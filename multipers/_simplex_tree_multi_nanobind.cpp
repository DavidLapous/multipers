#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/make_iterator.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "ext_interface/nanobind_registry_helpers.hpp"
#include "simplextree_conversion_core.hpp"
#include "nanobind_array_utils.hpp"
#include "nanobind_object_utils.hpp"
#include "multi_parameter_rank_invariant/euler_characteristic.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpst {

using tensor_dtype = int32_t;
using indices_type = int32_t;
using signed_measure_type = std::pair<std::vector<std::vector<indices_type>>, std::vector<tensor_dtype>>;

using multipers::core::SimplexTreeConversion;
using multipers::nanobind_helpers::dispatch_simplextree_by_template_id;
using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::SimplexTreeDescriptorList;
using multipers::nanobind_helpers::SlicerDescriptorList;
using multipers::nanobind_helpers::type_list;
using multipers::nanobind_helpers::visit_const_simplextree_wrapper;
using multipers::nanobind_utils::lowercase_copy;
using multipers::nanobind_utils::numpy_dtype_name;
using multipers::nanobind_utils::numpy_dtype_type;
using multipers::nanobind_utils::owned_array;
using multipers::nanobind_utils::template_id_of;
using multipers::nanobind_utils::vector_from_handle;

template <typename... Ds>
nb::object get_simplextree_class(type_list<Ds...>,
                                 const nb::handle& dtype,
                                 bool kcritical,
                                 std::string filtration_container) {
  std::string dtype_name = numpy_dtype_name(dtype);
  filtration_container = lowercase_copy(std::move(filtration_container));
  std::string_view normalized_filtration_container = filtration_container;
  bool matched = false;
  nb::object result;
  (
      [&]<typename D>() {
        if (!matched && D::dtype_name == dtype_name && D::is_kcritical == kcritical &&
            D::filtration_container == normalized_filtration_container) {
          result = nb::borrow<nb::object>(nb::type<typename D::interface_type>());
          matched = true;
        }
      }.template operator()<Ds>(),
      ...);
  if (!matched) {
    throw nb::type_error("Unknown SimplexTreeMulti implementation.");
  }
  return result;
}

nb::object get_simplextree_class_from_template_id(int template_id) {
  return dispatch_simplextree_by_template_id(template_id, [&]<typename Desc>() -> nb::object {
    return nb::borrow<nb::object>(nb::type<typename Desc::interface_type>());
  });
}

inline bool is_simplextree_multi(nb::handle input) { return is_simplextree_object(input); }

nb::tuple signed_measure_to_python(const signed_measure_type& sm, size_t width) {
  std::vector<indices_type> flat_pts;
  flat_pts.reserve(sm.first.size() * width);
  for (const auto& row : sm.first) {
    flat_pts.insert(flat_pts.end(), row.begin(), row.end());
  }
  std::vector<tensor_dtype> weights(sm.second.begin(), sm.second.end());
  return nb::make_tuple(nb::cast(owned_array<indices_type>(std::move(flat_pts), {sm.first.size(), width})),
                        nb::cast(owned_array<tensor_dtype>(std::move(weights), {sm.second.size()})));
}

template <typename... Desc>
nb::tuple compute_euler_signed_measure(type_list<Desc...>,
                                       nb::handle simplextree,
                                       std::vector<tensor_dtype>& container,
                                       const std::vector<indices_type>& grid_shape,
                                       size_t width,
                                       bool zero_pad,
                                       bool verbose) {
  if (!is_simplextree_multi(simplextree)) {
    throw std::runtime_error("Unsupported SimplexTreeMulti type.");
  }
  return dispatch_simplextree_by_template_id(template_id_of(simplextree), [&]<typename D>() -> nb::tuple {
    if constexpr (D::is_kcritical) {
      throw std::runtime_error("Unsupported SimplexTreeMulti type.");
    } else {
      auto& st = nb::cast<typename D::interface_type&>(simplextree);
      signed_measure_type sm;
      {
        nb::gil_scoped_release release;
        sm = Gudhi::multiparameter::euler_characteristic::get_euler_signed_measure(
            st, container.data(), grid_shape, zero_pad, verbose);
      }
      return signed_measure_to_python(sm, width);
    }
  });
}

template <typename Target>
bool try_copy_from_any(Target& self, nb::handle source) {
  if (!is_simplextree_object(source)) {
    return false;
  }
  visit_const_simplextree_wrapper(source, [&]<typename D>(const typename D::interface_type& sourceWrapper) {
    SimplexTreeConversion<Target, typename D::interface_type>::run(self, sourceWrapper);
  });
  return true;
}

template <typename Target, typename Source>
Target construct_from_simplex_tree(const Source& source, int numParam) {
  Target out;
  SimplexTreeConversion<Target, Source>::run(out, source, numParam);
  return out;
}

template <typename Target, typename Source>
Target construct_from_slicer(const Source& source, int max_dim, int numParam = -1) {
  Target out;
  out.copy_from(source.get_slicer(), max_dim, numParam);
  return out;
}

template <typename Target, typename Class, typename... SourceDesc>
void bind_simplextree_source_constructors(Class& cls, type_list<SourceDesc...>) {
  (cls.def(
       "__init__",
       [](Target* self, const typename SourceDesc::interface_type& source, int numParam) {
         new (self) Target(construct_from_simplex_tree<Target>(source, numParam));
       },
       "source"_a,
       "num_parameters"_a = -1),
   ...);
}

template <typename Target, typename Class, typename... SourceDesc>
void bind_slicer_source_constructors(Class& cls, type_list<SourceDesc...>) {
  (cls.def(
       "__init__",
       [](Target* self, const typename SourceDesc::interface& source, int max_dim, int numParam) {
         new (self) Target(construct_from_slicer<Target>(source, max_dim, numParam));
       },
       "source"_a,
       "max_dim"_a = -1,
       "num_parameters"_a = -1),
   ...);
}

template <class Interface, typename Class>
void bind_simplex_tree_constructors(Class& cls) {
  cls.def(nb::init<>()).def(nb::init<int>(), "num_parameters"_a = -1);

  bind_simplextree_source_constructors<Interface>(cls, SimplexTreeDescriptorList{});
  bind_slicer_source_constructors<Interface>(cls, SlicerDescriptorList{});

  cls.def("_copy_from_any",
          [](Interface& self, nb::handle other) -> Interface& {
            if (!try_copy_from_any<Interface>(self, other)) {
              throw std::runtime_error("Unsupported SimplexTreeMulti input type. Got " +
                                       std::string(nb::inst_name(other).c_str()) + ".");
            }
            return self;
          })
      .def("_from_gudhi_state", &Interface::from_std);
}

template <class Interface, typename Class>
void bind_simplex_tree_dunders(Class& cls) {
  cls.def("__getstate__", [](Interface& self) -> nanobind::tuple { return self.serialize(); })
      .def("__setstate__",
           [](Interface& self, nanobind::tuple state) {
             auto st = Gudhi::multi_persistence::deserialize_multi_simplex_tree_from_python<Interface>(state);
             new (&self) Interface(std::move(st));
           })
      .def(
          "__iter__", [](Interface& self) { return self.get_simplex_python_iterator(); }, nb::keep_alive<0, 1>())
      .def("__eq__", [](Interface& self, Interface& other) { return self == other; });
}

template <class Interface, typename Desc, typename Class>
void bind_simplex_tree_properties(Class& cls) {
  cls.def_prop_rw("filtration_grid", &Interface::get_filtration_grid, &Interface::set_filtration_grid, "value"_a.none())
      .def_prop_ro("num_parameters", &Interface::num_parameters)
      .def_prop_ro("is_kcritical", [](const Interface&) -> bool { return Desc::is_kcritical; })
      .def_prop_ro("dtype", [](const Interface&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); })
      .def_prop_ro("ftype", [](const Interface&) -> std::string { return std::string(Desc::ftype_name); })
      .def_prop_ro("filtration_container",
                   [](const Interface&) -> std::string { return std::string(Desc::filtration_container_name); })
      .def_prop_ro("_template_id", [](const Interface&) -> int { return Desc::template_id; });

  cls.def("get_simplices", &Interface::get_simplex_python_iterator, nb::keep_alive<0, 1>())
      .def("get_skeleton", &Interface::get_skeleton_python_iterator, nb::keep_alive<0, 1>())
      .def("get_boundaries", &Interface::get_boundary_python_iterator, nb::keep_alive<0, 1>());

  cls.def("num_vertices", &Interface::num_vertices, nb::call_guard<nb::gil_scoped_release>())
      .def("num_simplices",
           nb::overload_cast<>(&Interface::num_simplices, nb::const_),
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "dimension", nb::overload_cast<>(&Interface::dimension, nb::const_), nb::call_guard<nb::gil_scoped_release>())
      .def("upper_bound_dimension", &Interface::upper_bound_dimension, nb::call_guard<nb::gil_scoped_release>())
      .def("find_simplex", &Interface::find_simplex)
      .def("get_simplices_of_dimension", &Interface::get_simplices_of_dimension)
      .def("_get_filtration",
           &Interface::get_simplex_filtration_value,
           "simplex"_a,
           "copy_only_when_necessary"_a = true,
           "raw"_a = false)
      .def("_get_filtration_values", &Interface::get_filtration_values);

  cls.def("get_edge_list", &Interface::template get_edge_list<>).def("pts_to_indices", &Interface::get_point_indices);
}

template <class Interface, typename Class>
void bind_simplex_tree_modifiers(Class& cls) {
  using Value = typename Interface::value_type;
  using Tensor1D = nanobind::ndarray<const Value, nanobind::ndim<1>, nanobind::any_contig>;

  cls.def("_insert", &Interface::insert_single_simplex, "simplex"_a, "filtration"_a = nb::none())
      .def("_insert_batch", &Interface::insert_batch)
      .def("remove_maximal_simplex", &Interface::remove_maximal_simplex)
      .def("prune_above_dimension", &Interface::prune_above_dimension, nb::call_guard<nb::gil_scoped_release>())
      .def("expansion", &Interface::expand)
      .def("make_filtration_non_decreasing",
           &Interface::make_filtration_non_decreasing,
           nb::call_guard<nb::gil_scoped_release>())
      .def("_assign_filtration", &Interface::assign_simplex_filtration, "vertices"_a, "filtration"_a = nb::none())
      .def("_normalize_filtrations_raw", &Interface::template normalize_filtration_values<Value>, "box"_a = nb::none())
      .def("_simplify_filtration_raw", &Interface::simplify_all_filtration_values)
      .def("_fill_lowerstar", &Interface::fill_lowerstar)
      .def("_fill_distance_matrix",
           &Interface::fill_distance_matrix,
           "distance_matrix"_a,
           "parameter"_a,
           "node_value"_a = 0)
      .def("_squeeze_inplace",
           nanobind::overload_cast<const std::vector<Tensor1D>&, bool>(&Interface::template coarsen_on_grid<Value>),
           nanobind::rv_policy::reference_internal)
      .def("_squeeze_inplace",
           nanobind::overload_cast<const std::vector<std::vector<Value>>&, bool>(
               &Interface::template coarsen_on_grid<Value>),
           nanobind::rv_policy::reference_internal)
      .def("_clean_filtration_grid_raw", &Interface::clean_filtration_grid);

  cls.def("_get_to_std_state", &Interface::template project_on_line_to_std<>)
      .def("_unsqueeze_to", &Interface::build_unsqueezed_from)
      .def("_reconstruct_from_edge_array",
           &Interface::template build_bifiltration_from_edges<>,
           "edges"_a,
           "expand_dimension"_a = 0);
}

template <typename Desc>
void bind_simplextree_class(nb::module_& m, nb::list& available_simplex_trees) {
  using Interface = typename Desc::interface_type;

  auto cls = nb::class_<Interface>(m, Desc::python_name.data());

  bind_simplex_tree_constructors<Interface>(cls);
  bind_simplex_tree_dunders<Interface>(cls);
  bind_simplex_tree_properties<Interface, Desc>(cls);
  bind_simplex_tree_modifiers<Interface>(cls);

  available_simplex_trees.append(cls);
}

template <typename... Desc>
void bind_all_simplex_trees(type_list<Desc...>, nb::module_& m, nb::list& available_simplex_trees) {
  (bind_simplextree_class<Desc>(m, available_simplex_trees), ...);
}

}  // namespace mpst

NB_MODULE(_simplex_tree_multi_nanobind, m) {
  m.doc() = "nanobind SimplexTreeMulti bindings";
  nb::list available_simplex_trees;

  mpst::bind_all_simplex_trees(mpst::SimplexTreeDescriptorList{}, m, available_simplex_trees);

  m.def(
      "_get_simplextree_class",
      [](nb::handle dtype, bool kcritical, std::string filtration_container) {
        return mpst::get_simplextree_class(
            mpst::SimplexTreeDescriptorList{}, dtype, kcritical, std::move(filtration_container));
      },
      "dtype"_a,
      "kcritical"_a = false,
      "filtration_container"_a = "Contiguous");
  m.def("_get_simplextree_class_from_template_id", &mpst::get_simplextree_class_from_template_id, "template_id"_a);
  m.def("is_simplextree_multi", [](nb::object input) { return mpst::is_simplextree_multi(input); });

  m.def(
      "_compute_euler_signed_measure",
      [](nb::handle simplextree, nb::handle grid_shape_handle, bool zero_pad, bool verbose) {
        auto grid_shape = mpst::vector_from_handle<mpst::indices_type>(grid_shape_handle);
        size_t width = grid_shape.size();
        size_t total = 1;
        for (mpst::indices_type value : grid_shape) {
          total *= static_cast<size_t>(value);
        }
        std::vector<mpst::tensor_dtype> container(total, 0);
        return mpst::compute_euler_signed_measure(
            mpst::SimplexTreeDescriptorList{}, simplextree, container, grid_shape, width, zero_pad, verbose);
      },
      "simplextree"_a,
      "grid_shape"_a,
      "zero_pad"_a = false,
      "verbose"_a = false);

  m.attr("available_simplextrees") = available_simplex_trees;
}

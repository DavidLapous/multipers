#ifndef MP_PY_SLICER_NANOBIND_H_INCLUDED
#define MP_PY_SLICER_NANOBIND_H_INCLUDED

#include <cstdint>
#include <utility>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
// #include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "ext_interface/nanobind_registry_helpers.hpp"
#include "nanobind_object_utils.hpp"
#include "gudhi/interface_helper_structs.h"

namespace mpnb {

using namespace nanobind::literals;  // for the "argname"_a
using multipers::nanobind_helpers::type_list;
using multipers::nanobind_helpers::simplextree_wrapper_t;
using multipers::nanobind_helpers::SlicerDescriptorList;
using multipers::nanobind_helpers::SimplexTreeDescriptorList;
using multipers::nanobind_helpers::PySimplexTree;
using multipers::nanobind_utils::numpy_dtype_type;

inline void bind_generator_basis(nanobind::module_& m) {
  using Generator_basis_data = Gudhi::multi_persistence::detail::Generator_basis_data;
  using Index = Generator_basis_data::Index;  // std::uint32_t
  using Grade = Generator_basis_data::Grade;  // double

  nanobind::class_<Generator_basis_data>(m, "_GeneratorBasis")
      .def(nanobind::init<int,
                          std::vector<std::vector<Index>>,
                          std::vector<std::vector<Index>>,
                          std::vector<std::pair<Grade, Grade>>,
                          std::vector<std::pair<Grade, Grade>>>(),
           "degree"_a,
           "columns"_a,
           "row_boundaries"_a,
           "row_grades"_a = std::vector<std::pair<Grade, Grade>>{},
           "column_grades"_a = std::vector<std::pair<Grade, Grade>>{})
      .def_prop_ro("degree", [](const Generator_basis_data& self) { return self.degree; })
      .def_prop_ro("columns", [](const Generator_basis_data& self) { return self.columns; })
      .def_prop_ro("row_boundaries", [](const Generator_basis_data& self) { return self.rowBoundaries; })
      .def_prop_ro("row_grades", [](const Generator_basis_data& self) { return self.rowGrades; })
      .def_prop_ro("column_grades", [](const Generator_basis_data& self) { return self.columnGrades; })
      .def("keys", &Generator_basis_data::get_keys)
      .def("__getitem__",
           [](const Generator_basis_data& self, const std::string& key) -> nanobind::object { return self[key]; })
      .def(
          "__contains__",
          [](const Generator_basis_data&, const std::string& key) -> bool { return Generator_basis_data::is_key(key); })
      .def("__repr__", &Generator_basis_data::to_str)
      .def("__getstate__",
           [](const Generator_basis_data& self) -> nanobind::ndarray<nanobind::numpy, char> {
             std::size_t buffer_size;
             char* buffer;
             {
               nanobind::gil_scoped_release release;
               buffer_size = get_serialization_size_of(self);
               buffer = new char[buffer_size];
               serialize_value_to_char_buffer(self, buffer);
             }
             return _wrap_as_numpy_array(buffer, buffer_size);
           })
      .def("__setstate__",
           [](Generator_basis_data& self, nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy> state) {
             new (&self)
                 Generator_basis_data(Gudhi::multi_persistence::detail::deserialize_gen_basis_from_python(state));
           });
}

template <typename Class, typename... SourceDesc>
inline void bind_from_slicer_constructors(Class& cls, type_list<SourceDesc...>) {
  (cls.def(nanobind::init<const typename SourceDesc::interface&>()), ...);
}

template <typename Class, typename... SourceDesc>
inline void bind_from_simplex_tree_constructors(Class& cls, type_list<SourceDesc...>) {
  (cls.def(nanobind::init<PySimplexTree<typename SourceDesc::interface_type, typename SourceDesc::value_type>&>()),
   ...);
}

template <class Target, typename Class, typename... SourceDesc>
inline void bind_slicer_eq(Class& cls, type_list<SourceDesc...>) {
  (cls.def(
       "__eq__",
       [](const Target& a, const typename SourceDesc::interface& b) { return a == b; },
       nanobind::is_operator()),
   ...);
}

template <class Target, typename Class, typename... SourceDesc>
inline void bind_from_slicer_copy(Class& cls, type_list<SourceDesc...>) {
  (cls.def("_copy_from_any",
           nanobind::overload_cast<const typename SourceDesc::interface&>(
               &Target::template copy<typename SourceDesc::concrete::Filtration_value,
                                      typename SourceDesc::concrete::Persistence>)),
   ...);
}

template <class Target, typename Class, typename... SourceDesc>
inline void bind_from_simplex_tree_copy(Class& cls, type_list<SourceDesc...>) {
  (cls.def(
       "_copy_from_any",
       nanobind::overload_cast<PySimplexTree<typename SourceDesc::interface_type, typename SourceDesc::value_type>&>(
           &Target::template copy<typename SourceDesc::filtration_type>)),
   ...);
}

template <class Slicer, typename Class>
inline void bind_slicer_constructors(Class& cls) {
  using T = typename Slicer::value_type;
  using Tensor2D = nanobind::ndarray<const T, nanobind::ndim<2>>;

  // default constructor
  cls.def(nanobind::init<>());

  // constructors from all available slicers
  bind_from_slicer_constructors(cls, SlicerDescriptorList{});
  // constructors from all available simplex trees
  bind_from_simplex_tree_constructors(cls, SimplexTreeDescriptorList{});

  // from cubical image
  cls.def(nanobind::init<Tensor2D, const std::vector<unsigned int>&>());

  // from file
  cls.def(nanobind::init<const std::string&, int, bool, bool>(),
          "path"_a,
          "shift_dimension"_a,
          "is_rivet_compatible"_a = false,
          "is_reversed"_a = false);

  // from containers
  // allowing all combinations of tensor/sequence and dtype multiplies by more than 5 the compile time...
  // so generator_maps will automatically convert to std::vector<std::vector<Index>> and as a copy is made
  // anyway, it can directly be copied into the Index type
  // generator_dimensions is restricted to tensor types, but is allowed every dtype. But tensors of dtypes
  // which are not uint32, int32, uint64 or int64 will be copied
  // filtration_values is given the most freedom. The only real restriction is that the dtype has to be
  // uint32, int32, uint64 or int64 if the Slicer dtype is integer and float or double if the Slicer dtype
  // is floating point.
  cls.def(nanobind::init<const std::vector<std::vector<typename Slicer::Index>>&,
                         nanobind::ndarray<const std::uint32_t, nanobind::ndim<1>, nanobind::any_contig>,
                         nanobind::iterable>(),
          "generator_maps"_a,
          "generator_dimensions"_a.noconvert(),
          "filtration_values"_a)
      .def(nanobind::init<const std::vector<std::vector<typename Slicer::Index>>&,
                          nanobind::ndarray<const std::int32_t, nanobind::ndim<1>, nanobind::any_contig>,
                          nanobind::iterable>(),
           "generator_maps"_a,
           "generator_dimensions"_a.noconvert(),
           "filtration_values"_a)
      .def(nanobind::init<const std::vector<std::vector<typename Slicer::Index>>&,
                          nanobind::ndarray<const std::uint64_t, nanobind::ndim<1>, nanobind::any_contig>,
                          nanobind::iterable>(),
           "generator_maps"_a,
           "generator_dimensions"_a.noconvert(),
           "filtration_values"_a)
      .def(nanobind::init<const std::vector<std::vector<typename Slicer::Index>>&,
                          nanobind::ndarray<const std::int64_t, nanobind::ndim<1>, nanobind::any_contig>,
                          nanobind::iterable>());

  // handles None case, has to be bind last
  cls.def("__init__", [](Slicer* self, nanobind::handle arg) {
    if (!arg.is_none()) throw nanobind::next_overload();
    new (self) Slicer();
  });
}

template <class Slicer, typename Class>
inline void bind_slicer_dunders(Class& cls) {
  cls.def("__len__", &Slicer::size)
      .def("__getstate__",
           [](const Slicer& self) -> nanobind::tuple {
             std::size_t buffer_size;
             char* buffer;
             {
               nanobind::gil_scoped_release release;
               buffer_size = get_serialization_size_of(self);
               buffer = new char[buffer_size];
               serialize_value_to_char_buffer(self, buffer);
             }
             return nanobind::make_tuple(self.get_filtration_grid(), _wrap_as_numpy_array(buffer, buffer_size));
           })
      .def("__setstate__", [](Slicer& self, nanobind::tuple state) {
        new (&self) Slicer(Gudhi::multi_persistence::deserialize_slicer_from_python<Slicer>(state));
      });

  //__eq__
  bind_slicer_eq<Slicer>(cls, SlicerDescriptorList{});
}

template <class Slicer, typename Desc, typename Class>
inline void bind_slicer_properties(Class& cls) {
  cls.def_prop_rw(
         "filtration_grid", &Slicer::get_filtration_grid, &Slicer::set_filtration_grid, nanobind::arg("value").none())
      .def_prop_rw("minpres_degree",
                   &Slicer::get_min_pres_degree,
                   [](Slicer& self, int degree) { self.set_min_pres_degree(degree, self.is_min_res()); })
      .def_prop_rw("is_minres", &Slicer::is_min_res, &Slicer::set_is_min_res)
      .def_prop_rw(
          "_generator_basis",
          &Slicer::get_generator_basis,
          [](Slicer& self, nanobind::object value) {
            if (value.is_none() || nanobind::isinstance<nanobind::dict>(value)) {
              self.set_generator_basis(nanobind::cast<nanobind::dict>(value));
              return;
            }
            self.set_generator_basis(
                nanobind::cast<std::optional<Gudhi::multi_persistence::detail::Generator_basis_data>>(value));
          },
          nanobind::arg("value").none())
      .def_prop_ro("is_pres", &Slicer::is_pres)
      .def_prop_ro("pres_degree", &Slicer::get_pres_degree)
      .def_prop_ro("is_minpres", &Slicer::is_min_pres)
      .def_prop_ro("num_generators", &Slicer::size)
      .def_prop_ro("dimension", &Slicer::get_max_dimension)
      .def_prop_ro("num_parameters", &Slicer::get_number_of_parameters)
      .def_prop_ro("dtype", [](const Slicer&) -> nanobind::object { return numpy_dtype_type(Desc::dtype_name); })
      .def_prop_ro("col_type", [](const Slicer&) -> std::string { return std::string(Desc::column_type); })
      .def_prop_ro("filtration_container",
                   [](const Slicer&) -> std::string { return std::string(Desc::filtration_container); })
      .def_prop_ro("is_vine", [](const Slicer&) -> bool { return Desc::is_vine; })
      .def_prop_ro("is_kcritical", [](const Slicer&) -> bool { return Desc::is_kcritical; })
      .def_prop_ro("pers_backend", [](const Slicer&) -> std::string { return std::string(Desc::backend_type); })
      .def_prop_ro("ftype", [](const Slicer&) -> std::string { return std::string(Desc::filtration_type); })
      .def_prop_ro("_template_id", [](const Slicer&) -> int { return Desc::template_id; })
      .def_ro_static("_inf_value", &Slicer::T_inf)
      .def_ro_static("_minus_inf_value", &Slicer::T_m_inf);

  cls.def("get_dimensions", &Slicer::get_dimensions)
      .def(
          "get_boundaries",
          [](const Slicer& self, bool packed) -> nanobind::tuple {
            if (packed) return self.get_flat_boundaries();
            return self.get_boundaries();
          },
          "packed"_a = false)
      .def("get_filtration",
           &Slicer::get_filtration_value,
           "idx"_a,
           "copy_only_when_necessary"_a = true,
           "raw"_a = false)
      .def("get_filtrations_values",
           [](Slicer& self) -> nanobind::ndarray<nanobind::numpy, typename Slicer::value_type> {
             return nanobind::cast<nanobind::ndarray<nanobind::numpy, typename Slicer::value_type>>(
                 self.get_all_filtration_values(true, false, false)[1]);
           })
      .def("_get_filtrations_impl",
           &Slicer::get_all_filtration_values,
           "packed"_a = false,
           "view"_a = false,
           "raw"_a = false)
      .def("_mark_minpres", &Slicer::set_min_pres_degree, "degree"_a, "is_minres"_a = false)
      .def("_mark_pres", [](Slicer& self, int degree) { self.set_is_pres(degree, false); });
}

template <class Slicer, typename Desc, typename Class>
inline void bind_slicer_modifiers(Class& cls) {
  using T = typename Slicer::value_type;
  using Tensor1D = nanobind::ndarray<const T, nanobind::ndim<1>, nanobind::any_contig>;

  cls.def("prune_above_dimension", &Slicer::prune_above_dimension)
      .def("coarsen_on_grid_inplace",
           nanobind::overload_cast<const std::vector<Tensor1D>&, bool>(&Slicer::template coarsen_on_grid<T>))
      .def("coarsen_on_grid_inplace",
           nanobind::overload_cast<const std::vector<std::vector<T>>&, bool>(&Slicer::template coarsen_on_grid<T>))
      .def("to_colexical", &Slicer::build_colexical_permuted_slicer, "return_permutation"_a = false)
      .def("permute_generators", &Slicer::build_slicer_as_permutation)
      .def("push_to_line", &Slicer::template push_to_line<T>, "basepoint"_a, "direction"_a = nanobind::none())
      .def("initialize_persistence_computation",
           &Slicer::initialize_persistence_computation,
           "ignore_infinite_filtration_values"_a = true)
      .def("update_persistence_computation",
           &Slicer::update_persistence_computation,
           "ignore_infinite_filtration_values"_a = false)
      .def("get_barcode", &Slicer::get_barcode)
      .def("get_barcode_idx", &Slicer::get_barcode_as_indices)
      .def("_compute_persistence_on_slices",
           &Slicer::compute_persistence_on_slices,
           "values"_a,
           "ignore_infinite_filtration_values"_a = true)
      .def("_landscapes_on_grid",
           &Slicer::template compute_landscapes_on_grid<T>,
           "xgrid"_a,
           "ygrid"_a,
           "direction"_a,
           "stride_i"_a,
           "stride_j"_a,
           "dt"_a,
           "degree"_a,
           "ks"_a,
           "n_jobs"_a = 0,
           "ignore_infinite_filtration_values"_a = true)
      .def("_make_filtration_non_decreasing_raw", &Slicer::make_filtration_non_decreasing)
      .def("_simplify_filtration_raw", &Slicer::simplify_all_filtration_values)
      .def("_normalize_filtrations_raw",
           &Slicer::template normalize_filtration_values<double>,
           "box"_a = nanobind::none())
      .def("_clean_filtration_grid_raw", &Slicer::clean_filtration_grid);

  bind_from_slicer_copy<Slicer>(cls, SlicerDescriptorList{});
  bind_from_simplex_tree_copy<Slicer>(cls, SimplexTreeDescriptorList{});
  cls.def("copy", [](const Slicer& self) -> Slicer { return Slicer(self); });

  if constexpr (Desc::has_grid_methods) {
    cls.def("coarsen_on_grid_copy", &Slicer::template build_coarsen_on_grid<T>)
        .def("compute_kernel_projective_cover", &Slicer::build_from_projective_cover_kernel, "dim"_a = nb::none());
  }

  if constexpr (Desc::is_vine) {
    using Persistence = typename Desc::concrete::Persistence;
    if constexpr (Persistence::has_rep_cycles) {
      cls.def("get_representative_cycles",
              &Slicer::get_representative_cycles,
              "update"_a = true,
              "dim"_a = nanobind::none(),
              "idx"_a = nanobind::none(),
              "intersect_points"_a = nanobind::none())
          .def("get_most_persistent_cycles",
               &Slicer::get_most_persistent_cycles,
               "dim"_a = 1,
               "n"_a = 1,
               "update"_a = true,
               "idx"_a = false);
    }
  }
}

template <class Slicer, typename Class>
inline void bind_slicer_io(Class& cls) {
  cls.def("_info_string", &Slicer::to_string)
      .def("_to_scc_raw",
           &Slicer::write_to_scc_file,
           "path"_a,
           "degree"_a = -1,
           "rivet_compatible"_a = false,
           "ignore_last_generators"_a = false,
           "strip_comments"_a = false,
           "reverse"_a = false);
}

template <typename Desc>
inline void bind_slicer_class(nanobind::module_& m, nanobind::list& available_slicers) {
  using Slicer = typename Desc::interface;

  auto cls = nanobind::class_<Slicer>(m, Desc::python_name.data());

  bind_slicer_constructors<Slicer>(cls);
  bind_slicer_dunders<Slicer>(cls);
  bind_slicer_properties<Slicer, Desc>(cls);
  bind_slicer_modifiers<Slicer, Desc>(cls);
  bind_slicer_io<Slicer>(cls);

  available_slicers.append(cls);
}

template <typename... Desc>
inline void bind_all_slicers(type_list<Desc...>, nanobind::module_& m, nanobind::list& available_slicers) {
  (bind_slicer_class<Desc>(m, available_slicers), ...);
}

}  // namespace mpnb

#endif  // MP_PY_SLICER_NANOBIND_H_INCLUDED

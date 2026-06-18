#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#include <python_interfaces/numpy_utils.h>
#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Summand.h>
#include "gudhi/Module_interface.h"
#include "nanobind_array_utils.hpp"
#include "nanobind_dense_array_utils.hpp"
#include "nanobind_mma_registry_helpers.hpp"
#include "nanobind_object_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpmma {

using multipers::nanobind_dense_utils::vector_from_array;
using multipers::nanobind_mma_helpers::is_mma_module_object;
using multipers::nanobind_mma_helpers::MMADescriptorList;
using multipers::nanobind_mma_helpers::type_list;
using multipers::nanobind_utils::matrix_from_handle;
using multipers::nanobind_utils::numpy_dtype_type;
using multipers::nanobind_utils::owned_array;
using multipers::nanobind_utils::vector_from_handle;

template <typename T>
using BoxArray = nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig>;

template <typename T>
nb::ndarray<nb::numpy, T> corner_matrix_to_python(std::vector<T>&& flat, size_t rows, size_t cols) {
  return owned_array<T>(std::move(flat), {rows, cols});
}

template <typename T>
nb::ndarray<nb::numpy, T> corner_matrix_to_python(const std::vector<T>& flat, size_t rows, size_t cols) {
  return owned_array<T>(std::vector<T>(flat.begin(), flat.end()), {rows, cols});
}

template <typename T>
nb::ndarray<nb::numpy, T> corner_pair_to_python(const std::vector<T>& lower, const std::vector<T>& upper) {
  std::vector<T> flat;
  flat.reserve(lower.size() + upper.size());
  flat.insert(flat.end(), lower.begin(), lower.end());
  flat.insert(flat.end(), upper.begin(), upper.end());
  return owned_array<T>(std::move(flat), {size_t(2), lower.size()});
}

template <typename Desc, typename Class>
void bind_float_module_methods(Class& cls) {
  if constexpr (Desc::is_float) {
    using T = typename Desc::value_type;
    using Box = Gudhi::multi_persistence::Box<T>;
    using Module = Gudhi::multi_persistence::Module_interface<T>;
    using NDArray1 = typename Module::Tensor1D;
    using NDArray2 = typename Module::Tensor2D;

    cls.def("_get_barcode_from_line",
            &Module::get_barcode_from_line,
            "basepoint"_a,
            "direction"_a = nb::none(),
            "degree"_a = -1)
        .def("_get_barcodes_from_lines",
             &Module::get_barcode_from_lines,
             "basepoints"_a,
             "directions"_a = nb::none(),
             "degree"_a = -1,
             "keep_inf"_a = true)
        // .def("evaluate_in_grid", nb::overload_cast<const std::vector<std::vector<T>>&>(&Module::evaluate_in_grid))
        .def("evaluate_in_grid", nb::overload_cast<const std::vector<NDArray1>&>(&Module::evaluate_in_grid))
        .def("evaluate_in_grid", nb::overload_cast<NDArray2>(&Module::evaluate_in_grid))
        .def("_compute_landscapes_box",
             &Module::template compute_landscapes_from_box<std::int32_t>,
             "degree"_a,
             "ks"_a,
             "box"_a,
             "resolution"_a,
             "n_jobs"_a = 0)
        .def("_compute_landscapes_box",
             &Module::template compute_landscapes_from_box<std::int64_t>,
             "degree"_a,
             "ks"_a,
             "box"_a,
             "resolution"_a,
             "n_jobs"_a = 0)
        .def("_compute_landscapes_grid",
             &Module::template compute_landscapes_from_grid<std::int32_t>,
             "degree"_a,
             "ks"_a,
             "grid"_a,
             "n_jobs"_a = 0)
        .def("_compute_landscapes_grid",
             &Module::template compute_landscapes_from_grid<std::int64_t>,
             "degree"_a,
             "ks"_a,
             "grid"_a,
             "n_jobs"_a = 0)
        .def("_compute_pixels",
             &Module::template compute_pixels<std::int32_t>,
             "coordinates"_a,
             "degrees"_a,
             "box"_a,
             "delta"_a,
             "p"_a,
             "normalize"_a = false,
             "n_jobs"_a = 0)
        .def("_compute_pixels",
             &Module::template compute_pixels<std::int64_t>,
             "coordinates"_a,
             "degrees"_a,
             "box"_a,
             "delta"_a,
             "p"_a,
             "normalize"_a = false,
             "n_jobs"_a = 0)
        // .def("distance_to",
        //      nb::overload_cast<const std::vector<std::vector<T>>&, bool, int>(&Module::compute_distance_to),
        //      "pts"_a,
        //      "signed"_a = false,
        //      "n_jobs"_a = 0)
        .def("distance_to",
             nb::overload_cast<NDArray2, bool, int>(&Module::compute_distance_to),
             "pts"_a,
             "signed"_a = false,
             "n_jobs"_a = 0)
        .def("get_interleavings", nb::overload_cast<>(&Module::compute_interleavings))
        .def("get_interleavings", nb::overload_cast<NDArray2>(&Module::compute_interleavings));
  }
}

template <typename Desc>
void bind_summand_class(nb::module_& m) {
  using T = typename Desc::value_type;
  using Summand = Gudhi::multi_persistence::Summand<T>;

  nb::class_<Summand>(m, Desc::summand_name.data())
      .def(nb::init<>())
      .def("get_birth_list",
           [](Summand& self) -> nb::ndarray<nb::numpy, T> {
             std::vector<T> births;
             const size_t num_parameters = static_cast<size_t>(self.get_number_of_parameters());
             const size_t num_birth_corners = static_cast<size_t>(self.get_number_of_birth_corners());
             {
               nb::gil_scoped_release release;
               births = self.compute_flat_upset();
             }
             return corner_matrix_to_python<T>(std::move(births), num_birth_corners, num_parameters);
           })
      .def("get_death_list",
           [](Summand& self) -> nb::ndarray<nb::numpy, T> {
             std::vector<T> deaths;
             const size_t num_parameters = static_cast<size_t>(self.get_number_of_parameters());
             const size_t num_death_corners = static_cast<size_t>(self.get_number_of_death_corners());
             {
               nb::gil_scoped_release release;
               deaths = self.compute_flat_downset();
             }
             return corner_matrix_to_python<T>(std::move(deaths), num_death_corners, num_parameters);
           })
      .def_prop_ro("degree", [](const Summand& self) -> int { return self.get_dimension(); })
      .def("get_bounds",
           [](Summand& self) -> nb::ndarray<nb::numpy, T> {
             std::pair<std::vector<T>, std::vector<T>> cbounds;
             {
               nb::gil_scoped_release release;
               auto bounds = self.compute_bounds();
               auto cpp_bounds = bounds.get_bounding_corners();
               cbounds.first.assign(cpp_bounds.first.begin(), cpp_bounds.first.end());
               cbounds.second.assign(cpp_bounds.second.begin(), cpp_bounds.second.end());
             }
             return corner_pair_to_python<T>(cbounds.first, cbounds.second);
           })
      .def("num_parameters", [](Summand& self) -> int { return self.get_number_of_parameters(); })
      .def_prop_ro("_template_id", [](const Summand&) -> int { return Desc::template_id; })
      .def_prop_ro("dtype", [](const Summand&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); })
      .def("__eq__", [](Summand& self, Summand& other) { return self == other; });
}

template <typename Desc>
void bind_box_class(nb::module_& m) {
  using T = typename Desc::value_type;
  using Box = Gudhi::multi_persistence::Box<T>;

  nb::class_<Box>(m, Desc::box_name.data())
      .def(nb::new_([](nb::handle bottom, nb::handle top) {
             auto lower = vector_from_handle<T>(bottom);
             auto upper = vector_from_handle<T>(top);
             return Box(lower, upper);
           }),
           "bottomCorner"_a,
           "topCorner"_a)
      .def(nb::new_([](nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> bottom,
                       nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> top) {
             return Box(vector_from_array(bottom), vector_from_array(top));
           }),
           "bottomCorner"_a,
           "topCorner"_a)
      .def_prop_ro("num_parameters", [](const Box& self) -> int { return self.get_lower_corner().size(); })
      .def("contains",
           [](Box& self, nb::handle x) {
             auto values = vector_from_handle<T>(x);
             return self.contains(values);
           })
      .def("contains",
           [](Box& self, nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig> x) {
             return self.contains(vector_from_array(x));
           })
      .def("get",
           [](Box& self) -> nb::ndarray<nb::numpy, T> {
             auto lower = std::vector<T>(self.get_lower_corner().begin(), self.get_lower_corner().end());
             auto upper = std::vector<T>(self.get_upper_corner().begin(), self.get_upper_corner().end());
             return corner_pair_to_python<T>(lower, upper);
           })
      .def("to_multipers",
           [](Box& self) -> nb::ndarray<nb::numpy, T> {
             auto lower = std::vector<T>(self.get_lower_corner().begin(), self.get_lower_corner().end());
             auto upper = std::vector<T>(self.get_upper_corner().begin(), self.get_upper_corner().end());
             std::vector<T> flat;
             flat.reserve(lower.size() * 2);
             for (size_t i = 0; i < lower.size(); ++i) {
               flat.push_back(lower[i]);
               flat.push_back(upper[i]);
             }
             return owned_array<T>(std::move(flat), {size_t(2), lower.size()});
           })
      .def_prop_ro("_template_id", [](const Box&) -> int { return Desc::template_id; })
      .def_prop_ro("dtype", [](const Box&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); });
}

template <typename Desc>
void bind_module_class(nb::module_& m) {
  using T = typename Desc::value_type;
  using Module = Gudhi::multi_persistence::Module_interface<T>;
  using Summand = typename Module::Summand_t;
  using Box = Gudhi::multi_persistence::Box<T>;
  using NDArray1 = typename Module::Tensor1D;
  using NDArray2 = typename Module::Tensor2D;

  std::string iterator_name = std::string("_PyModuleIterator_") + std::string(Desc::short_name);

  auto module_cls =
      nb::class_<Module>(m, Desc::module_name.data(), nb::dynamic_attr())
          .def(nb::init<>())
          .def(nb::init<Box>())
          .def(nb::init<NDArray2>())
          .def_prop_ro("dtype", [](const Module&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); })
          .def_prop_ro("_template_id", [](const Module&) -> int { return Desc::template_id; })
          .def(nb::self == nb::self)
          .def("__len__", [](Module& self) -> int { return self.size(); })
          .def("__getitem__",
               [](Module& self, int key) -> Summand& {
                 int size = self.size();
                 if (size == 0) throw nb::index_error("Module is empty.");
                 if (key < 0) key += size;
                 if (key < 0 || key >= size) throw nb::index_error("Summand index out of range.");
                 return self.get_summand(key);
               })
          .def(
              "__iter__",
              [iterator_name](const Module& self) {
                return nb::make_iterator(nb::type<Module>(),
                                         iterator_name.c_str(),
                                         self.begin(),
                                         self.end(),
                                         nb::rv_policy::reference_internal);
              },
              nb::keep_alive<0, 1>())
          .def("summands_of_dimension_range", &Module::get_summands_of_dimension_range)
          .def_prop_ro("max_degree", &Module::get_max_dimension)
          .def_prop_ro("num_parameters", &Module::get_number_of_parameters)
          .def_prop_rw("box",
                       &Module::get_box_view,
                       [](Module& self, nb::object box) {
                         if (nb::ndarray_check(box.ptr())) {
                           self.set_box(nb::cast<NDArray2>(box));
                           return;
                         }
                         if (nb::isinstance<Box>(box)) {
                           self.set_box(nb::cast<const Box&>(box));
                           return;
                         }
                         try {
                           self.set_box(nb::cast<std::vector<std::vector<T>>>(box));
                           return;
                         } catch (const nb::cast_error&) {
                           throw nb::type_error("Box has to be set with a 2D array, nested sequence or a PyBox.");
                         }
                       })
          .def("set_box",
               [](Module& self, const Box& box) {
                 PyErr_WarnEx(PyExc_DeprecationWarning, "set_box() is deprecated, use the .box property instead", 1);
                 return self.set_box(box);
               })
          .def("set_box",
               [](Module& self, NDArray2 box) {
                 PyErr_WarnEx(PyExc_DeprecationWarning, "set_box() is deprecated, use the .box property instead", 1);
                 return self.set_box(box);
               })
          .def("set_box",
               [](Module& self, const std::vector<std::vector<T>>& box) {
                 PyErr_WarnEx(PyExc_DeprecationWarning, "set_box() is deprecated, use the .box property instead", 1);
                 return self.set_box(box);
               })
          .def("get_bottom", &Module::get_box_lower_corner_view)
          .def("get_top", &Module::get_box_upper_corner_view)
          .def("get_box",
               [](const Module& self) {
                 PyErr_WarnEx(PyExc_DeprecationWarning, "get_box() is deprecated, use the .box property instead", 1);
                 return self.get_box_view();
               })
          .def("get_bounds", &Module::compute_bounds)
          .def("get_filtration_values", &Module::get_flat_filtration_values, "unique"_a = true)
          .def("get_dimensions", &Module::get_all_dimension)
          .def("merge", nb::overload_cast<const Module&>(&Module::merge))
          .def("merge", nb::overload_cast<const Module&, int>(&Module::merge))
          .def("_add_mmas", nb::overload_cast<nb::iterable>(&Module::merge))  // rename this also as 'merge'?
          .def("rescale", nb::overload_cast<NDArray1, int>(&Module::rescale), "rescale_factors"_a, "degree"_a = -1)
          // .def("rescale",
          //      nb::overload_cast<const std::vector<T>&, int>(&Module::rescale),
          //      "rescale_factors"_a,
          //      "degree"_a = -1)
          .def("translate", nb::overload_cast<NDArray1, int>(&Module::translate), "translation"_a, "degree"_a = -1)
          // .def("translate",
          //      nb::overload_cast<const std::vector<T>&, int>(&Module::translate),
          //      "translation"_a,
          //      "degree"_a = -1)
          .def("permute_summands", &Module::permute_summands)
          .def("get_module_of_degree", &Module::get_module_of_degree)
          .def("get_module_of_degrees", nb::overload_cast<NDArray1>(&Module::get_module_of_degrees))
          .def("get_module_of_degrees", nb::overload_cast<const std::vector<T>&>(&Module::get_module_of_degrees))
          .def("__getstate__",
               [](const Module& self) -> nb::ndarray<nb::numpy, char> {
                 std::size_t buffer_size;
                 char* buffer;
                 {
                   nb::gil_scoped_release release;
                   buffer_size = get_serialization_size_of(self);
                   buffer = new char[buffer_size];
                   serialize_value_to_char_buffer(self, buffer);
                 }
                 return _wrap_as_numpy_array(buffer, buffer_size);
               })
          .def("__setstate__", [](Module& self, nb::ndarray<const char, nb::ndim<1>, nb::numpy> state) {
            new (&self) Module(Gudhi::multi_persistence::deserialize_from_python<T>(state));
          });

  bind_float_module_methods<Desc>(module_cls);

  std::string range_name = std::string("_SummandByDimensionRange_") + std::string(Desc::short_name);
  std::string range_it_name = std::string("_SummandByDimensionIt_") + std::string(Desc::short_name);

  nb::class_<typename Module::Summand_of_dimension_range>(m, range_name.c_str())
      .def(nb::init<>())
      .def(
          "__iter__",
          [range_it_name](typename Module::Summand_of_dimension_range& r) {
            return nb::make_iterator(nb::type<typename Module::Summand_of_dimension_range>(),
                                     range_it_name.c_str(),
                                     r.begin(),
                                     r.end(),
                                     nb::rv_policy::reference_internal);
          },
          nb::keep_alive<0, 1>());
}

template <typename Desc>
void bind_mma_type(nb::module_& m) {
  bind_summand_class<Desc>(m);
  bind_box_class<Desc>(m);
  bind_module_class<Desc>(m);
}

template <typename... Desc>
void bind_all_mma(type_list<Desc...>, nb::module_& m) {
  (bind_mma_type<Desc>(m), ...);
}

inline bool is_mma(nb::handle stuff) { return is_mma_module_object(stuff); }

}  // namespace mpmma

NB_MODULE(_mma_nanobind, m) {
  m.doc() = "nanobind MMA bindings";
  mpmma::bind_all_mma(mpmma::MMADescriptorList{}, m);
  m.def("is_mma", [](nb::handle stuff) { return mpmma::is_mma(stuff); });
}

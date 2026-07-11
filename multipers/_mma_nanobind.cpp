#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>

#include <python_interfaces/numpy_utils.h>
#include <gudhi/simple_mdspan.h>
#include <gudhi/Multi_persistence/Box.h>
#include <gudhi/Multi_persistence/Summand.h>

#include "gudhi/Module_interface.h"
#include "gudhi/summand_interface.h"
#include "nanobind_mma_registry_helpers.hpp"
#include "nanobind_object_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpmma {

using multipers::nanobind_mma_helpers::is_mma_module_object;
using multipers::nanobind_mma_helpers::MMADescriptorList;
using multipers::nanobind_mma_helpers::type_list;
using multipers::nanobind_utils::numpy_dtype_type;

template <typename Desc>
void bind_summand_class(nb::module_& m) {
  using T = typename Desc::value_type;
  using Summand = Gudhi::multi_persistence::Summand<T>;
  using Box = Gudhi::multi_persistence::Box<T>;

  nb::class_<Summand>(m, Desc::summand_name.data())
      .def(nb::init<>())
      .def_prop_ro("_template_id", [](const Summand&) -> int { return Desc::template_id; })
      .def_prop_ro("dtype", [](const Summand&) -> nb::object { return numpy_dtype_type(Desc::dtype_name); })
      .def("__eq__",
           [](const Summand& self, const Summand& other) {
             bool res;
             {
               // comparing two summands should be expensive enough that it is worth releasing the GIL?
               nanobind::gil_scoped_release release;
               res = (self == other);
             }
             return res;
           })
      .def_prop_ro("num_parameters", &Summand::get_number_of_parameters)
      .def_prop_ro("degree", &Summand::get_dimension)
      .def("get_birth_list",
           [](const Summand& self) { return Gudhi::multi_persistence::compute_flat_corners<T>(self.get_upset()); })
      .def("get_death_list",
           [](const Summand& self) { return Gudhi::multi_persistence::compute_flat_corners<T>(self.get_downset()); })
      .def("get_bounds",
           [](const Summand& self) {
             std::vector<T> res;
             Box bounds;
             {
               nanobind::gil_scoped_release release;
               bounds = self.compute_bounds();
               res.reserve(bounds.get_number_of_coordinates() * 2);
               res.insert(res.end(), bounds.get_lower_corner().begin(), bounds.get_lower_corner().end());
               res.insert(res.end(), bounds.get_upper_corner().begin(), bounds.get_upper_corner().end());
             }
             return _wrap_as_numpy_array(std::move(res), 2, bounds.get_number_of_coordinates());
           })
      .def("__getstate__",
           [](const Summand& self) -> nb::ndarray<nb::numpy, char> {
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
      .def("__setstate__", [](Summand& self, nb::ndarray<const char, nb::ndim<1>, nb::numpy> state) {
        new (&self) Summand(Gudhi::multi_persistence::deserialize_summand_from_python<T>(state));
      });
}

template <typename Desc, typename Class>
void bind_float_module_methods(Class& cls) {
  if constexpr (Desc::is_float) {
    using T = typename Desc::value_type;
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
        // .def("distance_to", &Module::compute_distance_to_iterable, "pts"_a, "signed"_a = false, "n_jobs"_a = 0)
        .def("distance_to", &Module::compute_distance_to_tensor, "pts"_a, "signed"_a = false, "n_jobs"_a = 0)
        .def("get_interleavings", &Module::compute_interleavings)
        .def("get_interleavings", &Module::compute_interleavings_from_box);
  }
}

template <typename Desc>
void bind_module_class(nb::module_& m) {
  using T = typename Desc::value_type;
  using Module = Gudhi::multi_persistence::Module_interface<T>;
  using Summand = typename Module::Summand_t;
  using NDArray1 = typename Module::Tensor1D;
  using NDArray2 = typename Module::Tensor2D;
  using IntNDArray = typename Module::template IntTensor1D<int>;

  std::string iterator_name = std::string("_PyModuleIterator_") + std::string(Desc::short_name);

  auto module_cls =
      nb::class_<Module>(m, Desc::module_name.data())
          .def(nb::init<>())
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
                         try {
                           if (nb::ndarray_check(box.ptr())) {
                             self.set_box(nb::cast<NDArray2>(box));
                             return;
                           }
                           self.set_box(nb::cast<std::vector<std::vector<T>>>(box));
                           return;
                         } catch (const nb::cast_error&) {
                           throw nb::type_error(
                               "Box has to be set with an array or a nested sequence of shape (2, p).");
                         }
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
                 return self.get_box_view_ro();
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
          .def("get_module_of_degrees", nb::overload_cast<IntNDArray>(&Module::get_module_of_degrees))
          .def("get_module_of_degrees", nb::overload_cast<const std::vector<int>&>(&Module::get_module_of_degrees))
          .def("to_flat_idx", nb::overload_cast<const std::vector<NDArray1>&>(&Module::get_flat_indices_in_grid))
          .def("to_flat_idx", nb::overload_cast<const std::vector<std::vector<T>>&>(&Module::get_flat_indices_in_grid))
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
            new (&self) Module(Gudhi::multi_persistence::deserialize_module_from_python<T>(state));
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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "ext_interface/backend_log_policy.hpp"
#include "graph_mph0/nanobind_interface.hpp"
#include "gudhi/Multi_persistence/Box.h"
#include <python_interfaces/numpy_utils.h>
#include <gudhi/multiparameter_module_approximation.h>
#include "nanobind_dense_array_utils.hpp"
#include "nanobind_object_utils.hpp"

#include "_slicer_nanobind.h"
#include "_slicer_algorithms_nanobind.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpnb {

using tensor_dtype = int32_t;
using indices_type = int32_t;
using signed_measure_type = std::pair<std::vector<std::vector<indices_type>>, std::vector<tensor_dtype>>;

using multipers::nanobind_dense_utils::matrix_from_array;
using multipers::nanobind_dense_utils::vector_from_array;
using multipers::nanobind_helpers::SlicerDescriptorList;
using multipers::nanobind_helpers::type_list;
using multipers::nanobind_utils::cast_vector;

template <typename Desc>
inline constexpr bool is_kcritical_contiguous_f64_matrix_slicer_v =
    std::is_same_v<typename Desc::value_type, double> && !Desc::is_vine && Desc::is_kcritical &&
    !Desc::is_degree_rips && Desc::column_type == std::string_view("UNORDERED_SET") &&
    Desc::backend_type == std::string_view("Matrix") && Desc::filtration_container == std::string_view("Contiguous");

template <typename Desc>
inline constexpr bool is_contiguous_f64_matrix_slicer_v =
    std::is_same_v<typename Desc::value_type, double> && !Desc::is_vine && !Desc::is_kcritical &&
    !Desc::is_degree_rips && Desc::column_type == std::string_view("UNORDERED_SET") &&
    Desc::backend_type == std::string_view("Matrix") && Desc::filtration_container == std::string_view("Contiguous");

template <typename List>
struct contiguous_f64_matrix_slicer_desc_impl;

template <>
struct contiguous_f64_matrix_slicer_desc_impl<type_list<>> {
  using type = void;
  static constexpr bool found = false;
  static constexpr int matches = 0;
};

template <typename Head, typename... Tail>
struct contiguous_f64_matrix_slicer_desc_impl<type_list<Head, Tail...>> {
  using tail = contiguous_f64_matrix_slicer_desc_impl<type_list<Tail...>>;
  static constexpr bool is_match = is_contiguous_f64_matrix_slicer_v<Head>;
  static constexpr bool found = is_match || tail::found;
  static constexpr int matches = tail::matches + (is_match ? 1 : 0);
  using type = std::conditional_t<is_match, Head, typename tail::type>;
};

using ContiguousF64MatrixSlicerDesc = typename contiguous_f64_matrix_slicer_desc_impl<SlicerDescriptorList>::type;

static_assert(!std::is_void_v<ContiguousF64MatrixSlicerDesc>,
              "Expected exactly one one-critical contiguous float64 matrix slicer template.");
static_assert(contiguous_f64_matrix_slicer_desc_impl<SlicerDescriptorList>::matches == 1,
              "One-critical contiguous float64 matrix slicer template must be unique.");

template <typename List>
struct kcritical_contiguous_f64_matrix_slicer_desc_impl;

template <>
struct kcritical_contiguous_f64_matrix_slicer_desc_impl<type_list<>> {
  using type = void;
  static constexpr bool found = false;
  static constexpr int matches = 0;
};

template <typename Head, typename... Tail>
struct kcritical_contiguous_f64_matrix_slicer_desc_impl<type_list<Head, Tail...>> {
  using tail = kcritical_contiguous_f64_matrix_slicer_desc_impl<type_list<Tail...>>;
  static constexpr bool is_match = is_kcritical_contiguous_f64_matrix_slicer_v<Head>;
  static constexpr bool found = is_match || tail::found;
  static constexpr int matches = tail::matches + (is_match ? 1 : 0);
  using type = std::conditional_t<is_match, Head, typename tail::type>;
};

using KcriticalContiguousF64MatrixSlicerDesc =
    typename kcritical_contiguous_f64_matrix_slicer_desc_impl<SlicerDescriptorList>::type;

static_assert(!std::is_void_v<KcriticalContiguousF64MatrixSlicerDesc>,
              "Expected exactly one k-critical contiguous float64 matrix slicer template.");
static_assert(kcritical_contiguous_f64_matrix_slicer_desc_impl<SlicerDescriptorList>::matches == 1,
              "k-critical contiguous float64 matrix slicer template must be unique.");

}  // namespace mpnb

NB_MODULE(_slicer_nanobind, m) {
  m.doc() = "nanobind slicer bindings";
  nb::list available_slicers;

  m.def("_get_backend_log_mask", []() { return multipers::backend_log_policy::get_backend_log_mask(); });
  m.def(
      "_set_backend_log_mask",
      [](uint32_t mask) { multipers::backend_log_policy::set_backend_log_mask(mask); },
      "mask"_a);

  mpnb::bind_generator_basis(m);
  mpnb::bind_all_slicers(mpnb::SlicerDescriptorList{}, m, available_slicers);

  m.def("_graph_mph0_minimal_presentation",
        &mpnb::graph_mph0_minimal_presentation,
        "slicer"_a,
        "degree"_a,
        "full_resolution"_a);

  m.def(
      "_compute_hilbert_signed_measure",
      [](nb::handle slicer,
         nb::handle grid_shape_handle,
         nb::handle degrees_handle,
         bool zero_pad,
         mpnb::indices_type n_jobs,
         bool verbose,
         bool ignore_inf) {
        auto grid_shape = mpnb::cast_vector<mpnb::indices_type>(grid_shape_handle);
        auto degrees = mpnb::cast_vector<mpnb::indices_type>(degrees_handle);
        std::vector<mpnb::indices_type> full_shape;
        full_shape.reserve(grid_shape.size() + 1);
        full_shape.push_back((mpnb::indices_type)degrees.size());
        full_shape.insert(full_shape.end(), grid_shape.begin(), grid_shape.end());
        size_t width = grid_shape.size() + 1;
        size_t total = 1;
        for (mpnb::indices_type value : full_shape) {
          total *= (size_t)value;
        }
        std::vector<mpnb::tensor_dtype> container(total, 0);
        return mpnb::compute_hilbert_signed_measure(mpnb::SlicerDescriptorList{},
                                                    slicer,
                                                    container,
                                                    full_shape,
                                                    degrees,
                                                    width,
                                                    zero_pad,
                                                    n_jobs,
                                                    verbose,
                                                    ignore_inf);
      },
      "slicer"_a,
      "grid_shape"_a,
      "degrees"_a,
      "zero_pad"_a = false,
      "n_jobs"_a = 0,
      "verbose"_a = false,
      "ignore_inf"_a = true);

  m.def(
      "_compute_hilbert_signed_measure_sparse",
      [](nb::handle slicer,
         nb::handle grid_shape_handle,
         nb::handle degrees_handle,
         bool zero_pad,
         mpnb::indices_type n_jobs,
         bool ignore_inf) {
        auto grid_shape = mpnb::cast_vector<mpnb::indices_type>(grid_shape_handle);
        auto degrees = mpnb::cast_vector<mpnb::indices_type>(degrees_handle);
        const size_t width = grid_shape.size() + 1;
        return mpnb::compute_hilbert_signed_measure_sparse(
            mpnb::SlicerDescriptorList{}, slicer, grid_shape, degrees, width, zero_pad, n_jobs, ignore_inf);
      },
      "slicer"_a,
      "grid_shape"_a,
      "degrees"_a,
      "zero_pad"_a = false,
      "n_jobs"_a = 0,
      "ignore_inf"_a = true);

  m.def(
      "_compute_rank_tensor",
      [](nb::handle slicer,
         nb::handle grid_shape_handle,
         nb::handle degrees_handle,
         mpnb::indices_type n_jobs,
         bool ignore_inf) {
        auto grid_shape = mpnb::cast_vector<mpnb::indices_type>(grid_shape_handle);
        auto degrees = mpnb::cast_vector<mpnb::indices_type>(degrees_handle);
        std::vector<mpnb::indices_type> full_shape;
        full_shape.reserve(1 + 2 * grid_shape.size());
        full_shape.push_back((mpnb::indices_type)degrees.size());
        full_shape.insert(full_shape.end(), grid_shape.begin(), grid_shape.end());
        full_shape.insert(full_shape.end(), grid_shape.begin(), grid_shape.end());
        size_t total = 1;
        for (mpnb::indices_type value : full_shape) {
          total *= (size_t)value;
        }
        std::vector<mpnb::tensor_dtype> container(total, 0);
        return mpnb::compute_rank_tensor(
            mpnb::SlicerDescriptorList{}, slicer, container, full_shape, degrees, total, n_jobs, ignore_inf);
      },
      "slicer"_a,
      "grid_shape"_a,
      "degrees"_a,
      "n_jobs"_a = 0,
      "ignore_inf"_a = true);

  m.def(
      "_compute_rank_signed_measure_sparse",
      [](nb::handle slicer,
         nb::handle grid_shape_handle,
         nb::handle degrees_handle,
         bool zero_pad,
         mpnb::indices_type n_jobs,
         bool ignore_inf) {
        auto grid_shape = mpnb::cast_vector<mpnb::indices_type>(grid_shape_handle);
        auto degrees = mpnb::cast_vector<mpnb::indices_type>(degrees_handle);
        const size_t width = 1 + 2 * grid_shape.size();
        return mpnb::compute_rank_signed_measure_sparse(
            mpnb::SlicerDescriptorList{}, slicer, grid_shape, degrees, width, zero_pad, n_jobs, ignore_inf);
      },
      "slicer"_a,
      "grid_shape"_a,
      "degrees"_a,
      "zero_pad"_a = false,
      "n_jobs"_a = 0,
      "ignore_inf"_a = true);

  m.def(
      "_compute_module_approximation_from_slicer",
      [](nb::handle slicer,
         nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> direction,
         double max_error,
         nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> box_array,
         bool threshold,
         bool complete,
         bool verbose,
         int n_jobs) {
        if (box_array.shape(0) != 2 || (direction.shape(0) != 0 && box_array.shape(1) != direction.shape(0))) {
          throw nb::value_error("box must have shape (2, num_parameters).");
        }
        auto direction_values = mpnb::vector_from_array(direction);
        auto box_values = mpnb::matrix_from_array(box_array);
        Gudhi::multi_persistence::Box<double> box(box_values[0], box_values[1]);
        return mpnb::compute_module_approximation_from_slicer(mpnb::SlicerDescriptorList{},
                                                              slicer,
                                                              direction_values,
                                                              max_error,
                                                              box,
                                                              threshold,
                                                              complete,
                                                              verbose,
                                                              n_jobs);
      },
      "slicer"_a,
      "direction"_a,
      "max_error"_a,
      "box"_a,
      "threshold"_a = false,
      "complete"_a = true,
      "verbose"_a = false,
      "n_jobs"_a = -1);

  m.def(
      "_get_slicer_class",
      [](bool is_vineyard,
         bool is_k_critical,
         nb::handle dtype,
         std::string col,
         std::string pers_backend,
         std::string filtration_container) {
        return mpnb::get_slicer_class(mpnb::SlicerDescriptorList{},
                                      is_vineyard,
                                      is_k_critical,
                                      dtype,
                                      std::move(col),
                                      std::move(pers_backend),
                                      std::move(filtration_container));
      },
      "is_vineyard"_a,
      "is_k_critical"_a,
      "dtype"_a,
      "col"_a,
      "pers_backend"_a,
      "filtration_container"_a);

  // m.def("_get_slicer_class_from_template_id", &mpnb::get_slicer_class_from_template_id, "template_id"_a);

  // mpnb::bind_bitmap_builders(mpnb::SlicerDescriptorList{}, m);

  m.attr("available_slicers") = available_slicers;
}

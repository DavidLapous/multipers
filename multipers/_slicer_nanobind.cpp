#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "ext_interface/backend_log_policy.hpp"
#include "graph_mph0/nanobind_interface.hpp"
#include "nanobind_object_utils.hpp"

#include "_slicer_nanobind.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpnb {

using multipers::nanobind_helpers::SlicerDescriptorList;
using multipers::nanobind_helpers::type_list;
using multipers::nanobind_utils::lowercase_copy;
using multipers::nanobind_utils::numpy_dtype_name;

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

template <typename... Ds>
inline nb::object get_slicer_class(type_list<Ds...>,
                                   bool is_vineyard,
                                   bool is_k_critical,
                                   const nb::handle& dtype,
                                   std::string col,
                                   std::string pers_backend,
                                   std::string filtration_container) {
  std::string dtype_name = numpy_dtype_name(dtype);
  col = lowercase_copy(std::move(col));
  pers_backend = lowercase_copy(std::move(pers_backend));
  filtration_container = lowercase_copy(std::move(filtration_container));
  bool matched = false;
  nb::object result;
  (
      [&]<typename D>() {
        if (!matched && D::is_vine == is_vineyard && D::is_kcritical == is_k_critical && D::dtype_name == dtype_name &&
            lowercase_copy(std::string(D::column_type)) == col &&
            lowercase_copy(std::string(D::backend_type)) == pers_backend &&
            lowercase_copy(std::string(D::filtration_container)) == filtration_container) {
          result = nb::borrow<nb::object>(nb::type<typename D::interface>());
          matched = true;
        }
      }.template operator()<Ds>(),
      ...);
  if (!matched) {
    throw nb::value_error("Unimplemented slicer combination.");
  }
  return result;
}

}  // namespace mpnb

NB_MODULE(_slicer_nanobind, m) {
  using tensor_dtype = int32_t;
  using indices_type = int32_t;

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
        "full_resolution"_a,
        "finite_grid_masks"_a = nb::none());

  m.def("_compute_hilbert_signed_measure",
        &mpnb::compute_hilbert_signed_measure<tensor_dtype, indices_type>,
        "slicer"_a,
        "grid_shape"_a,
        "degrees"_a,
        "zero_pad"_a = false,
        "n_jobs"_a = 0,
        "verbose"_a = false,
        "ignore_inf"_a = true);

  m.def("_compute_hilbert_signed_measure_sparse",
        &mpnb::compute_hilbert_signed_measure_sparse<tensor_dtype, indices_type>,
        "slicer"_a,
        "grid_shape"_a,
        "degrees"_a,
        "zero_pad"_a = false,
        "n_jobs"_a = 0,
        "ignore_inf"_a = true,
        "verbose"_a = false);

  m.def("_compute_rank_tensor",
        &mpnb::compute_rank_tensor<tensor_dtype, indices_type>,
        "slicer"_a,
        "grid_shape"_a,
        "degrees"_a,
        "n_jobs"_a = 0,
        "ignore_inf"_a = true);

  m.def("_compute_rank_signed_measure_sparse",
        &mpnb::compute_rank_signed_measure_sparse<tensor_dtype, indices_type>,
        "slicer"_a,
        "grid_shape"_a,
        "degrees"_a,
        "zero_pad"_a = false,
        "n_jobs"_a = 0,
        "ignore_inf"_a = true);

  m.def("_compute_module_approximation_from_slicer",
        &mpnb::compute_module_approximation_from_slicer<double>,
        "slicer"_a,
        "direction"_a,
        "max_error"_a,
        "box"_a,
        "threshold"_a = false,
        "complete"_a = true,
        "verbose"_a = false,
        "n_jobs"_a = -1);

  m.def("_get_slicer_class",
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
        });

  m.attr("available_slicers") = available_slicers;
}

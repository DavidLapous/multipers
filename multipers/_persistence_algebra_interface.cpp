#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "ext_interface/persistence_algebra_interface.hpp"
#include "ext_interface/nanobind_registry_runtime.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mppai {

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;

#if MULTIPERS_HAS_PERSISTENCE_ALGEBRA_INTERFACE
inline nb::list cast_degrees(const std::vector<multipers::persistence_algebra_detail::pa_degree>& degrees) {
  nb::list out;
  for (const auto& degree : degrees) {
    out.append(nb::make_tuple(degree.first, degree.second));
  }
  return out;
}

template <typename Matrix>
inline nb::list cast_columns(const Matrix& matrix) {
  nb::list out;
  for (const auto& col : matrix.data) {
    out.append(nb::cast(col));
  }
  return out;
}
#endif

inline nb::object minimal_presentation_for_target(nb::object target, int degree, bool full_resolution) {
  auto& input_wrapper = nb::cast<CanonicalWrapper&>(target);
  auto complex =
      multipers::persistence_algebra_minpres_contiguous_interface(input_wrapper.truc, degree, full_resolution);
  return multipers::nanobind_helpers::build_canonical_contiguous_f64_slicer_object_from_complex(target, complex);
}

}  // namespace mppai

NB_MODULE(_persistence_algebra_interface, m) {
  auto available = []() { return multipers::persistence_algebra_interface_available(); };
  m.def("_is_available", available);
  m.def("available", available);
  m.def("require", [available]() {
    if (!available()) {
      throw std::runtime_error(
          "Persistence-Algebra interface is not available in this build. Rebuild multipers with Persistence-Algebra support to enable this backend.");
    }
  });

  m.def(
      "_debug_degree_block",
      [](nb::object slicer, int degree, bool with_homology) {
        if (!multipers::persistence_algebra_interface_available()) {
          throw std::runtime_error("Persistence-Algebra interface is not available.");
        }
#if MULTIPERS_HAS_PERSISTENCE_ALGEBRA_INTERFACE
        nb::object target = multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
        auto& input_wrapper = nb::cast<mppai::CanonicalWrapper&>(target);
        auto first = multipers::persistence_algebra_detail::build_boundary_matrix(input_wrapper.truc, degree);
        auto second = multipers::persistence_algebra_detail::build_boundary_matrix(input_wrapper.truc, degree + 1);
        auto first_sorted = first;
        first_sorted.sort_rows_lexicographically();
        const auto relation_row_permutation = first_sorted.sort_columns_lexicographically_with_output();
        auto kernel = first_sorted.graded_kernel();
        auto second_aligned = second;
        second_aligned.permute_rows_graded(relation_row_permutation);
        second_aligned.sort_columns_lexicographically();
        second_aligned.sort_rows_lexicographically();

        nb::dict out;
        out["first_row_degrees"] = mppai::cast_degrees(first.row_degrees);
        out["first_col_degrees"] = mppai::cast_degrees(first.col_degrees);
        out["first_sorted_row_degrees"] = mppai::cast_degrees(first_sorted.row_degrees);
        out["first_sorted_col_degrees"] = mppai::cast_degrees(first_sorted.col_degrees);
        out["second_row_degrees"] = mppai::cast_degrees(second.row_degrees);
        out["second_col_degrees"] = mppai::cast_degrees(second.col_degrees);
        out["kernel_row_degrees"] = mppai::cast_degrees(kernel.row_degrees);
        out["kernel_col_degrees"] = mppai::cast_degrees(kernel.col_degrees);
        out["aligned_second_row_degrees"] = mppai::cast_degrees(second_aligned.row_degrees);
        out["aligned_second_col_degrees"] = mppai::cast_degrees(second_aligned.col_degrees);
        out["first_shapes"] = nb::make_tuple(first.get_num_rows(), first.get_num_cols());
        out["first_sorted_shapes"] = nb::make_tuple(first_sorted.get_num_rows(), first_sorted.get_num_cols());
        out["second_shapes"] = nb::make_tuple(second.get_num_rows(), second.get_num_cols());
        out["kernel_shapes"] = nb::make_tuple(kernel.get_num_rows(), kernel.get_num_cols());
        out["first_sorted_data"] = mppai::cast_columns(first_sorted);
        out["second_aligned_data"] = mppai::cast_columns(second_aligned);
        out["kernel_data"] = mppai::cast_columns(kernel);

        auto ambient = multipers::persistence_algebra_detail::pa_matrix(0, kernel.get_num_rows());
        ambient.row_degrees = kernel.row_degrees;
        auto kernel_presentation = ambient.submodule_generated_by(kernel);
        auto kernel_for_pullback = kernel;
        auto image_in_kernel = kernel_for_pullback.inverse_image(ambient, second_aligned);
        auto image_submodule_presentation = image_in_kernel.presentation_of_submodule(kernel_presentation);
        auto quotient_presentation = kernel_presentation;
        quotient_presentation.quotient_by(image_in_kernel);
        out["kernel_presentation_row_degrees"] = mppai::cast_degrees(kernel_presentation.row_degrees);
        out["kernel_presentation_col_degrees"] = mppai::cast_degrees(kernel_presentation.col_degrees);
        out["kernel_presentation_shapes"] =
            nb::make_tuple(kernel_presentation.get_num_rows(), kernel_presentation.get_num_cols());
        out["kernel_presentation_data"] = mppai::cast_columns(kernel_presentation);
        out["image_in_kernel_row_degrees"] = mppai::cast_degrees(image_in_kernel.row_degrees);
        out["image_in_kernel_col_degrees"] = mppai::cast_degrees(image_in_kernel.col_degrees);
        out["image_in_kernel_shapes"] = nb::make_tuple(image_in_kernel.get_num_rows(), image_in_kernel.get_num_cols());
        out["image_in_kernel_data"] = mppai::cast_columns(image_in_kernel);
        out["image_submodule_presentation_row_degrees"] = mppai::cast_degrees(image_submodule_presentation.row_degrees);
        out["image_submodule_presentation_col_degrees"] = mppai::cast_degrees(image_submodule_presentation.col_degrees);
        out["image_submodule_presentation_shapes"] =
            nb::make_tuple(image_submodule_presentation.get_num_rows(), image_submodule_presentation.get_num_cols());
        out["image_submodule_presentation_data"] = mppai::cast_columns(image_submodule_presentation);
        out["quotient_presentation_row_degrees"] = mppai::cast_degrees(quotient_presentation.row_degrees);
        out["quotient_presentation_col_degrees"] = mppai::cast_degrees(quotient_presentation.col_degrees);
        out["quotient_presentation_shapes"] =
            nb::make_tuple(quotient_presentation.get_num_rows(), quotient_presentation.get_num_cols());
        out["quotient_presentation_data"] = mppai::cast_columns(quotient_presentation);
        if (with_homology) {
          auto sequence = graded_linalg::R2Sequence<multipers::persistence_algebra_detail::pa_index>(first, second);
          auto homology = sequence.get_homology();
          out["homology_row_degrees"] = mppai::cast_degrees(homology.row_degrees);
          out["homology_col_degrees"] = mppai::cast_degrees(homology.col_degrees);
          out["homology_shapes"] = nb::make_tuple(homology.get_num_rows(), homology.get_num_cols());
          out["homology_data"] = mppai::cast_columns(homology);
        }
        return out;
#else
        (void)slicer;
        (void)degree;
        (void)with_homology;
        throw std::runtime_error("Persistence-Algebra interface is not available.");
#endif
      },
      "slicer"_a,
      "degree"_a,
      "with_homology"_a = false);

  m.def(
      "minimal_presentation",
      [](nb::object slicer,
         int degree,
         bool full_resolution,
         bool use_clearing,
         bool use_chunk,
         bool verbose) {
        (void)use_clearing;
        (void)use_chunk;
        (void)verbose;
        if (!multipers::persistence_algebra_interface_available()) {
          throw std::runtime_error("Persistence-Algebra interface is not available.");
        }
        nb::object target = multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
        nb::object out = mppai::minimal_presentation_for_target(target, degree, full_resolution);
        if (target.ptr() == slicer.ptr()) {
          return out;
        }
        return multipers::nanobind_helpers::astype_slicer_to_original_type(slicer, out);
      },
      "slicer"_a,
      "degree"_a,
      "full_resolution"_a = true,
      "use_clearing"_a = true,
      "use_chunk"_a = true,
      "verbose"_a = false);

  m.def(
      "death_curve_presentation",
      [](nb::object slicer, int degree) {
        if (!multipers::persistence_algebra_interface_available()) {
          throw std::runtime_error("Persistence-Algebra interface is not available.");
        }
        nb::object target = multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
        auto& input_wrapper = nb::cast<mppai::CanonicalWrapper&>(target);
        auto complex = multipers::persistence_algebra_death_curve_contiguous_interface(input_wrapper.truc, degree);
        nb::object out = multipers::nanobind_helpers::build_canonical_contiguous_f64_slicer_object_from_complex(target, complex);
        if (target.ptr() == slicer.ptr()) {
          return out;
        }
        return multipers::nanobind_helpers::astype_slicer_to_original_type(slicer, out);
      },
      "slicer"_a,
      "degree"_a);
}

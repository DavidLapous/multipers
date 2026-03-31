#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "ext_interface/mpfree_interface.hpp"

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
#include "mpfree/global.h"

namespace mpmi {

inline void set_backend_stdout(bool enabled) { mpfree::verbose = enabled; }

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;

inline nb::object ensure_supported_target(nb::object slicer) {
  return multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
}

nb::object minimal_presentation_for_target(nb::object target,
                                           int degree,
                                           bool full_resolution,
                                           bool use_clearing,
                                           bool use_chunk,
                                           bool backend_stdout,
                                           bool keep_generators) {
  auto& input_wrapper = nb::cast<CanonicalWrapper&>(target);
  nb::object out = target.type()();
  auto& out_wrapper = nb::cast<CanonicalWrapper&>(out);

  if (!keep_generators) {
    auto complex = multipers::mpfree_minpres_contiguous_interface(
        input_wrapper.truc, degree, full_resolution, use_chunk, use_clearing, backend_stdout);
    multipers::build_slicer_from_complex(out_wrapper.truc, complex);
    return out;
  }

  auto result = multipers::mpfree_minpres_with_generators_contiguous_interface(
      input_wrapper.truc, degree, full_resolution, use_chunk, use_clearing, backend_stdout);
  multipers::build_slicer_from_complex(out_wrapper.truc, result.first);

  const auto dimensions = input_wrapper.truc.get_dimensions();
  const auto& boundaries = input_wrapper.truc.get_boundaries();
  auto& filtrations = input_wrapper.truc.get_filtration_values();
  std::vector<size_t> degree_indices;
  degree_indices.reserve(dimensions.size());
  for (size_t i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == degree) {
      degree_indices.push_back(i);
    }
  }
  std::stable_sort(degree_indices.begin(), degree_indices.end(), [&](size_t a, size_t b) {
    const auto& fa = filtrations[a];
    const auto& fb = filtrations[b];
    return fa(0, 1) < fb(0, 1) || (fa(0, 1) == fb(0, 1) && fa(0, 0) < fb(0, 0));
  });

  if (result.second.row_indices.size() != result.second.row_grades.size()) {
    throw std::runtime_error("mpfree generator-basis extraction failed: row count mismatch.");
  }
  for (size_t i = 0; i < result.second.row_indices.size(); ++i) {
    const auto row_idx = static_cast<size_t>(result.second.row_indices[i]);
    if (row_idx >= degree_indices.size()) {
      throw std::runtime_error("mpfree generator-basis extraction failed: row index out of range.");
    }
    const auto& filtration = filtrations[degree_indices[row_idx]];
    const auto& grade = result.second.row_grades[i];
    if (filtration(0, 0) != grade.first || filtration(0, 1) != grade.second) {
      throw std::runtime_error(
          "mpfree generator-basis extraction failed: row grades do not match the original degree block.");
    }
  }

  nb::list row_boundaries;
  for (auto raw_row_idx : result.second.row_indices) {
    const auto row_idx = static_cast<size_t>(raw_row_idx);
    const auto idx = degree_indices[row_idx];
    std::vector<uint32_t> boundary;
    boundary.reserve(boundaries[idx].size());
    for (auto value : boundaries[idx]) {
      boundary.push_back(static_cast<uint32_t>(value));
    }
    row_boundaries.append(nb::cast(std::move(boundary)));
  }

  nb::dict basis;
  basis["degree"] = degree;
  basis["row_boundaries"] = std::move(row_boundaries);
  basis["columns"] = nb::cast(std::move(result.second.columns));
  basis["row_grades"] = nb::cast(std::move(result.second.row_grades));
  basis["column_grades"] = nb::cast(std::move(result.second.column_grades));
  out.attr("_generator_basis") = std::move(basis);
  return out;
}

}  // namespace mpmi
#endif

NB_MODULE(_mpfree_interface, m) {
  bool ext_log_enabled = false;
  try {
    ext_log_enabled = nb::cast<bool>(nb::module_::import_("multipers.logs").attr("ext_log_enabled")());
  } catch (...) {
    ext_log_enabled = false;
  }

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
  mpmi::set_backend_stdout(ext_log_enabled);
#else
  (void)ext_log_enabled;
#endif

  m.def("_is_available", []() { return multipers::mpfree_interface_available(); });

#if MULTIPERS_DISABLE_MPFREE_INTERFACE
  m.def("_set_backend_stdout", [](bool) {}, "enabled"_a);
#else
  m.def("_set_backend_stdout", [](bool enabled) { mpmi::set_backend_stdout(enabled); }, "enabled"_a);
#endif

  m.def(
      "minimal_presentation",
      [](nb::object slicer,
         int degree,
         bool full_resolution,
         bool use_clearing,
         bool use_chunk,
         bool keep_generators,
         bool verbose,
         bool backend_stdout) {
#if MULTIPERS_DISABLE_MPFREE_INTERFACE
        throw std::runtime_error("mpfree in-memory interface is disabled at compile time.");
#else
        if (!multipers::mpfree_interface_available()) {
          throw std::runtime_error("mpfree in-memory interface is not available.");
        }
        nb::object target = mpmi::ensure_supported_target(slicer);
        nb::object out = mpmi::minimal_presentation_for_target(
            target, degree, full_resolution, use_clearing, use_chunk, backend_stdout, keep_generators);
        if (target.ptr() == slicer.ptr()) {
          return out;
        }
        return multipers::nanobind_helpers::astype_slicer_to_original_type(slicer, out);
#endif
      },
      "slicer"_a,
      "degree"_a,
      "full_resolution"_a = true,
      "use_clearing"_a = true,
      "use_chunk"_a = true,
      "keep_generators"_a = false,
      "verbose"_a = false,
      "_backend_stdout"_a = false);
}

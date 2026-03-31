#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "ext_interface/2pac_interface.hpp"

#if !MULTIPERS_DISABLE_2PAC_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_2PAC_INTERFACE
namespace mtpi {

inline nb::object numpy_float64() { return nb::module_::import_("numpy").attr("float64"); }

inline nb::object ensure_supported_target(nb::object slicer) {
  return slicer.attr("astype")("vineyard"_a = false,
                               "kcritical"_a = false,
                               "dtype"_a = numpy_float64(),
                               "col"_a = slicer.attr("col_type"),
                               "pers_backend"_a = "matrix",
                               "filtration_container"_a = "contiguous");
}

nb::object minimal_presentation_for_target(nb::object target,
                                           int degree,
                                           bool full_resolution,
                                           bool use_clearing,
                                           bool use_chunk,
                                           bool verbose,
                                           bool backend_stdout,
                                           bool keep_generators) {
  return multipers::nanobind_helpers::visit_slicer_wrapper(
      target, [&]<typename Desc>(typename Desc::wrapper& input_wrapper) {
        nb::object out = target.type()();
        auto& out_wrapper = nb::cast<typename Desc::wrapper&>(out);

        if (!keep_generators) {
          auto complex = multipers::twopac_minpres_contiguous_interface(
              input_wrapper.truc, degree, full_resolution, use_chunk, use_clearing, backend_stdout || verbose);
          multipers::build_slicer_from_complex(out_wrapper.truc, complex);
          return out;
        }

        auto result = multipers::twopac_minpres_with_generators_contiguous_interface(
            input_wrapper.truc, degree, full_resolution, use_chunk, use_clearing, backend_stdout || verbose);
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
          throw std::runtime_error("2pac generator-basis extraction failed: row count mismatch.");
        }
        for (size_t i = 0; i < result.second.row_indices.size(); ++i) {
          const auto row_idx = static_cast<size_t>(result.second.row_indices[i]);
          if (row_idx >= degree_indices.size()) {
            throw std::runtime_error("2pac generator-basis extraction failed: row index out of range.");
          }
          const auto& filtration = filtrations[degree_indices[row_idx]];
          const auto& grade = result.second.row_grades[i];
          if (filtration(0, 0) != grade.first || filtration(0, 1) != grade.second) {
            throw std::runtime_error(
                "2pac generator-basis extraction failed: row grades do not match the original degree block.");
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
      });
}

}  // namespace mtpi
#endif

NB_MODULE(_2pac_interface, m) {
  m.def("_is_available", []() { return multipers::twopac_interface_available(); });

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
#if MULTIPERS_DISABLE_2PAC_INTERFACE
        throw std::runtime_error("2pac in-memory interface is disabled at compile time.");
#else
        if (!multipers::twopac_interface_available()) {
          throw std::runtime_error("2pac in-memory interface is not available.");
        }
        nb::object target = mtpi::ensure_supported_target(slicer);
        nb::object out = mtpi::minimal_presentation_for_target(
            target, degree, full_resolution, use_clearing, use_chunk, verbose, backend_stdout, keep_generators);
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

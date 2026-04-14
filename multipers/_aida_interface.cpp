#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ext_interface/aida_interface.hpp"

#if !MULTIPERS_DISABLE_AIDA_INTERFACE
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_AIDA_INTERFACE
namespace mpaida {

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;

inline nb::object ensure_supported_target(nb::object slicer) {
  return multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
}

inline nb::object to_colexical_target(const nb::object& target) {
  return nb::cast(multipers::nanobind_helpers::colexical_slicer_copy(nb::cast<const CanonicalWrapper&>(target)));
}

multipers::nanobind_helpers::BifiltrationMinpresDegreeBlock build_input_from_slicer(const CanonicalWrapper& wrapper) {
  if (wrapper.minpres_degree < 0) {
    throw std::runtime_error("AIDA takes a minimal presentation as an input.");
  }
  if (wrapper.truc.get_number_of_parameters() != 2) {
    throw std::runtime_error("AIDA is only compatible with 2-parameter minimal presentations.");
  }
  return multipers::nanobind_helpers::extract_bifiltration_minpres_degree_block(wrapper, wrapper.minpres_degree);
}

template <typename Summand>
nb::object summand_to_slicer(nb::object target,
                             const Summand& summand,
                             int degree,
                             bool is_squeezed,
                             const nb::object& filtration_grid) {
  std::vector<int> dimensions(summand.row_degrees.size() + summand.col_degrees.size());
  std::fill_n(dimensions.begin(), summand.row_degrees.size(), degree);
  std::fill(dimensions.begin() + static_cast<std::ptrdiff_t>(summand.row_degrees.size()), dimensions.end(), degree + 1);

  std::vector<std::vector<int> > boundaries(dimensions.size());
  for (std::size_t i = 0; i < summand.matrix.size(); ++i) {
    boundaries[summand.row_degrees.size() + i] = summand.matrix[i];
  }

  std::vector<std::pair<double, double> > filtration_values;
  filtration_values.reserve(summand.row_degrees.size() + summand.col_degrees.size());
  filtration_values.insert(filtration_values.end(), summand.row_degrees.begin(), summand.row_degrees.end());
  filtration_values.insert(filtration_values.end(), summand.col_degrees.begin(), summand.col_degrees.end());

  nb::object compact_grid = nb::none();
  if (is_squeezed) {
    std::vector<std::vector<int64_t> > used_coordinates(2);
    used_coordinates[0].reserve(filtration_values.size());
    used_coordinates[1].reserve(filtration_values.size());
    for (const auto& degree : filtration_values) {
      used_coordinates[0].push_back(multipers::nanobind_helpers::squeezed_raw_index_from_value(degree.first, 0));
      used_coordinates[1].push_back(multipers::nanobind_helpers::squeezed_raw_index_from_value(degree.second, 1));
    }
    auto compacted = multipers::nanobind_helpers::compact_squeezed_filtration_grid(
        filtration_grid, std::move(used_coordinates));
    compact_grid = compacted.filtration_grid;
    for (auto& degree : filtration_values) {
      degree.first = multipers::nanobind_helpers::remap_squeezed_coordinate(degree.first, 0, compacted.remap);
      degree.second = multipers::nanobind_helpers::remap_squeezed_coordinate(degree.second, 1, compacted.remap);
    }
  }

  auto complex = multipers::build_contiguous_f64_slicer_from_output(filtration_values, boundaries, dimensions);

  nb::object out = multipers::nanobind_helpers::build_canonical_contiguous_f64_slicer_object_from_complex(target, complex);
  auto& out_wrapper = nb::cast<CanonicalWrapper&>(out);
  out_wrapper.minpres_degree = degree;
  if (is_squeezed) {
    out_wrapper.filtration_grid = compact_grid;
  }
  return out;
}

}  // namespace mpaida
#endif

NB_MODULE(_aida_interface, m) {
  auto available = []() {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
    return false;
#else
    return true;
#endif
  };
  m.def("_is_available", available);
  m.def("available", available);
  m.def("require", [available]() {
    if (!available()) {
      throw std::runtime_error(
          "AIDA interface is not available in this build. Rebuild multipers with AIDA support to enable this backend.");
    }
  });

  m.def(
      "aida",
      [](nb::object s, bool sort, bool verbose, bool progress) {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
        throw std::runtime_error("AIDA interface is disabled at compile time.");
#else
        nb::object target = mpaida::ensure_supported_target(s);
        if (sort) {
          target = mpaida::to_colexical_target(target);
        }
        const auto& target_wrapper = nb::cast<const mpaida::CanonicalWrapper&>(target);
        auto prepared = mpaida::build_input_from_slicer(target_wrapper);

        aida::AIDA_functor functor;
        functor.config.show_info = verbose;
        functor.config.sort_output = false;
        functor.config.sort = sort;
        functor.config.progress = progress;
        auto input = aida::multipers_interface_input<int>(
            prepared.relation_grades, prepared.row_grades, prepared.relation_boundaries);
        auto output = functor.multipers_interface(input);

        nb::list out;
        for (const auto& summand : output.summands) {
          nb::object slicer = mpaida::summand_to_slicer(
              target, summand, prepared.degree, prepared.is_squeezed, prepared.filtration_grid);
          slicer = multipers::nanobind_helpers::rewrap_slicer_output_to_original_type(s, target, slicer);
          out.append(slicer);
        }
        return out;
#endif
      },
      "s"_a,
      "sort"_a = true,
      "verbose"_a = false,
      "progress"_a = false);
}

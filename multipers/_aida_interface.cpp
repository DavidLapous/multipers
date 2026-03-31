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

struct prepared_input {
  std::vector<std::pair<double, double> > row_degree;
  std::vector<std::pair<double, double> > col_degree;
  std::vector<std::vector<int> > matrix;
  nb::object filtration_grid = nb::none();
  int degree = -1;
  bool is_squeezed = false;
};

inline bool has_nonempty_filtration_grid(const nb::handle& grid) {
  if (!grid.is_valid() || grid.is_none() || !nb::hasattr(grid, "__len__") || nb::len(grid) == 0) {
    return false;
  }

  for (nb::handle row : nb::iter(grid)) {
    return nb::hasattr(row, "__len__") && nb::len(row) > 0;
  }
  return false;
}

inline nb::object ensure_supported_target(nb::object slicer) {
  return multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
}

prepared_input build_input_from_slicer(const CanonicalWrapper& wrapper) {
  if (wrapper.minpres_degree < 0) {
    throw std::runtime_error("AIDA takes a minimal presentation as an input.");
  }
  if (wrapper.truc.get_number_of_parameters() != 2) {
    throw std::runtime_error("AIDA is only compatible with 2-parameter minimal presentations.");
  }

  prepared_input out;
  out.degree = wrapper.minpres_degree;
  out.filtration_grid = wrapper.filtration_grid;
  out.is_squeezed = has_nonempty_filtration_grid(wrapper.filtration_grid);

  const auto& dimensions = wrapper.truc.get_dimensions();
  const auto& filtrations = wrapper.truc.get_filtration_values();
  const auto& boundaries = wrapper.truc.get_boundaries();

  const std::size_t row_start = std::lower_bound(dimensions.begin(), dimensions.end(), out.degree) - dimensions.begin();
  const std::size_t row_end =
      std::lower_bound(dimensions.begin(), dimensions.end(), out.degree + 1) - dimensions.begin();
  const std::size_t col_end =
      std::lower_bound(dimensions.begin(), dimensions.end(), out.degree + 2) - dimensions.begin();

  out.row_degree.reserve(row_end - row_start);
  for (std::size_t i = row_start; i < row_end; ++i) {
    out.row_degree.emplace_back(filtrations[i](0, 0), filtrations[i](0, 1));
  }

  out.col_degree.reserve(col_end - row_end);
  out.matrix.reserve(col_end - row_end);
  for (std::size_t i = row_end; i < col_end; ++i) {
    out.col_degree.emplace_back(filtrations[i](0, 0), filtrations[i](0, 1));
    auto& boundary = out.matrix.emplace_back();
    boundary.reserve(boundaries[i].size());
    for (const auto boundary_index : boundaries[i]) {
      boundary.push_back(static_cast<int>(boundary_index));
    }
  }

  return out;
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

  auto complex = multipers::build_contiguous_f64_slicer_from_output(filtration_values, boundaries, dimensions);

  nb::object out = target.type()();
  auto& out_wrapper = nb::cast<CanonicalWrapper&>(out);
  multipers::build_slicer_from_complex(out_wrapper.truc, complex);
  out_wrapper.minpres_degree = degree;
  if (is_squeezed) {
    out_wrapper.filtration_grid = filtration_grid;
    out.attr("_clean_filtration_grid")();
  }
  return out;
}

}  // namespace mpaida
#endif

NB_MODULE(_aida_interface, m) {
  m.def("_is_available", []() {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
    return false;
#else
    return true;
#endif
  });

  m.def(
      "aida",
      [](nb::object s, bool sort, bool verbose, bool progress) {
#if MULTIPERS_DISABLE_AIDA_INTERFACE
        throw std::runtime_error("AIDA in-memory interface is disabled at compile time.");
#else
        if (sort) {
          s = s.attr("to_colexical")();
        }

        nb::object target = mpaida::ensure_supported_target(s);
        const auto& target_wrapper = nb::cast<const mpaida::CanonicalWrapper&>(target);
        auto prepared = mpaida::build_input_from_slicer(target_wrapper);

        aida::AIDA_functor functor;
        functor.config.show_info = verbose;
        functor.config.sort_output = false;
        functor.config.sort = sort;
        functor.config.progress = progress;
        auto input = aida::multipers_interface_input<int>(prepared.col_degree, prepared.row_degree, prepared.matrix);
        auto output = functor.multipers_interface(input);

        nb::list out;
        for (const auto& summand : output.summands) {
          nb::object slicer = mpaida::summand_to_slicer(
              target, summand, prepared.degree, prepared.is_squeezed, prepared.filtration_grid);
          if (target.ptr() != s.ptr()) {
            slicer = multipers::nanobind_helpers::astype_slicer_to_original_type(s, slicer);
          }
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

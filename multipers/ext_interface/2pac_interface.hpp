#pragma once

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace multipers {

template <typename index_type>
struct twopac_interface_input {
  std::vector<std::pair<double, double>> filtration_values;
  std::vector<std::vector<index_type>> boundaries;
  std::vector<int> dimensions;
};

template <typename index_type>
struct twopac_interface_output {
  std::vector<std::pair<double, double>> filtration_values;
  std::vector<std::vector<index_type>> boundaries;
  std::vector<int> dimensions;
};

inline bool twopac_interface_available();

template <typename index_type>
twopac_interface_output<index_type> twopac_minpres_interface(const twopac_interface_input<index_type>& input,
                                                             int degree,
                                                             bool full_resolution = true,
                                                             bool use_chunk = true,
                                                             bool use_clearing = false,
                                                             bool verbose_output = false);

}  // namespace multipers

#ifndef MULTIPERS_DISABLE_2PAC_INTERFACE
#if defined(_WIN32)
#define MULTIPERS_DISABLE_2PAC_INTERFACE 1
#else
#define MULTIPERS_DISABLE_2PAC_INTERFACE 0
#endif
#endif

#if !MULTIPERS_DISABLE_2PAC_INTERFACE
#include "contiguous_slicer_bridge.hpp"

namespace multipers {

template <typename contiguous_slicer_type>
contiguous_f64_complex twopac_minpres_contiguous_interface(contiguous_slicer_type& input,
                                                           int degree,
                                                           bool full_resolution = true,
                                                           bool use_chunk = true,
                                                           bool use_clearing = false,
                                                           bool verbose_output = false);

}  // namespace multipers
#endif

#ifndef MULTIPERS_HAS_2PAC_INTERFACE
#define MULTIPERS_HAS_2PAC_INTERFACE 0
#endif

#if !MULTIPERS_DISABLE_2PAC_INTERFACE && MULTIPERS_HAS_2PAC_INTERFACE

#include "chunk.hpp"
#include "factor.hpp"
#include "lw.hpp"
#include "matrices.hpp"
#include "minimize.hpp"
#include "time_measurement.hpp"

namespace multipers {

inline bool twopac_interface_available() { return true; }

namespace twopac_detail {

using twopac_resolution_t = std::pair<GradedMatrix, GradedMatrix>;

inline std::mutex& twopac_interface_mutex() {
  static std::mutex m;
  return m;
}

struct timing_mute_guard {
  explicit timing_mute_guard(bool verbose_output) : old_muted(Timing::muted) {
    Timing::contexts.clear();
    if (verbose_output) {
      Timing::unmute();
    } else {
      Timing::mute();
    }
  }

  ~timing_mute_guard() {
    Timing::contexts.clear();
    if (old_muted) {
      Timing::mute();
    } else {
      Timing::unmute();
    }
  }

  bool old_muted;
};

class VectorChainComplex : public Complex {
 public:
  explicit VectorChainComplex(std::vector<GradedMatrix> matrices) : matrices_(std::move(matrices)) {}

  GradedMatrix next_matrix() override {
    if (index_ >= matrices_.size()) {
      throw EOI();
    }
    return std::move(matrices_[index_++]);
  }

  bool is_cochain() override { return false; }

 private:
  std::vector<GradedMatrix> matrices_;
  std::size_t index_ = 0;
};

inline std::size_t first_index_of_dimension(const std::vector<int>& dimensions, int dim) {
  return static_cast<std::size_t>(std::lower_bound(dimensions.begin(), dimensions.end(), dim) - dimensions.begin());
}

inline grade pair_to_grade(const std::pair<double, double>& value) { return grade{value.first, value.second}; }

template <typename index_type>
inline std::vector<grade> grades_from_pairs(const std::vector<std::pair<double, double>>& filtration_values,
                                            std::size_t begin,
                                            std::size_t end) {
  std::vector<grade> out;
  out.reserve(end - begin);
  for (std::size_t i = begin; i < end; ++i) {
    out.push_back(pair_to_grade(filtration_values[i]));
  }
  return out;
}

template <typename index_type>
inline std::vector<std::vector<uint>> local_boundaries(const std::vector<std::vector<index_type>>& boundaries,
                                                       std::size_t col_begin,
                                                       std::size_t col_end,
                                                       std::size_t row_begin,
                                                       std::size_t row_end) {
  std::vector<std::vector<uint>> out;
  out.reserve(col_end - col_begin);
  for (std::size_t i = col_begin; i < col_end; ++i) {
    std::vector<uint> column;
    column.reserve(boundaries[i].size());
    for (const auto raw_idx : boundaries[i]) {
      if (raw_idx < 0) {
        throw std::invalid_argument("2pac interface received a negative boundary index.");
      }
      const auto idx = static_cast<std::size_t>(raw_idx);
      if (idx < row_begin || idx >= row_end) {
        throw std::invalid_argument("2pac interface received a boundary index outside the previous dimension block.");
      }
      column.push_back(static_cast<uint>(idx - row_begin));
    }
    std::sort(column.begin(), column.end());
    out.push_back(std::move(column));
  }
  return out;
}

template <typename index_type>
inline GradedMatrix build_boundary_matrix(const twopac_interface_input<index_type>& input, int dim) {
  const std::size_t row_begin = first_index_of_dimension(input.dimensions, dim);
  const std::size_t row_end = first_index_of_dimension(input.dimensions, dim + 1);
  const std::size_t col_begin = row_end;
  const std::size_t col_end = first_index_of_dimension(input.dimensions, dim + 2);

  auto row_grades = grades_from_pairs<index_type>(input.filtration_values, row_begin, row_end);
  auto column_grades = grades_from_pairs<index_type>(input.filtration_values, col_begin, col_end);
  auto entries = local_boundaries(input.boundaries, col_begin, col_end, row_begin, row_end);

  GradedMatrix out(std::move(row_grades),
                   std::move(column_grades),
                   SparseMatrix(static_cast<uint>(row_end - row_begin), std::move(entries)));
  out.sort_rows();
  out.sort_columns();
  return out;
}

template <typename index_type>
inline std::vector<GradedMatrix> build_boundary_matrices(const twopac_interface_input<index_type>& input, int degree) {
  if (degree < 0) {
    throw std::invalid_argument("2pac interface expects a non-negative homological degree.");
  }
  if (input.filtration_values.size() != input.boundaries.size() ||
      input.filtration_values.size() != input.dimensions.size()) {
    throw std::invalid_argument("Invalid 2pac input: sizes of filtrations, boundaries and dimensions differ.");
  }
  if (!std::is_sorted(input.dimensions.begin(), input.dimensions.end())) {
    throw std::invalid_argument("Dimensions are expected to be sorted in non-decreasing order.");
  }

  std::vector<GradedMatrix> out;
  out.reserve(static_cast<std::size_t>(degree + 1));
  for (int dim = 0; dim <= degree; ++dim) {
    out.push_back(build_boundary_matrix(input, dim));
  }
  return out;
}

inline twopac_resolution_t homology_step(GradedMatrix D,
                                         std::optional<GradedMatrix>& cycles,
                                         std::optional<GradedMatrix>* generators = nullptr) {
  if (!cycles) {
    auto E = GradedMatrix::ones({{grade::min, grade::min}}, D.row_grades);
    cycles = std::get<0>(kernel_mgs(E));
  }

  GradedMatrix next_cycles;
  GradedMatrix boundaries;
  GradedMatrix kernel_of_mgs;
  std::tie(next_cycles, boundaries, kernel_of_mgs) = kernel_mgs(D);

  auto factorized = factor_matrix(boundaries, *cycles);
  std::vector<size_t> non_local_rows;
  std::vector<size_t> non_local_columns;
  std::tie(non_local_rows, std::ignore, kernel_of_mgs) = minimize(kernel_of_mgs);
  factorized = std::move(factorized).get_columns(non_local_rows);
  std::tie(non_local_rows, non_local_columns, factorized) = minimize(factorized);
  kernel_of_mgs = std::move(kernel_of_mgs).get_rows(non_local_columns);

  if (generators != nullptr) {
    *generators = std::move(*cycles).get_columns(non_local_rows);
  }
  cycles = std::move(next_cycles);
  return {std::move(factorized), std::move(kernel_of_mgs)};
}

template <typename index_type>
inline twopac_interface_output<index_type> convert_resolution_to_output(twopac_resolution_t resolution,
                                                                        int degree,
                                                                        bool full_resolution) {
  auto& f0 = resolution.first;
  auto& f1 = resolution.second;
  f0.data.consolidate();
  f1.data.consolidate();

  twopac_interface_output<index_type> out;
  out.filtration_values.reserve(f0.rows() + f0.columns() + (full_resolution ? f1.columns() : 0));
  out.boundaries.reserve(f0.rows() + f0.columns() + (full_resolution ? f1.columns() : 0));
  out.dimensions.reserve(f0.rows() + f0.columns() + (full_resolution ? f1.columns() : 0));

  for (const auto& row_grade : f0.row_grades) {
    out.filtration_values.emplace_back(row_grade.x, row_grade.y);
    out.boundaries.emplace_back();
    out.dimensions.push_back(degree);
  }

  for (uint col = 0; col < f0.columns(); ++col) {
    out.filtration_values.emplace_back(f0.column_grades[col].x, f0.column_grades[col].y);
    std::vector<index_type> boundary;
    boundary.reserve(f0.data[col].data.size());
    for (const auto row_idx : f0.data[col].data) {
      boundary.push_back(static_cast<index_type>(row_idx));
    }
    out.boundaries.push_back(std::move(boundary));
    out.dimensions.push_back(degree + 1);
  }

  if (!full_resolution) {
    return out;
  }

  const auto row_shift = static_cast<index_type>(f0.rows());
  for (uint col = 0; col < f1.columns(); ++col) {
    out.filtration_values.emplace_back(f1.column_grades[col].x, f1.column_grades[col].y);
    std::vector<index_type> boundary;
    boundary.reserve(f1.data[col].data.size());
    for (const auto row_idx : f1.data[col].data) {
      boundary.push_back(static_cast<index_type>(row_idx) + row_shift);
    }
    out.boundaries.push_back(std::move(boundary));
    out.dimensions.push_back(degree + 2);
  }

  return out;
}

template <typename index_type, typename contiguous_slicer_type>
inline twopac_interface_input<index_type> convert_contiguous_slicer_to_input(contiguous_slicer_type& slicer) {
  twopac_interface_input<index_type> input;

  const auto dimensions = slicer.get_dimensions();
  const auto& boundaries = slicer.get_boundaries();
  auto& filtrations = slicer.get_filtration_values();
  const std::size_t num_generators = dimensions.size();

  if (boundaries.size() != num_generators || filtrations.size() != num_generators) {
    throw std::invalid_argument("Invalid slicer content: sizes of filtrations, boundaries and dimensions differ.");
  }

  input.dimensions = dimensions;
  input.boundaries.resize(num_generators);
  input.filtration_values.resize(num_generators);

  for (std::size_t i = 0; i < num_generators; ++i) {
    input.boundaries[i].reserve(boundaries[i].size());
    for (const auto idx : boundaries[i]) {
      input.boundaries[i].push_back(static_cast<index_type>(idx));
    }

    if (filtrations[i].num_parameters() != 2 || filtrations[i].num_generators() != 1) {
      throw std::invalid_argument(
          "2pac contiguous bridge expects 1-critical contiguous slicers with exactly 2 filtration parameters.");
    }
    input.filtration_values[i] = std::make_pair(filtrations[i](0, 0), filtrations[i](0, 1));
  }

  return input;
}

}  // namespace twopac_detail

template <typename index_type>
twopac_interface_output<index_type> twopac_minpres_interface(const twopac_interface_input<index_type>& input,
                                                             int degree,
                                                             bool full_resolution,
                                                             bool use_chunk,
                                                             bool use_clearing,
                                                             bool verbose_output) {
  std::lock_guard<std::mutex> lock(twopac_detail::twopac_interface_mutex());
  twopac_detail::timing_mute_guard timing_guard(verbose_output);
  (void)use_clearing;

  auto matrices = twopac_detail::build_boundary_matrices(input, degree);
  if (matrices.empty()) {
    return twopac_interface_output<index_type>();
  }

  std::shared_ptr<Complex> complex = std::make_shared<twopac_detail::VectorChainComplex>(std::move(matrices));
  if (use_chunk) {
    complex = std::make_shared<Chunk>(std::move(complex), static_cast<uint>(degree + 1));
  }

  std::optional<GradedMatrix> cycles;
  twopac_detail::twopac_resolution_t resolution;
  for (int dim = 0; dim <= degree; ++dim) {
    resolution = twopac_detail::homology_step(complex->next_matrix(), cycles);
  }

  return twopac_detail::convert_resolution_to_output<index_type>(std::move(resolution), degree, full_resolution);
}

#if !MULTIPERS_DISABLE_2PAC_INTERFACE
template <typename contiguous_slicer_type>
inline contiguous_f64_complex twopac_minpres_contiguous_interface(contiguous_slicer_type& input,
                                                                  int degree,
                                                                  bool full_resolution,
                                                                  bool use_chunk,
                                                                  bool use_clearing,
                                                                  bool verbose_output) {
  auto converted_input = twopac_detail::convert_contiguous_slicer_to_input<int>(input);
  auto out =
      twopac_minpres_interface<int>(converted_input, degree, full_resolution, use_chunk, use_clearing, verbose_output);
  return build_contiguous_f64_slicer_from_output<int>(out.filtration_values, out.boundaries, out.dimensions);
}
#endif

}  // namespace multipers

#else

namespace multipers {

inline bool twopac_interface_available() { return false; }

template <typename index_type>
twopac_interface_output<index_type> twopac_minpres_interface(const twopac_interface_input<index_type>&,
                                                             int,
                                                             bool,
                                                             bool,
                                                             bool,
                                                             bool) {
  throw std::runtime_error(
      "2pac in-memory interface is not available at compile time. Initialize ext/2pac (or set "
      "MULTIPERS_2PAC_SOURCE_DIR) and rebuild.");
}

#if !MULTIPERS_DISABLE_2PAC_INTERFACE
template <typename contiguous_slicer_type>
inline contiguous_f64_complex twopac_minpres_contiguous_interface(contiguous_slicer_type&,
                                                                  int,
                                                                  bool,
                                                                  bool,
                                                                  bool,
                                                                  bool) {
  throw std::runtime_error(
      "2pac in-memory interface is not available at compile time. Initialize ext/2pac (or set "
      "MULTIPERS_2PAC_SOURCE_DIR) and rebuild.");
}
#endif

}  // namespace multipers

#endif

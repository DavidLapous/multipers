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

template <typename index_type>
struct twopac_generator_matrix_output {
  std::vector<index_type> row_indices;
  std::vector<std::pair<double, double>> row_grades;
  std::vector<std::pair<double, double>> column_grades;
  std::vector<std::vector<index_type>> columns;
};

template <typename index_type>
struct twopac_minpres_with_generators_output {
  twopac_interface_output<index_type> minimal_presentation;
  twopac_generator_matrix_output<index_type> generator_matrix;
};

inline bool twopac_interface_available();

template <typename index_type>
twopac_interface_output<index_type> twopac_minpres_interface(const twopac_interface_input<index_type>& input,
                                                             int degree,
                                                             bool full_resolution = true,
                                                             bool use_chunk = true,
                                                             bool use_clearing = false,
                                                             bool verbose_output = false);

template <typename index_type>
twopac_minpres_with_generators_output<index_type> twopac_minpres_with_generators_interface(
    const twopac_interface_input<index_type>& input,
    int degree,
    bool full_resolution = true,
    bool use_chunk = false,
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

template <typename contiguous_slicer_type>
std::pair<contiguous_f64_complex, twopac_generator_matrix_output<int>>
twopac_minpres_with_generators_contiguous_interface(contiguous_slicer_type& input,
                                                    int degree,
                                                    bool full_resolution = true,
                                                    bool use_chunk = false,
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

struct twopac_raw_result {
  twopac_resolution_t resolution;
  std::optional<GradedMatrix> generators;
  std::vector<size_t> generator_row_indices;
};

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
inline twopac_generator_matrix_output<index_type> convert_generator_matrix_to_output(GradedMatrix matrix,
                                                                                     std::vector<size_t> row_indices) {
  matrix.data.consolidate();

  twopac_generator_matrix_output<index_type> out;
  out.row_indices.reserve(row_indices.size());
  for (const auto row_idx : row_indices) {
    out.row_indices.push_back(static_cast<index_type>(row_idx));
  }
  out.row_grades.reserve(matrix.row_grades.size());
  for (const auto& row_grade : matrix.row_grades) {
    out.row_grades.emplace_back(row_grade.x, row_grade.y);
  }

  out.column_grades.reserve(matrix.column_grades.size());
  out.columns.reserve(matrix.columns());
  for (uint col = 0; col < matrix.columns(); ++col) {
    out.column_grades.emplace_back(matrix.column_grades[col].x, matrix.column_grades[col].y);
    std::vector<index_type> support;
    support.reserve(matrix.data[col].data.size());
    for (const auto row_idx : matrix.data[col].data) {
      support.push_back(static_cast<index_type>(row_idx));
    }
    out.columns.push_back(std::move(support));
  }

  return out;
}

inline std::vector<size_t> identity_indices(std::size_t size) {
  std::vector<size_t> indices(size);
  for (std::size_t i = 0; i < size; ++i) {
    indices[i] = i;
  }
  return indices;
}

inline std::vector<GradedMatrix> chunk_chain_matrices_with_provenance(std::vector<GradedMatrix> matrices,
                                                                      int target_degree,
                                                                      std::vector<size_t>& target_row_indices) {
  std::vector<GradedMatrix> out;
  if (matrices.empty()) {
    target_row_indices.clear();
    return out;
  }

  out.reserve(matrices.size());
  std::vector<size_t> non_local_rows_d1;
  std::vector<size_t> current_basis = identity_indices(matrices[0].columns());
  GradedMatrix on_hold;
  std::tie(non_local_rows_d1, current_basis, on_hold) = minimize(matrices[0]);
  if (target_degree == 0) {
    target_row_indices = non_local_rows_d1;
  }

  if (matrices.size() == 1) {
    out.push_back(std::move(on_hold));
    return out;
  }

  for (std::size_t idx = 1; idx < matrices.size(); ++idx) {
    GradedMatrix next = std::move(matrices[idx]).get_rows(current_basis);
    std::vector<size_t> non_local_rows;
    std::vector<size_t> non_local_columns;
    std::tie(non_local_rows, non_local_columns, next) = minimize(next);
    on_hold = std::move(on_hold).get_columns(non_local_rows);
    if (static_cast<int>(idx) == target_degree) {
      target_row_indices = get_elements(current_basis, non_local_rows);
    }
    out.push_back(std::move(on_hold));
    on_hold = std::move(next);
    current_basis = std::move(non_local_columns);
  }

  out.push_back(std::move(on_hold));
  return out;
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

template <typename index_type>
inline twopac_raw_result compute_twopac_minpres_raw(const twopac_interface_input<index_type>& input,
                                                    int degree,
                                                    bool use_chunk,
                                                    bool verbose_output,
                                                    bool capture_generators) {
  std::lock_guard<std::mutex> lock(twopac_interface_mutex());
  timing_mute_guard timing_guard(verbose_output);

  auto matrices = build_boundary_matrices(input, degree);
  if (matrices.empty()) {
    return twopac_raw_result{};
  }

  if (capture_generators && use_chunk) {
    twopac_raw_result out;
    auto chunked = chunk_chain_matrices_with_provenance(std::move(matrices), degree, out.generator_row_indices);
    std::optional<GradedMatrix> cycles;
    std::optional<GradedMatrix> generators;
    twopac_resolution_t resolution;
    for (int dim = 0; dim <= degree; ++dim) {
      const bool want_generators = dim == degree;
      resolution = homology_step(std::move(chunked[dim]), cycles, want_generators ? &generators : nullptr);
    }
    out.resolution = std::move(resolution);
    out.generators = std::move(generators);
    return out;
  }

  std::shared_ptr<Complex> complex = std::make_shared<VectorChainComplex>(std::move(matrices));
  if (use_chunk) {
    complex = std::make_shared<Chunk>(std::move(complex), static_cast<uint>(degree + 1));
  }

  std::optional<GradedMatrix> cycles;
  twopac_resolution_t resolution;
  std::optional<GradedMatrix> generators;
  for (int dim = 0; dim <= degree; ++dim) {
    const bool want_generators = capture_generators && dim == degree;
    resolution = homology_step(complex->next_matrix(), cycles, want_generators ? &generators : nullptr);
  }

  twopac_raw_result out{std::move(resolution), std::move(generators), {}};
  if (capture_generators && out.generators.has_value()) {
    out.generator_row_indices = identity_indices(out.generators->row_grades.size());
  }
  return out;
}

}  // namespace twopac_detail

template <typename index_type>
twopac_interface_output<index_type> twopac_minpres_interface(const twopac_interface_input<index_type>& input,
                                                             int degree,
                                                             bool full_resolution,
                                                             bool use_chunk,
                                                             bool use_clearing,
                                                             bool verbose_output) {
  (void)use_clearing;

  auto raw = twopac_detail::compute_twopac_minpres_raw(input, degree, use_chunk, verbose_output, false);
  if (raw.resolution.first.rows() == 0 && raw.resolution.first.columns() == 0 && raw.resolution.second.rows() == 0 &&
      raw.resolution.second.columns() == 0) {
    return twopac_interface_output<index_type>();
  }

  return twopac_detail::convert_resolution_to_output<index_type>(std::move(raw.resolution), degree, full_resolution);
}

template <typename index_type>
twopac_minpres_with_generators_output<index_type> twopac_minpres_with_generators_interface(
    const twopac_interface_input<index_type>& input,
    int degree,
    bool full_resolution,
    bool use_chunk,
    bool use_clearing,
    bool verbose_output) {
  (void)use_clearing;

  auto raw = twopac_detail::compute_twopac_minpres_raw(input, degree, use_chunk, verbose_output, true);
  if (!raw.generators.has_value() && raw.resolution.first.rows() == 0 && raw.resolution.first.columns() == 0 &&
      raw.resolution.second.rows() == 0 && raw.resolution.second.columns() == 0) {
    return twopac_minpres_with_generators_output<index_type>();
  }

  twopac_minpres_with_generators_output<index_type> out;
  out.minimal_presentation =
      twopac_detail::convert_resolution_to_output<index_type>(std::move(raw.resolution), degree, full_resolution);
  if (raw.generators.has_value()) {
    out.generator_matrix = twopac_detail::convert_generator_matrix_to_output<index_type>(
        std::move(*raw.generators), std::move(raw.generator_row_indices));
  }
  return out;
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

template <typename contiguous_slicer_type>
inline std::pair<contiguous_f64_complex, twopac_generator_matrix_output<int>>
twopac_minpres_with_generators_contiguous_interface(contiguous_slicer_type& input,
                                                    int degree,
                                                    bool full_resolution,
                                                    bool use_chunk,
                                                    bool use_clearing,
                                                    bool verbose_output) {
  auto converted_input = twopac_detail::convert_contiguous_slicer_to_input<int>(input);
  auto out = twopac_minpres_with_generators_interface<int>(
      converted_input, degree, full_resolution, use_chunk, use_clearing, verbose_output);
  return {build_contiguous_f64_slicer_from_output<int>(out.minimal_presentation.filtration_values,
                                                       out.minimal_presentation.boundaries,
                                                       out.minimal_presentation.dimensions),
          std::move(out.generator_matrix)};
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

template <typename index_type>
twopac_minpres_with_generators_output<index_type>
twopac_minpres_with_generators_interface(const twopac_interface_input<index_type>&, int, bool, bool, bool, bool) {
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

template <typename contiguous_slicer_type>
inline std::pair<contiguous_f64_complex, twopac_generator_matrix_output<int>>
twopac_minpres_with_generators_contiguous_interface(contiguous_slicer_type&, int, bool, bool, bool, bool) {
  throw std::runtime_error(
      "2pac in-memory interface is not available at compile time. Initialize ext/2pac (or set "
      "MULTIPERS_2PAC_SOURCE_DIR) and rebuild.");
}
#endif

}  // namespace multipers

#endif

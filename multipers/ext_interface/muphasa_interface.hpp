#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace multipers {

template <typename index_type>
struct muphasa_interface_input {
  std::vector<std::vector<std::int64_t>> filtration_values;
  std::vector<std::vector<index_type>> boundaries;
  std::vector<int> dimensions;
  std::size_t num_parameters = 0;
};

template <typename index_type>
struct muphasa_interface_output {
  std::vector<std::vector<std::int64_t>> filtration_values;
  std::vector<std::vector<index_type>> boundaries;
  std::vector<int> dimensions;
};

inline bool muphasa_interface_available();

template <typename index_type>
muphasa_interface_output<index_type> muphasa_minpres_interface(const muphasa_interface_input<index_type>& input,
                                                               int degree,
                                                               bool full_resolution = false,
                                                               bool verbose_output = false);

}  // namespace multipers

#if !MULTIPERS_DISABLE_MUPHASA_INTERFACE
#include "contiguous_slicer_bridge.hpp"

namespace multipers {

template <typename contiguous_slicer_type>
contiguous_i32_complex muphasa_minpres_contiguous_interface(contiguous_slicer_type& input,
                                                            int degree,
                                                            bool full_resolution = false,
                                                            bool verbose_output = false);

}  // namespace multipers
#endif

#ifndef MULTIPERS_HAS_MUPHASA_INTERFACE
#if !MULTIPERS_DISABLE_MUPHASA_INTERFACE && __has_include("main.cpp") && __has_include("matrix.h")
#define MULTIPERS_HAS_MUPHASA_INTERFACE 1
#else
#define MULTIPERS_HAS_MUPHASA_INTERFACE 0
#endif
#endif

namespace multipers {

inline bool muphasa_interface_available() { return MULTIPERS_HAS_MUPHASA_INTERFACE; }

}  // namespace multipers

#if !MULTIPERS_DISABLE_MUPHASA_INTERFACE && MULTIPERS_HAS_MUPHASA_INTERFACE

// Muphasa exposes its library routines from a monolithic main.cpp. Build it in
// this extension only, with CLI main renamed away from the Python module entry.
#define main multipers_muphasa_cli_main
#include "main.cpp"
#undef main

namespace multipers {

namespace muphasa_detail {

struct raw_result {
  Matrix presentation;
  std::vector<grade_t> row_grades;
};

inline std::mutex& muphasa_interface_mutex() {
  static std::mutex m;
  return m;
}

inline std::size_t first_index_of_dimension(const std::vector<int>& dimensions, int dim) {
  return static_cast<std::size_t>(std::lower_bound(dimensions.begin(), dimensions.end(), dim) - dimensions.begin());
}

inline grade_t vector_to_grade(const std::vector<std::int64_t>& values) {
  grade_t grade;
  grade.reserve(values.size());
  for (const auto value : values) {
    grade.push_back(static_cast<index_t>(value));
  }
  return grade;
}

inline std::vector<std::int64_t> grade_to_vector(const grade_t& grade) {
  std::vector<std::int64_t> out;
  out.reserve(grade.size());
  for (const auto value : grade) {
    out.push_back(static_cast<std::int64_t>(value));
  }
  return out;
}

template <typename index_type>
Matrix build_boundary_matrix(const muphasa_interface_input<index_type>& input,
                             std::size_t col_begin,
                             std::size_t col_end,
                             std::size_t row_begin,
                             std::size_t row_end,
                             const std::string& matrix_name) {
  Matrix out;
  out.reserve(col_end - col_begin);
  for (std::size_t i = col_begin; i < col_end; ++i) {
    auto grade = vector_to_grade(input.filtration_values[i]);
    SignatureColumn column(grade, static_cast<index_t>(i - col_begin));
    for (const auto raw_idx : input.boundaries[i]) {
      if (raw_idx < 0) {
        throw std::invalid_argument(matrix_name + ": negative boundary index.");
      }
      const auto idx = static_cast<std::size_t>(raw_idx);
      if (idx < row_begin || idx >= row_end) {
        throw std::invalid_argument(matrix_name + ": boundary index outside expected dimension block.");
      }
      column.push(column_entry_t(1, static_cast<index_t>(idx - row_begin)));
    }
    column.syzygy.push(column_entry_t(1, static_cast<index_t>(i - col_begin)));
    out.push_back(std::move(column));
  }
  return out;
}

template <typename index_type>
Matrix build_identity_kernel(const muphasa_interface_input<index_type>& input,
                             std::size_t col_begin,
                             std::size_t col_end) {
  Matrix out;
  out.reserve(col_end - col_begin);
  for (std::size_t i = col_begin; i < col_end; ++i) {
    auto grade = vector_to_grade(input.filtration_values[i]);
    SignatureColumn column(grade, static_cast<index_t>(i - col_begin));
    column.push(column_entry_t(1, static_cast<index_t>(i - col_begin)));
    column.syzygy.push(column_entry_t(1, static_cast<index_t>(i - col_begin)));
    out.push_back(std::move(column));
  }
  return out;
}

inline std::vector<grade_t> ordered_row_grades(const hash_map<size_t, grade_t>& row_grade_map) {
  std::vector<size_t> rows;
  rows.reserve(row_grade_map.size());
  for (const auto& entry : row_grade_map) {
    rows.push_back(entry.first);
  }
  std::sort(rows.begin(), rows.end());

  std::vector<grade_t> out;
  out.reserve(rows.size());
  for (const auto row : rows) {
    out.push_back(row_grade_map.at(row));
  }
  return out;
}

inline std::vector<grade_t> row_grades_from_matrix(const Matrix& matrix) {
  std::vector<grade_t> out;
  out.reserve(matrix.size());
  for (const auto& column : matrix) {
    out.push_back(column.grade);
  }
  return out;
}

inline bool grade_colex_less(const std::vector<std::int64_t>& left, const std::vector<std::int64_t>& right) {
  const auto common_size = std::min(left.size(), right.size());
  for (std::size_t offset = 0; offset < common_size; ++offset) {
    const auto p = common_size - 1 - offset;
    if (left[p] != right[p]) {
      return left[p] < right[p];
    }
  }
  return left.size() < right.size();
}

template <typename index_type>
muphasa_interface_input<index_type> sort_input_by_dimension_colex(const muphasa_interface_input<index_type>& input) {
  const std::size_t n = input.dimensions.size();
  std::vector<std::size_t> permutation(n);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::stable_sort(permutation.begin(), permutation.end(), [&](std::size_t left, std::size_t right) {
    if (input.dimensions[left] != input.dimensions[right]) {
      return input.dimensions[left] < input.dimensions[right];
    }
    return grade_colex_less(input.filtration_values[left], input.filtration_values[right]);
  });

  std::vector<std::size_t> old_to_new(n);
  for (std::size_t new_idx = 0; new_idx < n; ++new_idx) {
    old_to_new[permutation[new_idx]] = new_idx;
  }

  muphasa_interface_input<index_type> out;
  out.num_parameters = input.num_parameters;
  out.dimensions.resize(n);
  out.filtration_values.resize(n);
  out.boundaries.resize(n);

  for (std::size_t new_idx = 0; new_idx < n; ++new_idx) {
    const std::size_t old_idx = permutation[new_idx];
    out.dimensions[new_idx] = input.dimensions[old_idx];
    out.filtration_values[new_idx] = input.filtration_values[old_idx];
    out.boundaries[new_idx].reserve(input.boundaries[old_idx].size());
    for (const auto raw_boundary_idx : input.boundaries[old_idx]) {
      if (raw_boundary_idx < 0 || static_cast<std::size_t>(raw_boundary_idx) >= n) {
        throw std::invalid_argument("Invalid Muphasa input: boundary index out of range.");
      }
      out.boundaries[new_idx].push_back(
          static_cast<index_type>(old_to_new[static_cast<std::size_t>(raw_boundary_idx)]));
    }
    std::sort(out.boundaries[new_idx].begin(), out.boundaries[new_idx].end());
  }
  return out;
}

template <int NumParameters, typename index_type>
raw_result compute_minpres(Matrix high_matrix, Matrix low_matrix) {
  if constexpr (NumParameters == 2) {
    Matrix kernel = computeKernel_2p(low_matrix);
    if (high_matrix.empty()) {
      return raw_result{Matrix{}, row_grades_from_matrix(kernel)};
    }
    auto presentation = compute_presentation_2p(high_matrix, kernel);
    return raw_result{std::move(presentation.first), ordered_row_grades(presentation.second)};
  } else if constexpr (NumParameters == 3) {
    auto presentation = computeMinimalPresentation_3p(high_matrix, low_matrix, false);
    return raw_result{std::move(presentation.first), std::move(presentation.second)};
  } else {
    auto presentation = computeMinimalPresentation(high_matrix, low_matrix, false);
    return raw_result{std::move(presentation.first), std::move(presentation.second)};
  }
}

template <typename index_type>
raw_result compute_minpres_for_parameter_count(const muphasa_interface_input<index_type>& input, int degree) {
  const std::size_t start_dm1 = first_index_of_dimension(input.dimensions, degree - 1);
  const std::size_t start_d = first_index_of_dimension(input.dimensions, degree);
  const std::size_t start_dp1 = first_index_of_dimension(input.dimensions, degree + 1);
  const std::size_t start_dp2 = first_index_of_dimension(input.dimensions, degree + 2);

  Matrix high_matrix = build_boundary_matrix(input, start_dp1, start_dp2, start_d, start_dp1, "Muphasa upper matrix");
  Matrix low_matrix = build_boundary_matrix(input, start_d, start_dp1, start_dm1, start_d, "Muphasa lower matrix");

  if (start_dm1 == start_d) {
    Matrix kernel = build_identity_kernel(input, start_d, start_dp1);
    if (high_matrix.empty()) {
      return raw_result{Matrix{}, row_grades_from_matrix(kernel)};
    }
    if (input.num_parameters == 2) {
      auto presentation = compute_presentation_2p(high_matrix, kernel);
      return raw_result{std::move(presentation.first), ordered_row_grades(presentation.second)};
    }
    if (input.num_parameters == 3) {
      return compute_minpres<3, index_type>(std::move(high_matrix), std::move(kernel));
    }
    return compute_minpres<-1, index_type>(std::move(high_matrix), std::move(kernel));
  }

  if (input.num_parameters == 2) {
    return compute_minpres<2, index_type>(std::move(high_matrix), std::move(low_matrix));
  }
  if (input.num_parameters == 3) {
    return compute_minpres<3, index_type>(std::move(high_matrix), std::move(low_matrix));
  }
  return compute_minpres<-1, index_type>(std::move(high_matrix), std::move(low_matrix));
}

template <typename index_type>
muphasa_interface_output<index_type> convert_raw_to_output(raw_result raw, int degree) {
  muphasa_interface_output<index_type> out;
  out.filtration_values.reserve(raw.row_grades.size() + raw.presentation.size());
  out.boundaries.reserve(raw.row_grades.size() + raw.presentation.size());
  out.dimensions.reserve(raw.row_grades.size() + raw.presentation.size());

  for (const auto& row_grade : raw.row_grades) {
    out.filtration_values.push_back(grade_to_vector(row_grade));
    out.boundaries.emplace_back();
    out.dimensions.push_back(degree);
  }

  const auto row_count = raw.row_grades.size();
  for (const auto& column : raw.presentation) {
    out.filtration_values.push_back(grade_to_vector(column.grade));
    std::vector<index_type> boundary;
    boundary.reserve(column.size());
    for (const auto row_idx : column) {
      if (row_idx < 0 || static_cast<std::size_t>(row_idx) >= row_count) {
        throw std::runtime_error("Muphasa output contains a relation row outside the minimal-generator block.");
      }
      boundary.push_back(static_cast<index_type>(row_idx));
    }
    out.boundaries.push_back(std::move(boundary));
    out.dimensions.push_back(degree + 1);
  }

  return out;
}

template <typename index_type, typename contiguous_slicer_type>
muphasa_interface_input<index_type> convert_contiguous_slicer_to_input(contiguous_slicer_type& slicer) {
  muphasa_interface_input<index_type> input;

  const auto dimensions = slicer.get_dimensions();
  const auto& boundaries = slicer.get_boundaries();
  auto& filtrations = slicer.get_filtration_values();
  const std::size_t num_generators = dimensions.size();

  if (boundaries.size() != num_generators || filtrations.size() != num_generators) {
    throw std::invalid_argument("Invalid slicer content: sizes of filtrations, boundaries and dimensions differ.");
  }
  if (!std::is_sorted(dimensions.begin(), dimensions.end())) {
    throw std::invalid_argument("Dimensions are expected to be sorted in non-decreasing order.");
  }

  input.dimensions = dimensions;
  input.boundaries.resize(num_generators);
  input.filtration_values.resize(num_generators);

  for (std::size_t i = 0; i < num_generators; ++i) {
    if (filtrations[i].num_generators() != 1) {
      throw std::invalid_argument("Muphasa contiguous bridge expects 1-critical contiguous slicers.");
    }
    if (i == 0) {
      input.num_parameters = filtrations[i].num_parameters();
    } else if (input.num_parameters != filtrations[i].num_parameters()) {
      throw std::invalid_argument("Muphasa contiguous bridge got inconsistent filtration arities.");
    }

    input.filtration_values[i].reserve(input.num_parameters);
    for (std::size_t p = 0; p < input.num_parameters; ++p) {
      input.filtration_values[i].push_back(filtrations[i](0, p));
    }

    input.boundaries[i].reserve(boundaries[i].size());
    for (const auto idx : boundaries[i]) {
      if (idx >= num_generators) {
        throw std::invalid_argument("Invalid slicer content: boundary index out of range.");
      }
      for (std::size_t p = 0; p < input.num_parameters; ++p) {
        if (filtrations[i](0, p) < filtrations[idx](0, p)) {
          throw std::invalid_argument(
              "Muphasa backend expects filtration values to be non-decreasing along boundaries.");
        }
      }
      input.boundaries[i].push_back(static_cast<index_type>(idx));
    }
  }

  return input;
}

}  // namespace muphasa_detail

template <typename index_type>
muphasa_interface_output<index_type> muphasa_minpres_interface(const muphasa_interface_input<index_type>& input,
                                                               int degree,
                                                               bool full_resolution,
                                                               bool verbose_output) {
  (void)verbose_output;
  if (full_resolution) {
    throw std::invalid_argument("Muphasa backend currently supports only full_resolution=False.");
  }
  if (degree < 0) {
    throw std::invalid_argument("Muphasa interface expects a non-negative homological degree.");
  }
  if (input.num_parameters == 0 && !input.filtration_values.empty()) {
    throw std::invalid_argument("Muphasa interface received empty filtration grades.");
  }
  if (input.num_parameters < 2) {
    throw std::invalid_argument("Muphasa backend expects at least 2-parameter slicers.");
  }
  if (input.filtration_values.size() != input.boundaries.size() ||
      input.filtration_values.size() != input.dimensions.size()) {
    throw std::invalid_argument("Invalid Muphasa input: sizes of filtrations, boundaries and dimensions differ.");
  }

  auto sorted_input = muphasa_detail::sort_input_by_dimension_colex(input);

  std::lock_guard<std::mutex> lock(muphasa_detail::muphasa_interface_mutex());
  auto raw = muphasa_detail::compute_minpres_for_parameter_count(sorted_input, degree);
  return muphasa_detail::convert_raw_to_output<index_type>(std::move(raw), degree);
}

template <typename contiguous_slicer_type>
inline contiguous_i32_complex muphasa_minpres_contiguous_interface(contiguous_slicer_type& input,
                                                                   int degree,
                                                                   bool full_resolution,
                                                                   bool verbose_output) {
  auto converted_input = muphasa_detail::convert_contiguous_slicer_to_input<int>(input);
  auto out = muphasa_minpres_interface<int>(converted_input, degree, full_resolution, verbose_output);
  return build_contiguous_i32_slicer_from_output<int>(out.filtration_values, out.boundaries, out.dimensions);
}

}  // namespace multipers

#else

namespace multipers {

[[noreturn]] inline void throw_muphasa_interface_unavailable() {
  throw std::runtime_error(
      "Muphasa interface is not available at compile time. Initialize ext/muphasa (or set "
      "MULTIPERS_MUPHASA_SOURCE_DIR) and rebuild.");
}

template <typename index_type>
muphasa_interface_output<index_type> muphasa_minpres_interface(const muphasa_interface_input<index_type>&,
                                                               int,
                                                               bool,
                                                               bool) {
  throw_muphasa_interface_unavailable();
}

#if !MULTIPERS_DISABLE_MUPHASA_INTERFACE
template <typename contiguous_slicer_type>
inline contiguous_i32_complex muphasa_minpres_contiguous_interface(contiguous_slicer_type&, int, bool, bool) {
  throw_muphasa_interface_unavailable();
}
#endif

}  // namespace multipers

#endif

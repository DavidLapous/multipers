#pragma once

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace multipers {

template <typename index_type>
struct mpfree_interface_input {
  std::vector<std::pair<double, double> > filtration_values;
  std::vector<std::vector<index_type> > boundaries;
  std::vector<int> dimensions;
};

template <typename index_type>
struct mpfree_interface_output {
  std::vector<std::pair<double, double> > filtration_values;
  std::vector<std::vector<index_type> > boundaries;
  std::vector<int> dimensions;
};

inline bool mpfree_interface_available();

template <typename index_type>
mpfree_interface_output<index_type> mpfree_minpres_interface(const mpfree_interface_input<index_type>& input,
                                                             int degree,
                                                             bool full_resolution = true,
                                                             bool use_chunk = true,
                                                             bool use_clearing = false,
                                                             bool verbose_output = false);

}  // namespace multipers

#ifndef MULTIPERS_DISABLE_MPFREE_INTERFACE
#define MULTIPERS_DISABLE_MPFREE_INTERFACE 0
#endif

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE && __has_include(<mpfree/mpfree.h>) && __has_include(<mpp_utils/Graded_matrix.h>)
#define MULTIPERS_HAS_MPFREE_INTERFACE 1
#include <mpfree/mpfree.h>
#include <mpp_utils/Graded_matrix.h>
#include <mpp_utils/Pre_column_struct.h>
#include <mpp_utils/create_graded_matrices_from_pre_column_struct.h>
#else
#define MULTIPERS_HAS_MPFREE_INTERFACE 0
#endif

namespace multipers {

inline bool mpfree_interface_available() { return MULTIPERS_HAS_MPFREE_INTERFACE; }

#if MULTIPERS_HAS_MPFREE_INTERFACE

namespace detail {

inline std::mutex& mpfree_interface_mutex() {
  static std::mutex m;
  return m;
}

using Graded_matrix = mpp_utils::Graded_matrix<phat::vector_vector>;
using Grade = typename Graded_matrix::Grade;

inline std::size_t first_index_of_dimension(const std::vector<int>& dimensions, int dim) {
  return static_cast<std::size_t>(std::lower_bound(dimensions.begin(), dimensions.end(), dim) - dimensions.begin());
}

inline Grade pair_to_grade(const std::pair<double, double>& grade) {
  Grade out;
  out.at = {grade.first, grade.second};
  return out;
}

template <typename index_type>
inline std::vector<phat::index> convert_boundary(const std::vector<index_type>& boundary,
                                                 std::size_t row_begin,
                                                 std::size_t row_end,
                                                 const std::string& matrix_name,
                                                 std::size_t column) {
  std::vector<phat::index> out;
  out.reserve(boundary.size());
  for (const auto idx : boundary) {
    if (idx < 0) {
      throw std::invalid_argument(matrix_name + ": negative boundary index at column " + std::to_string(column));
    }
    const auto id = static_cast<std::size_t>(idx);
    if (id < row_begin || id >= row_end) {
      throw std::invalid_argument(matrix_name + ": boundary index out of expected dimension block at column " +
                                  std::to_string(column));
    }
    out.push_back(static_cast<phat::index>(id - row_begin));
  }
  std::sort(out.begin(), out.end());
  return out;
}

template <typename index_type>
inline void assign_column_grades(Graded_matrix& matrix,
                                 const std::vector<std::pair<double, double> >& filtration_values,
                                 std::size_t col_begin,
                                 std::size_t col_end) {
  matrix.grades.resize(col_end - col_begin);
  for (std::size_t i = col_begin; i < col_end; ++i) {
    matrix.grades[i - col_begin] = pair_to_grade(filtration_values[i]);
  }
}

template <typename index_type>
inline void assign_row_grades(Graded_matrix& matrix,
                              const std::vector<std::pair<double, double> >& filtration_values,
                              std::size_t row_begin,
                              std::size_t row_end) {
  matrix.row_grades.clear();
  matrix.row_grades.reserve(row_end - row_begin);
  for (std::size_t i = row_begin; i < row_end; ++i) {
    matrix.row_grades.push_back(pair_to_grade(filtration_values[i]));
  }
}

template <typename index_type>
inline void fill_boundaries(Graded_matrix& matrix,
                            const std::vector<std::vector<index_type> >& boundaries,
                            std::size_t col_begin,
                            std::size_t col_end,
                            std::size_t row_begin,
                            std::size_t row_end,
                            const std::string& matrix_name) {
  for (std::size_t i = col_begin; i < col_end; ++i) {
    matrix.set_col(i - col_begin, convert_boundary(boundaries[i], row_begin, row_end, matrix_name, i - col_begin));
  }
}

template <typename index_type>
inline void append_rows_as_generators(const Graded_matrix& matrix,
                                      int degree,
                                      mpfree_interface_output<index_type>& out) {
  for (const auto& row_grade : matrix.row_grades) {
    out.filtration_values.emplace_back(row_grade.at[0], row_grade.at[1]);
    out.boundaries.emplace_back();
    out.dimensions.push_back(degree);
  }
}

template <typename index_type>
inline void append_columns_as_generators(const Graded_matrix& matrix,
                                         int degree,
                                         index_type row_shift,
                                         mpfree_interface_output<index_type>& out) {
  for (phat::index col_idx = 0; col_idx < matrix.get_num_cols(); ++col_idx) {
    out.filtration_values.emplace_back(matrix.grades[col_idx].at[0], matrix.grades[col_idx].at[1]);
    std::vector<phat::index> col;
    matrix.get_col(col_idx, col);
    std::vector<index_type> shifted_col;
    shifted_col.reserve(col.size());
    for (const auto row_idx : col) {
      shifted_col.push_back(static_cast<index_type>(row_idx) + row_shift);
    }
    out.boundaries.push_back(std::move(shifted_col));
    out.dimensions.push_back(degree);
  }
}

template <typename index_type>
inline mpfree_interface_output<index_type> convert_minpres_to_output(Graded_matrix& min_rep,
                                                                     int degree,
                                                                     bool full_resolution) {
  mpfree_interface_output<index_type> out;

  if (full_resolution) {
    mpp_utils::to_colex_order(min_rep);
  }

  append_rows_as_generators(min_rep, degree, out);
  append_columns_as_generators(min_rep, degree + 1, 0, out);

  if (!full_resolution) {
    return out;
  }

  Graded_matrix syzygy_matrix_base;
  Graded_matrix dummy_base;

  using Graded_matrix_extended = typename mpfree::Extend_matrix<Graded_matrix>::Type;
  Graded_matrix_extended min_rep_ext(&min_rep);
  Graded_matrix_extended dummy_ext(&dummy_base);
  Graded_matrix_extended syzygy_matrix_ext(&syzygy_matrix_base);

  min_rep_ext.grid_scheduler = mpfree::Grid_scheduler(min_rep);
  min_rep_ext.pq_row.resize(min_rep.num_grades_y);
  min_rep_ext.assign_pivots();
  min_rep_ext.assign_slave_matrix();
  mpfree::ker_basis(min_rep_ext, syzygy_matrix_ext, dummy_ext, false);

  append_columns_as_generators(syzygy_matrix_base, degree + 2, static_cast<index_type>(min_rep.row_grades.size()), out);

  return out;
}

}  // namespace detail

template <typename index_type>
mpfree_interface_output<index_type> mpfree_minpres_interface(const mpfree_interface_input<index_type>& input,
                                                             int degree,
                                                             bool full_resolution,
                                                             bool use_chunk,
                                                             bool use_clearing,
                                                             bool verbose_output) {
  std::lock_guard<std::mutex> lock(detail::mpfree_interface_mutex());

  if (degree < 0) {
    throw std::invalid_argument("mpfree interface expects a non-negative homological degree.");
  }

  if (input.filtration_values.size() != input.boundaries.size() ||
      input.filtration_values.size() != input.dimensions.size()) {
    throw std::invalid_argument("Invalid multipers input: sizes of filtrations, boundaries and dimensions differ.");
  }

  // for (const auto& grade : input.filtration_values) {
  //   if (!std::isfinite(grade.first) || !std::isfinite(grade.second)) {
  //     throw std::invalid_argument("mpfree interface expects finite bifiltration values.");
  //   }
  // }

  if (!std::is_sorted(input.dimensions.begin(), input.dimensions.end())) {
    throw std::invalid_argument("Dimensions are expected to be sorted in non-decreasing order.");
  }

  const std::size_t start_dm1 = detail::first_index_of_dimension(input.dimensions, degree - 1);
  const std::size_t start_d = detail::first_index_of_dimension(input.dimensions, degree);
  const std::size_t start_dp1 = detail::first_index_of_dimension(input.dimensions, degree + 1);
  const std::size_t start_dp2 = detail::first_index_of_dimension(input.dimensions, degree + 2);

  const auto n_dm1 = start_d - start_dm1;
  const auto n_d = start_dp1 - start_d;
  const auto n_dp1 = start_dp2 - start_dp1;

  using Pre_column = mpp_utils::Pre_column_struct<detail::Grade>;
  std::vector<std::vector<Pre_column> > pre_matrices(2);
  pre_matrices[0].reserve(n_dp1);
  pre_matrices[1].reserve(n_d);

  for (std::size_t i = start_dp1; i < start_dp2; ++i) {
    auto boundary = detail::convert_boundary(input.boundaries[i], start_d, start_dp1, "Upper matrix", i - start_dp1);
    auto grade = detail::pair_to_grade(input.filtration_values[i]);
    pre_matrices[0].emplace_back(static_cast<mpp_utils::index>(i - start_dp1), grade, boundary);
  }

  for (std::size_t i = start_d; i < start_dp1; ++i) {
    auto boundary = detail::convert_boundary(input.boundaries[i], start_dm1, start_d, "Lower matrix", i - start_d);
    auto grade = detail::pair_to_grade(input.filtration_values[i]);
    pre_matrices[1].emplace_back(static_cast<mpp_utils::index>(i - start_d), grade, boundary);
  }

  std::vector<detail::Graded_matrix> matrices;
  mpp_utils::create_graded_matrices_from_pre_column_struct(
      pre_matrices, matrices, static_cast<int>(n_dm1), false, true);

  if (matrices.size() != 2) {
    throw std::runtime_error("Internal mpfree conversion failure: expected two graded matrices.");
  }

  detail::Graded_matrix gm_upper = std::move(matrices[0]);
  detail::Graded_matrix gm_lower = std::move(matrices[1]);

  detail::Graded_matrix min_rep;
  const bool old_verbose = mpfree::verbose;
  mpfree::verbose = verbose_output;
  mpfree::compute_minimal_presentation(gm_upper, gm_lower, min_rep, use_chunk, use_clearing);
  mpfree::verbose = old_verbose;

  return detail::convert_minpres_to_output<index_type>(min_rep, degree, full_resolution);
}

#else

template <typename index_type>
mpfree_interface_output<index_type> mpfree_minpres_interface(const mpfree_interface_input<index_type>&,
                                                             int,
                                                             bool,
                                                             bool,
                                                             bool,
                                                             bool) {
  throw std::runtime_error(
      "mpfree in-memory interface is not available at compile time. Install/checkout mpfree headers and rebuild.");
}

#endif

}  // namespace multipers

#pragma once

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

template <typename index_type>
struct mpfree_generator_matrix_output {
  std::vector<index_type> row_indices;
  std::vector<std::pair<double, double> > row_grades;
  std::vector<std::pair<double, double> > column_grades;
  std::vector<std::vector<index_type> > columns;
};

template <typename index_type>
struct mpfree_minpres_with_generators_output {
  mpfree_interface_output<index_type> minimal_presentation;
  mpfree_generator_matrix_output<index_type> generator_matrix;
};

inline bool mpfree_interface_available();

template <typename index_type>
mpfree_interface_output<index_type> mpfree_minpres_interface(const mpfree_interface_input<index_type>& input,
                                                             int degree,
                                                             bool full_resolution = true,
                                                             bool use_chunk = true,
                                                             bool use_clearing = false,
                                                             bool verbose_output = false);

template <typename index_type>
mpfree_minpres_with_generators_output<index_type> mpfree_minpres_with_generators_interface(
    const mpfree_interface_input<index_type>& input,
    int degree,
    bool full_resolution = true,
    bool use_chunk = false,
    bool use_clearing = false,
    bool verbose_output = false);

}  // namespace multipers

#ifndef MULTIPERS_DISABLE_MPFREE_INTERFACE
#define MULTIPERS_DISABLE_MPFREE_INTERFACE 0
#endif

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
#include "backend_log_policy.hpp"
#include "contiguous_slicer_bridge.hpp"

namespace multipers {

template <typename contiguous_slicer_type>
contiguous_f64_complex mpfree_minpres_contiguous_interface(contiguous_slicer_type& input,
                                                           int degree,
                                                           bool full_resolution = true,
                                                           bool use_chunk = true,
                                                           bool use_clearing = false,
                                                           bool verbose_output = false);

template <typename contiguous_slicer_type>
std::pair<contiguous_f64_complex, mpfree_generator_matrix_output<int> >
mpfree_minpres_with_generators_contiguous_interface(contiguous_slicer_type& input,
                                                    int degree,
                                                    bool full_resolution = true,
                                                    bool use_chunk = false,
                                                    bool use_clearing = false,
                                                    bool verbose_output = false);

}  // namespace multipers
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
  // mpfree only needs serialization when timer globals are compiled in.
  static std::mutex m;
  return m;
}

using Graded_matrix = mpp_utils::Graded_matrix<phat::vector_vector>;
using Grade = typename Graded_matrix::Grade;

struct mpfree_raw_result {
  Graded_matrix min_rep;
  Graded_matrix kernel_basis;
  std::vector<Grade> chain_row_grades;
  std::vector<phat::index> chain_row_indices;
  std::vector<phat::index> surviving_rows;
  bool has_generators = false;
};

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

inline std::vector<std::size_t> row_order_for_output(const Graded_matrix& matrix, bool full_resolution) {
  std::vector<std::size_t> order(matrix.row_grades.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  if (!full_resolution) {
    return order;
  }
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const auto& ga = matrix.row_grades[a].at;
    const auto& gb = matrix.row_grades[b].at;
    if (ga[1] != gb[1]) {
      return ga[1] < gb[1];
    }
    if (ga[0] != gb[0]) {
      return ga[0] < gb[0];
    }
    return a < b;
  });
  return order;
}

template <typename GradedMatrix1, typename GradedMatrix2>
concept has_chunk_preprocessing_with_surviving_rows =
    requires(GradedMatrix1& M1, GradedMatrix2& M2, std::vector<phat::index>* surviving_rows) {
      mpfree::chunk_preprocessing(M1, M2, surviving_rows);
    };

template <typename GradedMatrix>
inline void chunk_preprocessing_with_surviving_rows_fallback(GradedMatrix& M1,
                                                             GradedMatrix& M2,
                                                             std::vector<phat::index>& surviving_rows) {
  using index = phat::index;

  if (mpfree::verbose) std::cout << "Num entries at start chunk: " << M1.get_num_entries() << std::endl;

  std::vector<int> local_pivots;
  local_pivots.reserve(M1.num_rows);
  for (index i = 0; i < M1.num_rows; i++) {
    local_pivots.push_back(-1);
  }
  if (mpfree::verbose) std::cout << "Local reduction" << std::endl;
  std::vector<index> global_indices;
  for (index i = 0; i < M1.get_num_cols(); i++) {
    while (M1.is_local(i)) {
      index p = M1.get_max_index(i);
      index j = local_pivots[p];
      if (j != -1) {
        M1.add_to(j, i);
      } else {
        local_pivots[p] = i;
        break;
      }
    }
    if (!M1.is_local(i)) {
      global_indices.push_back(i);
    }
    if (M1.is_empty(i)) {
      M1.clear(i);
    }
  }

  if (mpfree::verbose) std::cout << "Num entries after local reduce: " << M1.get_num_entries() << std::endl;
  if (mpfree::verbose) std::cout << "Sparsification" << std::endl;

  std::vector<index> col;
  M1.sync();
#pragma omp parallel for schedule(guided, 1), private(col)
  for (index r = 0; r < global_indices.size(); r++) {
    col.clear();
    index i = global_indices[r];
    while (!M1.is_empty(i)) {
      index p = M1.get_max_index(i);
      index j = local_pivots[p];
      if (j != -1) {
        M1.add_to(j, i);
      } else {
        col.push_back(p);
        M1.remove_max(i);
      }
    }
    std::reverse(col.begin(), col.end());
    M1.set_col(i, col);
  }
  M1.sync();

  if (mpfree::verbose) std::cout << "Build up smaller matrices" << std::endl;
  std::vector<int> new_row_index;
  new_row_index.resize(M1.num_rows);
  index row_count = 0;
  surviving_rows.clear();
  surviving_rows.reserve(M1.num_rows);
  for (index i = 0; i < M1.num_rows; i++) {
    if (local_pivots[i] == -1) {
      new_row_index[i] = row_count++;
      surviving_rows.push_back(i);
    } else {
      new_row_index[i] = -1;
    }
  }

  std::vector<int> new_col_index;
  new_col_index.resize(M1.get_num_cols());
  index col_count = 0;
  for (index i = 0; i < M1.get_num_cols(); i++) {
    if (M1.is_empty(i) || M1.is_local(i)) {
      new_col_index[i] = -1;
    } else {
      new_col_index[i] = col_count++;
    }
  }

  for (index i = 0; i < M1.get_num_cols(); i++) {
    if (new_col_index[i] != -1) {
      index j = new_col_index[i];
      assert(j <= i);
      M1.grades[j] = M1.grades[i];
      std::vector<index> current_col;
      M1.get_col(i, current_col);
      for (index k = 0; k < current_col.size(); k++) {
        current_col[k] = new_row_index[current_col[k]];
      }
      M1.set_col(j, current_col);
    }
  }

  M1.set_dimensions(row_count, col_count);

  for (index i = 0; i < M1.num_rows; i++) {
    if (local_pivots[i] == -1) {
      int j = new_row_index[i];
      assert(j <= i);
      M1.row_grades[j] = M1.row_grades[i];
      M2.grades[j] = M1.row_grades[i];
      std::vector<index> current_col;
      M2.get_col(i, current_col);
      M2.set_col(j, current_col);
    }
  }
  M2.set_dimensions(M2.num_rows, row_count);
  M2.grades.resize(row_count);
  M1.row_grades.resize(row_count);
  M1.num_rows = row_count;

  M1.assign_slave_matrix();
  M1.assign_pivots();
  M2.assign_slave_matrix();
  M2.assign_pivots();

  M1.pq_row.resize(M1.num_grades_y);
  M2.pq_row.resize(M2.num_grades_y);

  if (mpfree::verbose)
    std::cout << "After chunk reduction, matrix has " << M1.get_num_cols() << " columns and " << M1.num_rows << " rows"
              << std::endl;
  if (mpfree::verbose) std::cout << "N' is " << M1.get_num_cols() + M1.num_rows << std::endl;
  if (mpfree::verbose) std::cout << "Num entries after chunk: " << M1.get_num_entries() << std::endl;
}

template <typename GradedMatrix1, typename GradedMatrix2>
inline void chunk_preprocessing_with_surviving_rows(GradedMatrix1& M1,
                                                    GradedMatrix2& M2,
                                                    std::vector<phat::index>& surviving_rows) {
  if constexpr (has_chunk_preprocessing_with_surviving_rows<GradedMatrix1, GradedMatrix2>) {
    mpfree::chunk_preprocessing(M1, M2, &surviving_rows);
  } else {
    chunk_preprocessing_with_surviving_rows_fallback(M1, M2, surviving_rows);
  }
}

template <typename GradedMatrixInput, typename GradedMatrixOutput>
concept has_minimize_with_kept_rows =
    requires(GradedMatrixInput& M, GradedMatrixOutput& result, std::vector<phat::index>* kept_rows) {
      mpfree::minimize(M, result, kept_rows);
    };

template <typename GradedMatrixInput, typename GradedMatrixOutput>
inline void minimize_with_kept_rows_fallback(GradedMatrixInput& M,
                                             GradedMatrixOutput& result,
                                             std::vector<phat::index>& kept_rows) {
  using Grade = typename GradedMatrixInput::Grade;
  using index = phat::index;
  using Column = std::vector<index>;

  GradedMatrixInput& VVM = M;
  VVM.assign_pivots();

  std::set<index> rows_to_delete;
  std::vector<index> cols_to_keep;
  std::vector<Grade> col_grades;
  for (index i = 0; i < VVM.get_num_cols(); i++) {
    while (!VVM.is_empty(i)) {
      index col_grade_x = VVM.grades[i].index_at[0];
      index col_grade_y = VVM.grades[i].index_at[1];
      index p = VVM.get_max_index(i);
      index row_grade_x = VVM.row_grades[p].index_at[0];
      index row_grade_y = VVM.row_grades[p].index_at[1];

      if (col_grade_x != row_grade_x || col_grade_y != row_grade_y) {
        cols_to_keep.push_back(i);
        col_grades.push_back(VVM.grades[i]);
        break;
      }
      if (VVM.pivots[p] == -1) {
        rows_to_delete.insert(p);
        VVM.pivots[p] = i;
        break;
      }
      VVM.add_to(VVM.pivots[p], i);
    }
    assert(!VVM.is_empty(i));
  }

  std::vector<Column> new_cols(cols_to_keep.size());
  VVM.sync();
#pragma omp parallel for schedule(guided, 1)
  for (index k = 0; k < cols_to_keep.size(); k++) {
    Column& col = new_cols[k];
    index i = cols_to_keep[k];
    while (!VVM.is_empty(i)) {
      index p = VVM.get_max_index(i);
      if (VVM.pivots[p] == -1) {
        col.push_back(p);
        VVM.remove_max(i);
      } else {
        VVM.add_to(VVM.pivots[p], i);
      }
    }
    std::reverse(col.begin(), col.end());
  }
  VVM.sync();

  index nr = VVM.num_rows;
  index count = 0;
  std::unordered_map<index, index> index_map;
  std::vector<Grade> res_row_grades;
  for (index i = 0; i < nr; i++) {
    if (rows_to_delete.count(i) == 0) {
      index_map[i] = count++;
      res_row_grades.push_back(VVM.row_grades[i]);
    }
  }

  kept_rows.clear();
  kept_rows.reserve(index_map.size());
  for (index i = 0; i < nr; i++) {
    if (rows_to_delete.count(i) == 0) {
      kept_rows.push_back(i);
    }
  }

  for (index i = 0; i < cols_to_keep.size(); i++) {
    Column& col = new_cols[i];
    for (int j = 0; j < col.size(); j++) {
      assert(index_map.count(col[j]));
      col[j] = index_map[col[j]];
    }
  }

  result.number_of_parameters = M.number_of_parameters;
  result.set_dimensions(index_map.size(), cols_to_keep.size());
  result.num_rows = index_map.size();
  result.grades = col_grades;
  std::copy(res_row_grades.begin(), res_row_grades.end(), std::back_inserter(result.row_grades));
  for (int i = 0; i < cols_to_keep.size(); i++) {
    result.set_col(i, new_cols[i]);
  }
  mpp_utils::assign_grade_indices(result);
}

template <typename GradedMatrixInput, typename GradedMatrixOutput>
inline void minimize_with_kept_rows(GradedMatrixInput& M,
                                    GradedMatrixOutput& result,
                                    std::vector<phat::index>& kept_rows) {
  if constexpr (has_minimize_with_kept_rows<GradedMatrixInput, GradedMatrixOutput>) {
    mpfree::minimize(M, result, &kept_rows);
  } else {
    minimize_with_kept_rows_fallback(M, result, kept_rows);
  }
}

template <typename index_type>
inline mpfree_generator_matrix_output<index_type> convert_generator_matrix_to_output(
    const Graded_matrix& kernel_basis,
    const std::vector<Grade>& chain_row_grades,
    const std::vector<phat::index>& chain_row_indices,
    const std::vector<phat::index>& surviving_rows,
    const std::vector<std::size_t>& row_order,
    const Graded_matrix& min_rep) {
  mpfree_generator_matrix_output<index_type> out;
  out.row_indices.reserve(chain_row_indices.size());
  for (const auto row_idx : chain_row_indices) {
    out.row_indices.push_back(static_cast<index_type>(row_idx));
  }
  out.row_grades.reserve(chain_row_grades.size());
  for (const auto& row_grade : chain_row_grades) {
    out.row_grades.emplace_back(row_grade.at[0], row_grade.at[1]);
  }

  out.column_grades.reserve(row_order.size());
  out.columns.reserve(row_order.size());
  for (std::size_t output_idx : row_order) {
    const auto generator_idx = surviving_rows[output_idx];
    out.column_grades.emplace_back(min_rep.row_grades[output_idx].at[0], min_rep.row_grades[output_idx].at[1]);
    std::vector<phat::index> column;
    kernel_basis.get_col(generator_idx, column);
    std::vector<index_type> support;
    support.reserve(column.size());
    for (const auto row_idx : column) {
      support.push_back(static_cast<index_type>(row_idx));
    }
    out.columns.push_back(std::move(support));
  }
  return out;
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

template <typename index_type, typename contiguous_slicer_type>
inline mpfree_interface_input<index_type> convert_contiguous_slicer_to_input(contiguous_slicer_type& slicer) {
  mpfree_interface_input<index_type> input;

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
          "mpfree contiguous bridge expects 1-critical contiguous slicers with exactly 2 filtration parameters.");
    }
    input.filtration_values[i] = std::make_pair(filtrations[i](0, 0), filtrations[i](0, 1));
  }

  return input;
}

template <typename index_type>
inline mpfree_raw_result compute_mpfree_minpres_raw(const mpfree_interface_input<index_type>& input,
                                                    int degree,
                                                    bool use_chunk,
                                                    bool use_clearing,
                                                    bool verbose_output,
                                                    bool capture_generators) {
  (void)verbose_output;
#if MPFREE_TIMERS
  const bool need_global_state_lock = true;
#else
  const bool need_global_state_lock = false;
#endif
  std::optional<std::lock_guard<std::mutex> > global_state_lock;
  if (need_global_state_lock) {
    global_state_lock.emplace(mpfree_interface_mutex());
  }

  if (degree < 0) {
    throw std::invalid_argument("mpfree interface expects a non-negative homological degree.");
  }

  if (input.filtration_values.size() != input.boundaries.size() ||
      input.filtration_values.size() != input.dimensions.size()) {
    throw std::invalid_argument("Invalid multipers input: sizes of filtrations, boundaries and dimensions differ.");
  }

  if (!std::is_sorted(input.dimensions.begin(), input.dimensions.end())) {
    throw std::invalid_argument("Dimensions are expected to be sorted in non-decreasing order.");
  }

  const std::size_t start_dm1 = first_index_of_dimension(input.dimensions, degree - 1);
  const std::size_t start_d = first_index_of_dimension(input.dimensions, degree);
  const std::size_t start_dp1 = first_index_of_dimension(input.dimensions, degree + 1);
  const std::size_t start_dp2 = first_index_of_dimension(input.dimensions, degree + 2);

  const auto n_dm1 = start_d - start_dm1;
  const auto n_d = start_dp1 - start_d;
  const auto n_dp1 = start_dp2 - start_dp1;

  using Pre_column = mpp_utils::Pre_column_struct<Grade>;
  std::vector<std::vector<Pre_column> > pre_matrices(2);
  pre_matrices[0].reserve(n_dp1);
  pre_matrices[1].reserve(n_d);

  for (std::size_t i = start_dp1; i < start_dp2; ++i) {
    auto boundary = convert_boundary(input.boundaries[i], start_d, start_dp1, "Upper matrix", i - start_dp1);
    auto grade = pair_to_grade(input.filtration_values[i]);
    pre_matrices[0].emplace_back(static_cast<mpp_utils::index>(i - start_dp1), grade, boundary);
  }

  for (std::size_t i = start_d; i < start_dp1; ++i) {
    auto boundary = convert_boundary(input.boundaries[i], start_dm1, start_d, "Lower matrix", i - start_d);
    auto grade = pair_to_grade(input.filtration_values[i]);
    pre_matrices[1].emplace_back(static_cast<mpp_utils::index>(i - start_d), grade, boundary);
  }

  std::vector<Graded_matrix> matrices;
  mpp_utils::create_graded_matrices_from_pre_column_struct(
      pre_matrices, matrices, static_cast<int>(n_dm1), false, true);

  if (matrices.size() != 2) {
    throw std::runtime_error("Internal mpfree conversion failure: expected two graded matrices.");
  }

  Graded_matrix gm_upper = std::move(matrices[0]);
  Graded_matrix gm_lower = std::move(matrices[1]);

  if (!capture_generators) {
    mpfree_raw_result out;
    mpfree::compute_minimal_presentation(gm_upper, gm_lower, out.min_rep, use_chunk, use_clearing);
    return out;
  }

  typedef typename mpfree::Extend_matrix<Graded_matrix>::Type Graded_matrix_extended;

  if (!mpp_utils::is_colex_sorted(gm_upper)) {
    mpp_utils::to_colex_order(gm_upper, true, false);
    gm_upper.grade_indices_assigned = false;
  }
  if (!mpp_utils::is_colex_sorted_columns(gm_lower)) {
    mpp_utils::to_colex_order(gm_lower, false, false);
    gm_lower.grade_indices_assigned = false;
  }
  if (!gm_upper.grade_indices_assigned || !gm_lower.grade_indices_assigned) {
    mpp_utils::assign_grade_indices_of_pair(gm_upper, gm_lower);
  }

  Graded_matrix_extended GM1(&gm_upper);
  Graded_matrix_extended GM2(&gm_lower);
  GM1.assign_slave_matrix();
  GM1.assign_pivots();
  GM2.assign_slave_matrix();
  GM2.assign_pivots();
  GM1.pq_row.resize(GM1.num_grades_y);
  GM2.pq_row.resize(GM2.num_grades_y);

  mpfree_raw_result out;

  if (use_chunk) {
    chunk_preprocessing_with_surviving_rows(GM1, GM2, out.chain_row_indices);
  } else {
    out.chain_row_indices.reserve(gm_lower.grades.size());
    for (phat::index i = 0; i < static_cast<phat::index>(gm_lower.grades.size()); ++i) {
      out.chain_row_indices.push_back(i);
    }
  }

  Graded_matrix MG_base, Ker_base;
  Graded_matrix_extended MG(&MG_base), Ker(&Ker_base);
  GM1.grid_scheduler = mpfree::Grid_scheduler(GM1);
  mpfree::min_gens(GM1, MG, use_clearing);
  gm_upper = Graded_matrix();

  GM2.grid_scheduler = mpfree::Grid_scheduler(GM2);
  mpfree::ker_basis(GM2, Ker, MG, use_clearing);
  out.chain_row_grades = gm_lower.grades;
  gm_lower = Graded_matrix();

  Graded_matrix semi_min_rep_base;
  Graded_matrix_extended semi_min_rep(&semi_min_rep_base);
  mpfree::reparameterize(MG, Ker, semi_min_rep);
  MG_base = Graded_matrix();

  out.kernel_basis = Ker_base;
  out.has_generators = true;
  minimize_with_kept_rows(semi_min_rep, out.min_rep, out.surviving_rows);
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
  auto raw = detail::compute_mpfree_minpres_raw(input, degree, use_chunk, use_clearing, verbose_output, false);
  return detail::convert_minpres_to_output<index_type>(raw.min_rep, degree, full_resolution);
}

template <typename index_type>
mpfree_minpres_with_generators_output<index_type> mpfree_minpres_with_generators_interface(
    const mpfree_interface_input<index_type>& input,
    int degree,
    bool full_resolution,
    bool use_chunk,
    bool use_clearing,
    bool verbose_output) {
  auto raw = detail::compute_mpfree_minpres_raw(input, degree, use_chunk, use_clearing, verbose_output, true);
  auto row_order = detail::row_order_for_output(raw.min_rep, full_resolution);
  auto generator_matrix = detail::convert_generator_matrix_to_output<index_type>(
      raw.kernel_basis, raw.chain_row_grades, raw.chain_row_indices, raw.surviving_rows, row_order, raw.min_rep);

  mpfree_minpres_with_generators_output<index_type> out;
  out.minimal_presentation = detail::convert_minpres_to_output<index_type>(raw.min_rep, degree, full_resolution);
  out.generator_matrix = std::move(generator_matrix);
  return out;
}

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
template <typename contiguous_slicer_type>
inline contiguous_f64_complex mpfree_minpres_contiguous_interface(contiguous_slicer_type& input,
                                                                  int degree,
                                                                  bool full_resolution,
                                                                  bool use_chunk,
                                                                  bool use_clearing,
                                                                  bool verbose_output) {
  auto converted_input = detail::convert_contiguous_slicer_to_input<int>(input);
  auto out =
      mpfree_minpres_interface<int>(converted_input, degree, full_resolution, use_chunk, use_clearing, verbose_output);
  return build_contiguous_f64_slicer_from_output<int>(out.filtration_values, out.boundaries, out.dimensions);
}

template <typename contiguous_slicer_type>
inline std::pair<contiguous_f64_complex, mpfree_generator_matrix_output<int> >
mpfree_minpres_with_generators_contiguous_interface(contiguous_slicer_type& input,
                                                    int degree,
                                                    bool full_resolution,
                                                    bool use_chunk,
                                                    bool use_clearing,
                                                    bool verbose_output) {
  auto converted_input = detail::convert_contiguous_slicer_to_input<int>(input);
  auto out = mpfree_minpres_with_generators_interface<int>(
      converted_input, degree, full_resolution, use_chunk, use_clearing, verbose_output);
  return std::make_pair(build_contiguous_f64_slicer_from_output<int>(out.minimal_presentation.filtration_values,
                                                                     out.minimal_presentation.boundaries,
                                                                     out.minimal_presentation.dimensions),
                        std::move(out.generator_matrix));
}
#endif

#else

template <typename index_type>
mpfree_interface_output<index_type> mpfree_minpres_interface(const mpfree_interface_input<index_type>&,
                                                             int,
                                                             bool,
                                                             bool,
                                                             bool,
                                                             bool) {
  throw std::runtime_error(
      "mpfree interface is not available at compile time. Install/checkout mpfree headers and rebuild.");
}

template <typename index_type>
mpfree_minpres_with_generators_output<index_type>
mpfree_minpres_with_generators_interface(const mpfree_interface_input<index_type>&, int, bool, bool, bool, bool) {
  throw std::runtime_error(
      "mpfree interface is not available at compile time. Install/checkout mpfree headers and rebuild.");
}

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
template <typename contiguous_slicer_type>
inline contiguous_f64_complex mpfree_minpres_contiguous_interface(contiguous_slicer_type&,
                                                                  int,
                                                                  bool,
                                                                  bool,
                                                                  bool,
                                                                  bool) {
  throw std::runtime_error(
      "mpfree interface is not available at compile time. Install/checkout mpfree headers and rebuild.");
}

template <typename contiguous_slicer_type>
inline std::pair<contiguous_f64_complex, mpfree_generator_matrix_output<int> >
mpfree_minpres_with_generators_contiguous_interface(contiguous_slicer_type&, int, bool, bool, bool, bool) {
  throw std::runtime_error(
      "mpfree interface is not available at compile time. Install/checkout mpfree headers and rebuild.");
}
#endif

#endif

}  // namespace multipers

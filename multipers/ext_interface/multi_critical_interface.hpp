#pragma once

#include "backend_log_policy.hpp"

#include <algorithm>
#include <cmath>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace multipers {

template <typename index_type>
struct multi_critical_interface_input {
  std::vector<std::vector<std::pair<double, double> > > filtration_values;
  std::vector<std::vector<index_type> > boundaries;
  std::vector<int> dimensions;
};

template <typename index_type>
struct multi_critical_interface_output {
  std::vector<std::pair<double, double> > filtration_values;
  std::vector<std::vector<index_type> > boundaries;
  std::vector<int> dimensions;
};

inline bool multi_critical_interface_available();

template <typename index_type>
multi_critical_interface_output<index_type> multi_critical_resolution_interface(
    const multi_critical_interface_input<index_type>& input,
    bool use_logpath = true,
    bool use_multi_chunk = true,
    bool verbose_output = false);

template <typename index_type>
multi_critical_interface_output<index_type> multi_critical_minpres_interface(
    const multi_critical_interface_input<index_type>& input,
    int degree,
    bool use_logpath = true,
    bool use_multi_chunk = true,
    bool verbose_output = false,
    bool swedish = false);

template <typename index_type>
std::vector<multi_critical_interface_output<index_type> > multi_critical_minpres_all_interface(
    const multi_critical_interface_input<index_type>& input,
    bool use_logpath = true,
    bool use_multi_chunk = true,
    bool verbose_output = false,
    bool swedish = false);

}  // namespace multipers

#ifndef MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE
#define MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE 0
#endif

#if !MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE
#include "contiguous_slicer_bridge.hpp"

namespace multipers {

template <typename kcontiguous_slicer_type>
contiguous_f64_complex multi_critical_resolution_contiguous_interface(kcontiguous_slicer_type& input,
                                                                      bool use_logpath = true,
                                                                      bool use_multi_chunk = true,
                                                                      bool verbose_output = false);

}  // namespace multipers
#endif

#if !MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE && __has_include(<multi_critical/free_resolution.h>) && \
    __has_include(<mpfree/mpfree.h>) && __has_include(<multi_chunk/multi_chunk.h>)
#define MULTIPERS_HAS_MULTI_CRITICAL_INTERFACE 1

#ifndef MULTI_CRITICAL_TIMERS
#define MULTI_CRITICAL_TIMERS 0
#endif

#if MULTI_CRITICAL_TIMERS
#if __has_include(<multi_critical/boost_timers.h>)
#include <multi_critical/boost_timers.h>
#endif
#else
namespace multi_critical {
struct multipers_timer_stub {
  void start() {}

  void stop() {}

  void resume() {}
};

static multipers_timer_stub test_timer_1;
static multipers_timer_stub test_timer_2;
}  // namespace multi_critical
#endif

#include <mpfree/mpfree.h>
#include <mpp_utils/Graded_matrix.h>
#include <mpp_utils/sorting_utility.h>
#include <multi_chunk/multi_chunk.h>
#include <multi_critical/basic.h>
#include <multi_critical/free_resolution.h>

#else
#define MULTIPERS_HAS_MULTI_CRITICAL_INTERFACE 0
#endif

namespace multipers {

inline bool multi_critical_interface_available() { return MULTIPERS_HAS_MULTI_CRITICAL_INTERFACE; }

#if MULTIPERS_HAS_MULTI_CRITICAL_INTERFACE

namespace multi_critical_detail {

inline std::mutex& multi_critical_interface_mutex() {
  // Only timer globals still require process-global serialization.
  static std::mutex m;
  return m;
}

inline bool multi_critical_interface_needs_global_state_lock() {
#if MULTI_CRITICAL_TIMERS || MULTI_CHUNK_TIMERS || MPFREE_TIMERS
  return true;
#else
  return false;
#endif
}

using Graded_matrix = mpp_utils::Graded_matrix<phat::vector_vector>;

template <typename index_type>
struct parser_column {
  std::vector<double> raw_grades;
  std::vector<long> boundary;
};

template <typename index_type>
class multi_critical_input_parser {
 public:
  explicit multi_critical_input_parser(const multi_critical_interface_input<index_type>& input)
      : levels_(build_levels(input)), offsets_(levels_.size(), 0) {}

  int number_of_parameters() { return 2; }

  int number_of_levels() { return static_cast<int>(levels_.size()); }

  bool has_next_column(int level) {
    validate_level(level);
    return offsets_[level - 1] < levels_[level - 1].size();
  }

  bool has_grades_on_last_level() { return false; }

  template <typename OutputIterator1, typename OutputIterator2>
  void next_column(int level, OutputIterator1 out1, OutputIterator2 out2) {
    validate_level(level);
    if (!has_next_column(level)) {
      throw std::out_of_range("No more columns on requested level.");
    }
    const auto& col = levels_[level - 1][offsets_[level - 1]++];
    for (const auto val : col.raw_grades) {
      *out1++ = val;
    }
    for (const auto idx : col.boundary) {
      *out2++ = std::make_pair(idx, 1);
    }
  }

  int number_of_generators(int level) {
    validate_level(level);
    return static_cast<int>(levels_[level - 1].size());
  }

  void reset(int level) {
    validate_level(level);
    offsets_[level - 1] = 0;
  }

  void reset_all() { std::fill(offsets_.begin(), offsets_.end(), 0); }

 private:
  std::vector<std::vector<parser_column<index_type> > > levels_;
  std::vector<std::size_t> offsets_;

  void validate_level(int level) const {
    if (level < 1 || static_cast<std::size_t>(level) > levels_.size()) {
      throw std::out_of_range("Requested level is out of parser range.");
    }
  }

  static std::vector<std::vector<parser_column<index_type> > > build_levels(
      const multi_critical_interface_input<index_type>& input) {
    if (input.filtration_values.size() != input.boundaries.size() ||
        input.filtration_values.size() != input.dimensions.size()) {
      throw std::invalid_argument(
          "Invalid multi_critical input: sizes of filtrations, boundaries and dimensions differ.");
    }

    const std::size_t num_generators = input.dimensions.size();
    if (num_generators == 0) {
      return std::vector<std::vector<parser_column<index_type> > >();
    }

    if (!std::is_sorted(input.dimensions.begin(), input.dimensions.end())) {
      throw std::invalid_argument("Dimensions are expected to be sorted in non-decreasing order.");
    }

    const int max_dim = input.dimensions.back();
    if (max_dim < 0) {
      throw std::invalid_argument("Dimensions must be non-negative.");
    }

    std::vector<std::vector<std::size_t> > indices_by_level(static_cast<std::size_t>(max_dim) + 1);
    std::vector<long> shifted_indices(num_generators, -1);

    for (std::size_t i = 0; i < num_generators; ++i) {
      const int dim = input.dimensions[i];
      if (dim < 0 || dim > max_dim) {
        throw std::invalid_argument("Generator dimension out of valid range.");
      }
      const std::size_t level_idx = static_cast<std::size_t>(max_dim - dim);
      shifted_indices[i] = static_cast<long>(indices_by_level[level_idx].size());
      indices_by_level[level_idx].push_back(i);
    }

    std::vector<std::vector<parser_column<index_type> > > levels(static_cast<std::size_t>(max_dim) + 2);

    for (std::size_t level_idx = 0; level_idx < indices_by_level.size(); ++level_idx) {
      auto& level_cols = levels[level_idx];
      level_cols.reserve(indices_by_level[level_idx].size());
      for (const auto global_idx : indices_by_level[level_idx]) {
        const int simplex_dim = input.dimensions[global_idx];
        parser_column<index_type> col;

        const auto& grades = input.filtration_values[global_idx];
        if (grades.empty()) {
          throw std::invalid_argument("Each generator must have at least one filtration grade.");
        }
        col.raw_grades.reserve(grades.size() * 2);
        for (const auto& grade : grades) {
          // if (!std::isfinite(grade.first) || !std::isfinite(grade.second)) {
          //   throw std::invalid_argument("multi_critical interface expects finite bifiltration values.");
          // }
          col.raw_grades.push_back(grade.first);
          col.raw_grades.push_back(grade.second);
        }

        const auto& boundary = input.boundaries[global_idx];
        if (simplex_dim == 0 && !boundary.empty()) {
          throw std::invalid_argument("Dimension-0 generators must have empty boundaries.");
        }
        col.boundary.reserve(boundary.size());
        for (const auto bd_idx_typed : boundary) {
          if (bd_idx_typed < 0) {
            throw std::invalid_argument("Boundary index cannot be negative.");
          }
          const auto bd_idx = static_cast<std::size_t>(bd_idx_typed);
          if (bd_idx >= num_generators) {
            throw std::invalid_argument("Boundary index out of range.");
          }
          if (simplex_dim > 0 && input.dimensions[bd_idx] != simplex_dim - 1) {
            throw std::invalid_argument("Boundary index does not point to previous dimension.");
          }
          if (shifted_indices[bd_idx] < 0) {
            throw std::invalid_argument("Internal index conversion failed.");
          }
          col.boundary.push_back(shifted_indices[bd_idx]);
        }
        std::sort(col.boundary.begin(), col.boundary.end());

        level_cols.push_back(std::move(col));
      }
    }

    return levels;
  }
};

template <typename index_type>
inline multi_critical_interface_output<index_type> append_columns(
    const Graded_matrix& matrix,
    int out_dimension,
    index_type row_shift,
    multi_critical_interface_output<index_type> out = multi_critical_interface_output<index_type>()) {
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
    out.dimensions.push_back(out_dimension);
  }
  return out;
}

template <typename index_type>
inline multi_critical_interface_output<index_type> convert_chain_complex(const std::vector<Graded_matrix>& matrices) {
  multi_critical_interface_output<index_type> out;
  if (matrices.empty()) {
    return out;
  }

  const std::size_t num_dims = matrices.size();
  std::vector<index_type> counts(num_dims, 0);
  for (std::size_t i = 0; i < num_dims; ++i) {
    const std::size_t matrix_idx = num_dims - 1 - i;
    counts[i] = static_cast<index_type>(matrices[matrix_idx].get_num_cols());
  }
  std::vector<index_type> offsets(num_dims, 0);
  for (std::size_t i = 1; i < num_dims; ++i) {
    offsets[i] = offsets[i - 1] + counts[i - 1];
  }

  for (std::size_t i = 0; i < num_dims; ++i) {
    const std::size_t matrix_idx = num_dims - 1 - i;
    const int out_dimension = static_cast<int>(i);
    const index_type row_shift = out_dimension == 0 ? 0 : offsets[i - 1];
    out = append_columns(matrices[matrix_idx], out_dimension, row_shift, std::move(out));
  }

  return out;
}

template <typename index_type>
inline multi_critical_interface_output<index_type> convert_minpres(Graded_matrix& min_rep, int degree) {
  multi_critical_interface_output<index_type> out;

  for (const auto& row_grade : min_rep.row_grades) {
    out.filtration_values.emplace_back(row_grade.at[0], row_grade.at[1]);
    out.boundaries.emplace_back();
    out.dimensions.push_back(degree);
  }

  return append_columns(min_rep, degree + 1, 0, std::move(out));
}

template <typename index_type>
inline std::vector<Graded_matrix> compute_free_resolution_matrices(
    const multi_critical_interface_input<index_type>& input,
    bool use_logpath,
    bool use_multi_chunk,
    bool verbose_output) {
  if (input.filtration_values.empty()) {
    return std::vector<Graded_matrix>();
  }

  multi_critical_input_parser<index_type> parser(input);
  (void)verbose_output;

  std::vector<Graded_matrix> matrices;
  multi_critical::free_resolution(parser, matrices, use_logpath);

  for (std::size_t i = 0; i < matrices.size(); ++i) {
    Graded_matrix& mat = matrices[i];
    if (!mpp_utils::is_lex_sorted(mat)) {
      mpp_utils::to_lex_order(mat, i + 1 < matrices.size(), false);
    }
  }

  if (use_multi_chunk) {
    multi_chunk::compress(matrices);
  }

  return matrices;
}

template <typename kcontiguous_slicer_type>
inline multi_critical_interface_input<int> multi_critical_input_from_kcontiguous_slicer(
    kcontiguous_slicer_type& slicer) {
  multi_critical_interface_input<int> input;

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
      input.boundaries[i].push_back(static_cast<int>(idx));
    }

    if (filtrations[i].num_parameters() != 2) {
      throw std::invalid_argument("multi_critical contiguous bridge expects bifiltration values with 2 parameters.");
    }
    const std::size_t num_grades = filtrations[i].num_generators();
    if (num_grades == 0) {
      throw std::invalid_argument("multi_critical contiguous bridge expects at least one grade per generator.");
    }
    input.filtration_values[i].reserve(num_grades);
    for (std::size_t g = 0; g < num_grades; ++g) {
      input.filtration_values[i].emplace_back(filtrations[i](g, 0), filtrations[i](g, 1));
    }
  }

  return input;
}

}  // namespace multi_critical_detail

template <typename index_type>
multi_critical_interface_output<index_type> multi_critical_resolution_interface(
    const multi_critical_interface_input<index_type>& input,
    bool use_logpath,
    bool use_multi_chunk,
    bool verbose_output) {
  std::optional<std::lock_guard<std::mutex> > lock;
  if (multi_critical_detail::multi_critical_interface_needs_global_state_lock()) {
    lock.emplace(multi_critical_detail::multi_critical_interface_mutex());
  }

  auto matrices =
      multi_critical_detail::compute_free_resolution_matrices(input, use_logpath, use_multi_chunk, verbose_output);
  if (matrices.size() <= 1) {
    return multi_critical_interface_output<index_type>();
  }

  // Mirrors the historical shift_dimension=-2 behavior from the SCC file pipeline.
  std::vector<multi_critical_detail::Graded_matrix> shifted_matrices(matrices.begin(), matrices.end() - 1);
  return multi_critical_detail::convert_chain_complex<index_type>(shifted_matrices);
}

#if !MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE
template <typename kcontiguous_slicer_type>
inline contiguous_f64_complex multi_critical_resolution_contiguous_interface(kcontiguous_slicer_type& input,
                                                                             bool use_logpath,
                                                                             bool use_multi_chunk,
                                                                             bool verbose_output) {
  auto converted_input = multi_critical_detail::multi_critical_input_from_kcontiguous_slicer(input);
  auto out = multi_critical_resolution_interface<int>(converted_input, use_logpath, use_multi_chunk, verbose_output);
  return build_contiguous_f64_slicer_from_output<int>(out.filtration_values, out.boundaries, out.dimensions);
}
#endif

template <typename index_type>
multi_critical_interface_output<index_type> multi_critical_minpres_interface(
    const multi_critical_interface_input<index_type>& input,
    int degree,
    bool use_logpath,
    bool use_multi_chunk,
    bool verbose_output,
    bool /*swedish*/) {
  std::optional<std::lock_guard<std::mutex> > lock;
  if (multi_critical_detail::multi_critical_interface_needs_global_state_lock()) {
    lock.emplace(multi_critical_detail::multi_critical_interface_mutex());
  }

  if (degree < 0) {
    throw std::invalid_argument("multi_critical minimal presentation expects a non-negative degree.");
  }

  auto matrices =
      multi_critical_detail::compute_free_resolution_matrices(input, use_logpath, use_multi_chunk, verbose_output);
  if (matrices.size() < 2) {
    return multi_critical_interface_output<index_type>();
  }

  const int matrix_index = static_cast<int>(matrices.size()) - 1 - degree;
  if (matrix_index < 1 || matrix_index >= static_cast<int>(matrices.size())) {
    return multi_critical_interface_output<index_type>();
  }

  auto first = matrices[matrix_index - 1];
  auto second = matrices[matrix_index];
  multi_critical_detail::Graded_matrix min_rep;

  (void)verbose_output;
  mpfree::compute_minimal_presentation(first, second, min_rep, false, false);

  return multi_critical_detail::convert_minpres<index_type>(min_rep, degree);
}

template <typename index_type>
std::vector<multi_critical_interface_output<index_type> > multi_critical_minpres_all_interface(
    const multi_critical_interface_input<index_type>& input,
    bool use_logpath,
    bool use_multi_chunk,
    bool verbose_output,
    bool /*swedish*/) {
  std::optional<std::lock_guard<std::mutex> > lock;
  if (multi_critical_detail::multi_critical_interface_needs_global_state_lock()) {
    lock.emplace(multi_critical_detail::multi_critical_interface_mutex());
  }

  std::vector<multi_critical_interface_output<index_type> > out;

  auto matrices =
      multi_critical_detail::compute_free_resolution_matrices(input, use_logpath, use_multi_chunk, verbose_output);
  if (matrices.size() < 2) {
    return out;
  }

  out.reserve(matrices.size() - 1);
  (void)verbose_output;

  for (std::size_t i = 0; i + 1 < matrices.size(); ++i) {
    auto first = matrices[i];
    auto second = matrices[i + 1];
    multi_critical_detail::Graded_matrix min_rep;
    mpfree::compute_minimal_presentation(first, second, min_rep, false, false);
    out.push_back(multi_critical_detail::convert_minpres<index_type>(min_rep, static_cast<int>(i)));
  }
  return out;
}

#else

template <typename index_type>
multi_critical_interface_output<index_type>
multi_critical_resolution_interface(const multi_critical_interface_input<index_type>&, bool, bool, bool) {
  throw std::runtime_error(
      "multi_critical interface is not available at compile time. Install/checkout headers and rebuild.");
}

#if !MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE
template <typename kcontiguous_slicer_type>
inline contiguous_f64_complex multi_critical_resolution_contiguous_interface(kcontiguous_slicer_type&,
                                                                             bool,
                                                                             bool,
                                                                             bool) {
  throw std::runtime_error(
      "multi_critical interface is not available at compile time. Install/checkout headers and rebuild.");
}
#endif

template <typename index_type>
multi_critical_interface_output<index_type>
multi_critical_minpres_interface(const multi_critical_interface_input<index_type>&, int, bool, bool, bool, bool) {
  throw std::runtime_error(
      "multi_critical interface is not available at compile time. Install/checkout headers and rebuild.");
}

template <typename index_type>
std::vector<multi_critical_interface_output<index_type> >
multi_critical_minpres_all_interface(const multi_critical_interface_input<index_type>&, bool, bool, bool, bool) {
  throw std::runtime_error(
      "multi_critical interface is not available at compile time. Install/checkout headers and rebuild.");
}

#endif

}  // namespace multipers

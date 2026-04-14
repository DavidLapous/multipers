#pragma once

#include "backend_log_flags.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef FUNCTION_DELAUNAY_TIMERS
#define FUNCTION_DELAUNAY_TIMERS 0
#endif

#ifndef MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
#define MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE 0
#endif

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
#include "contiguous_slicer_bridge.hpp"
#include "Simplex_tree_multi_interface.h"
#endif

#if FUNCTION_DELAUNAY_TIMERS
#if __has_include(<function_delaunay/boost_timers.h>)
#include <function_delaunay/boost_timers.h>
#endif
#else
struct multipers_function_delaunay_timer_stub {
  void start() {}

  void stop() {}

  void resume() {}
};

static multipers_function_delaunay_timer_stub test_timer_1;
static multipers_function_delaunay_timer_stub test_timer_2;
static multipers_function_delaunay_timer_stub test_timer_3;
static multipers_function_delaunay_timer_stub test_timer_4;
#endif

namespace multipers {

template <typename index_type>
struct function_delaunay_interface_input {
  std::vector<std::vector<double> > points;
  std::vector<double> function_values;
  bool recover_ids = false;
};

template <typename index_type>
struct function_delaunay_interface_output {
  std::vector<std::pair<double, double> > filtration_values;
  std::vector<std::vector<index_type> > boundaries;
  std::vector<int> dimensions;
};

inline bool function_delaunay_interface_available();

template <typename index_type>
function_delaunay_interface_output<index_type> function_delaunay_interface(
    const function_delaunay_interface_input<index_type>& input,
    int degree = -1,
    bool use_multi_chunk = false,
    bool verbose_output = false);

}  // namespace multipers

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
namespace multipers {

using function_delaunay_simplextree_filtration =
    Gudhi::multi_filtration::Multi_parameter_filtration<double, false, !false>;
using function_delaunay_simplextree_interface_output =
    Gudhi::multiparameter::python_interface::Simplex_tree_multi_interface<function_delaunay_simplextree_filtration,
                                                                          double>;

template <typename index_type>
contiguous_f64_complex function_delaunay_interface_contiguous_slicer(
    const function_delaunay_interface_input<index_type>& input,
    int degree = -1,
    bool use_multi_chunk = false,
    bool verbose_output = false);

template <typename index_type>
function_delaunay_simplextree_interface_output function_delaunay_simplextree_interface(
    const function_delaunay_interface_input<index_type>& input,
    bool verbose_output = false);

}  // namespace multipers
#endif

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE && __has_include(<function_delaunay/function_delaunay_with_meb.h>) && \
    __has_include(<function_delaunay/Point_with_densities.h>)
#define MULTIPERS_HAS_FUNCTION_DELAUNAY_INTERFACE 1
#include <Eigen/Core>
#include <function_delaunay/Point_with_densities.h>
#include <function_delaunay/function_delaunay_with_meb.h>
#include <mpp_utils/Graded_matrix.h>
#include <multi_chunk/multi_chunk.h>
#include <mpfree/mpfree.h>
#else
#define MULTIPERS_HAS_FUNCTION_DELAUNAY_INTERFACE 0
#endif

namespace multipers {

inline bool function_delaunay_interface_available() { return MULTIPERS_HAS_FUNCTION_DELAUNAY_INTERFACE; }

#if MULTIPERS_HAS_FUNCTION_DELAUNAY_INTERFACE

namespace detail {

inline std::mutex& function_delaunay_interface_mutex() {
  // stream_silencer swaps the process-global std::cout/std::cerr buffers, so
  // function_delaunay calls must serialize while that redirection is possible.
  static std::mutex m;
  return m;
}

class null_streambuf : public std::streambuf {
 protected:
  int overflow(int c) override { return traits_type::not_eof(c); }
};

class stream_silencer {
 public:
  explicit stream_silencer(bool silence) : silence_(silence), null_stream_(&null_buffer_) {
    if (!silence_) {
      return;
    }
    old_cout_ = std::cout.rdbuf(null_stream_.rdbuf());
    old_cerr_ = std::cerr.rdbuf(null_stream_.rdbuf());
  }

  ~stream_silencer() {
    if (!silence_) {
      return;
    }
    std::cout.rdbuf(old_cout_);
    std::cerr.rdbuf(old_cerr_);
  }

 private:
  bool silence_;
  null_streambuf null_buffer_;
  std::ostream null_stream_;
  std::streambuf* old_cout_ = nullptr;
  std::streambuf* old_cerr_ = nullptr;
};

using Graded_matrix = mpp_utils::Graded_matrix<>;

template <typename index_type>
inline std::vector<index_type> sorted_to_original_vertex_ids(
    const std::vector<function_delaunay::Point_with_densities>& points) {
  std::vector<index_type> out;
  out.reserve(points.size());
  for (const auto& point : points) {
    if (point.idx < 0) {
      throw std::runtime_error("function_delaunay bridge expected non-negative point ids.");
    }
    out.push_back(static_cast<index_type>(point.idx));
  }
  return out;
}

template <typename index_type>
inline void recover_vertex_ids(function_delaunay_interface_output<index_type>& out,
                               const std::vector<index_type>& sorted_to_original) {
  const std::size_t num_vertices = sorted_to_original.size();
  if (out.dimensions.size() < num_vertices || out.filtration_values.size() < num_vertices ||
      out.boundaries.size() < num_vertices) {
    throw std::runtime_error("function_delaunay bridge expected a full 0-simplex block.");
  }

  std::vector<std::pair<double, double> > vertex_filtrations(num_vertices);
  std::vector<std::vector<index_type> > vertex_boundaries(num_vertices);
  std::vector<uint8_t> seen(num_vertices, 0);
  for (std::size_t sorted_idx = 0; sorted_idx < num_vertices; ++sorted_idx) {
    if (out.dimensions[sorted_idx] != 0) {
      throw std::runtime_error("function_delaunay bridge expected vertices to occupy the leading dimension-0 block.");
    }
    const auto original_idx = static_cast<std::size_t>(sorted_to_original[sorted_idx]);
    if (original_idx >= num_vertices) {
      throw std::runtime_error("function_delaunay bridge recovered vertex id out of range.");
    }
    vertex_filtrations[original_idx] = std::move(out.filtration_values[sorted_idx]);
    vertex_boundaries[original_idx] = std::move(out.boundaries[sorted_idx]);
    seen[original_idx] = 1;
  }
  if (std::find(seen.begin(), seen.end(), uint8_t{0}) != seen.end()) {
    throw std::runtime_error("function_delaunay bridge recovered a non-permutation of vertex ids.");
  }
  std::copy(vertex_filtrations.begin(), vertex_filtrations.end(), out.filtration_values.begin());
  std::copy(vertex_boundaries.begin(), vertex_boundaries.end(), out.boundaries.begin());

  for (std::size_t i = num_vertices; i < out.dimensions.size(); ++i) {
    if (out.dimensions[i] != 1) {
      continue;
    }
    for (auto& boundary_idx : out.boundaries[i]) {
      if (boundary_idx < 0 || static_cast<std::size_t>(boundary_idx) >= num_vertices) {
        throw std::runtime_error("function_delaunay bridge expected edge boundaries to reference vertices only.");
      }
      boundary_idx = sorted_to_original[boundary_idx];
    }
  }
}

inline std::pair<double, double> grade_to_pair(const Graded_matrix::Grade& grade) {
  if (grade.at.size() < 2) {
    throw std::invalid_argument("function_delaunay interface expects bifiltration grades.");
  }
  return {grade.at[0], grade.at[1]};
}

template <typename index_type>
inline function_delaunay_interface_output<index_type> append_columns(
    const Graded_matrix& matrix,
    int out_dimension,
    index_type row_shift,
    function_delaunay_interface_output<index_type> out = function_delaunay_interface_output<index_type>()) {
  for (phat::index col_idx = 0; col_idx < matrix.get_num_cols(); ++col_idx) {
    out.filtration_values.push_back(grade_to_pair(matrix.grades[col_idx]));
    std::vector<phat::index> local_boundary;
    matrix.get_col(col_idx, local_boundary);
    std::vector<index_type> boundary;
    boundary.reserve(local_boundary.size());
    for (const auto row_idx : local_boundary) {
      boundary.push_back(static_cast<index_type>(row_idx) + row_shift);
    }
    out.boundaries.push_back(std::move(boundary));
    out.dimensions.push_back(out_dimension);
  }
  return out;
}

template <typename index_type>
inline function_delaunay_interface_output<index_type> convert_chain_complex(
    const std::vector<Graded_matrix>& matrices) {
  function_delaunay_interface_output<index_type> out;
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

inline std::vector<double> lowerstar_values_from_points(
    const std::vector<function_delaunay::Point_with_densities>& points) {
  std::vector<double> lowerstar_values;
  lowerstar_values.reserve(points.size());
  for (const auto& point : points) {
    if (point.densities.empty()) {
      throw std::runtime_error("function_delaunay simplex interface expects one function value per point.");
    }
    lowerstar_values.push_back(point.densities[0]);
  }
  return lowerstar_values;
}

inline Gudhi::Simplex_tree<> relabel_simplex_tree_vertices(const Gudhi::Simplex_tree<>& simplex_tree,
                                                           const std::vector<int>& sorted_to_original) {
  struct simplex_record {
    std::vector<int> simplex;
    double filtration;
  };

  std::vector<simplex_record> simplices;
  simplices.reserve(simplex_tree.num_simplices());
  for (const auto simplex_handle : simplex_tree.complex_simplex_range()) {
    simplex_record record;
    record.filtration = simplex_tree.filtration(simplex_handle);
    for (const auto vertex : simplex_tree.simplex_vertex_range(simplex_handle)) {
      if (vertex < 0 || static_cast<std::size_t>(vertex) >= sorted_to_original.size()) {
        throw std::runtime_error("function_delaunay simplex interface recovered vertex id out of range.");
      }
      record.simplex.push_back(sorted_to_original[vertex]);
    }
    std::sort(record.simplex.begin(), record.simplex.end());
    simplices.push_back(std::move(record));
  }

  std::stable_sort(simplices.begin(), simplices.end(), [](const simplex_record& a, const simplex_record& b) {
    return a.simplex.size() < b.simplex.size();
  });

  Gudhi::Simplex_tree<> out;
  for (const auto& record : simplices) {
    out.insert_simplex(record.simplex, record.filtration);
  }
  return out;
}

inline function_delaunay_simplextree_interface_output convert_simplex_tree(
    Gudhi::Simplex_tree<>& simplex_tree,
    const std::vector<double>& lowerstar_values) {
  function_delaunay_simplextree_interface_output out;

  const std::size_t serialized_size = simplex_tree.get_serialization_size();
  std::vector<char> serialized_simplextree(serialized_size);
  if (serialized_size > 0) {
    simplex_tree.serialize(serialized_simplextree.data(), serialized_size);
  }

  std::vector<double> default_values(2, -std::numeric_limits<double>::infinity());
  if (serialized_size > 0) {
    out.from_std(serialized_simplextree.data(), serialized_size, 0, default_values);
  }

  out.fill_lowerstar(lowerstar_values, 1);

  return out;
}

template <typename index_type>
inline function_delaunay_interface_output<index_type> convert_minpres(Graded_matrix& min_rep, int degree) {
  function_delaunay_interface_output<index_type> out;

  for (const auto& row_grade : min_rep.row_grades) {
    out.filtration_values.push_back(grade_to_pair(row_grade));
    out.boundaries.emplace_back();
    out.dimensions.push_back(degree);
  }

  return append_columns(min_rep, degree + 1, 0, std::move(out));
}

}  // namespace detail

template <typename index_type>
function_delaunay_interface_output<index_type> function_delaunay_interface(
    const function_delaunay_interface_input<index_type>& input,
    int degree,
    bool use_multi_chunk,
    bool verbose_output) {
  std::lock_guard<std::mutex> lock(detail::function_delaunay_interface_mutex());

  if (input.points.size() != input.function_values.size()) {
    throw std::invalid_argument("function_delaunay interface expects as many function values as input points.");
  }
  if (input.points.empty()) {
    return function_delaunay_interface_output<index_type>();
  }

  const std::size_t point_dim = input.points[0].size();
  if (point_dim == 0) {
    throw std::invalid_argument("function_delaunay interface expects points with positive ambient dimension.");
  }
  for (const auto& point : input.points) {
    if (point.size() != point_dim) {
      throw std::invalid_argument("All points must have the same ambient dimension.");
    }
  }

  std::vector<function_delaunay::Point_with_densities> points;
  points.reserve(input.points.size());
  for (std::size_t i = 0; i < input.points.size(); ++i) {
    const auto& coords = input.points[i];
    std::vector<double> densities(1, input.function_values[i]);
    points.emplace_back(coords.begin(), coords.end(), densities.begin(), densities.end());
    points.back().idx = static_cast<int>(i);
  }
  std::sort(points.begin(), points.end(), function_delaunay::Lex_sort_by_density());
  if (degree >= 0 && input.recover_ids) {
    throw std::invalid_argument("function_delaunay recover_ids=True is only supported for full-complex outputs.");
  }
  const auto sorted_to_original = detail::sorted_to_original_vertex_ids<index_type>(points);

  const bool effective_verbose_output = backend_log_flags::function_delaunay && verbose_output;
  multi_chunk::verbose = effective_verbose_output;
  const bool old_mpfree_verbose = mpfree::verbose;
  if (backend_log_flags::mpfree && effective_verbose_output) {
    mpfree::verbose = true;
  }

  detail::stream_silencer silencer(!effective_verbose_output);

  std::vector<detail::Graded_matrix> matrices;
  function_delaunay::function_delaunay_with_meb<detail::Graded_matrix>(points, matrices, false);

  if (use_multi_chunk) {
    multi_chunk::compress(matrices);
  }

  if (degree >= 0) {
    if (matrices.size() < 2) {
      if (backend_log_flags::mpfree && effective_verbose_output) {
        mpfree::verbose = old_mpfree_verbose;
      }
      return function_delaunay_interface_output<index_type>();
    }
    const int matrix_idx = static_cast<int>(matrices.size()) - degree - 2;
    if (matrix_idx < 0 || matrix_idx + 1 >= static_cast<int>(matrices.size())) {
      if (backend_log_flags::mpfree && effective_verbose_output) {
        mpfree::verbose = old_mpfree_verbose;
      }
      throw std::invalid_argument("Invalid homological degree for function_delaunay minimal presentation.");
    }
    auto first_matrix = matrices[matrix_idx];
    auto second_matrix = matrices[matrix_idx + 1];
    detail::Graded_matrix min_rep;
    mpfree::compute_minimal_presentation(first_matrix, second_matrix, min_rep, false, false);
    auto out = detail::convert_minpres<index_type>(min_rep, degree);
    if (backend_log_flags::mpfree && effective_verbose_output) {
      mpfree::verbose = old_mpfree_verbose;
    }
    return out;
  }

  auto out = detail::convert_chain_complex<index_type>(matrices);
  if (input.recover_ids) {
    detail::recover_vertex_ids(out, sorted_to_original);
  }
  if (backend_log_flags::mpfree && effective_verbose_output) {
    mpfree::verbose = old_mpfree_verbose;
  }
  return out;
}

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
template <typename index_type>
function_delaunay_simplextree_interface_output function_delaunay_simplextree_interface(
    const function_delaunay_interface_input<index_type>& input,
    bool verbose_output) {
  std::lock_guard<std::mutex> lock(detail::function_delaunay_interface_mutex());

  if (input.points.size() != input.function_values.size()) {
    throw std::invalid_argument("function_delaunay interface expects as many function values as input points.");
  }
  if (input.points.empty()) {
    return function_delaunay_simplextree_interface_output();
  }

  const std::size_t point_dim = input.points[0].size();
  if (point_dim == 0) {
    throw std::invalid_argument("function_delaunay interface expects points with positive ambient dimension.");
  }
  for (const auto& point : input.points) {
    if (point.size() != point_dim) {
      throw std::invalid_argument("All points must have the same ambient dimension.");
    }
  }

  std::vector<function_delaunay::Point_with_densities> points;
  points.reserve(input.points.size());
  for (std::size_t i = 0; i < input.points.size(); ++i) {
    const auto& coords = input.points[i];
    std::vector<double> densities(1, input.function_values[i]);
    points.emplace_back(coords.begin(), coords.end(), densities.begin(), densities.end());
    points.back().idx = static_cast<int>(i);
  }
  std::sort(points.begin(), points.end(), function_delaunay::Lex_sort_by_density());
  const auto sorted_to_original = detail::sorted_to_original_vertex_ids<int>(points);

  const bool effective_verbose_output = backend_log_flags::function_delaunay && verbose_output;
  detail::stream_silencer silencer(!effective_verbose_output);
  Gudhi::Simplex_tree<> simplex_tree;
  function_delaunay::incremental_delaunay_complex(points, simplex_tree, false);
  if (input.recover_ids) {
    simplex_tree = detail::relabel_simplex_tree_vertices(simplex_tree, sorted_to_original);
    return detail::convert_simplex_tree(simplex_tree, input.function_values);
  }
  return detail::convert_simplex_tree(simplex_tree, detail::lowerstar_values_from_points(points));
}

template <typename index_type>
contiguous_f64_complex function_delaunay_interface_contiguous_slicer(
    const function_delaunay_interface_input<index_type>& input,
    int degree,
    bool use_multi_chunk,
    bool verbose_output) {
  auto out = function_delaunay_interface<index_type>(input, degree, use_multi_chunk, verbose_output);
  return build_contiguous_f64_slicer_from_output<index_type>(out.filtration_values, out.boundaries, out.dimensions);
}
#endif

#else

template <typename index_type>
function_delaunay_interface_output<index_type>
function_delaunay_interface(const function_delaunay_interface_input<index_type>&, int, bool, bool) {
  throw std::runtime_error(
      "function_delaunay interface is not available at compile time. Install/checkout headers and rebuild.");
}

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
template <typename index_type>
contiguous_f64_complex
function_delaunay_interface_contiguous_slicer(const function_delaunay_interface_input<index_type>&, int, bool, bool) {
  throw std::runtime_error(
      "function_delaunay interface is not available at compile time. Install/checkout headers and rebuild.");
}

template <typename index_type>
function_delaunay_simplextree_interface_output function_delaunay_simplextree_interface(
    const function_delaunay_interface_input<index_type>&,
    bool) {
  throw std::runtime_error(
      "function_delaunay interface is not available at compile time. Install/checkout headers and rebuild.");
}
#endif

#endif

}  // namespace multipers

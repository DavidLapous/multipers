#pragma once

#include "backend_log_policy.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gudhi/simple_mdspan.h>

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
  std::vector<double> points;
  std::size_t num_points = 0;
  std::size_t num_point_coordinates = 0;
  std::vector<double> function_values;
  std::size_t num_function_parameters = 0;
  bool recover_ids = false;
};

template <typename index_type>
struct function_delaunay_interface_output {
  std::vector<double> filtration_values;
  std::size_t num_parameters = 0;
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
  // Only timer globals still require process-global serialization.
  static std::mutex m;
  return m;
}

inline bool function_delaunay_wrapper_silences_backend_output() { return false; }

inline bool function_delaunay_interface_needs_global_state_lock() {
#if FUNCTION_DELAUNAY_TIMERS || MULTI_CHUNK_TIMERS || MPFREE_TIMERS
  return true;
#else
  return false;
#endif
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
inline std::size_t validate_function_delaunay_input(const function_delaunay_interface_input<index_type>& input) {
  if (input.num_points == 0) {
    if (!input.points.empty()) {
      throw std::invalid_argument("function_delaunay interface got point coordinates without input points.");
    }
    if (!input.function_values.empty()) {
      throw std::invalid_argument("function_delaunay interface got function values without input points.");
    }
    if (input.num_function_parameters > 2) {
      throw std::invalid_argument("function_delaunay interface expects one or two function parameters.");
    }
    return input.num_function_parameters;
  }

  if (input.num_point_coordinates == 0) {
    throw std::invalid_argument("function_delaunay interface expects points with positive ambient dimension.");
  }
  if (input.points.size() != input.num_points * input.num_point_coordinates) {
    throw std::invalid_argument("function_delaunay interface got malformed point cloud data.");
  }

  const std::size_t num_function_parameters = input.num_function_parameters;
  if (num_function_parameters == 0 || num_function_parameters > 2) {
    throw std::invalid_argument("function_delaunay interface expects one or two function parameters.");
  }
  if (input.function_values.size() != input.num_points * num_function_parameters) {
    throw std::invalid_argument("function_delaunay interface expects one function row per input point.");
  }
  return num_function_parameters;
}

template <typename index_type>
inline Gudhi::Simple_mdspan<const double, Gudhi::dextents<std::size_t, 2> > point_cloud_view(
    const function_delaunay_interface_input<index_type>& input) {
  return Gudhi::Simple_mdspan<const double, Gudhi::dextents<std::size_t, 2> >(
      input.points.data(), input.num_points, input.num_point_coordinates);
}

template <typename index_type>
inline Gudhi::Simple_mdspan<const double, Gudhi::dextents<std::size_t, 2> > function_values_view(
    const function_delaunay_interface_input<index_type>& input) {
  return Gudhi::Simple_mdspan<const double, Gudhi::dextents<std::size_t, 2> >(
      input.function_values.data(), input.num_points, input.num_function_parameters);
}

template <typename index_type>
inline std::size_t filtration_value_rows(const function_delaunay_interface_output<index_type>& out) {
  if (out.num_parameters == 0) {
    if (!out.filtration_values.empty()) {
      throw std::runtime_error("function_delaunay bridge got filtration values without grade dimension.");
    }
    return 0;
  }
  if (out.filtration_values.size() % out.num_parameters != 0) {
    throw std::runtime_error("function_delaunay bridge got malformed contiguous filtration values.");
  }
  return out.filtration_values.size() / out.num_parameters;
}

template <typename index_type>
inline void append_grade(function_delaunay_interface_output<index_type>& out, const Graded_matrix::Grade& grade) {
  if (grade.at.size() < 2) {
    throw std::invalid_argument("function_delaunay interface expects at least bifiltration grades.");
  }
  if (out.num_parameters == 0) {
    out.num_parameters = grade.at.size();
  } else if (grade.at.size() != out.num_parameters) {
    throw std::invalid_argument("function_delaunay interface got inconsistent grade dimensions.");
  }
  out.filtration_values.insert(out.filtration_values.end(), grade.at.begin(), grade.at.end());
}

template <typename index_type>
inline std::vector<function_delaunay::Point_with_densities> make_sorted_function_delaunay_points(
    const function_delaunay_interface_input<index_type>& input,
    std::size_t* num_function_parameters = nullptr) {
  const auto num_functions = validate_function_delaunay_input(input);
  if (num_function_parameters != nullptr) {
    *num_function_parameters = num_functions;
  }

  std::vector<function_delaunay::Point_with_densities> points;
  points.reserve(input.num_points);
  const auto point_view = point_cloud_view(input);
  const auto densities_view = function_values_view(input);
  for (std::size_t i = 0; i < input.num_points; ++i) {
    const double* coords_begin = point_view.data_handle() + i * point_view.stride(0);
    const double* densities_begin = densities_view.data_handle() + i * densities_view.stride(0);
    points.emplace_back(
        coords_begin, coords_begin + input.num_point_coordinates, densities_begin, densities_begin + num_functions);
    points.back().idx = static_cast<int>(i);
  }
  std::sort(points.begin(), points.end(), function_delaunay::Lex_sort_by_density());
  return points;
}

template <typename index_type>
inline void recover_vertex_ids(function_delaunay_interface_output<index_type>& out,
                               const std::vector<index_type>& sorted_to_original) {
  const std::size_t num_vertices = sorted_to_original.size();
  const std::size_t num_filtration_rows = filtration_value_rows(out);
  if (out.num_parameters == 0 || out.dimensions.size() < num_vertices || num_filtration_rows < num_vertices ||
      out.boundaries.size() < num_vertices) {
    throw std::runtime_error("function_delaunay bridge expected a full 0-simplex block.");
  }

  std::vector<double> vertex_filtrations(num_vertices * out.num_parameters);
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
    std::copy(out.filtration_values.begin() + static_cast<std::ptrdiff_t>(sorted_idx * out.num_parameters),
              out.filtration_values.begin() + static_cast<std::ptrdiff_t>((sorted_idx + 1) * out.num_parameters),
              vertex_filtrations.begin() + static_cast<std::ptrdiff_t>(original_idx * out.num_parameters));
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

template <typename index_type>
inline function_delaunay_interface_output<index_type> append_columns(
    const Graded_matrix& matrix,
    int out_dimension,
    index_type row_shift,
    function_delaunay_interface_output<index_type> out = function_delaunay_interface_output<index_type>()) {
  for (phat::index col_idx = 0; col_idx < matrix.get_num_cols(); ++col_idx) {
    std::vector<phat::index> local_boundary;
    matrix.get_col(col_idx, local_boundary);
    std::vector<index_type> boundary;
    boundary.reserve(local_boundary.size());
    for (const auto row_idx : local_boundary) {
      boundary.push_back(static_cast<index_type>(row_idx) + row_shift);
    }
    append_grade(out, matrix.grades[col_idx]);
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
  std::size_t total_generators = 0;
  std::size_t grade_dim = 0;
  for (std::size_t i = 0; i < num_dims; ++i) {
    const std::size_t matrix_idx = num_dims - 1 - i;
    const auto num_cols = static_cast<std::size_t>(matrices[matrix_idx].get_num_cols());
    counts[i] = static_cast<index_type>(num_cols);
    total_generators += num_cols;
    if (grade_dim == 0 && num_cols > 0) {
      grade_dim = matrices[matrix_idx].grades[0].at.size();
    }
  }
  out.boundaries.reserve(total_generators);
  out.dimensions.reserve(total_generators);
  if (grade_dim > 0) {
    out.filtration_values.reserve(total_generators * grade_dim);
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
    const std::vector<function_delaunay::Point_with_densities>& points,
    std::size_t num_function_parameters) {
  std::vector<double> lowerstar_values;
  lowerstar_values.reserve(points.size() * num_function_parameters);
  for (const auto& point : points) {
    if (point.densities.empty()) {
      throw std::runtime_error("function_delaunay simplex interface expects one function value per point.");
    }
    if (point.densities.size() != num_function_parameters) {
      throw std::runtime_error("function_delaunay simplex interface got inconsistent function dimensions.");
    }
    lowerstar_values.insert(lowerstar_values.end(), point.densities.begin(), point.densities.end());
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

inline function_delaunay_simplextree_interface_output convert_simplex_tree(Gudhi::Simplex_tree<>& simplex_tree,
                                                                           const std::vector<double>& lowerstar_values,
                                                                           std::size_t num_function_parameters) {
  function_delaunay_simplextree_interface_output out;

  if (num_function_parameters == 0) {
    throw std::invalid_argument("function_delaunay simplex interface expects at least one function parameter.");
  }
  if (lowerstar_values.size() % num_function_parameters != 0) {
    throw std::invalid_argument("function_delaunay simplex interface got malformed function values.");
  }
  const std::size_t num_vertices = lowerstar_values.size() / num_function_parameters;
  const auto lowerstar_view = Gudhi::Simple_mdspan<const double, Gudhi::dextents<std::size_t, 2> >(
      lowerstar_values.data(), num_vertices, num_function_parameters);

  const std::size_t serialized_size = simplex_tree.get_serialization_size();
  std::vector<char> serialized_simplextree(serialized_size);
  if (serialized_size > 0) {
    simplex_tree.serialize(serialized_simplextree.data(), serialized_size);
  }

  std::vector<double> default_values(1 + num_function_parameters, -std::numeric_limits<double>::infinity());
  if (serialized_size > 0) {
    out.from_std(serialized_simplextree.data(), serialized_size, 0, default_values);
  }

  for (std::size_t parameter = 0; parameter < num_function_parameters; ++parameter) {
    std::vector<double> column_values;
    column_values.reserve(num_vertices);
    for (std::size_t vertex = 0; vertex < num_vertices; ++vertex) {
      column_values.push_back(lowerstar_view(vertex, parameter));
    }
    out.fill_lowerstar(column_values, static_cast<int>(1 + parameter));
  }

  return out;
}

template <typename index_type>
inline function_delaunay_interface_output<index_type> convert_minpres(Graded_matrix& min_rep, int degree) {
  function_delaunay_interface_output<index_type> out;
  const auto num_cols = static_cast<std::size_t>(min_rep.get_num_cols());
  const std::size_t total_generators = min_rep.row_grades.size() + num_cols;
  std::size_t grade_dim = 0;
  if (!min_rep.row_grades.empty()) {
    grade_dim = min_rep.row_grades.front().at.size();
  } else if (num_cols > 0) {
    grade_dim = min_rep.grades[0].at.size();
  }
  out.boundaries.reserve(total_generators);
  out.dimensions.reserve(total_generators);
  if (grade_dim > 0) {
    out.filtration_values.reserve(total_generators * grade_dim);
  }

  for (const auto& row_grade : min_rep.row_grades) {
    append_grade(out, row_grade);
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
  std::optional<std::lock_guard<std::mutex> > global_state_lock;
  if (detail::function_delaunay_interface_needs_global_state_lock()) {
    global_state_lock.emplace(detail::function_delaunay_interface_mutex());
  }

  if (input.num_points == 0) {
    return function_delaunay_interface_output<index_type>();
  }

  auto points = detail::make_sorted_function_delaunay_points(input);
  if (degree >= 0 && input.recover_ids) {
    throw std::invalid_argument("function_delaunay recover_ids=True is only supported for full-complex outputs.");
  }
  const auto sorted_to_original = detail::sorted_to_original_vertex_ids<index_type>(points);

  (void)verbose_output;
  detail::stream_silencer silencer(false);

  std::vector<detail::Graded_matrix> matrices;
  function_delaunay::function_delaunay_with_meb<detail::Graded_matrix>(points, matrices, false);

  if (use_multi_chunk) {
    multi_chunk::compress(matrices);
  }

  if (degree >= 0) {
    if (matrices.size() < 2) {
      return function_delaunay_interface_output<index_type>();
    }
    const int matrix_idx = static_cast<int>(matrices.size()) - degree - 2;
    if (matrix_idx < 0 || matrix_idx + 1 >= static_cast<int>(matrices.size())) {
      throw std::invalid_argument("Invalid homological degree for function_delaunay minimal presentation.");
    }
    auto first_matrix = matrices[matrix_idx];
    auto second_matrix = matrices[matrix_idx + 1];
    detail::Graded_matrix min_rep;
    mpfree::compute_minimal_presentation(first_matrix, second_matrix, min_rep, false, false);
    return detail::convert_minpres<index_type>(min_rep, degree);
  }

  auto out = detail::convert_chain_complex<index_type>(matrices);
  if (input.recover_ids) {
    detail::recover_vertex_ids(out, sorted_to_original);
  }
  return out;
}

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
template <typename index_type>
function_delaunay_simplextree_interface_output function_delaunay_simplextree_interface(
    const function_delaunay_interface_input<index_type>& input,
    bool verbose_output) {
  std::optional<std::lock_guard<std::mutex> > global_state_lock;
  if (detail::function_delaunay_interface_needs_global_state_lock()) {
    global_state_lock.emplace(detail::function_delaunay_interface_mutex());
  }

  if (input.num_points == 0) {
    return function_delaunay_simplextree_interface_output();
  }

  std::size_t num_function_parameters = 0;
  auto points = detail::make_sorted_function_delaunay_points(input, &num_function_parameters);
  const auto sorted_to_original = detail::sorted_to_original_vertex_ids<int>(points);

  (void)verbose_output;
  detail::stream_silencer silencer(false);
  Gudhi::Simplex_tree<> simplex_tree;
  function_delaunay::incremental_delaunay_complex(points, simplex_tree, false);
  if (input.recover_ids) {
    simplex_tree = detail::relabel_simplex_tree_vertices(simplex_tree, sorted_to_original);
    return detail::convert_simplex_tree(simplex_tree, input.function_values, num_function_parameters);
  }
  return detail::convert_simplex_tree(
      simplex_tree, detail::lowerstar_values_from_points(points, num_function_parameters), num_function_parameters);
}

template <typename index_type>
contiguous_f64_complex function_delaunay_interface_contiguous_slicer(
    const function_delaunay_interface_input<index_type>& input,
    int degree,
    bool use_multi_chunk,
    bool verbose_output) {
  auto out = function_delaunay_interface<index_type>(input, degree, use_multi_chunk, verbose_output);
  return build_contiguous_f64_slicer_from_output<index_type>(
      out.filtration_values, out.num_parameters, out.boundaries, out.dimensions);
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

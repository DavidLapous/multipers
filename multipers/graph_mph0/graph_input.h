#pragma once

#include "graph_mph0.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace multipers::graph_mph0 {

template <typename Value>
double graph_grade_coordinate(Value value) {
  if constexpr (std::is_integral_v<Value>) {
    if (value == std::numeric_limits<Value>::max() || value == std::numeric_limits<Value>::lowest()) {
      throw std::invalid_argument("graph requires finite filtration values");
    }
    if constexpr (std::numeric_limits<Value>::digits > std::numeric_limits<double>::digits) {
      constexpr std::uint64_t max_exact = std::uint64_t{1} << std::numeric_limits<double>::digits;
      if constexpr (std::is_signed_v<Value>) {
        if (value < -static_cast<Value>(max_exact) || value > static_cast<Value>(max_exact)) {
          throw std::invalid_argument("graph requires integer filtration values exactly representable as float64");
        }
      } else if (value > static_cast<Value>(max_exact)) {
        throw std::invalid_argument("graph requires integer filtration values exactly representable as float64");
      }
    }
  }
  return static_cast<double>(value);
}

namespace input_detail {

inline void add_capacity(std::size_t& total, std::size_t count) {
  constexpr std::size_t limit = std::numeric_limits<std::uint32_t>::max();
  if (count > limit - total) {
    throw std::overflow_error("graph expanded input exceeds uint32 generator capacity");
  }
  total += count;
}

// Lexicographic order puts the smallest y first at equal x. A subsequent
// corner is minimal exactly when its y is below every retained predecessor.
inline void normalize_corners(std::vector<Grade>& grades, std::size_t begin = 0) {
  if (grades.size() - begin < 2) return;
  std::sort(grades.begin() + begin, grades.end());
  std::size_t write = begin;
  double least_y = std::numeric_limits<double>::infinity();
  for (std::size_t read = begin; read < grades.size(); ++read) {
    if (grades[read][1] >= least_y) continue;
    least_y = grades[read][1];
    if (write != read) grades[write] = grades[read];
    ++write;
  }
  grades.resize(write);
}

}  // namespace input_detail

// Corner access is synchronous and read-only. A null corner denotes an
// explicitly inactive lifetime corner, not a missing grade or an error.
// canonical_lifetimes requires lexicographic, minimal antichains in storage.
template <class DimensionAt, class BoundaryAt, class CornerCountAt, class GradeAt>
Graph build_graph_mph0_input(std::size_t num_cells,
                             std::int32_t degree,
                             DimensionAt&& dimension_at,
                             BoundaryAt&& boundary_at,
                             CornerCountAt&& corner_count_at,
                             GradeAt&& grade_at,
                             bool canonical_lifetimes = false) {
  if (degree < 0) throw std::invalid_argument("graph degree must be nonnegative");
  if (num_cells > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("graph input exceeds uint32 generator capacity");
  }
  const auto relation_degree = static_cast<std::int64_t>(degree) + 1;
  std::size_t vertex_capacity = 0;
  std::size_t edge_capacity = 0;
  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    const auto dimension = dimension_at(cell);
    if (dimension < 0) throw std::invalid_argument("graph generator dimensions must be nonnegative");
    if (dimension != degree && static_cast<std::int64_t>(dimension) != relation_degree) continue;
    auto&& boundary = boundary_at(cell);
    const std::size_t corners = corner_count_at(cell);
    if (dimension == degree) {
      if (!boundary.empty()) {
        throw std::invalid_argument(
            "Graph presentation generators must have empty boundaries (empty lower differential)");
      }
      input_detail::add_capacity(vertex_capacity, corners);
      input_detail::add_capacity(edge_capacity, corners ? corners - 1 : 0);
    } else if (!boundary.empty()) {
      if (boundary.size() != 2) {
        throw std::invalid_argument(
            "Every nonempty graph relation must contain exactly two generators (two distinct endpoints)");
      }
      const std::size_t u = boundary[0];
      const std::size_t v = boundary[1];
      if (u >= num_cells || v >= num_cells) throw std::invalid_argument("Graph relation endpoint is out of range");
      if (dimension_at(u) != degree || dimension_at(v) != degree) {
        throw std::invalid_argument("Graph relations must reference generators in the requested degree");
      }
      if (u == v) throw std::invalid_argument("Graph relations must reference two distinct generators");
      input_detail::add_capacity(edge_capacity, corners);
    }
  }

  Graph out;
  out.vertices.reserve(vertex_capacity);
  out.edges.reserve(edge_capacity);
  std::vector<std::size_t> vertex_offsets(num_cells + 1);
  const auto corner = [&](std::size_t cell, std::size_t local) -> std::optional<Grade> {
    auto value = grade_at(cell, local);
    if (value && (!std::isfinite((*value)[0]) || !std::isfinite((*value)[1]))) {
      throw std::invalid_argument("graph requires finite filtration values");
    }
    return value;
  };
  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    const auto begin = out.vertices.size();
    vertex_offsets[cell] = begin;
    if (dimension_at(cell) != degree) continue;
    const std::size_t count = corner_count_at(cell);
    for (std::size_t local = 0; local < count; ++local) {
      if (const auto value = corner(cell, local)) out.vertices.push_back(*value);
    }
    if (!canonical_lifetimes) input_detail::normalize_corners(out.vertices, begin);
    for (std::size_t vertex = begin + 1; vertex < out.vertices.size(); ++vertex) {
      const auto& previous = out.vertices[vertex - 1];
      const auto& current = out.vertices[vertex];
      out.edges.push_back({out.edges.size(),
                           vertex - 1,
                           vertex,
                           {std::max(previous[0], current[0]), std::max(previous[1], current[1])}});
    }
  }
  vertex_offsets[num_cells] = out.vertices.size();

  const auto lift = [&](std::size_t cell, const Grade& grade) {
    const auto begin = out.vertices.begin() + vertex_offsets[cell];
    const auto end = out.vertices.begin() + vertex_offsets[cell + 1];
    const auto upper =
        std::upper_bound(begin, end, grade[0], [](double x, const Grade& birth) { return x < birth[0]; });
    if (upper == begin || (*(upper - 1))[1] > grade[1]) {
      throw std::invalid_argument("Graph relation grade must dominate both endpoint grades");
    }
    return static_cast<std::size_t>((upper - 1) - out.vertices.begin());
  };
  std::vector<Grade> scratch;
  for (std::size_t cell = 0; cell < num_cells; ++cell) {
    if (static_cast<std::int64_t>(dimension_at(cell)) != relation_degree) continue;
    auto&& boundary = boundary_at(cell);
    const std::size_t count = corner_count_at(cell);
    if (boundary.empty()) {
      for (std::size_t local = 0; local < count; ++local) (void)corner(cell, local);
      continue;
    }
    const auto append = [&](const Grade& grade) {
      const auto u = lift(boundary[0], grade);
      const auto v = lift(boundary[1], grade);
      out.edges.push_back({out.edges.size(), u, v, grade});
    };
    if (canonical_lifetimes) {
      for (std::size_t local = 0; local < count; ++local) {
        if (const auto value = corner(cell, local)) append(*value);
      }
    } else {
      scratch.clear();
      scratch.reserve(count);
      for (std::size_t local = 0; local < count; ++local) {
        if (const auto value = corner(cell, local)) scratch.push_back(*value);
      }
      input_detail::normalize_corners(scratch);
      for (const auto& grade : scratch) append(grade);
    }
  }
  return out;
}

}  // namespace multipers::graph_mph0

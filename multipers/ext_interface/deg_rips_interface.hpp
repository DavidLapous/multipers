#pragma once

#include "backend_log_policy.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <streambuf>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef MULTIPERS_DISABLE_DEG_RIPS_INTERFACE
#define MULTIPERS_DISABLE_DEG_RIPS_INTERFACE 0
#endif

#if !MULTIPERS_DISABLE_DEG_RIPS_INTERFACE && __has_include(<deg_rips/basic.h>) && \
    __has_include(<deg_rips/Edge_domination_checker.h>) && __has_include(<deg_rips/Vertex_domination_matrix.h>)
#define MULTIPERS_HAS_DEG_RIPS_INTERFACE 1
#include <deg_rips/Edge_domination_checker.h>
#include <deg_rips/Vertex_domination_matrix.h>
#include <deg_rips/basic.h>
#include <tbb/parallel_for.h>
#else
#define MULTIPERS_HAS_DEG_RIPS_INTERFACE 0
#endif

namespace multipers {

inline bool deg_rips_interface_available() { return MULTIPERS_HAS_DEG_RIPS_INTERFACE; }

struct deg_rips_options {
  double min_scale = 0.0;
  double max_scale = std::numeric_limits<double>::infinity();
  bool with_vertex_domination = false;
  int whole_edge_iterations = 0;
  int edge_copy_iterations = 0;
  bool use_domination_for_whole_edge_removal = false;
  bool use_domination_for_edge_copy_removal = false;
  bool verbose_output = false;
};

struct deg_rips_stats {
  std::size_t input_vertices = 0;
  std::size_t output_vertices = 0;
  std::size_t inserted_vertices = 0;
  std::size_t inserted_edges = 0;
  std::size_t eliminated_vertices = 0;
  std::size_t edge_copy_count = 0;
};

#if MULTIPERS_HAS_DEG_RIPS_INTERFACE

namespace detail {

inline std::mutex& deg_rips_interface_mutex() {
  static std::mutex m;
  return m;
}

class deg_rips_null_streambuf : public std::streambuf {
 protected:
  int overflow(int c) override { return traits_type::not_eof(c); }

  std::streamsize xsputn(const char*, std::streamsize count) override { return count; }
};

class deg_rips_stream_silencer {
 public:
  explicit deg_rips_stream_silencer(bool silence) : silence_(silence), null_stream_(&null_buffer_) {
    if (!silence_) {
      return;
    }
    old_cout_ = std::cout.rdbuf(null_stream_.rdbuf());
    old_cerr_ = std::cerr.rdbuf(null_stream_.rdbuf());
  }

  deg_rips_stream_silencer(const deg_rips_stream_silencer&) = delete;
  deg_rips_stream_silencer& operator=(const deg_rips_stream_silencer&) = delete;

  ~deg_rips_stream_silencer() {
    if (!silence_) {
      return;
    }
    std::cout.rdbuf(old_cout_);
    std::cerr.rdbuf(old_cerr_);
  }

 private:
  bool silence_;
  deg_rips_null_streambuf null_buffer_;
  std::ostream null_stream_;
  std::streambuf* old_cout_ = nullptr;
  std::streambuf* old_cerr_ = nullptr;
};

inline double inclusive_max_scale(double max_scale) {
  if (max_scale < 0.0) {
    throw std::invalid_argument("deg_rips max_scale must be nonnegative.");
  }
  if (!std::isfinite(max_scale)) {
    return std::numeric_limits<double>::max();
  }
  return std::nextafter(max_scale, std::numeric_limits<double>::infinity());
}

inline bool close_distance_value(double a, double b) {
  const double scale = std::max({1.0, std::abs(a), std::abs(b)});
  return std::abs(a - b) <= 1e-12 * scale;
}

inline void validate_deg_rips_options(const deg_rips_options& options) {
  if (!std::isfinite(options.min_scale) || options.min_scale < 0.0) {
    throw std::invalid_argument("deg_rips min_scale must be a finite nonnegative value.");
  }
  if (std::isnan(options.max_scale) || options.max_scale < 0.0) {
    throw std::invalid_argument("deg_rips max_scale must be nonnegative.");
  }
  if (options.min_scale > inclusive_max_scale(options.max_scale)) {
    throw std::invalid_argument("deg_rips min_scale cannot exceed max_scale.");
  }
  if (options.whole_edge_iterations < 0 || options.edge_copy_iterations < 0) {
    throw std::invalid_argument("deg_rips domination iteration counts must be nonnegative.");
  }
}

inline void validate_distance_matrix(const double* distance_matrix, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const double diagonal = distance_matrix[i * n + i];
    if (!std::isfinite(diagonal) || diagonal < 0.0) {
      throw std::invalid_argument("deg_rips distance_matrix entries must be finite nonnegative values.");
    }
    if (!close_distance_value(diagonal, 0.0)) {
      throw std::invalid_argument("deg_rips distance_matrix diagonal entries must be zero.");
    }
    for (std::size_t j = i + 1; j < n; ++j) {
      const double value = distance_matrix[i * n + j];
      const double transpose = distance_matrix[j * n + i];
      if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument("deg_rips distance_matrix entries must be finite nonnegative values.");
      }
      if (!std::isfinite(transpose) || transpose < 0.0) {
        throw std::invalid_argument("deg_rips distance_matrix entries must be finite nonnegative values.");
      }
      if (!close_distance_value(value, transpose)) {
        throw std::invalid_argument("deg_rips distance_matrix must be symmetric.");
      }
    }
  }
}

struct deg_rips_distance_graph_data {
  deg_rips::Distance_graph dist;
  std::vector<std::vector<double> > critical_values_of_vertices;
};

inline deg_rips_distance_graph_data distance_graph_data_from_matrix(const double* distance_matrix,
                                                                    std::size_t n,
                                                                    double max_scale) {
  deg_rips_distance_graph_data data;
  data.dist._is_vertex.assign(n, true);
  data.dist.d.resize(n);
  data.dist._adj_lists.resize(n);
  data.critical_values_of_vertices.resize(n);

  tbb::parallel_for(std::size_t{0}, n, [&](std::size_t i) {
    auto& row = data.dist.d[i];
    row.resize(n);
    auto& adj = data.dist._adj_lists[i];
    auto& critical_values = data.critical_values_of_vertices[i];
    critical_values.reserve(n);
    for (std::size_t j = 0; j < n; ++j) {
      const double value = distance_matrix[i * n + j];
      if (value < max_scale) {
        row[j] = value;
        critical_values.push_back(value);
      } else {
        row[j] = data.dist.infty;
      }
      if (i != j && value < max_scale) {
        adj.insert(static_cast<int>(j));
      }
    }
    std::sort(critical_values.begin(), critical_values.end());
  });
  return data;
}

template <typename Filtration>
inline bool fill_flat_filtration(const std::vector<std::pair<int, double> >& critical_grades,
                                 const std::optional<std::vector<int> >& requested_degrees,
                                 std::vector<double>& flat) {
  flat.clear();
  if (critical_grades.empty()) {
    return false;
  }

  if (!requested_degrees) {
    flat.reserve(2 * critical_grades.size());
    for (const auto& [degree, radius] : critical_grades) {
      flat.push_back(radius);
      flat.push_back(-static_cast<double>(degree));
    }
    return true;
  }

  flat.reserve(2 * requested_degrees->size());
  std::size_t grade_index = 0;
  bool have_pending = false;
  int pending_degree = 0;
  double pending_radius = 0.0;

  for (int degree : *requested_degrees) {
    while (grade_index < critical_grades.size() && critical_grades[grade_index].first < degree) {
      ++grade_index;
    }
    if (grade_index == critical_grades.size()) {
      break;
    }

    const double radius = critical_grades[grade_index].second;
    if (!have_pending) {
      pending_degree = degree;
      pending_radius = radius;
      have_pending = true;
    } else if (radius == pending_radius) {
      pending_degree = degree;
    } else {
      flat.push_back(pending_radius);
      flat.push_back(-static_cast<double>(pending_degree));
      pending_degree = degree;
      pending_radius = radius;
    }
  }

  if (have_pending) {
    flat.push_back(pending_radius);
    flat.push_back(-static_cast<double>(pending_degree));
  }
  return !flat.empty();
}

template <typename Tree, typename Filtration, typename VertexRange>
inline bool emit_kcritical_simplex(Tree& tree,
                                   const VertexRange& vertices,
                                   const std::vector<std::pair<int, double> >& critical_grades,
                                   const std::optional<std::vector<int> >& requested_degrees,
                                   std::vector<double>& flat_filtration) {
  if (!fill_flat_filtration<Filtration>(critical_grades, requested_degrees, flat_filtration)) {
    return false;
  }
  const Filtration filtration(flat_filtration.begin(), flat_filtration.end(), 2);
  using BaseTree = typename Tree::Base_tree;
  auto& base_tree = static_cast<BaseTree&>(tree);
  auto result =
      base_tree.insert_simplex_and_subfaces(BaseTree::Filtration_maintenance::LOWER_EXISTING, vertices, filtration);
  return result.first != tree.null_simplex();
}

}  // namespace detail

template <typename Desc, typename SimplexTreeWrapper>
deg_rips_stats degree_rips_build_simplextree(SimplexTreeWrapper& wrapper,
                                             const double* distance_matrix,
                                             std::size_t n,
                                             const deg_rips_options& options,
                                             const std::optional<std::vector<int> >& requested_degrees) {
  if constexpr (Desc::is_kcritical && std::is_same_v<typename Desc::value_type, double> &&
                Desc::filtration_container == std::string_view("contiguous")) {
    using Filtration = typename Desc::filtration_type;

    detail::validate_deg_rips_options(options);
    detail::validate_distance_matrix(distance_matrix, n);

    const bool backend_logs_enabled = options.verbose_output || backend_log_policy::backend_log_enabled(
                                                                    backend_log_policy::backend_log_bit::deg_rips);
    std::lock_guard<std::mutex> global_state_lock(detail::deg_rips_interface_mutex());
    detail::deg_rips_stream_silencer silencer(!backend_logs_enabled);

    const double max_scale = detail::inclusive_max_scale(options.max_scale);
    auto graph_data = detail::distance_graph_data_from_matrix(distance_matrix, n, max_scale);
    auto& dist = graph_data.dist;
    auto& critical_values_of_vertices = graph_data.critical_values_of_vertices;
    int num_vertices = static_cast<int>(n);

    std::vector<double> domination_values;
    if (options.with_vertex_domination) {
      deg_rips::Vertex_domination_matrix vertex_dom(max_scale, dist);
      domination_values.assign(vertex_dom.first_domination_of.begin(), vertex_dom.first_domination_of.end());
    } else {
      domination_values.assign(static_cast<std::size_t>(num_vertices), std::numeric_limits<double>::max());
    }

    deg_rips_stats stats;
    stats.input_vertices = n;
    for (int i = 0; i < num_vertices; ++i) {
      const double domination_value = domination_values[static_cast<std::size_t>(i)];
      auto& critical_values = critical_values_of_vertices[static_cast<std::size_t>(i)];
      if (options.min_scale > domination_value) {
        ++stats.eliminated_vertices;
        dist.remove_vertex(i);
        critical_values.clear();
        continue;
      }

      std::size_t out = 0;
      for (double val : critical_values) {
        if (val <= options.min_scale) {
          critical_values[out++] = options.min_scale;
        } else if (val < domination_value) {
          critical_values[out++] = val;
        }
      }
      critical_values.resize(out);
    }

    if (stats.eliminated_vertices > 0) {
      int c = 0;
      for (int i = 0; i < num_vertices; ++i) {
        if (dist.is_vertex(i)) {
          if (c < i) {
            domination_values[static_cast<std::size_t>(c)] = domination_values[static_cast<std::size_t>(i)];
            critical_values_of_vertices[static_cast<std::size_t>(c)] =
                critical_values_of_vertices[static_cast<std::size_t>(i)];
          }
          ++c;
        }
      }
      critical_values_of_vertices.resize(static_cast<std::size_t>(c));
      domination_values.resize(static_cast<std::size_t>(c));
      num_vertices = dist.trim();
      assert(c == num_vertices);
    }
    stats.output_vertices = static_cast<std::size_t>(num_vertices);

    deg_rips::Edge_domination_checker edge_checker(dist,
                                                   max_scale,
                                                   critical_values_of_vertices,
                                                   domination_values,
                                                   options.whole_edge_iterations,
                                                   options.edge_copy_iterations,
                                                   options.use_domination_for_whole_edge_removal,
                                                   options.use_domination_for_edge_copy_removal);

    wrapper.tree.clear();
    wrapper.tree.set_num_parameters(2);
    bool inserted = false;
    std::vector<double> flat_filtration;
    std::vector<std::pair<int, double> > critical_grades;

    std::array<int, 1> vertex_simplex{};
    for (int i = 0; i < num_vertices; ++i) {
      vertex_simplex[0] = i;
      critical_grades.clear();
      critical_grades.reserve(critical_values_of_vertices[static_cast<std::size_t>(i)].size());
      deg_rips::filter_critical_grades(critical_values_of_vertices[static_cast<std::size_t>(i)], critical_grades);
      const bool did_insert = detail::emit_kcritical_simplex<decltype(wrapper.tree), Filtration>(
          wrapper.tree, vertex_simplex, critical_grades, requested_degrees, flat_filtration);
      inserted |= did_insert;
      stats.inserted_vertices += did_insert ? 1 : 0;
    }

    std::array<int, 2> edge_simplex{};
    for (const auto& curr : edge_checker.edge_copy) {
      stats.edge_copy_count += curr.second.size();
      if (curr.second.empty()) {
        continue;
      }
      int a = curr.first.first;
      int b = curr.first.second;
      if (a > b) {
        std::swap(a, b);
      }
      edge_simplex[0] = a;
      edge_simplex[1] = b;
      critical_grades.clear();
      critical_grades.reserve(curr.second.size());
      for (const auto& copy : curr.second) {
        critical_grades.emplace_back(-static_cast<int>(copy.g.x), static_cast<double>(copy.g.y));
      }
      const bool did_insert = detail::emit_kcritical_simplex<decltype(wrapper.tree), Filtration>(
          wrapper.tree, edge_simplex, critical_grades, requested_degrees, flat_filtration);
      inserted |= did_insert;
      stats.inserted_edges += did_insert ? 1 : 0;
    }
    if (inserted) {
      wrapper.tree.clear_filtration();
    }
    return stats;
  } else {
    throw std::invalid_argument("deg_rips backend expects a k-critical contiguous float64 SimplexTreeMulti target.");
  }
}

#else

template <typename Desc, typename SimplexTreeWrapper>
deg_rips_stats degree_rips_build_simplextree(SimplexTreeWrapper&,
                                             const double*,
                                             std::size_t,
                                             const deg_rips_options&,
                                             const std::optional<std::vector<int> >&) {
  throw std::runtime_error(
      "deg_rips interface is not available at compile time. Install/checkout headers and rebuild.");
}

#endif

}  // namespace multipers

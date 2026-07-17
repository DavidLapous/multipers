#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef MULTIPERS_DISABLE_GRAPHCODE_INTERFACE
#define MULTIPERS_DISABLE_GRAPHCODE_INTERFACE 0
#endif

namespace multipers {

struct graphcode_interface_input {
  std::vector<std::pair<double, double> > row_grades;
  std::vector<std::pair<double, double> > relation_grades;
  std::vector<std::vector<int> > relation_boundaries;
};

struct graphcode_vertex_output {
  double birth = 0.0;
  double death = 0.0;
  int slice_index = 0;
  double slice_value = 0.0;
};

struct graphcode_interface_output {
  std::vector<graphcode_vertex_output> vertices;
  std::vector<std::pair<int, int> > edges;
  std::vector<double> slice_values;
};

inline bool graphcode_interface_available();

graphcode_interface_output graphcode_interface(const graphcode_interface_input& input,
                                               int primary_parameter = 1,
                                               int slices = 0,
                                               bool include_infinite_bars = true,
                                               bool filter_out_disjoint_pairs = true,
                                               bool do_exhaustive_reduction = false,
                                               double relevance_threshold = -1.0,
                                               double secondary_threshold = std::numeric_limits<double>::max(),
                                               bool compressed_graphcode = true,
                                               bool double_edges = true);

}  // namespace multipers

#if !MULTIPERS_DISABLE_GRAPHCODE_INTERFACE && __has_include(<graphcode/boost_graph.h>) && \
    __has_include(<graphcode/Dynamic_boundary_matrix.h>) && __has_include(<graphcode/determine_slice_values.h>)
#define MULTIPERS_HAS_GRAPHCODE_INTERFACE 1

#include <boost/graph/adjacency_list.hpp>

#include <cstddef>
#include <iterator>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <graphcode/boost_graph.h>
#include <graphcode/determine_slice_values.h>
#include <graphcode/Dynamic_boundary_matrix.h>

#else
#define MULTIPERS_HAS_GRAPHCODE_INTERFACE 0
#endif

namespace multipers {

inline bool graphcode_interface_available() { return MULTIPERS_HAS_GRAPHCODE_INTERFACE; }

#if MULTIPERS_HAS_GRAPHCODE_INTERFACE
namespace detail {

inline std::vector<long> graphcode_normalized_column(std::vector<long> column) {
  std::sort(column.begin(), column.end());
  std::vector<long> out;
  for (std::size_t i = 0; i < column.size();) {
    const auto value = column[i];
    std::size_t count = 0;
    while (i < column.size() && column[i] == value) {
      ++count;
      ++i;
    }
    if (count % 2 == 1) {
      out.push_back(value);
    }
  }
  return out;
}

class GraphcodeLongBoundaryMatrix {
 public:
  void set_num_cols(long n) {
    columns_.assign(static_cast<std::size_t>(n), {});
    dims_.assign(static_cast<std::size_t>(n), -1);
  }

  long get_num_cols() const { return static_cast<long>(columns_.size()); }

  long get_dim(long idx) const { return dims_.at(static_cast<std::size_t>(idx)); }

  void set_dim(long idx, long dim) { dims_.at(static_cast<std::size_t>(idx)) = dim; }

  bool is_empty(long idx) const { return columns_.at(static_cast<std::size_t>(idx)).empty(); }

  void set_col(long idx, const std::vector<long>& column) {
    columns_.at(static_cast<std::size_t>(idx)) = graphcode_normalized_column(column);
  }

  void get_col(long idx, std::vector<long>& column) const { column = columns_.at(static_cast<std::size_t>(idx)); }

  long get_max_index(long idx) const {
    const auto& column = columns_.at(static_cast<std::size_t>(idx));
    return column.empty() ? -1 : column.back();
  }

  void remove_max(long idx) { columns_.at(static_cast<std::size_t>(idx)).pop_back(); }

  void add_to(long source, long target) {
    const auto& left = columns_.at(static_cast<std::size_t>(source));
    const auto& right = columns_.at(static_cast<std::size_t>(target));
    std::vector<long> out;
    std::set_symmetric_difference(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(out));
    columns_.at(static_cast<std::size_t>(target)) = std::move(out);
  }

 private:
  std::vector<std::vector<long> > columns_;
  std::vector<long> dims_;
};

inline void graphcode_validate_input(const graphcode_interface_input& input) {
  if (input.relation_grades.size() != input.relation_boundaries.size()) {
    throw std::invalid_argument("graphcode input relation grade/boundary counts differ.");
  }
  for (std::size_t col = 0; col < input.relation_boundaries.size(); ++col) {
    const auto& relation_grade = input.relation_grades[col];
    for (const auto row : input.relation_boundaries[col]) {
      if (row < 0 || static_cast<std::size_t>(row) >= input.row_grades.size()) {
        throw std::invalid_argument("graphcode input relation boundary index out of range.");
      }
      const auto& row_grade = input.row_grades[static_cast<std::size_t>(row)];
      if (relation_grade.first < row_grade.first || relation_grade.second < row_grade.second) {
        throw std::invalid_argument("graphcode input relation grades must dominate boundary row grades.");
      }
    }
  }
}

inline void graphcode_generators_relations_from_input(const graphcode_interface_input& input,
                                                      int primary_parameter,
                                                      double secondary_threshold,
                                                      std::vector<graphcode::Generator>& generators,
                                                      std::vector<graphcode::Relation>& relations) {
  std::unordered_map<long, long> row_to_generator;
  for (std::size_t row = 0; row < input.row_grades.size(); ++row) {
    const auto& grade = input.row_grades[row];
    const double values[2] = {grade.first, grade.second};
    if (values[1 - primary_parameter] <= secondary_threshold) {
      generators.emplace_back(values[0], values[1], static_cast<int>(row));
    }
  }
  std::sort(generators.begin(), generators.end(), graphcode::Sort_by_secondary(primary_parameter));
  for (std::size_t idx = 0; idx < generators.size(); ++idx) {
    generators[idx].index = static_cast<int>(idx);
    row_to_generator[generators[idx].pos_in_scc] = static_cast<long>(idx);
  }

  for (std::size_t col = 0; col < input.relation_grades.size(); ++col) {
    const auto& grade = input.relation_grades[col];
    const double values[2] = {grade.first, grade.second};
    if (values[1 - primary_parameter] > secondary_threshold) {
      continue;
    }
    std::vector<int> boundary;
    for (const auto row : input.relation_boundaries[col]) {
      const auto it = row_to_generator.find(row);
      if (it != row_to_generator.end()) {
        boundary.push_back(static_cast<int>(it->second));
      }
    }
    std::sort(boundary.begin(), boundary.end());
    relations.emplace_back(values[0], values[1], boundary);
  }
  std::sort(relations.begin(), relations.end(), graphcode::Sort_by_secondary(primary_parameter));
}

inline graphcode_interface_output output_from_graph(const graphcode::Graph& graph,
                                                    const std::vector<double>& slice_values,
                                                    bool double_edges) {
  graphcode_interface_output out;
  out.slice_values = slice_values;
  out.vertices.resize(boost::num_vertices(graph));
  typename boost::graph_traits<graphcode::Graph>::vertex_iterator vi, vi_end;
  for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
    const auto& vertex = graph[*vi];
    const auto idx = static_cast<std::size_t>(vertex.idx);
    if (idx >= out.vertices.size()) {
      throw std::runtime_error("graphcode returned a vertex index out of range.");
    }
    out.vertices[idx] = {vertex.birth, vertex.death, static_cast<int>(vertex.index_of_slice), vertex.slice_value};
  }

  out.edges.reserve(boost::num_edges(graph) * (double_edges ? 2 : 1));
  typename boost::graph_traits<graphcode::Graph>::edge_iterator ei, ei_end;
  for (boost::tie(ei, ei_end) = boost::edges(graph); ei != ei_end; ++ei) {
    const auto source = boost::source(*ei, graph);
    const auto target = boost::target(*ei, graph);
    const auto source_idx = static_cast<int>(graph[source].idx);
    const auto target_idx = static_cast<int>(graph[target].idx);
    out.edges.emplace_back(source_idx, target_idx);
    if (double_edges) {
      out.edges.emplace_back(target_idx, source_idx);
    }
  }
  return out;
}

}  // namespace detail

inline graphcode_interface_output graphcode_interface(const graphcode_interface_input& input,
                                                      int primary_parameter,
                                                      int slices,
                                                      bool include_infinite_bars,
                                                      bool filter_out_disjoint_pairs,
                                                      bool do_exhaustive_reduction,
                                                      double relevance_threshold,
                                                      double secondary_threshold,
                                                      bool compressed_graphcode,
                                                      bool double_edges) {
  if (primary_parameter != 0 && primary_parameter != 1) {
    throw std::invalid_argument("graphcode primary_parameter must be 0 or 1.");
  }
  if (slices < 0) {
    throw std::invalid_argument("graphcode slices must be nonnegative.");
  }
  if (input.row_grades.empty() && input.relation_grades.empty()) {
    return graphcode_interface_output();
  }

  detail::graphcode_validate_input(input);

  std::vector<graphcode::Generator> generators;
  std::vector<graphcode::Relation> relations;
  detail::graphcode_generators_relations_from_input(input,
                                                    primary_parameter,
                                                    secondary_threshold,
                                                    generators,
                                                    relations);

  std::vector<double> slice_values;
  graphcode::determine_slice_values(generators, relations, slices, primary_parameter, std::back_inserter(slice_values));
  if (slice_values.empty()) {
    graphcode_interface_output out;
    out.slice_values = std::move(slice_values);
    return out;
  }

  graphcode::Graph graph;
  std::vector<std::vector<std::pair<int, graphcode::Generator> > > generator_buckets(slice_values.size());
  std::vector<std::vector<std::pair<int, graphcode::Relation> > > relation_buckets(slice_values.size());
  std::unordered_map<double, int> slice_value_to_index;
  for (std::size_t i = 0; i < slice_values.size(); ++i) {
    slice_value_to_index[slice_values[i]] = static_cast<int>(i);
  }
  for (std::size_t i = 0; i < generators.size(); ++i) {
    const auto it = std::lower_bound(slice_values.begin(), slice_values.end(), generators[i].gr[primary_parameter]);
    if (it != slice_values.end()) {
      generator_buckets[slice_value_to_index[*it]].push_back(std::make_pair(static_cast<int>(i), generators[i]));
    }
  }
  for (std::size_t i = 0; i < relations.size(); ++i) {
    const auto it = std::lower_bound(slice_values.begin(), slice_values.end(), relations[i].gr[primary_parameter]);
    if (it != slice_values.end()) {
      relation_buckets[slice_value_to_index[*it]].push_back(std::make_pair(static_cast<int>(i), relations[i]));
    }
  }

  graphcode::Vertex_relevance_checker vertex_relevance_check(generators,
                                                             relations,
                                                             primary_parameter,
                                                             relevance_threshold,
                                                             include_infinite_bars);
  graphcode::Dynamic_boundary_matrix<detail::GraphcodeLongBoundaryMatrix, graphcode::Graph> dynamic_matrix(
      graph,
      generators,
      relations,
      vertex_relevance_check,
      primary_parameter,
      filter_out_disjoint_pairs,
      include_infinite_bars,
      do_exhaustive_reduction,
      compressed_graphcode);
  for (std::size_t i = 0; i < slice_values.size(); ++i) {
    dynamic_matrix.new_batch(generator_buckets[i], relation_buckets[i], static_cast<int>(i), slice_values[i]);
  }
  if (compressed_graphcode) {
    dynamic_matrix.last_batch(static_cast<int>(slice_values.size() - 1));
  }
  return detail::output_from_graph(graph, slice_values, double_edges);
}

#else

inline graphcode_interface_output graphcode_interface(const graphcode_interface_input&,
                                                      int,
                                                      int,
                                                      bool,
                                                      bool,
                                                      bool,
                                                      double,
                                                      double,
                                                      bool,
                                                      bool) {
  throw std::runtime_error("graphcode interface is not available at compile time. Install/checkout headers and rebuild.");
}

#endif

}  // namespace multipers

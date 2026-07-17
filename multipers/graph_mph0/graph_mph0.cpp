#include "graph_mph0/graph_mph0.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "graph_mph0/dynamic_merge_forest.h"

namespace multipers::graph_mph0 {
namespace {

bool equal(const Grade& a, const Grade& b) { return a[0] == b[0] && a[1] == b[1]; }

bool strictly_above(const Grade& a, const Grade& b) { return !equal(a, b) && b[0] <= a[0] && b[1] <= a[1]; }

std::vector<std::vector<std::size_t>> adjacency(const Graph& graph) {
  std::vector<std::vector<std::size_t>> out(graph.vertices.size());
  for (std::size_t i = 0; i < graph.edges.size(); ++i) {
    out[graph.edges[i].u].push_back(i);
    if (graph.edges[i].u != graph.edges[i].v) out[graph.edges[i].v].push_back(i);
  }
  return out;
}

Graph compact(const Graph& graph, std::vector<std::size_t> representative, const std::vector<bool>& removed_edges) {
  for (std::size_t i = 0; i < representative.size(); ++i) {
    std::size_t root = i;
    while (representative[root] != root) root = representative[root];
    for (std::size_t current = i; representative[current] != current;) {
      const std::size_t next = representative[current];
      representative[current] = root;
      current = next;
    }
  }

  Graph out;
  std::vector<std::size_t> new_id(graph.vertices.size(), static_cast<std::size_t>(-1));
  for (std::size_t i = 0; i < graph.vertices.size(); ++i) {
    if (representative[i] == i) {
      new_id[i] = out.vertices.size();
      out.vertices.push_back(graph.vertices[i]);
    }
  }
  for (std::size_t i = 0; i < graph.vertices.size(); ++i) new_id[i] = new_id[representative[i]];

  out.edges.reserve(graph.edges.size());
  for (std::size_t i = 0; i < graph.edges.size(); ++i) {
    if (removed_edges[i]) continue;
    const auto& edge = graph.edges[i];
    out.edges.push_back({out.edges.size(), new_id[edge.u], new_id[edge.v], edge.grade});
  }
  return out;
}

Graph collapse_local_edges(const Graph& graph) {
  const auto adj = adjacency(graph);
  std::vector<bool> visited(graph.vertices.size(), false);
  std::vector<bool> removed(graph.edges.size(), false);
  std::vector<std::size_t> representative(graph.vertices.size());
  std::iota(representative.begin(), representative.end(), 0);

  struct Step {
    std::size_t vertex;
    std::size_t edge;
  };

  const std::size_t no_edge = static_cast<std::size_t>(-1);
  for (std::size_t start = 0; start < graph.vertices.size(); ++start) {
    if (visited[start]) continue;
    std::vector<Step> stack{{start, no_edge}};
    while (!stack.empty()) {
      const Step step = stack.back();
      stack.pop_back();
      if (visited[step.vertex]) continue;
      visited[step.vertex] = true;
      representative[step.vertex] = start;
      if (step.edge != no_edge) removed[step.edge] = true;
      for (std::size_t edge_id : adj[step.vertex]) {
        if (removed[edge_id]) continue;
        const auto& edge = graph.edges[edge_id];
        const std::size_t other = edge.u == step.vertex ? edge.v : edge.u;
        if (!visited[other] && equal(edge.grade, graph.vertices[step.vertex]) &&
            equal(edge.grade, graph.vertices[other])) {
          stack.push_back({other, edge_id});
        }
      }
    }
  }
  return compact(graph, std::move(representative), removed);
}

Graph collapse_to_vertex_minimal(const Graph& input) {
  Graph graph = collapse_local_edges(input);
  const auto adj = adjacency(graph);
  std::vector<bool> visited(graph.vertices.size(), false);
  std::vector<bool> removed(graph.edges.size(), false);
  std::vector<std::size_t> representative(graph.vertices.size());
  std::iota(representative.begin(), representative.end(), 0);

  struct Step {
    std::size_t vertex;
    std::size_t edge;
  };

  const std::size_t no_edge = static_cast<std::size_t>(-1);
  for (std::size_t start = 0; start < graph.vertices.size(); ++start) {
    if (visited[start]) continue;
    bool minimal = true;
    for (std::size_t edge_id : adj[start]) {
      const auto& edge = graph.edges[edge_id];
      const std::size_t other = edge.u == start ? edge.v : edge.u;
      if (equal(edge.grade, graph.vertices[start]) && strictly_above(graph.vertices[start], graph.vertices[other])) {
        minimal = false;
        break;
      }
    }
    if (!minimal) continue;

    std::vector<Step> stack{{start, no_edge}};
    while (!stack.empty()) {
      const Step step = stack.back();
      stack.pop_back();
      if (visited[step.vertex]) continue;
      visited[step.vertex] = true;
      representative[step.vertex] = start;
      if (step.edge != no_edge) removed[step.edge] = true;
      for (std::size_t edge_id : adj[step.vertex]) {
        if (removed[edge_id]) continue;
        const auto& edge = graph.edges[edge_id];
        const std::size_t other = edge.u == step.vertex ? edge.v : edge.u;
        if (!visited[other] && equal(edge.grade, graph.vertices[other]) &&
            strictly_above(graph.vertices[other], graph.vertices[step.vertex])) {
          stack.push_back({other, edge_id});
        }
      }
    }
  }
  return compact(graph, std::move(representative), removed);
}

void validate(const Graph& graph) {
  std::unordered_set<std::size_t> edge_ids;
  for (const Grade& grade : graph.vertices) {
    if (!std::isfinite(grade[0]) || !std::isfinite(grade[1])) {
      throw std::invalid_argument("graph requires finite vertex grades");
    }
  }
  for (const auto& edge : graph.edges) {
    if (edge.u >= graph.vertices.size() || edge.v >= graph.vertices.size()) {
      throw std::invalid_argument("graph edge endpoint is out of range");
    }
    if (!edge_ids.insert(edge.id).second) throw std::invalid_argument("graph edge ids must be unique");
    if (!std::isfinite(edge.grade[0]) || !std::isfinite(edge.grade[1])) {
      throw std::invalid_argument("graph requires finite edge grades");
    }
    for (int parameter = 0; parameter < 2; ++parameter) {
      if (edge.grade[parameter] < graph.vertices[edge.u][parameter] ||
          edge.grade[parameter] < graph.vertices[edge.v][parameter]) {
        throw std::invalid_argument("graph edge grade precedes endpoint grade");
      }
    }
  }
}

}  // namespace

Result compute(const Graph& input, Compute_options options) {
  validate(input);
  const Graph graph = collapse_to_vertex_minimal(input);
  Result out;

  using Event = std::variant<std::size_t, const Edge*>;
  std::vector<Event> events;
  events.reserve(graph.vertices.size() + graph.edges.size());
  for (std::size_t vertex = 0; vertex < graph.vertices.size(); ++vertex) events.emplace_back(vertex);
  for (const auto& edge : graph.edges) events.emplace_back(&edge);
  auto event_grade = [&](const Event& event) -> const Grade& {
    if (const auto* vertex = std::get_if<std::size_t>(&event)) return graph.vertices[*vertex];
    return std::get<const Edge*>(event)->grade;
  };
  std::sort(events.begin(), events.end(), [&](const Event& left, const Event& right) {
    const Grade& a = event_grade(left);
    const Grade& b = event_grade(right);
    if (a[0] != b[0]) return a[0] < b[0];
    if (a[1] != b[1]) return a[1] < b[1];
    if (left.index() != right.index()) return left.index() < right.index();
    if (const auto* vertex = std::get_if<std::size_t>(&left)) return *vertex < std::get<std::size_t>(right);
    return std::get<const Edge*>(left)->id < std::get<const Edge*>(right)->id;
  });

  Dynamic_merge_forest<double> forest(graph.vertices.size());
  std::vector<std::size_t> row(graph.vertices.size(), static_cast<std::size_t>(-1));
  for (const Event& event : events) {
    if (const auto* vertex = std::get_if<std::size_t>(&event)) {
      row[*vertex] = out.beta_0.size();
      out.beta_0.push_back(graph.vertices[*vertex]);
      continue;
    }

    const Edge& edge = **std::get_if<const Edge*>(&event);
    if (edge.u == edge.v) {
      if (options.h1_betti) out.beta_0_h1.push_back(edge.grade);
      continue;
    }
    const double merge = forest.time_of_merge(edge.u, edge.v);
    if (merge <= edge.grade[1]) {
      if (options.h1_betti) out.beta_0_h1.push_back(edge.grade);
      continue;
    }
    std::vector<std::size_t> syzygy;
    if (options.full_resolution && merge != std::numeric_limits<double>::max()) {
      syzygy = forest.path_edges(edge.u, edge.v);
      syzygy.push_back(out.beta_1.size());
      std::sort(syzygy.begin(), syzygy.end());
    }
    forest.merge_at_time(edge.u, edge.v, edge.grade[1]);
    out.beta_1.push_back(edge.grade);
    out.relations.push_back({row[edge.u], row[edge.v]});
    if (merge != std::numeric_limits<double>::max()) {
      const Grade witness{edge.grade[0], merge};
      if (options.full_resolution) {
        out.beta_2.push_back(witness);
        out.syzygies.push_back(std::move(syzygy));
      }
      if (options.h1_betti) out.beta_0_h1.push_back(witness);
    }
  }
  return out;
}

}  // namespace multipers::graph_mph0

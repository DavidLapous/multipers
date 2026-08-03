#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "graph_mph0/link_cut_forest.h"

namespace multipers::graph_mph0 {

// Monotone dendrogram interface used by graph's lexicographic sweep.
template <class Weight>
class Dynamic_merge_forest {
 public:
  using Edge = typename Link_cut_forest<Weight>::Edge;

  explicit Dynamic_merge_forest(std::size_t vertices = 0)
      : forest_(vertices), component_parent_(vertices), component_rank_(vertices) {
    std::iota(component_parent_.begin(), component_parent_.end(), 0);
  }

  std::optional<Edge> merge_bottleneck(std::size_t u, std::size_t v) {
    check_vertex(u);
    check_vertex(v);
    if (u == v || component_root(u) != component_root(v)) return std::nullopt;
    return forest_.path_bottleneck_assuming_connected(u, v);
  }

  std::vector<std::size_t> path_edges(std::size_t u, std::size_t v) { return forest_.path_edges(u, v); }

  // `outgoing` must be the immediately preceding merge_bottleneck(u, v) result.
  void merge_at_time(std::size_t u, std::size_t v, Weight time, const std::optional<Edge>& outgoing) {
    const std::size_t incoming = next_edge_++;
    if (!outgoing) {
      forest_.link_assuming_disconnected(incoming, u, v, time);
      component_union(u, v);
      return;
    }
    if (std::tie(time, incoming) < std::tie(outgoing->weight, outgoing->id)) {
      forest_.cut(outgoing->id);
      forest_.link_assuming_disconnected(incoming, u, v, time);
    }
  }

 private:
  void check_vertex(std::size_t vertex) const {
    if (vertex >= component_parent_.size()) throw std::out_of_range("link-cut vertex is out of range");
  }

  std::size_t component_root(std::size_t vertex) {
    std::size_t root = vertex;
    while (component_parent_[root] != root) root = component_parent_[root];
    while (component_parent_[vertex] != vertex) {
      const std::size_t next = component_parent_[vertex];
      component_parent_[vertex] = root;
      vertex = next;
    }
    return root;
  }

  void component_union(std::size_t u, std::size_t v) {
    u = component_root(u);
    v = component_root(v);
    if (u == v) return;
    if (component_rank_[u] < component_rank_[v]) std::swap(u, v);
    component_parent_[v] = u;
    if (component_rank_[u] == component_rank_[v]) ++component_rank_[u];
  }

  Link_cut_forest<Weight> forest_;
  // This is valid only for this monotone, serialized merge protocol.
  std::vector<std::size_t> component_parent_;
  std::vector<std::uint8_t> component_rank_;
  std::size_t next_edge_ = 0;
};

}  // namespace multipers::graph_mph0

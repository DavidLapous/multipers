#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "graph_mph0/graph_mph0.h"
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
    return forest_.path_bottleneck(u, v);
  }

  std::vector<std::size_t> path_edges(std::size_t u, std::size_t v) { return forest_.path_edges(u, v); }

  // A caller-supplied witness is checked against the current path before mutation.
  void merge_at_time(std::size_t u, std::size_t v, Weight time, const std::optional<Edge>& outgoing) {
    const auto current = merge_bottleneck(u, v);
    if (u == v) throw std::invalid_argument("link-cut link would create a cycle");
    bool witness_matches = outgoing.has_value() == current.has_value();
    if (witness_matches && outgoing) {
      witness_matches = outgoing->id == current->id && outgoing->u == current->u && outgoing->v == current->v &&
                        !(outgoing->weight < current->weight) && !(current->weight < outgoing->weight);
      if constexpr (std::is_floating_point_v<Weight>) {
        witness_matches = witness_matches && !std::isnan(outgoing->weight) && !std::isnan(current->weight);
      }
    }
    if (!witness_matches) throw std::invalid_argument("merge-forest outgoing witness does not match the current path");
    if (next_edge_ == std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("merge-forest edge ids are exhausted");
    }
    merge_at_time_trusted(u, v, time, current);
  }

 private:
  friend Result compute(const Graph&, Compute_options);

  // compute() validates and owns its graph; these endpoints are distinct.
  std::optional<Edge> merge_bottleneck_trusted(std::size_t u, std::size_t v) {
    if (component_root(u) != component_root(v)) return std::nullopt;
    return forest_.path_bottleneck_trusted(u, v);
  }

  // A fresh bottleneck proves that this distinct pair is connected.
  std::vector<std::size_t> path_edges_trusted(std::size_t u, std::size_t v) { return forest_.path_edges_trusted(u, v); }

  // Admission is either the checked public witness or compute()'s immediate
  // query -> read-only path enumeration -> merge, with no intervening mutation.
  // Public admission rejects ID exhaustion; compute's finite event count bounds IDs.
  void merge_at_time_trusted(std::size_t u, std::size_t v, Weight time, const std::optional<Edge>& outgoing) {
    const std::size_t incoming = next_edge_;
    if (!outgoing) {
      forest_.link_disconnected_trusted(incoming, u, v, time);
      component_union(u, v);
    } else if (std::tie(time, incoming) < std::tie(outgoing->weight, outgoing->id)) {
      forest_.cut_active_trusted(outgoing->id);
      try {
        forest_.link_disconnected_trusted(incoming, u, v, time);
      } catch (...) {
        forest_.link_disconnected_trusted(outgoing->id, outgoing->u, outgoing->v, outgoing->weight);
        throw;
      }
    }
    ++next_edge_;
  }

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
  // Initially singleton components match the edgeless forest. A disconnected
  // link joins both partitions; replacing a fresh path edge reconnects exactly
  // the same component and leaves this monotone partition unchanged.
  std::vector<std::size_t> component_parent_;
  std::vector<std::uint8_t> component_rank_;
  std::size_t next_edge_ = 0;
};

}  // namespace multipers::graph_mph0

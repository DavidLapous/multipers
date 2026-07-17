#pragma once

#include <cstddef>
#include <limits>
#include <tuple>
#include <vector>

#include "graph_mph0/link_cut_forest.h"

namespace multipers::graph_mph0 {

// Monotone dendrogram interface used by graph's lexicographic sweep.
template <class Weight>
class Dynamic_merge_forest {
 public:
  explicit Dynamic_merge_forest(std::size_t vertices = 0) : forest_(vertices) {}

  Weight time_of_merge(std::size_t u, std::size_t v) {
    const auto edge = forest_.path_bottleneck(u, v);
    return edge ? edge->weight : std::numeric_limits<Weight>::max();
  }

  std::vector<std::size_t> path_edges(std::size_t u, std::size_t v) { return forest_.path_edges(u, v); }

  void merge_at_time(std::size_t u, std::size_t v, Weight time) {
    const std::size_t incoming = next_edge_++;
    const auto outgoing = forest_.path_bottleneck(u, v);
    if (!outgoing) {
      forest_.link(incoming, u, v, time);
      return;
    }
    if (std::tie(time, incoming) < std::tie(outgoing->weight, outgoing->id)) {
      forest_.cut(outgoing->id);
      forest_.link(incoming, u, v, time);
    }
  }

 private:
  Link_cut_forest<Weight> forest_;
  std::size_t next_edge_ = 0;
};

}  // namespace multipers::graph_mph0

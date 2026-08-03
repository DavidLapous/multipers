#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <gudhi/persistence_interval.h>

#include "link_cut_forest.h"

namespace multipers::graph_mph0 {

template <typename Dimensions, typename Boundaries>
void validate_graph_shape(const Dimensions& dimensions, const Boundaries& boundaries) {
  if (boundaries.size() != dimensions.size()) throw std::invalid_argument("Graph backend malformed boundaries");
  if (dimensions.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("Graph backend supports at most uint32 generator indices");
  }
  if (dimensions.empty()) return;

  const auto base_degree = *std::min_element(dimensions.begin(), dimensions.end());
  if (base_degree < 0) throw std::invalid_argument("Graph backend dimensions must be nonnegative");
  for (std::size_t generator = 0; generator < dimensions.size(); ++generator) {
    if (dimensions[generator] == base_degree) {
      if (!boundaries[generator].empty()) throw std::invalid_argument("Graph vertex boundary must be empty");
      continue;
    }
    if (dimensions[generator] != base_degree + 1) {
      throw std::invalid_argument("Graph backend requires two adjacent generator dimensions");
    }
    if (boundaries[generator].empty()) continue;
    if (boundaries[generator].size() != 2) {
      throw std::invalid_argument("Graph nonempty relation boundary must contain exactly two generators");
    }
    if (boundaries[generator][0] == boundaries[generator][1]) {
      throw std::invalid_argument("Graph relation endpoints must be distinct");
    }
    for (const auto vertex : boundaries[generator]) {
      if (vertex >= generator || dimensions[vertex] != base_degree) {
        throw std::invalid_argument("Graph edge endpoint is invalid");
      }
    }
  }
}

// Graph-specialized H0/H1 persistence backend. The vineyard specialization
// replays matrix-vineyard swaps and preserves barcode slot identities.
template <class Structure, bool Vineyard>
class Slicer_backend {
 public:
  using Dimension = int;
  using Index = std::uint32_t;
  using Bar = Gudhi::persistence_matrix::Persistence_interval<Dimension, Index>;
  using Cycle = std::vector<Index>;
  using Map = std::vector<Index>;
  template <class Complex>
  using As_type = Slicer_backend<Complex, Vineyard>;

  static constexpr auto nullDeath = Bar::inf;
  static constexpr bool is_vine = Vineyard;
  static constexpr bool is_graph = true;
  static constexpr bool has_rep_cycles = false;

  Slicer_backend() = default;

  template <class Complex, class Filtration_range>
  Slicer_backend(const Complex& complex, const Filtration_range& values, bool ignore_inf = false) {
    initialize(complex, values, ignore_inf);
  }

  template <class Complex, class Filtration_range>
  void initialize(const Complex& complex, const Filtration_range& values, bool = false) {
    const auto& dimensions = complex.get_dimensions();
    const auto& boundaries = complex.get_boundaries();
    validate_graph_shape(dimensions, boundaries);

    Slicer_backend replacement;
    replacement.dimensions_.assign(dimensions.begin(), dimensions.end());
    replacement.boundaries_.resize(replacement.dimensions_.size());
    for (std::size_t generator = 0; generator < boundaries.size(); ++generator) {
      const auto& boundary = boundaries[generator];
      if (!boundary.empty()) {
        replacement.boundaries_[generator].endpoints = {static_cast<Index>(boundary[0]),
                                                        static_cast<Index>(boundary[1])};
      }
    }
    if (!replacement.dimensions_.empty()) {
      replacement.base_degree_ = *std::min_element(replacement.dimensions_.begin(), replacement.dimensions_.end());
    }
    replacement.update(values);
    *this = std::move(replacement);
  }

  template <class Filtration_range>
  void update(const Filtration_range& values, bool = false) {
    if (values.size() != dimensions_.size()) throw std::invalid_argument("Graph backend filtration size mismatch");
    if constexpr (!Vineyard) initialized_ = false;
    if (!initialized_) {
      order_.resize(dimensions_.size());
      std::iota(order_.begin(), order_.end(), 0);
      std::sort(order_.begin(), order_.end(), [&](Index a, Index b) {
        if (dimensions_[a] != dimensions_[b]) return dimensions_[a] < dimensions_[b];
        if (values[a] < values[b]) return true;
        if (values[b] < values[a]) return false;
        return a < b;
      });
      bars_ = compute_bars();
      std::sort(bars_.begin(), bars_.end(), bar_less);
      if constexpr (Vineyard) initialize_vineyard_state();
      initialized_ = true;
      return;
    }

    // Match Vineyard_base::update: insertion-sort each dimension block with
    // adjacent swaps. Each swap transports current barcode slots along vines.
    for (Index i = 1; i < order_.size(); ++i) {
      Index current = i;
      while (current > 0 && dimensions_[order_[current]] == dimensions_[order_[current - 1]] &&
             values[order_[current]] < values[order_[current - 1]]) {
        vine_swap(current - 1);
        --current;
      }
    }
  }

  [[nodiscard]] bool is_initialized() const { return initialized_; }

  const Map& get_current_order() const { return order_; }

  const std::vector<Bar>& get_barcode() { return bars_; }

  std::vector<Cycle> get_all_representative_cycles(bool = true, Dimension = -1) {
    throw std::logic_error("Graph backend does not provide representative cycles");
  }

  Cycle get_representative_cycle(Index, bool = true) {
    throw std::logic_error("Graph backend does not provide representative cycles");
  }

  friend std::ostream& operator<<(std::ostream& stream, const Slicer_backend& backend) {
    return stream << "graph backend: " << backend.dimensions_.size() << " generators, " << backend.bars_.size()
                  << " bars";
  }

 private:
  struct Boundary {
    static constexpr Index empty_endpoint = std::numeric_limits<Index>::max();

    std::array<Index, 2> endpoints = {empty_endpoint, empty_endpoint};

    [[nodiscard]] bool empty() const { return endpoints[0] == empty_endpoint; }

    [[nodiscard]] const Index& operator[](std::size_t index) const { return endpoints[index]; }

    [[nodiscard]] auto begin() const { return endpoints.begin(); }

    [[nodiscard]] auto end() const { return empty() ? endpoints.begin() : endpoints.end(); }
  };

  static bool bar_less(const Bar& a, const Bar& b) {
    return std::tie(a.dim, a.birth, a.death) < std::tie(b.dim, b.birth, b.death);
  }

  std::vector<Bar> compute_bars() const {
    std::vector<Bar> bars;
    const Index count = dimensions_.size();
    std::vector<Index> vertex_number(count, count);
    std::vector<Index> order_position(count);
    Index number_of_vertices = 0;
    for (Index position = 0; position < count; ++position) {
      const Index generator = order_[position];
      order_position[generator] = position;
      if (dimensions_[generator] == base_degree_) vertex_number[generator] = number_of_vertices++;
    }

    std::vector<Index> parent(number_of_vertices);
    std::vector<Index> component_birth(number_of_vertices, count);
    std::iota(parent.begin(), parent.end(), 0);
    for (Index generator = 0; generator < count; ++generator) {
      if (dimensions_[generator] == base_degree_) component_birth[vertex_number[generator]] = generator;
    }
    auto root = [&](Index vertex) {
      Index current = vertex;
      while (parent[current] != current) current = parent[current];
      while (parent[vertex] != vertex) {
        const Index next = parent[vertex];
        parent[vertex] = current;
        vertex = next;
      }
      return current;
    };

    for (Index generator : order_) {
      if (dimensions_[generator] == base_degree_) continue;

      if (boundaries_[generator].empty()) {
        bars.emplace_back(generator, Bar::inf, base_degree_ + 1);
        continue;
      }
      const Index u = vertex_number[boundaries_[generator][0]];
      const Index v = vertex_number[boundaries_[generator][1]];
      Index ru = root(u);
      Index rv = root(v);
      if (ru == rv) {
        bars.emplace_back(generator, Bar::inf, base_degree_ + 1);
        continue;
      }

      if (order_position[component_birth[rv]] < order_position[component_birth[ru]]) std::swap(ru, rv);
      bars.emplace_back(component_birth[rv], generator, base_degree_);
      parent[rv] = ru;
    }

    for (Index vertex = 0; vertex < number_of_vertices; ++vertex) {
      if (root(vertex) == vertex) bars.emplace_back(component_birth[vertex], Bar::inf, base_degree_);
    }
    return bars;
  }

  void initialize_vineyard_state() {
    const std::size_t no_slot = std::numeric_limits<std::size_t>::max();
    const Index count = dimensions_.size();
    order_position_.resize(count);
    vertex_number_.assign(count, count);
    positive_edges_.assign(count, false);
    death_by_birth_.assign(count, Bar::inf);
    birth_by_death_.assign(count, Bar::inf);
    h0_slot_by_birth_.assign(count, no_slot);
    h1_slot_by_edge_.assign(count, no_slot);

    number_of_vertices_ = 0;
    for (Index position = 0; position < count; ++position) {
      const Index generator = order_[position];
      order_position_[generator] = position;
      if (dimensions_[generator] == base_degree_) vertex_number_[generator] = number_of_vertices_++;
    }

    for (std::size_t slot = 0; slot < bars_.size(); ++slot) {
      const Bar& bar = bars_[slot];
      if (bar.dim == base_degree_) {
        death_by_birth_[bar.birth] = bar.death;
        if (bar.death != Bar::inf) birth_by_death_[bar.death] = bar.birth;
        h0_slot_by_birth_[bar.birth] = slot;
      } else {
        positive_edges_[bar.birth] = true;
        h1_slot_by_edge_[bar.birth] = slot;
      }
    }

    tree_forest_.reset(number_of_vertices_);
    for (Index generator = 0; generator < count; ++generator) {
      if (dimensions_[generator] == base_degree_ + 1 && !boundaries_[generator].empty() &&
          !positive_edges_[generator]) {
        tree_forest_.link(generator,
                          vertex_number_[boundaries_[generator][0]],
                          vertex_number_[boundaries_[generator][1]],
                          order_position_[generator]);
      }
    }

    // The graph's connected components do not depend on filtration order.
    // Cache them once so dynamic-tree path queries can skip connectivity work.
    component_id_.resize(number_of_vertices_);
    std::iota(component_id_.begin(), component_id_.end(), 0);
    auto component_root = [&](Index vertex) {
      while (component_id_[vertex] != vertex) {
        component_id_[vertex] = component_id_[component_id_[vertex]];
        vertex = component_id_[vertex];
      }
      return vertex;
    };
    for (Index generator = 0; generator < count; ++generator) {
      if (dimensions_[generator] != base_degree_ + 1 || boundaries_[generator].empty()) continue;
      const Index first = component_root(vertex_number_[boundaries_[generator][0]]);
      const Index second = component_root(vertex_number_[boundaries_[generator][1]]);
      if (first != second) component_id_[second] = first;
    }
    for (Index vertex = 0; vertex < number_of_vertices_; ++vertex) {
      component_id_[vertex] = component_root(vertex);
    }
  }

  [[nodiscard]] bool is_tree_edge(Index edge) const { return !boundaries_[edge].empty() && !positive_edges_[edge]; }

  void swap_order(Index position, Index first, Index second) {
    std::swap(order_[position], order_[position + 1]);
    order_position_[first] = position + 1;
    order_position_[second] = position;
  }

  std::optional<typename Link_cut_forest<Index>::Edge> path_bottleneck(Index first_vertex, Index second_vertex) {
    const Index first = vertex_number_[first_vertex];
    const Index second = vertex_number_[second_vertex];
    if (first == second || component_id_[first] != component_id_[second]) return std::nullopt;
    return tree_forest_.path_bottleneck_assuming_connected(first, second);
  }

  void swap_vertices(Index position, Index first, Index second) {
    const auto merge = path_bottleneck(first, second);
    const Index first_death = death_by_birth_[first];
    const Index second_death = death_by_birth_[second];
    const bool exchange =
        merge && second_death == merge->id && (first_death == Bar::inf || order_position_[first_death] > merge->weight);
    if (exchange) {
      const std::size_t first_slot = h0_slot_by_birth_[first];
      const std::size_t second_slot = h0_slot_by_birth_[second];
      if (first_slot == std::numeric_limits<std::size_t>::max() ||
          second_slot == std::numeric_limits<std::size_t>::max()) {
        throw std::logic_error("Missing Graph vineyard H0 slot");
      }

      death_by_birth_[first] = second_death;
      death_by_birth_[second] = first_death;
      birth_by_death_[second_death] = first;
      if (first_death != Bar::inf) birth_by_death_[first_death] = second;
      bars_[first_slot] = Bar(second, first_death, base_degree_);
      bars_[second_slot] = Bar(first, second_death, base_degree_);
      std::swap(h0_slot_by_birth_[first], h0_slot_by_birth_[second]);
    }
    swap_order(position, first, second);
  }

  bool path_reaches_before(Index source, Index target, Index edge_position) {
    if (source == target) return true;
    const auto maximum = path_bottleneck(source, target);
    return maximum && maximum->weight < edge_position;
  }

  void swap_tree_edges(Index position, Index first, Index second) {
    const Index first_birth = birth_by_death_[first];
    const Index second_birth = birth_by_death_[second];
    if (first_birth == Bar::inf || second_birth == Bar::inf) {
      throw std::logic_error("Missing Graph vineyard H0 death inverse");
    }

    bool attaches_to_first_birth = false;
    for (Index endpoint : boundaries_[second]) {
      if (path_reaches_before(first_birth, endpoint, position)) {
        attaches_to_first_birth = true;
        break;
      }
    }
    bool exchange = false;
    if (attaches_to_first_birth) {
      const auto merge = path_bottleneck(first_birth, second_birth);
      if (!merge) throw std::logic_error("Graph vineyard tree births are disconnected");
      exchange =
          merge->id == first || (merge->id == second && order_position_[first_birth] > order_position_[second_birth]);
    }

    if (exchange) {
      death_by_birth_[first_birth] = second;
      death_by_birth_[second_birth] = first;
      birth_by_death_[first] = second_birth;
      birth_by_death_[second] = first_birth;
      bars_[h0_slot_by_birth_[first_birth]] = Bar(first_birth, second, base_degree_);
      bars_[h0_slot_by_birth_[second_birth]] = Bar(second_birth, first, base_degree_);
    }
    tree_forest_.update_weight(first, position + 1);
    tree_forest_.update_weight(second, position);
    swap_order(position, first, second);
  }

  bool positive_path_uses_tree_edge(Index positive_edge, Index tree_edge) {
    const auto maximum = path_bottleneck(boundaries_[positive_edge][0], boundaries_[positive_edge][1]);
    return maximum && maximum->id == tree_edge;
  }

  void exchange_tree_and_positive(Index position, Index tree_edge, Index positive_edge) {
    const Index birth = birth_by_death_[tree_edge];
    const std::size_t h0_slot = birth == Bar::inf ? std::numeric_limits<std::size_t>::max() : h0_slot_by_birth_[birth];
    const std::size_t h1_slot = h1_slot_by_edge_[positive_edge];
    if (h0_slot == std::numeric_limits<std::size_t>::max()) {
      throw std::logic_error("Missing Graph vineyard H0 slot");
    }
    if (h1_slot == std::numeric_limits<std::size_t>::max()) {
      throw std::logic_error("Missing Graph vineyard H1 slot");
    }

    tree_forest_.cut(tree_edge);
    try {
      tree_forest_.link(positive_edge,
                        vertex_number_[boundaries_[positive_edge][0]],
                        vertex_number_[boundaries_[positive_edge][1]],
                        position);
    } catch (...) {
      tree_forest_.link(
          tree_edge, vertex_number_[boundaries_[tree_edge][0]], vertex_number_[boundaries_[tree_edge][1]], position);
      throw;
    }

    death_by_birth_[birth] = positive_edge;
    birth_by_death_[tree_edge] = Bar::inf;
    birth_by_death_[positive_edge] = birth;
    bars_[h0_slot] = Bar(birth, positive_edge, base_degree_);

    positive_edges_[tree_edge] = true;
    positive_edges_[positive_edge] = false;
    h1_slot_by_edge_[positive_edge] = std::numeric_limits<std::size_t>::max();
    h1_slot_by_edge_[tree_edge] = h1_slot;
    bars_[h1_slot] = Bar(tree_edge, Bar::inf, base_degree_ + 1);
    swap_order(position, tree_edge, positive_edge);
  }

  void swap_edges(Index position, Index first, Index second) {
    const bool first_tree = is_tree_edge(first);
    const bool second_tree = is_tree_edge(second);
    if (first_tree && second_tree) {
      swap_tree_edges(position, first, second);
      return;
    }
    if (first_tree && !boundaries_[second].empty() && positive_path_uses_tree_edge(second, first)) {
      exchange_tree_and_positive(position, first, second);
      return;
    }

    // The persistence pairing is unchanged. Keep the dynamic forest's edge
    // ranks synchronized with the new filtration order.
    if (first_tree) tree_forest_.update_weight(first, position + 1);
    if (second_tree) tree_forest_.update_weight(second, position);
    swap_order(position, first, second);
  }

  void vine_swap(Index position) {
    const Index first = order_[position];
    const Index second = order_[position + 1];
    if (dimensions_[first] == base_degree_) {
      swap_vertices(position, first, second);
    } else {
      swap_edges(position, first, second);
    }
  }

  std::vector<Dimension> dimensions_;
  std::vector<Boundary> boundaries_;
  Map order_;
  Map order_position_;
  Map vertex_number_;
  Map component_id_;
  std::vector<Index> death_by_birth_;
  std::vector<Index> birth_by_death_;
  std::vector<Bar> bars_;
  std::vector<bool> positive_edges_;
  std::vector<std::size_t> h0_slot_by_birth_;
  std::vector<std::size_t> h1_slot_by_edge_;
  Link_cut_forest<Index> tree_forest_;
  Index number_of_vertices_ = 0;
  Dimension base_degree_ = 0;
  bool initialized_ = false;
};

}  // namespace multipers::graph_mph0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <gudhi/persistence_interval.h>

namespace multipers::graph_mph0 {

// Graph-specialized H0/H1 persistence backend. Updates replay the same adjacent
// filtration swaps as matrix vineyards and preserve barcode slot identities.
template <class Structure>
class Slicer_backend {
 public:
  using Dimension = int;
  using Index = std::uint32_t;
  using Bar = Gudhi::persistence_matrix::Persistence_interval<Dimension, Index>;
  using Cycle = std::vector<Index>;
  using Map = std::vector<Index>;
  template <class Complex>
  using As_type = Slicer_backend<Complex>;

  static constexpr auto nullDeath = Bar::inf;
  static constexpr bool is_vine = true;
  static constexpr bool has_rep_cycles = false;

  Slicer_backend() = default;

  template <class Complex, class Filtration_range>
  Slicer_backend(const Complex& complex, const Filtration_range& values, bool ignore_inf = false) {
    initialize(complex, values, ignore_inf);
  }

  template <class Complex, class Filtration_range>
  void initialize(const Complex& complex, const Filtration_range& values, bool = false) {
    Slicer_backend replacement;
    replacement.dimensions_.assign(complex.get_dimensions().begin(), complex.get_dimensions().end());
    replacement.boundaries_.reserve(replacement.dimensions_.size());
    for (const auto& boundary : complex.get_boundaries()) {
      replacement.boundaries_.emplace_back(boundary.begin(), boundary.end());
    }
    replacement.validate();
    replacement.update(values);
    *this = std::move(replacement);
  }

  template <class Filtration_range>
  void update(const Filtration_range& values, bool = false) {
    if (values.size() != dimensions_.size()) throw std::invalid_argument("Graph backend filtration size mismatch");
    if (!initialized_) {
      order_.resize(dimensions_.size());
      std::iota(order_.begin(), order_.end(), 0);
      std::sort(order_.begin(), order_.end(), [&](Index a, Index b) {
        if (dimensions_[a] != dimensions_[b]) return dimensions_[a] < dimensions_[b];
        return values[a] < values[b];
      });
      bars_ = compute_bars();
      std::sort(bars_.begin(), bars_.end(), bar_less);
      initialize_vineyard_state();
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
  void validate() {
    if (boundaries_.size() != dimensions_.size()) throw std::invalid_argument("Graph backend malformed boundaries");
    if (dimensions_.empty()) return;
    base_degree_ = *std::min_element(dimensions_.begin(), dimensions_.end());
    if (base_degree_ < 0) throw std::invalid_argument("Graph backend dimensions must be nonnegative");
    for (Index generator = 0; generator < dimensions_.size(); ++generator) {
      if (dimensions_[generator] == base_degree_) {
        if (!boundaries_[generator].empty()) throw std::invalid_argument("Graph vertex boundary must be empty");
        continue;
      }
      if (dimensions_[generator] != base_degree_ + 1) {
        throw std::invalid_argument("Graph backend requires two adjacent generator dimensions");
      }
      if (boundaries_[generator].empty()) continue;
      if (boundaries_[generator].size() != 2) {
        throw std::invalid_argument("Graph nonempty relation boundary must contain exactly two generators");
      }
      if (boundaries_[generator][0] == boundaries_[generator][1]) {
        throw std::invalid_argument("Graph relation endpoints must be distinct");
      }
      for (Index vertex : boundaries_[generator]) {
        if (vertex >= generator || dimensions_[vertex] != base_degree_) {
          throw std::invalid_argument("Graph edge endpoint is invalid");
        }
      }
    }
  }

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

  static bool endpoint_matches(Index old_endpoint, Index new_endpoint, Index first, Index second) {
    const bool old_swapped = old_endpoint == first || old_endpoint == second;
    const bool new_swapped = new_endpoint == first || new_endpoint == second;
    return old_swapped ? new_swapped : !new_swapped && old_endpoint == new_endpoint;
  }

  static bool same_external_endpoints(const Bar& old_bar, const Bar& new_bar, Index first, Index second) {
    return old_bar.dim == new_bar.dim && endpoint_matches(old_bar.birth, new_bar.birth, first, second) &&
           endpoint_matches(old_bar.death, new_bar.death, first, second);
  }

  void initialize_vineyard_state() {
    const std::size_t no_slot = std::numeric_limits<std::size_t>::max();
    const Index count = dimensions_.size();
    order_position_.resize(count);
    vertex_number_.assign(count, count);
    positive_edges_.assign(count, false);
    death_by_birth_.assign(count, Bar::inf);
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
        h0_slot_by_birth_[bar.birth] = slot;
      } else {
        positive_edges_[bar.birth] = true;
        h1_slot_by_edge_[bar.birth] = slot;
      }
    }

    tree_edges_.clear();
    for (Index generator = 0; generator < count; ++generator) {
      if (dimensions_[generator] == base_degree_ + 1 && !boundaries_[generator].empty() &&
          !positive_edges_[generator]) {
        tree_edges_.push_back(generator);
      }
    }
  }

  bool path_uses_tree_edge(Index positive_edge, Index tree_edge) const {
    std::vector<Index> parent(number_of_vertices_);
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&](Index vertex) {
      while (parent[vertex] != vertex) {
        parent[vertex] = parent[parent[vertex]];
        vertex = parent[vertex];
      }
      return vertex;
    };
    for (Index edge : tree_edges_) {
      if (edge == tree_edge) continue;
      Index u = root(vertex_number_[boundaries_[edge][0]]);
      Index v = root(vertex_number_[boundaries_[edge][1]]);
      if (u != v) parent[v] = u;
    }
    return root(vertex_number_[boundaries_[positive_edge][0]]) != root(vertex_number_[boundaries_[positive_edge][1]]);
  }

  std::vector<Index> compute_h0_deaths(const std::vector<Index>& tree_edges) const {
    const Index count = dimensions_.size();
    std::vector<Index> deaths(count, Bar::inf);
    std::vector<Index> parent(number_of_vertices_);
    std::vector<Index> component_birth(number_of_vertices_, count);
    std::iota(parent.begin(), parent.end(), 0);
    for (Index generator = 0; generator < count; ++generator) {
      if (dimensions_[generator] == base_degree_) component_birth[vertex_number_[generator]] = generator;
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

    std::vector<Index> ordered_tree_edges = tree_edges;
    std::sort(ordered_tree_edges.begin(), ordered_tree_edges.end(), [&](Index a, Index b) {
      return order_position_[a] < order_position_[b];
    });
    for (Index edge : ordered_tree_edges) {
      Index ru = root(vertex_number_[boundaries_[edge][0]]);
      Index rv = root(vertex_number_[boundaries_[edge][1]]);
      if (ru == rv) throw std::logic_error("Graph vineyard tree edge created a cycle");
      if (order_position_[component_birth[rv]] < order_position_[component_birth[ru]]) std::swap(ru, rv);
      deaths[component_birth[rv]] = edge;
      parent[rv] = ru;
    }
    return deaths;
  }

  void transport_h0(const std::vector<Index>& new_deaths, Index first, Index second) {
    const std::size_t no_slot = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> new_slots(dimensions_.size(), no_slot);
    std::vector<bool> consumed(dimensions_.size(), false);
    std::vector<Index> changed_births;
    std::vector<std::pair<std::size_t, Bar>> updates;
    updates.reserve(number_of_vertices_);

    for (Index birth = 0; birth < dimensions_.size(); ++birth) {
      if (dimensions_[birth] != base_degree_) continue;
      if (death_by_birth_[birth] == new_deaths[birth]) {
        const std::size_t slot = h0_slot_by_birth_[birth];
        if (slot == no_slot) throw std::logic_error("Missing Graph vineyard H0 slot");
        updates.emplace_back(slot, Bar(birth, new_deaths[birth], base_degree_));
        new_slots[birth] = slot;
        consumed[birth] = true;
      } else {
        changed_births.push_back(birth);
      }
    }

    for (Index old_birth : changed_births) {
      const Bar old_bar(old_birth, death_by_birth_[old_birth], base_degree_);
      Index match = dimensions_.size();
      for (Index new_birth = 0; new_birth < dimensions_.size(); ++new_birth) {
        if (dimensions_[new_birth] != base_degree_ || consumed[new_birth]) continue;
        if (same_external_endpoints(old_bar, Bar(new_birth, new_deaths[new_birth], base_degree_), first, second)) {
          if (match != dimensions_.size()) throw std::logic_error("Ambiguous Graph vineyard H0 transport");
          match = new_birth;
        }
      }
      if (match == dimensions_.size()) throw std::logic_error("Missing Graph vineyard H0 transport");
      const std::size_t slot = h0_slot_by_birth_[old_birth];
      if (slot == no_slot) throw std::logic_error("Missing Graph vineyard H0 slot");
      updates.emplace_back(slot, Bar(match, new_deaths[match], base_degree_));
      new_slots[match] = slot;
      consumed[match] = true;
    }

    for (Index birth = 0; birth < dimensions_.size(); ++birth) {
      if (dimensions_[birth] == base_degree_ && !consumed[birth]) {
        throw std::logic_error("Incomplete Graph vineyard H0 transport");
      }
    }
    for (const auto& [slot, bar] : updates) bars_[slot] = bar;
    death_by_birth_ = new_deaths;
    h0_slot_by_birth_ = std::move(new_slots);
  }

  void vine_swap(Index position) {
    const Index first = order_[position];
    const Index second = order_[position + 1];
    bool replace_tree_edge = false;
    std::vector<Index> new_tree_edges;
    if (dimensions_[first] == base_degree_ + 1) {
      if (boundaries_[first].empty() || boundaries_[second].empty() || positive_edges_[first]) {
        std::swap(order_[position], order_[position + 1]);
        order_position_[first] = position + 1;
        order_position_[second] = position;
        return;
      }
      if (positive_edges_[second]) {
        if (!path_uses_tree_edge(second, first)) {
          std::swap(order_[position], order_[position + 1]);
          order_position_[first] = position + 1;
          order_position_[second] = position;
          return;
        }
        new_tree_edges = tree_edges_;
        const auto tree_edge = std::find(new_tree_edges.begin(), new_tree_edges.end(), first);
        if (tree_edge == new_tree_edges.end()) throw std::logic_error("Missing Graph vineyard tree edge");
        *tree_edge = second;
        if (h1_slot_by_edge_[second] == std::numeric_limits<std::size_t>::max()) {
          throw std::logic_error("Missing Graph vineyard H1 slot");
        }
        replace_tree_edge = true;
      }
    }

    std::swap(order_[position], order_[position + 1]);
    order_position_[first] = position + 1;
    order_position_[second] = position;
    try {
      transport_h0(compute_h0_deaths(replace_tree_edge ? new_tree_edges : tree_edges_), first, second);
    } catch (...) {
      std::swap(order_[position], order_[position + 1]);
      order_position_[first] = position;
      order_position_[second] = position + 1;
      throw;
    }

    if (replace_tree_edge) {
      tree_edges_ = std::move(new_tree_edges);
      positive_edges_[first] = true;
      positive_edges_[second] = false;
      const std::size_t h1_slot = h1_slot_by_edge_[second];
      h1_slot_by_edge_[second] = std::numeric_limits<std::size_t>::max();
      h1_slot_by_edge_[first] = h1_slot;
      bars_[h1_slot] = Bar(first, Bar::inf, base_degree_ + 1);
    }
  }

  std::vector<Dimension> dimensions_;
  std::vector<std::vector<Index>> boundaries_;
  Map order_;
  Map order_position_;
  Map vertex_number_;
  std::vector<Index> tree_edges_;
  std::vector<Index> death_by_birth_;
  std::vector<Bar> bars_;
  std::vector<bool> positive_edges_;
  std::vector<std::size_t> h0_slot_by_birth_;
  std::vector<std::size_t> h1_slot_by_edge_;
  Index number_of_vertices_ = 0;
  Dimension base_degree_ = 0;
  bool initialized_ = false;
};

}  // namespace multipers::graph_mph0

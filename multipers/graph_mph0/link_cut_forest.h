#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace multipers::graph_mph0 {

// Edge-as-node link-cut forest. Path maxima use deterministic (weight, edge id) keys.
template <class Weight>
class Link_cut_forest {
 public:
  struct Edge {
    std::size_t id;
    std::size_t u;
    std::size_t v;
    Weight weight;
  };

  explicit Link_cut_forest(std::size_t vertices = 0) { reset(vertices); }

  void reset(std::size_t vertices) {
    vertices_ = vertices;
    nodes_.assign(vertices, Node{});
    edges_.clear();
  }

  std::size_t num_vertices() const { return vertices_; }

  bool same_component(std::size_t u, std::size_t v) {
    check_vertex(u);
    check_vertex(v);
    return u == v || find_root(u) == find_root(v);
  }

  void link(std::size_t id, std::size_t u, std::size_t v, Weight weight) {
    check_vertex(u);
    check_vertex(v);
    if (u == v || same_component(u, v)) throw std::invalid_argument("link-cut link would create a cycle");
    if (edges_.contains(id)) throw std::invalid_argument("link-cut edge id already exists");

    const std::size_t node = nodes_.size();
    Node edge_node;
    edge_node.is_edge = true;
    edge_node.edge_id = id;
    edge_node.weight = weight;
    edge_node.maximum = node;
    nodes_.push_back(edge_node);
    edges_.emplace(id, Edge_record{u, v, node, true});
    link_nodes(u, node);
    link_nodes(node, v);
  }

  void cut(std::size_t id) {
    auto it = edges_.find(id);
    if (it == edges_.end() || !it->second.active) throw std::invalid_argument("link-cut edge is not active");
    cut_nodes(it->second.u, it->second.node);
    cut_nodes(it->second.node, it->second.v);
    it->second.active = false;
  }

  void update_weight(std::size_t id, Weight weight) {
    auto it = edges_.find(id);
    if (it == edges_.end() || !it->second.active) throw std::invalid_argument("link-cut edge is not active");
    access(it->second.node);
    nodes_[it->second.node].weight = weight;
    pull(it->second.node);
  }

  std::optional<Edge> path_bottleneck(std::size_t u, std::size_t v) {
    check_vertex(u);
    check_vertex(v);
    if (u == v || !same_component(u, v)) return std::nullopt;
    make_root(u);
    access(v);
    const std::size_t node = nodes_[v].maximum;
    if (!nodes_[node].is_edge) throw std::logic_error("link-cut path has no edge maximum");
    const auto& record = edges_.at(nodes_[node].edge_id);
    return Edge{nodes_[node].edge_id, record.u, record.v, nodes_[node].weight};
  }

  std::vector<std::size_t> path_edges(std::size_t u, std::size_t v) {
    check_vertex(u);
    check_vertex(v);
    if (u == v || !same_component(u, v)) return {};
    make_root(u);
    access(v);

    std::vector<std::size_t> out;
    std::vector<std::size_t> stack{v};
    while (!stack.empty()) {
      const std::size_t node = stack.back();
      stack.pop_back();
      push(node);
      if (nodes_[node].is_edge) out.push_back(nodes_[node].edge_id);
      for (const std::size_t child : nodes_[node].child) {
        if (child != null) stack.push_back(child);
      }
    }
    return out;
  }

 private:
  static constexpr std::size_t null = static_cast<std::size_t>(-1);

  struct Node {
    std::size_t child[2] = {null, null};
    std::size_t parent = null;
    std::size_t maximum = null;
    std::size_t edge_id = null;
    Weight weight{};
    bool reverse = false;
    bool is_edge = false;
  };

  struct Edge_record {
    std::size_t u;
    std::size_t v;
    std::size_t node;
    bool active;
  };

  std::size_t vertices_ = 0;
  std::vector<Node> nodes_;
  std::unordered_map<std::size_t, Edge_record> edges_;

  void check_vertex(std::size_t v) const {
    if (v >= vertices_) throw std::out_of_range("link-cut vertex is out of range");
  }

  bool is_root(std::size_t x) const {
    const std::size_t p = nodes_[x].parent;
    return p == null || (nodes_[p].child[0] != x && nodes_[p].child[1] != x);
  }

  bool key_less(std::size_t a, std::size_t b) const {
    if (a == null || !nodes_[a].is_edge) return b != null && nodes_[b].is_edge;
    if (b == null || !nodes_[b].is_edge) return false;
    return nodes_[a].weight < nodes_[b].weight ||
           (!(nodes_[b].weight < nodes_[a].weight) && nodes_[a].edge_id < nodes_[b].edge_id);
  }

  void pull(std::size_t x) {
    nodes_[x].maximum = nodes_[x].is_edge ? x : null;
    for (std::size_t child : nodes_[x].child) {
      if (child != null && key_less(nodes_[x].maximum, nodes_[child].maximum)) {
        nodes_[x].maximum = nodes_[child].maximum;
      }
    }
  }

  void apply_reverse(std::size_t x) {
    if (x == null) return;
    std::swap(nodes_[x].child[0], nodes_[x].child[1]);
    nodes_[x].reverse = !nodes_[x].reverse;
  }

  void push(std::size_t x) {
    if (!nodes_[x].reverse) return;
    apply_reverse(nodes_[x].child[0]);
    apply_reverse(nodes_[x].child[1]);
    nodes_[x].reverse = false;
  }

  void push_path(std::size_t x) {
    if (!is_root(x)) push_path(nodes_[x].parent);
    push(x);
  }

  void rotate(std::size_t x) {
    const std::size_t p = nodes_[x].parent;
    const std::size_t g = nodes_[p].parent;
    const int side = nodes_[p].child[1] == x;
    const std::size_t middle = nodes_[x].child[side ^ 1];
    if (!is_root(p)) nodes_[g].child[nodes_[g].child[1] == p] = x;
    nodes_[x].parent = g;
    nodes_[x].child[side ^ 1] = p;
    nodes_[p].parent = x;
    nodes_[p].child[side] = middle;
    if (middle != null) nodes_[middle].parent = p;
    pull(p);
    pull(x);
  }

  void splay(std::size_t x) {
    push_path(x);
    while (!is_root(x)) {
      const std::size_t p = nodes_[x].parent;
      const std::size_t g = nodes_[p].parent;
      if (!is_root(p)) {
        if ((nodes_[p].child[1] == x) == (nodes_[g].child[1] == p))
          rotate(p);
        else
          rotate(x);
      }
      rotate(x);
    }
  }

  void access(std::size_t x) {
    std::size_t previous = null;
    for (std::size_t y = x; y != null; y = nodes_[y].parent) {
      splay(y);
      nodes_[y].child[1] = previous;
      pull(y);
      previous = y;
    }
    splay(x);
  }

  void make_root(std::size_t x) {
    access(x);
    apply_reverse(x);
  }

  std::size_t find_root(std::size_t x) {
    access(x);
    push(x);
    while (nodes_[x].child[0] != null) {
      x = nodes_[x].child[0];
      push(x);
    }
    splay(x);
    return x;
  }

  void link_nodes(std::size_t u, std::size_t v) {
    make_root(u);
    if (find_root(v) == u) throw std::logic_error("link-cut internal cycle");
    nodes_[u].parent = v;
  }

  void cut_nodes(std::size_t u, std::size_t v) {
    make_root(u);
    access(v);
    if (nodes_[v].child[0] != u || nodes_[u].child[1] != null) {
      throw std::logic_error("link-cut endpoints are not adjacent");
    }
    nodes_[v].child[0] = null;
    nodes_[u].parent = null;
    pull(v);
  }
};

}  // namespace multipers::graph_mph0

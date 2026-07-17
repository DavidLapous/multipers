#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace multipers::graph_mph0 {

using Grade = std::array<double, 2>;

struct Edge {
  std::size_t id;
  std::size_t u;
  std::size_t v;
  Grade grade;
};

struct Graph {
  std::vector<Grade> vertices;
  std::vector<Edge> edges;
};

struct Result {
  std::vector<Grade> beta_0;
  std::vector<Grade> beta_1;
  std::vector<Grade> beta_2;
  std::vector<Grade> beta_0_h1;
  std::vector<std::array<std::size_t, 2>> relations;
  std::vector<std::vector<std::size_t>> syzygies;
};

struct Compute_options {
  bool full_resolution = true;
  bool h1_betti = true;
};

Result compute(const Graph& graph, Compute_options options = {});

}  // namespace multipers::graph_mph0

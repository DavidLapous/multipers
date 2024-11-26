#pragma once

#include "gudhi/Flag_complex_edge_collapser.h"
#include <gudhi/Persistent_cohomology.h>
#include <gudhi/Simplex_tree_multi.h>
#include <gudhi/Multi_persistence/Box.h>

namespace Gudhi {
namespace multiparameter {

struct Simplex_tree_float {  // smaller simplextrees
  typedef linear_indexing_tag Indexing_tag;
  typedef std::int32_t Vertex_handle;
  typedef float Filtration_value;
  typedef std::uint32_t Simplex_key;
  static const bool store_key = true;
  static const bool store_filtration = true;
  static const bool contiguous_vertices = false;  // TODO OPTIMIZATION : maybe make the simplextree contiguous when
                                                  // calling grid_squeeze ?
  static const bool link_nodes_by_label = true;
  static const bool stable_simplex_handles = false;
  static const bool is_multi_parameter = false;
};

// using Simplex_tree_float = Simplex_tree_options_fast_persistence;

using Gudhi::multi_persistence::Box;
using Simplex_tree_std = Simplex_tree<Simplex_tree_float>;

using Barcode = std::vector<std::pair<Simplex_tree_std::Filtration_value, Simplex_tree_std::Filtration_value>>;

inline Barcode compute_dgm(Simplex_tree_std &st, int degree) {
  st.initialize_filtration(true);
  constexpr int coeff_field_characteristic = 11;
  constexpr Simplex_tree_std::Filtration_value min_persistence = 0;
  bool persistence_dim_max = st.dimension() == degree;
  Gudhi::persistent_cohomology::Persistent_cohomology<Simplex_tree_std, Gudhi::persistent_cohomology::Field_Zp> pcoh(
      st, persistence_dim_max);
  pcoh.init_coefficients(coeff_field_characteristic);
  pcoh.compute_persistent_cohomology(min_persistence);
  const auto &persistent_pairs = pcoh.intervals_in_dimension(degree);
  if constexpr (false) {
    std::cout << "Number of bars : " << persistent_pairs.size() << "\n";
  }
  return persistent_pairs;
}

template <typename degree_type, class interface_std_like>
inline std::vector<Barcode> compute_dgms(interface_std_like &st,
                                         const std::vector<degree_type> &degrees,
                                         int num_collapses,
                                         int expansion_dim) {
  std::vector<Barcode> out(degrees.size());
  static_assert(!interface_std_like::Options::is_multi_parameter,
                "Can only compute persistence for 1-parameter simplextrees.");
  const bool verbose = false;
  if (num_collapses > 0) {
    auto collapsed_st = collapse_edges(st, num_collapses);
    return compute_dgms(collapsed_st, degrees, 0, expansion_dim);
  }
  // if (expansion_dim == -1){
  // 	int max_dim = *std::max_element(degrees.begin(), degrees.end()) +1;
  // 	st.expansion(max_dim);
  // }
  else if (expansion_dim > 0) {
    st.expansion(expansion_dim);
  }

  st.initialize_filtration(true);  // true is ignore_infinite_values
  constexpr int coeff_field_characteristic = 11;
  constexpr typename interface_std_like::Filtration_value min_persistence = 0;

  bool persistence_dim_max = false;
  for (auto degree : degrees) {
    if (st.dimension() == degree) {
      persistence_dim_max = true;
      break;
    }
  }
  //
  if constexpr (verbose) {
    std::cout << "Computing dgm of st:\n";
    for (auto &sh : st.filtration_simplex_range()) {
      std::cout << "dim: " << st.dimension(sh) << " vertices: ";
      for (auto v : st.simplex_vertex_range(sh)) {
        std::cout << v << " ";
      }
      std::cout << " filtration: ";
      std::cout << st.filtration(sh) << "\n";
    }
    return out;
  }

  Gudhi::persistent_cohomology::Persistent_cohomology<interface_std_like, Gudhi::persistent_cohomology::Field_Zp> pcoh(
      st, persistence_dim_max);
  pcoh.init_coefficients(coeff_field_characteristic);
  pcoh.compute_persistent_cohomology(min_persistence);
  for (auto i = 0u; i < degrees.size(); i++) {
    out[i] = pcoh.intervals_in_dimension(degrees[i]);
  }
  return out;
}

// small wrapper
template <typename degree_type, class interface_std_like>
inline std::vector<Barcode> compute_dgms(interface_std_like &st,
                                         const std::vector<degree_type> &degrees,
                                         int expand_collapse_dim = 0) {
  if (expand_collapse_dim > 0) return compute_dgms(st, degrees, 10, expand_collapse_dim);
  return compute_dgms(st, degrees, 0, 0);
}

// Adapted version from the Simplextree interface
template <class simplextree_like>
inline simplextree_like collapse_edges(simplextree_like &st, int num_collapses) {
  using Filtered_edge = std::tuple<typename simplextree_like::Vertex_handle,
                                   typename simplextree_like::Vertex_handle,
                                   typename simplextree_like::Filtration_value>;
  std::vector<Filtered_edge> edges;
  for (auto sh : st.skeleton_simplex_range(1)) {
    if (st.dimension(sh) == 1) {
      const auto filtration = st.filtration(sh);
      if (filtration == std::numeric_limits<typename simplextree_like::Filtration_value>::infinity()) continue;
      typename simplextree_like::Simplex_vertex_range rg = st.simplex_vertex_range(sh);
      auto vit = rg.begin();
      typename simplextree_like::Vertex_handle v = *vit;
      typename simplextree_like::Vertex_handle w = *++vit;
      edges.emplace_back(v, w, filtration);
    }
  }
  for (int iteration = 0; iteration < num_collapses; iteration++) {
    auto current_size = edges.size();
    edges = Gudhi::collapse::flag_complex_collapse_edges(std::move(edges));
    if (edges.size() >= current_size) break;  // no need to do more
  }
  simplextree_like collapsed_stree;
  // Copy the original 0-skeleton
  for (auto sh : st.skeleton_simplex_range(0)) {
    collapsed_stree.insert_simplex({*(st.simplex_vertex_range(sh).begin())}, st.filtration(sh));
  }
  // Insert remaining edges
  for (auto [x, y, filtration] : edges) {
    collapsed_stree.insert_simplex({x, y}, filtration);
  }
  return collapsed_stree;
}

}}  // namespace Gudhi::multiparameter

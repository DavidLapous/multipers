/*    This file is part of the MMA Library -
 * https://gitlab.inria.fr/dloiseau/multipers - which is released under MIT. See
 * file LICENSE for full license details. Author(s):       David Loiseaux
 *
 *    Copyright (C) 2021 Inria
 *
 *    Modification(s):
 *      - 2022/03 Hannah Schreiber: Integration of the new Vineyard_persistence
 * class, renaming and cleanup.
 */
/**
 * @file format_python-cpp.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Functions that change the format of data to communicate between C++
 * and python.
 */

#ifndef FORMAT_PYTHON_CPP_H_INCLUDED
#define FORMAT_PYTHON_CPP_H_INCLUDED

#include <cstddef>
#include <vector>

#include "multiparameter_module_approximation/utilities.h"
#include <gudhi/Simplex_tree.h>
#include <gudhi/Simplex_tree_multi.h>

namespace Gudhi::multiparameter::mma {

// Lexical order + dimension
bool inline is_strictly_smaller_simplex(const boundary_type &s1,
                                 const boundary_type &s2) {
  if (s1.size() < s2.size())
    return true;
  if (s1.size() > s2.size())
    return false;

  for (unsigned int i = 0; i < s1.size(); i++) {
    if (s1[i] < s2[i])
      return true;
    if (s1[i] > s2[i])
      return false;
  }
  return false;
}

// std::pair<boundary_matrix, multifiltration_type>
// inline simplextree_to_boundary_filtration(const uintptr_t splxptr) {
//   using option =
//       Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<>;
//   Gudhi::Simplex_tree<option> &simplexTree =
//       *(Gudhi::Simplex_tree<option> *)(splxptr);
//
//   unsigned int numberOfSimplices = simplexTree.num_simplices();
//   boundary_matrix boundaries(numberOfSimplices);
//   boundary_matrix simplices(numberOfSimplices);
//   if (simplexTree.num_simplices() <= 0)
//     return {{}, {{}}};
//   unsigned int filtration_number =
//       simplexTree.filtration(*(simplexTree.complex_simplex_range().begin()))
//           .size();
//   std::vector<filtration_type> filtration(filtration_number,
//                                           filtration_type(numberOfSimplices));
//
//   unsigned int count = 0;
//   for (auto sh : simplexTree.filtration_simplex_range())
//     simplexTree.assign_key(sh, count++);
//
//   unsigned int i = 0;
//   for (auto &simplex : simplexTree.filtration_simplex_range()) {
//     for (const auto &simplex_id : simplexTree.boundary_simplex_range(simplex)) {
//       boundaries[i].push_back(simplexTree.key(simplex_id));
//     }
//     for (const auto &vertex : simplexTree.simplex_vertex_range(simplex)) {
//       simplices[i].push_back(vertex);
//     }
//     const auto &temp = simplexTree.filtration(simplex);
//     for (unsigned int j = 0; j < temp.size(); j++)
//       filtration[j][i] = temp[j];
//     i++;
//   }
//   for (boundary_type &simplex : simplices) {
//     std::sort(simplex.begin(), simplex.end());
//   }
//   permutation_type p =
//       Combinatorics::sort_and_return_permutation<boundary_type>(
//           simplices, &is_strictly_smaller_simplex);
//
//   for (auto &F : filtration) {
//     Combinatorics::compose(F, p);
//   }
//
//   Combinatorics::compose(boundaries, p);
//
//   auto inv = Combinatorics::inverse(p);
//
//   for (boundary_type &simplex : boundaries) {
//     for (auto &b : simplex)
//       b = inv[b];
//     std::sort(simplex.begin(), simplex.end());
//   }
//
//   return std::make_pair(boundaries, filtration);
// }

template <typename Options>
using scc_type =
    std::vector<std::pair<std::vector<std::vector<typename Options::value_type>>, boundary_matrix>>;
template <typename STOptions>
inline scc_type<STOptions> simplextree_to_scc(Gudhi::Simplex_tree<STOptions> &st) {
  scc_type<STOptions> out(st.dimension() + 1);
  if (st.num_simplices() <= 0)
    return out;

  /* Assigns keys to simplices according their dimension */
  std::vector<int> simplices_per_block_dim(st.dimension() + 1);
  for (auto sh : st.filtration_simplex_range())
    st.assign_key(sh, simplices_per_block_dim[st.dimension(sh)]++);

  std::vector<unsigned int> key_boundary_container;
  for (auto &simplex : st.filtration_simplex_range()) {
    key_boundary_container.clear();
    for (const auto &simplex_id : st.boundary_simplex_range(simplex)) {
      key_boundary_container.push_back(st.key(simplex_id));
    }
    auto &[block_filtrations, block_matrix] = out[st.dimension(simplex)];
    const typename STOptions::Filtration_value &simplex_filtration = st.filtration(simplex);
    block_matrix.push_back(key_boundary_container);

    block_filtrations.push_back(static_cast<typename STOptions::Filtration_value::Generator>(simplex_filtration));
  }
  return out;
}
template <typename Options>
using kscc_type =
    std::vector<std::pair<std::vector<std::vector<std::vector< typename Options::value_type >>>, boundary_matrix>>;
template <typename STOptions>
inline kscc_type<STOptions> kcritical_simplextree_to_scc(Gudhi::Simplex_tree<STOptions> &st) {
  static_assert(STOptions::Filtration_value::is_multi_critical);
  kscc_type<STOptions> out(st.dimension() + 1);
  if (st.num_simplices() <= 0)
    return out;

  /* Assigns keys to simplices according their dimension */
  std::vector<int> simplices_per_block_dim(st.dimension() + 1);
  for (auto sh : st.complex_simplex_range())
    st.assign_key(sh, simplices_per_block_dim[st.dimension(sh)]++);

  std::vector<unsigned int> key_boundary_container;
  for (auto &simplex : st.complex_simplex_range()) {
    key_boundary_container.clear();
    for (const auto &simplex_id : st.boundary_simplex_range(simplex)) {
      key_boundary_container.push_back(st.key(simplex_id));
    }
    auto &[block_filtrations, block_matrix] = out[st.dimension(simplex)];
    const typename STOptions::Filtration_value &simplex_filtration = st.filtration(simplex);
    block_matrix.push_back(key_boundary_container);
    block_filtrations.push_back(
        std::vector<std::vector<typename STOptions::Filtration_value::value_type>>(
            simplex_filtration.begin(), simplex_filtration.end()));
  }
  return out;
}

template <typename value_type>
using function_scc_type = std::vector<
    std::pair<std::vector<std::vector<std::vector<value_type>>>, boundary_matrix>>;
template <typename STOptions>
inline function_scc_type<typename STOptions::Filtration_value::value_type>
function_simplextree_to_scc(Gudhi::Simplex_tree<STOptions> &st) {
  using value_type =
      typename STOptions::Filtration_value::value_type;
  function_scc_type<value_type> out(st.dimension() + 1);
  if (st.num_simplices() <= 0)
    return out;

  /* Assigns keys to simplices according their dimension */
  std::vector<int> simplices_per_block_dim(st.dimension() + 1);
  for (auto sh : st.filtration_simplex_range())
    st.assign_key(sh, simplices_per_block_dim[st.dimension(sh)]++);

  std::vector<unsigned int> key_boundary_container;
  for (auto &simplex : st.filtration_simplex_range()) {
    key_boundary_container.clear();
    for (const auto &simplex_id : st.boundary_simplex_range(simplex)) {
      key_boundary_container.push_back(st.key(simplex_id));
    }
    auto &[block_filtrations, block_matrix] = out[st.dimension(simplex)];
    const auto &simplex_filtration = st.filtration(simplex);
    block_matrix.push_back(key_boundary_container);
    std::vector<std::vector<value_type>> _filtration;
    for (std::size_t i = 0; i < simplex_filtration.size(); i++) {
      _filtration.push_back(
          {static_cast<value_type>(simplex_filtration[i]), static_cast<value_type>(i)});
    }
    block_filtrations.push_back(_filtration);
  }
  return out;
}

// scc_type inline simplextree_to_scc(const uintptr_t splxptr) {
//   using option =
//       Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<>;
//   Gudhi::Simplex_tree<option> &st = *(Gudhi::Simplex_tree<option> *)(splxptr);
//   return simplextree_to_scc<option>(st);
// }
// function_scc_type inline function_simplextree_to_scc(const uintptr_t splxptr) {
//   using option =
//       Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<>;
//   Gudhi::Simplex_tree<option> &st = *(Gudhi::Simplex_tree<option> *)(splxptr);
//   return function_simplextree_to_scc<option>(st);
// }
template <typename Options>
using flattened_scc_type =
    std::pair<std::vector<std::vector<typename Options::value_type>>,
              std::vector<std::vector<unsigned int>>>;

template <typename Options>
flattened_scc_type<Options> inline simplextree_to_ordered_bf(
    Gudhi::Simplex_tree<Options> &st) {
  auto scc = simplextree_to_scc<Options>(st);
  flattened_scc_type<Options> out;
  auto &[filtration, boundary] = out;
  std::size_t num_simplices = 0;
  std::vector<std::size_t> cumsum_sizes = {0, 0};
  for (auto &[f, b] : scc) {
    num_simplices += b.size();
    cumsum_sizes.push_back(num_simplices);
  }
  filtration.reserve(num_simplices);
  boundary.reserve(num_simplices);
  for (auto i = 0u; i < scc.size(); ++i) {
    auto shift = cumsum_sizes[i];
    auto &[f, b] = scc[i];
    for (auto j = 0u; j < b.size(); ++j) {
      filtration.push_back(f[j]);
      auto new_b = b[j];
      for (auto &stuff : new_b)
        stuff += shift;
      boundary.push_back(new_b);
    }
  }
  return out;
}
// template <
//     typename Options =
//         Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<>>
// flattened_scc_type<Options> simplextree_to_ordered_bf(const uintptr_t splxptr) {
//   Gudhi::Simplex_tree<Options> &st = *(Gudhi::Simplex_tree<Options> *)(splxptr);
//   return simplextree_to_ordered_bf<Options>(st);
// }
template <typename MultiFiltration>
using BoundaryFiltration =
    std::pair<boundary_matrix, std::vector<MultiFiltration>>;

template <class SimplexTree>
BoundaryFiltration<typename SimplexTree::Options::Filtration_value>
st2bf(SimplexTree &st) {

  BoundaryFiltration<typename SimplexTree::Options::Filtration_value> out;
  auto &[matrix, filtrations] = out;
  matrix.reserve(st.num_simplices());
  filtrations.reserve(st.num_simplices());
  if (st.num_simplices() <= 0)
    return out;

  /* Assigns keys to simplices according their dimension */
  int count = 0;
  for (auto sh : st.complex_simplex_range())
    st.assign_key(sh, count++);

  std::vector<unsigned int> key_boundary_container;
  for (auto &simplex : st.complex_simplex_range()) {
    key_boundary_container.clear();
    for (const auto &simplex_id : st.boundary_simplex_range(simplex)) {
      key_boundary_container.push_back(st.key(simplex_id));
    }
    matrix.push_back(key_boundary_container);
    filtrations.push_back(st.filtration(simplex));
  }
  return out;
}

} // namespace Gudhi::multiparameter::mma

#endif // FORMAT_PYTHON_CPP_H_INCLUDED

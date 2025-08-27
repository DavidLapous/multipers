#pragma once

#include "Simplex_tree_multi_interface.h"
#include "multi_parameter_rank_invariant/persistence_slices.h"
#include "tensor/tensor.h"
#include <gudhi/One_critical_filtration.h>
#include <gudhi/Simplex_tree_multi.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>

namespace Gudhi {
namespace multiparameter {
namespace function_rips {

using value_type = typename python_interface::interface_std::Filtration_value;
using _multifiltration = typename Gudhi::multi_filtration::One_critical_filtration<value_type>;
using _multist = python_interface::Simplex_tree_multi_interface<_multifiltration>;
using interface_multi = _multist;


std::pair<std::map<value_type, unsigned int>, std::vector<value_type>>
inline radius_to_coordinate(Simplex_tree_std &st) {
  unsigned int count = 0;
  std::map<value_type, unsigned int> out;
  std::vector<value_type> filtration_values;
  filtration_values.reserve(st.num_simplices());
  for (auto sh : st.filtration_simplex_range()) { // ordered by filtration, so
                                                  // should be sorted
    auto filtration = st.filtration(sh);
    if (!out.contains(filtration)) {
      out[filtration] = count;
      filtration_values.push_back(filtration);
      count++;
    }
  }
  return {out, filtration_values};
}

// Takes a standard simplextree and turn it into a simplextreemulti, whose first
// axis is the rips, and the others are the filtrations of the node at each
// degree in degrees Assumes that the degrees are sorted, and unique

std::tuple<_multist,
           std::vector<value_type>, int>
inline get_degree_filtrations( // also return max_degree,filtration_values
    python_interface::interface_std &st, const std::vector<int> &degrees) {
  constexpr const bool verbose = false;
  using filtration_lists = std::vector<std::vector<value_type>>;

  assert(st.dimension() ==
         1); // the st slices will be expanded + collapsed after being filled.
  _multist st_multi;
  std::vector<value_type> rips_filtration_values = {
      0}; // vector that will hold the used filtration values
  rips_filtration_values.reserve(st.num_simplices());
  int max_st_degree = 0;

  int num_degrees = degrees.size();
  Gudhi::multi_persistence::multify(st, st_multi, 0); // puts the st filtration in axis 0 + fitrations for
                            // each degrees afterward
  // preprocess
  filtration_lists edge_filtration_of_nodes(st.num_vertices());
  for (auto sh : st.complex_simplex_range()) {
    if (st.dimension(sh) == 0)
      continue;
    value_type filtration = st.filtration(sh);
    for (auto node : st.simplex_vertex_range(sh)) {
      edge_filtration_of_nodes[node].push_back(filtration);
    }
  }

  for (auto &filtrations : edge_filtration_of_nodes) { // todo : parallel ?
    std::sort(filtrations.begin(), filtrations.end());
    int node_degree = filtrations.size();
    max_st_degree = std::max(node_degree, max_st_degree);
    filtrations.resize(std::max(num_degrees, node_degree));
    if constexpr (verbose)
      std::cout << "Filtration of node ";
    for (auto degree_index = 0; degree_index < num_degrees; degree_index++) {
      if (degrees[degree_index] < node_degree)
        filtrations[degree_index] = filtrations[degrees[degree_index]];
      else
        filtrations[degree_index] = std::numeric_limits<value_type>::infinity();
      if constexpr (verbose)
        std::cout << filtrations[degree_index] << " ";
    }
    filtrations.resize(num_degrees);
    std::reverse(filtrations.begin(),
                 filtrations.end()); // degree is in opposite direction
    for (value_type filtration_value : filtrations)
      rips_filtration_values.push_back(
          filtration_value); // we only do that here to have a smaller grid.
    if constexpr (verbose)
      std::cout << "\n";
  }
  // sort + unique the filtration values
  std::sort(rips_filtration_values.begin(), rips_filtration_values.end());
  rips_filtration_values.erase(
      std::unique(rips_filtration_values.begin(), rips_filtration_values.end()),
      rips_filtration_values.end());

  // fills the degree_rips simplextree with lower star
  auto sh_standard =
      st.complex_simplex_range()
          .begin(); // waiting for c++23 & zip to remove this garbage
  auto _end = st.complex_simplex_range().end();
  auto sh_multi = st_multi.complex_simplex_range().begin();
  for (; sh_standard != _end; ++sh_multi, ++sh_standard) {
    if (st.dimension(*sh_standard) == 0) // only fill using the node
      continue;                          // will be filled afterward
    // dimension is 1 by assumption. fill using the node + rips value
    value_type edge_filtration = st.filtration(*sh_standard);
    _multifiltration &edge_degree_rips_filtration =
        st_multi.filtration_mutable(*sh_multi); // the filtration vector to fill
    edge_degree_rips_filtration.resize(num_degrees);
    for (auto degree_index = 0; degree_index < num_degrees; degree_index++) {
      value_type edge_filtration_of_degree =
          edge_filtration; // copy as we do the max with edges of degree index
      for (int node : st.simplex_vertex_range(*sh_standard)) {
        edge_filtration_of_degree =
            std::max(edge_filtration_of_degree,
                     edge_filtration_of_nodes[node][degree_index]);
      }
      edge_degree_rips_filtration[degree_index] =
          edge_filtration_of_degree; // fills the correct value in the edge
                                     // filtration
    }
  }

  // fills the dimension 0 simplices
  { // scope for count;
    for (auto vertex :
         st_multi.complex_vertex_range()) { // should be in increasing order
      auto &vertex_filtration =
          st_multi.filtration_mutable(st_multi.find({vertex}));
      if constexpr (verbose) {
        std::cout << "Setting filtration of node " << vertex << " to ";
        for (auto degree_index = 0u; degree_index < num_degrees;
             degree_index++) {
          std::cout << edge_filtration_of_nodes[vertex][degree_index] << " ";
        }
        std::cout << "\n";
      }
      vertex_filtration.swap(edge_filtration_of_nodes[vertex]);
    }
  }

  return {st_multi, rips_filtration_values, max_st_degree};
}

// assumes that the degree is 1
inline void fill_st_slice(Simplex_tree_std &st_container,
                          _multist &degree_rips_st, int degree) {
  auto sh_std = st_container.complex_simplex_range().begin();
  auto sh_multi = degree_rips_st.complex_simplex_range().begin();
  auto sh_end = st_container.complex_simplex_range().end();
  for (; sh_std != sh_end; ++sh_std, ++sh_multi) {
    value_type splx_filtration =
        degree_rips_st.filtration_mutable(*sh_multi)[degree];
    st_container.assign_filtration(*sh_std, splx_filtration);
  }
  return;
}

template <typename dtype, typename index_type>
inline void
compute_2d_function_rips(_multist &st_multi, // Function rips
                         // Simplex_tree_std &_st,
                         const tensor::static_tensor_view<dtype, index_type>
                             &out, // assumes its a zero tensor
                         const std::vector<index_type> degrees, index_type I,
                         index_type J, // grid_shape
                         bool mobius_inverion, bool zero_pad) {
  constexpr bool verbose = false;
  if constexpr (verbose)
    std::cout << "Grid shape : " << I << " " << J << std::endl;

  // inits default simplextrees
  Simplex_tree_std _st;
  Gudhi::multi_persistence::flatten(_st, st_multi,
          -1); // copies the st_multi to a standard 1-pers simplextree
  tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree(_st);
  int max_simplex_dimension =
      *std::max_element(degrees.begin(), degrees.end()) + 1;
  tbb::parallel_for(0, J, [&](index_type function_value) {
    auto &st_std = thread_simplex_tree.local();
    fill_st_slice(st_std, st_multi, function_value);
    constexpr int num_collapses = 10;
    const std::vector<Barcode> barcodes =
        compute_dgms(st_std, degrees, num_collapses, max_simplex_dimension);

    index_type degree_index = 0;
    for (const auto &barcode : barcodes) { // TODO range view cartesian product
      // coordinates_container[0] = degree_index;
      for (const auto &bar : barcode) {
        auto birth = bar.first; // float
        auto death = bar.second;
        if (birth > I) // some birth can be infinite
          continue;

        if (!mobius_inverion) {
          index_type shift_value =
              out.get_cum_resolution()[1]; // degree, x coord, y coord
          index_type border = I;
          // index_type border  = out.get_resolution()[i+1];
          // dtype* ptr = &out[coordinates_container];
          dtype *ptr = &out[{degree_index, static_cast<index_type>(birth),
                             function_value}];
          auto stop_value = death > static_cast<value_type>(border)
                                ? border
                                : static_cast<index_type>(death);
          // Warning : for some reason linux static casts float inf to -min_int
          // so min doesnt work.
          if constexpr (verbose) {
            std::cout << "Adding : (";
            // for (auto stuff : coordinates_container) std::cout << stuff << ",
            // ";
            std::cout << ") With death " << death << " casted at "
                      << static_cast<index_type>(death) << "with threshold at"
                      << stop_value << " with " << border << std::endl;
          }
          for (index_type b = birth; b < stop_value; b++) {
            (*ptr)++;           // adds one to the vector
            ptr += shift_value; // shift the pointer to the next element in the
                                // segment [birth, death]
          }
        } else {
          // adds birth
          out[{degree_index, static_cast<index_type>(birth), function_value}]++;
          if constexpr (verbose) {
            std::cout << "Coordinate : ";
            // for (auto c : coordinates_container) std::cout << c << ", ";
            std::cout << std::endl;
            std::cout << "axis, death, resolution : " << 1 << ", "
                      << std::to_string(death) << ", "
                      << out.get_resolution()[1];
            std::cout << std::endl;
          }
          // removes death
          if (death < I) {
            out[{degree_index, static_cast<index_type>(death),
                 function_value}]--;
          } else if (zero_pad) {
            out[{degree_index, I - 1, function_value}]--;
          }
        }
      }
      degree_index++;
    }
  });
}

// python interface

std::pair<std::vector<value_type>, int>
inline get_degree_rips_st_python(const intptr_t simplextree_ptr,
                          const intptr_t st_multi_ptr,
                          const std::vector<int> &degrees) {
  auto &st_std = python_interface::get_simplextree_from_pointer<python_interface::interface_std>(simplextree_ptr);
  auto &st_multi_python_container =
      python_interface::get_simplextree_from_pointer<_multist>(st_multi_ptr);
  auto [st_multi, rips_filtration_values, max_node_degree] =
      get_degree_filtrations(st_std, degrees);
  st_multi_python_container = std::move(st_multi);
  return {rips_filtration_values, max_node_degree};
}

template <typename dtype, typename indices_type>
void compute_function_rips_surface_python(
    const intptr_t st_multi_ptr, dtype *data_ptr,
    const std::vector<indices_type> degrees, indices_type I, indices_type J,
    const bool mobius_inversion = false, const bool zero_pad = false,
    indices_type n_jobs = 0) {
  if (degrees.size() == 0)
    return;
  // const bool verbose = false;
  auto &st_multi = python_interface::get_simplextree_from_pointer<interface_multi>(st_multi_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(
      data_ptr, {static_cast<indices_type>(degrees.size()), I, J});
  if (zero_pad) {
    I--;
    J--;
  }

  oneapi::tbb::task_arena arena(n_jobs); // limits the number of threads
  arena.execute([&] {
    compute_2d_function_rips(st_multi, container, degrees, I, J,
                             mobius_inversion, zero_pad);
  });
  if (mobius_inversion)
    container.differentiate(2); // degree,x axis (already inversed), y axis
}

template <typename dtype, typename indices_type>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>>
compute_function_rips_signed_measure_python(
    const intptr_t st_multi_ptr, dtype *data_ptr,
    const std::vector<indices_type> degrees, indices_type I, indices_type J,
    const bool mobius_inversion = false, const bool zero_pad = false,
    indices_type n_jobs = 0) {
  if (degrees.size() == 0)
    return {{}, {}};
  // const bool verbose = false;
  auto &st_multi = python_interface::get_simplextree_from_pointer<interface_multi>(st_multi_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(
      data_ptr, {static_cast<indices_type>(degrees.size()), I, J});
  if (zero_pad) {
    I--;
    J--;
  }

  oneapi::tbb::task_arena arena(n_jobs); // limits the number of threads
  arena.execute([&] {
    compute_2d_function_rips(st_multi, container, degrees, I, J,
                             mobius_inversion, zero_pad);
  });
  if (mobius_inversion)
    container.differentiate(2); // degree,x axis (already inversed), y axis
  return container.sparsify();
}

}}} // namespace Gudhi::multiparameter::function_rips

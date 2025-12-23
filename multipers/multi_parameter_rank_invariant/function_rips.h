#pragma once

#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <limits>
#include <stdexcept>

#include "../gudhi/Simplex_tree_multi_interface.h"
#include "../tensor/tensor.h"
#include "persistence_slices.h"

namespace Gudhi {
namespace multiparameter {
namespace function_rips {

using value_type = typename python_interface::interface_std::Filtration_value;
// using _multifiltration = multipers::tmp_interface::Filtration_value<value_type>;
using flat_multifiltration = Gudhi::multi_filtration::Degree_rips_bifiltration<value_type>;
using _multifiltration = Gudhi::multi_filtration::Degree_rips_bifiltration<value_type>;
using _multi_st = python_interface::Simplex_tree_multi_interface<flat_multifiltration>;
using flat_multi_st = python_interface::Simplex_tree_multi_interface<flat_multifiltration>;
using mult_opt = Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<flat_multifiltration>;
using interface_multi = _multi_st;

std::pair<std::map<value_type, unsigned int>, std::vector<value_type>> inline radius_to_coordinate(
    Simplex_tree_std &st) {
  unsigned int count = 0;
  std::map<value_type, unsigned int> out;
  std::vector<value_type> filtration_values;
  filtration_values.reserve(st.num_simplices());
  for (auto sh : st.filtration_simplex_range()) {  // ordered by filtration, so
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
// also return max_degree,filtration_values
inline flat_multi_st get_degree_filtrations(
    python_interface::interface_std &st,
    const std::vector<int> &degrees) {
  constexpr const bool verbose = false;
  using filtration_lists = std::vector<std::vector<value_type>>;

  if (st.dimension() != 1)
    throw std::invalid_argument("get_degree_filtrations only works for 1-dimensional simplex trees.");

  unsigned int num_degrees = degrees.size();
  // puts the st filtration in axis 0 + fitrations for each degrees afterward
  flat_multifiltration default_f(static_cast<int>(num_degrees), 0);
  flat_multi_st st_multi(Gudhi::multi_persistence::make_multi_dimensional<mult_opt>(st, default_f, 0));

  // preprocess
  filtration_lists edge_filtration_of_nodes(st.num_vertices());
  for (auto sh : st.complex_simplex_range()) {
    if (st.dimension(sh) == 0) continue;
    value_type filtration = st.filtration(sh);
    for (auto node : st.simplex_vertex_range(sh)) {
      edge_filtration_of_nodes[node].push_back(filtration);
    }
  }

  tbb::parallel_for(static_cast<size_t>(0u), st.num_vertices(), [&](size_t  i){
    auto& filtrations = edge_filtration_of_nodes[i];
    tbb::parallel_sort(filtrations.begin(), filtrations.end());
    unsigned int node_degree = filtrations.size();
    filtrations.resize(std::max(num_degrees, node_degree));
    for (unsigned int degree_index = 0; degree_index < num_degrees; degree_index++) {
      if (degrees[degree_index] < static_cast<int>(node_degree))
        filtrations[degree_index] = filtrations[degrees[degree_index]];
      else
        filtrations[degree_index] = std::numeric_limits<value_type>::infinity();
      if constexpr (verbose) std::cout << filtrations[degree_index] << " ";
    }
    filtrations.resize(num_degrees);
    std::reverse(filtrations.begin(),
                 filtrations.end());  // degree is in opposite direction
  });


  // fills the degree_rips simplextree with lower star
  auto sh_standard = st.complex_simplex_range().begin();  // waiting for c++23 & zip to remove this garbage
  auto _end = st.complex_simplex_range().end();
  auto sh_multi = st_multi.complex_simplex_range().begin();
  for (; sh_standard != _end; ++sh_multi, ++sh_standard) {
    // only fill using t
    // will be filled afterward
    if (st.dimension(*sh_standard) == 0) continue;
    // dimension is 1 by assumption. fill using the node + rips value
    value_type edge_filtration = st.filtration(*sh_standard);
    // the filtration vector to fill

    std::vector<value_type> edge_degree_rips_filtration(num_degrees, std::numeric_limits<value_type>::infinity());
    // = st_multi.get_filtration_value(*sh_multi);
    // auto &edge_degree_rips_filtration = st_multi.get_filtration_value(*sh_multi);
    // edge_degree_rips_filtration.set_num_generators(num_degrees);

    if constexpr (verbose) {
      std::cout << "\nSetting filtration of edge [";
      for (int node : st.simplex_vertex_range(*sh_standard)) {
        std::cout << std::to_string(node) << ",";
      }
      std::cout << "]  to : ";
    }
    for (auto degree_index = 0u; degree_index < num_degrees; degree_index++) {
      value_type edge_filtration_of_degree = edge_filtration;  // copy as we do the max with edges of degree index
      for (int node : st.simplex_vertex_range(*sh_standard)) {
        edge_filtration_of_degree = std::max(edge_filtration_of_degree, edge_filtration_of_nodes[node][degree_index]);
      }

      edge_degree_rips_filtration[degree_index] = edge_filtration_of_degree;
      if (edge_filtration_of_degree == std::numeric_limits<value_type>::infinity()) break;
    }

    if constexpr (verbose) {
      for (auto k = 0u; k < num_degrees; k++) std::cout << " " << std::to_string(edge_degree_rips_filtration[k]);
    }
    // std::reverse(edge_degree_rips_filtration.begin(), edge_degree_rips_filtration.end());
    auto &to_update = st_multi.get_filtration_value(*sh_multi);
    if (edge_degree_rips_filtration.size() != num_degrees)
      throw std::overflow_error("edge_degree_rips of invalid size");
    to_update.set_num_generators(edge_degree_rips_filtration.size());
    for (auto k = 0u; k < edge_degree_rips_filtration.size(); k++) {
      to_update(k, 0) = edge_degree_rips_filtration[k];
    }
    // st_multi.get_filtration_value(*sh_multi).swap((_multifiltration(std::move(edge_degree_rips_filtration))));
    if constexpr (verbose) std::cout << "\n" << st_multi.get_filtration_value(*sh_multi) << std::endl;
    // std::reverse(edge_degree_rips_filtration.begin(), edge_degree_rips_filtration.end());

    // std::cout <<  st_multi.get_filtration_value(*sh_multi) <<std::endl;

    // edge_degree_rips_filtration.clea
    // std::cout <<  st_multi.get_filtration_value(*sh_multi) <<std::endl;
  }

  // fills the dimension 0 simplices
  for (auto vertex : st_multi.complex_vertex_range()) {  // should be in increasing order
    auto &vertex_filtration = st_multi.get_filtration_value(st_multi.find({vertex}));
    if constexpr (verbose) {
      std::cout << "Setting filtration of node " << vertex << " to ";
      for (auto degree_index = 0u; degree_index < num_degrees; degree_index++) {
        std::cout << edge_filtration_of_nodes[vertex][degree_index] << " ";
      }
      std::cout << "\n";
    }

    if (edge_filtration_of_nodes[vertex].size() > num_degrees)
      throw std::overflow_error("Edge filtration of node of bad size");

    vertex_filtration.set_num_generators(num_degrees);
    for (auto k = 0u; k < num_degrees; k++) {
      vertex_filtration(k, 0) = edge_filtration_of_nodes[vertex][k];
    }
    if constexpr (verbose) {
      std::cout << "Filtration of node " << std::to_string(vertex) << " :\n" << vertex_filtration << "\n";
    }
  }
  if constexpr (verbose) std::cout << std::endl;

  return st_multi;
}

// assumes that the degree is 1
inline void fill_st_slice(Simplex_tree_std &st_container, flat_multi_st &degree_rips_st, int degree) {
  auto sh_std = st_container.complex_simplex_range().begin();
  auto sh_multi = degree_rips_st.complex_simplex_range().begin();
  auto sh_end = st_container.complex_simplex_range().end();
  for (; sh_std != sh_end; ++sh_std, ++sh_multi) {
    value_type splx_filtration = degree_rips_st.get_filtration_value(*sh_multi)(0, degree);
    st_container.assign_filtration(*sh_std, splx_filtration);
  }
}

template <typename dtype, typename index_type>
inline void compute_2d_function_rips(
    flat_multi_st &st_multi,                                   // Function rips
                                                               // Simplex_tree_std &_st,
    const tensor::static_tensor_view<dtype, index_type> &out,  // assumes its a zero tensor
    const std::vector<index_type> degrees,
    index_type I,
    index_type J,  // grid_shape
    bool mobius_inverion,
    bool zero_pad) {
  constexpr bool verbose = false;
  if constexpr (verbose) std::cout << "Grid shape : " << I << " " << J << std::endl;

  // inits default simplextrees
  // copies the st_multi to a standard 1-pers simplextree, and puts its filtration values to 0 for all.
  Simplex_tree_std _st(
      st_multi,
      []([[maybe_unused]] const flat_multifiltration &f) -> Simplex_tree_std::Filtration_value { return 0.; });
  tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree(_st);
  int max_simplex_dimension = *std::max_element(degrees.begin(), degrees.end()) + 1;
  tbb::parallel_for(0, J, [&](index_type function_value) {
    auto &st_std = thread_simplex_tree.local();
    fill_st_slice(st_std, st_multi, function_value);
    constexpr int num_collapses = 10;
    const std::vector<Barcode> barcodes = compute_dgms(st_std, degrees, num_collapses, max_simplex_dimension);

    index_type degree_index = 0;
    for (const auto &barcode : barcodes) {  // TODO range view cartesian product
      // coordinates_container[0] = degree_index;
      for (const auto &bar : barcode) {
        auto birth = bar.first;  // float
        auto death = bar.second;
        if (birth > I)  // some birth can be infinite
          continue;

        if (!mobius_inverion) {
          index_type shift_value = out.get_cum_resolution()[1];  // degree, x coord, y coord
          index_type border = I;
          // index_type border  = out.get_resolution()[i+1];
          // dtype* ptr = &out[coordinates_container];
          dtype *ptr = &out[{degree_index, static_cast<index_type>(birth), function_value}];
          auto stop_value = death > static_cast<value_type>(border) ? border : static_cast<index_type>(death);
          // Warning : for some reason linux static casts float inf to -min_int
          // so min doesnt work.
          if constexpr (verbose) {
            std::cout << "Adding : (";
            // for (auto stuff : coordinates_container) std::cout << stuff << ",
            // ";
            std::cout << ") With death " << death << " casted at " << static_cast<index_type>(death)
                      << "with threshold at" << stop_value << " with " << border << std::endl;
          }
          for (index_type b = birth; b < stop_value; b++) {
            (*ptr)++;            // adds one to the vector
            ptr += shift_value;  // shift the pointer to the next element in the
                                 // segment [birth, death]
          }
        } else {
          // adds birth
          out[{degree_index, static_cast<index_type>(birth), function_value}]++;
          if constexpr (verbose) {
            std::cout << "Coordinate : ";
            // for (auto c : coordinates_container) std::cout << c << ", ";
            std::cout << std::endl;
            std::cout << "axis, death, resolution : " << 1 << ", " << std::to_string(death) << ", "
                      << out.get_resolution()[1];
            std::cout << std::endl;
          }
          // removes death
          if (death < I) {
            out[{degree_index, static_cast<index_type>(death), function_value}]--;
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

std::pair<std::vector<value_type>, int> inline get_degree_rips_st_python(const intptr_t simplextree_ptr,
                                                                         const intptr_t st_multi_ptr,
                                                                         const std::vector<int> &degrees) {
  auto &st_std = python_interface::get_simplextree_from_pointer<python_interface::interface_std>(simplextree_ptr);
  auto &st_multi_python_container = python_interface::get_simplextree_from_pointer<flat_multi_st>(st_multi_ptr);
  auto st_multi= get_degree_filtrations(st_std, degrees);
  st_multi_python_container = std::move(st_multi);
  return {{}, 0};
}

template <typename dtype, typename indices_type>
void compute_function_rips_surface_python(const intptr_t st_multi_ptr,
                                          dtype *data_ptr,
                                          const std::vector<indices_type> degrees,
                                          indices_type I,
                                          indices_type J,
                                          const bool mobius_inversion = false,
                                          const bool zero_pad = false,
                                          indices_type n_jobs = 0) {
  if (degrees.size() == 0) return;
  // const bool verbose = false;
  auto &st_multi = python_interface::get_simplextree_from_pointer<flat_multi_st>(st_multi_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(data_ptr,
                                                            {static_cast<indices_type>(degrees.size()), I, J});
  if (zero_pad) {
    I--;
    J--;
  }

  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] { compute_2d_function_rips(st_multi, container, degrees, I, J, mobius_inversion, zero_pad); });
  if (mobius_inversion) container.differentiate(2);  // degree,x axis (already inversed), y axis
}

template <typename dtype, typename indices_type>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> compute_function_rips_signed_measure_python(
    const intptr_t st_multi_ptr,
    dtype *data_ptr,
    const std::vector<indices_type> degrees,
    indices_type I,
    indices_type J,
    const bool mobius_inversion = false,
    const bool zero_pad = false,
    indices_type n_jobs = 0) {
  if (degrees.size() == 0) return {{}, {}};
  // const bool verbose = false;
  auto &st_multi = python_interface::get_simplextree_from_pointer<interface_multi>(st_multi_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(data_ptr,
                                                            {static_cast<indices_type>(degrees.size()), I, J});
  if (zero_pad) {
    I--;
    J--;
  }

  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] { compute_2d_function_rips(st_multi, container, degrees, I, J, mobius_inversion, zero_pad); });
  if (mobius_inversion) container.differentiate(2);  // degree,x axis (already inversed), y axis
  return container.sparsify();
}

}  // namespace function_rips
}  // namespace multiparameter
}  // namespace Gudhi

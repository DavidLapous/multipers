#pragma once

#include "Simplex_tree_multi_interface.h"
#include "multi_parameter_rank_invariant/persistence_slices.h"
#include "tensor/tensor.h"
#include <algorithm>
#include <gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h>
#include <iostream>
#include <limits>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <ostream>
#include <utility> // std::pair
#include <vector>
#include <gudhi/truc.h>

namespace Gudhi::multiparameter::hilbert_function {

// TODO : this function is ugly
template <typename value_type, typename indices_type>
inline typename Simplex_tree_std::Filtration_value horizontal_line_filtration2(
    const std::vector<value_type> &x, indices_type height, indices_type i,
    indices_type j, const std::vector<indices_type> &fixed_values) {
  constexpr const Simplex_tree_std::Filtration_value inf = std::numeric_limits<typename Simplex_tree_std::Filtration_value>::infinity();
  for (indices_type k = 0u; k < static_cast<indices_type>(x.size()); k++) {
    if (k == i || k == j)
      continue;                 // coordinate in the plane
    if (x[k] > fixed_values[k]) // simplex appears after the plane
      return inf;
  }
  if (x[j] <= height) // simplex apppears in the plane, but is it in the line
                      // with height "height"
    return x[i];
  else
    return inf;
}

template <typename index_type>
using hilbert_thread_data = tbb::enumerable_thread_specific<
    std::pair<Simplex_tree_std, std::vector<index_type>>>;
template <typename Filtration, typename dtype, typename index_type>
inline void compute_2d_hilbert_surface(
    python_interface::Simplex_tree_multi_interface<Filtration> &st_multi,
    // Simplex_tree_std &_st,
    hilbert_thread_data<index_type> &thread_simplex_tree,
    const tensor::static_tensor_view<dtype, index_type>
        &out, // assumes its a zero tensor
    const std::vector<index_type> grid_shape,
    const std::vector<index_type> degrees, index_type i, index_type j,
    const std::vector<index_type> fixed_values, bool mobius_inverion,
    bool zero_pad, int expand_collapse_dim = 0) {
  using value_type = typename Filtration::value_type;
  // if (grid_shape.size() < 2 || st_multi.get_number_of_parameters() < 2)
  // 	throw std::invalid_argument("Grid shape has to have at least 2
  // element."); if (st_multi.get_number_of_parameters() - fixed_values.size()
  // != 2) 	throw std::invalid_argument("Fix more values for the
  // simplextree, which has a too big number of parameters");
  // assert(fixed_values.size() == st_multi.get_number_of_parameters());

  constexpr bool verbose = false;
  index_type I = grid_shape[i + 1], J = grid_shape[j + 1];
  if constexpr (verbose)
    std::cout << "Grid shape : " << I << " " << J << std::endl;

  // grid2d out(I, std::vector<int>(J,0)); // zero of good size
  // std::vector<std::vector<index_type>> free_coordinates(grid_shape.size());
  // for (auto [r,k] = std::views::zip(free_coordinates,
  // std::ranges::iota(0,free_coordinates.size()))){ 	if (k==i) r =
  // std::ranges::iota(0,I); 	else if (k==j) r= std::ranges::iota(0,J);
  // else fixed_values[k];
  // }
  // tensor::static_tensor_view_view<value_type, index_type> dim2_view(out,
  // free_coordinates); auto coordinates_container =

  // Simplex_tree_std _st;
  // flatten<Simplex_tree_float,
  // Simplex_tree_options_multidimensional_filtration>(_st, st_multi,-1); //
  // copies the st_multi to a standard 1-pers simplextree
  // tbb::enumerable_thread_specific<std::pair<Simplex_tree_std,
  // std::vector<index_type>>> thread_simplex_tree;
  tbb::parallel_for(0, J, [&](index_type height) {
    // SIMPLEXTREE INIT
    auto &[st_std, coordinates_container] = thread_simplex_tree.local();
    // if (st_std.num_simplices() == 0){ st_std = _st;}
    // COORDINATES INIT
    // if (coordinates_container.size() == 0) {
    // 	//This init is fine as only the j+1th coord is touched
    // 	coordinates_container.reserve(fixed_values.size()+1);
    // 	coordinates_container.push_back(0); // degree
    // 	for (auto c : fixed_values) coordinates_container.push_back(c);
    // }
    // coordinates_container.resize(fixed_values.size()+1); // Not necessary

    // if (coordinates_container.size() != fixed_values.size()+1 ||
    // st_std.num_simplices() == 0){ 	throw std::runtime_error("Bad tbb thread
    // local storage initialization.");
    // }
    // Coordinate initialization to fixed values
    // coordinates_container[0] = 0; // not necessary
    // for (auto [c, i] : std::views::zip(fixed_values, std::views::iota(0u,
    // fixed_values.size()))) // NIK APPLE CLANG coordinates_container[i+1] = c;
    for (auto i = 0u; i < fixed_values.size(); i++)
      coordinates_container[i + 1] = fixed_values[i];

    coordinates_container[j + 1] = height;

    Filtration multi_filtration(
        st_multi.get_number_of_parameters());
    auto sh_standard = st_std.complex_simplex_range().begin();
    auto _end = st_std.complex_simplex_range().end();
    auto sh_multi = st_multi.complex_simplex_range().begin();
    for (; sh_standard != _end; ++sh_multi, ++sh_standard) {
      // for (auto [sh_standard, sh_multi] :
      // std::ranges::views::zip(st_std.complex_simplex_range(),
      // st_multi.complex_simplex_range())){ // too bad apple clang exists
      multi_filtration = st_multi.filtration(*sh_multi);
typename Simplex_tree_std::Filtration_value horizontal_filtration ;
      if constexpr (Filtration::is_multi_critical)
      {
        horizontal_filtration = std::numeric_limits<typename Simplex_tree_std::Filtration_value>::infinity();
        for (const auto& stuff : multi_filtration)
          horizontal_filtration = std::min(
              horizontal_filtration,
              horizontal_line_filtration2(stuff, height, i, j, fixed_values));
      }
      else
        {horizontal_filtration = horizontal_line_filtration2(
            multi_filtration, height, i, j, fixed_values);}
      st_std.assign_filtration(*sh_standard, horizontal_filtration);

      if constexpr (verbose) {
        multi_filtrations::Finitely_critical_multi_filtration<int> splx;
        for (auto vertex : st_multi.simplex_vertex_range(*sh_multi))
          splx.push_back(vertex);
        std::cout << "Simplex " << splx << "/" << st_std.num_simplices()
                  << " Filtration multi " << st_multi.filtration(*sh_multi)
                  << " Filtration 1d " << st_std.filtration(*sh_standard)
                  << "\n";
      }
    }

    if constexpr (verbose) {
      std::cout << "Coords : " << height << " [";
      for (auto stuff : fixed_values)
        std::cout << stuff << " ";
      std::cout << "]" << std::endl;
    }
    const std::vector<Barcode> barcodes =
        compute_dgms(st_std, degrees, expand_collapse_dim);
    index_type degree_index = 0;
    for (const auto &barcode : barcodes) { // TODO range view cartesian product
      coordinates_container[0] = degree_index;
      for (const auto &bar : barcode) {
        auto birth = bar.first; // float
        auto death = bar.second;
        // if constexpr (verbose) std::cout << "BEFORE " << birth << " " <<
        // death << " " << I << " \n"; death = death > I ? I : death; // TODO
        // FIXME if constexpr (verbose) std::cout <<"AFTER" << birth << " " <<
        // death << " " << I << " \n";
        if (birth > I) // some birth can be infinite
          continue;

        if (!mobius_inverion) {
          // throw std::logic_error("Not implemented");
          // death = death > I ? I : death;
          // for (int index = static_cast<int>(birth); index <
          // static_cast<int>(death); index ++){
          // 	out[degree_index][index][height]++;
          // }

          // Seems to bug on linux ????

          coordinates_container[i + 1] = static_cast<index_type>(birth);
          index_type shift_value = out.get_cum_resolution()[i + 1];
          index_type border = I;
          // index_type border  = out.get_resolution()[i+1];
          dtype *ptr = &out[coordinates_container];
          auto stop_value = death > static_cast<value_type>(border)
                                ? border
                                : static_cast<index_type>(death);
          // Warning : for some reason linux static casts float inf to -min_int
          // so min doesnt work.
          if constexpr (verbose) {
            std::cout << "Adding : (";
            for (auto stuff : coordinates_container)
              std::cout << stuff << ", ";
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
          // out[degree_index][static_cast<int>(birth)][height]++; // No need to
          // do mobius inversion on this axis, it can be done here if (death <
          // I) 	out[degree_index][static_cast<int>(death)][height]--;
          // else if (zero_pad)
          // {
          // 	out[degree_index].back()[height]--;
          // }
          // coordinates_container[0] = degree_index;
          coordinates_container[i + 1] = static_cast<index_type>(birth);
          out[coordinates_container]++;

          if constexpr (verbose) {
            std::cout << "Coordinate : ";
            for (auto c : coordinates_container)
              std::cout << c << ", ";
            std::cout << std::endl;
            std::cout << "axis, death, resolution : " << i + 1 << ", "
                      << std::to_string(death) << ", "
                      << out.get_resolution()[i + 1];
            std::cout << std::endl;
          }

          if (death < I) {
            coordinates_container[i + 1] = static_cast<index_type>(death);
            out[coordinates_container]--;
          } else if (zero_pad) {
            coordinates_container[i + 1] = I - 1;
            out[coordinates_container]--;
          }
        }
        // else
        // 	out[I-1][height]--;
      }
      degree_index++;
    }
  });
  return;
}

template <typename Filtration, typename dtype, typename index_type>
void _rec_get_hilbert_surface(
    python_interface::Simplex_tree_multi_interface<Filtration> &st_multi,
    // Simplex_tree_std &_st,
    hilbert_thread_data<index_type> &thread_simplex_tree,
    const tensor::static_tensor_view<dtype, index_type>
        &out, // assumes its a zero tensor
    const std::vector<index_type> grid_shape,
    const std::vector<index_type> degrees,
    std::vector<index_type> coordinates_to_compute,
    const std::vector<index_type> fixed_values, bool mobius_inverion = true,
    bool zero_pad = false, int expand_collapse_dim = 0) {
  constexpr bool verbose = false;

  if constexpr (verbose) {
    std::cout << "Computing coordinates (";
    for (auto c : coordinates_to_compute)
      std::cout << c << ", ";
    std::cout << "). with fixed values (";
    for (auto c : fixed_values) {
      std::cout << c << ", ";
    }
    std::cout << ")." << std::endl;
  }
  if (coordinates_to_compute.size() == 2) {
    compute_2d_hilbert_surface(st_multi,
                               // _st,
                               thread_simplex_tree,
                               out, // assumes its a zero tensor
                               grid_shape, degrees, coordinates_to_compute[0],
                               coordinates_to_compute[1], fixed_values,
                               mobius_inverion, zero_pad, expand_collapse_dim);
    return;
  }

  // coordinate to iterate.size --
  auto coordinate_to_iterate = coordinates_to_compute.back();
  coordinates_to_compute.pop_back();
  tbb::parallel_for(
      0, grid_shape[coordinate_to_iterate + 1], [&](index_type z) {
        // Updates fixes values that defines the slice
        std::vector<index_type> _fixed_values =
            fixed_values; // TODO : do not copy this //thread local
        _fixed_values[coordinate_to_iterate] = z;
        _rec_get_hilbert_surface(st_multi, thread_simplex_tree, out, grid_shape,
                                 degrees, coordinates_to_compute, _fixed_values,
                                 mobius_inverion, zero_pad,
                                 expand_collapse_dim);
      });
  // rmq : with mobius_inversion + rec, the coordinates to compute size is 2 =>
  // first coord is always the initial 1st coord.
  // => inversion is only needed for coords > 2
}

template <typename Filtration, typename dtype, typename index_type>
void get_hilbert_surface(python_interface::Simplex_tree_multi_interface<Filtration> &st_multi,
                         const tensor::static_tensor_view<dtype, index_type>
                             &out, // assumes its a zero tensor
                         const std::vector<index_type> &grid_shape,
                         const std::vector<index_type> &degrees,
                         std::vector<index_type> coordinates_to_compute,
                         const std::vector<index_type> &fixed_values,
                         bool mobius_inverion = true, bool zero_pad = false,
                         bool expand_collapse = false) {
  if (degrees.size() == 0)
    return;
  // wrapper arount the rec version, that initialize the thread variables.
  if (coordinates_to_compute.size() < 2)
    throw std::logic_error("Not implemented for " +
                           std::to_string(coordinates_to_compute.size()) +
                           "<2 parameters.");

  Simplex_tree_std _st;
  flatten(_st, st_multi,
          -1); // copies the st_multi to a standard 1-pers simplextree
  std::vector<index_type> coordinates_container(
      st_multi.get_number_of_parameters() + 1); // +1 for degree
  // coordinates_container.reserve(fixed_values.size()+1);
  // coordinates_container.push_back(0); // degree
  // for (auto c : fixed_values) coordinates_container.push_back(c);
  std::pair<Simplex_tree_std, std::vector<index_type>>
      thread_data_initialization = {_st, coordinates_container};
  const int max_dim =
      expand_collapse ? *std::max_element(degrees.begin(), degrees.end()) + 1
                      : 0;
  tbb::enumerable_thread_specific<
      std::pair<Simplex_tree_std, std::vector<index_type>>>
      thread_simplex_tree(
          thread_data_initialization); // this has a fixed size, so init should
                                       // be benefic
  _rec_get_hilbert_surface(st_multi, thread_simplex_tree, out, grid_shape,
                           degrees, coordinates_to_compute, fixed_values,
                           mobius_inverion, zero_pad, max_dim);
}

template <typename Filtration, typename dtype = int, typename indices_type>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>>
get_hilbert_signed_measure(python_interface::Simplex_tree_multi_interface<Filtration> &st_multi, dtype *data_ptr,
                           std::vector<indices_type> grid_shape,
                           const std::vector<indices_type> degrees,
                           bool zero_pad = false, indices_type n_jobs = 0,
                           const bool verbose = false,
                           const bool expand_collapse = false) {
  if (degrees.size() == 0)
    return {{}, {}};
  // const bool verbose = false;
  // auto &st_multi =
  //     get_simplextree_from_pointer<python_interface::interface_multi<Filtration>>(simplextree_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(
      data_ptr, grid_shape); // assumes its a zero tensor
  std::vector<indices_type> coordinates_to_compute(
      st_multi.get_number_of_parameters());
  for (auto i = 0u; i < coordinates_to_compute.size(); i++)
    coordinates_to_compute[i] = i;
  // for (auto [c,i] : std::views::zip(coordinates_to_compute,
  // std::views::iota(0,st_multi.get_number_of_parameters()))) c=i; // NIK apple
  // clang
  std::vector<indices_type> fixed_values(st_multi.get_number_of_parameters());

  if (verbose) {
    std::cout << "Container shape : ";
    for (auto r : container.get_resolution())
      std::cout << r << ", ";
    std::cout << "\nContainer size : " << container.size();
    std::cout << "\nComputing hilbert invariant ..." << std::flush;
  }
  if (zero_pad) {
    // +1 is bc degree is on first axis.
    for (auto i = 1; i < st_multi.get_number_of_parameters() + 1; i++)
      grid_shape[i]--; // get hilbert surface computes according to grid_shape.
    // for (auto i : std::views::iota(1,st_multi.get_number_of_parameters()+1))
    // grid_shape[i]--; // get hilbert surface computes according to grid_shape.
  }

  oneapi::tbb::task_arena arena(n_jobs); // limits the number of threads
  arena.execute([&] {
    get_hilbert_surface(st_multi, container, grid_shape, degrees,
                        coordinates_to_compute, fixed_values, true, zero_pad,
                        expand_collapse);
  });

  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Computing mobius inversion ..." << std::flush;
  }

  // for (indices_type axis :
  // std::views::iota(2,st_multi.get_number_of_parameters()+1)) // +1 for the
  // degree in axis 0
  for (indices_type axis = 2u; axis < st_multi.get_number_of_parameters() + 1;
       axis++)
    container.differentiate(axis);
  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Sparsifying the measure ..." << std::flush;
  }
  auto raw_signed_measure = container.sparsify();
  if (verbose) {
    std::cout << "Done.\n";
  }
  return raw_signed_measure;
}

template <typename Filtration, typename dtype, typename indices_type, typename... Args>
void get_hilbert_surface_python(python_interface::Simplex_tree_multi_interface<Filtration> &st_multi, dtype *data_ptr,
                                std::vector<indices_type> grid_shape,
                                const std::vector<indices_type> degrees,
                                const bool mobius_inversion,
                                const bool zero_pad, indices_type n_jobs,
                                bool expand_collapse) {
  const bool verbose = false;
  if (degrees.size() == 0)
    return;
  // const bool verbose = false;
  // auto &st_multi =
  //     get_simplextree_from_pointer<python_interface::interface_multi<Filtration>>(simplextree_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(
      data_ptr, grid_shape); // assumes its a zero tensor
  std::vector<indices_type> coordinates_to_compute(
      st_multi.get_number_of_parameters());
  for (auto i = 0u; i < coordinates_to_compute.size(); i++)
    coordinates_to_compute[i] = i;
  // for (auto [c,i] : std::views::zip(coordinates_to_compute,
  // std::views::iota(0,st_multi.get_number_of_parameters()))) c=i; // NIK apple
  // clang
  std::vector<indices_type> fixed_values(st_multi.get_number_of_parameters());

  if (verbose) {
    std::cout << "Container shape : ";
    for (auto r : container.get_resolution())
      std::cout << r << ", ";
    std::cout << "\nContainer size : " << container.size();
    std::cout << "\nComputing hilbert invariant ...";
  }
  if (zero_pad) {
    // +1 is bc degree is on first axis.
    for (auto i = 1; i < st_multi.get_number_of_parameters() + 1; i++)
      grid_shape[i]--; // get hilbert surface computes according to grid_shape.
    // for (auto i : std::views::iota(1,st_multi.get_number_of_parameters()+1))
    // grid_shape[i]--; // get hilbert surface computes according to grid_shape.
  }

  oneapi::tbb::task_arena arena(n_jobs); // limits the number of threads
  arena.execute([&] {
    get_hilbert_surface(st_multi, container, grid_shape, degrees,
                        coordinates_to_compute, fixed_values, mobius_inversion,
                        zero_pad, expand_collapse);
  });

  if (mobius_inversion)
    for (indices_type axis = 2u; axis < st_multi.get_number_of_parameters() + 1;
         axis++)
      container.differentiate(axis);
  return;
}



/// FROM SLICER
///
///

template <typename PersBackend, typename Structure, typename Filtration,
          typename dtype, typename indices_type, typename... Args>
void get_hilbert_surface_python(
    interface::Truc<PersBackend, Structure, Filtration> &slicer,
    dtype *data_ptr, std::vector<indices_type> grid_shape,
    const std::vector<indices_type> degrees, const bool mobius_inversion,
    const bool zero_pad, indices_type n_jobs, bool expand_collapse) {
  const bool verbose = false;
  if (degrees.size() == 0)
    return;
  // const bool verbose = false;
  // auto &st_multi =
  //     get_simplextree_from_pointer<python_interface::interface_multi<Filtration>>(simplextree_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(
      data_ptr, grid_shape); // assumes its a zero tensor
  std::size_t num_parameters = slicer.num_parameters();
  std::vector<indices_type> coordinates_to_compute(num_parameters);
  for (auto i = 0u; i < coordinates_to_compute.size(); i++)
    coordinates_to_compute[i] = i;
  // for (auto [c,i] : std::views::zip(coordinates_to_compute,
  // std::views::iota(0,st_multi.get_number_of_parameters()))) c=i; // NIK apple
  // clang
  std::vector<indices_type> fixed_values(num_parameters);

  if (verbose) {
    std::cout << "Container shape : ";
    for (auto r : container.get_resolution())
      std::cout << r << ", ";
    std::cout << "\nContainer size : " << container.size();
    std::cout << "\nComputing hilbert invariant ...";
  }
  if (zero_pad) {
    // +1 is bc degree is on first axis.
    for (auto i = 1; i < num_parameters + 1; i++)
      grid_shape[i]--; // get hilbert surface computes according to grid_shape.
    // for (auto i : std::views::iota(1,st_multi.get_number_of_parameters()+1))
    // grid_shape[i]--; // get hilbert surface computes according to grid_shape.
  }

  oneapi::tbb::task_arena arena(n_jobs); // limits the number of threads
  arena.execute([&] {
    get_hilbert_surface(slicer, container, grid_shape, degrees,
                        coordinates_to_compute, fixed_values, mobius_inversion,
                        zero_pad, expand_collapse);
  });

  if (mobius_inversion)
    for (indices_type axis = 2u; axis < num_parameters+ 1;
         axis++)
      container.differentiate(axis);
  return;
}

} // namespace Gudhi::multiparameter::hilbert_function

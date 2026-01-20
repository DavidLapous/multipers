#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <ostream>
#include <utility>  // std::pair
#include <vector>

#include "../gudhi/Simplex_tree_multi_interface.h"
#include "../gudhi/gudhi/Slicer.h"
#include "../tensor/tensor.h"
#include "persistence_slices.h"

namespace Gudhi {
namespace multiparameter {
namespace hilbert_function {

// TODO : this function is ugly
template <typename Filtration, typename indices_type>
inline typename Filtration::value_type horizontal_line_filtration2(const Filtration &x,
                                                                   unsigned int gen_index,
                                                                   indices_type height,
                                                                   indices_type i,
                                                                   indices_type j,
                                                                   const std::vector<indices_type> &fixed_values) {
  const auto &inf = Filtration::T_inf;
  for (indices_type k = 0u; k < static_cast<indices_type>(x.num_parameters()); k++) {
    if (k == i || k == j) continue;         // coordinate in the plane
    if (x(gen_index, k) > fixed_values[k])  // simplex appears after the plane
      return inf;
  }
  // simplex appears in the plane, but is it in the line with height "height"
  if (x(gen_index, j) <= height) return x(gen_index, i);
  return inf;
}

template <typename index_type>
using hilbert_thread_data = tbb::enumerable_thread_specific<std::pair<Simplex_tree_std, std::vector<index_type>>>;

template <typename Filtration, typename dtype, typename index_type>
inline void compute_2d_hilbert_surface(
    python_interface::Simplex_tree_multi_interface<Filtration> &st_multi,
    // Simplex_tree_std &_st,
    hilbert_thread_data<index_type> &thread_simplex_tree,
    const tensor::static_tensor_view<dtype, index_type> &out,  // assumes its a zero tensor
    const std::vector<index_type> grid_shape,
    const std::vector<index_type> degrees,
    index_type i,
    index_type j,
    const std::vector<index_type> fixed_values,
    bool mobius_inverion,
    bool zero_pad,
    int expand_collapse_dim = 0) {
  using value_type = typename Filtration::value_type;

  constexpr const bool verbose = false;
  index_type I = grid_shape[i + 1], J = grid_shape[j + 1];
  if constexpr (verbose) std::cout << "Grid shape : " << I << " " << J << std::endl;

  tbb::parallel_for(0, J, [&](index_type height) {
    // SIMPLEXTREE INIT
    //
    tbb::task_arena arena(1);
    arena.execute([&] {
      auto &[st_std, coordinates_container] = thread_simplex_tree.local();
      for (auto i = 0u; i < fixed_values.size(); i++) coordinates_container[i + 1] = fixed_values[i];

      coordinates_container[j + 1] = height;

      // Filtration multi_filtration(st_multi.num_parameters());
      auto sh_standard = st_std.complex_simplex_range().begin();
      auto _end = st_std.complex_simplex_range().end();
      auto sh_multi = st_multi.complex_simplex_range().begin();
      for (; sh_standard != _end; ++sh_multi, ++sh_standard) {
        // for (auto [sh_standard, sh_multi] :
        // std::ranges::views::zip(st_std.complex_simplex_range(),
        // st_multi.complex_simplex_range())){ // too bad apple clang exists
        Filtration multi_filtration = st_multi.filtration(*sh_multi);
        typename Simplex_tree_std::Filtration_value horizontal_filtration;
        horizontal_filtration = std::numeric_limits<typename Simplex_tree_std::Filtration_value>::infinity();
        for (unsigned int g = 0; g < multi_filtration.num_generators(); ++g) {
          horizontal_filtration = std::min(horizontal_filtration,
                                           static_cast<Simplex_tree_std::Filtration_value>(horizontal_line_filtration2(
                                               multi_filtration, g, height, i, j, fixed_values)));
        }
        st_std.assign_filtration(*sh_standard, horizontal_filtration);

        if constexpr (verbose) {
          std::cout << "Simplex {";
          for (auto vertex : st_multi.simplex_vertex_range(*sh_multi)) std::cout << vertex << " ";
          std::cout << "} / " << st_std.num_simplices() << " Filtration multi " << st_multi.filtration(*sh_multi)
                    << " Filtration 1d " << st_std.filtration(*sh_standard) << "\n";
        }
      }

      if constexpr (verbose) {
        std::cout << "Coords : " << height << " [";
        for (auto stuff : fixed_values) std::cout << stuff << " ";
        std::cout << "]" << std::endl;
      }
      const std::vector<Barcode> barcodes = compute_dgms(st_std, degrees, expand_collapse_dim);
      index_type degree_index = 0;
      for (const auto &barcode : barcodes) {  // TODO range view cartesian product
        coordinates_container[0] = degree_index;
        for (const auto &bar : barcode) {
          auto birth = bar.first;  // float
          auto death = bar.second;
          if (birth > I)  // some birth can be infinite
            continue;

          if (!mobius_inverion) {
            coordinates_container[i + 1] = static_cast<index_type>(birth);
            index_type shift_value = out.get_cum_resolution()[i + 1];
            index_type border = I;
            // index_type border  = out.get_resolution()[i+1];
            dtype *ptr = &out[coordinates_container];
            auto stop_value = death > static_cast<value_type>(border) ? border : static_cast<index_type>(death);
            // Warning : for some reason linux static casts float inf to -min_int
            // so min doesn't work.
            if constexpr (verbose) {
              std::cout << "Adding : (";
              for (auto stuff : coordinates_container) std::cout << stuff << ", ";
              std::cout << ") With death " << death << " casted at " << static_cast<index_type>(death)
                        << "with threshold at" << stop_value << " with " << border << std::endl;
            }
            for (index_type b = birth; b < stop_value; b++) {
              (*ptr)++;            // adds one to the vector
              ptr += shift_value;  // shift the pointer to the next element in the
                                   // segment [birth, death]
            }
          } else {
            coordinates_container[i + 1] = static_cast<index_type>(birth);
            out[coordinates_container]++;

            if constexpr (verbose) {
              std::cout << "Coordinate : ";
              for (auto c : coordinates_container) std::cout << c << ", ";
              std::cout << std::endl;
              std::cout << "axis, death, resolution : " << i + 1 << ", " << std::to_string(death) << ", "
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
        }
        degree_index++;
      }
    });
  });
  return;
}

template <typename Filtration, typename dtype, typename index_type>
void _rec_get_hilbert_surface(python_interface::Simplex_tree_multi_interface<Filtration> &st_multi,
                              // Simplex_tree_std &_st,
                              hilbert_thread_data<index_type> &thread_simplex_tree,
                              const tensor::static_tensor_view<dtype, index_type> &out,  // assumes its a zero tensor
                              const std::vector<index_type> grid_shape,
                              const std::vector<index_type> degrees,
                              std::vector<index_type> coordinates_to_compute,
                              const std::vector<index_type> fixed_values,
                              bool mobius_inverion = true,
                              bool zero_pad = false,
                              int expand_collapse_dim = 0) {
  constexpr const bool verbose = false;

  if constexpr (verbose) {
    std::cout << "Computing coordinates (";
    for (auto c : coordinates_to_compute) std::cout << c << ", ";
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
                               out,  // assumes its a zero tensor
                               grid_shape,
                               degrees,
                               coordinates_to_compute[0],
                               coordinates_to_compute[1],
                               fixed_values,
                               mobius_inverion,
                               zero_pad,
                               expand_collapse_dim);
    return;
  }

  // coordinate to iterate.size --
  auto coordinate_to_iterate = coordinates_to_compute.back();
  coordinates_to_compute.pop_back();
  tbb::parallel_for(0, grid_shape[coordinate_to_iterate + 1], [&](index_type z) {
    // Updates fixes values that defines the slice
    std::vector<index_type> _fixed_values = fixed_values;  // TODO : do not copy this //thread local
    _fixed_values[coordinate_to_iterate] = z;
    _rec_get_hilbert_surface(st_multi,
                             thread_simplex_tree,
                             out,
                             grid_shape,
                             degrees,
                             coordinates_to_compute,
                             _fixed_values,
                             mobius_inverion,
                             zero_pad,
                             expand_collapse_dim);
  });
  // rmq : with mobius_inversion + rec, the coordinates to compute size is 2 =>
  // first coord is always the initial 1st coord.
  // => inversion is only needed for coords > 2
}

template <typename Filtration, typename dtype, typename index_type>
void get_hilbert_surface(python_interface::Simplex_tree_multi_interface<Filtration> &st_multi,
                         const tensor::static_tensor_view<dtype, index_type> &out,  // assumes its a zero tensor
                         const std::vector<index_type> &grid_shape,
                         const std::vector<index_type> &degrees,
                         std::vector<index_type> coordinates_to_compute,
                         const std::vector<index_type> &fixed_values,
                         bool mobius_inverion = true,
                         bool zero_pad = false,
                         bool expand_collapse = false) {
  if (degrees.size() == 0) return;
  // wrapper arount the rec version, that initialize the thread variables.
  if (coordinates_to_compute.size() < 2)
    throw std::logic_error("Not implemented for " + std::to_string(coordinates_to_compute.size()) + "<2 parameters.");

  // copies the st_multi to a standard 1-pers simplextree without the filtration values
  Simplex_tree_std _st(st_multi,
                       []([[maybe_unused]] const Filtration &f) -> Simplex_tree_std::Filtration_value { return 0.; });
  std::vector<index_type> coordinates_container(st_multi.num_parameters() + 1);  // +1 for degree
  // coordinates_container.reserve(fixed_values.size()+1);
  // coordinates_container.push_back(0); // degree
  // for (auto c : fixed_values) coordinates_container.push_back(c);
  std::pair<Simplex_tree_std, std::vector<index_type>> thread_data_initialization = {_st, coordinates_container};
  const int max_dim = expand_collapse ? *std::max_element(degrees.begin(), degrees.end()) + 1 : 0;
  tbb::enumerable_thread_specific<std::pair<Simplex_tree_std, std::vector<index_type>>> thread_simplex_tree(
      thread_data_initialization);  // this has a fixed size, so init should
                                    // be benefic
  _rec_get_hilbert_surface(st_multi,
                           thread_simplex_tree,
                           out,
                           grid_shape,
                           degrees,
                           coordinates_to_compute,
                           fixed_values,
                           mobius_inverion,
                           zero_pad,
                           max_dim);
}

template <typename Filtration, typename dtype = int, typename indices_type>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> get_hilbert_signed_measure(
    python_interface::Simplex_tree_multi_interface<Filtration> &st_multi,
    dtype *data_ptr,
    std::vector<indices_type> grid_shape,
    const std::vector<indices_type> degrees,
    bool zero_pad = false,
    indices_type n_jobs = 0,
    const bool verbose = false,
    const bool expand_collapse = false) {
  if (degrees.size() == 0) return {{}, {}};
  // const bool verbose = false;
  // auto &st_multi =
  //     get_simplextree_from_pointer<python_interface::interface_multi<Filtration>>(simplextree_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(data_ptr, grid_shape);  // assumes its a zero tensor
  std::vector<indices_type> coordinates_to_compute(st_multi.num_parameters());
  for (auto i = 0u; i < coordinates_to_compute.size(); i++) coordinates_to_compute[i] = i;
  // for (auto [c,i] : std::views::zip(coordinates_to_compute,
  // std::views::iota(0,st_multi.num_parameters()))) c=i; // NIK apple
  // clang
  std::vector<indices_type> fixed_values(st_multi.num_parameters());

  if (verbose) {
    std::cout << "Container shape : ";
    for (auto r : container.get_resolution()) std::cout << r << ", ";
    std::cout << "\nContainer size : " << container.size();
    std::cout << "\nComputing hilbert invariant ..." << std::flush;
  }
  if (zero_pad) {
    // +1 is bc degree is on first axis.
    for (auto i = 1; i < st_multi.num_parameters() + 1; i++)
      grid_shape[i]--;  // get hilbert surface computes according to grid_shape.
    // for (auto i : std::views::iota(1,st_multi.num_parameters()+1))
    // grid_shape[i]--; // get hilbert surface computes according to grid_shape.
  }

  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] {
    get_hilbert_surface(st_multi,
                        container,
                        grid_shape,
                        degrees,
                        coordinates_to_compute,
                        fixed_values,
                        true,
                        zero_pad,
                        expand_collapse);
  });

  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Computing mobius inversion ..." << std::flush;
  }

  // for (indices_type axis :
  // std::views::iota(2,st_multi.num_parameters()+1)) // +1 for the
  // degree in axis 0
  for (indices_type axis = 2u; axis < st_multi.num_parameters() + 1; axis++) container.differentiate(axis);
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
void get_hilbert_surface_python(python_interface::Simplex_tree_multi_interface<Filtration> &st_multi,
                                dtype *data_ptr,
                                std::vector<indices_type> grid_shape,
                                const std::vector<indices_type> degrees,
                                const bool mobius_inversion,
                                const bool zero_pad,
                                indices_type n_jobs,
                                bool expand_collapse) {
  constexpr const bool verbose = false;
  if (degrees.size() == 0) return;
  // const bool verbose = false;
  // auto &st_multi =
  //     get_simplextree_from_pointer<python_interface::interface_multi<Filtration>>(simplextree_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(data_ptr, grid_shape);  // assumes its a zero tensor
  std::vector<indices_type> coordinates_to_compute(st_multi.num_parameters());
  for (auto i = 0u; i < coordinates_to_compute.size(); i++) coordinates_to_compute[i] = i;
  // for (auto [c,i] : std::views::zip(coordinates_to_compute,
  // std::views::iota(0,st_multi.num_parameters()))) c=i; // NIK apple
  // clang
  std::vector<indices_type> fixed_values(st_multi.num_parameters());

  if (verbose) {
    std::cout << "Container shape : ";
    for (auto r : container.get_resolution()) std::cout << r << ", ";
    std::cout << "\nContainer size : " << container.size();
    std::cout << "\nComputing hilbert invariant ...";
  }
  if (zero_pad) {
    // +1 is bc degree is on first axis.
    for (auto i = 1; i < st_multi.num_parameters() + 1; i++)
      grid_shape[i]--;  // get hilbert surface computes according to grid_shape.
    // for (auto i : std::views::iota(1,st_multi.num_parameters()+1))
    // grid_shape[i]--; // get hilbert surface computes according to grid_shape.
  }

  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] {
    get_hilbert_surface(st_multi,
                        container,
                        grid_shape,
                        degrees,
                        coordinates_to_compute,
                        fixed_values,
                        mobius_inversion,
                        zero_pad,
                        expand_collapse);
  });

  if (mobius_inversion)
    for (indices_type axis = 2u; axis < st_multi.num_parameters() + 1; axis++) container.differentiate(axis);
  return;
}

/// FROM SLICER
///
///

template <typename PersBackend, typename Filtration, typename dtype, typename index_type>
inline void compute_2d_hilbert_surface(
    tbb::enumerable_thread_specific<
        std::pair<typename Gudhi::multi_persistence::Slicer<Filtration, PersBackend>::Thread_safe,
                  std::vector<index_type>>> &thread_stuff,
    const tensor::static_tensor_view<dtype, index_type> &out,  // assumes its a zero tensor
    const std::vector<index_type> grid_shape,
    const std::vector<index_type> degrees,
    index_type i,
    index_type j,
    const std::vector<index_type> fixed_values,
    const bool mobius_inverion,
    const bool zero_pad,
    const bool ignore_inf = true) {
  using value_type = typename Filtration::value_type;

  constexpr const bool verbose = false;
  index_type I = grid_shape[i + 1], J = grid_shape[j + 1];
  if constexpr (verbose) std::cout << "Grid shape : " << I << " " << J << std::endl;
  tbb::parallel_for(0, J, [&](index_type height) {
    // SIMPLEXTREE INIT
    auto &[slicer, coordinates_container] = thread_stuff.local();
    for (auto i = 0u; i < fixed_values.size(); i++) coordinates_container[i + 1] = fixed_values[i];

    coordinates_container[j + 1] = height;

    auto &slice_filtration = slicer.get_slice();
    const auto &multi_filtration = slicer.get_filtration_values();

    for (std::size_t k = 0; k < multi_filtration.size(); k++) {
      slice_filtration[k] = Filtration::T_inf;
      for (unsigned int g = 0; g < multi_filtration[k].num_generators(); ++g) {
        slice_filtration[k] = std::min(
            slice_filtration[k],
            static_cast<value_type>(horizontal_line_filtration2(multi_filtration[k], g, height, i, j, fixed_values)));
      }
    }

    if constexpr (verbose) {
      std::cout << "Coords : " << height << " [";
      for (auto stuff : fixed_values) std::cout << stuff << " ";
      std::cout << "]" << std::endl;
    }

    using bc_type =
        typename Gudhi::multi_persistence::Slicer<Filtration, PersBackend>::template Multi_dimensional_flat_barcode<>;

    if (!slicer.persistence_computation_is_initialized()) [[unlikely]] {
      slicer.initialize_persistence_computation(ignore_inf);
    } else {
      slicer.update_persistence_computation(ignore_inf);
    }

    bc_type barcodes = slicer.template get_flat_barcode<true>();
    index_type degree_index = 0;
    for (auto degree : degrees) {  // TODO range view cartesian product
      const auto &barcode = barcodes[degree];
      coordinates_container[0] = degree_index;
      for (const auto &bar : barcode) {
        auto birth = bar[0];  // float
        auto death = bar[1];
        if (birth > I)  // some birth can be infinite
          continue;

        if (!mobius_inverion) {
          coordinates_container[i + 1] = static_cast<index_type>(birth);
          index_type shift_value = out.get_cum_resolution()[i + 1];
          index_type border = I;
          // index_type border  = out.get_resolution()[i+1];
          dtype *ptr = &out[coordinates_container];
          auto stop_value = death > static_cast<value_type>(border) ? border : static_cast<index_type>(death);
          // Warning : for some reason linux static casts float inf to -min_int
          // so min doesnt work.
          if constexpr (verbose) {
            std::cout << "Adding : (";
            for (auto stuff : coordinates_container) std::cout << stuff << ", ";
            std::cout << ") With death " << death << " casted at " << static_cast<index_type>(death)
                      << "with threshold at" << stop_value << " with " << border << std::endl;
          }
          for (index_type b = birth; b < stop_value; b++) {
            (*ptr)++;            // adds one to the vector
            ptr += shift_value;  // shift the pointer to the next element in the
                                 // segment [birth, death]
          }
        } else {
          coordinates_container[i + 1] = static_cast<index_type>(birth);
          out[coordinates_container]++;

          if constexpr (verbose) {
            std::cout << "Coordinate : ";
            for (auto c : coordinates_container) std::cout << c << ", ";
            std::cout << std::endl;
            std::cout << "axis, death, resolution : " << i + 1 << ", " << std::to_string(death) << ", "
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
      }
      degree_index++;
    }
  });
  return;
}

template <typename PersBackend, typename Filtration, typename dtype, typename index_type>
void _rec_get_hilbert_surface(
    tbb::enumerable_thread_specific<
        std::pair<typename Gudhi::multi_persistence::Slicer<Filtration, PersBackend>::Thread_safe,
                  std::vector<index_type>>> &thread_stuff,
    const tensor::static_tensor_view<dtype, index_type> &out,  // assumes its a zero tensor
    const std::vector<index_type> grid_shape,
    const std::vector<index_type> degrees,
    std::vector<index_type> coordinates_to_compute,
    const std::vector<index_type> fixed_values,
    const bool mobius_inverion = true,
    const bool zero_pad = false,
    const bool ignore_inf = true) {
  constexpr const bool verbose = false;

  if constexpr (verbose) {
    std::cout << "Computing coordinates (";
    for (auto c : coordinates_to_compute) std::cout << c << ", ";
    std::cout << "). with fixed values (";
    for (auto c : fixed_values) {
      std::cout << c << ", ";
    }
    std::cout << ")." << std::endl;
  }
  if (coordinates_to_compute.size() == 2) {
    compute_2d_hilbert_surface<PersBackend, Filtration, dtype, index_type>(thread_stuff,
                                                                           out,  // assumes its a zero tensor
                                                                           grid_shape,
                                                                           degrees,
                                                                           coordinates_to_compute[0],
                                                                           coordinates_to_compute[1],
                                                                           fixed_values,
                                                                           mobius_inverion,
                                                                           zero_pad,
                                                                           ignore_inf);
    return;
  }

  // coordinate to iterate.size --
  auto coordinate_to_iterate = coordinates_to_compute.back();
  coordinates_to_compute.pop_back();
  tbb::parallel_for(0, grid_shape[coordinate_to_iterate + 1], [&](index_type z) {
    // Updates fixes values that defines the slice
    std::vector<index_type> _fixed_values = fixed_values;  // TODO : do not copy this //thread local
    _fixed_values[coordinate_to_iterate] = z;
    _rec_get_hilbert_surface<PersBackend, Filtration, dtype, index_type>(thread_stuff,
                                                                         out,
                                                                         grid_shape,
                                                                         degrees,
                                                                         coordinates_to_compute,
                                                                         _fixed_values,
                                                                         mobius_inverion,
                                                                         zero_pad,
                                                                         ignore_inf);
  });
  // rmq : with mobius_inversion + rec, the coordinates to compute size is 2 =>
  // first coord is always the initial 1st coord.
  // => inversion is only needed for coords > 2
}

template <typename PersBackend, typename Filtration, typename dtype, typename index_type>
void get_hilbert_surface(Gudhi::multi_persistence::Slicer<Filtration, PersBackend> &slicer,
                         const tensor::static_tensor_view<dtype, index_type> &out,  // assumes its a zero tensor
                         const std::vector<index_type> &grid_shape,
                         const std::vector<index_type> &degrees,
                         std::vector<index_type> coordinates_to_compute,
                         const std::vector<index_type> &fixed_values,
                         const bool mobius_inverion = true,
                         const bool zero_pad = false,
                         const bool ignore_inf = true) {
  if (degrees.size() == 0) return;
  // wrapper arount the rec version, that initialize the thread variables.
  if (coordinates_to_compute.size() < 2)
    throw std::logic_error("Not implemented for " + std::to_string(coordinates_to_compute.size()) + "<2 parameters.");
  using ThreadSafe = typename Gudhi::multi_persistence::Slicer<Filtration, PersBackend>::Thread_safe;
  ThreadSafe slicer_thread(slicer);
  std::vector<index_type> coordinates_container(slicer_thread.get_number_of_parameters() + 1);  // +1 for degree
  // coordinates_container.reserve(fixed_values.size()+1);
  // coordinates_container.push_back(0); // degree
  // for (auto c : fixed_values) coordinates_container.push_back(c);
  std::pair<ThreadSafe, std::vector<index_type>> thread_data_initialization = {slicer_thread, coordinates_container};
  tbb::enumerable_thread_specific<std::pair<ThreadSafe, std::vector<index_type>>> thread_stuff(
      thread_data_initialization);  // this has a fixed size, so
                                    // init should be benefic
  _rec_get_hilbert_surface<PersBackend, Filtration, dtype, index_type>(thread_stuff,
                                                                       out,
                                                                       grid_shape,
                                                                       degrees,
                                                                       coordinates_to_compute,
                                                                       fixed_values,
                                                                       mobius_inverion,
                                                                       zero_pad,
                                                                       ignore_inf);
}

template <typename PersBackend, typename Filtration, typename dtype, typename indices_type, typename... Args>
void get_hilbert_surface_python(Gudhi::multi_persistence::Slicer<Filtration, PersBackend> &slicer,
                                dtype *data_ptr,
                                std::vector<indices_type> grid_shape,
                                const std::vector<indices_type> degrees,
                                const bool mobius_inversion,
                                const bool zero_pad,
                                const bool ignore_inf,
                                indices_type n_jobs) {
  const bool verbose = false;
  if (degrees.size() == 0) return;
  // const bool verbose = false;
  // auto &st_multi =
  //     get_simplextree_from_pointer<python_interface::interface_multi<Filtration>>(simplextree_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(data_ptr, grid_shape);  // assumes its a zero tensor
  int num_parameters = slicer.get_number_of_parameters();
  std::vector<indices_type> coordinates_to_compute(num_parameters);
  for (auto i = 0u; i < coordinates_to_compute.size(); i++) coordinates_to_compute[i] = i;
  // for (auto [c,i] : std::views::zip(coordinates_to_compute,
  // std::views::iota(0,st_multi.num_parameters()))) c=i; // NIK apple
  // clang
  std::vector<indices_type> fixed_values(num_parameters);

  if (verbose) {
    std::cout << "Container shape : ";
    for (auto r : container.get_resolution()) std::cout << r << ", ";
    std::cout << "\nContainer size : " << container.size();
    std::cout << "\nComputing hilbert invariant ...";
  }
  if (zero_pad) {
    // +1 is bc degree is on first axis.
    for (auto i = 1; i < num_parameters + 1; i++)
      grid_shape[i]--;  // get hilbert surface computes according to grid_shape.
    // for (auto i : std::views::iota(1,st_multi.num_parameters()+1))
    // grid_shape[i]--; // get hilbert surface computes according to grid_shape.
  }

  oneapi::tbb::task_arena arena(PersBackend::is_vine ? 1 : n_jobs);  // limits the number of threads
  arena.execute([&] {
    get_hilbert_surface(slicer,
                        container,
                        grid_shape,
                        degrees,
                        coordinates_to_compute,
                        fixed_values,
                        mobius_inversion,
                        zero_pad,
                        ignore_inf);
  });

  if (mobius_inversion)
    for (indices_type axis = 2u; axis < num_parameters + 1; axis++) container.differentiate(axis);
  return;
}

template <typename PersBackend, typename Filtration, typename dtype, typename indices_type, typename... Args>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> get_hilbert_signed_measure(
    Gudhi::multi_persistence::Slicer<Filtration, PersBackend> &slicer,
    dtype *data_ptr,
    std::vector<indices_type> grid_shape,
    const std::vector<indices_type> degrees,
    bool zero_pad = false,
    indices_type n_jobs = 0,
    const bool verbose = false,
    const bool ignore_inf = true) {
  if (degrees.size() == 0) return {{}, {}};
  // const bool verbose = false;
  // auto &st_multi =
  //     get_simplextree_from_pointer<python_interface::interface_multi<Filtration>>(simplextree_ptr);
  tensor::static_tensor_view<dtype, indices_type> container(data_ptr, grid_shape);  // assumes its a zero tensor
  std::vector<indices_type> coordinates_to_compute(slicer.get_number_of_parameters());
  for (auto i = 0u; i < coordinates_to_compute.size(); i++) coordinates_to_compute[i] = i;
  // for (auto [c,i] : std::views::zip(coordinates_to_compute,
  // std::views::iota(0,st_multi.num_parameters()))) c=i; // NIK apple
  // clang
  std::vector<indices_type> fixed_values(slicer.get_number_of_parameters());

  if (verbose) {
    std::cout << "Container shape : ";
    for (auto r : container.get_resolution()) std::cout << r << ", ";
    std::cout << "\nContainer size : " << container.size();
    std::cout << "\nComputing hilbert invariant ..." << std::flush;
  }
  if (zero_pad) {
    // +1 is bc degree is on first axis.
    for (auto i = 1u; i < slicer.get_number_of_parameters() + 1; i++)
      grid_shape[i]--;  // get hilbert surface computes according to grid_shape.
    // for (auto i : std::views::iota(1,st_multi.num_parameters()+1))
    // grid_shape[i]--; // get hilbert surface computes according to grid_shape.
  }

  oneapi::tbb::task_arena arena(PersBackend::is_vine ? 1 : n_jobs);  // limits the number of threads
  arena.execute([&] {
    get_hilbert_surface(
        slicer, container, grid_shape, degrees, coordinates_to_compute, fixed_values, true, zero_pad, ignore_inf);
  });

  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Computing mobius inversion ..." << std::flush;
  }

  // for (indices_type axis :
  // std::views::iota(2,st_multi.num_parameters()+1)) // +1 for the
  // degree in axis 0
  for (indices_type axis = 2; axis < static_cast<indices_type>(slicer.get_number_of_parameters() + 1); axis++)
    container.differentiate(axis);
  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Sparsifying the measure ..." << std::flush;
  }
  auto raw_signed_measure = container.sparsify();
  if (verbose) {
    std::cout << "Done." << std::endl;
  }
  return raw_signed_measure;
}
}  // namespace hilbert_function
}  // namespace multiparameter
}  // namespace Gudhi

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>  // std::pair
#include <vector>

#include <gudhi/multi_simplex_tree_helpers.h>
#include <gudhi/simple_mdspan.h>
#include <gudhi/Slicer.h>
#include "mobius_inversion.h"
#include "persistence_slices.h"

namespace Gudhi {
namespace multiparameter {
namespace rank_invariant {
// using Index = truc_interface::index_type;

// using Elbow = std::vector<std::pair<>>;grid
template <typename index_type>
inline void push_in_elbow(index_type &i, index_type &j, const index_type I, const index_type J) {
  if (j < J) {
    j++;
    return;
  }
  if (i < I) {
    i++;
    return;
  }
  j++;
  return;
}

template <typename index_type, typename value_type>
inline value_type get_slice_rank_filtration(const value_type x,
                                            const value_type y,
                                            const index_type I,
                                            const index_type J) {
  if (x > static_cast<value_type>(I))
    return std::numeric_limits<value_type>::has_infinity ? std::numeric_limits<value_type>::infinity()
                                                         : std::numeric_limits<value_type>::max();
  if (y > static_cast<value_type>(J)) return I + static_cast<index_type>(y);
  return J + static_cast<index_type>(x);
}

template <typename index_type>
inline std::pair<index_type, index_type> get_coordinates(index_type in_slice_value, index_type I, index_type J) {
  if (in_slice_value <= J) return {0, J};
  if (in_slice_value <= I + J) return {in_slice_value - J, J};
  return {I, in_slice_value - I};
}

template <typename Output, typename... Indices>
inline void increment_output(const Output &out, Indices... coordinates) {
  using index_type = std::common_type_t<Indices...>;
  const std::array<index_type, sizeof...(Indices)> coordinates_array{coordinates...};
  out[coordinates_array]++;
}

template <typename T, class Extents, class LayoutPolicy, typename... Indices>
inline void increment_output(const Gudhi::Simple_mdspan<T, Extents, LayoutPolicy> &out, Indices... coordinates) {
  using index_type = typename Gudhi::Simple_mdspan<T, Extents, LayoutPolicy>::index_type;
  out(static_cast<index_type>(coordinates)...)++;
}

template <typename index_type, typename Output>
inline void add_bar_contribution(const Output &out,
                                 index_type degree_index,
                                 index_type birth,
                                 index_type death,
                                 index_type I,
                                 index_type J,
                                 bool flip_death) {
  const index_type corner = I + J;
  if (birth > corner || death <= corner) return;

  const index_type last_birth = std::min(death, corner + 1);
  if (flip_death) {
    for (index_type intermediate_birth = birth; intermediate_birth < last_birth; ++intermediate_birth) {
      const auto [i, j] = get_coordinates(intermediate_birth, I, J);
      for (index_type l = J; l < death - I; ++l) {
        increment_output(out, degree_index, i, j, I - 1 - I, J - 1 - l);
      }
    }
  } else {
    for (index_type intermediate_birth = birth; intermediate_birth < last_birth; ++intermediate_birth) {
      const auto [i, j] = get_coordinates(intermediate_birth, I, J);
      for (index_type l = J; l < death - I; ++l) {
        increment_output(out, degree_index, i, j, I, l);
      }
    }
  }
}

template <typename index_type, typename Filtration, typename Output>
inline void compute_2d_rank_invariant_of_elbow(
    Simplex_tree<Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<Filtration>> &st_multi,
    Simplex_tree_std &_st_container,  // copy of st_multi
    const Output &out,                // assumes its a zero tensor
    const index_type I,
    const index_type J,
    const std::array<index_type, 5> &grid_shape,
    const std::vector<index_type> &degrees,
    const int expand_collapse_max_dim = 0) {
  // const bool verbose = false; // verbose
  using value_type = typename Simplex_tree_std::Filtration_value;
  constexpr const value_type inf = std::numeric_limits<value_type>::infinity();

  // const auto X = grid_shape[1],  Y = grid_shape[2]; // First axis is degree
  const auto Y = grid_shape[2];
  // Fills the filtration in the container
  // TODO : C++23 zip, when Apples clang will stop being obsolete
  auto sh_standard = _st_container.complex_simplex_range().begin();
  auto _end = _st_container.complex_simplex_range().end();
  auto sh_multi = st_multi.complex_simplex_range().begin();
  for (; sh_standard != _end; ++sh_multi, ++sh_standard) {
    const Filtration &multi_filtration = st_multi.filtration(*sh_multi);
    value_type filtration_in_slice = inf;
    for (unsigned int g = 0; g < multi_filtration.num_generators(); ++g) {
      value_type x = multi_filtration(g, 0);
      value_type y = multi_filtration(g, 1);
      filtration_in_slice = std::min(filtration_in_slice, get_slice_rank_filtration(x, y, I, J));
    }
    _st_container.assign_filtration(*sh_standard, filtration_in_slice);
  }
  const std::vector<Barcode> &barcodes = compute_dgms(_st_container, degrees, expand_collapse_max_dim);
  index_type degree_index = 0;
  for (const auto &barcode : barcodes) {  // TODO range view cartesian product
    for (const auto &bar : barcode) {
      auto birth = static_cast<index_type>(bar.first);
      auto death = static_cast<index_type>(
          std::min(bar.second,
                   static_cast<typename Simplex_tree_std::Filtration_value>(Y + I)));  // I,J atteints, pas X ni Y

      add_bar_contribution(out, degree_index, birth, death, I, J, false);
    }
    degree_index++;
  }
}

template <typename index_type, typename Filtration, typename Output>
inline void compute_2d_rank_invariant(
    Simplex_tree<Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<Filtration>> &st_multi,
    const Output &out,  // assumes its a zero tensor
    const std::array<index_type, 5> &grid_shape,
    const std::vector<index_type> &degrees,
    bool expand_collapse) {
  if (degrees.size() == 0) return;
  assert(st_multi.num_parameters() == 2);
  // copies the st_multi to a standard 1-pers simplextree
  Simplex_tree_std st_ =
      Gudhi::multi_persistence::make_one_dimensional<Gudhi::multiparameter::Simplex_tree_float>(st_multi, 0);
  const int max_dim = expand_collapse ? *std::max_element(degrees.begin(), degrees.end()) + 1 : 0;
  index_type X = grid_shape[1];
  index_type Y = grid_shape[2];                                                // First axis is degree
  tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree(st_);  // initialize with a good simplextree
  tbb::parallel_for(0, X, [&](index_type I) {
    tbb::parallel_for(0, Y, [&](index_type J) {
      auto &st_container = thread_simplex_tree.local();
      compute_2d_rank_invariant_of_elbow(st_multi, st_container, out, I, J, grid_shape, degrees, max_dim);
    });
  });
}

template <typename Filtration, typename dtype, typename indices_type, typename... Args>
void compute_rank_invariant_python(
    Simplex_tree<Gudhi::multi_persistence::Simplex_tree_options_multidimensional_filtration<Filtration>> &st_multi,
    dtype *data_ptr,
    const std::vector<indices_type> &grid_shape,
    const std::vector<indices_type> &degrees,
    indices_type n_jobs,
    bool expand_collapse) {
  if (degrees.size() == 0) return;
  if (grid_shape.size() != 5) [[unlikely]] {
    throw std::runtime_error("Internal error: rank invariant expects a 5-dimensional grid shape.");
  }
  const std::array<indices_type, 5> shape = {grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3], grid_shape[4]};
  auto container = Gudhi::Simple_mdspan(data_ptr,
                                        static_cast<std::size_t>(shape[0]),
                                        static_cast<std::size_t>(shape[1]),
                                        static_cast<std::size_t>(shape[2]),
                                        static_cast<std::size_t>(shape[3]),
                                        static_cast<std::size_t>(shape[4]));  // assumes its a zero tensor

  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  arena.execute([&] { compute_2d_rank_invariant(st_multi, container, shape, degrees, expand_collapse); });

  return;
}

template <class PersBackend, class MultiFiltration, typename index_type, typename Output>
inline void compute_2d_rank_invariant_of_elbow(
    typename Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend>::Thread_safe &slicer,  // truc slicer
    const Output &out,  // assumes its a zero tensor
    const index_type I,
    const index_type J,
    const std::array<index_type, 5> &grid_shape,
    const std::vector<index_type> &degrees,
    // std::vector<Index> &order_container,                                 // constant size
    // std::vector<typename MultiFiltration::value_type> &one_persistence,  // constant size
    const bool flip_death = false,
    const bool ignore_inf = true) {
  using value_type = typename MultiFiltration::value_type;
  const auto &filtrations_values = slicer.get_filtration_values();
  auto num_generators = filtrations_values.size();
  // one_persistence.resize(num_generators); // local variable should be
  // initialized correctly
  const auto Y = grid_shape[2];
  constexpr const bool verbose = false;
  if constexpr (verbose) std::cout << "filtration_in_slice : [ ";
  for (auto i = 0u; i < num_generators; ++i) {
    const auto &f = filtrations_values[i];
    value_type filtration_in_slice = MultiFiltration::T_inf;
    for (unsigned int g = 0; g < f.num_generators(); ++g) {
      value_type x = f(g, 0);
      value_type y = f(g, 1);

      filtration_in_slice = std::min(filtration_in_slice, get_slice_rank_filtration(x, y, I, J));
    }
    if constexpr (verbose) std::cout << filtration_in_slice << ",";
    slicer.get_slice()[i] = filtration_in_slice;
  }
  if constexpr (verbose) std::cout << "\b]" << std::endl;

  index_type degree_index = 0;
  // order_container.resize(slicer.num_generators()); // local variable should
  // be initialized correctly
  // TODO : use slicer::Thread_safe instead of maintaining one_pers & order
  // BUG : This will break as soon as slicer interface change

  using bc_type = typename Gudhi::multi_persistence::Slicer<MultiFiltration,
                                                            PersBackend>::template Multi_dimensional_flat_barcode<>;
  if (!slicer.persistence_computation_is_initialized()) [[unlikely]] {
    slicer.initialize_persistence_computation(ignore_inf);
  } else {
    slicer.update_persistence_computation(ignore_inf);
  }
  bc_type barcodes = slicer.template get_flat_barcode<true>();

  // note one_pers not necesary when vine, but does the same computation

  for (auto degree : degrees) {
    // this assumes barcodes degrees starts from 0
    if constexpr (verbose) std::cout << "Adding Barcode of degree " << degree << std::endl;
    if (degree >= static_cast<index_type>(barcodes.size())) continue;
    const auto &barcode = barcodes[degree];
    for (const auto &bar : barcode) {
      if (bar[0] > Y + I) continue;
      if constexpr (verbose)
        std::cout << bar[0] << " " << bar[1] << "checkinf: " << MultiFiltration::T_inf << " ==? "
                  << (bar[0] == MultiFiltration::T_inf) << std::endl;
      auto birth = static_cast<index_type>(bar[0]);
      auto death = static_cast<index_type>(
          std::min(bar[1],
                   static_cast<typename MultiFiltration::value_type>(Y + I)));  // I,J atteints, pas X ni Y
      if constexpr (false) std::cout << "Birth " << birth << " Death " << death << std::endl;
      add_bar_contribution(out, degree_index, birth, death, I, J, flip_death);
    }
    degree_index++;
  }
};

template <class PersBackend, class MultiFiltration, typename index_type, typename Output>
inline void compute_2d_rank_invariant(Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend> &slicer,
                                      const Output &out,  // assumes its a zero tensor
                                      const std::array<index_type, 5> &grid_shape,
                                      const std::vector<index_type> &degrees,
                                      const bool flip_death,
                                      const bool ignore_inf) {
  if (degrees.size() == 0) return;
  index_type X = grid_shape[1];
  index_type Y = grid_shape[2];  // First axis is degree
  constexpr const bool verbose = false;
  if constexpr (verbose)
    std::cout << "Shape " << grid_shape[0] << " " << grid_shape[1] << " " << grid_shape[2] << " " << grid_shape[3]
              << " " << grid_shape[4] << std::endl;

  using ThreadSafe = typename Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend>::Thread_safe;
  ThreadSafe slicer_thread(slicer);
  tbb::enumerable_thread_specific<ThreadSafe> thread_locals(slicer_thread);
  tbb::parallel_for(0, X, [&](index_type I) {
    tbb::parallel_for(0, Y, [&](index_type J) {
      if constexpr (verbose) std::cout << "Computing elbow " << I << " " << J << "...";
      ThreadSafe &slicer = thread_locals.local();
      compute_2d_rank_invariant_of_elbow<PersBackend, MultiFiltration, index_type>(
          slicer, out, I, J, grid_shape, degrees, flip_death, ignore_inf);
      if constexpr (verbose) std::cout << "Done!" << std::endl;
    });
  });
}

template <class PersBackend, class MultiFiltration, typename dtype, typename indices_type>
void compute_rank_invariant_python(Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend> &slicer,
                                   dtype *data_ptr,
                                   const std::vector<indices_type> &grid_shape,
                                   const std::vector<indices_type> &degrees,
                                   indices_type n_jobs,
                                   const bool ignore_inf) {
  if (degrees.size() == 0) return;
  if (grid_shape.size() != 5) [[unlikely]] {
    throw std::runtime_error("Internal error: rank invariant expects a 5-dimensional grid shape.");
  }
  const std::array<indices_type, 5> shape = {grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3], grid_shape[4]};
  auto container = Gudhi::Simple_mdspan(data_ptr,
                                        static_cast<std::size_t>(shape[0]),
                                        static_cast<std::size_t>(shape[1]),
                                        static_cast<std::size_t>(shape[2]),
                                        static_cast<std::size_t>(shape[3]),
                                        static_cast<std::size_t>(shape[4]));  // assumes its a zero tensor
  if constexpr (false) {
    std::cout << "ignore_inf " << ignore_inf << std::endl;
  }

  oneapi::tbb::task_arena arena(PersBackend::is_vine ? 1 : n_jobs);  // limits the number of threads
  arena.execute([&] { compute_2d_rank_invariant(slicer, container, shape, degrees, false, ignore_inf); });

  return;
}

template <typename PersBackend, typename MultiFiltration, typename dtype = int, typename indices_type = int>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> compute_rank_signed_measure(
    Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend> &slicer,
    dtype *data_ptr,
    const std::vector<indices_type> &grid_shape,
    const std::vector<indices_type> &degrees,
    indices_type n_jobs,
    bool verbose,
    const bool ignore_inf) {
  if (degrees.size() == 0) return {{}, {}};
  if (grid_shape.size() != 5) [[unlikely]] {
    throw std::runtime_error("Internal error: rank signed measure expects a 5-dimensional grid shape.");
  }
  const std::array<indices_type, 5> shape = {grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3], grid_shape[4]};
  mobius_inversion::dense_tensor_view<dtype, indices_type> container(data_ptr, grid_shape);  // assumes zero tensor
  oneapi::tbb::task_arena arena(n_jobs);  // limits the number of threads
  constexpr bool flip_death = true;
  arena.execute([&] { compute_2d_rank_invariant(slicer, container, shape, degrees, flip_death, ignore_inf); });

  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Computing mobius inversion ..." << std::flush;
  }

  // for (indices_type axis :
  // std::views::iota(2,st_multi.num_parameters()+1)) // +1 for the
  // degree in axis 0
  for (std::size_t axis = 0u; axis < slicer.get_number_of_parameters() + 1; axis++)
    mobius_inversion::differentiate(data_ptr, grid_shape, static_cast<indices_type>(axis));
  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Sparsifying the measure ..." << std::flush;
  }
  auto raw_signed_measure = mobius_inversion::sparsify(container, {false, false, true, true});
  if (verbose) {
    std::cout << "Done.\n";
  }
  return raw_signed_measure;
}

}  // namespace rank_invariant
}  // namespace multiparameter
}  // namespace Gudhi

#pragma once
#include "gudhi/truc.h"
#include "multi_parameter_rank_invariant/persistence_slices.h"
#include "tensor/tensor.h"
#include <algorithm>
#include <cstddef>
#include <gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <ostream>
#include <utility> // std::pair
#include <vector>
namespace Gudhi::multiparameter::rank_invariant {

// using Elbow = std::vector<std::pair<>>;grid
template <typename index_type>
inline void push_in_elbow(index_type &i, index_type &j, const index_type I,
                          const index_type J) {
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
inline value_type
get_slice_rank_filtration(const value_type x, const value_type y,
                          const index_type I, const index_type J) {
  if (x > static_cast<value_type>(I))
    return std::numeric_limits<value_type>::has_infinity
               ? std::numeric_limits<value_type>::infinity()
               : std::numeric_limits<value_type>::max();
  if (y > static_cast<value_type>(J))
    return I + static_cast<index_type>(y);
  return J + static_cast<index_type>(x);
}

template <typename index_type>
inline std::pair<index_type, index_type>
get_coordinates(index_type in_slice_value, index_type I, index_type J) {
  if (in_slice_value <= J)
    return {0, J};
  if (in_slice_value <= I + J)
    return {in_slice_value - J, J};
  return {I, in_slice_value - I};
}

template <typename dtype, typename index_type, typename Filtration>
inline void compute_2d_rank_invariant_of_elbow(
    Simplex_tree<Simplex_tree_options_multidimensional_filtration<Filtration>> &st_multi,
    Simplex_tree_std &_st_container, // copy of st_multi
    const tensor::static_tensor_view<dtype, index_type>
        &out, // assumes its a zero tensor
    const index_type I, const index_type J,
    const std::vector<index_type> &grid_shape,
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
    value_type filtration_in_slice;
    if constexpr (Filtration::is_multi_critical) {
      filtration_in_slice = inf;
      for (const auto &stuff : multi_filtration) {
        value_type x = stuff[0];
        value_type y = stuff[1];
        filtration_in_slice = std::min(filtration_in_slice,
                                       get_slice_rank_filtration(x, y, I, J));
      }
    } else {
      value_type x = multi_filtration[0];
      value_type y = multi_filtration[1];
      filtration_in_slice = get_slice_rank_filtration(x, y, I, J);
    }
    _st_container.assign_filtration(*sh_standard, filtration_in_slice);

  }
  const std::vector<Barcode> &barcodes =
      compute_dgms(_st_container, degrees, expand_collapse_max_dim);
  index_type degree_index = 0;
  for (const auto &barcode : barcodes) { // TODO range view cartesian product
    for (const auto &bar : barcode) {
      auto birth = static_cast<index_type>(bar.first);
      auto death = static_cast<index_type>(
          std::min(bar.second,
                   static_cast<typename Simplex_tree_std::Filtration_value>(Y + I))); // I,J atteints, pas X ni Y

      // todo : optimize
      // auto [a,b] = get_coordinates(birth, I,J);
      for (auto intermediate_birth = birth; intermediate_birth < death;
           intermediate_birth++) {
        for (auto intermediate_death = intermediate_birth;
             intermediate_death < death; intermediate_death++) {
          auto [i, j] = get_coordinates(intermediate_birth, I, J);
          auto [k, l] = get_coordinates(intermediate_death, I, J);
          if (((i < k || j == J) && (j < l || k == I))) {
            // std::vector<index_type> coordinates_to_remove =
            // {degree_index,i,j,k,l}; out[coordinates_to_remove]++;
            out[{degree_index, i, j, k, l}]++;
          }
        }
      }
    }
    degree_index++;
  }
}

template <typename dtype, typename index_type, typename Filtration>
inline void
compute_2d_rank_invariant(Simplex_tree<Simplex_tree_options_multidimensional_filtration<Filtration>> &st_multi,
                          const tensor::static_tensor_view<dtype, index_type>
                              &out, // assumes its a zero tensor
                          const std::vector<index_type> &grid_shape,
                          const std::vector<index_type> &degrees,
                          bool expand_collapse) {
  if (degrees.size() == 0)
    return;
  assert(st_multi.get_number_of_parameters() == 2);
  Simplex_tree_std st_;
  flatten(st_, st_multi,
          0); // copies the st_multi to a standard 1-pers simplextree
  const int max_dim =
      expand_collapse ? *std::max_element(degrees.begin(), degrees.end()) + 1
                      : 0;
  index_type X = grid_shape[1];
  index_type Y = grid_shape[2]; // First axis is degree
  tbb::enumerable_thread_specific<Simplex_tree_std> thread_simplex_tree(
      st_); // initialize with a good simplextree
  tbb::parallel_for(0, X, [&](index_type I) {
    tbb::parallel_for(0, Y, [&](index_type J) {
      auto &st_container = thread_simplex_tree.local();
      compute_2d_rank_invariant_of_elbow<dtype, index_type, Filtration>(st_multi, st_container, out, I, J,
                                         grid_shape, degrees, max_dim);
    });
  });
}

template <typename Filtration, typename dtype, typename indices_type, typename... Args>
void compute_rank_invariant_python(Simplex_tree<Simplex_tree_options_multidimensional_filtration<Filtration>>& st_multi,
                                   dtype *data_ptr,
                                   const std::vector<indices_type> grid_shape,
                                   const std::vector<indices_type> degrees,
                                   indices_type n_jobs, bool expand_collapse) {
  if (degrees.size() == 0)
    return;
  tensor::static_tensor_view<dtype, indices_type> container(
      data_ptr, grid_shape); // assumes its a zero tensor

  oneapi::tbb::task_arena arena(n_jobs); // limits the number of threads
  arena.execute([&] {
    compute_2d_rank_invariant<dtype, indices_type, Filtration>(st_multi, container, grid_shape, degrees,
                              expand_collapse);
  });

  return;
}
template <class PersBackend, class Structure,
          class MultiFiltration = Gudhi::multiparameter::multi_filtrations::
              Finitely_critical_multi_filtration<float>,
          typename dtype, typename index_type>
inline void compute_2d_rank_invariant_of_elbow(
    interface::Truc<PersBackend, Structure, MultiFiltration>
        &slicer, // truc slicer
    const tensor::static_tensor_view<dtype, index_type>
        &out, // assumes its a zero tensor
    const index_type I, const index_type J,
    const std::vector<index_type> &grid_shape,
    const std::vector<index_type> &degrees,
    std::vector<std::size_t> &order_container, // constant size
    std::vector<typename MultiFiltration::value_type>
        &one_persistence, // constant size
    const bool flip_death = false
) {
  using value_type = typename MultiFiltration::value_type;
  const auto &filtrations_values = slicer.get_filtrations();
  auto num_generators = filtrations_values.size();
  // one_persistence.resize(num_generators); // local variable should be
  // initialized correctly
  const auto Y = grid_shape[2];
  const bool verbose = false;
  if constexpr (verbose)
    std::cout << "filtration_in_slice : [ ";
  for (auto i = 0u; i < num_generators; ++i) {
    const auto &f = filtrations_values[i];
    value_type filtration_in_slice = MultiFiltration::T_inf;
    if constexpr (MultiFiltration::is_multi_critical) {
      for (const auto &stuff : f) {

        value_type x = stuff[0];
        value_type y = stuff[1];

        filtration_in_slice = std::min(filtration_in_slice,
                                       get_slice_rank_filtration(x, y, I, J));
      }
    } else {
      value_type x = f[0];
      value_type y = f[1];
      filtration_in_slice = get_slice_rank_filtration(x, y, I, J);
    }
    if constexpr (verbose)
      std::cout << filtration_in_slice << ",";
    one_persistence[i] = filtration_in_slice;
  }
  if constexpr (verbose)
    std::cout << "\b]" << std::endl;

  index_type degree_index = 0;
  // order_container.resize(slicer.num_generators()); // local variable should
  // be initialized correctly
  // TODO : use slicer::ThreadSafe instead of maintaining one_pers & order
  // BUG : This will break as soon as slicer interface change

  using bc_type = typename interface::Truc<PersBackend, Structure,
                                           MultiFiltration>::split_barcode;
  bc_type barcodes;
  if constexpr (PersBackend::is_vine) {
    slicer.set_one_filtration(one_persistence);
    if (I == 0 && J == 0)
        [[unlikely]] // this is dangerous, assumes it starts at 0 0
    {
      // TODO : This is a good optimization but needs a patch on PersistenceMatrix
      // std::vector<bool> degrees_index(slicer.get_dimensions().back()+1, false);
      // for (const auto &degree : degrees) {
      //   if (degree <= slicer.get_dimensions())
      //     degrees_index[degree] = true;
      // }
      // slicer.compute_persistence(degrees_index);
      slicer.compute_persistence();
    } else {
      slicer.vineyard_update();
    }
    barcodes = slicer.get_barcode();
  } else {
    PersBackend pers =
        slicer.compute_persistence_out(one_persistence, order_container);
    barcodes = slicer.get_barcode(pers, one_persistence);
  }

  // note one_pers not necesary when vine, but does the same computation

  for (auto degree : degrees) {
    // this assumes barcodes degrees starts from 0
    if constexpr (verbose)
      std::cout << "Adding Barcode of degree " << degree << std::endl;
    if (degree >= static_cast<index_type>(barcodes.size()))
      continue;
    const auto &barcode = barcodes[degree];
    for (const auto &bar : barcode) {
      if (bar.first > Y + I)
        continue;
      if constexpr (verbose)
        std::cout << bar.first << " " << bar.second
                  << "checkinf: " << MultiFiltration::T_inf << " ==? "
                  << (bar.first == MultiFiltration::T_inf) << std::endl;
      auto birth = static_cast<index_type>(bar.first);
      auto death = static_cast<index_type>(
          std::min(bar.second,
                   static_cast<typename MultiFiltration::value_type>(Y + I))); // I,J atteints, pas X ni Y
      if constexpr (false)
        std::cout << "Birth " << birth << " Death " << death << std::endl;
      for (auto intermediate_birth = birth; intermediate_birth < death;
           intermediate_birth++) {
        for (auto intermediate_death = intermediate_birth;
             intermediate_death < death; intermediate_death++) {
          auto [i, j] = get_coordinates(intermediate_birth, I, J);
          auto [k, l] = get_coordinates(intermediate_death, I, J);
          if (((i < k || j == J) && (j < l || k == I))) {
            if (flip_death)
              out[{degree_index, i, j, I-k, J-l}]++;
            else
              out[{degree_index, i, j, k, l}]++;
          }
          if constexpr (false)
            std::cout << degree_index << " " << i << " " << j << " " << k << " "
                      << l << std::endl;
        }
      }
    }
    degree_index++;
  }
};

template <class PersBackend, class Structure,
          class MultiFiltration = Gudhi::multiparameter::multi_filtrations::
              Finitely_critical_multi_filtration<float>,
          typename dtype, typename index_type>
inline void compute_2d_rank_invariant(
    interface::Truc<PersBackend, Structure, MultiFiltration> &slicer,
    const tensor::static_tensor_view<dtype, index_type>
        &out, // assumes its a zero tensor
    const std::vector<index_type> &grid_shape,
    const std::vector<index_type> &degrees, const bool flip_death) {
  if (degrees.size() == 0)
    return;
  index_type X = grid_shape[1];
  index_type Y = grid_shape[2]; // First axis is degree
  const bool verbose = false;
  if constexpr (verbose)
    std::cout << "Shape " << grid_shape[0] << " " << grid_shape[1] << " "
              << grid_shape[2] << " " << grid_shape[3] << " " << grid_shape[4]
              << std::endl;

  using local_type =
      std::tuple<std::vector<std::size_t>,                         // order
                 std::vector<typename MultiFiltration::value_type> // one
                                                                   // filtration
                 >;
  local_type local_template = {
      std::vector<std::size_t>(slicer.num_generators()),
      std::vector<typename MultiFiltration::value_type>(
          slicer.num_generators())};
  tbb::enumerable_thread_specific<local_type> thread_locals(local_template);
  tbb::parallel_for(0, X, [&](index_type I) {
    tbb::parallel_for(0, Y, [&](index_type J) {
      if constexpr (verbose)
        std::cout << "Computing elbow " << I << " " << J << "...";
      auto &[order_container, one_filtration_container] = thread_locals.local();
      compute_2d_rank_invariant_of_elbow(slicer, out, I, J, grid_shape, degrees,
                                         order_container,
                                         one_filtration_container,flip_death);
      if constexpr (verbose)
        std::cout << "Done!" << std::endl;
    });
  });
}

template <class PersBackend, class Structure,
          class MultiFiltration = Gudhi::multiparameter::multi_filtrations::
              Finitely_critical_multi_filtration<float>,
          typename dtype, typename indices_type>
void compute_rank_invariant_python(
    interface::Truc<PersBackend, Structure, MultiFiltration> slicer,
    dtype *data_ptr, const std::vector<indices_type> grid_shape,
    const std::vector<indices_type> degrees, indices_type n_jobs) {
  if (degrees.size() == 0)
    return;
  tensor::static_tensor_view<dtype, indices_type> container(
      data_ptr, grid_shape); // assumes its a zero tensor

  oneapi::tbb::task_arena arena(PersBackend::is_vine ? 1 : n_jobs); // limits the number of threads
  arena.execute([&] {
    compute_2d_rank_invariant(slicer, container, grid_shape, degrees, false);
  });

  return;
}








template <typename PersBackend,typename Structure, typename MultiFiltration, typename dtype = int, typename indices_type=int>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>>
 compute_rank_signed_measure(
    interface::Truc<PersBackend, Structure, MultiFiltration> slicer,
    dtype *data_ptr, const std::vector<indices_type> grid_shape,
    const std::vector<indices_type> degrees, indices_type n_jobs, bool verbose) {

  if (degrees.size() == 0)
    return {{},{}};
  tensor::static_tensor_view<dtype, indices_type> container(
      data_ptr, grid_shape); // assumes its a zero tensor
  oneapi::tbb::task_arena arena(n_jobs); // limits the number of threads
  constexpr bool flip_death = true;
  arena.execute([&] {
    compute_2d_rank_invariant(slicer, container, grid_shape, degrees, flip_death);
  });

  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Computing mobius inversion ..." << std::flush;
  }

  // for (indices_type axis :
  // std::views::iota(2,st_multi.get_number_of_parameters()+1)) // +1 for the
  // degree in axis 0
  for (std::size_t axis = 0u; axis < slicer.num_parameters()+ 1;
       axis++)
    container.differentiate(axis);
  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Sparsifying the measure ..." << std::flush;
  }
  auto raw_signed_measure = container.sparsify({false,false,true,true}); 
  if (verbose) {
    std::cout << "Done.\n";
  }
  return raw_signed_measure;
}

} // namespace Gudhi::multiparameter::rank_invariant

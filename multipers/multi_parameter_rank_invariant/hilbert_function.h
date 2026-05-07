#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>  // std::pair
#include <vector>

#include <gudhi/Slicer.h>
#include "mobius_inversion.h"

namespace Gudhi {
namespace multiparameter {
namespace hilbert_function {

template <std::size_t N, typename index_type>
inline std::array<index_type, 1 + N> hilbert_sparse_key_shape(const std::array<index_type, N> &grid_shape,
                                                             const index_type degree_count) {
  std::array<index_type, 1 + N> key_shape{};
  key_shape[0] = degree_count;
  for (std::size_t axis = 0; axis < N; ++axis) key_shape[1 + axis] = grid_shape[axis];
  return key_shape;
}

template <std::size_t N, typename dtype, typename index_type>
class packed_hilbert_sparse_accumulator {
 public:
  using key_type = std::uint64_t;
  using coordinate_type = std::array<index_type, 1 + N>;
  using entry_type = std::pair<key_type, dtype>;

  explicit packed_hilbert_sparse_accumulator(const coordinate_type &key_shape,
                                             const std::size_t initial_reserve = 1024)
      : key_shape_(key_shape), initial_reserve_(initial_reserve) {
    validate_key_shape();
    weights_.reserve(initial_reserve_);
  }

  packed_hilbert_sparse_accumulator(const packed_hilbert_sparse_accumulator &other)
      : key_shape_(other.key_shape_), initial_reserve_(other.initial_reserve_) {
    weights_.reserve(initial_reserve_);
  }

  void reserve(const std::size_t size) { weights_.reserve(size); }

  [[nodiscard]] std::size_t size() const { return weights_.size(); }

  void add(const coordinate_type &coordinates, const dtype weight) { add_key(pack(coordinates), weight); }

  void add_key(const key_type key, const dtype weight) {
    if (weight == dtype{0}) return;
    auto [it, inserted] = weights_.emplace(key, weight);
    if (inserted) return;
    it->second += weight;
    if (it->second == dtype{0}) weights_.erase(it);
  }

  void merge_from(const packed_hilbert_sparse_accumulator &other) {
    for (const auto &[key, weight] : other.weights_) add_key(key, weight);
  }

  void append_entries_to(std::vector<entry_type> &out) const {
    out.reserve(out.size() + weights_.size());
    for (const auto &[key, weight] : weights_) out.emplace_back(key, weight);
  }

  [[nodiscard]] coordinate_type unpack(key_type key) const {
    coordinate_type coordinates{};
    for (std::size_t axis = 1 + N; axis-- > 0;) {
      const auto extent = static_cast<key_type>(key_shape_[axis]);
      coordinates[axis] = static_cast<index_type>(key % extent);
      key /= extent;
    }
    return coordinates;
  }

 private:
  void validate_key_shape() const {
    key_type capacity = 1;
    constexpr key_type max_key = std::numeric_limits<key_type>::max();
    for (const auto extent_value : key_shape_) {
      if (extent_value <= index_type{0}) [[unlikely]] {
        throw std::runtime_error("Internal error: invalid sparse Hilbert key shape.");
      }
      const auto extent = static_cast<key_type>(extent_value);
      if (capacity > max_key / extent) [[unlikely]] {
        throw std::runtime_error("Sparse Hilbert key space exceeds 64-bit capacity.");
      }
      capacity *= extent;
    }
  }

  [[nodiscard]] key_type pack(const coordinate_type &coordinates) const {
    key_type key = 0;
    for (std::size_t axis = 0; axis < 1 + N; ++axis) {
      const auto extent = static_cast<key_type>(key_shape_[axis]);
      const auto coordinate = static_cast<key_type>(coordinates[axis]);
      if (coordinate >= extent) [[unlikely]] {
        throw std::runtime_error("Internal error: invalid sparse Hilbert coordinate.");
      }
      key = key * extent + coordinate;
    }
    return key;
  }

  coordinate_type key_shape_{};
  std::size_t initial_reserve_ = 1024;
  std::unordered_map<key_type, dtype> weights_;
};

template <std::size_t N, typename dtype, typename index_type, typename Accumulator>
inline void add_hilbert_atom(Accumulator &sm_pts,
                             const index_type degree_index,
                             const std::array<index_type, N> &coordinates,
                             const std::array<index_type, N> &output_shape,
                             const dtype weight) {
  using key_type = typename Accumulator::key_type;
  key_type key = static_cast<key_type>(degree_index);
  for (std::size_t axis = 0; axis < N; ++axis) {
    key = key * static_cast<key_type>(output_shape[axis]) + static_cast<key_type>(coordinates[axis]);
  }
  sm_pts.add_key(key, weight);
}

template <std::size_t N, typename index_type>
inline std::array<index_type, N> hilbert_active_shape(const std::array<index_type, N> &output_shape,
                                                     const bool zero_pad) {
  std::array<index_type, N> active_shape = output_shape;
  if (zero_pad) {
    for (auto &extent : active_shape) --extent;
  }
  for (const auto extent : active_shape) {
    if (extent <= index_type{0}) [[unlikely]] {
      throw std::runtime_error("Internal error: invalid sparse Hilbert grid shape.");
    }
  }
  return active_shape;
}

template <std::size_t N, typename index_type>
inline std::size_t hilbert_sparse_barcode_axis(const std::array<index_type, N> &active_shape) {
  std::size_t best_axis = 0;
  for (std::size_t axis = 1; axis < N; ++axis) {
    if (active_shape[axis] > active_shape[best_axis]) best_axis = axis;
  }
  return best_axis;
}

template <std::size_t N, typename index_type>
inline index_type hilbert_fixed_slice_count(const std::array<index_type, N> &active_shape,
                                           const std::size_t barcode_axis) {
  index_type count = 1;
  for (std::size_t axis = 0; axis < N; ++axis) {
    if (axis != barcode_axis) count *= active_shape[axis];
  }
  return count;
}

template <std::size_t N, typename index_type>
inline std::array<index_type, N> hilbert_fixed_from_linear(index_type linear,
                                                          const std::array<index_type, N> &active_shape,
                                                          const std::size_t barcode_axis) {
  std::array<index_type, N> fixed{};
  for (std::size_t axis = N; axis-- > 0;) {
    if (axis == barcode_axis) continue;
    fixed[axis] = linear % active_shape[axis];
    linear /= active_shape[axis];
  }
  return fixed;
}

template <std::size_t N, typename Filtration, typename index_type>
inline typename Filtration::value_type get_hilbert_line_filtration_on_axis(
    const Filtration &filtration,
    unsigned int generator,
    const std::array<index_type, N> &fixed,
    const std::size_t barcode_axis) {
  using value_type = typename Filtration::value_type;
  for (std::size_t axis = 0; axis < N; ++axis) {
    if (axis == barcode_axis) continue;
    if (filtration(generator, static_cast<unsigned int>(axis)) > static_cast<value_type>(fixed[axis])) {
      return Filtration::T_inf;
    }
  }
  return filtration(generator, static_cast<unsigned int>(barcode_axis));
}

template <std::size_t Axis, std::size_t N, typename dtype, typename index_type, typename Accumulator>
inline void emit_hilbert_endpoint_atom_combinations(
    Accumulator &sm_pts,
    const std::array<std::array<index_type, 2>, N> &endpoint_coordinates,
    const std::array<std::array<dtype, 2>, N> &endpoint_weights,
    const std::array<std::size_t, N> &endpoint_counts,
    std::array<index_type, 1 + N> &coordinates,
    const dtype weight) {
  if constexpr (Axis == N) {
    sm_pts.add(coordinates, weight);
  } else {
    for (std::size_t choice = 0; choice < endpoint_counts[Axis]; ++choice) {
      coordinates[1 + Axis] = endpoint_coordinates[Axis][choice];
      emit_hilbert_endpoint_atom_combinations<Axis + 1, N>(sm_pts,
                                                           endpoint_coordinates,
                                                           endpoint_weights,
                                                           endpoint_counts,
                                                           coordinates,
                                                           static_cast<dtype>(weight * endpoint_weights[Axis][choice]));
    }
  }
}

template <std::size_t N, typename dtype, typename index_type, typename value_type, typename Accumulator>
inline void emit_hilbert_bar_signed_measure_atoms(Accumulator &sm_pts,
                                                  const index_type degree_index,
                                                  const value_type birth_value,
                                                  const value_type death_value,
                                                  const std::array<index_type, N> &fixed,
                                                  const std::array<index_type, N> &active_shape,
                                                  const std::array<index_type, N> &output_shape,
                                                  const std::size_t barcode_axis,
                                                  const bool zero_pad) {
  if (birth_value >= static_cast<value_type>(active_shape[barcode_axis])) return;

  if constexpr (N == 2) {
    const auto fixed_axis = barcode_axis == 0 ? std::size_t{1} : std::size_t{0};
    const auto birth = static_cast<index_type>(birth_value);

    auto add_barcode_endpoint = [&](const index_type barcode_coordinate, const dtype sign) {
      std::array<index_type, N> coordinates = fixed;
      coordinates[barcode_axis] = barcode_coordinate;
      add_hilbert_atom<N, dtype, index_type, Accumulator>(sm_pts, degree_index, coordinates, output_shape, sign);
      if (fixed[fixed_axis] + 1 < output_shape[fixed_axis]) {
        coordinates[fixed_axis] = fixed[fixed_axis] + 1;
        add_hilbert_atom<N, dtype, index_type, Accumulator>(
            sm_pts, degree_index, coordinates, output_shape, static_cast<dtype>(-sign));
      }
    };

    add_barcode_endpoint(birth, dtype{1});
    if (death_value < static_cast<value_type>(active_shape[barcode_axis])) {
      add_barcode_endpoint(static_cast<index_type>(death_value), dtype{-1});
    } else if (zero_pad) {
      add_barcode_endpoint(active_shape[barcode_axis] - 1, dtype{-1});
    }
    return;
  }

  std::array<std::array<index_type, 2>, N> endpoint_coordinates{};
  std::array<std::array<dtype, 2>, N> endpoint_weights{};
  std::array<std::size_t, N> endpoint_counts{};

  for (std::size_t axis = 0; axis < N; ++axis) {
    endpoint_coordinates[axis][0] = fixed[axis];
    endpoint_weights[axis][0] = dtype{1};
    endpoint_counts[axis] = 1;
    if (axis != barcode_axis && fixed[axis] + 1 < output_shape[axis]) {
      endpoint_coordinates[axis][1] = fixed[axis] + 1;
      endpoint_weights[axis][1] = dtype{-1};
      endpoint_counts[axis] = 2;
    }
  }

  const auto birth = static_cast<index_type>(birth_value);
  endpoint_coordinates[barcode_axis][0] = birth;
  endpoint_weights[barcode_axis][0] = dtype{1};
  endpoint_counts[barcode_axis] = 1;

  if (death_value < static_cast<value_type>(active_shape[barcode_axis])) {
    endpoint_coordinates[barcode_axis][1] = static_cast<index_type>(death_value);
    endpoint_weights[barcode_axis][1] = dtype{-1};
    endpoint_counts[barcode_axis] = 2;
  } else if (zero_pad) {
    endpoint_coordinates[barcode_axis][1] = active_shape[barcode_axis] - 1;
    endpoint_weights[barcode_axis][1] = dtype{-1};
    endpoint_counts[barcode_axis] = 2;
  }

  std::array<index_type, 1 + N> coordinates{};
  coordinates[0] = degree_index;

  if constexpr (N == 3 || N == 4) {
    constexpr std::size_t total_choices = std::size_t{1} << N;
    for (std::size_t mask = 0; mask < total_choices; ++mask) {
      std::array<index_type, N> atom_coordinates{};
      dtype atom_weight = dtype{1};
      bool valid = true;
      for (std::size_t axis = 0; axis < N; ++axis) {
        const std::size_t choice = (mask >> axis) & std::size_t{1};
        if (choice >= endpoint_counts[axis]) {
          valid = false;
          break;
        }
        atom_coordinates[axis] = endpoint_coordinates[axis][choice];
        atom_weight = static_cast<dtype>(atom_weight * endpoint_weights[axis][choice]);
      }
      if (valid) {
        add_hilbert_atom<N, dtype, index_type, Accumulator>(
            sm_pts, degree_index, atom_coordinates, output_shape, atom_weight);
      }
    }
    return;
  }

  emit_hilbert_endpoint_atom_combinations<0, N>(
      sm_pts, endpoint_coordinates, endpoint_weights, endpoint_counts, coordinates, dtype{1});
}

template <std::size_t N, typename dtype, typename index_type>
inline std::pair<std::vector<std::vector<index_type>>, std::vector<dtype>> merge_sparse_hilbert_accumulators(
    tbb::enumerable_thread_specific<packed_hilbert_sparse_accumulator<N, dtype, index_type>> &thread_sm_pts,
    const std::array<index_type, 1 + N> &key_shape) {
  std::size_t total_size = 0;
  for (const auto &local_sm_pts : thread_sm_pts) total_size += local_sm_pts.size();
  packed_hilbert_sparse_accumulator<N, dtype, index_type> merged(key_shape, total_size);
  for (const auto &local_sm_pts : thread_sm_pts) merged.merge_from(local_sm_pts);

  std::vector<typename packed_hilbert_sparse_accumulator<N, dtype, index_type>::entry_type> entries;
  entries.reserve(merged.size());
  merged.append_entries_to(entries);
  std::sort(entries.begin(), entries.end(), [](const auto &left, const auto &right) { return left.first < right.first; });

  std::vector<std::vector<index_type>> coordinates;
  std::vector<dtype> weights;
  coordinates.reserve(entries.size());
  weights.reserve(entries.size());

  for (std::size_t i = 0; i < entries.size();) {
    const auto key = entries[i].first;
    dtype weight = dtype{0};
    do {
      weight = static_cast<dtype>(weight + entries[i].second);
      ++i;
    } while (i < entries.size() && entries[i].first == key);
    if (weight != dtype{0}) {
      const auto coordinate = merged.unpack(key);
      coordinates.emplace_back(coordinate.begin(), coordinate.end());
      weights.push_back(weight);
    }
  }
  return {coordinates, weights};
}

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

/// FROM SLICER
///
///

template <typename PersBackend, typename Filtration, typename dtype, typename index_type>
inline void compute_2d_hilbert_surface(
    tbb::enumerable_thread_specific<
        std::pair<typename Gudhi::multi_persistence::Slicer<Filtration, PersBackend>::Thread_safe,
                  std::vector<index_type>>> &thread_stuff,
    const mobius_inversion::dense_tensor_view<dtype, index_type> &out,  // assumes its a zero tensor
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
        if (birth >= I)  // some births are the squeezed infinity sentinel
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
    const mobius_inversion::dense_tensor_view<dtype, index_type> &out,  // assumes zero tensor
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
                         const mobius_inversion::dense_tensor_view<dtype, index_type> &out,  // assumes zero tensor
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
  mobius_inversion::dense_tensor_view<dtype, indices_type> container(data_ptr, grid_shape);  // assumes zero tensor
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
    for (indices_type axis = 2u; axis < num_parameters + 1; axis++)
      mobius_inversion::differentiate(data_ptr, container.get_resolution(), axis);
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
  mobius_inversion::dense_tensor_view<dtype, indices_type> container(data_ptr, grid_shape);  // assumes zero tensor
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
    mobius_inversion::differentiate(data_ptr, container.get_resolution(), axis);
  if (verbose) {
    std::cout << "Done.\n";
    std::cout << "Sparsifying the measure ..." << std::flush;
  }
  auto raw_signed_measure = mobius_inversion::sparsify(container);
  if (verbose) {
    std::cout << "Done." << std::endl;
  }
  return raw_signed_measure;
}

template <std::size_t N, class PersBackend, class Filtration, typename dtype, typename index_type>
inline std::pair<std::vector<std::vector<index_type>>, std::vector<dtype>>
compute_hilbert_signed_measure_sparse_by_slices(Gudhi::multi_persistence::Slicer<Filtration, PersBackend> &slicer,
                                                const std::array<index_type, N> &output_shape,
                                                const std::vector<index_type> &degrees,
                                                const bool zero_pad,
                                                const bool ignore_inf) {
  if (degrees.size() == 0) return {{}, {}};

  using value_type = typename Filtration::value_type;
  using ThreadSafe = typename Gudhi::multi_persistence::Slicer<Filtration, PersBackend>::Thread_safe;
  using bc_type = typename Gudhi::multi_persistence::Slicer<Filtration,
                                                            PersBackend>::template Multi_dimensional_flat_barcode<>;

  const auto active_shape = hilbert_active_shape<N>(output_shape, zero_pad);
  const auto barcode_axis = hilbert_sparse_barcode_axis<N>(active_shape);
  const auto fixed_slice_count = hilbert_fixed_slice_count<N>(active_shape, barcode_axis);
  const auto key_shape = hilbert_sparse_key_shape<N>(output_shape, static_cast<index_type>(degrees.size()));
  const std::size_t max_atoms_per_bar = std::size_t{1} << N;
  const std::size_t reserve_estimate = std::max<std::size_t>(
      1024,
      std::min<std::size_t>(65536,
                            static_cast<std::size_t>(fixed_slice_count) * degrees.size() * max_atoms_per_bar / 4));

  ThreadSafe slicer_thread(slicer);
  tbb::enumerable_thread_specific<ThreadSafe> thread_locals(slicer_thread);
  packed_hilbert_sparse_accumulator<N, dtype, index_type> sm_pts_accumulator(key_shape, reserve_estimate);
  tbb::enumerable_thread_specific<packed_hilbert_sparse_accumulator<N, dtype, index_type>> thread_sm_pts(
      sm_pts_accumulator);

  tbb::parallel_for(index_type{0}, fixed_slice_count, [&](index_type fixed_index) {
    const auto fixed = hilbert_fixed_from_linear<N>(fixed_index, active_shape, barcode_axis);
    ThreadSafe &local_slicer = thread_locals.local();
    auto &slice_filtration = local_slicer.get_slice();
    const auto &multi_filtration = local_slicer.get_filtration_values();

    for (std::size_t simplex_index = 0; simplex_index < multi_filtration.size(); ++simplex_index) {
      const auto &filtration = multi_filtration[simplex_index];
      value_type filtration_in_slice = Filtration::T_inf;
      for (unsigned int generator = 0; generator < filtration.num_generators(); ++generator) {
        filtration_in_slice = std::min(filtration_in_slice,
                                       get_hilbert_line_filtration_on_axis<N>(
                                           filtration, generator, fixed, barcode_axis));
      }
      slice_filtration[simplex_index] = filtration_in_slice;
    }

    if (!local_slicer.persistence_computation_is_initialized()) [[unlikely]] {
      local_slicer.initialize_persistence_computation(ignore_inf);
    } else {
      local_slicer.update_persistence_computation(ignore_inf);
    }
    bc_type barcodes = local_slicer.template get_flat_barcode<true>();

    auto &sm_pts = thread_sm_pts.local();
    index_type degree_index = 0;
    for (auto degree : degrees) {
      if (degree >= index_type{0} && degree < static_cast<index_type>(barcodes.size())) {
        const auto &barcode = barcodes[degree];
        for (const auto &bar : barcode) {
          emit_hilbert_bar_signed_measure_atoms<N, dtype, index_type, value_type>(sm_pts,
                                                                                  degree_index,
                                                                                  bar[0],
                                                                                  bar[1],
                                                                                  fixed,
                                                                                  active_shape,
                                                                                  output_shape,
                                                                                  barcode_axis,
                                                                                  zero_pad);
        }
      }
      ++degree_index;
    }
  });

  return merge_sparse_hilbert_accumulators<N, dtype, index_type>(thread_sm_pts, key_shape);
}

template <class PersBackend, class Filtration, typename dtype = int, typename indices_type = int>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> compute_hilbert_signed_measure_sparse_python(
    Gudhi::multi_persistence::Slicer<Filtration, PersBackend> &slicer,
    const std::vector<indices_type> &grid_shape,
    const std::vector<indices_type> &degrees,
    const bool zero_pad,
    indices_type n_jobs,
    const bool ignore_inf) {
  if (degrees.size() == 0) return {{}, {}};
  const auto num_parameters = slicer.get_number_of_parameters();
  oneapi::tbb::task_arena arena(PersBackend::is_vine ? 1 : n_jobs);
  switch (num_parameters) {
    case 2: {
      if (grid_shape.size() != 2) [[unlikely]] throw std::runtime_error("Internal error: invalid Hilbert grid shape.");
      const std::array<indices_type, 2> parameter_shape = {grid_shape[0], grid_shape[1]};
      std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> out;
      arena.execute([&] {
        out = compute_hilbert_signed_measure_sparse_by_slices<2, PersBackend, Filtration, dtype, indices_type>(
            slicer, parameter_shape, degrees, zero_pad, ignore_inf);
      });
      return out;
    }
    case 3: {
      if (grid_shape.size() != 3) [[unlikely]] throw std::runtime_error("Internal error: invalid Hilbert grid shape.");
      const std::array<indices_type, 3> parameter_shape = {grid_shape[0], grid_shape[1], grid_shape[2]};
      std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> out;
      arena.execute([&] {
        out = compute_hilbert_signed_measure_sparse_by_slices<3, PersBackend, Filtration, dtype, indices_type>(
            slicer, parameter_shape, degrees, zero_pad, ignore_inf);
      });
      return out;
    }
    case 4: {
      if (grid_shape.size() != 4) [[unlikely]] throw std::runtime_error("Internal error: invalid Hilbert grid shape.");
      const std::array<indices_type, 4> parameter_shape = {grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3]};
      std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> out;
      arena.execute([&] {
        out = compute_hilbert_signed_measure_sparse_by_slices<4, PersBackend, Filtration, dtype, indices_type>(
            slicer, parameter_shape, degrees, zero_pad, ignore_inf);
      });
      return out;
    }
    default:
      throw std::runtime_error("Sparse Hilbert signed measure is implemented for 2, 3, and 4 parameters.");
  }
}
}  // namespace hilbert_function
}  // namespace multiparameter
}  // namespace Gudhi

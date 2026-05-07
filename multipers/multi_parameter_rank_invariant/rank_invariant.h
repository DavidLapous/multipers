#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_for.h>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
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

template <std::size_t N, typename index_type>
inline std::array<index_type, 1 + 2 * N> rank_sparse_key_shape(const std::array<index_type, N> &grid_shape,
                                                              const index_type degree_count,
                                                              const bool zero_pad) {
  std::array<index_type, 1 + 2 * N> key_shape{};
  key_shape[0] = degree_count;
  for (std::size_t axis = 0; axis < N; ++axis) {
    key_shape[1 + axis] = grid_shape[axis];
    key_shape[1 + N + axis] = grid_shape[axis] + (zero_pad ? index_type{0} : index_type{1});
  }
  return key_shape;
}

template <std::size_t N, typename dtype, typename index_type>
class packed_rank_sparse_accumulator {
 public:
  using key_type = std::uint64_t;
  using coordinate_type = std::array<index_type, 1 + 2 * N>;

  explicit packed_rank_sparse_accumulator(const coordinate_type &key_shape) : key_shape_(key_shape) {
    key_type capacity = 1;
    constexpr key_type max_key = std::numeric_limits<key_type>::max();
    for (const auto extent_value : key_shape_) {
      if (extent_value <= index_type{0}) [[unlikely]] {
        throw std::runtime_error("Internal error: invalid sparse rank key shape.");
      }
      const auto extent = static_cast<key_type>(extent_value);
      if (capacity > max_key / extent) [[unlikely]] {
        throw std::runtime_error("Sparse rank key space exceeds 64-bit capacity.");
      }
      capacity *= extent;
    }
    weights_.reserve(1024);
  }

  void reserve(const std::size_t size) { weights_.reserve(size); }

  [[nodiscard]] std::size_t size() const { return weights_.size(); }

  void add(const coordinate_type &coordinates, const dtype weight) { add_key(pack(coordinates), weight); }

  void merge_from(const packed_rank_sparse_accumulator &other) {
    for (const auto &[key, weight] : other.weights_) add_key(key, weight);
  }

  [[nodiscard]] std::vector<key_type> sorted_keys() const {
    std::vector<key_type> keys;
    keys.reserve(weights_.size());
    for (const auto &[key, weight] : weights_) keys.push_back(key);
    std::sort(keys.begin(), keys.end());
    return keys;
  }

  [[nodiscard]] dtype weight(const key_type key) const { return weights_.at(key); }

  [[nodiscard]] coordinate_type unpack(key_type key) const {
    coordinate_type coordinates{};
    for (std::size_t axis = 1 + 2 * N; axis-- > 0;) {
      const auto extent = static_cast<key_type>(key_shape_[axis]);
      coordinates[axis] = static_cast<index_type>(key % extent);
      key /= extent;
    }
    return coordinates;
  }

 private:
  [[nodiscard]] key_type pack(const coordinate_type &coordinates) const {
    key_type key = 0;
    for (std::size_t axis = 0; axis < 1 + 2 * N; ++axis) {
      const auto extent = static_cast<key_type>(key_shape_[axis]);
      const auto coordinate = static_cast<key_type>(coordinates[axis]);
      if (coordinate >= extent) [[unlikely]] {
        throw std::runtime_error("Internal error: invalid sparse rank coordinate.");
      }
      key = key * extent + coordinate;
    }
    return key;
  }

  void add_key(const key_type key, const dtype weight) {
    if (weight == dtype{0}) return;
    auto [it, inserted] = weights_.emplace(key, weight);
    if (inserted) return;
    it->second += weight;
    if (it->second == dtype{0}) weights_.erase(it);
  }

  coordinate_type key_shape_{};
  std::unordered_map<key_type, dtype> weights_;
};

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

template <std::size_t N, typename index_type>
inline std::array<index_type, 1 + 2 * N> vector_to_full_rank_shape(const std::vector<index_type> &values) {
  if (values.size() != 1 + 2 * N) [[unlikely]] {
    throw std::runtime_error("Internal error: invalid fixed rank tensor shape dispatch.");
  }
  std::array<index_type, 1 + 2 * N> out{};
  std::copy_n(values.begin(), 1 + 2 * N, out.begin());
  return out;
}

template <std::size_t N, typename Output, typename index_type>
inline void increment_rank_output(const Output &out,
                                  index_type degree_index,
                                  const std::array<index_type, N> &birth,
                                  const std::array<index_type, N> &death) {
  std::array<index_type, 1 + 2 * N> coordinates{};
  coordinates[0] = degree_index;
  for (std::size_t axis = 0; axis < N; ++axis) {
    coordinates[1 + axis] = birth[axis];
    coordinates[1 + N + axis] = death[axis];
  }
  out[coordinates]++;
}

template <std::size_t N, typename index_type>
inline index_type monotone_pair_count(index_type extent) {
  return extent * (extent + 1) / 2;
}

template <typename index_type>
inline std::pair<index_type, index_type> monotone_pair_from_linear(index_type linear, index_type extent) {
  for (index_type lower = 0; lower < extent; ++lower) {
    const index_type count = extent - lower;
    if (linear < count) return {lower, lower + linear};
    linear -= count;
  }
  return {extent - 1, extent - 1};
}

template <std::size_t N, typename index_type>
inline index_type slice_path_family_size(const std::array<index_type, N> &grid_shape) {
  index_type size = grid_shape[0] * grid_shape[N - 1];
  for (std::size_t axis = 1; axis + 1 < N; ++axis) size *= monotone_pair_count<N>(grid_shape[axis]);
  return size;
}

template <std::size_t N, typename index_type>
inline std::array<std::size_t, N> sparse_slice_path_axis_order(const std::array<index_type, N> &grid_shape) {
  std::array<std::size_t, N> identity{};
  for (std::size_t axis = 0; axis < N; ++axis) identity[axis] = axis;
  if constexpr (N > 2) {
    std::array<std::size_t, N> sorted_axes = identity;
    std::sort(sorted_axes.begin(), sorted_axes.end(), [&](std::size_t left, std::size_t right) {
      if (grid_shape[left] != grid_shape[right]) return grid_shape[left] > grid_shape[right];
      return left < right;
    });
    std::array<std::size_t, N> order{};
    order[0] = sorted_axes[0];
    order[N - 1] = sorted_axes[1];
    for (std::size_t axis = 1; axis + 1 < N; ++axis) order[axis] = sorted_axes[axis + 1];
    std::array<index_type, N> ordered_shape{};
    for (std::size_t axis = 0; axis < N; ++axis) ordered_shape[axis] = grid_shape[order[axis]];
    if (slice_path_family_size<N>(ordered_shape) >= slice_path_family_size<N>(grid_shape)) return identity;
    return order;
  }
  return identity;
}

template <std::size_t N>
inline bool axis_order_is_identity(const std::array<std::size_t, N> &axis_order) {
  for (std::size_t axis = 0; axis < N; ++axis)
    if (axis_order[axis] != axis) return false;
  return true;
}

template <std::size_t N, typename index_type>
inline std::array<index_type, N> ordered_grid_shape(const std::array<index_type, N> &grid_shape,
                                                   const std::array<std::size_t, N> &axis_order) {
  std::array<index_type, N> out{};
  for (std::size_t axis = 0; axis < N; ++axis) out[axis] = grid_shape[axis_order[axis]];
  return out;
}

template <std::size_t N, typename index_type>
inline void slice_path_bounds_from_linear(index_type linear,
                                          const std::array<index_type, N> &grid_shape,
                                          std::array<index_type, N> &lower,
                                          std::array<index_type, N> &upper) {
  lower.fill(index_type{0});
  upper = grid_shape;
  for (auto &extent : upper) --extent;

  upper[0] = linear % grid_shape[0];
  linear /= grid_shape[0];

  for (std::size_t axis = 1; axis + 1 < N; ++axis) {
    const index_type count = monotone_pair_count<N>(grid_shape[axis]);
    const auto [axis_lower, axis_upper] = monotone_pair_from_linear(linear % count, grid_shape[axis]);
    linear /= count;
    lower[axis] = axis_lower;
    upper[axis] = axis_upper;
  }

  lower[N - 1] = linear % grid_shape[N - 1];
}

template <std::size_t N, typename index_type>
inline index_type path_scalar_min(const std::array<index_type, N> &lower) {
  index_type value = 0;
  for (std::size_t axis = 1; axis < N; ++axis) value += lower[axis];
  return value;
}

template <std::size_t N, typename index_type>
inline index_type path_last_segment_start(const std::array<index_type, N> &upper,
                                          const std::array<index_type, N> &lower) {
  index_type value = lower[N - 1];
  for (std::size_t axis = 0; axis + 1 < N; ++axis) value += upper[axis];
  return value;
}

template <std::size_t N, typename index_type>
inline index_type path_last_segment_stop(const std::array<index_type, N> &upper,
                                         const std::array<index_type, N> &grid_shape) {
  index_type value = grid_shape[N - 1];
  for (std::size_t axis = 0; axis + 1 < N; ++axis) value += upper[axis];
  return value;
}

template <std::size_t N, typename Filtration, typename index_type>
inline typename Filtration::value_type get_slice_rank_filtration_on_slice_path(
    const Filtration &filtration,
    unsigned int generator,
    const std::array<index_type, N> &lower,
    const std::array<index_type, N> &upper) {
  using value_type = typename Filtration::value_type;
  if constexpr (N == 2) {
    return get_slice_rank_filtration<index_type, value_type>(filtration(generator, 0),
                                                            filtration(generator, 1),
                                                            upper[0],
                                                            lower[1]);
  }
  for (std::size_t segment = 0; segment < N; ++segment) {
    bool appears_on_segment = true;
    for (std::size_t axis = 0; axis < segment; ++axis) {
      if (filtration(generator, static_cast<unsigned int>(axis)) > static_cast<value_type>(upper[axis])) {
        appears_on_segment = false;
        break;
      }
    }
    if (!appears_on_segment) continue;
    for (std::size_t axis = segment + 1; axis < N; ++axis) {
      if (filtration(generator, static_cast<unsigned int>(axis)) > static_cast<value_type>(lower[axis])) {
        appears_on_segment = false;
        break;
      }
    }
    if (!appears_on_segment) continue;
    const auto coordinate = static_cast<index_type>(filtration(generator, static_cast<unsigned int>(segment)));
    if (coordinate > upper[segment]) continue;
    index_type scalar = std::max(lower[segment], coordinate);
    for (std::size_t axis = 0; axis < segment; ++axis) scalar += upper[axis];
    for (std::size_t axis = segment + 1; axis < N; ++axis) scalar += lower[axis];
    return static_cast<value_type>(scalar);
  }
  return Filtration::T_inf;
}

template <std::size_t N, typename Filtration, typename index_type>
inline typename Filtration::value_type get_slice_rank_filtration_on_ordered_slice_path(
    const Filtration &filtration,
    unsigned int generator,
    const std::array<index_type, N> &lower,
    const std::array<index_type, N> &upper,
    const std::array<std::size_t, N> &axis_order) {
  using value_type = typename Filtration::value_type;
  for (std::size_t segment = 0; segment < N; ++segment) {
    bool appears_on_segment = true;
    for (std::size_t axis = 0; axis < segment; ++axis) {
      if (filtration(generator, static_cast<unsigned int>(axis_order[axis])) > static_cast<value_type>(upper[axis])) {
        appears_on_segment = false;
        break;
      }
    }
    if (!appears_on_segment) continue;
    for (std::size_t axis = segment + 1; axis < N; ++axis) {
      if (filtration(generator, static_cast<unsigned int>(axis_order[axis])) > static_cast<value_type>(lower[axis])) {
        appears_on_segment = false;
        break;
      }
    }
    if (!appears_on_segment) continue;
    const auto coordinate = static_cast<index_type>(filtration(generator, static_cast<unsigned int>(axis_order[segment])));
    if (coordinate > upper[segment]) continue;
    index_type scalar = std::max(lower[segment], coordinate);
    for (std::size_t axis = 0; axis < segment; ++axis) scalar += upper[axis];
    for (std::size_t axis = segment + 1; axis < N; ++axis) scalar += lower[axis];
    return static_cast<value_type>(scalar);
  }
  return Filtration::T_inf;
}

template <std::size_t Axis, std::size_t N, typename dtype, typename index_type, typename Accumulator>
inline void emit_endpoint_atom_combinations(
    Accumulator &atoms,
    const std::array<std::array<index_type, 2>, 2 * N> &endpoint_coordinates,
    const std::array<std::array<dtype, 2>, 2 * N> &endpoint_weights,
    const std::array<std::size_t, 2 * N> &endpoint_counts,
    std::array<index_type, 1 + 2 * N> &coordinates,
    const dtype weight) {
  if constexpr (Axis == 2 * N) {
    bool valid_rank_coordinate = true;
    for (std::size_t axis = 0; axis < N; ++axis) {
      if (coordinates[1 + axis] >= coordinates[1 + N + axis]) {
        valid_rank_coordinate = false;
        break;
      }
    }
    if (valid_rank_coordinate) atoms.add(coordinates, weight);
  } else {
    for (std::size_t choice = 0; choice < endpoint_counts[Axis]; ++choice) {
      coordinates[1 + Axis] = endpoint_coordinates[Axis][choice];
      emit_endpoint_atom_combinations<Axis + 1, N>(
          atoms,
          endpoint_coordinates,
          endpoint_weights,
          endpoint_counts,
          coordinates,
          static_cast<dtype>(weight * endpoint_weights[Axis][choice]));
    }
  }
}

template <std::size_t N, typename index_type, typename Output>
inline void add_slice_path_bar_contribution(const Output &out,
                                            index_type degree_index,
                                            index_type birth,
                                            index_type death,
                                            const std::array<index_type, N> &lower,
                                            const std::array<index_type, N> &upper,
                                            const std::array<index_type, N> &grid_shape) {
  const index_type first_start = path_scalar_min<N>(lower);
  const index_type first_stop = first_start + upper[0] + 1;
  const index_type last_start = path_last_segment_start<N>(upper, lower);
  const index_type last_stop = path_last_segment_stop<N>(upper, grid_shape);
  if (birth >= first_stop || death <= last_start) return;

  const index_type birth_stop = std::min(death, first_stop);
  const index_type death_stop = std::min(death, last_stop);
  std::array<index_type, N> birth_coordinate = lower;
  std::array<index_type, N> death_coordinate = upper;
  for (index_type birth_scalar = std::max(birth, first_start); birth_scalar < birth_stop; ++birth_scalar) {
    birth_coordinate[0] = birth_scalar - first_start;
    for (index_type death_scalar = last_start; death_scalar < death_stop; ++death_scalar) {
      death_coordinate[N - 1] = death_scalar - (last_start - lower[N - 1]);
      increment_rank_output<N>(out, degree_index, birth_coordinate, death_coordinate);
    }
  }
}

template <std::size_t N, typename dtype, typename index_type, typename Accumulator>
inline void emit_interval_endpoint_atoms(Accumulator &atoms,
                                          index_type degree_index,
                                          const std::array<index_type, N> &birth_low,
                                          const std::array<index_type, N> &birth_high,
                                         const std::array<index_type, N> &death_low,
                                         const std::array<index_type, N> &death_high,
                                         const std::array<index_type, N> &grid_shape,
                                         bool zero_pad) {
  std::array<std::array<index_type, 2>, 2 * N> endpoint_coordinates{};
  std::array<std::array<dtype, 2>, 2 * N> endpoint_weights{};
  std::array<std::size_t, 2 * N> endpoint_counts{};

  for (std::size_t axis = 0; axis < N; ++axis) {
    if (birth_high[axis] <= birth_low[axis]) return;
    endpoint_coordinates[axis][0] = birth_low[axis];
    endpoint_weights[axis][0] = dtype{1};
    endpoint_counts[axis] = 1;
    if (birth_high[axis] < grid_shape[axis]) {
      endpoint_coordinates[axis][1] = birth_high[axis];
      endpoint_weights[axis][1] = dtype{-1};
      endpoint_counts[axis] = 2;
    }
  }

  for (std::size_t axis = 0; axis < N; ++axis) {
    const auto high = zero_pad ? std::min(death_high[axis], grid_shape[axis] - 1) : death_high[axis];
    if (high <= death_low[axis]) return;
    const auto endpoint_axis = N + axis;
    endpoint_coordinates[endpoint_axis][0] = high;
    endpoint_weights[endpoint_axis][0] = dtype{1};
    endpoint_counts[endpoint_axis] = 1;
    if (death_low[axis] > 0) {
      endpoint_coordinates[endpoint_axis][1] = death_low[axis];
      endpoint_weights[endpoint_axis][1] = dtype{-1};
      endpoint_counts[endpoint_axis] = 2;
    }
  }

  std::array<index_type, 1 + 2 * N> coordinates{};
  coordinates[0] = degree_index;
  emit_endpoint_atom_combinations<0, N>(
      atoms, endpoint_coordinates, endpoint_weights, endpoint_counts, coordinates, dtype{1});
}

template <std::size_t N, typename dtype, typename index_type, typename Accumulator>
inline void emit_slice_path_bar_signed_measure_atoms(Accumulator &atoms,
                                                     index_type degree_index,
                                                     index_type birth,
                                                     index_type death,
                                                     const std::array<index_type, N> &lower,
                                                     const std::array<index_type, N> &upper,
                                                     const std::array<index_type, N> &grid_shape,
                                                     bool zero_pad) {
  if constexpr (N == 2) {
    const index_type first_start = lower[1];
    const index_type first_stop = first_start + upper[0] + 1;
    const index_type last_start = upper[0] + lower[1];
    const index_type last_stop = upper[0] + grid_shape[1];
    if (birth >= first_stop || death <= last_start) return;

    const index_type birth_stop = std::min(death, first_stop);
    const index_type death_stop = std::min(death, last_stop);
    const index_type birth_x_low = std::max(birth, first_start) - first_start;
    const index_type birth_x_high = birth_stop - first_start;
    const index_type birth_y_low = lower[1];
    const index_type birth_y_high = lower[1] + 1;
    const index_type death_x_low = upper[0];
    const index_type death_x_high = upper[0] + 1;
    const index_type death_y_low = lower[1];
    const index_type death_y_high = lower[1] + death_stop - last_start;

    if (birth_x_high <= birth_x_low || death_y_high <= death_y_low) return;

    std::array<std::array<index_type, 2>, 3> endpoint_coordinates{};
    std::array<std::array<dtype, 2>, 3> endpoint_weights{};
    std::array<std::size_t, 3> endpoint_counts{};

    endpoint_coordinates[0][0] = birth_x_low;
    endpoint_weights[0][0] = dtype{1};
    endpoint_counts[0] = 1;
    if (birth_x_high < grid_shape[0]) {
      endpoint_coordinates[0][1] = birth_x_high;
      endpoint_weights[0][1] = dtype{-1};
      endpoint_counts[0] = 2;
    }

    endpoint_coordinates[1][0] = birth_y_low;
    endpoint_weights[1][0] = dtype{1};
    endpoint_counts[1] = 1;
    if (birth_y_high < grid_shape[1]) {
      endpoint_coordinates[1][1] = birth_y_high;
      endpoint_weights[1][1] = dtype{-1};
      endpoint_counts[1] = 2;
    }

    const index_type death_x_high_endpoint =
        zero_pad ? std::min(death_x_high, grid_shape[0] - 1) : death_x_high;
    if (death_x_high_endpoint <= death_x_low) return;
    endpoint_coordinates[2][0] = death_x_high_endpoint;
    endpoint_weights[2][0] = dtype{1};
    endpoint_counts[2] = 1;
    if (death_x_low > 0) {
      endpoint_coordinates[2][1] = death_x_low;
      endpoint_weights[2][1] = dtype{-1};
      endpoint_counts[2] = 2;
    }

    const index_type death_y_high_endpoint =
        zero_pad ? std::min(death_y_high, grid_shape[1] - 1) : death_y_high;
    if (death_y_high_endpoint <= death_y_low) return;
    std::array<index_type, 5> coordinates{};
    coordinates[0] = degree_index;
    coordinates[4] = death_y_high_endpoint;
    for (std::size_t bx = 0; bx < endpoint_counts[0]; ++bx) {
      coordinates[1] = endpoint_coordinates[0][bx];
      const dtype bx_weight = endpoint_weights[0][bx];
      for (std::size_t by = 0; by < endpoint_counts[1]; ++by) {
        coordinates[2] = endpoint_coordinates[1][by];
        const dtype birth_weight = static_cast<dtype>(bx_weight * endpoint_weights[1][by]);
        for (std::size_t dx = 0; dx < endpoint_counts[2]; ++dx) {
          coordinates[3] = endpoint_coordinates[2][dx];
          const dtype death_x_weight = static_cast<dtype>(birth_weight * endpoint_weights[2][dx]);
          if (coordinates[1] < coordinates[3] && coordinates[2] < coordinates[4]) {
            atoms.add(coordinates, death_x_weight);
          }
        }
      }
    }
    return;
  }

  const index_type first_start = path_scalar_min<N>(lower);
  const index_type first_stop = first_start + upper[0] + 1;
  const index_type last_start = path_last_segment_start<N>(upper, lower);
  const index_type last_stop = path_last_segment_stop<N>(upper, grid_shape);
  if (birth >= first_stop || death <= last_start) return;

  const index_type birth_stop = std::min(death, first_stop);
  const index_type death_stop = std::min(death, last_stop);

  std::array<index_type, N> birth_low = lower;
  std::array<index_type, N> birth_high = lower;
  std::array<index_type, N> death_low = upper;
  std::array<index_type, N> death_high = upper;

  birth_low[0] = std::max(birth, first_start) - first_start;
  birth_high[0] = birth_stop - first_start;
  for (std::size_t axis = 1; axis < N; ++axis) birth_high[axis] = lower[axis] + 1;

  for (std::size_t axis = 0; axis + 1 < N; ++axis) death_high[axis] = upper[axis] + 1;
  death_low[N - 1] = lower[N - 1];
  death_high[N - 1] = lower[N - 1] + death_stop - last_start;

  emit_interval_endpoint_atoms<N, dtype, index_type, Accumulator>(atoms,
                                                                  degree_index,
                                                                  birth_low,
                                                                  birth_high,
                                                                  death_low,
                                                                  death_high,
                                                                  grid_shape,
                                                                  zero_pad);
}

template <std::size_t N, typename dtype, typename index_type, typename Accumulator>
inline void emit_ordered_slice_path_bar_signed_measure_atoms(Accumulator &atoms,
                                                             index_type degree_index,
                                                             index_type birth,
                                                             index_type death,
                                                             const std::array<index_type, N> &lower,
                                                             const std::array<index_type, N> &upper,
                                                             const std::array<index_type, N> &grid_shape,
                                                             const std::array<index_type, N> &ordered_shape,
                                                             const std::array<std::size_t, N> &axis_order,
                                                             bool zero_pad) {
  const index_type first_start = path_scalar_min<N>(lower);
  const index_type first_stop = first_start + upper[0] + 1;
  const index_type last_start = path_last_segment_start<N>(upper, lower);
  const index_type last_stop = path_last_segment_stop<N>(upper, ordered_shape);
  if (birth >= first_stop || death <= last_start) return;

  const index_type birth_stop = std::min(death, first_stop);
  const index_type death_stop = std::min(death, last_stop);

  std::array<index_type, N> birth_low{};
  std::array<index_type, N> birth_high{};
  std::array<index_type, N> death_low{};
  std::array<index_type, N> death_high{};
  for (std::size_t axis = 0; axis < N; ++axis) {
    const auto original_axis = axis_order[axis];
    birth_low[original_axis] = lower[axis];
    birth_high[original_axis] = lower[axis] + 1;
    death_low[original_axis] = upper[axis];
    death_high[original_axis] = upper[axis] + 1;
  }

  birth_low[axis_order[0]] = std::max(birth, first_start) - first_start;
  birth_high[axis_order[0]] = birth_stop - first_start;

  death_low[axis_order[N - 1]] = lower[N - 1];
  death_high[axis_order[N - 1]] = lower[N - 1] + death_stop - last_start;

  emit_interval_endpoint_atoms<N, dtype, index_type, Accumulator>(atoms,
                                                                  degree_index,
                                                                  birth_low,
                                                                  birth_high,
                                                                  death_low,
                                                                  death_high,
                                                                  grid_shape,
                                                                  zero_pad);
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

template <std::size_t N, class PersBackend, class MultiFiltration, typename index_type, typename Output>
inline void compute_nd_rank_invariant_by_slice_paths(
    Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend> &slicer,
    const Output &out,  // assumes its a zero tensor
    const std::array<index_type, N> &grid_shape,
    const std::vector<index_type> &degrees,
    const bool ignore_inf) {
  if (degrees.size() == 0) return;
  const index_type total_slice_paths = slice_path_family_size<N>(grid_shape);
  using value_type = typename MultiFiltration::value_type;
  using ThreadSafe = typename Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend>::Thread_safe;
  ThreadSafe slicer_thread(slicer);
  tbb::enumerable_thread_specific<ThreadSafe> thread_locals(slicer_thread);

  tbb::parallel_for(index_type{0}, total_slice_paths, [&](index_type slice_path_index) {
    std::array<index_type, N> lower{};
    std::array<index_type, N> upper{};
    slice_path_bounds_from_linear<N>(slice_path_index, grid_shape, lower, upper);

    ThreadSafe &local_slicer = thread_locals.local();
    auto &slice_filtration = local_slicer.get_slice();
    const auto &filtration_values = local_slicer.get_filtration_values();
    for (std::size_t i = 0; i < filtration_values.size(); ++i) {
      const auto &filtration = filtration_values[i];
      value_type filtration_in_slice = MultiFiltration::T_inf;
      for (unsigned int generator = 0; generator < filtration.num_generators(); ++generator) {
        filtration_in_slice = std::min(
            filtration_in_slice,
            get_slice_rank_filtration_on_slice_path<N>(filtration, generator, lower, upper));
      }
      slice_filtration[i] = filtration_in_slice;
    }

    using bc_type = typename Gudhi::multi_persistence::Slicer<MultiFiltration,
                                                              PersBackend>::template Multi_dimensional_flat_barcode<>;
    if (!local_slicer.persistence_computation_is_initialized()) [[unlikely]] {
      local_slicer.initialize_persistence_computation(ignore_inf);
    } else {
      local_slicer.update_persistence_computation(ignore_inf);
    }
    bc_type barcodes = local_slicer.template get_flat_barcode<true>();

    index_type degree_index = 0;
    for (auto degree : degrees) {
      if (degree < static_cast<index_type>(barcodes.size())) {
        const auto &barcode = barcodes[degree];
        for (const auto &bar : barcode) {
          const auto birth = static_cast<index_type>(bar[0]);
          const auto death = static_cast<index_type>(
              std::min(bar[1], static_cast<value_type>(path_last_segment_stop<N>(upper, grid_shape))));
          add_slice_path_bar_contribution<N>(out, degree_index, birth, death, lower, upper, grid_shape);
        }
      }
      ++degree_index;
    }
  });
}

template <std::size_t N, typename dtype, typename index_type>
inline std::pair<std::vector<std::vector<index_type>>, std::vector<dtype>> merge_sparse_rank_accumulators(
    tbb::enumerable_thread_specific<packed_rank_sparse_accumulator<N, dtype, index_type>> &thread_atoms,
    const std::array<index_type, 1 + 2 * N> &key_shape) {
  packed_rank_sparse_accumulator<N, dtype, index_type> merged(key_shape);
  std::size_t total_size = 0;
  for (const auto &local_atoms : thread_atoms) total_size += local_atoms.size();
  merged.reserve(total_size);
  for (const auto &local_atoms : thread_atoms) merged.merge_from(local_atoms);

  const auto keys = merged.sorted_keys();
  std::vector<std::vector<index_type>> coordinates;
  std::vector<dtype> weights;
  coordinates.reserve(keys.size());
  weights.reserve(keys.size());

  for (const auto key : keys) {
    const auto coordinate = merged.unpack(key);
    const auto weight = merged.weight(key);
    coordinates.emplace_back(coordinate.begin(), coordinate.end());
    weights.push_back(weight);
  }
  return {coordinates, weights};
}

template <std::size_t N, class PersBackend, class MultiFiltration, typename dtype, typename index_type>
inline std::pair<std::vector<std::vector<index_type>>, std::vector<dtype>> compute_rank_signed_measure_by_slice_paths(
    Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend> &slicer,
    const std::array<index_type, N> &grid_shape,
    const std::vector<index_type> &degrees,
    const bool zero_pad,
    const bool ignore_inf) {
  if (degrees.size() == 0) return {{}, {}};
  if constexpr (N == 2) {
    const auto key_shape = rank_sparse_key_shape<N>(grid_shape, static_cast<index_type>(degrees.size()), zero_pad);
    using value_type = typename MultiFiltration::value_type;
    using ThreadSafe = typename Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend>::Thread_safe;
    ThreadSafe slicer_thread(slicer);
    tbb::enumerable_thread_specific<ThreadSafe> thread_locals(slicer_thread);
    packed_rank_sparse_accumulator<N, dtype, index_type> atom_accumulator(key_shape);
    tbb::enumerable_thread_specific<packed_rank_sparse_accumulator<N, dtype, index_type>> thread_atoms(atom_accumulator);

    const index_type X = grid_shape[0];
    const index_type Y = grid_shape[1];
    tbb::parallel_for(index_type{0}, X * Y, [&](index_type slice_path_index) {
      const index_type I = slice_path_index % X;
      const index_type J = slice_path_index / X;
      const std::array<index_type, N> lower{index_type{0}, J};
      const std::array<index_type, N> upper{I, static_cast<index_type>(Y - 1)};

      ThreadSafe &local_slicer = thread_locals.local();
      auto &slice_filtration = local_slicer.get_slice();
      const auto &filtration_values = local_slicer.get_filtration_values();
      for (std::size_t i = 0; i < filtration_values.size(); ++i) {
        const auto &filtration = filtration_values[i];
        value_type filtration_in_slice = MultiFiltration::T_inf;
        for (unsigned int generator = 0; generator < filtration.num_generators(); ++generator) {
          filtration_in_slice =
              std::min(filtration_in_slice,
                       get_slice_rank_filtration<index_type, value_type>(filtration(generator, 0),
                                                                         filtration(generator, 1),
                                                                         I,
                                                                         J));
        }
        slice_filtration[i] = filtration_in_slice;
      }

      using bc_type = typename Gudhi::multi_persistence::Slicer<MultiFiltration,
                                                                PersBackend>::template Multi_dimensional_flat_barcode<>;
      if (!local_slicer.persistence_computation_is_initialized()) [[unlikely]] {
        local_slicer.initialize_persistence_computation(ignore_inf);
      } else {
        local_slicer.update_persistence_computation(ignore_inf);
      }
      bc_type barcodes = local_slicer.template get_flat_barcode<true>();

      auto &sm_pts = thread_atoms.local();
      index_type degree_index = 0;
      for (auto degree : degrees) {
        if (degree < static_cast<index_type>(barcodes.size())) {
          const auto &barcode = barcodes[degree];
          for (const auto &bar : barcode) {
            const auto birth = static_cast<index_type>(bar[0]);
            const auto death = static_cast<index_type>(std::min(bar[1], static_cast<value_type>(I + Y)));
            emit_slice_path_bar_signed_measure_atoms<N, dtype>(
                sm_pts, degree_index, birth, death, lower, upper, grid_shape, zero_pad);
          }
        }
        ++degree_index;
      }
    });

    return merge_sparse_rank_accumulators<N, dtype, index_type>(thread_atoms, key_shape);
  }

  const auto axis_order = sparse_slice_path_axis_order<N>(grid_shape);
  const bool reorder_axes = !axis_order_is_identity<N>(axis_order);
  const auto ordered_shape = ordered_grid_shape<N>(grid_shape, axis_order);
  const auto slice_path_grid_shape = reorder_axes ? ordered_shape : grid_shape;
  const index_type total_slice_paths = slice_path_family_size<N>(slice_path_grid_shape);
  const auto key_shape = rank_sparse_key_shape<N>(grid_shape, static_cast<index_type>(degrees.size()), zero_pad);
  using value_type = typename MultiFiltration::value_type;
  using ThreadSafe = typename Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend>::Thread_safe;
  ThreadSafe slicer_thread(slicer);
  tbb::enumerable_thread_specific<ThreadSafe> thread_locals(slicer_thread);
  packed_rank_sparse_accumulator<N, dtype, index_type> atom_accumulator(key_shape);
  tbb::enumerable_thread_specific<packed_rank_sparse_accumulator<N, dtype, index_type>> thread_atoms(atom_accumulator);

  tbb::parallel_for(index_type{0}, total_slice_paths, [&](index_type slice_path_index) {
    std::array<index_type, N> lower{};
    std::array<index_type, N> upper{};
    slice_path_bounds_from_linear<N>(slice_path_index, slice_path_grid_shape, lower, upper);

    ThreadSafe &local_slicer = thread_locals.local();
    auto &slice_filtration = local_slicer.get_slice();
    const auto &filtration_values = local_slicer.get_filtration_values();
    for (std::size_t i = 0; i < filtration_values.size(); ++i) {
      const auto &filtration = filtration_values[i];
      value_type filtration_in_slice = MultiFiltration::T_inf;
      for (unsigned int generator = 0; generator < filtration.num_generators(); ++generator) {
        filtration_in_slice =
            std::min(filtration_in_slice,
                     reorder_axes ? get_slice_rank_filtration_on_ordered_slice_path<N>(filtration,
                                                                                        generator,
                                                                                        lower,
                                                                                        upper,
                                                                                        axis_order)
                                  : get_slice_rank_filtration_on_slice_path<N>(filtration, generator, lower, upper));
      }
      slice_filtration[i] = filtration_in_slice;
    }

    using bc_type = typename Gudhi::multi_persistence::Slicer<MultiFiltration,
                                                              PersBackend>::template Multi_dimensional_flat_barcode<>;
    if (!local_slicer.persistence_computation_is_initialized()) [[unlikely]] {
      local_slicer.initialize_persistence_computation(ignore_inf);
    } else {
      local_slicer.update_persistence_computation(ignore_inf);
    }
    bc_type barcodes = local_slicer.template get_flat_barcode<true>();

    auto &sm_pts = thread_atoms.local();
    index_type degree_index = 0;
    for (auto degree : degrees) {
      if (degree < static_cast<index_type>(barcodes.size())) {
        const auto &barcode = barcodes[degree];
        for (const auto &bar : barcode) {
          const auto birth = static_cast<index_type>(bar[0]);
          const auto death = static_cast<index_type>(
              std::min(bar[1], static_cast<value_type>(path_last_segment_stop<N>(upper, slice_path_grid_shape))));
          if (reorder_axes) {
            emit_ordered_slice_path_bar_signed_measure_atoms<N, dtype>(sm_pts,
                                                                       degree_index,
                                                                       birth,
                                                                       death,
                                                                       lower,
                                                                       upper,
                                                                       grid_shape,
                                                                       slice_path_grid_shape,
                                                                       axis_order,
                                                                       zero_pad);
          } else {
            emit_slice_path_bar_signed_measure_atoms<N, dtype>(
                sm_pts, degree_index, birth, death, lower, upper, grid_shape, zero_pad);
          }
        }
      }
      ++degree_index;
    }
  });

  return merge_sparse_rank_accumulators<N, dtype, index_type>(thread_atoms, key_shape);
}

template <class PersBackend, class MultiFiltration, typename dtype = int, typename indices_type = int>
std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> compute_rank_signed_measure_sparse_python(
    Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend> &slicer,
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
      if (grid_shape.size() != 2) [[unlikely]] throw std::runtime_error("Internal error: invalid rank grid shape.");
      const std::array<indices_type, 2> parameter_shape = {grid_shape[0], grid_shape[1]};
      std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> out;
      arena.execute([&] {
        out = compute_rank_signed_measure_by_slice_paths<2, PersBackend, MultiFiltration, dtype, indices_type>(
            slicer, parameter_shape, degrees, zero_pad, ignore_inf);
      });
      return out;
    }
    case 3: {
      if (grid_shape.size() != 3) [[unlikely]] throw std::runtime_error("Internal error: invalid rank grid shape.");
      const std::array<indices_type, 3> parameter_shape = {grid_shape[0], grid_shape[1], grid_shape[2]};
      std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> out;
      arena.execute([&] {
        out = compute_rank_signed_measure_by_slice_paths<3, PersBackend, MultiFiltration, dtype, indices_type>(
            slicer, parameter_shape, degrees, zero_pad, ignore_inf);
      });
      return out;
    }
    case 4: {
      if (grid_shape.size() != 4) [[unlikely]] throw std::runtime_error("Internal error: invalid rank grid shape.");
      const std::array<indices_type, 4> parameter_shape = {grid_shape[0], grid_shape[1], grid_shape[2], grid_shape[3]};
      std::pair<std::vector<std::vector<indices_type>>, std::vector<dtype>> out;
      arena.execute([&] {
        out = compute_rank_signed_measure_by_slice_paths<4, PersBackend, MultiFiltration, dtype, indices_type>(
            slicer, parameter_shape, degrees, zero_pad, ignore_inf);
      });
      return out;
    }
    default:
      throw std::runtime_error("Sparse rank signed measure is implemented for 2, 3, and 4 parameters.");
  }
}

template <class PersBackend, class MultiFiltration, typename dtype, typename indices_type>
void compute_rank_invariant_python(Gudhi::multi_persistence::Slicer<MultiFiltration, PersBackend> &slicer,
                                   dtype *data_ptr,
                                   const std::vector<indices_type> &grid_shape,
                                   const std::vector<indices_type> &degrees,
                                   indices_type n_jobs,
                                   const bool ignore_inf) {
  if (degrees.size() == 0) return;
  const auto num_parameters = slicer.get_number_of_parameters();
  if (grid_shape.size() != 1 + 2 * num_parameters) [[unlikely]] {
    throw std::runtime_error("Internal error: invalid rank invariant tensor shape.");
  }
  if constexpr (false) {
    std::cout << "ignore_inf " << ignore_inf << std::endl;
  }

  oneapi::tbb::task_arena arena(PersBackend::is_vine ? 1 : n_jobs);  // limits the number of threads
  switch (num_parameters) {
    case 2: {
      const auto shape = vector_to_full_rank_shape<2>(grid_shape);
      auto container = Gudhi::Simple_mdspan(data_ptr,
                                            static_cast<std::size_t>(shape[0]),
                                            static_cast<std::size_t>(shape[1]),
                                            static_cast<std::size_t>(shape[2]),
                                            static_cast<std::size_t>(shape[3]),
                                            static_cast<std::size_t>(shape[4]));  // assumes zero tensor
      arena.execute([&] { compute_2d_rank_invariant(slicer, container, shape, degrees, false, ignore_inf); });
      break;
    }
    case 3: {
      mobius_inversion::dense_tensor_view<dtype, indices_type> container(data_ptr, grid_shape);  // assumes zero tensor
      const auto shape = vector_to_full_rank_shape<3>(grid_shape);
      const std::array<indices_type, 3> parameter_shape = {shape[1], shape[2], shape[3]};
      arena.execute(
          [&] { compute_nd_rank_invariant_by_slice_paths<3>(slicer, container, parameter_shape, degrees, ignore_inf); });
      break;
    }
    case 4: {
      mobius_inversion::dense_tensor_view<dtype, indices_type> container(data_ptr, grid_shape);  // assumes zero tensor
      const auto shape = vector_to_full_rank_shape<4>(grid_shape);
      const std::array<indices_type, 4> parameter_shape = {shape[1], shape[2], shape[3], shape[4]};
      arena.execute(
          [&] { compute_nd_rank_invariant_by_slice_paths<4>(slicer, container, parameter_shape, degrees, ignore_inf); });
      break;
    }
    default:
      throw std::runtime_error("Rank invariant is implemented for 2, 3, and 4 parameters.");
  }

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

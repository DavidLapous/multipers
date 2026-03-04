#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "contiguous_slicer_bridge.hpp"

namespace multipers {

template <typename index_type>
struct rhomboid_tiling_interface_input {
  std::vector<std::vector<double> > points;
};

template <typename index_type>
struct rhomboid_tiling_interface_output {
  std::vector<std::pair<double, double> > filtration_values;
  std::vector<std::vector<index_type> > boundaries;
  std::vector<int> dimensions;
};

inline bool rhomboid_tiling_interface_available();

template <typename index_type>
rhomboid_tiling_interface_output<index_type> rhomboid_tiling_to_slicer_interface(
    const rhomboid_tiling_interface_input<index_type>& input,
    int k_max,
    int degree,
    bool verbose_output = false);

template <typename index_type>
contiguous_f64_complex rhomboid_tiling_to_contiguous_slicer_interface(
    const rhomboid_tiling_interface_input<index_type>& input,
    int k_max,
    int degree,
    bool verbose_output = false);

}  // namespace multipers

#ifndef MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE
#define MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE 0
#endif

#if !MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE && __has_include(<CGAL/Exact_predicates_exact_constructions_kernel.h>) && \
    __has_include(<dimensional_traits_2.h>) && __has_include(<dimensional_traits_3.h>) && __has_include(<rhomboid_tiling.h>)
#define MULTIPERS_HAS_RHOMBOID_TILING_INTERFACE 1
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/number_utils.h>

#include <bifiltration_cell.h>
#include <dimensional_traits_2.h>
#include <dimensional_traits_3.h>
#include <rhomboid_tiling.h>
#else
#define MULTIPERS_HAS_RHOMBOID_TILING_INTERFACE 0
#endif

namespace multipers {

inline bool rhomboid_tiling_interface_available() { return MULTIPERS_HAS_RHOMBOID_TILING_INTERFACE; }

#if MULTIPERS_HAS_RHOMBOID_TILING_INTERFACE

namespace detail {

template <typename FT>
inline double firep_round_radius(const FT& value) {
  return CGAL::to_double(value);
}

template <typename index_type, typename FT>
inline rhomboid_tiling_interface_output<index_type> bifiltration_to_slicer_output(
    const std::vector<BifiltrationCell<FT> >& bifiltration,
    int repr_dimension,
    int shift_dimensions) {
  std::size_t no_dim_plus_1 = 0;
  std::size_t no_dim = 0;
  std::size_t no_dim_minus_1 = 0;
  int max_id_dim_minus_1 = -1;
  int max_id_dim = -1;

  for (const auto& bc : bifiltration) {
    if (bc.d == repr_dimension - 1) {
      ++no_dim_minus_1;
      if (bc.id > max_id_dim_minus_1) {
        max_id_dim_minus_1 = bc.id;
      }
    }
    if (bc.d == repr_dimension) {
      ++no_dim;
      if (bc.id > max_id_dim) {
        max_id_dim = bc.id;
      }
    }
    if (bc.d == repr_dimension + 1) {
      ++no_dim_plus_1;
    }
  }

  const index_type invalid_index = std::numeric_limits<index_type>::max();
  std::vector<index_type> new_id_d_minus_1(
      max_id_dim_minus_1 >= 0 ? static_cast<std::size_t>(max_id_dim_minus_1 + 1) : 0,
      invalid_index);
  std::vector<index_type> new_id_d(
      max_id_dim >= 0 ? static_cast<std::size_t>(max_id_dim + 1) : 0,
      invalid_index);

  index_type next_dim_minus_1 = 0;
  index_type next_dim = 0;
  for (const auto& bc : bifiltration) {
    if (bc.id < 0) {
      throw std::invalid_argument("rhomboid_tiling conversion failed: negative cell id.");
    }
    if (bc.d == repr_dimension - 1) {
      new_id_d_minus_1[static_cast<std::size_t>(bc.id)] = next_dim_minus_1;
      ++next_dim_minus_1;
    }
    if (bc.d == repr_dimension) {
      new_id_d[static_cast<std::size_t>(bc.id)] = next_dim;
      ++next_dim;
    }
  }

  const std::size_t number_of_cells = no_dim_plus_1 + no_dim + no_dim_minus_1;
  std::vector<std::size_t> counts = {no_dim_plus_1, no_dim, no_dim_minus_1};
  if (shift_dimensions != 0) {
    const long long resized = static_cast<long long>(counts.size()) + static_cast<long long>(shift_dimensions);
    if (resized <= 0) {
      return rhomboid_tiling_interface_output<index_type>();
    }
    counts.resize(static_cast<std::size_t>(resized), 0);
  }

  std::size_t dim_it = 0;
  while (dim_it < counts.size() && counts[dim_it] == 0) {
    ++dim_it;
  }
  if (dim_it == counts.size() || number_of_cells == 0) {
    return rhomboid_tiling_interface_output<index_type>();
  }

  rhomboid_tiling_interface_output<index_type> out;
  const double minus_inf = -std::numeric_limits<double>::infinity();
  out.filtration_values.assign(number_of_cells, std::make_pair(minus_inf, minus_inf));
  out.boundaries.resize(number_of_cells);
  out.dimensions.assign(number_of_cells, 0);

  std::size_t shift = counts[dim_it];
  std::size_t next_shift = (dim_it < counts.size() - 1) ? counts[dim_it + 1] : 0;
  std::size_t i = 0;

  auto append_cell = [&](const BifiltrationCell<FT>& bc, const std::vector<index_type>& lower_dim_new_id_map) {
    if (dim_it >= counts.size() || i >= number_of_cells) {
      return;
    }

    out.filtration_values[i] = std::make_pair(firep_round_radius(bc.r), -static_cast<double>(bc.k));
    std::vector<index_type> boundary;
    boundary.reserve(bc.boundary.size());
    for (const int idx : bc.boundary) {
      if (idx < 0 || static_cast<std::size_t>(idx) >= lower_dim_new_id_map.size()) {
        throw std::invalid_argument("rhomboid_tiling conversion failed: boundary index is outside expected dimension block.");
      }
      const index_type local_id = lower_dim_new_id_map[static_cast<std::size_t>(idx)];
      if (local_id == invalid_index) {
        throw std::invalid_argument("rhomboid_tiling conversion failed: boundary index is outside expected dimension block.");
      }
      const std::size_t shifted_idx = static_cast<std::size_t>(local_id) + shift;
      if (shifted_idx >= number_of_cells) {
        throw std::invalid_argument("rhomboid_tiling conversion failed: shifted boundary index out of range.");
      }
      boundary.push_back(static_cast<index_type>(number_of_cells - 1 - shifted_idx));
    }
    std::sort(boundary.begin(), boundary.end());
    out.boundaries[i] = std::move(boundary);
    out.dimensions[i] = static_cast<int>(counts.size() - 1 - dim_it);

    --counts[dim_it];
    while (dim_it < counts.size() && counts[dim_it] == 0) {
      ++dim_it;
      if (dim_it != counts.size()) {
        shift += next_shift;
        next_shift = dim_it < counts.size() - 1 ? counts[dim_it + 1] : 0;
      }
    }
    ++i;
  };

  for (const auto& bc : bifiltration) {
    if (bc.d == repr_dimension + 1) {
      append_cell(bc, new_id_d);
    }
  }
  for (const auto& bc : bifiltration) {
    if (bc.d == repr_dimension) {
      append_cell(bc, new_id_d_minus_1);
    }
  }

  std::reverse(out.filtration_values.begin(), out.filtration_values.end());
  std::reverse(out.boundaries.begin(), out.boundaries.end());
  std::reverse(out.dimensions.begin(), out.dimensions.end());
  return out;
}

template <class Dt, typename index_type>
inline rhomboid_tiling_interface_output<index_type> build_from_points(
    const rhomboid_tiling_interface_input<index_type>& input,
    int k_max,
    int degree,
    bool verbose_output) {
  using clock = std::chrono::steady_clock;
  auto t_total_start = clock::now();

  using Point = typename Dt::Point;

  std::vector<Point> points;
  points.reserve(input.points.size());
  for (const auto& coords : input.points) {
    if (coords.size() != static_cast<std::size_t>(Dt::dimension)) {
      throw std::invalid_argument("rhomboid_tiling interface expects all points to have the same ambient dimension.");
    }
    points.push_back(Dt::make_point(coords));
  }
  auto t_points_ready = clock::now();

  if (points.size() < 2) {
    throw std::invalid_argument("rhomboid_tiling interface expects at least two input points.");
  }

  int max_order = k_max;
  if (max_order > static_cast<int>(points.size())) {
    max_order = static_cast<int>(points.size()) - 1;
  }
  if (max_order <= 0) {
    throw std::invalid_argument("rhomboid_tiling interface expects k_max > 0.");
  }

  auto t_core_start = clock::now();
  RhomboidTiling<Dt> rhomboid_tiling(points, max_order);
  const auto bifiltration = rhomboid_tiling.get_bifiltration();
  auto t_core_done = clock::now();

  auto t_convert_start = clock::now();
  const int shift_dimensions = degree <= 0 ? -1 : degree - 1;
  auto out = bifiltration_to_slicer_output<index_type>(bifiltration, degree, shift_dimensions);
  auto t_convert_done = clock::now();

  if (verbose_output) {
    const double points_sec = std::chrono::duration<double>(t_points_ready - t_total_start).count();
    const double core_sec = std::chrono::duration<double>(t_core_done - t_core_start).count();
    const double convert_sec = std::chrono::duration<double>(t_convert_done - t_convert_start).count();
    const double total_sec = std::chrono::duration<double>(t_convert_done - t_total_start).count();
    std::cout << "[multipers.rhomboid][timing] points_to_kernel=" << points_sec
              << "s rhomboid_core=" << core_sec << "s interface_convert=" << convert_sec
              << "s total=" << total_sec << "s bifiltration_cells=" << bifiltration.size()
              << " output_cells=" << out.dimensions.size() << std::endl;
  }

  return out;
}

}  // namespace detail

template <typename index_type>
rhomboid_tiling_interface_output<index_type> rhomboid_tiling_to_slicer_interface(
    const rhomboid_tiling_interface_input<index_type>& input,
    int k_max,
    int degree,
    bool verbose_output) {
  if (input.points.empty()) {
    throw std::invalid_argument("rhomboid_tiling interface expects a non-empty point cloud.");
  }

  if (k_max <= 0) {
    throw std::invalid_argument("rhomboid_tiling interface expects k_max > 0.");
  }

  const std::size_t point_dim = input.points.front().size();
  if (point_dim != 2 && point_dim != 3) {
    throw std::invalid_argument("rhomboid_tiling interface supports only 2D and 3D point clouds.");
  }
  for (const auto& point : input.points) {
    if (point.size() != point_dim) {
      throw std::invalid_argument("rhomboid_tiling interface expects all points to have the same ambient dimension.");
    }
  }

  using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
  using Dt2 = DimensionalTraits_2<Kernel>;
  using Dt3 = DimensionalTraits_3<Kernel>;
  if (point_dim == 2) {
    return detail::build_from_points<Dt2, index_type>(input, k_max, degree, verbose_output);
  }
  return detail::build_from_points<Dt3, index_type>(input, k_max, degree, verbose_output);
}

template <typename index_type>
contiguous_f64_complex rhomboid_tiling_to_contiguous_slicer_interface(
    const rhomboid_tiling_interface_input<index_type>& input,
    int k_max,
    int degree,
    bool verbose_output) {
  auto out = rhomboid_tiling_to_slicer_interface<index_type>(input, k_max, degree, verbose_output);
  return build_contiguous_f64_slicer_from_output<index_type>(out.filtration_values, out.boundaries, out.dimensions);
}

#else

template <typename index_type>
rhomboid_tiling_interface_output<index_type> rhomboid_tiling_to_slicer_interface(
    const rhomboid_tiling_interface_input<index_type>&,
    int,
    int,
    bool) {
  throw std::runtime_error("rhomboid_tiling in-memory interface is not available at compile time. Install/checkout "
                           "rhomboidtiling + CGAL headers and rebuild.");
}

template <typename index_type>
contiguous_f64_complex rhomboid_tiling_to_contiguous_slicer_interface(
    const rhomboid_tiling_interface_input<index_type>&,
    int,
    int,
    bool) {
  throw std::runtime_error("rhomboid_tiling in-memory interface is not available at compile time. Install/checkout "
                           "rhomboidtiling + CGAL headers and rebuild.");
}

#endif

}  // namespace multipers

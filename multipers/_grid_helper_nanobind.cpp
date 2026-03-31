#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nanobind_array_utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace mpgnb {

using multipers::nanobind_utils::owned_array;

template <typename T>
std::vector<size_t> shape_of(const nb::ndarray<nb::numpy, T, nb::c_contig>& array) {
  std::vector<size_t> shape(array.ndim());
  for (size_t i = 0; i < shape.size(); ++i) {
    shape[i] = array.shape(i);
  }
  return shape;
}

inline std::vector<size_t> row_major_strides(const std::vector<size_t>& shape) {
  std::vector<size_t> strides(shape.size(), 1);
  for (ptrdiff_t i = static_cast<ptrdiff_t>(shape.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i) + 1] * shape[static_cast<size_t>(i) + 1];
  }
  return strides;
}

template <typename T>
std::vector<T> unique_sorted(std::vector<T>&& values) {
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

template <typename T>
std::vector<int64_t> regular_closest_1d_indices_impl(
    const nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig>& sorted_values,
    size_t resolution,
    bool unique) {
  if (sorted_values.shape(0) == 0) {
    return {};
  }
  if (resolution == 0) {
    return {};
  }
  if (resolution == 1) {
    return {0};
  }

  const T* sorted = sorted_values.data();
  const size_t num_values = sorted_values.shape(0);
  const double lo = static_cast<double>(sorted[0]);
  const double hi = static_cast<double>(sorted[num_values - 1]);
  const double step = (hi - lo) / static_cast<double>(resolution - 1);
  std::vector<int64_t> out;
  out.reserve(resolution);
  std::vector<T> selected_values;
  if (unique) {
    selected_values.reserve(resolution);
  }
  for (size_t i = 0; i < resolution; ++i) {
    const double target = lo + static_cast<double>(i) * step;
    auto right = std::lower_bound(sorted, sorted + num_values, static_cast<T>(target));
    size_t chosen_idx = 0;
    if (right == sorted) {
      chosen_idx = 0;
    } else if (right == sorted + num_values) {
      chosen_idx = num_values - 1;
    } else {
      const size_t right_idx = static_cast<size_t>(right - sorted);
      const size_t left_idx = right_idx - 1;
      const T right_value = sorted[right_idx];
      const T left_value = sorted[left_idx];
      chosen_idx =
          std::abs(target - static_cast<double>(left_value)) <= std::abs(static_cast<double>(right_value) - target)
              ? left_idx
              : right_idx;
    }

    if (unique) {
      const T chosen_value = sorted[chosen_idx];
      if (!selected_values.empty() && selected_values.back() == chosen_value) {
        continue;
      }
      selected_values.push_back(chosen_value);
    }
    out.push_back(static_cast<int64_t>(chosen_idx));
  }
  return out;
}

template <typename T>
std::vector<nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig>> cast_grid_sequence(const nb::tuple& grid) {
  std::vector<nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig>> out;
  out.reserve(grid.size());
  for (size_t i = 0; i < grid.size(); ++i) {
    out.push_back(nb::cast<nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig>>(grid[i]));
  }
  return out;
}

template <typename Point, typename Weight>
std::vector<int64_t> grid_coordinates_impl(
    const nb::ndarray<nb::numpy, const Point, nb::ndim<2>, nb::c_contig>& points,
    const std::vector<nb::ndarray<nb::numpy, const Point, nb::ndim<1>, nb::c_contig>>& grid) {
  const size_t num_points = points.shape(0);
  const size_t num_parameters = points.shape(1);
  if (grid.size() != num_parameters) {
    throw std::runtime_error("Grid dimension does not match point dimension.");
  }
  std::vector<int64_t> coords(num_points * num_parameters, 0);
  for (size_t p = 0; p < num_parameters; ++p) {
    const Point* grid_data = grid[p].data();
    const size_t grid_size = grid[p].shape(0);
    for (size_t i = 0; i < num_points; ++i) {
      const Point value = points(i, p);
      coords[i * num_parameters + p] =
          static_cast<int64_t>(std::lower_bound(grid_data, grid_data + grid_size, value) - grid_data);
    }
  }
  return coords;
}

template <typename Point, typename Weight>
std::vector<int32_t> integrate_measure_impl(
    const nb::ndarray<nb::numpy, const Point, nb::ndim<2>, nb::c_contig>& points,
    const nb::ndarray<nb::numpy, const Weight, nb::ndim<1>, nb::c_contig>& weights,
    const std::vector<nb::ndarray<nb::numpy, const Point, nb::ndim<1>, nb::c_contig>>& grid) {
  const size_t num_points = points.shape(0);
  const size_t num_parameters = points.shape(1);
  if (weights.shape(0) != num_points) {
    throw std::runtime_error("Weights do not match number of points.");
  }
  if (grid.size() != num_parameters) {
    throw std::runtime_error("Grid dimension does not match point dimension.");
  }

  std::vector<size_t> shape(num_parameters);
  size_t total_size = 1;
  for (size_t p = 0; p < num_parameters; ++p) {
    shape[p] = grid[p].shape(0);
    total_size *= shape[p];
  }
  auto strides = row_major_strides(shape);
  std::vector<int32_t> out(total_size, 0);

  for (size_t i = 0; i < num_points; ++i) {
    size_t linear_index = 0;
    bool inside = true;
    for (size_t p = 0; p < num_parameters; ++p) {
      const Point* grid_data = grid[p].data();
      const size_t grid_size = shape[p];
      const size_t coord =
          static_cast<size_t>(std::lower_bound(grid_data, grid_data + grid_size, points(i, p)) - grid_data);
      if (coord >= grid_size) {
        inside = false;
        break;
      }
      linear_index += coord * strides[p];
    }
    if (inside) {
      out[linear_index] += static_cast<int32_t>(weights(i));
    }
  }

  for (size_t axis = 0; axis < num_parameters; ++axis) {
    const size_t axis_stride = strides[axis];
    const size_t axis_size = shape[axis];
    const size_t block = axis_stride * axis_size;
    const size_t repeat = total_size / block;
    for (size_t rep = 0; rep < repeat; ++rep) {
      const size_t base = rep * block;
      for (size_t offset = 0; offset < axis_stride; ++offset) {
        int32_t running = 0;
        for (size_t i = 0; i < axis_size; ++i) {
          const size_t idx = base + i * axis_stride + offset;
          running += out[idx];
          out[idx] = running;
        }
      }
    }
  }

  return out;
}

template <typename T>
void apply_threshold_last_plane(T* data,
                                const std::vector<size_t>& shape,
                                const std::vector<size_t>& strides,
                                size_t axis) {
  const size_t total_size = strides[0] * shape[0];
  const size_t axis_stride = strides[axis];
  const size_t axis_size = shape[axis];
  const size_t block = axis_stride * axis_size;
  const size_t repeat = total_size / block;
  for (size_t rep = 0; rep < repeat; ++rep) {
    const size_t base = rep * block;
    for (size_t offset = 0; offset < axis_stride; ++offset) {
      data[base + (axis_size - 1) * axis_stride + offset] = static_cast<T>(0);
    }
  }
}

template <typename T>
void signed_betti_inplace_impl(nb::ndarray<nb::numpy, T, nb::c_contig> array, bool threshold) {
  auto shape = shape_of(array);
  if (shape.empty()) {
    return;
  }
  auto strides = row_major_strides(shape);
  T* data = array.data();
  const size_t total_size = strides[0] * shape[0];

  if (threshold) {
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      apply_threshold_last_plane(data, shape, strides, axis);
    }
  }

  for (size_t axis = 0; axis < shape.size(); ++axis) {
    const size_t axis_stride = strides[axis];
    const size_t axis_size = shape[axis];
    const size_t block = axis_stride * axis_size;
    const size_t repeat = total_size / block;
    for (size_t rep = 0; rep < repeat; ++rep) {
      const size_t base = rep * block;
      for (size_t offset = 0; offset < axis_stride; ++offset) {
        for (size_t i = axis_size - 1; i > 0; --i) {
          const size_t idx = base + i * axis_stride + offset;
          data[idx] -= data[idx - axis_stride];
        }
      }
    }
  }
}

}  // namespace mpgnb

NB_MODULE(_grid_helper_nanobind, m) {
  m.def(
      "regular_closest_1d_indices",
      [](nb::ndarray<nb::numpy, const float, nb::ndim<1>, nb::c_contig> values, int resolution, bool unique) {
        auto out = mpgnb::regular_closest_1d_indices_impl(values, static_cast<size_t>(resolution), unique);
        return mpgnb::owned_array<int64_t>(std::move(out), {out.size()});
      },
      "values"_a,
      "resolution"_a,
      "unique"_a = true);
  m.def(
      "regular_closest_1d_indices",
      [](nb::ndarray<nb::numpy, const double, nb::ndim<1>, nb::c_contig> values, int resolution, bool unique) {
        auto out = mpgnb::regular_closest_1d_indices_impl(values, static_cast<size_t>(resolution), unique);
        return mpgnb::owned_array<int64_t>(std::move(out), {out.size()});
      },
      "values"_a,
      "resolution"_a,
      "unique"_a = true);

  m.def(
      "push_to_grid_coordinates",
      [](nb::ndarray<nb::numpy, const float, nb::ndim<2>, nb::c_contig> points, nb::tuple grid) {
        auto grids = mpgnb::cast_grid_sequence<float>(grid);
        auto coords = mpgnb::grid_coordinates_impl<float, int32_t>(points, grids);
        return mpgnb::owned_array<int64_t>(std::move(coords), {points.shape(0), points.shape(1)});
      },
      "points"_a,
      "grid"_a);
  m.def(
      "push_to_grid_coordinates",
      [](nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> points, nb::tuple grid) {
        auto grids = mpgnb::cast_grid_sequence<double>(grid);
        auto coords = mpgnb::grid_coordinates_impl<double, int32_t>(points, grids);
        return mpgnb::owned_array<int64_t>(std::move(coords), {points.shape(0), points.shape(1)});
      },
      "points"_a,
      "grid"_a);

  m.def(
      "integrate_measure",
      [](nb::ndarray<nb::numpy, const float, nb::ndim<2>, nb::c_contig> points,
         nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> weights,
         nb::tuple grid) {
        auto grids = mpgnb::cast_grid_sequence<float>(grid);
        auto out = mpgnb::integrate_measure_impl(points, weights, grids);
        std::vector<size_t> shape;
        shape.reserve(grid.size());
        for (const auto& g : grids) {
          shape.push_back(g.shape(0));
        }
        return mpgnb::owned_array<int32_t>(std::move(out), shape);
      },
      "points"_a,
      "weights"_a,
      "grid"_a);
  m.def(
      "integrate_measure",
      [](nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> points,
         nb::ndarray<nb::numpy, const int64_t, nb::ndim<1>, nb::c_contig> weights,
         nb::tuple grid) {
        auto grids = mpgnb::cast_grid_sequence<double>(grid);
        auto out = mpgnb::integrate_measure_impl(points, weights, grids);
        std::vector<size_t> shape;
        shape.reserve(grid.size());
        for (const auto& g : grids) {
          shape.push_back(g.shape(0));
        }
        return mpgnb::owned_array<int32_t>(std::move(out), shape);
      },
      "points"_a,
      "weights"_a,
      "grid"_a);
  m.def(
      "integrate_measure",
      [](nb::ndarray<nb::numpy, const float, nb::ndim<2>, nb::c_contig> points,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> weights,
         nb::tuple grid) {
        auto grids = mpgnb::cast_grid_sequence<float>(grid);
        auto out = mpgnb::integrate_measure_impl(points, weights, grids);
        std::vector<size_t> shape;
        shape.reserve(grid.size());
        for (const auto& g : grids) {
          shape.push_back(g.shape(0));
        }
        return mpgnb::owned_array<int32_t>(std::move(out), shape);
      },
      "points"_a,
      "weights"_a,
      "grid"_a);
  m.def(
      "integrate_measure",
      [](nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> points,
         nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> weights,
         nb::tuple grid) {
        auto grids = mpgnb::cast_grid_sequence<double>(grid);
        auto out = mpgnb::integrate_measure_impl(points, weights, grids);
        std::vector<size_t> shape;
        shape.reserve(grid.size());
        for (const auto& g : grids) {
          shape.push_back(g.shape(0));
        }
        return mpgnb::owned_array<int32_t>(std::move(out), shape);
      },
      "points"_a,
      "weights"_a,
      "grid"_a);

  m.def(
      "signed_betti_inplace",
      [](nb::ndarray<nb::numpy, int32_t, nb::c_contig> array, bool threshold) {
        mpgnb::signed_betti_inplace_impl<int32_t>(array, threshold);
        return array;
      },
      "array"_a,
      "threshold"_a = false);
  m.def(
      "signed_betti_inplace",
      [](nb::ndarray<nb::numpy, int64_t, nb::c_contig> array, bool threshold) {
        mpgnb::signed_betti_inplace_impl<int64_t>(array, threshold);
        return array;
      },
      "array"_a,
      "threshold"_a = false);
}

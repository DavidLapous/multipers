#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(GUDHI_USE_TBB)
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#include <gudhi/Multi_persistence/Line.h>
#include <python_interfaces/numpy_utils.h>

#include "nanobind_array_utils.hpp"

namespace nb = nanobind;

namespace mpnb {

using multipers::nanobind_utils::owned_array;

template <typename Bars>
inline void insert_landscape_barcode_values(std::vector<double>& out,
                                            std::size_t base,
                                            std::size_t plane_size,
                                            const Bars& bars,
                                            double t,
                                            const std::vector<std::int32_t>& ks,
                                            std::vector<double>& top) {
  std::fill(top.begin(), top.end(), 0.0);
  auto* data = bars.data();
  for (std::size_t b = 0; b < bars.size(); ++b) {
    const double value =
        std::max(0.0, std::min(t - static_cast<double>(data[b][0]), static_cast<double>(data[b][1]) - t));
    if (value <= top.back()) continue;
    auto it = std::upper_bound(top.begin(), top.end(), value, std::greater<double>{});
    top.insert(it, value);
    top.pop_back();
  }
  for (std::size_t k = 0; k < ks.size(); ++k) {
    out[k * plane_size + base] = top[static_cast<std::size_t>(ks[k])];
  }
}

struct Landscape_grid_line_start {
  std::size_t i;
  std::size_t j;
};

inline std::vector<Landscape_grid_line_start> landscape_grid_line_starts(std::size_t nx,
                                                                         std::size_t ny,
                                                                         std::size_t stride_i,
                                                                         std::size_t stride_j) {
  std::vector<Landscape_grid_line_start> starts;
  starts.reserve(std::min(stride_i, nx) * ny + (stride_i < nx ? (nx - stride_i) * std::min(stride_j, ny) : 0));
  for (std::size_t i = 0; i < std::min(stride_i, nx); ++i) {
    for (std::size_t j = 0; j < ny; ++j) starts.push_back({i, j});
  }
  for (std::size_t i = stride_i; i < nx; ++i) {
    for (std::size_t j = 0; j < std::min(stride_j, ny); ++j) starts.push_back({i, j});
  }
  return starts;
}

template <typename Desc, typename Wrapper, typename Value>
nb::ndarray<nb::numpy, double> landscapes_on_grid(Wrapper& self,
                                                  nb::ndarray<const Value, nb::ndim<1>, nb::c_contig> xgrid,
                                                  nb::ndarray<const Value, nb::ndim<1>, nb::c_contig> ygrid,
                                                  nb::ndarray<const Value, nb::ndim<1>, nb::c_contig> direction,
                                                  std::size_t stride_i,
                                                  std::size_t stride_j,
                                                  double dt,
                                                  int degree,
                                                  nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> ks_array,
                                                  int n_jobs,
                                                  bool ignore_infinite_filtration_values) {
  const std::size_t nx = xgrid.shape(0);
  const std::size_t ny = ygrid.shape(0);
  if (nx == 0 || ny == 0) throw nb::value_error("Landscape grid axes must be non-empty.");
  if (direction.shape(0) != 2) throw nb::value_error("Landscape direction must be two-dimensional.");
  if (stride_i == 0 || stride_j == 0) throw nb::value_error("Landscape grid strides must be positive.");
  if (!std::isfinite(dt) || dt <= 0.0) throw nb::value_error("Landscape grid step must be finite and positive.");
  if (nx > std::numeric_limits<std::size_t>::max() / ny) throw nb::value_error("Landscape output grid is too large.");

  auto x_range = make_element_range(xgrid.data(), xgrid.view(), false);
  auto y_range = make_element_range(ygrid.data(), ygrid.view(), false);
  auto direction_range = make_element_range(direction.data(), direction.view(), false);
  std::vector<Value> x_values(x_range.begin(), x_range.end());
  std::vector<Value> y_values(y_range.begin(), y_range.end());
  std::vector<Value> direction_vec(direction_range.begin(), direction_range.end());

  for (Value value : x_values) {
    if (!std::isfinite(static_cast<double>(value))) throw nb::value_error("Landscape x-grid must be finite.");
  }
  for (Value value : y_values) {
    if (!std::isfinite(static_cast<double>(value))) throw nb::value_error("Landscape y-grid must be finite.");
  }
  for (Value value : direction_vec) {
    const double direction_value = static_cast<double>(value);
    if (!std::isfinite(direction_value)) throw nb::value_error("Landscape direction must be finite.");
    if (direction_value <= 0.0) throw nb::value_error("Landscape direction must be strictly positive.");
  }

  std::vector<std::int32_t> ks(ks_array.data(), ks_array.data() + ks_array.shape(0));
  std::int32_t need = 0;
  for (std::int32_t k : ks) {
    if (k < 0) throw nb::value_error("Landscape ks must be nonnegative.");
    need = std::max(need, k + 1);
  }
  const std::size_t plane_size = nx * ny;
  std::vector<double> out(ks.size() * plane_size, 0.0);
  if (ks.empty()) return owned_array<double>(std::move(out), {ks.size(), nx, ny});

  const std::vector<Landscape_grid_line_start> starts = landscape_grid_line_starts(nx, ny, stride_i, stride_j);
  auto compute_line = [&](auto& slicer, std::size_t line, bool& initialized, std::vector<double>& top) {
    const auto [i0, j0] = starts[line];
    const std::size_t length = std::min((nx - 1 - i0) / stride_i + 1, (ny - 1 - j0) / stride_j + 1);
    std::vector<Value> basepoint{x_values[i0], y_values[j0]};
    slicer.push_to(Gudhi::multi_persistence::Line<Value>(basepoint, direction_vec));
    if constexpr (Desc::is_vine) {
      if (initialized) {
        slicer.update_persistence_computation(ignore_infinite_filtration_values);
      } else {
        slicer.initialize_persistence_computation(ignore_infinite_filtration_values);
        initialized = true;
      }
    } else {
      slicer.initialize_persistence_computation(ignore_infinite_filtration_values);
    }
    auto barcode = slicer.template get_flat_barcode<true, Value, false>();
    if (degree < 0 || static_cast<std::size_t>(degree) >= barcode.size()) {
      throw std::out_of_range("Landscape degree is outside barcode degree range.");
    }
    const auto& bars = barcode[static_cast<std::size_t>(degree)];
    for (std::size_t step = 0; step < length; ++step) {
      const std::size_t i = i0 + step * stride_i;
      const std::size_t j = j0 + step * stride_j;
      insert_landscape_barcode_values(out, i * ny + j, plane_size, bars, dt * static_cast<double>(step), ks, top);
    }
  };

  {
    nb::gil_scoped_release release;
#if defined(GUDHI_USE_TBB)
    auto compute_range = [&](std::size_t begin, std::size_t end) {
      auto slicer = self.truc.weak_copy();
      bool initialized = false;
      std::vector<double> top(static_cast<std::size_t>(need), 0.0);
      for (std::size_t line = begin; line < end; ++line) compute_line(slicer, line, initialized, top);
    };
    if (n_jobs == 1) {
      compute_range(0, starts.size());
    } else {
      const std::size_t target_chunks = n_jobs > 0 ? static_cast<std::size_t>(n_jobs) * 4 : std::size_t(64);
      const std::size_t grain_size = std::max<std::size_t>(1, (starts.size() + target_chunks - 1) / target_chunks);
      auto run = [&] {
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, starts.size(), grain_size), [&](const auto& range) {
          tbb::this_task_arena::isolate([&] { compute_range(range.begin(), range.end()); });
        });
      };
      if (n_jobs > 0) {
        tbb::task_arena arena(n_jobs);
        arena.execute(run);
      } else {
        run();
      }
    }
#else
    auto slicer = self.truc.weak_copy();
    bool initialized = false;
    std::vector<double> top(static_cast<std::size_t>(need), 0.0);
    for (std::size_t line = 0; line < starts.size(); ++line) compute_line(slicer, line, initialized, top);
#endif
  }
  return owned_array<double>(std::move(out), {ks.size(), nx, ny});
}

}  // namespace mpnb

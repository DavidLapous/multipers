#pragma once

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace multipers::nanobind_helpers {

using squeezed_coordinate_remap = std::vector<std::unordered_map<int64_t, int64_t>>;

struct CompactedSqueezedFiltrationGrid {
  nanobind::object filtration_grid;
  std::vector<std::vector<int64_t>> coordinates;
  squeezed_coordinate_remap remap;

  CompactedSqueezedFiltrationGrid() : filtration_grid(nanobind::none()) {}
};

inline bool has_nonempty_filtration_grid(const nanobind::handle& grid) {
  if (!grid.is_valid() || grid.is_none() || !nanobind::hasattr(grid, "__len__") || nanobind::len(grid) == 0) {
    return false;
  }

  for (nanobind::handle row : nanobind::iter(grid)) {
    return nanobind::hasattr(row, "__len__") && nanobind::len(row) > 0;
  }
  return false;
}

inline int64_t squeezed_raw_index_from_value(double value, size_t parameter) {
  if (!std::isfinite(value)) {
    throw std::runtime_error("Expected finite squeezed filtration coordinates for parameter " +
                             std::to_string(parameter) + ".");
  }
  const double rounded = std::round(value);
  if (std::fabs(value - rounded) > 1e-9) {
    throw std::runtime_error("Expected integer squeezed filtration coordinates for parameter " +
                             std::to_string(parameter) + ".");
  }
  return static_cast<int64_t>(rounded);
}

inline int64_t normalized_squeezed_index(int64_t raw_index, int64_t row_size, size_t parameter) {
  int64_t normalized = raw_index;
  if (normalized < 0) {
    normalized += row_size;
  }
  if (normalized < 0 || normalized >= row_size) {
    throw std::runtime_error("Squeezed filtration coordinate is outside the filtration grid for parameter " +
                             std::to_string(parameter) + ".");
  }
  return normalized;
}

inline CompactedSqueezedFiltrationGrid compact_squeezed_filtration_grid(
    const nanobind::object& filtration_grid,
    std::vector<std::vector<int64_t>> used_coordinates) {
  const size_t num_parameters = used_coordinates.size();
  auto compact_grid = nanobind::steal<nanobind::tuple>(PyTuple_New(static_cast<Py_ssize_t>(num_parameters)));
  if (!compact_grid.is_valid()) {
    throw nanobind::python_error();
  }

  CompactedSqueezedFiltrationGrid out;
  out.coordinates = std::move(used_coordinates);
  out.remap.resize(num_parameters);

  for (size_t parameter = 0; parameter < num_parameters; ++parameter) {
    auto& current_coordinates = out.coordinates[parameter];
    std::sort(current_coordinates.begin(), current_coordinates.end());
    current_coordinates.erase(std::unique(current_coordinates.begin(), current_coordinates.end()), current_coordinates.end());

    nanobind::object row = filtration_grid[parameter];
    const int64_t row_size = static_cast<int64_t>(nanobind::len(row));
    nanobind::list selection;
    auto& remap = out.remap[parameter];
    for (size_t i = 0; i < current_coordinates.size(); ++i) {
      const int64_t raw_index = current_coordinates[i];
      (void)normalized_squeezed_index(raw_index, row_size, parameter);
      remap.emplace(raw_index, static_cast<int64_t>(i));
      selection.append(nanobind::int_(raw_index));
    }

    nanobind::object compact_row = row.attr("__getitem__")(selection);
    PyTuple_SET_ITEM(compact_grid.ptr(), static_cast<Py_ssize_t>(parameter), compact_row.release().ptr());
  }

  out.filtration_grid = compact_grid;
  return out;
}

inline double remap_squeezed_coordinate(double value, size_t parameter, const squeezed_coordinate_remap& remap) {
  return static_cast<double>(remap.at(parameter).at(squeezed_raw_index_from_value(value, parameter)));
}

template <typename Value>
inline std::vector<std::vector<Value>> cast_squeezed_coordinate_grid(
    const std::vector<std::vector<int64_t>>& coordinates) {
  std::vector<std::vector<Value>> out(coordinates.size());
  for (size_t parameter = 0; parameter < coordinates.size(); ++parameter) {
    const auto& row = coordinates[parameter];
    auto& out_row = out[parameter];
    out_row.reserve(row.size());
    for (int64_t value : row) {
      out_row.push_back(static_cast<Value>(value));
    }
  }
  return out;
}

struct PySlicerPythonState {
  nanobind::object filtration_grid;
  nanobind::object generator_basis;
  int minpres_degree;

  PySlicerPythonState() : filtration_grid(nanobind::none()), generator_basis(nanobind::none()), minpres_degree(-1) {}
};

template <typename TargetState, typename SourceState>
inline void copy_slicer_python_state(TargetState& target, const SourceState& source) {
  target.filtration_grid = source.filtration_grid;
  target.generator_basis = source.generator_basis;
  target.minpres_degree = source.minpres_degree;
}

template <typename State>
inline void reset_slicer_python_state(State& state) {
  state.filtration_grid = nanobind::none();
  state.generator_basis = nanobind::none();
  state.minpres_degree = -1;
}

template <typename Slicer>
struct PySlicer : PySlicerPythonState {
  Slicer truc;
};

struct PySimplexTreePythonState {
  nanobind::object filtration_grid;

  PySimplexTreePythonState() : filtration_grid(nanobind::none()) {}
};

template <typename TargetState, typename SourceState>
inline void copy_simplextree_python_state(TargetState& target, const SourceState& source) {
  target.filtration_grid = source.filtration_grid;
}

template <typename State>
inline void reset_simplextree_python_state(State& state) {
  state.filtration_grid = nanobind::none();
}

template <typename Interface, typename T>
struct PySimplexTree : PySimplexTreePythonState {
  Interface tree;
};

}  // namespace multipers::nanobind_helpers

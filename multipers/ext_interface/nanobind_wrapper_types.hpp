#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <vector>

namespace multipers::nanobind_helpers {

inline bool has_nonempty_filtration_grid(const nanobind::handle& grid) {
  if (!grid.is_valid() || grid.is_none() || !nanobind::hasattr(grid, "__len__") || nanobind::len(grid) == 0) {
    return false;
  }

  for (nanobind::handle row : nanobind::iter(grid)) {
    return nanobind::hasattr(row, "__len__") && nanobind::len(row) > 0;
  }
  return false;
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

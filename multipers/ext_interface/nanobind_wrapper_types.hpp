#pragma once

#include <nanobind/nanobind.h>

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

}  // namespace multipers::nanobind_helpers

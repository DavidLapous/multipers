/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       David Loiseaux, Hannah Schreiber
 *
 *    Copyright (C) 2026 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

/**
 * @file interface_helper_structs.h
 * @author David Loiseaux, Hannah Schreiber
 * @brief Contains helper structs for python bindings.
 */

#ifndef MP_PY_INTER_HELPER_STRUCTS_H_INCLUDED
#define MP_PY_INTER_HELPER_STRUCTS_H_INCLUDED

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <python_interfaces/construction_utils.h>

#include "slicer_interface_helpers.h"

namespace Gudhi {
namespace multi_persistence {
namespace detail {

struct Generator_basis_data {
  using Index = std::uint32_t;
  using Grade = double;

  int degree = -1;
  std::vector<std::vector<Index>> columns;
  std::vector<std::vector<Index>> rowBoundaries;
  std::vector<std::pair<Grade, Grade>> rowGrades;
  std::vector<std::pair<Grade, Grade>> columnGrades;

  Generator_basis_data() = default;

  Generator_basis_data(int degree_,
                       std::vector<std::vector<Index>> columns_,
                       std::vector<std::vector<Index>> rowBoundaries_,
                       std::vector<std::pair<Grade, Grade>> rowGrades_ = {},
                       std::vector<std::pair<Grade, Grade>> columnGrades_ = {})
      : degree(degree_),
        columns(std::move(columns_)),
        rowBoundaries(std::move(rowBoundaries_)),
        rowGrades(std::move(rowGrades_)),
        columnGrades(std::move(columnGrades_)) {}

  Generator_basis_data(nanobind::dict basis) {
    if (basis.is_none()) return;

    if (!basis.contains("degree") || !basis.contains("columns") || !basis.contains("row_boundaries")) {
      throw std::invalid_argument(
          "Invalid generator basis dictionary: expected keys `degree`, `columns`, and `row_boundaries`.");
    }

    bool success = nanobind::try_cast<int>(basis["degree"], degree);
    if (!success) {
      throw std::invalid_argument("_generator_basis['degree'] has to be of type int.");
    }
    success = nanobind::try_cast<std::vector<std::vector<Index>>>(basis["columns"], columns);
    if (!success) {
      throw std::invalid_argument("_generator_basis['columns'] has to be of an iterable of iterable of uint32.");
    }
    success = nanobind::try_cast<std::vector<std::vector<Index>>>(basis["row_boundaries"], rowBoundaries);
    if (!success) {
      throw std::invalid_argument("_generator_basis['row_boundaries'] has to be of an iterable of iterable of uint32.");
    }
    if (basis.contains("row_grades")) {
      success = nanobind::try_cast<std::vector<std::pair<Grade, Grade>>>(basis["row_grades"], rowGrades);
      if (!success) {
        throw std::invalid_argument("_generator_basis['row_grades'] has to be an iterable of pairs of float.");
      }
    }
    if (basis.contains("column_grades")) {
      success = nanobind::try_cast<std::vector<std::pair<Grade, Grade>>>(basis["column_grades"], columnGrades);
      if (!success) {
        throw std::invalid_argument("_generator_basis['column_grades'] has to be an iterable of pairs of float.");
      }
    }
  }

  std::vector<std::vector<Index>> expand_cycle(const std::vector<Index>& cycle) {
    std::vector<std::uint8_t> activeRows(rowBoundaries.size(), 0);
    for (Index genIdx : cycle) {
      if (genIdx >= columns.size()) {
        throw std::runtime_error("Representative cycle refers to a generator outside `_generator_basis`.");
      }
      for (Index rowIdx : columns[genIdx]) {
        if (rowIdx >= rowBoundaries.size()) {
          throw std::runtime_error("`_generator_basis` column support refers to a row outside `row_boundaries`.");
        }
        activeRows[rowIdx] ^= 1;
      }
    }

    std::vector<std::vector<Index>> out;
    for (std::size_t rowIdx = 0; rowIdx < activeRows.size(); ++rowIdx) {
      if (activeRows[rowIdx] != 0) {
        out.push_back(rowBoundaries[rowIdx]);
      }
    }
    return out;
  }
};

struct Compacted_squeezed_filtration_grid {
  using Index = std::int64_t;
  using squeezed_coordinate_remap = std::vector<std::unordered_map<Index, Index>>;

  nanobind::tuple filtrationGrid;
  std::vector<std::vector<Index>> coordinates;
  squeezed_coordinate_remap remap;

  Compacted_squeezed_filtration_grid() : filtrationGrid(nanobind::none()) {}

  Compacted_squeezed_filtration_grid(const nanobind::object& grid,
                                     const std::vector<std::vector<Index>>& usedCoordinates)
      : filtrationGrid(nanobind::none()), coordinates(usedCoordinates), remap(usedCoordinates.size()) {
    // special case of ndarray should be more efficient then general nanobind::iterable
    if (nanobind::ndarray<> arr; nanobind::try_cast<nanobind::ndarray<>>(grid, arr, false)) {
      if (arr.ndim() != 2) throw nanobind::type_error("Expected a 2D grid.");
      _dispatch_dtype(
          grid, [&]<typename U>() { return _get_compact_grid<U>(nanobind::ndarray<const U, nanobind::ndim<2>>(arr)); });
      return;
    }

    if (!nanobind::isinstance<nanobind::iterable>(grid))
      throw nanobind::type_error("Expected a grid as a 2D array or an iterable of iterables.");

    _get_compact_grid(nanobind::cast<nanobind::iterable>(grid));
  }

  template <typename T>
  T remap_squeezed_coordinate(T coordinate, std::size_t parameter) {
    if (parameter >= remap.size()) throw std::out_of_range("Parameter is out of range");
    // careful: just assumes that it will fit into T
    return static_cast<T>(remap[parameter].at(python::_cast_to_int<Index>(
        coordinate,
        "Expected integer squeezed filtration coordinates for parameter " + std::to_string(parameter) + ".")));
  }

  template <class Slicer>
  static std::vector<std::vector<Index>> collect_used_squeezed_coordinates(const Slicer& slicer) {
    const auto numParam = slicer.get_number_of_parameters();
    std::vector<std::vector<Index>> usedCoordinates(numParam);
    {
      nanobind::gil_scoped_release release;
      for (const auto& f : slicer.get_filtration_values()) {
        for (size_t g = 0; g < f.num_generators(); ++g) {
          for (size_t p = 0; p < numParam; ++p) {
            usedCoordinates[p].push_back(python::_cast_to_int<Index>(
                f(g, p), "Expected integer squeezed filtration coordinates for parameter " + std::to_string(p) + "."));
          }
        }
      }
    }
    return usedCoordinates;
  }

 private:
  static Index normalize_squeezed_index_or_sentinel(Index rawIndex, Index rowSize, std::size_t parameter) {
    if (rawIndex < 0) {
      rawIndex += rowSize;
    }
    if (rawIndex == rowSize) {
      return rawIndex;  // sentinel
    }
    if (rawIndex < 0 || rawIndex > rowSize) {
      throw std::runtime_error("Squeezed filtration coordinate is outside the filtration grid for parameter " +
                               std::to_string(parameter) + ".");
    }
    return rawIndex;
  }

  template <typename U>
  void _get_compact_grid(nanobind::ndarray<const U, nanobind::ndim<2>> grid) {
    auto view = grid.view();
    const Index rowSize = view.shape(1);

    if (view.shape(0) != coordinates.size())
      throw std::invalid_argument("Grid size and number of coordinates do not match.");

    filtrationGrid = Gudhi::python::_build_tuple(coordinates.size(), [&](std::size_t p) {
      auto& currentCoordinates = coordinates[p];
      std::ranges::sort(currentCoordinates);
      const auto uniq = std::ranges::unique(currentCoordinates);
      currentCoordinates.erase(uniq.begin(), uniq.end());

      nanobind::list selection;
      auto& m = remap[p];
      for (size_t i = 0; i < currentCoordinates.size(); ++i) {
        const Index rawIdx = currentCoordinates[i];
        const Index normalized = normalize_squeezed_index_or_sentinel(rawIdx, rowSize, p);
        m.emplace(rawIdx, i);
        if (normalized != rowSize) {
          selection.append(view(p, normalized));
        }
      }
      return selection;
    });
  }

  void _get_compact_grid(nanobind::iterable grid) {
    auto itGrid = grid.begin();
    filtrationGrid = Gudhi::python::_build_tuple(coordinates.size(), [&](std::size_t p) {
      auto& currentCoordinates = coordinates[p];
      std::ranges::sort(currentCoordinates);
      const auto uniq = std::ranges::unique(currentCoordinates);
      currentCoordinates.erase(uniq.begin(), uniq.end());

      nanobind::list selection;
      auto& m = remap[p];
      nanobind::list row(*itGrid);
      const auto rowSize = static_cast<Index>(row.size());
      for (std::size_t i = 0; i < currentCoordinates.size(); ++i) {
        const Index rawIdx = currentCoordinates[i];
        const Index normalized = normalize_squeezed_index_or_sentinel(rawIdx, rowSize, p);
        m.emplace(rawIdx, i);
        if (normalized != rowSize) {
          selection.append(row[normalized]);
        }
      }
      ++itGrid;

      if (itGrid == grid.end() && p + 1 != coordinates.size())
        throw std::invalid_argument("Grid size and number of coordinates do not match.");

      return selection;
    });
  }
};

template <class Boundaries, class Dimensions, typename Index>
class Representative_cycle_intersection {
 public:
  Representative_cycle_intersection(const Boundaries& boundaries,
                                    const Dimensions& dimensions,
                                    const std::unordered_set<Index>& points)
      : boundaries_(&boundaries), dimensions_(&dimensions), points_(&points), cache_(boundaries.size(), -1) {}

  bool intersects(const std::vector<Index>& cycle) {
    if (points_->empty()) {
      return false;
    }
    for (Index cell : cycle) {
      if (_cell_intersects(cell)) {
        return true;
      }
    }
    return false;
  }

  // for Generator_basis_data case, temporary
  bool dim_1_boundaries_intersects(const std::vector<std::vector<Index>>& cycle) {
    if (points_->empty()) {
      return false;
    }
    for (const auto& boundary : cycle) {
      for (Index vertex : boundary) {
        if (points_->contains(vertex)) {
          return true;
        }
      }
    }
    return false;
  }

  template <class F>
  void initialize_cache(std::size_t n, F&& get_cycle) {
    for (std::size_t i = 0; i < n; ++i) {
      intersects(std::forward<F>(get_cycle)(i));
    }
  }

 private:
  Boundaries const* boundaries_;
  Dimensions const* dimensions_;
  std::unordered_set<Index> const* points_;
  std::vector<std::int8_t> cache_;

  bool _cell_intersects(Index cell) {
    auto& cached = cache_[static_cast<std::size_t>(cell)];
    if (cached >= 0) {
      return cached != 0;
    }

    bool intersects = false;
    if (dimensions_[cell] == 0) {
      intersects = points_->find(cell) != points_->end();
    } else {
      for (auto face : boundaries_[cell]) {
        if (_cell_intersects(face)) {
          intersects = true;
          break;
        }
      }
    }
    cached = static_cast<std::int8_t>(intersects);
    return intersects;
  }
};

}  // namespace detail
}  // namespace multi_persistence
}  // namespace Gudhi

#endif  // MP_PY_INTER_HELPER_STRUCTS_H_INCLUDED

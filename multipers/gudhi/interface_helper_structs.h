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

#include <gudhi/Slicer.h>
#include <gudhi/Multi_persistence/utils.h>
#include <python_interfaces/construction_utils.h>

#include "ext_interface/nanobind_wrapper_types.hpp"
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
  std::vector<std::pair<Grade, Grade>> rowGrades; // change pair to array<Grade, 2>?
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

  template <class Complex, class GeneratorMatrix>
  Generator_basis_data(const Complex& complex, int degree_, GeneratorMatrix& generatorMatrix)
      : degree(degree_),
        columns(generatorMatrix.columns.size()),
        rowBoundaries(generatorMatrix.row_indices.size()),
        rowGrades(generatorMatrix.row_grades),
        columnGrades(generatorMatrix.column_grades) {
    nanobind::gil_scoped_release release;
    const auto& dimensions = complex.get_dimensions();
    const auto& boundaries = complex.get_boundaries();
    const auto& filtrations = complex.get_filtration_values();
    std::vector<std::size_t> degreeIndices;
    degreeIndices.reserve(dimensions.size());
    for (std::size_t i = 0; i < dimensions.size(); ++i) {
      if (dimensions[i] == degree) {
        degreeIndices.push_back(i);
      }
    }
    std::stable_sort(degreeIndices.begin(), degreeIndices.end(), [&](std::size_t a, std::size_t b) {
      const auto& fa = filtrations[a];
      const auto& fb = filtrations[b];
      return fa(0, 1) < fb(0, 1) || (fa(0, 1) == fb(0, 1) && fa(0, 0) < fb(0, 0));
    });

    for (std::size_t i = 0; i < generatorMatrix.row_indices.size(); ++i) {
      const auto row_idx = static_cast<std::size_t>(generatorMatrix.row_indices[i]);
      if (row_idx >= degreeIndices.size()) {
        throw std::runtime_error("generator-basis extraction failed: row index out of range.");
      }
      const auto& filtration = filtrations[degreeIndices[row_idx]];
      const auto& grade = generatorMatrix.row_grades[i];
      if (filtration(0, 0) != grade.first || filtration(0, 1) != grade.second) {
        throw std::runtime_error(
            "generator-basis extraction failed: row grades do not match the original degree block.");
      }
    }

    for (std::size_t i = 0; i < generatorMatrix.columns.size(); ++i) {
      columns[i].reserve(generatorMatrix.columns[i].size());
      for (const auto row_idx : generatorMatrix.columns[i]) {
        columns[i].push_back(Gudhi::python::_cast_to_int<Index>(
            row_idx, "generator-basis extraction failed: column support index does not fit into uint32."));
      }
    }

    for (std::size_t i = 0; i < generatorMatrix.row_indices.size(); ++i) {
      const auto rowIdx = static_cast<std::size_t>(generatorMatrix.row_indices[i]);
      const auto idx = degreeIndices[rowIdx];
      rowBoundaries[i].reserve(boundaries[idx].size());
      for (auto value : boundaries[idx]) {
        rowBoundaries[i].push_back(Gudhi::python::_cast_to_int<Index>(
            value, "generator-basis extraction failed: row boundary index does not fit into uint32."));
      }
    }
  }

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

  nanobind::object operator[](std::string_view key) const {
    if (key == "degree") return nanobind::cast(degree);
    if (key == "columns") return nanobind::cast(columns);
    if (key == "row_boundaries") return nanobind::cast(rowBoundaries);
    if (key == "row_grades") return nanobind::cast(rowGrades);
    if (key == "column_grades") return nanobind::cast(columnGrades);

    throw nanobind::key_error("Invalid `_GeneratorBasis` key.");
  }

  static bool is_key(std::string_view key) {
    return key == "degree" || key == "columns" || key == "row_boundaries" || key == "row_grades" ||
           key == "column_grades";
  }

  static nanobind::tuple get_keys() {
    return nanobind::make_tuple("degree", "columns", "row_boundaries", "row_grades", "column_grades");
  }

  std::vector<std::vector<Index>> expand_cycle(const std::vector<Index>& cycle) const {
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

  [[nodiscard]] std::string to_str() const {
    return "_GeneratorBasis(degree=" + std::to_string(degree) + ", columns=" + std::to_string(columns.size()) +
           ", row_boundaries=" + std::to_string(rowBoundaries.size()) + ")";
  }

  /**
   * @brief Serialize given value into the buffer at given pointer.
   *
   * @param value Value to serialize.
   * @param start Pointer to the start of the space in the buffer where to store the serialization.
   * @return End position of the serialization in the buffer.
   */
  friend char* serialize_value_to_char_buffer(const Generator_basis_data& value, char* start) {
    char* curr = start;
    curr = serialize_value_to_char_buffer(value.degree, curr);
    curr = serialize_value_to_char_buffer(value.columns, curr);
    curr = serialize_value_to_char_buffer(value.rowBoundaries, curr);
    curr = serialize_value_to_char_buffer(value.rowGrades, curr);
    curr = serialize_value_to_char_buffer(value.columnGrades, curr);
    return curr;
  }

  /**
   * @brief Deserialize the value from a buffer at given pointer and stores it in given value.
   *
   * @param value Value to fill with the deserialized summand.
   * @param start Pointer to the start of the space in the buffer where the serialization is stored.
   * @return End position of the serialization in the buffer.
   */
  friend const char* deserialize_value_from_char_buffer(Generator_basis_data& value, const char* start) {
    const char* curr = start;
    curr = deserialize_value_from_char_buffer(value.degree, curr);
    curr = deserialize_value_from_char_buffer(value.columns, curr);
    curr = deserialize_value_from_char_buffer(value.rowBoundaries, curr);
    curr = deserialize_value_from_char_buffer(value.rowGrades, curr);
    curr = deserialize_value_from_char_buffer(value.columnGrades, curr);
    return curr;
  }

  /**
   * @brief Returns the serialization size of the given summand.
   */
  friend std::size_t get_serialization_size_of(const Generator_basis_data& value) {
    std::size_t size = get_serialization_size_of(value.degree);
    size += get_serialization_size_of(value.columns);
    size += get_serialization_size_of(value.rowBoundaries);
    size += get_serialization_size_of(value.rowGrades);
    size += get_serialization_size_of(value.columnGrades);
    return size;
  }
};

inline Generator_basis_data deserialize_gen_basis_from_python(
    nanobind::ndarray<const char, nanobind::ndim<1>, nanobind::numpy> state) {
  Generator_basis_data basis;
  {
    nanobind::gil_scoped_release release;
    deserialize_value_from_char_buffer(basis, state.data());
  }
  return basis;
}

struct Compacted_squeezed_filtration_grid {
  using Index = std::int64_t;
  using squeezed_coordinate_remap = std::vector<std::unordered_map<Index, Index>>;

  nanobind::tuple filtrationGrid;
  std::vector<std::vector<Index>> coordinates;
  squeezed_coordinate_remap remap;

  Compacted_squeezed_filtration_grid() = default;

  Compacted_squeezed_filtration_grid(const nanobind::object& grid,
                                     const std::vector<std::vector<Index>>& usedCoordinates)
      : coordinates(usedCoordinates), remap(usedCoordinates.size()) {
    // special case of ndarray should be more efficient then general nanobind::iterable
    if (nanobind::ndarray<> arr; nanobind::try_cast<nanobind::ndarray<>>(grid, arr, false)) {
      if (arr.ndim() != 2) throw nanobind::type_error("Expected a 2D grid.");
      _dispatch_float_dtype(
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

  template <class MultiFiltrationValue, class PersistenceAlgorithm>
  static std::vector<std::vector<Index>> collect_used_squeezed_coordinates(
      const Gudhi::multi_persistence::Slicer<MultiFiltrationValue, PersistenceAlgorithm>& slicer) {
    const auto numParam = slicer.get_number_of_parameters();
    std::vector<std::vector<Index>> usedCoordinates(numParam);
    {
      nanobind::gil_scoped_release release;
      for (const auto& f : slicer.get_filtration_values()) {
        for (std::size_t g = 0; g < f.num_generators(); ++g) {
          for (std::size_t p = 0; p < numParam; ++p) {
            usedCoordinates[p].push_back(python::_cast_to_int<Index>(
                f(g, p),
                "Expected slicer integer squeezed filtration coordinates for parameter " + std::to_string(p) + "."));
          }
        }
      }
    }
    return usedCoordinates;
  }

  template <typename Interface, typename T>
  static std::vector<std::vector<Index>> collect_used_squeezed_coordinates(
      multipers::nanobind_helpers::PySimplexTree<Interface, T>& simplexTree) {
    const auto numParam = simplexTree.tree.num_parameters();
    std::vector<std::vector<Index>> usedCoordinates(numParam);
    {
      nanobind::gil_scoped_release release;
      for (auto simplex_handle : simplexTree.tree.complex_simplex_range()) {
        auto pair = simplexTree.tree.get_simplex_and_filtration(simplex_handle);
        const auto& f = *pair.second;
        for (std::size_t g = 0; g < f.num_generators(); ++g) {
          for (std::size_t p = 0; p < numParam; ++p) {
            usedCoordinates[p].push_back(python::_cast_to_int<Index>(
                f(g, p),
                "Expected simplex tree integer squeezed filtration coordinates for parameter " + std::to_string(p) +
                    "."));
          }
        }
      }
    }
    return usedCoordinates;
  }

  template <typename T>
  static std::vector<std::vector<Index>> collect_used_squeezed_coordinates(
      const std::vector<std::pair<T, T>>& biFiltrationValues) {
    std::vector<std::vector<Index>> usedCoordinates(2);
    {
      nanobind::gil_scoped_release release;
      usedCoordinates[0].reserve(biFiltrationValues.size());
      usedCoordinates[1].reserve(biFiltrationValues.size());
      for (const auto& degree : biFiltrationValues) {
        usedCoordinates[0].push_back(python::_cast_to_int<Index>(
            degree.first, "Expected integer squeezed filtration coordinates for parameter 0."));
        usedCoordinates[1].push_back(python::_cast_to_int<Index>(
            degree.second, "Expected integer squeezed filtration coordinates for parameter 1."));
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

    if (view.shape(0) < coordinates.size())
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
          selection.append(normalized);
        }
      }

      nanobind::object gridTmp = nanobind::cast(grid);
      auto rowTmp = gridTmp.attr("__getitem__")(p);
      return rowTmp.attr("__getitem__")(selection);
    });
  }

  void _get_compact_grid(nanobind::iterable grid) {
    if (!nanobind::hasattr(grid, "__getitem__")) throw nanobind::type_error("Grid has to support subscripting.");
    Index gridSize = 0;
    if (nanobind::hasattr(grid, "__len__")) {
      gridSize = static_cast<Index>(nanobind::len(grid));
    } else {
      for (auto it = grid.begin(); it != grid.end(); ++it) ++gridSize;
    }
    if (gridSize < coordinates.size()) throw std::invalid_argument("Grid size and number of coordinates do not match.");

    filtrationGrid = Gudhi::python::_build_tuple(coordinates.size(), [&](std::size_t p) {
      auto& currentCoordinates = coordinates[p];
      std::ranges::sort(currentCoordinates);
      const auto uniq = std::ranges::unique(currentCoordinates);
      currentCoordinates.erase(uniq.begin(), uniq.end());

      nanobind::list selection;
      auto& m = remap[p];
      if (!nanobind::hasattr(grid[p], "__getitem__"))
        throw nanobind::type_error("Grid rows have to support subscripting.");
      nanobind::object row = nanobind::cast(grid[p]);
      Index rowSize;
      if (nanobind::hasattr(row, "__len__"))
        rowSize = static_cast<Index>(nanobind::len(row));
      else
        rowSize = static_cast<Index>(nanobind::list(row).size());
      for (std::size_t i = 0; i < currentCoordinates.size(); ++i) {
        const Index rawIdx = currentCoordinates[i];
        const Index normalized = normalize_squeezed_index_or_sentinel(rawIdx, rowSize, p);
        m.emplace(rawIdx, i);
        if (normalized != rowSize) {
          selection.append(normalized);
        }
      }

      return row.attr("__getitem__")(selection);
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
    if ((*dimensions_)[cell] == 0) {
      intersects = points_->find(cell) != points_->end();
    } else {
      for (auto face : (*boundaries_)[cell]) {
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

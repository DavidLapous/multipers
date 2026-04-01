#pragma once

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace multipers::nanobind_helpers {

template <typename Wrapper, typename GeneratorMatrix>
nanobind::dict generator_basis_from_degree_rows(const Wrapper& input_wrapper,
                                                int degree,
                                                GeneratorMatrix& generator_matrix,
                                                const char* backend_name) {
  const std::string error_prefix = std::string(backend_name) + " generator-basis extraction failed: ";
  const auto dimensions = input_wrapper.truc.get_dimensions();
  const auto& boundaries = input_wrapper.truc.get_boundaries();
  const auto& filtrations = input_wrapper.truc.get_filtration_values();
  std::vector<size_t> degree_indices;
  degree_indices.reserve(dimensions.size());
  for (size_t i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == degree) {
      degree_indices.push_back(i);
    }
  }
  std::stable_sort(degree_indices.begin(), degree_indices.end(), [&](size_t a, size_t b) {
    const auto& fa = filtrations[a];
    const auto& fb = filtrations[b];
    return fa(0, 1) < fb(0, 1) || (fa(0, 1) == fb(0, 1) && fa(0, 0) < fb(0, 0));
  });

  if (generator_matrix.row_indices.size() != generator_matrix.row_grades.size()) {
    throw std::runtime_error(error_prefix + "row count mismatch.");
  }
  for (size_t i = 0; i < generator_matrix.row_indices.size(); ++i) {
    const auto row_idx = static_cast<size_t>(generator_matrix.row_indices[i]);
    if (row_idx >= degree_indices.size()) {
      throw std::runtime_error(error_prefix + "row index out of range.");
    }
    const auto& filtration = filtrations[degree_indices[row_idx]];
    const auto& grade = generator_matrix.row_grades[i];
    if (filtration(0, 0) != grade.first || filtration(0, 1) != grade.second) {
      throw std::runtime_error(error_prefix + "row grades do not match the original degree block.");
    }
  }

  nanobind::list row_boundaries;
  for (auto raw_row_idx : generator_matrix.row_indices) {
    const auto row_idx = static_cast<size_t>(raw_row_idx);
    const auto idx = degree_indices[row_idx];
    std::vector<uint32_t> boundary;
    boundary.reserve(boundaries[idx].size());
    for (auto value : boundaries[idx]) {
      boundary.push_back(static_cast<uint32_t>(value));
    }
    row_boundaries.append(nanobind::cast(std::move(boundary)));
  }

  nanobind::dict basis;
  basis["degree"] = degree;
  basis["row_boundaries"] = std::move(row_boundaries);
  basis["columns"] = nanobind::cast(std::move(generator_matrix.columns));
  basis["row_grades"] = nanobind::cast(std::move(generator_matrix.row_grades));
  basis["column_grades"] = nanobind::cast(std::move(generator_matrix.column_grades));
  return basis;
}

}  // namespace multipers::nanobind_helpers

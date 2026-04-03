#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "contiguous_slicer_bridge.hpp"

namespace multipers::nanobind_helpers {

struct GeneratorBasisData {
  bool active = false;
  int degree = -1;
  std::vector<std::vector<uint32_t>> columns;
  std::vector<std::vector<uint32_t>> row_boundaries;
  std::vector<std::pair<double, double>> row_grades;
  std::vector<std::pair<double, double>> column_grades;

  GeneratorBasisData() = default;

  GeneratorBasisData(int degree_,
                     std::vector<std::vector<uint32_t>> columns_,
                     std::vector<std::vector<uint32_t>> row_boundaries_,
                     std::vector<std::pair<double, double>> row_grades_ = {},
                     std::vector<std::pair<double, double>> column_grades_ = {})
      : active(true),
        degree(degree_),
        columns(std::move(columns_)),
        row_boundaries(std::move(row_boundaries_)),
        row_grades(std::move(row_grades_)),
        column_grades(std::move(column_grades_)) {}
};

template <typename Index>
inline uint32_t checked_uint32_index(Index raw_idx, const std::string& error_prefix, const char* label) {
  if constexpr (std::is_signed_v<Index>) {
    if (raw_idx < 0) {
      throw std::runtime_error(error_prefix + label + " index is negative.");
    }
  }

  const auto wide = static_cast<uint64_t>(raw_idx);
  if (wide > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error(error_prefix + label + " index exceeds uint32 range.");
  }
  return static_cast<uint32_t>(wide);
}

template <typename Columns>
inline std::vector<std::vector<uint32_t>> convert_generator_columns(const Columns& columns,
                                                                    const std::string& error_prefix,
                                                                    const char* label) {
  std::vector<std::vector<uint32_t>> out;
  out.reserve(columns.size());
  for (const auto& column : columns) {
    std::vector<uint32_t> support;
    support.reserve(column.size());
    for (const auto row_idx : column) {
      support.push_back(checked_uint32_index(row_idx, error_prefix, label));
    }
    out.push_back(std::move(support));
  }
  return out;
}

inline GeneratorBasisData generator_basis_from_legacy_dict(const nanobind::dict& basis) {
  if (!basis.contains("degree") || !basis.contains("columns") || !basis.contains("row_boundaries")) {
    throw std::runtime_error("Invalid `_generator_basis`: expected keys `degree`, `columns`, and `row_boundaries`.");
  }

  GeneratorBasisData out;
  out.active = true;
  out.degree = nanobind::cast<int>(basis["degree"]);
  out.columns = nanobind::cast<std::vector<std::vector<uint32_t>>>(basis["columns"]);
  out.row_boundaries = nanobind::cast<std::vector<std::vector<uint32_t>>>(basis["row_boundaries"]);
  if (basis.contains("row_grades")) {
    out.row_grades = nanobind::cast<std::vector<std::pair<double, double>>>(basis["row_grades"]);
  }
  if (basis.contains("column_grades")) {
    out.column_grades = nanobind::cast<std::vector<std::pair<double, double>>>(basis["column_grades"]);
  }
  return out;
}

inline GeneratorBasisData generator_basis_from_object(const nanobind::handle& basis_handle) {
  if (!basis_handle.is_valid() || basis_handle.is_none()) {
    return GeneratorBasisData{};
  }

  if (nanobind::isinstance<nanobind::dict>(basis_handle)) {
    return generator_basis_from_legacy_dict(nanobind::borrow<nanobind::dict>(basis_handle));
  }

  GeneratorBasisData out = nanobind::cast<GeneratorBasisData>(basis_handle);
  out.active = true;
  return out;
}

inline nanobind::object generator_basis_to_python_object(GeneratorBasisData basis) {
  nanobind::module_::import_("multipers._slicer_nanobind");
  basis.active = true;
  return nanobind::cast(std::move(basis));
}

template <typename Wrapper, typename GeneratorMatrix>
GeneratorBasisData generator_basis_from_degree_rows(const Wrapper& input_wrapper,
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

  GeneratorBasisData basis;
  basis.active = true;
  basis.degree = degree;
  basis.row_grades = generator_matrix.row_grades;
  basis.column_grades = generator_matrix.column_grades;
  basis.columns = convert_generator_columns(generator_matrix.columns, error_prefix, "column support");
  basis.row_boundaries.reserve(generator_matrix.row_indices.size());
  for (auto raw_row_idx : generator_matrix.row_indices) {
    const auto row_idx = static_cast<size_t>(raw_row_idx);
    const auto idx = degree_indices[row_idx];
    std::vector<uint32_t> boundary;
    boundary.reserve(boundaries[idx].size());
    for (auto value : boundaries[idx]) {
      boundary.push_back(checked_uint32_index(value, error_prefix, "row boundary"));
    }
    basis.row_boundaries.push_back(std::move(boundary));
  }

  return basis;
}

template <typename Wrapper, typename GeneratorMatrix>
nanobind::object generator_basis_object_from_degree_rows(const Wrapper& input_wrapper,
                                                         int degree,
                                                         GeneratorMatrix& generator_matrix,
                                                         const char* backend_name) {
  return generator_basis_to_python_object(
      generator_basis_from_degree_rows(input_wrapper, degree, generator_matrix, backend_name));
}

template <typename Wrapper, typename ComplexFactory, typename ResultFactory>
nanobind::object build_minpres_slicer_output_for_target(nanobind::object target,
                                                        const Wrapper& input_wrapper,
                                                        int degree,
                                                        bool keep_generators,
                                                        const char* backend_name,
                                                        ComplexFactory&& compute_complex,
                                                        ResultFactory&& compute_with_generators) {
  nanobind::object out = target.type()();
  auto& out_wrapper = nanobind::cast<Wrapper&>(out);

  if (!keep_generators) {
    auto complex = std::forward<ComplexFactory>(compute_complex)();
    build_slicer_from_complex(out_wrapper.truc, complex);
    return out;
  }

  auto result = std::forward<ResultFactory>(compute_with_generators)();
  build_slicer_from_complex(out_wrapper.truc, result.first);
  out_wrapper.generator_basis = generator_basis_object_from_degree_rows(
      input_wrapper, degree, result.second, backend_name);
  return out;
}

}  // namespace multipers::nanobind_helpers

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gudhi/simple_mdspan.h>

#include "Persistence_slices_interface.h"

namespace multipers {

template <typename value_type>
using contiguous_filtration = multipers::tmp_interface::filtration_options<
    multipers::tmp_interface::Filtration_containers_strs::Multi_parameter_filtration,
    false,
    value_type>;

using contiguous_i32_filtration = contiguous_filtration<std::int32_t>;

using contiguous_f64_filtration = contiguous_filtration<double>;

using kcontiguous_f64_filtration = multipers::tmp_interface::
    filtration_options<multipers::tmp_interface::Filtration_containers_strs::Multi_parameter_filtration, true, double>;

template <typename value_type>
using contiguous_slicer = multipers::tmp_interface::TrucPythonInterface<
    multipers::tmp_interface::BackendsEnum::Matrix,
    false,
    false,
    value_type,
    multipers::tmp_interface::Available_columns::UNORDERED_SET,
    multipers::tmp_interface::Filtration_containers_strs::Multi_parameter_filtration>;

using contiguous_i32_slicer = contiguous_slicer<std::int32_t>;

using contiguous_f64_slicer = contiguous_slicer<double>;

using kcontiguous_f64_slicer = multipers::tmp_interface::TrucPythonInterface<
    multipers::tmp_interface::BackendsEnum::Matrix,
    false,
    true,
    double,
    multipers::tmp_interface::Available_columns::UNORDERED_SET,
    multipers::tmp_interface::Filtration_containers_strs::Multi_parameter_filtration>;

template <typename value_type>
using contiguous_complex =
    Gudhi::multi_persistence::Multi_parameter_filtered_complex<contiguous_filtration<value_type>>;

using contiguous_i32_complex = contiguous_complex<std::int32_t>;

using contiguous_f64_complex = contiguous_complex<double>;

template <typename value_type, typename index_type>
inline contiguous_complex<value_type> build_contiguous_slicer_from_output(
    const std::vector<value_type>& filtration_values,
    std::size_t num_parameters,
    const std::vector<std::vector<index_type>>& boundaries,
    const std::vector<int>& dimensions) {
  const std::size_t num_generators = dimensions.size();
  if (boundaries.size() != num_generators) {
    throw std::invalid_argument("Invalid interface output: sizes of filtrations, boundaries and dimensions differ.");
  }
  if (num_generators == 0) {
    if (!filtration_values.empty()) {
      throw std::invalid_argument("Invalid interface output: filtration data without generators.");
    }
  } else {
    if (num_parameters == 0) {
      throw std::invalid_argument("Invalid interface output: empty filtration grade.");
    }
    if (filtration_values.size() != num_generators * num_parameters) {
      throw std::invalid_argument("Invalid interface output: filtration data has wrong shape.");
    }
  }

  std::vector<std::vector<uint32_t>> c_boundaries;
  c_boundaries.resize(num_generators);
  for (std::size_t i = 0; i < num_generators; ++i) {
    c_boundaries[i].reserve(boundaries[i].size());
    for (const auto idx : boundaries[i]) {
      if (idx < 0) {
        throw std::invalid_argument("Invalid boundary index: negative index.");
      }
      if (static_cast<unsigned long long>(idx) > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Invalid boundary index: exceeds uint32 range.");
      }
      c_boundaries[i].push_back(static_cast<uint32_t>(idx));
    }
  }

  std::vector<contiguous_filtration<value_type>> c_filtrations;
  c_filtrations.reserve(num_generators);
  const auto grades = Gudhi::Simple_mdspan<const value_type, Gudhi::dextents<std::size_t, 2>>(
      filtration_values.data(), num_generators, num_parameters);
  for (std::size_t i = 0; i < num_generators; ++i) {
    const value_type* row_begin = grades.data_handle() + i * grades.stride(0);
    c_filtrations.emplace_back(row_begin, row_begin + num_parameters);
  }

  std::vector<int> c_dimensions = dimensions;
  return contiguous_complex<value_type>(c_boundaries, c_dimensions, c_filtrations);
}

template <typename index_type>
inline contiguous_f64_complex build_contiguous_f64_slicer_from_output(
    const std::vector<double>& filtration_values,
    std::size_t num_parameters,
    const std::vector<std::vector<index_type>>& boundaries,
    const std::vector<int>& dimensions) {
  return build_contiguous_slicer_from_output<double, index_type>(
      filtration_values, num_parameters, boundaries, dimensions);
}

template <typename index_type>
inline contiguous_i32_complex build_contiguous_i32_slicer_from_output(
    const std::vector<std::vector<std::int64_t>>& filtration_values,
    const std::vector<std::vector<index_type>>& boundaries,
    const std::vector<int>& dimensions) {
  const std::size_t num_parameters = filtration_values.empty() ? 0 : filtration_values.front().size();
  std::vector<std::int32_t> grade_values;
  grade_values.reserve(filtration_values.size() * num_parameters);
  for (const auto& grade : filtration_values) {
    if (grade.size() != num_parameters) {
      throw std::invalid_argument("Invalid interface output: inconsistent filtration grade sizes.");
    }
    for (const auto value : grade) {
      if (value < std::numeric_limits<std::int32_t>::min() || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("Invalid interface output: filtration coordinate exceeds int32 range.");
      }
      grade_values.push_back(static_cast<std::int32_t>(value));
    }
  }
  return build_contiguous_slicer_from_output<std::int32_t, index_type>(
      grade_values, num_parameters, boundaries, dimensions);
}

template <typename index_type>
inline contiguous_f64_complex build_contiguous_f64_slicer_from_output(
    const std::vector<std::vector<double>>& filtration_values,
    const std::vector<std::vector<index_type>>& boundaries,
    const std::vector<int>& dimensions) {
  const std::size_t num_parameters = filtration_values.empty() ? 0 : filtration_values.front().size();
  std::vector<double> grade_values;
  grade_values.reserve(filtration_values.size() * num_parameters);
  for (const auto& grade : filtration_values) {
    if (grade.size() != num_parameters) {
      throw std::invalid_argument("Invalid interface output: inconsistent filtration grade sizes.");
    }
    grade_values.insert(grade_values.end(), grade.begin(), grade.end());
  }
  return build_contiguous_f64_slicer_from_output<index_type>(grade_values, num_parameters, boundaries, dimensions);
}

template <typename index_type>
inline contiguous_f64_complex build_contiguous_f64_slicer_from_output(
    const std::vector<std::pair<double, double>>& filtration_values,
    const std::vector<std::vector<index_type>>& boundaries,
    const std::vector<int>& dimensions) {
  std::vector<double> grade_values;
  grade_values.reserve(2 * filtration_values.size());
  for (const auto& grade : filtration_values) {
    grade_values.push_back(grade.first);
    grade_values.push_back(grade.second);
  }
  return build_contiguous_f64_slicer_from_output<index_type>(grade_values, 2, boundaries, dimensions);
}

template <typename slicer_type, typename complex_type>
inline void build_slicer_from_complex(slicer_type& target, complex_type& complex) {
  target = slicer_type(std::move(complex));
}

}  // namespace multipers

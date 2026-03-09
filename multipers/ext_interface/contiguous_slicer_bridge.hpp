#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Persistence_slices_interface.h"

namespace multipers {

using contiguous_f64_filtration = multipers::tmp_interface::filtration_options<
    multipers::tmp_interface::Filtration_containers_strs::Multi_parameter_filtration,
    false,
    double>;

using kcontiguous_f64_filtration = multipers::tmp_interface::filtration_options<
    multipers::tmp_interface::Filtration_containers_strs::Multi_parameter_filtration,
    true,
    double>;

using contiguous_f64_slicer = multipers::tmp_interface::TrucPythonInterface<
    multipers::tmp_interface::BackendsEnum::Matrix,
    false,
    false,
    double,
    multipers::tmp_interface::Available_columns::UNORDERED_SET,
    multipers::tmp_interface::Filtration_containers_strs::Multi_parameter_filtration>;

using kcontiguous_f64_slicer = multipers::tmp_interface::TrucPythonInterface<
    multipers::tmp_interface::BackendsEnum::Matrix,
    false,
    true,
    double,
    multipers::tmp_interface::Available_columns::UNORDERED_SET,
    multipers::tmp_interface::Filtration_containers_strs::Multi_parameter_filtration>;

using contiguous_f64_complex = Gudhi::multi_persistence::Multi_parameter_filtered_complex<contiguous_f64_filtration>;

template <typename index_type>
inline contiguous_f64_complex build_contiguous_f64_slicer_from_output(
    const std::vector<std::pair<double, double> >& filtration_values,
    const std::vector<std::vector<index_type> >& boundaries,
    const std::vector<int>& dimensions) {
  const std::size_t num_generators = dimensions.size();
  if (filtration_values.size() != num_generators || boundaries.size() != num_generators) {
    throw std::invalid_argument("Invalid interface output: sizes of filtrations, boundaries and dimensions differ.");
  }

  std::vector<std::vector<uint32_t> > c_boundaries;
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

  std::vector<contiguous_f64_filtration> c_filtrations;
  c_filtrations.reserve(num_generators);
  for (const auto& grade : filtration_values) {
    std::vector<double> values = {grade.first, grade.second};
    c_filtrations.emplace_back(values);
  }

  std::vector<int> c_dimensions = dimensions;
  return contiguous_f64_complex(c_boundaries, c_dimensions, c_filtrations);
}

template <typename slicer_type>
inline void build_slicer_from_complex(slicer_type& target,
                                      contiguous_f64_complex& complex) {
  target = slicer_type(complex);
}

}  // namespace multipers

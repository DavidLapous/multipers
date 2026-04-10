#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ext_interface/nanobind_wrapper_types.hpp"
#include "gudhi/Multi_parameter_filtered_complex.h"
#include "nanobind_array_utils.hpp"

namespace mpnb {

namespace nb = nanobind;

inline constexpr uint32_t kSlicerSerializationMagic = 0x4d50534c;
inline constexpr uint32_t kSlicerSerializationVersion = 1;

enum class SlicerSerializationMode : uint32_t {
  OneCritical = 0,
  KCritical = 1,
  DegreeRips = 2,
};

struct SlicerSerializationHeaderV1 {
  uint32_t magic;
  uint32_t version;
  uint32_t mode;
  uint32_t reserved;
  uint64_t num_generators;
  uint64_t boundary_flat_size;
  uint64_t num_parameters;
  uint64_t filtration_rows;
  uint64_t boundary_indptr_offset;
  uint64_t boundary_flat_offset;
  uint64_t dimensions_offset;
  uint64_t grade_indptr_offset;
  uint64_t grades_offset;
  uint64_t total_size;
};

static_assert(std::is_trivially_copyable_v<SlicerSerializationHeaderV1>);

struct SlicerSerializationLayout {
  size_t boundary_indptr_offset;
  size_t boundary_flat_offset;
  size_t dimensions_offset;
  size_t grade_indptr_offset;
  size_t grades_offset;
  size_t total_size;
};

template <typename T>
size_t serialized_array_bytes(size_t count) {
  return count * sizeof(T);
}

inline size_t align_serialized_offset(size_t offset, size_t alignment) {
  size_t remainder = offset % alignment;
  if (remainder == 0) {
    return offset;
  }
  return offset + alignment - remainder;
}

template <bool IsKCritical, bool IsDegreeRips>
constexpr uint32_t expected_slicer_serialization_mode() {
  return static_cast<uint32_t>(IsDegreeRips ? SlicerSerializationMode::DegreeRips
                                            : (IsKCritical ? SlicerSerializationMode::KCritical
                                                           : SlicerSerializationMode::OneCritical));
}

template <typename Value>
SlicerSerializationLayout make_slicer_serialization_layout(size_t num_generators,
                                                           size_t boundary_flat_size,
                                                           size_t num_parameters,
                                                           size_t filtration_rows,
                                                           bool include_grade_indptr) {
  SlicerSerializationLayout layout{};
  size_t offset = sizeof(SlicerSerializationHeaderV1);
  offset = align_serialized_offset(offset, alignof(uint64_t));
  layout.boundary_indptr_offset = offset;
  offset += serialized_array_bytes<uint64_t>(num_generators + 1);
  offset = align_serialized_offset(offset, alignof(uint32_t));
  layout.boundary_flat_offset = offset;
  offset += serialized_array_bytes<uint32_t>(boundary_flat_size);
  offset = align_serialized_offset(offset, alignof(int32_t));
  layout.dimensions_offset = offset;
  offset += serialized_array_bytes<int32_t>(num_generators);
  if (include_grade_indptr) {
    offset = align_serialized_offset(offset, alignof(int64_t));
    layout.grade_indptr_offset = offset;
    offset += serialized_array_bytes<int64_t>(num_generators + 1);
  } else {
    layout.grade_indptr_offset = 0;
  }
  offset = align_serialized_offset(offset, alignof(Value));
  layout.grades_offset = offset;
  offset += serialized_array_bytes<Value>(filtration_rows * num_parameters);
  layout.total_size = offset;
  return layout;
}

template <typename T>
T* mutable_serialized_block(std::vector<uint8_t>& buffer, size_t offset, size_t count) {
  if (offset % alignof(T) != 0) {
    throw std::runtime_error("Invalid serialized slicer state.");
  }
  size_t num_bytes = serialized_array_bytes<T>(count);
  if (offset > buffer.size() || buffer.size() - offset < num_bytes) {
    throw std::runtime_error("Invalid serialized slicer state.");
  }
  return reinterpret_cast<T*>(buffer.data() + offset);
}

template <typename T>
const T* serialized_block_view(const uint8_t* data, size_t buffer_size, uint64_t offset_value, uint64_t count_value) {
  size_t offset = static_cast<size_t>(offset_value);
  size_t count = static_cast<size_t>(count_value);
  if (offset % alignof(T) != 0) {
    throw std::runtime_error("Invalid serialized slicer state.");
  }
  size_t num_bytes = serialized_array_bytes<T>(count);
  if (offset > buffer_size || buffer_size - offset < num_bytes) {
    throw std::runtime_error("Invalid serialized slicer state.");
  }
  return reinterpret_cast<const T*>(data + offset);
}

template <typename Wrapper, typename Concrete>
void load_slicer_from_generator_data(Wrapper& self,
                                     std::vector<std::vector<uint32_t>>&& boundaries,
                                     std::vector<int>&& dimensions,
                                     std::vector<typename Concrete::Filtration_value>&& filtrations) {
  if (boundaries.empty()) {
    self.truc = Concrete();
    return;
  }
  Gudhi::multi_persistence::Multi_parameter_filtered_complex<typename Concrete::Filtration_value> cpx(
      std::move(boundaries), std::move(dimensions), std::move(filtrations));
  self.truc = Concrete(std::move(cpx));
}

template <typename FiltrationValue, typename Value, bool IsDegreeRips>
FiltrationValue filtration_from_serialized_rows(const Value* grades_flat,
                                                size_t row_begin,
                                                size_t row_end,
                                                size_t num_parameters) {
  using Underlying_container = typename FiltrationValue::Underlying_container;
  if constexpr (IsDegreeRips) {
    Underlying_container generators(row_end - row_begin);
    for (size_t row = row_begin; row < row_end; ++row) {
      generators[row - row_begin] = grades_flat[2 * row];
    }
    return FiltrationValue(std::move(generators), static_cast<int>(num_parameters));
  } else if constexpr (std::is_same_v<typename Underlying_container::value_type, Value>) {
    Underlying_container generators((row_end - row_begin) * num_parameters);
    if (!generators.empty()) {
      size_t offset = row_begin * num_parameters;
      std::memcpy(generators.data(), grades_flat + offset, generators.size() * sizeof(Value));
    }
    return FiltrationValue(std::move(generators), static_cast<int>(num_parameters));
  } else {
    Underlying_container generators;
    generators.reserve(row_end - row_begin);
    for (size_t row = row_begin; row < row_end; ++row) {
      size_t offset = row * num_parameters;
      generators.emplace_back(grades_flat + offset, grades_flat + offset + num_parameters);
    }
    return FiltrationValue(std::move(generators), static_cast<int>(num_parameters));
  }
}

template <typename Wrapper, typename Concrete, typename Value, bool IsKCritical, bool IsDegreeRips>
void load_state_v1(Wrapper& self, const uint8_t* data, size_t buffer_size) {
  if (buffer_size < sizeof(SlicerSerializationHeaderV1)) {
    throw std::runtime_error("Invalid serialized slicer state.");
  }
  SlicerSerializationHeaderV1 header;
  std::memcpy(&header, data, sizeof(header));
  if (header.magic != kSlicerSerializationMagic || header.version != kSlicerSerializationVersion ||
      header.total_size != buffer_size) {
    throw std::runtime_error("Invalid serialized slicer state.");
  }
  constexpr uint32_t expected_mode = expected_slicer_serialization_mode<IsKCritical, IsDegreeRips>();
  if (header.mode != expected_mode) {
    throw std::runtime_error("Serialized slicer state does not match target type.");
  }

  size_t num_generators = static_cast<size_t>(header.num_generators);
  size_t boundary_flat_size = static_cast<size_t>(header.boundary_flat_size);
  size_t num_parameters = static_cast<size_t>(header.num_parameters);
  size_t filtration_rows = static_cast<size_t>(header.filtration_rows);
  if constexpr (IsDegreeRips) {
    if (num_parameters != 2) {
      throw std::runtime_error("Invalid serialized slicer filtrations.");
    }
  }
  if constexpr (!IsKCritical) {
    if (filtration_rows != num_generators) {
      throw std::runtime_error("Invalid serialized slicer filtrations.");
    }
  }

  const uint64_t* boundary_indptr =
      serialized_block_view<uint64_t>(data, buffer_size, header.boundary_indptr_offset, header.num_generators + 1);
  const uint32_t* boundary_flat =
      serialized_block_view<uint32_t>(data, buffer_size, header.boundary_flat_offset, header.boundary_flat_size);
  const int32_t* dimensions32 =
      serialized_block_view<int32_t>(data, buffer_size, header.dimensions_offset, header.num_generators);
  if (boundary_indptr[num_generators] != header.boundary_flat_size) {
    throw std::runtime_error("Invalid serialized slicer boundaries.");
  }

  std::vector<std::vector<uint32_t>> boundaries(num_generators);
  for (size_t i = 0; i < num_generators; ++i) {
    uint64_t begin = boundary_indptr[i];
    uint64_t finish = boundary_indptr[i + 1];
    if (begin > finish || finish > boundary_flat_size) {
      throw std::runtime_error("Invalid serialized slicer boundaries.");
    }
    boundaries[i].assign(boundary_flat + begin, boundary_flat + finish);
  }

  std::vector<int> dimensions(num_generators);
  for (size_t i = 0; i < num_generators; ++i) {
    dimensions[i] = static_cast<int>(dimensions32[i]);
  }

  std::vector<typename Concrete::Filtration_value> c_filtrations;
  c_filtrations.reserve(num_generators);

  if constexpr (IsKCritical) {
    const int64_t* grade_indptr =
        serialized_block_view<int64_t>(data, buffer_size, header.grade_indptr_offset, header.num_generators + 1);
    const Value* grades_flat =
        serialized_block_view<Value>(data, buffer_size, header.grades_offset, filtration_rows * num_parameters);
    if (grade_indptr[num_generators] != static_cast<int64_t>(header.filtration_rows)) {
      throw std::runtime_error("Invalid serialized slicer filtrations.");
    }
    for (size_t i = 0; i < num_generators; ++i) {
      int64_t begin = grade_indptr[i];
      int64_t finish = grade_indptr[i + 1];
      if (begin > finish || finish > static_cast<int64_t>(filtration_rows)) {
        throw std::runtime_error("Invalid serialized slicer filtrations.");
      }
      c_filtrations.push_back(filtration_from_serialized_rows<typename Concrete::Filtration_value, Value, IsDegreeRips>(
          grades_flat, static_cast<size_t>(begin), static_cast<size_t>(finish), num_parameters));
    }
  } else {
    const Value* grades_flat =
        serialized_block_view<Value>(data, buffer_size, header.grades_offset, num_generators * num_parameters);
    for (size_t i = 0; i < num_generators; ++i) {
      size_t offset = i * num_parameters;
      if (num_parameters == 0) {
        c_filtrations.emplace_back(std::vector<Value>());
      } else {
        c_filtrations.emplace_back(grades_flat + offset, grades_flat + offset + num_parameters);
      }
    }
  }

  load_slicer_from_generator_data<Wrapper, Concrete>(
      self, std::move(boundaries), std::move(dimensions), std::move(c_filtrations));
}

template <typename Wrapper, typename Value, bool IsKCritical, bool IsDegreeRips>
nb::object serialized_state(Wrapper& self) {
  std::vector<uint8_t> buffer;
  {
    nb::gil_scoped_release release;
    const auto& boundaries = self.truc.get_boundaries();
    const auto& dims = self.truc.get_dimensions();
    const auto& filtrations = self.truc.get_filtration_values();

    size_t num_generators = boundaries.size();
    size_t total_boundary_size = 0;
    size_t total_rows = 0;
    for (size_t i = 0; i < num_generators; ++i) {
      total_boundary_size += boundaries[i].size();
      if constexpr (IsKCritical) {
        total_rows += filtrations[i].num_generators();
      }
    }

    size_t num_parameters = 0;
    size_t filtration_rows = 0;

    if constexpr (IsKCritical) {
      num_parameters = IsDegreeRips ? size_t(2) : static_cast<size_t>(self.truc.get_number_of_parameters());
      filtration_rows = total_rows;
    } else {
      num_parameters = static_cast<size_t>(self.truc.get_number_of_parameters());
      filtration_rows = num_generators;
    }

    SlicerSerializationLayout layout = make_slicer_serialization_layout<Value>(
        num_generators, total_boundary_size, num_parameters, filtration_rows, IsKCritical);
    buffer.resize(layout.total_size);

    SlicerSerializationHeaderV1 header{kSlicerSerializationMagic,
                                       kSlicerSerializationVersion,
                                       expected_slicer_serialization_mode<IsKCritical, IsDegreeRips>(),
                                       0,
                                       static_cast<uint64_t>(num_generators),
                                       static_cast<uint64_t>(total_boundary_size),
                                       static_cast<uint64_t>(num_parameters),
                                       static_cast<uint64_t>(filtration_rows),
                                       static_cast<uint64_t>(layout.boundary_indptr_offset),
                                       static_cast<uint64_t>(layout.boundary_flat_offset),
                                       static_cast<uint64_t>(layout.dimensions_offset),
                                       static_cast<uint64_t>(layout.grade_indptr_offset),
                                       static_cast<uint64_t>(layout.grades_offset),
                                       static_cast<uint64_t>(layout.total_size)};
    std::memcpy(buffer.data(), &header, sizeof(header));

    auto* boundary_indptr = mutable_serialized_block<uint64_t>(buffer, layout.boundary_indptr_offset, num_generators + 1);
    auto* boundary_flat = mutable_serialized_block<uint32_t>(buffer, layout.boundary_flat_offset, total_boundary_size);
    auto* dimensions = mutable_serialized_block<int32_t>(buffer, layout.dimensions_offset, num_generators);
    auto* grades_flat =
        mutable_serialized_block<Value>(buffer, layout.grades_offset, filtration_rows * num_parameters);
    boundary_indptr[0] = 0;

    size_t boundary_offset = 0;
    for (size_t i = 0; i < num_generators; ++i) {
      const auto& row = boundaries[i];
      dimensions[i] = static_cast<int32_t>(dims[i]);
      if (!row.empty()) {
        std::memcpy(boundary_flat + boundary_offset, row.data(), serialized_array_bytes<uint32_t>(row.size()));
      }
      boundary_offset += row.size();
      boundary_indptr[i + 1] = static_cast<uint64_t>(boundary_offset);
    }

    if constexpr (IsKCritical) {
      auto* grade_indptr = mutable_serialized_block<int64_t>(buffer, layout.grade_indptr_offset, num_generators + 1);
      grade_indptr[0] = 0;
      size_t offset = 0;
      for (size_t i = 0; i < num_generators; ++i) {
        size_t k = filtrations[i].num_generators();
        for (size_t g = 0; g < k; ++g) {
          if constexpr (IsDegreeRips) {
            grades_flat[2 * (offset + g)] = filtrations[i](g, 0);
            grades_flat[2 * (offset + g) + 1] = static_cast<Value>(g);
          } else {
            Value* out_row = grades_flat + (offset + g) * num_parameters;
            for (size_t p = 0; p < num_parameters; ++p) {
              out_row[p] = filtrations[i](g, p);
            }
          }
        }
        offset += k;
        grade_indptr[i + 1] = static_cast<int64_t>(offset);
      }
    } else {
      for (size_t i = 0; i < num_generators; ++i) {
        Value* out_row = grades_flat + i * num_parameters;
        if (!filtrations[i].is_finite()) {
          std::fill_n(out_row, num_parameters, filtrations[i](0, 0));
        } else if (num_parameters > 0) {
          std::memcpy(out_row, &filtrations[i](0, 0), num_parameters * sizeof(Value));
        }
      }
    }
  }
  return nb::cast(multipers::nanobind_utils::owned_array<uint8_t>(std::move(buffer), {buffer.size()}));
}

template <typename Wrapper, typename Concrete, typename Value, bool IsKCritical, bool IsDegreeRips>
void load_state(Wrapper& self, nb::handle state) {
  auto buffer = nb::cast<nb::ndarray<nb::numpy, const uint8_t, nb::ndim<1>, nb::c_contig>>(state);
  {
    nb::gil_scoped_release release;
    load_state_v1<Wrapper, Concrete, Value, IsKCritical, IsDegreeRips>(self, buffer.data(), buffer.size());
  }
  multipers::nanobind_helpers::reset_slicer_python_state(self);
}

}  // namespace mpnb

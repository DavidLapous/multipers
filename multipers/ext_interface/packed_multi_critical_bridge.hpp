#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace multipers {

struct packed_multi_critical_bridge_input {
  // Legacy jagged boundary representation — may be empty in CSR mode.
  std::vector<std::vector<uint32_t>> boundaries;

  // CSR boundary backend — active when csr_boundaries_indptr non-empty.
  // When active, boundaries must be empty (or vice versa).
  std::vector<int64_t> csr_boundaries_indptr;
  std::vector<uint32_t> csr_boundaries_indices;

  std::vector<int32_t> dimensions;
  std::vector<int64_t> grade_indptr;
  std::vector<double> grade_values;

  struct RowView {
    const uint32_t* data;
    std::size_t size;
  };

  RowView boundary_row(std::size_t idx) const noexcept {
    if (!csr_boundaries_indptr.empty()) {
      auto begin = static_cast<std::size_t>(csr_boundaries_indptr[idx]);
      auto end = static_cast<std::size_t>(csr_boundaries_indptr[idx + 1]);
      return {csr_boundaries_indices.data() + begin, end - begin};
    }
    return {boundaries[idx].data(), boundaries[idx].size()};
  }

  std::size_t boundary_size() const noexcept {
    if (!csr_boundaries_indptr.empty()) {
      return csr_boundaries_indptr.size() - 1;
    }
    return boundaries.size();
  }
};

}  // namespace multipers

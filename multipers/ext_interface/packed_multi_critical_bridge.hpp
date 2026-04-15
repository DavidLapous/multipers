#pragma once

#include <cstdint>
#include <vector>

namespace multipers {

struct packed_multi_critical_bridge_input {
  std::vector<std::vector<uint32_t>> boundaries;
  std::vector<int32_t> dimensions;
  std::vector<int64_t> grade_indptr;
  std::vector<double> grade_values;
};

}  // namespace multipers

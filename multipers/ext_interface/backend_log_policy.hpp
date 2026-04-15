#pragma once

#include <cstdint>

namespace multipers::backend_log_policy {

enum class backend_log_bit : uint32_t {
  mpfree = 1u << 0,
  multi_critical = 1u << 1,
  function_delaunay = 1u << 2,
  twopac = 1u << 3,
};

using backend_log_mask = uint32_t;

constexpr backend_log_mask bit_mask(backend_log_bit bit) noexcept {
  return static_cast<backend_log_mask>(bit);
}

backend_log_mask get_backend_log_mask() noexcept;
void set_backend_log_mask(backend_log_mask mask) noexcept;
void set_backend_log_enabled(backend_log_bit bit, bool enabled) noexcept;
bool backend_log_enabled(backend_log_bit bit) noexcept;

}  // namespace multipers::backend_log_policy

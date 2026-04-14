#include "ext_interface/backend_log_policy.hpp"

#include <atomic>

namespace multipers::backend_log_policy {

namespace {

std::atomic<backend_log_mask> g_backend_log_mask{0u};

}  // namespace

backend_log_mask get_backend_log_mask() noexcept {
  return g_backend_log_mask.load(std::memory_order_relaxed);
}

void set_backend_log_mask(backend_log_mask mask) noexcept {
  g_backend_log_mask.store(mask, std::memory_order_relaxed);
}

void set_backend_log_enabled(backend_log_bit bit, bool enabled) noexcept {
  backend_log_mask mask = bit_mask(bit);
  if (enabled) {
    g_backend_log_mask.fetch_or(mask, std::memory_order_relaxed);
  } else {
    g_backend_log_mask.fetch_and(~mask, std::memory_order_relaxed);
  }
}

bool backend_log_enabled(backend_log_bit bit) noexcept {
  return (get_backend_log_mask() & bit_mask(bit)) != 0;
}

}  // namespace multipers::backend_log_policy

#pragma once

#include "backend_log_policy.hpp"

namespace multipers::backend_log_policy {

// Vendored backends read `verbose` as a global bool. In Multipers builds,
// route those reads to the shared runtime bitmask without reintroducing
// per-call global writes.
template <backend_log_bit Bit>
struct runtime_flag {
  constexpr runtime_flag() noexcept = default;

  template <typename T>
  runtime_flag& operator=(const T&) noexcept {
    return *this;
  }

  operator bool() const noexcept { return backend_log_enabled(Bit); }
};

template <bool Value>
struct constant_flag {
  constexpr constant_flag() noexcept = default;

  template <typename T>
  constant_flag& operator=(const T&) noexcept {
    return *this;
  }

  constexpr operator bool() const noexcept { return Value; }
};

}  // namespace multipers::backend_log_policy

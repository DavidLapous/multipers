#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "gudhi/Multi_persistence/Module.h"
#include "nanobind_object_utils.hpp"

namespace multipers::nanobind_mma_helpers {

using multipers::nanobind_utils::has_template_id;
using multipers::nanobind_utils::maybe_template_id_of;
using multipers::nanobind_utils::template_id_of;

template <typename T>
struct PyModule {
  Gudhi::multi_persistence::Module<T> mod;
};

template <typename... Types>
struct type_list {};

#include <_mma_nanobind_registry.inc>

template <typename Func>
decltype(auto) dispatch_mma_by_template_id(int template_id, Func&& func) {
  switch (template_id) {
#define MP_MMA_CASE(desc) \
  case desc::template_id: \
    return std::forward<Func>(func).template operator()<desc>();
    MP_FOR_EACH_MMA_DESC(MP_MMA_CASE)
#undef MP_MMA_CASE
    default:
      throw nanobind::type_error("Unknown MMA template id.");
  }
}

inline bool is_known_mma_template_id(int template_id) {
  switch (template_id) {
#define MP_MMA_CASE(desc) \
  case desc::template_id: \
    return true;
    MP_FOR_EACH_MMA_DESC(MP_MMA_CASE)
#undef MP_MMA_CASE
    default:
      return false;
  }
}

template <typename Desc>
using module_wrapper_t = PyModule<typename Desc::value_type>;

template <typename List>
struct mma_desc_for_double_impl;

template <>
struct mma_desc_for_double_impl<type_list<>> {
  using type = void;
  static constexpr int matches = 0;
};

template <typename Head, typename... Tail>
struct mma_desc_for_double_impl<type_list<Head, Tail...>> {
  using tail = mma_desc_for_double_impl<type_list<Tail...>>;
  static constexpr bool is_match = std::is_same_v<typename Head::value_type, double>;
  using type = std::conditional_t<is_match, Head, typename tail::type>;
  static constexpr int matches = tail::matches + (is_match ? 1 : 0);
};

using canonical_double_mma_desc = typename mma_desc_for_double_impl<MMADescriptorList>::type;
static_assert(!std::is_void_v<canonical_double_mma_desc>, "Expected an MMA descriptor with double value type.");
static_assert(mma_desc_for_double_impl<MMADescriptorList>::matches == 1,
              "Expected exactly one MMA descriptor with double value type.");

template <typename Func>
decltype(auto) visit_mma_module_wrapper(const nanobind::handle& input, Func&& func) {
  return dispatch_mma_by_template_id(template_id_of(input), [&]<typename Desc>() -> decltype(auto) {
    auto& wrapper = nanobind::cast<module_wrapper_t<Desc>&>(input);
    return std::forward<Func>(func).template operator()<Desc>(wrapper);
  });
}

template <typename Func>
decltype(auto) visit_const_mma_module_wrapper(const nanobind::handle& input, Func&& func) {
  return dispatch_mma_by_template_id(template_id_of(input), [&]<typename Desc>() -> decltype(auto) {
    const auto& wrapper = nanobind::cast<const module_wrapper_t<Desc>&>(input);
    return std::forward<Func>(func).template operator()<Desc>(wrapper);
  });
}

inline bool is_mma_module_object(const nanobind::handle& input) {
  std::optional<int> template_id = maybe_template_id_of(input);
  if (!template_id || !is_known_mma_template_id(*template_id)) {
    return false;
  }
  return dispatch_mma_by_template_id(
      *template_id, [&]<typename Desc>() -> bool { return nanobind::isinstance<module_wrapper_t<Desc>>(input); });
}

}  // namespace multipers::nanobind_mma_helpers

#pragma once

#include <nanobind/nanobind.h>

#include <type_traits>
#include <utility>

#include "Persistence_slices_interface.h"
#include "Simplex_tree_multi_interface.h"
#include "nanobind_wrapper_types.hpp"
#include "nanobind_object_utils.hpp"

namespace nb = nanobind;

namespace multipers::nanobind_helpers {

template <typename... Types>
struct type_list {};

#include <_slicer_nanobind_registry.inc>

using multipers::nanobind_utils::has_template_id;
using multipers::nanobind_utils::template_id_of;

template <typename Desc>
using simplextree_wrapper_t = PySimplexTree<typename Desc::interface_type, typename Desc::value_type>;

template <typename Func>
decltype(auto) dispatch_slicer_by_template_id(int template_id, Func&& func) {
  switch (template_id) {
#define MP_SLICER_CASE(desc) \
  case desc::template_id:    \
    return std::forward<Func>(func).template operator()<desc>();
    MP_FOR_EACH_SLICER_DESC(MP_SLICER_CASE)
#undef MP_SLICER_CASE
    default:
      throw nb::type_error("Unknown slicer template id.");
  }
}

template <typename Func>
decltype(auto) dispatch_simplextree_by_template_id(int template_id, Func&& func) {
  switch (template_id) {
#define MP_SIMPLEXTREE_CASE(desc) \
  case desc::template_id:         \
    return std::forward<Func>(func).template operator()<desc>();
    MP_FOR_EACH_SIMPLEXTREE_DESC(MP_SIMPLEXTREE_CASE)
#undef MP_SIMPLEXTREE_CASE
    default:
      throw nb::type_error("Unknown SimplexTreeMulti template id.");
  }
}

template <typename Func>
decltype(auto) visit_slicer_wrapper(const nb::handle& input, Func&& func) {
  return dispatch_slicer_by_template_id(template_id_of(input), [&]<typename Desc>() -> decltype(auto) {
    auto& wrapper = nb::cast<typename Desc::wrapper&>(input);
    return std::forward<Func>(func).template operator()<Desc>(wrapper);
  });
}

template <typename Func>
decltype(auto) visit_const_slicer_wrapper(const nb::handle& input, Func&& func) {
  return dispatch_slicer_by_template_id(template_id_of(input), [&]<typename Desc>() -> decltype(auto) {
    const auto& wrapper = nb::cast<const typename Desc::wrapper&>(input);
    return std::forward<Func>(func).template operator()<Desc>(wrapper);
  });
}

template <typename Func>
decltype(auto) visit_simplextree_wrapper(const nb::handle& input, Func&& func) {
  return dispatch_simplextree_by_template_id(template_id_of(input), [&]<typename Desc>() -> decltype(auto) {
    auto& wrapper = nb::cast<simplextree_wrapper_t<Desc>&>(input);
    return std::forward<Func>(func).template operator()<Desc>(wrapper);
  });
}

template <typename Func>
decltype(auto) visit_const_simplextree_wrapper(const nb::handle& input, Func&& func) {
  return dispatch_simplextree_by_template_id(template_id_of(input), [&]<typename Desc>() -> decltype(auto) {
    const auto& wrapper = nb::cast<const simplextree_wrapper_t<Desc>&>(input);
    return std::forward<Func>(func).template operator()<Desc>(wrapper);
  });
}

inline bool is_slicer_object(const nb::handle& input) {
  if (!has_template_id(input)) {
    return false;
  }
  try {
    return visit_const_slicer_wrapper(input,
                                      [&]<typename Desc>(const typename Desc::wrapper&) -> bool { return true; });
  } catch (...) {
    return false;
  }
}

inline bool is_simplextree_object(const nb::handle& input) {
  if (!has_template_id(input)) {
    return false;
  }
  try {
    return visit_const_simplextree_wrapper(
        input, [&]<typename Desc>(const simplextree_wrapper_t<Desc>&) -> bool { return true; });
  } catch (...) {
    return false;
  }
}

}  // namespace multipers::nanobind_helpers

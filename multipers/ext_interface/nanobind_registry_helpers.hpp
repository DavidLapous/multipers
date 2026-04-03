#pragma once

#include <nanobind/nanobind.h>

#include <optional>
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
using multipers::nanobind_utils::maybe_template_id_of;
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

inline bool is_known_slicer_template_id(int template_id) {
  switch (template_id) {
#define MP_SLICER_CASE(desc) \
  case desc::template_id:    \
    return true;
    MP_FOR_EACH_SLICER_DESC(MP_SLICER_CASE)
#undef MP_SLICER_CASE
    default:
      return false;
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

inline bool is_known_simplextree_template_id(int template_id) {
  switch (template_id) {
#define MP_SIMPLEXTREE_CASE(desc) \
  case desc::template_id:         \
    return true;
    MP_FOR_EACH_SIMPLEXTREE_DESC(MP_SIMPLEXTREE_CASE)
#undef MP_SIMPLEXTREE_CASE
    default:
      return false;
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

template <typename... Ds>
int select_slicer_template_id(type_list<Ds...>,
                              bool is_vineyard,
                              bool is_k_critical,
                              std::string_view dtype_name,
                              std::string col,
                              std::string pers_backend,
                              std::string filtration_container) {
  col = nanobind_utils::lowercase_copy(std::move(col));
  pers_backend = nanobind_utils::lowercase_copy(std::move(pers_backend));
  filtration_container = nanobind_utils::lowercase_copy(std::move(filtration_container));

  bool matched = false;
  int result = -1;
  (
      [&]<typename D>() {
        if (!matched && D::is_vine == is_vineyard && D::is_kcritical == is_k_critical &&
            D::dtype_name == dtype_name && nanobind_utils::lowercase_copy(std::string(D::column_type)) == col &&
            nanobind_utils::lowercase_copy(std::string(D::backend_type)) == pers_backend &&
            nanobind_utils::lowercase_copy(std::string(D::filtration_container)) == filtration_container) {
          result = D::template_id;
          matched = true;
        }
      }.template operator()<Ds>(),
      ...);

  if (!matched) {
    throw nb::value_error("Unimplemented slicer combination.");
  }
  return result;
}

template <typename... Ds>
bool has_slicer_filtration_container(type_list<Ds...>, std::string filtration_container) {
  filtration_container = nanobind_utils::lowercase_copy(std::move(filtration_container));
  bool found = false;
  (
      [&]<typename D>() {
        if (!found && nanobind_utils::lowercase_copy(std::string(D::filtration_container)) == filtration_container) {
          found = true;
        }
      }.template operator()<Ds>(),
      ...);
  return found;
}

inline int related_slicer_template_id(const nb::handle& source, bool is_kcritical, const std::string& filtration_container) {
  return visit_const_slicer_wrapper(source, [&]<typename Desc>(const typename Desc::wrapper&) -> int {
    return select_slicer_template_id(SlicerDescriptorList{},
                                     Desc::is_vine,
                                     is_kcritical,
                                     Desc::dtype_name,
                                     std::string(Desc::column_type),
                                     std::string(Desc::backend_type),
                                     filtration_container);
  });
}

inline bool is_slicer_object(const nb::handle& input) {
  std::optional<int> template_id = maybe_template_id_of(input);
  if (!template_id || !is_known_slicer_template_id(*template_id)) {
    return false;
  }
  return dispatch_slicer_by_template_id(
      *template_id, [&]<typename Desc>() -> bool { return nb::isinstance<typename Desc::wrapper>(input); });
}

inline bool is_simplextree_object(const nb::handle& input) {
  std::optional<int> template_id = maybe_template_id_of(input);
  if (!template_id || !is_known_simplextree_template_id(*template_id)) {
    return false;
  }
  return dispatch_simplextree_by_template_id(
      *template_id, [&]<typename Desc>() -> bool { return nb::isinstance<simplextree_wrapper_t<Desc>>(input); });
}

}  // namespace multipers::nanobind_helpers

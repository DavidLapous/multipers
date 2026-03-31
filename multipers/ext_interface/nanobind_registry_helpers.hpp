#pragma once

#include <nanobind/nanobind.h>

#include <type_traits>
#include <utility>

#include "Persistence_slices_interface.h"
#include "Simplex_tree_multi_interface.h"
#include "nanobind_object_utils.hpp"
#include "contiguous_slicer_bridge.hpp"

namespace nb = nanobind;

namespace multipers::nanobind_helpers {

template <typename Slicer>
struct PySlicer {
  Slicer truc;
  nb::object filtration_grid;
  nb::object generator_basis;
  int minpres_degree;

  PySlicer() : filtration_grid(nb::none()), generator_basis(nb::none()), minpres_degree(-1) {}
};

template <typename Interface, typename T>
struct PySimplexTree {
  Interface tree;
  nb::object filtration_grid;

  PySimplexTree() : filtration_grid(nb::list()) {}
};

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

template <typename Desc>
inline constexpr bool is_canonical_contiguous_f64_slicer_v =
    std::is_same_v<typename Desc::concrete, multipers::contiguous_f64_slicer>;

template <typename List>
struct canonical_contiguous_f64_slicer_desc_impl;

template <>
struct canonical_contiguous_f64_slicer_desc_impl<type_list<>> {
  using type = void;
  static constexpr int matches = 0;
};

template <typename Head, typename... Tail>
struct canonical_contiguous_f64_slicer_desc_impl<type_list<Head, Tail...>> {
  using tail = canonical_contiguous_f64_slicer_desc_impl<type_list<Tail...>>;
  static constexpr bool is_match = is_canonical_contiguous_f64_slicer_v<Head>;
  using type = std::conditional_t<is_match, Head, typename tail::type>;
  static constexpr int matches = tail::matches + (is_match ? 1 : 0);
};

using canonical_contiguous_f64_slicer_desc =
    typename canonical_contiguous_f64_slicer_desc_impl<SlicerDescriptorList>::type;
using canonical_contiguous_f64_slicer = typename canonical_contiguous_f64_slicer_desc::concrete;

inline constexpr int canonical_contiguous_f64_slicer_template_id = canonical_contiguous_f64_slicer_desc::template_id;

static_assert(!std::is_void_v<canonical_contiguous_f64_slicer_desc>,
              "Expected exactly one canonical contiguous float64 matrix slicer template.");
static_assert(canonical_contiguous_f64_slicer_desc_impl<SlicerDescriptorList>::matches == 1,
              "Canonical contiguous float64 matrix slicer template must be unique.");

inline bool is_canonical_contiguous_f64_slicer_object(const nb::handle& input) {
  return has_template_id(input) && template_id_of(input) == canonical_contiguous_f64_slicer_template_id;
}

inline nb::object canonical_contiguous_f64_slicer_class() {
  return nb::borrow<nb::object>(nb::type<typename canonical_contiguous_f64_slicer_desc::wrapper>());
}

inline nb::object slicer_class_from_template_id(int template_id) {
  return dispatch_slicer_by_template_id(template_id, [&]<typename Desc>() -> nb::object {
    return nb::borrow<nb::object>(nb::type<typename Desc::wrapper>());
  });
}

inline nb::object simplextree_class_from_template_id(int template_id) {
  return dispatch_simplextree_by_template_id(template_id, [&]<typename Desc>() -> nb::object {
    return nb::borrow<nb::object>(nb::type<simplextree_wrapper_t<Desc>>());
  });
}

inline nb::object astype_slicer_to_template_id(const nb::object& source, int template_id) {
  if (template_id_of(source) == template_id) {
    return source;
  }
  return slicer_class_from_template_id(template_id)(source);
}

inline nb::object astype_simplextree_to_template_id(const nb::object& source, int template_id) {
  if (template_id_of(source) == template_id) {
    return source;
  }
  nb::object out = simplextree_class_from_template_id(template_id)();
  out.attr("_copy_from_any")(source);
  return out;
}

inline nb::object astype_slicer_to_original_type(const nb::object& original, const nb::object& source) {
  return astype_slicer_to_template_id(source, template_id_of(original));
}

inline nb::object astype_simplextree_to_original_type(const nb::object& original, const nb::object& source) {
  return astype_simplextree_to_template_id(source, template_id_of(original));
}

inline void copy_into_canonical_contiguous_f64_slicer(const nb::handle& input,
                                                      canonical_contiguous_f64_slicer& output) {
  visit_const_slicer_wrapper(input, [&]<typename Desc>(const typename Desc::wrapper& source) {
    output = canonical_contiguous_f64_slicer(source.truc);
  });
}

inline nb::object ensure_canonical_contiguous_f64_slicer_object(const nb::object& input) {
  if (is_canonical_contiguous_f64_slicer_object(input)) {
    return input;
  }
  nb::object out = canonical_contiguous_f64_slicer_class()();
  auto& out_wrapper = nb::cast<typename canonical_contiguous_f64_slicer_desc::wrapper&>(out);
  copy_into_canonical_contiguous_f64_slicer(input, out_wrapper.truc);
  visit_const_slicer_wrapper(input, [&]<typename Desc>(const typename Desc::wrapper& source) {
    out_wrapper.filtration_grid = source.filtration_grid;
    out_wrapper.generator_basis = source.generator_basis;
    out_wrapper.minpres_degree = source.minpres_degree;
  });
  return out;
}

}  // namespace multipers::nanobind_helpers

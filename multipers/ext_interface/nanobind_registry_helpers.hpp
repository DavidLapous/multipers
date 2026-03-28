#pragma once

#include <nanobind/nanobind.h>

#include <type_traits>

#include "Persistence_slices_interface.h"
#include "Simplex_tree_multi_interface.h"
#include "contiguous_slicer_bridge.hpp"

namespace nb = nanobind;

namespace multipers::nanobind_helpers {

template <typename Slicer>
struct PySlicer {
  Slicer truc;
  nb::object filtration_grid;
  int minpres_degree;

  PySlicer() : filtration_grid(nb::none()), minpres_degree(-1) {}
};

template <typename... Types>
struct type_list {};

#include <_slicer_nanobind_registry.inc>

inline bool has_template_id(const nb::handle& input) { return nb::hasattr(input, "_template_id"); }

inline int template_id_of(const nb::handle& input) {
  if (!has_template_id(input)) {
    throw nb::type_error("Object does not expose a template id.");
  }
  return nb::cast<int>(input.attr("_template_id"));
}

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

inline nb::module_ slicer_nanobind_module() { return nb::module_::import_("multipers._slicer_nanobind"); }

inline nb::module_ simplextree_nanobind_module() {
  return nb::module_::import_("multipers._simplex_tree_multi_nanobind");
}

inline nb::object canonical_contiguous_f64_slicer_class() {
  return slicer_nanobind_module().attr("_get_slicer_class_from_template_id")(
      canonical_contiguous_f64_slicer_template_id);
}

inline nb::object slicer_class_from_template_id(int template_id) {
  return slicer_nanobind_module().attr("_get_slicer_class_from_template_id")(template_id);
}

inline nb::object simplextree_class_from_template_id(int template_id) {
  return simplextree_nanobind_module().attr("_get_simplextree_class_from_template_id")(template_id);
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
  dispatch_slicer_by_template_id(template_id_of(input), [&]<typename Desc>() {
    auto* source = reinterpret_cast<const typename Desc::concrete*>(nb::cast<intptr_t>(input.attr("get_ptr")()));
    output = canonical_contiguous_f64_slicer(*source);
  });
}

inline nb::object ensure_canonical_contiguous_f64_slicer_object(const nb::object& input) {
  if (is_canonical_contiguous_f64_slicer_object(input)) {
    return input;
  }
  nb::object out = canonical_contiguous_f64_slicer_class()();
  auto* out_cpp = reinterpret_cast<canonical_contiguous_f64_slicer*>(nb::cast<intptr_t>(out.attr("get_ptr")()));
  copy_into_canonical_contiguous_f64_slicer(input, *out_cpp);
  if (nb::hasattr(input, "filtration_grid")) {
    out.attr("filtration_grid") = input.attr("filtration_grid");
  }
  if (nb::hasattr(input, "minpres_degree")) {
    out.attr("minpres_degree") = input.attr("minpres_degree");
  }
  return out;
}

}  // namespace multipers::nanobind_helpers

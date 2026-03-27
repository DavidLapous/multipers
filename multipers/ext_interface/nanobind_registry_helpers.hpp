#pragma once

#include <nanobind/nanobind.h>

#include <type_traits>

#include "Persistence_slices_interface.h"
#include "Simplex_tree_multi_interface.h"

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

#include "_slicer_nanobind_registry.inc"

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
inline constexpr bool is_contiguous_f64_slicer_v =
    std::is_same_v<typename Desc::value_type, double> && !Desc::is_kcritical && !Desc::is_degree_rips;

template <typename Desc>
inline constexpr bool is_contiguous_f64_simplextree_v =
    std::is_same_v<typename Desc::value_type, double> && !Desc::is_kcritical;

inline bool is_supported_contiguous_f64_slicer_object(const nb::handle& input) {
  return dispatch_slicer_by_template_id(template_id_of(input),
                                        []<typename Desc>() { return is_contiguous_f64_slicer_v<Desc>; });
}

inline bool is_supported_contiguous_f64_simplextree_object(const nb::handle& input) {
  return dispatch_simplextree_by_template_id(template_id_of(input),
                                             []<typename Desc>() { return is_contiguous_f64_simplextree_v<Desc>; });
}

}  // namespace multipers::nanobind_helpers

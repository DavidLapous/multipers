#include "ext_interface/nanobind_registry_runtime.hpp"

#include "ext_interface/nanobind_registry_helpers.hpp"

namespace nb = nanobind;

namespace multipers::nanobind_helpers {

namespace {

inline bool is_canonical_contiguous_f64_slicer_object(const nb::handle& input) {
  return nb::isinstance<canonical_contiguous_f64_slicer_wrapper>(input);
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

}  // namespace

nb::object astype_slicer_to_template_id(const nb::object& source, int template_id) {
  if (template_id_of(source) == template_id) {
    return source;
  }
  return slicer_class_from_template_id(template_id)(source);
}

nb::object astype_simplextree_to_template_id(const nb::object& source, int template_id) {
  if (template_id_of(source) == template_id) {
    return source;
  }
  nb::object out = simplextree_class_from_template_id(template_id)();
  out.attr("_copy_from_any")(source);
  return out;
}

nb::object astype_slicer_to_original_type(const nb::object& original, const nb::object& source) {
  return astype_slicer_to_template_id(source, template_id_of(original));
}

nb::object astype_simplextree_to_original_type(const nb::object& original, const nb::object& source) {
  return astype_simplextree_to_template_id(source, template_id_of(original));
}

void copy_into_canonical_contiguous_f64_slicer(const nb::handle& input, canonical_contiguous_f64_slicer& output) {
  visit_const_slicer_wrapper(input, [&]<typename Desc>(const typename Desc::wrapper& source) {
    output = canonical_contiguous_f64_slicer(source.truc);
  });
}

nb::object ensure_canonical_contiguous_f64_slicer_object(const nb::object& input) {
  if (is_canonical_contiguous_f64_slicer_object(input)) {
    return input;
  }

  nb::object out = nb::borrow<nb::object>(nb::type<canonical_contiguous_f64_slicer_wrapper>())();
  auto& out_wrapper = nb::cast<canonical_contiguous_f64_slicer_wrapper&>(out);
  copy_into_canonical_contiguous_f64_slicer(input, out_wrapper.truc);
  visit_const_slicer_wrapper(input, [&]<typename Desc>(const typename Desc::wrapper& source) {
    out_wrapper.filtration_grid = source.filtration_grid;
    out_wrapper.generator_basis = source.generator_basis;
    out_wrapper.minpres_degree = source.minpres_degree;
  });
  return out;
}

}  // namespace multipers::nanobind_helpers

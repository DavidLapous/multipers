#include "ext_interface/nanobind_registry_runtime.hpp"

#include "ext_interface/nanobind_registry_helpers.hpp"

namespace nb = nanobind;

namespace multipers::nanobind_helpers {

namespace {

inline bool is_canonical_contiguous_f64_slicer_object(const nb::handle& input) {
  return nb::isinstance<canonical_contiguous_f64_slicer_wrapper>(input);
}

inline bool is_canonical_kcontiguous_f64_slicer_object(const nb::handle& input) {
  return nb::isinstance<canonical_kcontiguous_f64_slicer_wrapper>(input);
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

template <typename CanonicalSlicer>
void copy_into_canonical_slicer_impl(const nb::handle& input, CanonicalSlicer& output) {
  visit_const_slicer_wrapper(
      input, [&]<typename Desc>(const typename Desc::wrapper& source) { output = CanonicalSlicer(source.truc); });
}

template <typename CanonicalWrapper, typename CanonicalSlicer, typename IsCanonical>
nb::object ensure_canonical_slicer_object_impl(const nb::object& input, IsCanonical&& is_canonical) {
  if (std::forward<IsCanonical>(is_canonical)(input)) {
    return input;
  }

  nb::object out = nb::borrow<nb::object>(nb::type<CanonicalWrapper>())();
  auto& out_wrapper = nb::cast<CanonicalWrapper&>(out);
  copy_into_canonical_slicer_impl(input, out_wrapper.truc);
  visit_const_slicer_wrapper(input, [&]<typename Desc>(const typename Desc::wrapper& source) {
    out_wrapper.filtration_grid = source.filtration_grid;
    out_wrapper.generator_basis = source.generator_basis;
    out_wrapper.minpres_degree = source.minpres_degree;
  });
  return out;
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
  copy_into_canonical_slicer_impl(input, output);
}

nb::object ensure_canonical_contiguous_f64_slicer_object(const nb::object& input) {
  return ensure_canonical_slicer_object_impl<canonical_contiguous_f64_slicer_wrapper, canonical_contiguous_f64_slicer>(
      input, is_canonical_contiguous_f64_slicer_object);
}

void copy_into_canonical_kcontiguous_f64_slicer(const nb::handle& input, canonical_kcontiguous_f64_slicer& output) {
  copy_into_canonical_slicer_impl(input, output);
}

nb::object ensure_canonical_kcontiguous_f64_slicer_object(const nb::object& input) {
  return ensure_canonical_slicer_object_impl<canonical_kcontiguous_f64_slicer_wrapper,
                                             canonical_kcontiguous_f64_slicer>(
      input, is_canonical_kcontiguous_f64_slicer_object);
}

}  // namespace multipers::nanobind_helpers

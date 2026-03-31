#pragma once

#include <nanobind/nanobind.h>

#include "contiguous_slicer_bridge.hpp"
#include "nanobind_wrapper_types.hpp"

namespace multipers::nanobind_helpers {

using canonical_contiguous_f64_slicer = multipers::contiguous_f64_slicer;
using canonical_contiguous_f64_slicer_wrapper = PySlicer<canonical_contiguous_f64_slicer>;

nanobind::object astype_slicer_to_template_id(const nanobind::object& source, int template_id);
nanobind::object astype_simplextree_to_template_id(const nanobind::object& source, int template_id);
nanobind::object astype_slicer_to_original_type(const nanobind::object& original, const nanobind::object& source);
nanobind::object astype_simplextree_to_original_type(const nanobind::object& original, const nanobind::object& source);
void copy_into_canonical_contiguous_f64_slicer(const nanobind::handle& input, canonical_contiguous_f64_slicer& output);
nanobind::object ensure_canonical_contiguous_f64_slicer_object(const nanobind::object& input);

}  // namespace multipers::nanobind_helpers

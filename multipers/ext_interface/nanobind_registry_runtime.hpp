#pragma once

#include <nanobind/nanobind.h>

#include <utility>
#include <vector>

#include "contiguous_slicer_bridge.hpp"
#include "nanobind_wrapper_types.hpp"

namespace multipers::nanobind_helpers {

using canonical_contiguous_f64_slicer = multipers::contiguous_f64_slicer;
using canonical_contiguous_f64_slicer_wrapper = PySlicer<canonical_contiguous_f64_slicer>;
using canonical_kcontiguous_f64_slicer = multipers::kcontiguous_f64_slicer;
using canonical_kcontiguous_f64_slicer_wrapper = PySlicer<canonical_kcontiguous_f64_slicer>;

template <typename Wrapper>
Wrapper colexical_slicer_copy(const Wrapper& source) {
  decltype(build_permuted_slicer(source.truc)) stuff;
  {
    nanobind::gil_scoped_release release;
    stuff = build_permuted_slicer(source.truc);
  }
  Wrapper out;
  out.truc = std::move(stuff.first);
  copy_slicer_python_state(out, source);
  return out;
}

template <typename Wrapper>
std::pair<Wrapper, std::vector<uint32_t> > colexical_slicer_copy_with_permutation(const Wrapper& source) {
  decltype(build_permuted_slicer(source.truc)) stuff;
  {
    nanobind::gil_scoped_release release;
    stuff = build_permuted_slicer(source.truc);
  }
  Wrapper out;
  out.truc = std::move(stuff.first);
  copy_slicer_python_state(out, source);
  return {std::move(out), std::vector<uint32_t>(stuff.second.begin(), stuff.second.end())};
}

template <typename Wrapper>
Wrapper permuted_slicer_copy(const Wrapper& source, const std::vector<uint32_t>& permutation) {
  Wrapper out;
  {
    nanobind::gil_scoped_release release;
    out.truc = build_permuted_slicer(source.truc, permutation);
  }
  copy_slicer_python_state(out, source);
  return out;
}

nanobind::object astype_slicer_to_template_id(const nanobind::object& source, int template_id);
nanobind::object astype_simplextree_to_template_id(const nanobind::object& source, int template_id);
nanobind::object astype_slicer_to_original_type(const nanobind::object& original, const nanobind::object& source);
nanobind::object astype_simplextree_to_original_type(const nanobind::object& original, const nanobind::object& source);
nanobind::object rewrap_slicer_output_to_original_type(const nanobind::object& original,
                                                       const nanobind::object& canonical_target,
                                                       const nanobind::object& output);
void copy_into_canonical_contiguous_f64_slicer(const nanobind::handle& input, canonical_contiguous_f64_slicer& output);
nanobind::object ensure_canonical_contiguous_f64_slicer_object(const nanobind::object& input);
void copy_into_canonical_kcontiguous_f64_slicer(const nanobind::handle& input,
                                                canonical_kcontiguous_f64_slicer& output);
nanobind::object ensure_canonical_kcontiguous_f64_slicer_object(const nanobind::object& input);

template <typename Func>
nanobind::object run_with_canonical_contiguous_f64_slicer_output(const nanobind::object& original, Func&& func) {
  nanobind::object canonical_target = ensure_canonical_contiguous_f64_slicer_object(original);
  nanobind::object output = std::forward<Func>(func)(canonical_target);
  return rewrap_slicer_output_to_original_type(original, canonical_target, output);
}

template <typename Func>
nanobind::object run_with_canonical_kcontiguous_f64_slicer_output(const nanobind::object& original, Func&& func) {
  nanobind::object canonical_target = ensure_canonical_kcontiguous_f64_slicer_object(original);
  nanobind::object output = std::forward<Func>(func)(canonical_target);
  return rewrap_slicer_output_to_original_type(original, canonical_target, output);
}

}  // namespace multipers::nanobind_helpers

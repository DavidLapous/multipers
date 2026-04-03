#pragma once

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "contiguous_slicer_bridge.hpp"
#include "nanobind_wrapper_types.hpp"

namespace multipers::nanobind_helpers {

using canonical_contiguous_f64_slicer = multipers::contiguous_f64_slicer;
using canonical_contiguous_f64_slicer_wrapper = PySlicer<canonical_contiguous_f64_slicer>;
using canonical_kcontiguous_f64_slicer = multipers::kcontiguous_f64_slicer;
using canonical_kcontiguous_f64_slicer_wrapper = PySlicer<canonical_kcontiguous_f64_slicer>;

struct BifiltrationMinpresDegreeBlock {
  int degree = -1;
  nanobind::object filtration_grid = nanobind::none();
  bool is_squeezed = false;
  std::size_t row_begin = 0;
  std::size_t row_end = 0;
  std::size_t col_end = 0;
  std::vector<std::pair<double, double>> row_grades;
  std::vector<std::pair<double, double>> relation_grades;
  std::vector<std::vector<int>> relation_boundaries;
};

template <typename Wrapper>
BifiltrationMinpresDegreeBlock extract_bifiltration_minpres_degree_block(const Wrapper& wrapper, int degree) {
  if (degree < 0) {
    throw std::runtime_error("Expected a minimal-presentation slicer.");
  }
  if (wrapper.truc.get_number_of_parameters() != 2) {
    throw std::runtime_error("Only 2-parameter minimal-presentation slicers are supported.");
  }

  BifiltrationMinpresDegreeBlock out;
  out.degree = degree;
  out.filtration_grid = wrapper.filtration_grid;
  out.is_squeezed = has_nonempty_filtration_grid(wrapper.filtration_grid);

  const auto& dimensions = wrapper.truc.get_dimensions();
  const auto& filtrations = wrapper.truc.get_filtration_values();
  const auto& boundaries = wrapper.truc.get_boundaries();

  out.row_begin = std::lower_bound(dimensions.begin(), dimensions.end(), degree) - dimensions.begin();
  out.row_end = std::lower_bound(dimensions.begin(), dimensions.end(), degree + 1) - dimensions.begin();
  out.col_end = std::lower_bound(dimensions.begin(), dimensions.end(), degree + 2) - dimensions.begin();

  out.row_grades.reserve(out.row_end - out.row_begin);
  for (std::size_t i = out.row_begin; i < out.row_end; ++i) {
    out.row_grades.emplace_back(filtrations[i](0, 0), filtrations[i](0, 1));
  }

  out.relation_grades.reserve(out.col_end - out.row_end);
  out.relation_boundaries.reserve(out.col_end - out.row_end);
  for (std::size_t i = out.row_end; i < out.col_end; ++i) {
    out.relation_grades.emplace_back(filtrations[i](0, 0), filtrations[i](0, 1));
    auto& relation = out.relation_boundaries.emplace_back();
    relation.reserve(boundaries[i].size());
    for (const auto boundary_index : boundaries[i]) {
      relation.push_back(static_cast<int>(boundary_index));
    }
  }

  return out;
}

inline std::vector<std::vector<int>> localize_degree_block_relation_boundaries(
    const BifiltrationMinpresDegreeBlock& block) {
  std::vector<std::vector<int>> out;
  out.reserve(block.relation_boundaries.size());
  for (const auto& relation : block.relation_boundaries) {
    auto& localized = out.emplace_back();
    localized.reserve(relation.size());
    for (int boundary_index : relation) {
      if (boundary_index < static_cast<int>(block.row_begin) || boundary_index >= static_cast<int>(block.row_end)) {
        throw std::runtime_error(
            "Invalid minimal presentation slicer: relation boundaries must reference degree-d generators only.");
      }
      localized.push_back(boundary_index - static_cast<int>(block.row_begin));
    }
  }
  return out;
}

template <typename Complex>
nanobind::object build_canonical_contiguous_f64_slicer_object_from_complex(const nanobind::object& target,
                                                                           Complex& complex) {
  nanobind::object out = target.type()();
  auto& out_wrapper = nanobind::cast<canonical_contiguous_f64_slicer_wrapper&>(out);
  build_slicer_from_complex(out_wrapper.truc, complex);
  return out;
}

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

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace multipers {

template <typename index_type>
struct hera_module_presentation_input {
  std::vector<std::pair<double, double> > generator_grades;
  std::vector<std::pair<double, double> > relation_grades;
  std::vector<std::vector<index_type> > relation_components;
};

struct hera_interface_params {
  double hera_epsilon = 0.001;
  double delta = 0.1;
  int max_depth = 8;
  int initialization_depth = 2;
  int bound_strategy = 4;
  int traverse_strategy = 1;
  bool tolerate_max_iter_exceeded = false;
  bool stop_asap = true;
};

struct hera_interface_result {
  double distance = 0.0;
  double actual_error = 0.0;
  int actual_max_depth = 0;
  int n_hera_calls = 0;
};

struct hera_wasserstein_params {
  double wasserstein_power = 1.0;
  double internal_p = std::numeric_limits<double>::infinity();
  double delta = 0.01;
};

inline bool hera_interface_available();

template <typename index_type>
hera_interface_result hera_matching_distance(const hera_module_presentation_input<index_type>& left,
                                             const hera_module_presentation_input<index_type>& right,
                                             const hera_interface_params& params = hera_interface_params());

double hera_bottleneck_distance(const std::vector<std::pair<double, double> >& left,
                                const std::vector<std::pair<double, double> >& right,
                                double delta = 0.01);

double hera_wasserstein_distance(const std::vector<std::pair<double, double> >& left,
                                 const std::vector<std::pair<double, double> >& right,
                                 const hera_wasserstein_params& params = hera_wasserstein_params());

}  // namespace multipers

#ifndef MULTIPERS_DISABLE_HERA_INTERFACE
#define MULTIPERS_DISABLE_HERA_INTERFACE 0
#endif

#if !MULTIPERS_DISABLE_HERA_INTERFACE && __has_include(<hera/matching_distance.h>) && __has_include(<hera/bottleneck.h>) && __has_include(<hera/wasserstein.h>)
#define MULTIPERS_HAS_HERA_INTERFACE 1
#include <hera/bottleneck.h>
#include <hera/matching_distance.h>
#include <hera/wasserstein.h>
#else
#define MULTIPERS_HAS_HERA_INTERFACE 0
#endif

namespace multipers {

inline bool hera_interface_available() { return MULTIPERS_HAS_HERA_INTERFACE; }

#if MULTIPERS_HAS_HERA_INTERFACE

namespace detail {

template <typename index_type>
bool hera_inputs_equal(const hera_module_presentation_input<index_type>& left,
                       const hera_module_presentation_input<index_type>& right) {
  return left.generator_grades == right.generator_grades && left.relation_grades == right.relation_grades &&
         left.relation_components == right.relation_components;
}

inline md::BoundStrategy hera_bound_strategy_from_int(int code) {
  switch (code) {
    case 0:
      return md::BoundStrategy::bruteforce;
    case 1:
      return md::BoundStrategy::local_dual_bound;
    case 2:
      return md::BoundStrategy::local_dual_bound_refined;
    case 3:
      return md::BoundStrategy::local_dual_bound_for_each_point;
    case 4:
      return md::BoundStrategy::local_combined;
    default:
      throw std::invalid_argument("Unknown Hera bound strategy code.");
  }
}

inline md::TraverseStrategy hera_traverse_strategy_from_int(int code) {
  switch (code) {
    case 0:
      return md::TraverseStrategy::depth_first;
    case 1:
      return md::TraverseStrategy::breadth_first;
    case 2:
      return md::TraverseStrategy::breadth_first_value;
    case 3:
      return md::TraverseStrategy::upper_bound;
    default:
      throw std::invalid_argument("Unknown Hera traverse strategy code.");
  }
}

inline std::vector<std::pair<double, double> > filter_diagonal_points(
    const std::vector<std::pair<double, double> >& input) {
  std::vector<std::pair<double, double> > out;
  out.reserve(input.size());
  for (const auto& point : input) {
    if (point.first != point.second) {
      out.push_back(point);
    }
  }
  return out;
}

inline double hera_internal_p_from_double(double value) {
  if (std::isinf(value)) {
    return hera::get_infinity<double>();
  }
  return value;
}

template <typename index_type>
md::ModulePresentation<double> to_hera_module(const hera_module_presentation_input<index_type>& input) {
  using Module = md::ModulePresentation<double>;

  if (input.relation_grades.size() != input.relation_components.size()) {
    throw std::invalid_argument("Invalid Hera input: relation grades and relation components differ in size.");
  }

  md::PointVec<double> generators;
  generators.reserve(input.generator_grades.size());
  for (const auto& grade : input.generator_grades) {
    generators.emplace_back(grade.first, grade.second);
  }

  typename Module::RelVec relations;
  relations.reserve(input.relation_grades.size());
  for (std::size_t rel_idx = 0; rel_idx < input.relation_grades.size(); ++rel_idx) {
    md::IndexVec components;
    components.reserve(input.relation_components[rel_idx].size());
    for (const auto raw_index : input.relation_components[rel_idx]) {
      if (raw_index < 0 || static_cast<std::size_t>(raw_index) >= input.generator_grades.size()) {
        throw std::invalid_argument("Invalid Hera input: relation component index is out of range.");
      }
      components.push_back(static_cast<md::Index>(raw_index));
    }

    const auto& grade = input.relation_grades[rel_idx];
    relations.emplace_back(md::Point<double>(grade.first, grade.second), components);
  }

  return Module(generators, relations);
}

}  // namespace detail

template <typename index_type>
hera_interface_result hera_matching_distance(const hera_module_presentation_input<index_type>& left,
                                             const hera_module_presentation_input<index_type>& right,
                                             const hera_interface_params& params) {
  if (detail::hera_inputs_equal(left, right)) {
    return hera_interface_result {};
  }

  md::CalculationParams<double> calc_params;
  calc_params.hera_epsilon = params.hera_epsilon;
  calc_params.delta = params.delta;
  calc_params.max_depth = params.max_depth;
  calc_params.initialization_depth = params.initialization_depth;
  calc_params.bound_strategy = detail::hera_bound_strategy_from_int(params.bound_strategy);
  calc_params.traverse_strategy = detail::hera_traverse_strategy_from_int(params.traverse_strategy);
  calc_params.tolerate_max_iter_exceeded = params.tolerate_max_iter_exceeded;
  calc_params.stop_asap = params.stop_asap;

  auto left_module = detail::to_hera_module(left);
  auto right_module = detail::to_hera_module(right);

  hera_interface_result out;
  out.distance = md::matching_distance(left_module, right_module, calc_params);
  out.actual_error = calc_params.actual_error;
  out.actual_max_depth = calc_params.actual_max_depth;
  out.n_hera_calls = calc_params.n_hera_calls;
  return out;
}

inline double hera_bottleneck_distance(const std::vector<std::pair<double, double> >& left,
                                       const std::vector<std::pair<double, double> >& right,
                                       double delta) {
  if (delta < 0.0) {
    throw std::invalid_argument("Hera bottleneck distance expects delta >= 0.");
  }
  auto filtered_left = detail::filter_diagonal_points(left);
  auto filtered_right = detail::filter_diagonal_points(right);
  if (delta == 0.0) {
    return hera::bottleneckDistExact(filtered_left, filtered_right);
  }
  return hera::bottleneckDistApprox(filtered_left, filtered_right, delta);
}

inline double hera_wasserstein_distance(const std::vector<std::pair<double, double> >& left,
                                        const std::vector<std::pair<double, double> >& right,
                                        const hera_wasserstein_params& params) {
  hera::AuctionParams<double> auction_params;
  auction_params.wasserstein_power = params.wasserstein_power;
  auction_params.internal_p = detail::hera_internal_p_from_double(params.internal_p);
  auction_params.delta = params.delta;
  auto filtered_left = detail::filter_diagonal_points(left);
  auto filtered_right = detail::filter_diagonal_points(right);
  return hera::wasserstein_dist(filtered_left, filtered_right, auction_params);
}

#else

template <typename index_type>
hera_interface_result hera_matching_distance(const hera_module_presentation_input<index_type>&,
                                             const hera_module_presentation_input<index_type>&,
                                             const hera_interface_params&) {
  throw std::runtime_error(
      "Hera in-memory interface is not available at compile time. Provide Hera headers and rebuild multipers.");
}

inline double hera_bottleneck_distance(const std::vector<std::pair<double, double> >&, const std::vector<std::pair<double, double> >&, double) {
  throw std::runtime_error(
      "Hera in-memory interface is not available at compile time. Provide Hera headers and rebuild multipers.");
}

inline double hera_wasserstein_distance(const std::vector<std::pair<double, double> >&,
                                        const std::vector<std::pair<double, double> >&,
                                        const hera_wasserstein_params&) {
  throw std::runtime_error(
      "Hera in-memory interface is not available at compile time. Provide Hera headers and rebuild multipers.");
}

#endif

}  // namespace multipers

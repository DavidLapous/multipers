#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ext_interface/hera_interface.hpp"

#if !MULTIPERS_DISABLE_HERA_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace mphera {

template <typename T>
std::vector<T> cast_vector(nb::handle h) {
  return nb::cast<std::vector<T>>(h);
}

template <typename T>
std::vector<std::vector<T>> cast_matrix(nb::handle h) {
  return nb::cast<std::vector<std::vector<T>>>(h);
}

std::vector<std::pair<double, double>> diagram_from_handle(nb::handle h, bool drop_diagonal = false) {
  std::vector<std::pair<double, double>> out;
  for (const auto& row : cast_matrix<double>(h)) {
    if (row.size() != 2) throw std::runtime_error("Hera diagram distances expect arrays of shape (n, 2).");
    if (drop_diagonal && row[0] == row[1]) continue;
    out.emplace_back(row[0], row[1]);
  }
  return out;
}

multipers::hera_module_presentation_input<int> module_input_from_slicer(nb::object slicer) {
#if !MULTIPERS_DISABLE_HERA_INTERFACE
  if (!multipers::nanobind_helpers::is_slicer_object(slicer)) {
    throw std::runtime_error("Input has to be a slicer.");
  }
  return multipers::nanobind_helpers::visit_const_slicer_wrapper(
      slicer, [&]<typename Desc>(const typename Desc::wrapper& wrapper) {
        if (wrapper.truc.get_number_of_parameters() != 2) {
          throw std::runtime_error("Matching distance only supports 2-parameter slicers.");
        }
        if constexpr (Desc::is_kcritical) {
          throw std::runtime_error("Matching distance expects 1-critical minimal-presentation slicers.");
        }

        auto block = multipers::nanobind_helpers::extract_bifiltration_minpres_degree_block(
            wrapper, wrapper.minpres_degree);

        multipers::hera_module_presentation_input<int> out;
        out.generator_grades = std::move(block.row_grades);
        out.relation_grades = std::move(block.relation_grades);
        out.relation_components = multipers::nanobind_helpers::localize_degree_block_relation_boundaries(block);
        return out;
      });
#else
  static_cast<void>(slicer);
  throw std::runtime_error("Hera in-memory interface is disabled at compile time.");
#endif
}

}  // namespace mphera

NB_MODULE(_hera_interface, m) {
  m.def("_is_available", []() { return multipers::hera_interface_available(); });

  m.def(
      "matching_distance",
      [](nb::object left,
         nb::object right,
         double hera_epsilon,
         double delta,
         int max_depth,
         int initialization_depth,
         int bound_strategy,
         int traverse_strategy,
         bool tolerate_max_iter_exceeded,
         bool stop_asap,
         bool return_stats) -> nb::object {
#if MULTIPERS_DISABLE_HERA_INTERFACE
        static_cast<void>(left);
        static_cast<void>(right);
        static_cast<void>(hera_epsilon);
        static_cast<void>(delta);
        static_cast<void>(max_depth);
        static_cast<void>(initialization_depth);
        static_cast<void>(bound_strategy);
        static_cast<void>(traverse_strategy);
        static_cast<void>(tolerate_max_iter_exceeded);
        static_cast<void>(stop_asap);
        static_cast<void>(return_stats);
        throw std::runtime_error("Hera in-memory interface is disabled at compile time.");
#else
        auto left_input = mphera::module_input_from_slicer(left);
        auto right_input = mphera::module_input_from_slicer(right);
        multipers::hera_interface_params params;
        params.hera_epsilon = hera_epsilon;
        params.delta = delta;
        params.max_depth = max_depth;
        params.initialization_depth = initialization_depth;
        params.bound_strategy = bound_strategy;
        params.traverse_strategy = traverse_strategy;
        params.tolerate_max_iter_exceeded = tolerate_max_iter_exceeded;
        params.stop_asap = stop_asap;
        auto result = multipers::hera_matching_distance(left_input, right_input, params);
        if (return_stats) {
          nb::dict stats;
          stats["actual_error"] = result.actual_error;
          stats["actual_max_depth"] = result.actual_max_depth;
          stats["n_hera_calls"] = result.n_hera_calls;
          return nb::cast(nb::make_tuple(result.distance, stats));
        }
        return nb::cast(result.distance);
#endif
      },
      "left"_a,
      "right"_a,
      "hera_epsilon"_a = 0.001,
      "delta"_a = 0.1,
      "max_depth"_a = 8,
      "initialization_depth"_a = 2,
      "bound_strategy"_a = 4,
      "traverse_strategy"_a = 1,
      "tolerate_max_iter_exceeded"_a = false,
      "stop_asap"_a = true,
      "return_stats"_a = false);

  m.def(
      "bottleneck_distance",
      [](nb::handle left, nb::handle right, double delta) {
        return multipers::hera_bottleneck_distance(
            mphera::diagram_from_handle(left), mphera::diagram_from_handle(right), delta);
      },
      "left"_a,
      "right"_a,
      "delta"_a = 0.01);

  m.def(
      "bottleneck_distances",
      [](nb::handle left_diagrams, nb::handle right_diagrams, double delta) {
        auto left_batch = nb::cast<std::vector<nb::object>>(left_diagrams);
        auto right_batch = nb::cast<std::vector<nb::object>>(right_diagrams);
        if (left_batch.size() != right_batch.size()) {
          throw std::runtime_error("Left and right diagram batches must contain the same number of diagrams.");
        }
        std::vector<double> out(left_batch.size());
        for (size_t i = 0; i < left_batch.size(); ++i) {
          out[i] = multipers::hera_bottleneck_distance(mphera::diagram_from_handle(left_batch[i], true),
                                                       mphera::diagram_from_handle(right_batch[i], true),
                                                       delta);
        }
        return out;
      },
      "left_diagrams"_a,
      "right_diagrams"_a,
      "delta"_a = 0.01);

  m.def(
      "wasserstein_distance",
      [](nb::handle left, nb::handle right, double order, double internal_p, double delta) {
        multipers::hera_wasserstein_params params;
        params.wasserstein_power = order;
        params.internal_p = internal_p;
        params.delta = delta;
        return multipers::hera_wasserstein_distance(
            mphera::diagram_from_handle(left), mphera::diagram_from_handle(right), params);
      },
      "left"_a,
      "right"_a,
      "order"_a = 1.0,
      "internal_p"_a = std::numeric_limits<double>::infinity(),
      "delta"_a = 0.01);
}

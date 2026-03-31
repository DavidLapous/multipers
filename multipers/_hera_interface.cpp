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
#include "ext_interface/nanobind_registry_helpers.hpp"

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

        const auto dimensions = wrapper.truc.get_dimensions();
        const auto& filtrations = wrapper.truc.get_filtration_values();
        const auto& boundaries = wrapper.truc.get_boundaries();
        int degree = wrapper.minpres_degree;

        multipers::hera_module_presentation_input<int> out;
        size_t row_start = std::lower_bound(dimensions.begin(), dimensions.end(), degree) - dimensions.begin();
        size_t row_end = std::lower_bound(dimensions.begin(), dimensions.end(), degree + 1) - dimensions.begin();
        size_t col_end = std::lower_bound(dimensions.begin(), dimensions.end(), degree + 2) - dimensions.begin();

        out.generator_grades.reserve(row_end - row_start);
        for (size_t i = row_start; i < row_end; ++i) {
          out.generator_grades.emplace_back(filtrations[i](0, 0), filtrations[i](0, 1));
        }
        out.relation_grades.reserve(col_end - row_end);
        out.relation_components.resize(col_end - row_end);
        for (size_t i = row_end; i < col_end; ++i) {
          out.relation_grades.emplace_back(filtrations[i](0, 0), filtrations[i](0, 1));
          for (auto boundary_index : boundaries[i]) {
            if (boundary_index < static_cast<int>(row_start) || boundary_index >= static_cast<int>(row_end)) {
              throw std::runtime_error(
                  "Invalid minimal presentation slicer: relation boundaries must reference degree-d generators only.");
            }
            out.relation_components[i - row_end].push_back(boundary_index - static_cast<int>(row_start));
          }
        }
        return out;
      });
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

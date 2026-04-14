#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(GUDHI_USE_TBB)
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#include "hera_monte_carlo_core.hpp"
#include "ext_interface/hera_interface.hpp"

#if !MULTIPERS_DISABLE_HERA_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace mphera {

using diagram_t = std::vector<std::pair<double, double>>;

struct monte_carlo_slicer_metadata {
  bool is_kcritical = false;
  bool is_squeezed = false;
  std::size_t num_parameters = 0;
};

template <typename T>
std::vector<T> cast_vector(nb::handle h) {
  return nb::cast<std::vector<T>>(h);
}

template <typename T>
std::vector<std::vector<T>> cast_matrix(nb::handle h) {
  return nb::cast<std::vector<std::vector<T>>>(h);
}

[[noreturn]] inline void throw_disabled_hera_interface() {
  throw std::runtime_error("Hera interface is disabled at compile time.");
}

diagram_t diagram_from_handle(nb::handle h, bool drop_diagonal = false) {
  diagram_t out;
  for (const auto& row : cast_matrix<double>(h)) {
    if (row.size() != 2) throw std::runtime_error("Hera diagram distances expect arrays of shape (n, 2).");
    if (drop_diagonal && row[0] == row[1]) continue;
    out.emplace_back(row[0], row[1]);
  }
  return out;
}

std::vector<double> bottleneck_distances_from_native_batches(const std::vector<diagram_t>& left_native,
                                                             const std::vector<diagram_t>& right_native,
                                                             double delta,
                                                             int n_jobs) {
  std::vector<double> out(left_native.size());
  {
    nb::gil_scoped_release release;
#if defined(GUDHI_USE_TBB)
    const int max_concurrency = n_jobs <= 0 ? tbb::task_arena::automatic : n_jobs;
    tbb::task_arena arena(max_concurrency);
    arena.execute([&] {
      tbb::parallel_for(size_t(0), left_native.size(), [&](size_t i) {
        out[i] = multipers::hera_bottleneck_distance(left_native[i], right_native[i], delta);
      });
    });
#else
    static_cast<void>(n_jobs);
    for (size_t i = 0; i < left_native.size(); ++i) {
      out[i] = multipers::hera_bottleneck_distance(left_native[i], right_native[i], delta);
    }
#endif
  }
  return out;
}

#if !MULTIPERS_DISABLE_HERA_INTERFACE
template <bool IsKcritical>
using native_f64_slicer_t = std::conditional_t<IsKcritical, multipers::kcontiguous_f64_slicer, multipers::contiguous_f64_slicer>;

template <bool IsKcritical, typename Func>
decltype(auto) with_native_f64_slicer(const nb::handle& input, Func&& func) {
  using NativeSlicer = native_f64_slicer_t<IsKcritical>;
  return multipers::nanobind_helpers::visit_const_slicer_wrapper(
      input, [&]<typename Desc>(const typename Desc::wrapper& wrapper) -> decltype(auto) {
        // Monte Carlo fast path always runs on non-vine float64 matrix slicers.
        if constexpr (std::is_same_v<typename Desc::concrete, NativeSlicer>) {
          return std::forward<Func>(func)(wrapper.truc);
        } else {
          NativeSlicer copy(wrapper.truc);
          return std::forward<Func>(func)(copy);
        }
      });
}
#endif

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
  throw std::runtime_error("Hera interface is disabled at compile time.");
#endif
}

#if !MULTIPERS_DISABLE_HERA_INTERFACE

inline monte_carlo_slicer_metadata metadata_from_slicer(nb::handle input) {
  return multipers::nanobind_helpers::visit_const_slicer_wrapper(
      input, [&]<typename Desc>(const typename Desc::wrapper& wrapper) {
        monte_carlo_slicer_metadata out;
        out.is_kcritical = Desc::is_kcritical;
        out.is_squeezed = multipers::nanobind_helpers::has_nonempty_filtration_grid(wrapper.filtration_grid);
        out.num_parameters = static_cast<std::size_t>(wrapper.truc.get_number_of_parameters());
        return out;
      });
}

inline void validate_monte_carlo_native_inputs(
    nb::handle left,
    nb::handle right,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& basepoints,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& directions,
    int degree,
    double delta) {
  if (!multipers::nanobind_helpers::is_slicer_object(left) ||
      !multipers::nanobind_helpers::is_slicer_object(right)) {
    throw std::runtime_error("Input has to be a slicer.");
  }
  if (degree < 0) {
    throw std::runtime_error("Monte Carlo matching distance expects degree >= 0.");
  }
  if (delta < 0.0) {
    throw std::runtime_error("Hera bottleneck distance expects delta >= 0.");
  }
  if (basepoints.shape(0) != directions.shape(0)) {
    throw std::runtime_error("Basepoint and direction batches must contain the same number of lines.");
  }
  if (basepoints.shape(1) != directions.shape(1)) {
    throw std::runtime_error("Basepoints and directions must have the same line dimension.");
  }
}

template <typename LeftSlicer, typename RightSlicer>
nb::object run_monte_carlo_on_native_slicers(
    const LeftSlicer& left,
    const RightSlicer& right,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& basepoints,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& directions,
    int degree,
    double delta,
    int n_jobs) {
  auto raw_distances = [&] {
    nb::gil_scoped_release release;
    return multipers::core::monte_carlo_bottleneck_distances_on_lines(left,
                                                                       right,
                                                                       basepoints.data(),
                                                                       directions.data(),
                                                                       static_cast<std::size_t>(basepoints.shape(0)),
                                                                       static_cast<std::size_t>(basepoints.shape(1)),
                                                                       degree,
                                                                       delta,
                                                                       true,
                                                                       n_jobs);
  }();
  return nb::cast(raw_distances);
}

template <bool LeftKcritical, bool RightKcritical>
nb::object run_monte_carlo_on_native_slicer_types(
    nb::handle left,
    nb::handle right,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& basepoints,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& directions,
    int degree,
    double delta,
    int n_jobs) {
  using LeftSlicer = native_f64_slicer_t<LeftKcritical>;
  using RightSlicer = native_f64_slicer_t<RightKcritical>;

  return with_native_f64_slicer<LeftKcritical>(left, [&](const LeftSlicer& left_native) -> nb::object {
    return with_native_f64_slicer<RightKcritical>(
        right, [&](const RightSlicer& right_native) -> nb::object {
          return run_monte_carlo_on_native_slicers(left_native, right_native, basepoints, directions, degree, delta, n_jobs);
        });
  });
}

inline nb::object monte_carlo_bottleneck_distances_on_lines(
    nb::handle left,
    nb::handle right,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& basepoints,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& directions,
    int degree,
    double delta,
    int n_jobs) {
  validate_monte_carlo_native_inputs(left, right, basepoints, directions, degree, delta);
  const auto left_meta = metadata_from_slicer(left);
  const auto right_meta = metadata_from_slicer(right);
  if (left_meta.is_squeezed || right_meta.is_squeezed) {
    throw std::runtime_error("Native Monte Carlo matching-distance fast path does not support squeezed slicers.");
  }
  if (left_meta.num_parameters != right_meta.num_parameters) {
    throw std::runtime_error("Matching distance expects slicers with the same number of parameters.");
  }
  if (left_meta.num_parameters != static_cast<std::size_t>(basepoints.shape(1))) {
    throw std::runtime_error("Basepoints and directions must match the slicer parameter dimension.");
  }

  if (left_meta.is_kcritical) {
    if (right_meta.is_kcritical) {
      return run_monte_carlo_on_native_slicer_types<true, true>(left, right, basepoints, directions, degree, delta, n_jobs);
    }
    return run_monte_carlo_on_native_slicer_types<true, false>(left, right, basepoints, directions, degree, delta, n_jobs);
  }
  if (right_meta.is_kcritical) {
    return run_monte_carlo_on_native_slicer_types<false, true>(left, right, basepoints, directions, degree, delta, n_jobs);
  }
  return run_monte_carlo_on_native_slicer_types<false, false>(left, right, basepoints, directions, degree, delta, n_jobs);
}

#endif

inline nb::object matching_distance_binding(nb::object left,
                                            nb::object right,
                                            double hera_epsilon,
                                            double delta,
                                            int max_depth,
                                            int initialization_depth,
                                            int bound_strategy,
                                            int traverse_strategy,
                                            bool tolerate_max_iter_exceeded,
                                            bool stop_asap,
                                            bool return_stats) {
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
  throw_disabled_hera_interface();
#else
  auto left_input = module_input_from_slicer(left);
  auto right_input = module_input_from_slicer(right);
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
}

inline nb::object monte_carlo_bottleneck_distances_binding(
    nb::handle left,
    nb::handle right,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& basepoints,
    const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& directions,
    int degree,
    double delta,
    int n_jobs) {
#if MULTIPERS_DISABLE_HERA_INTERFACE
  static_cast<void>(left);
  static_cast<void>(right);
  static_cast<void>(basepoints);
  static_cast<void>(directions);
  static_cast<void>(degree);
  static_cast<void>(delta);
  static_cast<void>(n_jobs);
  throw_disabled_hera_interface();
#else
  return monte_carlo_bottleneck_distances_on_lines(left, right, basepoints, directions, degree, delta, n_jobs);
#endif
}

}  // namespace mphera

NB_MODULE(_hera_interface, m) {
  auto available = []() { return multipers::hera_interface_available(); };
  m.def("_is_available", available);
  m.def("available", available);
  m.def("require", [available]() {
    if (!available()) {
      throw std::runtime_error(
          "Hera interface is not available in this build. Rebuild multipers with Hera headers to enable this backend.");
    }
  });

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
        return mphera::matching_distance_binding(left,
                                                 right,
                                                 hera_epsilon,
                                                 delta,
                                                 max_depth,
                                                 initialization_depth,
                                                 bound_strategy,
                                                 traverse_strategy,
                                                 tolerate_max_iter_exceeded,
                                                 stop_asap,
                                                 return_stats);
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
      [](nb::handle left_diagrams, nb::handle right_diagrams, double delta, int n_jobs) {
        auto left_batch = nb::cast<std::vector<nb::object>>(left_diagrams);
        auto right_batch = nb::cast<std::vector<nb::object>>(right_diagrams);
        if (left_batch.size() != right_batch.size()) {
          throw std::runtime_error("Left and right diagram batches must contain the same number of diagrams.");
        }

        std::vector<mphera::diagram_t> left_native(left_batch.size());
        std::vector<mphera::diagram_t> right_native(right_batch.size());
        for (size_t i = 0; i < left_batch.size(); ++i) {
          left_native[i] = mphera::diagram_from_handle(left_batch[i], true);
          right_native[i] = mphera::diagram_from_handle(right_batch[i], true);
        }
        return mphera::bottleneck_distances_from_native_batches(left_native, right_native, delta, n_jobs);
      },
      "left_diagrams"_a,
      "right_diagrams"_a,
      "delta"_a = 0.01,
      "n_jobs"_a = 0);

  m.def(
      "monte_carlo_bottleneck_distances",
      [](nb::handle left,
         nb::handle right,
         const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& basepoints,
         const nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig>& directions,
         int degree,
         double delta,
         int n_jobs) -> nb::object {
        return mphera::monte_carlo_bottleneck_distances_binding(left, right, basepoints, directions, degree, delta, n_jobs);
      },
      "left"_a,
      "right"_a,
      "basepoints"_a,
      "directions"_a,
      "degree"_a,
      "delta"_a = 0.01,
      "n_jobs"_a = 0);

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

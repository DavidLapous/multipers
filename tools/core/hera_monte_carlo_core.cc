#define MULTIPERS_BUILD_CORE_TEMPLATES 1

#include <hera_monte_carlo_core.hpp>

#if defined(GUDHI_USE_TBB)
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#include <tuple>
#include <utility>
#include <vector>

#include <ext_interface/hera_interface.hpp>

namespace multipers::core {

namespace {

template <typename LeftSlicer, typename RightSlicer, typename DistanceFn>
std::vector<double> run_monte_carlo_line_distances_on_lines_impl(
    const LeftSlicer& left,
    const RightSlicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    bool ignore_infinite_filtration_values,
    int n_jobs,
    DistanceFn&& distance_fn) {
  if (num_lines == 0) return {};

  auto dgm_on_line = [ignore_infinite_filtration_values, degree](auto& slicer,
                                                                 const auto& line,
                                                                 std::vector<std::pair<double, double> >& diagram) {
    slicer.push_to(line);
    if (!slicer.persistence_computation_is_initialized()) [[unlikely]] {
      slicer.initialize_persistence_computation(ignore_infinite_filtration_values);
    } else {
      slicer.update_persistence_computation(ignore_infinite_filtration_values);
    }
    auto bars = slicer.template get_barcode<false, double, false>(degree);
    diagram.clear();
    if (diagram.capacity() < bars.size()) diagram.reserve(bars.size());
    for (const auto& bar : bars) {
      if (bar.dim == degree) {
        diagram.emplace_back(bar.birth, bar.death);
      }
    }
  };

#if defined(GUDHI_USE_TBB)
  using LeftThreadSafe = typename LeftSlicer::Thread_safe;
  using RightThreadSafe = typename RightSlicer::Thread_safe;
  using Line = Gudhi::multi_persistence::Line<double>;
  using Point = typename Line::Point_t;
  using Diagram = std::vector<std::pair<double, double> >;
  using ThreadLocalState = std::tuple<LeftThreadSafe, RightThreadSafe, Line, Diagram, Diagram>;
  tbb::enumerable_thread_specific<ThreadLocalState> thread_locals(ThreadLocalState(
      left.weak_copy(), right.weak_copy(), Line(Point(num_parameters), Point(num_parameters, 1.0)), Diagram(), Diagram()));
  std::vector<double> out(num_lines);
  const int max_concurrency = n_jobs <= 0 ? tbb::task_arena::automatic : n_jobs;
  tbb::task_arena arena(max_concurrency);
  arena.execute([&] {
    tbb::parallel_for(std::size_t(0), num_lines, [&](std::size_t i) {
      auto& thread_local_state = thread_locals.local();
      auto& left_local = std::get<0>(thread_local_state);
      auto& right_local = std::get<1>(thread_local_state);
      auto& line = std::get<2>(thread_local_state);
      auto& left_diagram = std::get<3>(thread_local_state);
      auto& right_diagram = std::get<4>(thread_local_state);
      tbb::this_task_arena::isolate([&] {
        const double* basepoint = basepoints + i * num_parameters;
        const double* direction = directions + i * num_parameters;
        bool is_trivial = true;
        for (std::size_t j = 0; j < num_parameters; ++j) {
          const double direction_value = direction[j];
          if (direction_value) is_trivial = false;
          if (direction_value <= 0.0) {
            throw std::invalid_argument("Direction should have strictly positive entries.");
          }
          line.base_point()[j] = basepoint[j];
          line.direction()[j] = direction_value;
        }
        if (num_parameters != 0 && is_trivial) {
          throw std::invalid_argument("Direction should not be trivial.");
        }
        dgm_on_line(left_local, line, left_diagram);
        dgm_on_line(right_local, line, right_diagram);
        out[i] = distance_fn(left_diagram, right_diagram);
      });
    });
  });
  return out;
#else
  static_cast<void>(n_jobs);
  using Line = Gudhi::multi_persistence::Line<double>;
  using Point = typename Line::Point_t;
  using Diagram = std::vector<std::pair<double, double> >;
  auto left_local = left.weak_copy();
  auto right_local = right.weak_copy();
  Line line(Point(num_parameters), Point(num_parameters, 1.0));
  Diagram left_diagram;
  Diagram right_diagram;
  std::vector<double> out(num_lines);

  for (std::size_t i = 0; i < num_lines; ++i) {
    const double* basepoint = basepoints + i * num_parameters;
    const double* direction = directions + i * num_parameters;
    bool is_trivial = true;
    for (std::size_t j = 0; j < num_parameters; ++j) {
      const double direction_value = direction[j];
      if (direction_value) is_trivial = false;
      if (direction_value <= 0.0) {
        throw std::invalid_argument("Direction should have strictly positive entries.");
      }
      line.base_point()[j] = basepoint[j];
      line.direction()[j] = direction_value;
    }
    if (num_parameters != 0 && is_trivial) {
      throw std::invalid_argument("Direction should not be trivial.");
    }
    dgm_on_line(left_local, line, left_diagram);
    dgm_on_line(right_local, line, right_diagram);
    out[i] = distance_fn(left_diagram, right_diagram);
  }
  return out;
#endif
}

}  // namespace

#define MP_HERA_MC_SLICER_PAIRS(X)                  \
  X(contiguous_f64_slicer, contiguous_f64_slicer)   \
  X(contiguous_f64_slicer, kcontiguous_f64_slicer)  \
  X(kcontiguous_f64_slicer, contiguous_f64_slicer)  \
  X(kcontiguous_f64_slicer, kcontiguous_f64_slicer)

#define MP_DEFINE_HERA_MC_OVERLOADS(LeftSlicer, RightSlicer)                                          \
  std::vector<double> monte_carlo_bottleneck_distances_on_lines(                                      \
      const LeftSlicer& left,                                                                          \
      const RightSlicer& right,                                                                        \
      const double* basepoints,                                                                        \
      const double* directions,                                                                        \
      std::size_t num_lines,                                                                           \
      std::size_t num_parameters,                                                                      \
      int degree,                                                                                      \
      double delta,                                                                                    \
      bool ignore_infinite_filtration_values,                                                          \
      int n_jobs) {                                                                                    \
    return run_monte_carlo_line_distances_on_lines_impl(                                               \
        left,                                                                                          \
        right,                                                                                         \
        basepoints,                                                                                    \
        directions,                                                                                    \
        num_lines,                                                                                     \
        num_parameters,                                                                                \
        degree,                                                                                        \
        ignore_infinite_filtration_values,                                                             \
        n_jobs,                                                                                        \
        [delta](const auto& left_diagram, const auto& right_diagram) {                                \
          return multipers::hera_bottleneck_distance(left_diagram, right_diagram, delta);             \
        });                                                                                            \
  }                                                                                                    \
  std::vector<double> monte_carlo_wasserstein_distances_on_lines(                                     \
      const LeftSlicer& left,                                                                          \
      const RightSlicer& right,                                                                        \
      const double* basepoints,                                                                        \
      const double* directions,                                                                        \
      std::size_t num_lines,                                                                           \
      std::size_t num_parameters,                                                                      \
      int degree,                                                                                      \
      double order,                                                                                    \
      double internal_p,                                                                               \
      double delta,                                                                                    \
      bool ignore_infinite_filtration_values,                                                          \
      int n_jobs) {                                                                                    \
    multipers::hera_wasserstein_params params;                                                         \
    params.wasserstein_power = order;                                                                  \
    params.internal_p = internal_p;                                                                    \
    params.delta = delta;                                                                              \
    return run_monte_carlo_line_distances_on_lines_impl(                                               \
        left,                                                                                          \
        right,                                                                                         \
        basepoints,                                                                                    \
        directions,                                                                                    \
        num_lines,                                                                                     \
        num_parameters,                                                                                \
        degree,                                                                                        \
        ignore_infinite_filtration_values,                                                             \
        n_jobs,                                                                                        \
        [params](const auto& left_diagram, const auto& right_diagram) {                               \
          return multipers::hera_wasserstein_distance(left_diagram, right_diagram, params);           \
        });                                                                                            \
  }

MP_HERA_MC_SLICER_PAIRS(MP_DEFINE_HERA_MC_OVERLOADS)

#undef MP_DEFINE_HERA_MC_OVERLOADS
#undef MP_HERA_MC_SLICER_PAIRS

void hera_monte_carlo_core_anchor() {}

}  // namespace multipers::core

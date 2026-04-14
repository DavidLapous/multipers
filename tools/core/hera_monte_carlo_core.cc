#define MULTIPERS_BUILD_CORE_TEMPLATES 1

#include <hera_monte_carlo_core.hpp>

#if defined(GUDHI_USE_TBB)
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#include <utility>
#include <vector>

#include <ext_interface/hera_interface.hpp>

namespace multipers::core {

namespace {

template <typename SlicerLike>
void advance_slicer_state_for_line(SlicerLike& slicer,
                                   const double* basepoint,
                                   const double* direction,
                                   std::size_t num_parameters,
                                   bool ignore_infinite_filtration_values,
                                   std::vector<double>& bp,
                                   std::vector<double>& dir) {
  bp.resize(num_parameters);
  dir.resize(num_parameters);
  for (std::size_t i = 0; i < num_parameters; ++i) {
    bp[i] = basepoint[i];
    dir[i] = direction[i];
  }
  slicer.push_to(Gudhi::multi_persistence::Line<double>(bp, dir));
  slicer.initialize_persistence_computation(ignore_infinite_filtration_values);
}

template <typename SlicerLike>
std::vector<std::pair<double, double> > degree_diagram_from_current_state(SlicerLike& slicer, int degree) {
  auto barcodes = slicer.template get_flat_barcode<true, double, false>(degree);
  const auto& degree_bars = barcodes[degree];
  std::vector<std::pair<double, double> > out;
  out.reserve(degree_bars.size());
  for (const auto& bar : degree_bars) {
    out.emplace_back(bar[0], bar[1]);
  }
  return out;
}

template <typename LeftSlicer, typename RightSlicer>
std::vector<double> run_monte_carlo_bottleneck_distances_on_lines_impl(
    const LeftSlicer& left,
    const RightSlicer& right,
    const double* basepoints,
    const double* directions,
    std::size_t num_lines,
    std::size_t num_parameters,
    int degree,
    double delta,
    bool ignore_infinite_filtration_values,
    int n_jobs) {
  if (num_lines == 0) return {};

#if defined(GUDHI_USE_TBB)
  using LeftThreadSafe = typename LeftSlicer::Thread_safe;
  using RightThreadSafe = typename RightSlicer::Thread_safe;
  tbb::enumerable_thread_specific<std::pair<LeftThreadSafe, RightThreadSafe> > thread_locals(
      std::make_pair(left.weak_copy(), right.weak_copy()));
  std::vector<double> out(num_lines);
  const int max_concurrency = n_jobs <= 0 ? tbb::task_arena::automatic : n_jobs;
  tbb::task_arena arena(max_concurrency);
  arena.execute([&] {
    tbb::parallel_for(std::size_t(0), num_lines, [&](std::size_t i) {
      auto& thread_local_state = thread_locals.local();
      auto& left_local = thread_local_state.first;
      auto& right_local = thread_local_state.second;
      tbb::this_task_arena::isolate([&] {
        std::vector<double> left_bp(num_parameters);
        std::vector<double> left_dir(num_parameters);
        std::vector<double> right_bp(num_parameters);
        std::vector<double> right_dir(num_parameters);
        const double* basepoint = basepoints + i * num_parameters;
        const double* direction = directions + i * num_parameters;
        advance_slicer_state_for_line(
            left_local, basepoint, direction, num_parameters, ignore_infinite_filtration_values, left_bp, left_dir);
        advance_slicer_state_for_line(
            right_local, basepoint, direction, num_parameters, ignore_infinite_filtration_values, right_bp, right_dir);
        out[i] = multipers::hera_bottleneck_distance(
            degree_diagram_from_current_state(left_local, degree),
            degree_diagram_from_current_state(right_local, degree),
            delta);
      });
    });
  });
  return out;
#else
  static_cast<void>(n_jobs);
  auto left_local = left.weak_copy();
  auto right_local = right.weak_copy();
  std::vector<double> left_bp(num_parameters);
  std::vector<double> left_dir(num_parameters);
  std::vector<double> right_bp(num_parameters);
  std::vector<double> right_dir(num_parameters);
  std::vector<double> out(num_lines);

  for (std::size_t i = 0; i < num_lines; ++i) {
    const double* basepoint = basepoints + i * num_parameters;
    const double* direction = directions + i * num_parameters;
    advance_slicer_state_for_line(
        left_local, basepoint, direction, num_parameters, ignore_infinite_filtration_values, left_bp, left_dir);
    advance_slicer_state_for_line(
        right_local, basepoint, direction, num_parameters, ignore_infinite_filtration_values, right_bp, right_dir);
    out[i] = multipers::hera_bottleneck_distance(
        degree_diagram_from_current_state(left_local, degree),
        degree_diagram_from_current_state(right_local, degree),
        delta);
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

#define MP_DEFINE_HERA_MC_OVERLOAD(LeftSlicer, RightSlicer)                                           \
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
    return run_monte_carlo_bottleneck_distances_on_lines_impl(                                         \
        left, right, basepoints, directions, num_lines, num_parameters, degree, delta,                \
        ignore_infinite_filtration_values, n_jobs);                                                    \
  }

MP_HERA_MC_SLICER_PAIRS(MP_DEFINE_HERA_MC_OVERLOAD)

#undef MP_DEFINE_HERA_MC_OVERLOAD
#undef MP_HERA_MC_SLICER_PAIRS

void hera_monte_carlo_core_anchor() {}

}  // namespace multipers::core

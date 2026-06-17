#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "ext_interface/deg_rips_interface.hpp"

#if !MULTIPERS_DISABLE_DEG_RIPS_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace mpdr {

#if !MULTIPERS_DISABLE_DEG_RIPS_INTERFACE

using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::visit_simplextree_wrapper;

std::vector<int> degrees_from_array(nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> degrees) {
  std::vector<int> out;
  out.reserve(degrees.shape(0));
  for (size_t i = 0; i < degrees.shape(0); ++i) {
    const int degree = static_cast<int>(degrees(i));
    if (degree < 0) {
      throw std::invalid_argument("deg_rips degrees must be nonnegative.");
    }
    if (!out.empty() && degree < out.back()) {
      throw std::invalid_argument("deg_rips degrees must be sorted in nondecreasing order.");
    }
    out.push_back(degree);
  }
  if (out.empty()) {
    throw std::invalid_argument("deg_rips degrees must contain at least one value.");
  }
  return out;
}

std::optional<std::vector<int> > degrees_from_object(nb::object degrees) {
  if (degrees.is_none()) {
    return std::nullopt;
  }
  return degrees_from_array(nb::cast<nb::ndarray<nb::numpy, const int32_t, nb::ndim<1>, nb::c_contig> >(degrees));
}

nb::dict stats_to_dict(const multipers::deg_rips_stats& stats) {
  nb::dict out;
  out["input_vertices"] = nb::cast(stats.input_vertices);
  out["output_vertices"] = nb::cast(stats.output_vertices);
  out["inserted_vertices"] = nb::cast(stats.inserted_vertices);
  out["inserted_edges"] = nb::cast(stats.inserted_edges);
  out["eliminated_vertices"] = nb::cast(stats.eliminated_vertices);
  out["edge_copy_count"] = nb::cast(stats.edge_copy_count);
  return out;
}

struct degree_rips_call_result {
  nb::object target;
  multipers::deg_rips_stats stats;
};

degree_rips_call_result degree_rips_to_simplextree_impl(
    nb::object target,
    nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> distance_matrix,
    nb::object degrees,
    double threshold_radius,
    bool vanilla,
    bool vertex_domination,
    int whole_edge_iterations,
    int edge_copy_iterations,
    bool use_domination_for_whole_edge_removal,
    bool use_domination_for_edge_copy_removal,
    double min_scale,
    bool verbose) {
  if (!is_simplextree_object(target)) {
    throw nb::type_error("degree_rips_to_simplextree expects a SimplexTreeMulti target.");
  }
  if (distance_matrix.shape(0) != distance_matrix.shape(1)) {
    throw std::invalid_argument("deg_rips distance_matrix must be square.");
  }

  auto requested_degrees = degrees_from_object(degrees);
  multipers::deg_rips_options options;
  options.max_scale = threshold_radius;
  options.min_scale = min_scale;
  options.verbose_output = verbose;
  if (!vanilla) {
    options.with_vertex_domination = vertex_domination;
    options.whole_edge_iterations = whole_edge_iterations;
    options.edge_copy_iterations = edge_copy_iterations;
    options.use_domination_for_whole_edge_removal = use_domination_for_whole_edge_removal;
    options.use_domination_for_edge_copy_removal = use_domination_for_edge_copy_removal;
  }

  multipers::deg_rips_stats stats;
  const double* distance_matrix_data = distance_matrix.data();
  const size_t num_points = distance_matrix.shape(0);
  visit_simplextree_wrapper(target, [&]<typename Desc>(auto& wrapper) {
    nb::gil_scoped_release release;
    stats = multipers::degree_rips_build_simplextree<Desc>(
        wrapper, distance_matrix_data, num_points, options, requested_degrees);
  });
  return {target, stats};
}

#endif

}  // namespace mpdr

NB_MODULE(_deg_rips_interface, m) {
  auto available = []() { return multipers::deg_rips_interface_available(); };
  m.def("_is_available", available);
  m.def("available", available);
  m.def("require", [available]() {
    if (!available()) {
      throw std::runtime_error(
          "deg_rips interface is not available in this build. Rebuild multipers with deg_rips support to enable this "
          "backend.");
    }
  });

  auto build = [](nb::object target,
                  nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> distance_matrix,
                  nb::object degrees,
                  double threshold_radius,
                  bool vanilla,
                  bool vertex_domination,
                  int whole_edge_iterations,
                  int edge_copy_iterations,
                  bool use_domination_for_whole_edge_removal,
                  bool use_domination_for_edge_copy_removal,
                  double min_scale,
                  bool verbose) -> nb::object {
#if MULTIPERS_DISABLE_DEG_RIPS_INTERFACE
    throw std::runtime_error("deg_rips interface is disabled at compile time.");
#else
    return mpdr::degree_rips_to_simplextree_impl(target,
                                                 distance_matrix,
                                                 degrees,
                                                 threshold_radius,
                                                 vanilla,
                                                 vertex_domination,
                                                 whole_edge_iterations,
                                                 edge_copy_iterations,
                                                 use_domination_for_whole_edge_removal,
                                                 use_domination_for_edge_copy_removal,
                                                 min_scale,
                                                 verbose)
        .target;
#endif
  };

  m.def("degree_rips_to_simplextree",
        build,
        "target"_a,
        "distance_matrix"_a,
        "degrees"_a.none(),
        "threshold_radius"_a,
        "vanilla"_a = true,
        "vertex_domination"_a = true,
        "whole_edge_iterations"_a = 1,
        "edge_copy_iterations"_a = 1,
        "use_domination_for_whole_edge_removal"_a = true,
        "use_domination_for_edge_copy_removal"_a = false,
        "min_scale"_a = 0.0,
        "verbose"_a = false);

  m.def(
      "degree_rips_to_simplextree_with_stats",
      [](nb::object target,
         nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig> distance_matrix,
         nb::object degrees,
         double threshold_radius,
         bool vanilla,
         bool vertex_domination,
         int whole_edge_iterations,
         int edge_copy_iterations,
         bool use_domination_for_whole_edge_removal,
         bool use_domination_for_edge_copy_removal,
         double min_scale,
         bool verbose) -> nb::tuple {
#if MULTIPERS_DISABLE_DEG_RIPS_INTERFACE
        throw std::runtime_error("deg_rips interface is disabled at compile time.");
#else
        auto result = mpdr::degree_rips_to_simplextree_impl(target,
                                                            distance_matrix,
                                                            degrees,
                                                            threshold_radius,
                                                            vanilla,
                                                            vertex_domination,
                                                            whole_edge_iterations,
                                                            edge_copy_iterations,
                                                            use_domination_for_whole_edge_removal,
                                                            use_domination_for_edge_copy_removal,
                                                            min_scale,
                                                            verbose);
        return nb::make_tuple(result.target, mpdr::stats_to_dict(result.stats));
#endif
      },
      "target"_a,
      "distance_matrix"_a,
      "degrees"_a.none(),
      "threshold_radius"_a,
      "vanilla"_a = true,
      "vertex_domination"_a = true,
      "whole_edge_iterations"_a = 1,
      "edge_copy_iterations"_a = 1,
      "use_domination_for_whole_edge_removal"_a = true,
      "use_domination_for_edge_copy_removal"_a = false,
      "min_scale"_a = 0.0,
      "verbose"_a = false);
}

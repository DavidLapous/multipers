#include <nanobind/nanobind.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ext_interface/graphcode_interface.hpp"
#include "nanobind_array_utils.hpp"

#if !MULTIPERS_DISABLE_GRAPHCODE_INTERFACE
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_GRAPHCODE_INTERFACE
namespace mpgc {

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;

inline multipers::graphcode_interface_input input_from_slicer(nb::object slicer) {
  nb::object target = multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
  const auto& wrapper = nb::cast<const CanonicalWrapper&>(target);
  const int degree = multipers::nanobind_helpers::slicer_minpres_degree(wrapper);
  if (degree < 0) {
    throw std::runtime_error("graphcode expects a minimal-presentation slicer.");
  }
  if (wrapper.truc.get_number_of_parameters() != 2) {
    throw std::runtime_error("graphcode expects a 2-parameter minimal-presentation slicer.");
  }

  auto block = multipers::nanobind_helpers::extract_bifiltration_minpres_degree_block(wrapper, degree);
  multipers::graphcode_interface_input input;
  input.row_grades = std::move(block.row_grades);
  input.relation_grades = std::move(block.relation_grades);
  input.relation_boundaries = multipers::nanobind_helpers::localize_degree_block_relation_boundaries(block);
  return input;
}

inline nb::dict output_to_dict(multipers::graphcode_interface_output&& output) {
  std::vector<double> vertices;
  vertices.reserve(output.vertices.size() * 4);
  for (const auto& vertex : output.vertices) {
    vertices.insert(vertices.end(),
                    {vertex.birth, vertex.death, static_cast<double>(vertex.slice_index), vertex.slice_value});
  }

  std::vector<std::int64_t> edges;
  edges.reserve(output.edges.size() * 2);
  for (const auto& edge : output.edges) {
    edges.insert(edges.end(), {edge.first, edge.second});
  }

  const auto num_vertices = output.vertices.size();
  const auto num_edges = output.edges.size();
  const auto num_slice_values = output.slice_values.size();
  nb::dict out;
  out["vertices"] = multipers::nanobind_utils::owned_array<double>(std::move(vertices), {num_vertices, size_t(4)});
  out["edges"] = multipers::nanobind_utils::owned_array<std::int64_t>(std::move(edges), {num_edges, size_t(2)});
  out["slice_values"] =
      multipers::nanobind_utils::owned_array<double>(std::move(output.slice_values), {num_slice_values});
  return out;
}

}  // namespace mpgc
#endif

NB_MODULE(_graphcode_interface, m) {
  auto available = []() { return multipers::graphcode_interface_available(); };
  m.def("_is_available", available);
  m.def("available", available);
  m.def("require", [available]() {
    if (!available()) {
      throw std::runtime_error(
          "graphcode interface is not available in this build. Rebuild multipers with graphcode support to enable this "
          "backend.");
    }
  });

  m.def(
      "graphcode",
      [](nb::object slicer,
         int primary_parameter,
         int slices,
         bool include_infinite_bars,
         bool filter_out_disjoint_pairs,
         bool do_exhaustive_reduction,
         double relevance_threshold,
         double secondary_threshold,
         bool compressed_graphcode,
         bool double_edges) {
#if MULTIPERS_DISABLE_GRAPHCODE_INTERFACE
        throw std::runtime_error("graphcode interface is disabled at compile time.");
#else
        if (!multipers::graphcode_interface_available()) {
          throw std::runtime_error("graphcode interface is not available.");
        }
        auto input = mpgc::input_from_slicer(slicer);
        multipers::graphcode_interface_output output;
        {
          nb::gil_scoped_release release;
          output = multipers::graphcode_interface(input,
                                                  primary_parameter,
                                                  slices,
                                                  include_infinite_bars,
                                                  filter_out_disjoint_pairs,
                                                  do_exhaustive_reduction,
                                                  relevance_threshold,
                                                  secondary_threshold,
                                                  compressed_graphcode,
                                                  double_edges);
        }
        return mpgc::output_to_dict(std::move(output));
#endif
      },
      "slicer"_a,
      "primary_parameter"_a = 1,
      "slices"_a = 0,
      "include_infinite_bars"_a = true,
      "filter_out_disjoint_pairs"_a = true,
      "do_exhaustive_reduction"_a = false,
      "relevance_threshold"_a = -1.0,
      "secondary_threshold"_a = std::numeric_limits<double>::max(),
      "compressed_graphcode"_a = true,
      "double_edges"_a = true);
}

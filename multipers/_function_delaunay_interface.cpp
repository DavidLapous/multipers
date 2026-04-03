#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ext_interface/function_delaunay_interface.hpp"

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "ext_interface/nanobind_registry_runtime.hpp"
#include "nanobind_object_utils.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace mpfd {

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;
using multipers::nanobind_helpers::is_simplextree_object;
using multipers::nanobind_helpers::visit_simplextree_wrapper;
using multipers::nanobind_utils::matrix_from_handle;
using multipers::nanobind_utils::vector_from_handle;

template <typename Wrapper>
void build_function_delaunay_simplextree(Wrapper& wrapper,
                                         const multipers::function_delaunay_interface_input<int>& input,
                                         bool verbose) {
  using Interface = multipers::function_delaunay_simplextree_interface_output;
  Interface output = multipers::function_delaunay_simplextree_interface<int>(input, verbose);
  {
    nb::gil_scoped_release release;
    wrapper.tree.copy_from_interface_object(output);
  }
}

inline multipers::function_delaunay_interface_input<int> build_input(nb::handle point_cloud,
                                                                     nb::handle function_values,
                                                                     bool recover_ids) {
  auto points = matrix_from_handle<double>(point_cloud);
  auto values = vector_from_handle<double>(function_values);
  if (points.size() != values.size()) {
    throw std::runtime_error("point_cloud and function_values sizes do not match.");
  }
  multipers::function_delaunay_interface_input<int> input;
  input.points = std::move(points);
  input.function_values = std::move(values);
  input.recover_ids = recover_ids;
  return input;
}

nb::object function_delaunay_to_slicer_for_target(nb::object target,
                                                  const multipers::function_delaunay_interface_input<int>& input,
                                                  int degree,
                                                  bool multi_chunk,
                                                  bool verbose) {
  auto complex = multipers::function_delaunay_interface_contiguous_slicer<int>(input, degree, multi_chunk, verbose);
  auto& wrapper = nb::cast<CanonicalWrapper&>(target);
  multipers::build_slicer_from_complex(wrapper.truc, complex);
  return target;
}

nb::object function_delaunay_to_simplextree_for_target(nb::object target,
                                                       const multipers::function_delaunay_interface_input<int>& input,
                                                       bool verbose) {
  nb::object out = target.type()();
  visit_simplextree_wrapper(
      out, [&]<typename Desc>(auto& wrapper) { build_function_delaunay_simplextree(wrapper, input, verbose); });
  return out;
}

#endif

}  // namespace mpfd

NB_MODULE(_function_delaunay_interface, m) {
  m.def("_is_available", []() { return multipers::function_delaunay_interface_available(); });

  m.def(
      "function_delaunay_to_slicer",
      [](nb::object slicer,
         nb::handle point_cloud,
         nb::handle function_values,
         int degree,
         bool multi_chunk,
         bool recover_ids,
         bool verbose) {
#if MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
        throw std::runtime_error("function_delaunay in-memory interface is disabled at compile time.");
#else
        auto input = mpfd::build_input(point_cloud, function_values, recover_ids);
        return multipers::nanobind_helpers::run_with_canonical_contiguous_f64_slicer_output(
            slicer,
            [&](const nb::object& target) {
              return mpfd::function_delaunay_to_slicer_for_target(target, input, degree, multi_chunk, verbose);
            });
#endif
      },
      "slicer"_a,
      "point_cloud"_a,
      "function_values"_a,
      "degree"_a,
      "multi_chunk"_a,
      "recover_ids"_a = false,
      "verbose"_a = false);

  m.def(
      "function_delaunay_to_simplextree",
      [](nb::object simplextree, nb::handle point_cloud, nb::handle function_values, bool recover_ids, bool verbose) {
#if MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
        throw std::runtime_error("function_delaunay in-memory interface is disabled at compile time.");
#else
        if (!mpfd::is_simplextree_object(simplextree)) {
          throw nb::type_error("function_delaunay_to_simplextree expects a SimplexTreeMulti target.");
        }
        auto input = mpfd::build_input(point_cloud, function_values, recover_ids);
        return mpfd::function_delaunay_to_simplextree_for_target(simplextree, input, verbose);
#endif
      },
      "simplextree"_a,
      "point_cloud"_a,
      "function_values"_a,
      "recover_ids"_a = false,
      "verbose"_a = false);
}

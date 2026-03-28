#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ext_interface/function_delaunay_interface.hpp"

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace mpfd {

template <typename T>
std::vector<T> cast_vector(nb::handle h) {
  return nb::cast<std::vector<T>>(h);
}

template <typename T>
std::vector<std::vector<T>> cast_matrix(nb::handle h) {
  return nb::cast<std::vector<std::vector<T>>>(h);
}

#if !MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE

inline nb::object numpy_float64() { return nb::module_::import_("numpy").attr("float64"); }

inline nb::object ensure_supported_simplextree_target(nb::object simplextree) {
  return simplextree.attr("astype")(
      "dtype"_a = numpy_float64(), "kcritical"_a = false, "filtration_container"_a = "contiguous");
}

inline multipers::function_delaunay_interface_input<int> build_input(nb::handle point_cloud,
                                                                     nb::handle function_values) {
  auto points = cast_matrix<double>(point_cloud);
  auto values = cast_vector<double>(function_values);
  if (points.size() != values.size()) {
    throw std::runtime_error("point_cloud and function_values sizes do not match.");
  }
  multipers::function_delaunay_interface_input<int> input;
  input.points = std::move(points);
  input.function_values = std::move(values);
  return input;
}

nb::object function_delaunay_to_slicer_for_target(nb::object target,
                                                  const multipers::function_delaunay_interface_input<int>& input,
                                                  int degree,
                                                  bool multi_chunk,
                                                  bool verbose) {
  auto complex = multipers::function_delaunay_interface_contiguous_slicer<int>(input, degree, multi_chunk, verbose);
  auto* target_cpp = reinterpret_cast<multipers::contiguous_f64_slicer*>(nb::cast<intptr_t>(target.attr("get_ptr")()));
  multipers::build_slicer_from_complex(*target_cpp, complex);
  return target;
}

nb::object function_delaunay_to_simplextree_for_target(nb::object target,
                                                       const multipers::function_delaunay_interface_input<int>& input,
                                                       bool verbose) {
  auto output = multipers::function_delaunay_simplextree_interface<int>(input, verbose);
  nb::object out = target.type()();
  out.attr("set_num_parameter")(2);
  using Interface = multipers::function_delaunay_simplextree_interface_output;
  out.attr("_from_interface_ptr")(reinterpret_cast<intptr_t>(new Interface(output)));
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
         bool verbose) {
#if MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
        throw std::runtime_error("function_delaunay in-memory interface is disabled at compile time.");
#else
        auto input = mpfd::build_input(point_cloud, function_values);
        nb::object target = multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
        nb::object out = mpfd::function_delaunay_to_slicer_for_target(target, input, degree, multi_chunk, verbose);
        if (target.ptr() == slicer.ptr()) {
          return out;
        }
        return multipers::nanobind_helpers::astype_slicer_to_original_type(slicer, out);
#endif
      },
      "slicer"_a,
      "point_cloud"_a,
      "function_values"_a,
      "degree"_a,
      "multi_chunk"_a,
      "verbose"_a = false);

  m.def(
      "function_delaunay_to_simplextree",
      [](nb::object simplextree, nb::handle point_cloud, nb::handle function_values, bool verbose) {
#if MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
        throw std::runtime_error("function_delaunay in-memory interface is disabled at compile time.");
#else
        auto input = mpfd::build_input(point_cloud, function_values);
        nb::object target = mpfd::ensure_supported_simplextree_target(simplextree);
        nb::object out = mpfd::function_delaunay_to_simplextree_for_target(target, input, verbose);
        if (target.ptr() == simplextree.ptr()) {
          return out;
        }
        return multipers::nanobind_helpers::astype_simplextree_to_original_type(simplextree, out);
#endif
      },
      "simplextree"_a,
      "point_cloud"_a,
      "function_values"_a,
      "verbose"_a = false);
}

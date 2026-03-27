#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
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

inline nb::object ensure_supported_slicer_target(nb::object slicer) {
  if (multipers::nanobind_helpers::is_supported_contiguous_f64_slicer_object(slicer)) {
    return slicer;
  }
  return slicer.attr("astype")("vineyard"_a = slicer.attr("is_vine"),
                               "kcritical"_a = false,
                               "dtype"_a = numpy_float64(),
                               "col"_a = slicer.attr("col_type"),
                               "pers_backend"_a = "matrix",
                               "filtration_container"_a = "contiguous");
}

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

template <typename Desc>
nb::object function_delaunay_to_slicer_for_target(nb::object target,
                                                  const multipers::function_delaunay_interface_input<int>& input,
                                                  int degree,
                                                  bool multi_chunk,
                                                  bool verbose) {
  using Concrete = typename Desc::concrete;
  auto complex = multipers::function_delaunay_interface_contiguous_slicer<int>(input, degree, multi_chunk, verbose);
  nb::object out = target.type()();
  auto* out_cpp = reinterpret_cast<Concrete*>(nb::cast<intptr_t>(out.attr("get_ptr")()));
  multipers::build_slicer_from_complex(*out_cpp, complex);
  return out;
}

nb::object function_delaunay_to_simplextree_for_target(nb::object target,
                                                       const multipers::function_delaunay_interface_input<int>& input,
                                                       bool verbose) {
  auto output = multipers::function_delaunay_simplextree_interface<int>(input, verbose);
  nb::object out = target.type()();
  out.attr("set_num_parameter")(2);
  using Interface = multipers::function_delaunay_simplextree_interface_output;
  out.attr("_from_ptr")(reinterpret_cast<intptr_t>(new Interface(output)));
  return out;
}

inline nb::object convert_back_to_original_slicer_type(nb::object original, nb::object out) {
  return out.attr("astype")("vineyard"_a = original.attr("is_vine"),
                            "kcritical"_a = original.attr("is_kcritical"),
                            "dtype"_a = original.attr("dtype"),
                            "col"_a = original.attr("col_type"),
                            "pers_backend"_a = original.attr("pers_backend"),
                            "filtration_container"_a = original.attr("filtration_container"));
}

inline nb::object convert_back_to_original_simplextree_type(nb::object original, nb::object out) {
  return out.attr("astype")("dtype"_a = original.attr("dtype"),
                            "kcritical"_a = original.attr("is_kcritical"),
                            "filtration_container"_a = original.attr("filtration_container"));
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
        nb::object target = mpfd::ensure_supported_slicer_target(slicer);
        nb::object out = multipers::nanobind_helpers::dispatch_slicer_by_template_id(
            multipers::nanobind_helpers::template_id_of(target), [&]<typename Desc>() -> nb::object {
              if constexpr (multipers::nanobind_helpers::is_contiguous_f64_slicer_v<Desc>) {
                return mpfd::function_delaunay_to_slicer_for_target<Desc>(target, input, degree, multi_chunk, verbose);
              } else {
                throw nb::type_error("Unsupported slicer template for function_delaunay interface.");
              }
            });
        if (target.ptr() == slicer.ptr()) {
          return out;
        }
        return mpfd::convert_back_to_original_slicer_type(slicer, out);
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
        nb::object target;
        try {
          target = mpfd::ensure_supported_simplextree_target(simplextree);
        } catch (const std::exception& e) {
          throw std::runtime_error(std::string("ensure_supported_simplextree_target failed: ") + e.what());
        }
        nb::object out;
        try {
          out = mpfd::function_delaunay_to_simplextree_for_target(target, input, verbose);
        } catch (const std::exception& e) {
          throw std::runtime_error(std::string("function_delaunay_to_simplextree_for_target failed: ") + e.what());
        }
        if (target.ptr() == simplextree.ptr()) {
          return out;
        }
        try {
          return mpfd::convert_back_to_original_simplextree_type(simplextree, out);
        } catch (const std::exception& e) {
          throw std::runtime_error(std::string("convert_back_to_original_simplextree_type failed: ") + e.what());
        }
#endif
      },
      "simplextree"_a,
      "point_cloud"_a,
      "function_values"_a,
      "verbose"_a = false);
}

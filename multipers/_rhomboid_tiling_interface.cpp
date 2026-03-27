#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ext_interface/rhomboid_tiling_interface.hpp"

#if !MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE
namespace mprt {

inline nb::object numpy_float64() { return nb::module_::import_("numpy").attr("float64"); }

inline nb::object ensure_supported_target(nb::object slicer) {
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

template <typename Desc>
nb::object rhomboid_tiling_to_slicer_for_target(nb::object target,
                                                const multipers::rhomboid_tiling_interface_input<int>& input,
                                                int k_max,
                                                int degree,
                                                bool verbose) {
  using Concrete = typename Desc::concrete;
  auto complex = multipers::rhomboid_tiling_to_contiguous_slicer_interface<int>(input, k_max, degree, verbose);
  nb::object out = target.type()();
  auto* out_cpp = reinterpret_cast<Concrete*>(nb::cast<intptr_t>(out.attr("get_ptr")()));
  multipers::build_slicer_from_complex(*out_cpp, complex);
  return out;
}

inline nb::object convert_back_to_original_type(nb::object original, nb::object out) {
  return out.attr("astype")("vineyard"_a = original.attr("is_vine"),
                            "kcritical"_a = original.attr("is_kcritical"),
                            "dtype"_a = original.attr("dtype"),
                            "col"_a = original.attr("col_type"),
                            "pers_backend"_a = original.attr("pers_backend"),
                            "filtration_container"_a = original.attr("filtration_container"));
}

}  // namespace mprt
#endif

NB_MODULE(_rhomboid_tiling_interface, m) {
  m.def("_is_available", []() { return multipers::rhomboid_tiling_interface_available(); });

  m.def(
      "rhomboid_tiling_to_slicer",
      [](nb::object slicer, nb::handle point_cloud, int k_max, int degree, bool verbose) {
#if MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE
        throw std::runtime_error("rhomboid_tiling in-memory interface is disabled at compile time.");
#else
        multipers::rhomboid_tiling_interface_input<int> input;
        input.points = nb::cast<std::vector<std::vector<double> > >(point_cloud);
        nb::object target = mprt::ensure_supported_target(slicer);
        nb::object out = multipers::nanobind_helpers::dispatch_slicer_by_template_id(
            multipers::nanobind_helpers::template_id_of(target), [&]<typename Desc>() -> nb::object {
              if constexpr (multipers::nanobind_helpers::is_contiguous_f64_slicer_v<Desc>) {
                return mprt::rhomboid_tiling_to_slicer_for_target<Desc>(target, input, k_max, degree, verbose);
              } else {
                throw nb::type_error("Unsupported slicer template for rhomboid_tiling interface.");
              }
            });
        if (target.ptr() == slicer.ptr()) {
          return out;
        }
        return mprt::convert_back_to_original_type(slicer, out);
#endif
      },
      "slicer"_a,
      "point_cloud"_a,
      "k_max"_a,
      "degree"_a,
      "verbose"_a = false);
}

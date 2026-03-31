#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "ext_interface/rhomboid_tiling_interface.hpp"

#if !MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE
namespace mprt {

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;

inline nb::object numpy_float64() { return nb::module_::import_("numpy").attr("float64"); }

nb::object rhomboid_tiling_to_slicer_for_target(nb::object target,
                                                const multipers::rhomboid_tiling_interface_input<int>& input,
                                                int k_max,
                                                int degree,
                                                bool verbose) {
  auto complex = multipers::rhomboid_tiling_to_contiguous_slicer_interface<int>(input, k_max, degree, verbose);
  auto& wrapper = nb::cast<CanonicalWrapper&>(target);
  multipers::build_slicer_from_complex(wrapper.truc, complex);
  return target;
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
        nb::object target = multipers::nanobind_helpers::ensure_canonical_contiguous_f64_slicer_object(slicer);
        nb::object out = mprt::rhomboid_tiling_to_slicer_for_target(target, input, k_max, degree, verbose);
        if (target.ptr() == slicer.ptr()) {
          return out;
        }
        return multipers::nanobind_helpers::astype_slicer_to_original_type(slicer, out);
#endif
      },
      "slicer"_a,
      "point_cloud"_a,
      "k_max"_a,
      "degree"_a,
      "verbose"_a = false);
}

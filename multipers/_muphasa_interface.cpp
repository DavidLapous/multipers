#include <nanobind/nanobind.h>

#include <stdexcept>
#include <string>
#include <type_traits>

#include "ext_interface/muphasa_interface.hpp"

#if !MULTIPERS_DISABLE_MUPHASA_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_MUPHASA_INTERFACE
namespace mpmi {

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_i32_slicer_wrapper;

void require_grid_squeezed_integer_slicer(const nb::object& slicer) {
  multipers::nanobind_helpers::visit_const_slicer_wrapper(
      slicer, []<typename Desc>(const typename Desc::wrapper& wrapper) {
        if (!multipers::nanobind_helpers::has_nonempty_filtration_grid(wrapper.filtration_grid)) {
          throw std::invalid_argument("Muphasa backend expects a grid-squeezed slicer.");
        }
        if constexpr (std::is_floating_point_v<typename Desc::value_type>) {
          throw std::invalid_argument("Muphasa backend expects a grid-squeezed integer-coordinate slicer.");
        }
      });
}

nb::object minimal_presentation_for_target(nb::object target, int degree, bool full_resolution, bool verbose) {
  auto& input_wrapper = nb::cast<CanonicalWrapper&>(target);
  nb::object out = nb::borrow<nb::object>(nb::type<CanonicalWrapper>())();
  auto& out_wrapper = nb::cast<CanonicalWrapper&>(out);
  std::string error;
  {
    nb::gil_scoped_release release;
    try {
      auto complex =
          multipers::muphasa_minpres_contiguous_interface(input_wrapper.truc, degree, full_resolution, verbose);
      multipers::build_slicer_from_complex(out_wrapper.truc, complex);
    } catch (const std::exception& exc) {
      error = exc.what();
    } catch (...) {
      error = "unknown Muphasa backend error";
    }
  }
  if (!error.empty()) {
    throw std::runtime_error(error);
  }
  return out;
}

}  // namespace mpmi
#endif

NB_MODULE(_muphasa_interface, m) {
  auto available = []() { return multipers::muphasa_interface_available(); };
  m.def("_is_available", available);
  m.def("available", available);
  m.def("require", [available]() {
    if (!available()) {
      throw std::runtime_error(
          "Muphasa interface is not available in this build. Rebuild multipers with Muphasa support to enable this "
          "backend.");
    }
  });

  m.def(
      "minimal_presentation",
      [](nb::object slicer, int degree, bool full_resolution, bool keep_generators, bool verbose) {
#if MULTIPERS_DISABLE_MUPHASA_INTERFACE
        throw std::runtime_error("Muphasa interface is disabled at compile time.");
#else
        if (keep_generators) {
          throw std::invalid_argument("Muphasa backend does not support keep_generators yet.");
        }
        if (!multipers::muphasa_interface_available()) {
          throw std::runtime_error("Muphasa interface is not available.");
        }
        mpmi::require_grid_squeezed_integer_slicer(slicer);
        return multipers::nanobind_helpers::run_with_canonical_contiguous_i32_slicer_output(
            slicer, [&](const nb::object& target) {
              return mpmi::minimal_presentation_for_target(target, degree, full_resolution, verbose);
            });
#endif
      },
      "slicer"_a,
      "degree"_a,
      "full_resolution"_a = false,
      "keep_generators"_a = false,
      "verbose"_a = false);
}

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "ext_interface/backend_log_flags.hpp"
#include "ext_interface/mpfree_interface.hpp"

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
#include "ext_interface/nanobind_generator_basis.hpp"
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
#include "mpfree/global.h"

namespace mpmi {

inline void set_backend_stdout(bool enabled) {
#if MPFREE_LOGS
  mpfree::verbose = enabled;
#else
  (void)enabled;
#endif
}

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;

nb::object minimal_presentation_for_target(nb::object target,
                                           int degree,
                                           bool full_resolution,
                                           bool use_clearing,
                                           bool use_chunk,
                                           bool backend_stdout,
                                           bool keep_generators) {
  auto& input_wrapper = nb::cast<CanonicalWrapper&>(target);
  return multipers::nanobind_helpers::build_minpres_slicer_output_for_target(
      target,
      input_wrapper,
      degree,
      keep_generators,
      "mpfree",
      [&] {
        return multipers::mpfree_minpres_contiguous_interface(
            input_wrapper.truc, degree, full_resolution, use_chunk, use_clearing, backend_stdout);
      },
      [&] {
        return multipers::mpfree_minpres_with_generators_contiguous_interface(
            input_wrapper.truc, degree, full_resolution, use_chunk, use_clearing, backend_stdout);
      });
}

}  // namespace mpmi
#endif

NB_MODULE(_mpfree_interface, m) {
  bool ext_log_enabled = false;
  try {
    ext_log_enabled = nb::cast<bool>(nb::module_::import_("multipers.logs").attr("ext_log_enabled")());
  } catch (...) {
    ext_log_enabled = false;
  }

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
  mpmi::set_backend_stdout(ext_log_enabled);
#else
  (void)ext_log_enabled;
#endif

  auto available = []() { return multipers::mpfree_interface_available(); };
  m.def("_is_available", available);
  m.def("available", available);
  m.def("require", [available]() {
    if (!available()) {
      throw std::runtime_error(
          "mpfree interface is not available in this build. Rebuild multipers with mpfree support to enable this backend.");
    }
  });
  m.def("_compiled_log_flags", []() {
    nb::dict out;
    out["mpfree"] = nb::bool_(multipers::backend_log_flags::mpfree);
    return out;
  });

#if MULTIPERS_DISABLE_MPFREE_INTERFACE
  m.def("_set_backend_stdout", [](bool) {}, "enabled"_a);
#else
  m.def("_set_backend_stdout", [](bool enabled) { mpmi::set_backend_stdout(enabled); }, "enabled"_a);
#endif

  m.def(
      "minimal_presentation",
      [](nb::object slicer,
         int degree,
         bool full_resolution,
         bool use_clearing,
         bool use_chunk,
         bool keep_generators,
         bool verbose,
         bool backend_stdout) {
#if MULTIPERS_DISABLE_MPFREE_INTERFACE
        throw std::runtime_error("mpfree interface is disabled at compile time.");
#else
        if (!multipers::mpfree_interface_available()) {
          throw std::runtime_error("mpfree interface is not available.");
        }
        return multipers::nanobind_helpers::run_with_canonical_contiguous_f64_slicer_output(
            slicer,
            [&](const nb::object& target) {
              return mpmi::minimal_presentation_for_target(
                  target, degree, full_resolution, use_clearing, use_chunk, backend_stdout, keep_generators);
            });
#endif
      },
      "slicer"_a,
      "degree"_a,
      "full_resolution"_a = true,
      "use_clearing"_a = true,
      "use_chunk"_a = true,
      "keep_generators"_a = false,
      "verbose"_a = false,
      "_backend_stdout"_a = false);
}

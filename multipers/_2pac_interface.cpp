#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "ext_interface/backend_log_policy.hpp"
#include "ext_interface/2pac_interface.hpp"

#if !MULTIPERS_DISABLE_2PAC_INTERFACE
#include "ext_interface/nanobind_generator_basis.hpp"
#include "ext_interface/nanobind_registry_runtime.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_2PAC_INTERFACE
namespace mtpi {

using CanonicalWrapper = multipers::nanobind_helpers::canonical_contiguous_f64_slicer_wrapper;

nb::object minimal_presentation_for_target(nb::object target,
                                           int degree,
                                           bool full_resolution,
                                           bool use_clearing,
                                           bool use_chunk,
                                           bool verbose,
                                           bool keep_generators,
                                           bool use_cohomology) {
  auto& input_wrapper = nb::cast<CanonicalWrapper&>(target);
  (void)verbose;
  if (keep_generators && use_cohomology) {
    throw std::invalid_argument(
        "2pac cohomology backend does not support keep_generators; use backend='2pac-homology'.");
  }
  const auto algorithm = use_cohomology ? multipers::twopac_minpres_algorithm::cohomology
                                        : multipers::twopac_minpres_algorithm::homology;
  const char* backend_name = use_cohomology ? "2pac" : "2pac-homology";
  const bool effective_verbose_output = multipers::backend_log_policy::backend_log_enabled(
      multipers::backend_log_policy::backend_log_bit::twopac);
  return multipers::nanobind_helpers::build_minpres_slicer_output_for_target(
      target,
      input_wrapper,
      degree,
      keep_generators,
      backend_name,
      [&] {
        return multipers::twopac_minpres_contiguous_interface(
            input_wrapper.truc,
            degree,
            full_resolution,
            use_chunk,
            use_clearing,
            effective_verbose_output,
            algorithm);
      },
      [&] {
        return multipers::twopac_minpres_with_generators_contiguous_interface(
            input_wrapper.truc,
            degree,
            full_resolution,
            use_chunk,
            use_clearing,
            effective_verbose_output,
            algorithm);
      });
}

}  // namespace mtpi
#endif

NB_MODULE(_2pac_interface, m) {
  auto available = []() { return multipers::twopac_interface_available(); };
  m.def("_is_available", available);
  m.def("available", available);
  m.def("require", [available]() {
    if (!available()) {
      throw std::runtime_error(
          "2pac interface is not available in this build. Rebuild multipers with 2pac support to enable this backend.");
    }
  });

  m.def(
      "minimal_presentation",
      [](nb::object slicer,
         int degree,
         bool full_resolution,
         bool use_clearing,
         bool use_chunk,
         bool keep_generators,
         bool verbose,
         bool use_cohomology) {
#if MULTIPERS_DISABLE_2PAC_INTERFACE
        throw std::runtime_error("2pac interface is disabled at compile time.");
#else
        if (!multipers::twopac_interface_available()) {
          throw std::runtime_error("2pac interface is not available.");
        }
        return multipers::nanobind_helpers::run_with_canonical_contiguous_f64_slicer_output(
            slicer,
            [&](const nb::object& target) {
              return mtpi::minimal_presentation_for_target(
                  target,
                  degree,
                  full_resolution,
                  use_clearing,
                  use_chunk,
                  verbose,
                  keep_generators,
                  use_cohomology);
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
      "use_cohomology"_a = true);
}

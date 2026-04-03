#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

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
                                           bool backend_stdout,
                                           bool keep_generators) {
  auto& input_wrapper = nb::cast<CanonicalWrapper&>(target);
  nb::object out = target.type()();
  auto& out_wrapper = nb::cast<CanonicalWrapper&>(out);

  if (!keep_generators) {
    auto complex = multipers::twopac_minpres_contiguous_interface(
        input_wrapper.truc, degree, full_resolution, use_chunk, use_clearing, backend_stdout || verbose);
    multipers::build_slicer_from_complex(out_wrapper.truc, complex);
    return out;
  }

  auto result = multipers::twopac_minpres_with_generators_contiguous_interface(
      input_wrapper.truc, degree, full_resolution, use_chunk, use_clearing, backend_stdout || verbose);
  multipers::build_slicer_from_complex(out_wrapper.truc, result.first);
  out_wrapper.generator_basis = multipers::nanobind_helpers::generator_basis_object_from_degree_rows(
      input_wrapper, degree, result.second, "2pac");
  return out;
}

}  // namespace mtpi
#endif

NB_MODULE(_2pac_interface, m) {
  m.def("_is_available", []() { return multipers::twopac_interface_available(); });

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
#if MULTIPERS_DISABLE_2PAC_INTERFACE
        throw std::runtime_error("2pac in-memory interface is disabled at compile time.");
#else
        if (!multipers::twopac_interface_available()) {
          throw std::runtime_error("2pac in-memory interface is not available.");
        }
        return multipers::nanobind_helpers::run_with_canonical_contiguous_f64_slicer_output(
            slicer,
            [&](const nb::object& target) {
              return mtpi::minimal_presentation_for_target(
                  target, degree, full_resolution, use_clearing, use_chunk, verbose, backend_stdout, keep_generators);
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

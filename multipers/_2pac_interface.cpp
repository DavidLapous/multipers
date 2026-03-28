#include <nanobind/nanobind.h>

#include <stdexcept>

#include "ext_interface/2pac_interface.hpp"

#if !MULTIPERS_DISABLE_2PAC_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_2PAC_INTERFACE
namespace mtpi {

inline nb::object numpy_float64() { return nb::module_::import_("numpy").attr("float64"); }

inline nb::object ensure_supported_target(nb::object slicer) {
  return slicer.attr("astype")("vineyard"_a = false,
                               "kcritical"_a = false,
                               "dtype"_a = numpy_float64(),
                               "col"_a = slicer.attr("col_type"),
                               "pers_backend"_a = "matrix",
                               "filtration_container"_a = "contiguous");
}

nb::object minimal_presentation_for_target(nb::object target,
                                           int degree,
                                           bool full_resolution,
                                           bool use_clearing,
                                           bool use_chunk,
                                           bool verbose,
                                           bool backend_stdout) {
  auto* input_cpp = reinterpret_cast<multipers::contiguous_f64_slicer*>(nb::cast<intptr_t>(target.attr("get_ptr")()));
  auto complex = multipers::twopac_minpres_contiguous_interface(
      *input_cpp, degree, full_resolution, use_chunk, use_clearing, backend_stdout || verbose);
  nb::object out = target.type()();
  auto* out_cpp = reinterpret_cast<multipers::contiguous_f64_slicer*>(nb::cast<intptr_t>(out.attr("get_ptr")()));
  multipers::build_slicer_from_complex(*out_cpp, complex);
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
         bool verbose,
         bool backend_stdout) {
#if MULTIPERS_DISABLE_2PAC_INTERFACE
        throw std::runtime_error("2pac in-memory interface is disabled at compile time.");
#else
        if (!multipers::twopac_interface_available()) {
          throw std::runtime_error("2pac in-memory interface is not available.");
        }
        nb::object target = mtpi::ensure_supported_target(slicer);
        nb::object out = mtpi::minimal_presentation_for_target(
            target, degree, full_resolution, use_clearing, use_chunk, verbose, backend_stdout);
        if (target.ptr() == slicer.ptr()) {
          return out;
        }
        return multipers::nanobind_helpers::astype_slicer_to_original_type(slicer, out);
#endif
      },
      "slicer"_a,
      "degree"_a,
      "full_resolution"_a = true,
      "use_clearing"_a = true,
      "use_chunk"_a = true,
      "verbose"_a = false,
      "_backend_stdout"_a = false);
}

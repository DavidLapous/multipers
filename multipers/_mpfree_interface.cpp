#include <nanobind/nanobind.h>

#include <stdexcept>
#include <type_traits>

#include "mpfree/global.h"
#include "ext_interface/mpfree_interface.hpp"

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
#include "ext_interface/nanobind_registry_helpers.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

#if !MULTIPERS_DISABLE_MPFREE_INTERFACE
namespace mpmi {

inline void set_backend_stdout(bool enabled) { mpfree::verbose = enabled; }

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
nb::object minimal_presentation_for_target(nb::object target,
                                           int degree,
                                           bool full_resolution,
                                           bool use_clearing,
                                           bool use_chunk,
                                           bool backend_stdout) {
  using Concrete = typename Desc::concrete;
  auto* input_cpp = reinterpret_cast<Concrete*>(nb::cast<intptr_t>(target.attr("get_ptr")()));
  auto complex = multipers::mpfree_minpres_contiguous_interface(
      *input_cpp, degree, full_resolution, use_chunk, use_clearing, backend_stdout);
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

}  // namespace mpmi
#endif

NB_MODULE(_mpfree_interface, m) {
  bool ext_log_enabled = false;
  try {
    ext_log_enabled = nb::cast<bool>(nb::module_::import_("multipers.logs").attr("ext_log_enabled")());
  } catch (...) {
    ext_log_enabled = false;
  }
  mpmi::set_backend_stdout(ext_log_enabled);

  m.def("_is_available", []() { return multipers::mpfree_interface_available(); });
  m.def("_set_backend_stdout", [](bool enabled) { mpmi::set_backend_stdout(enabled); }, "enabled"_a);

  m.def(
      "minimal_presentation",
      [](nb::object slicer, int degree, bool full_resolution, bool use_clearing, bool use_chunk, bool verbose, bool backend_stdout) {
#if MULTIPERS_DISABLE_MPFREE_INTERFACE
        throw std::runtime_error("mpfree in-memory interface is disabled at compile time.");
#else
        if (!multipers::mpfree_interface_available()) {
          throw std::runtime_error("mpfree in-memory interface is not available.");
        }
        nb::object target = mpmi::ensure_supported_target(slicer);
        nb::object out = multipers::nanobind_helpers::dispatch_slicer_by_template_id(
            multipers::nanobind_helpers::template_id_of(target), [&]<typename Desc>() -> nb::object {
              if constexpr (multipers::nanobind_helpers::is_contiguous_f64_slicer_v<Desc>) {
                return mpmi::minimal_presentation_for_target<Desc>(
                    target, degree, full_resolution, use_clearing, use_chunk, backend_stdout);
              } else {
                throw nb::type_error("Unsupported slicer template for mpfree interface.");
              }
            });
        if (target.ptr() == slicer.ptr()) {
          return out;
        }
        return mpmi::convert_back_to_original_type(slicer, out);
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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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

inline multipers::packed_morphism_columns packed_columns(
    nb::ndarray<nb::numpy, const std::uint64_t, nb::ndim<1>, nb::c_contig> indptr,
    nb::ndarray<nb::numpy, const std::uint32_t, nb::ndim<1>, nb::c_contig> indices) {
  return {indptr.data(), indices.data(), indptr.shape(0), indices.shape(0)};
}

void require_grid_squeezed_integer_slicer(const nb::object& slicer) {
  multipers::nanobind_helpers::visit_const_slicer_wrapper(
      slicer, []<typename Desc>(const typename Desc::interface& wrapper) {
        if (!multipers::nanobind_helpers::has_nonempty_filtration_grid(wrapper.get_filtration_grid())) {
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
          multipers::muphasa_minpres_contiguous_interface(input_wrapper.get_slicer(), degree, full_resolution, verbose);
      multipers::build_slicer_from_complex(out_wrapper.get_slicer(), complex);
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

nb::object algebra_operation_for_targets(CanonicalWrapper& source_wrapper,
                                         CanonicalWrapper& target_wrapper,
                                         const multipers::packed_morphism_columns& columns,
                                         int degree,
                                         const std::string& op) {
  nb::object out = nb::borrow<nb::object>(nb::type<CanonicalWrapper>())();
  auto& out_wrapper = nb::cast<CanonicalWrapper&>(out);
  std::string error;
  {
    nb::gil_scoped_release release;
    try {
      auto source_input = multipers::muphasa_detail::convert_contiguous_slicer_to_input<int>(source_wrapper.get_slicer());
      auto target_input = multipers::muphasa_detail::convert_contiguous_slicer_to_input<int>(target_wrapper.get_slicer());
      if (source_input.num_parameters != target_input.num_parameters) {
        throw std::invalid_argument("Muphasa source/target parameter counts must agree.");
      }
      std::lock_guard<std::mutex> lock(multipers::muphasa_detail::muphasa_interface_mutex());
      auto raw = multipers::muphasa_detail::compute_free_morphism_op(source_input, target_input, columns, degree, op);
      auto converted = multipers::muphasa_detail::convert_raw_to_output<int>(std::move(raw), degree);
      auto complex = multipers::build_contiguous_i32_slicer_from_output<int>(
          converted.filtration_values, converted.boundaries, converted.dimensions);
      multipers::build_slicer_from_complex(out_wrapper.get_slicer(), complex);
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
      "algebra_operation",
      [](const std::string& op,
         nb::object source,
         nb::object target,
         nb::ndarray<nb::numpy, const std::uint64_t, nb::ndim<1>, nb::c_contig> column_indptr,
         nb::ndarray<nb::numpy, const std::uint32_t, nb::ndim<1>, nb::c_contig> row_indices,
         int degree) {
#if MULTIPERS_DISABLE_MUPHASA_INTERFACE
        throw std::runtime_error("Muphasa interface is disabled at compile time.");
#else
        if (op != "kernel" && op != "image") {
          throw std::invalid_argument(
              "Muphasa kernel/image are supported but quotient/coimage are not yet bound.");
        }
        if (!multipers::muphasa_interface_available()) {
          throw std::runtime_error("Muphasa interface is not available.");
        }
        mpmi::require_grid_squeezed_integer_slicer(source);
        mpmi::require_grid_squeezed_integer_slicer(target);
        auto source_canonical = multipers::nanobind_helpers::ensure_canonical_contiguous_i32_slicer_object(source);
        auto target_canonical = multipers::nanobind_helpers::ensure_canonical_contiguous_i32_slicer_object(target);
        auto& source_wrapper = nb::cast<mpmi::CanonicalWrapper&>(source_canonical);
        auto& target_wrapper = nb::cast<mpmi::CanonicalWrapper&>(target_canonical);
        const nb::object& owner = op == "kernel" ? source : target;
        const nb::object& owner_target = op == "kernel" ? source_canonical : target_canonical;
        auto columns = mpmi::packed_columns(column_indptr, row_indices);
        nb::object out = mpmi::algebra_operation_for_targets(source_wrapper, target_wrapper, columns, degree, op);
        if (owner_target.ptr() == owner.ptr()) {
          return out;
        }
        return multipers::nanobind_helpers::astype_slicer_to_original_type(owner, out);
#endif
      },
      "op"_a,
      "source"_a,
      "target"_a,
      "column_indptr"_a,
      "row_indices"_a,
      "degree"_a);

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

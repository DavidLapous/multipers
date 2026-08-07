#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "contiguous_slicer_bridge.hpp"

namespace multipers::nanobind_helpers {

template <typename Wrapper, typename ComplexFactory, typename ResultFactory>
nanobind::object build_minpres_slicer_output_for_target(nanobind::object target,
                                                        const Wrapper& input_wrapper,
                                                        int degree,
                                                        bool keep_generators,
                                                        const char* backend_name,
                                                        ComplexFactory&& compute_complex,
                                                        ResultFactory&& compute_with_generators) {
  nanobind::object out = target.type()();
  auto& out_wrapper = nanobind::cast<Wrapper&>(out);

  if (!keep_generators) {
    {
      nanobind::gil_scoped_release release;
      auto complex = std::forward<ComplexFactory>(compute_complex)();
      build_slicer_from_complex(out_wrapper.get_slicer(), complex);
    }
    return out;
  }

  auto result = [&]() {
    nanobind::gil_scoped_release release;
    auto result = std::forward<ResultFactory>(compute_with_generators)();
    build_slicer_from_complex(out_wrapper.get_slicer(), result.first);
    if (result.second.row_indices.size() != result.second.row_grades.size()) {
      throw std::runtime_error(std::string(backend_name) + " generator-basis extraction failed: row count mismatch.");
    }
    return result.second;
  }();
  out_wrapper.set_generator_basis(input_wrapper.get_slicer().get_filtered_complex(), degree, result);
  return out;
}

}  // namespace multipers::nanobind_helpers

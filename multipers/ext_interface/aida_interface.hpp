#pragma once

#ifndef MULTIPERS_DISABLE_AIDA_INTERFACE
#if defined(_WIN32)
#define MULTIPERS_DISABLE_AIDA_INTERFACE 1
#else
#define MULTIPERS_DISABLE_AIDA_INTERFACE 0
#endif
#endif

#if !MULTIPERS_DISABLE_AIDA_INTERFACE
#include <aida_interface.hpp>
#include <config.hpp>

namespace multipers {

inline aida::Block_list decompose_aida(aida::AIDA_functor& functor,
                                       const std::vector<std::pair<double, double>>& col_degrees,
                                       const std::vector<std::pair<double, double>>& row_degrees,
                                       const std::vector<std::vector<int>>& data) {
  aida::GradedMatrix presentation(col_degrees.size(), row_degrees.size(), data, col_degrees, row_degrees);
  if (!functor.config.sort) {
    if (!presentation.compatible_sorting_is_verified()) {
      presentation.refresh_compatible_sorted(graded_linalg::TraitLinearOrder<graded_linalg::r2degree>{
          graded_linalg::Degree_traits<graded_linalg::r2degree>::colex_lambda()});
    }
    presentation.require_compatibly_sorted("AIDA with sort=False");
  }
  aida::Block_list summands;
  functor(presentation, summands);
  return summands;
}

}  // namespace multipers
#endif

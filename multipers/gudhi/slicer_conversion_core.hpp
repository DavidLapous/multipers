#pragma once

#include "Persistence_slices_interface.h"

namespace multipers::core {

template <class TargetSlicer, class SourceSlicer>
struct SlicerConversion {
  static TargetSlicer run(const SourceSlicer& source) {
    if constexpr (std::is_same_v<TargetSlicer, SourceSlicer>) {
      return source;
    } else if constexpr (std::is_constructible_v<typename TargetSlicer::Complex, const typename SourceSlicer::Complex&>) {
      return TargetSlicer(source);
    } else {
      throw std::runtime_error("Unsupported slicer conversion.");
    }
  }
};

}  // namespace multipers::core

#if !defined(MULTIPERS_BUILD_CORE_TEMPLATES) && __has_include(<slicer_conversion_extern_templates.h>)
#include <slicer_conversion_extern_templates.h>
#endif

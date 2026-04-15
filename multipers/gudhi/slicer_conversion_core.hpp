#pragma once

#include "Persistence_slices_interface.h"

namespace multipers::core {

template <class TargetSlicer, class SourceSlicer>
struct SlicerConversion {
  static TargetSlicer run(const SourceSlicer& source) { return TargetSlicer(source); }
};

}  // namespace multipers::core

#if !defined(MULTIPERS_BUILD_CORE_TEMPLATES) && __has_include(<slicer_conversion_extern_templates.h>)
#include <slicer_conversion_extern_templates.h>
#endif

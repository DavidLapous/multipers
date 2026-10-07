#pragma once

#include <cstdint>

// #include "Simplex_tree_multi_interface.h"
#include "Multi_simplex_tree_interface.h"

namespace multipers::core {

template <class TargetInterface, class SourceInterface>
struct SimplexTreeConversion {
  static void run(TargetInterface& target, const SourceInterface& source, int numParam = -1) {
    target.copy_from(source, numParam);
  }
};

}  // namespace multipers::core

#if !defined(MULTIPERS_BUILD_CORE_TEMPLATES) && __has_include(<simplextree_conversion_extern_templates.h>)
#include <simplextree_conversion_extern_templates.h>
#endif

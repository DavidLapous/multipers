#pragma once

#include <cstdint>

#include "Simplex_tree_multi_interface.h"

namespace multipers::core {

template <class TargetInterface, class SourceInterface>
struct SimplexTreeConversion {
  static void run(TargetInterface& target, const SourceInterface& source) {
    target.template copy_from_interface<typename SourceInterface::Filtration_value>(
        reinterpret_cast<intptr_t>(const_cast<SourceInterface*>(&source)));
  }
};

}  // namespace multipers::core

#if !defined(MULTIPERS_BUILD_CORE_TEMPLATES) && __has_include(<simplextree_conversion_extern_templates.h>)
#include <simplextree_conversion_extern_templates.h>
#endif

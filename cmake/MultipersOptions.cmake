include_guard(GLOBAL)

if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

set_property(
  DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/options.py"
  "${CMAKE_SOURCE_DIR}/tools/codegen/_registry.py"
  "${CMAKE_SOURCE_DIR}/tools/codegen/query_config.py"
)

set(MULTIPERS_GUDHI_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/gudhi-devel" CACHE PATH "Path to a Gudhi source checkout")
set(MULTIPERS_SKYSCRAPER_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/Skyscraper-Invariant" CACHE PATH "Path to Skyscraper-Invariant")
set(MULTIPERS_GRAPHCODE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/graphcode" CACHE PATH "Path to a graphcode source checkout")
set(MULTIPERS_MUPHASA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/muphasa" CACHE PATH "Path to a Muphasa source checkout")
set(MULTIPERS_DEG_RIPS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/deg_rips" CACHE PATH "Path to a deg_rips source checkout")
set(MULTIPERS_HERA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/hera" CACHE PATH "Path to a Hera source checkout")
set(MULTIPERS_2PAC_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/2pac" CACHE PATH "Path to a 2pac source tree")

function(multipers_define_feature_option feature backend legacy)
  if(NOT DEFINED ${feature})
    set(_default ON)
    if(DEFINED ${legacy} AND ${legacy})
      set(_default OFF)
    endif()
    if(WIN32)
      set(_default OFF)
    endif()
    set(${feature} "${_default}" CACHE BOOL "Build the ${backend} interface")
  endif()
endfunction()

multipers_define_feature_option(MULTIPERS_ENABLE_MPFREE mpfree MULTIPERS_DISABLE_MPFREE_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_MUPHASA muphasa MULTIPERS_DISABLE_MUPHASA_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_FUNCTION_DELAUNAY function_delaunay MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_GRAPHCODE graphcode MULTIPERS_DISABLE_GRAPHCODE_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_DEG_RIPS deg_rips MULTIPERS_DISABLE_DEG_RIPS_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_MULTI_CRITICAL multi_critical MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_RHOMBOID_TILING rhomboid MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_2PAC 2pac MULTIPERS_DISABLE_2PAC_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_AIDA aida MULTIPERS_DISABLE_AIDA_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_HERA hera MULTIPERS_DISABLE_HERA_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_PERSISTENCE_ALGEBRA persistence_algebra MULTIPERS_DISABLE_PERSISTENCE_ALGEBRA_INTERFACE)
multipers_define_feature_option(MULTIPERS_ENABLE_SKYSCRAPER skyscraper MULTIPERS_DISABLE_SKYSCRAPER_INTERFACE)

set(MULTIPERS_EXT_PATCH_DIR "${CMAKE_SOURCE_DIR}/ext/patches")
set(MULTIPERS_GENERATED_EXT_PATCH_DIR "${CMAKE_BINARY_DIR}/generated_ext_patches")
set(MULTIPERS_EXT_PATCH_GENERATOR "${MULTIPERS_EXT_PATCH_DIR}/generate_ext_patches.py")
set(MULTIPERS_APPLY_PATCH_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/ApplyExtPatchOverlay.cmake")
set(MULTIPERS_GENERATED_ROOT "${CMAKE_BINARY_DIR}/generated")
set(MULTIPERS_CODEGEN_CACHE_DIR "${CMAKE_BINARY_DIR}/tmp")

set(MULTIPERS_COMPILED_MODULES_DIR "${CMAKE_BINARY_DIR}/compiled_modules/multipers")

if(WIN32)
  set(MULTIPERS_WINDOWS_RUNTIME_DEP_SET multipers_windows_runtime_deps)
  set(MULTIPERS_WINDOWS_RUNTIME_DEP_DIRECTORIES "")
  if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{CONDA_PREFIX}" _multipers_conda_prefix)
    list(APPEND MULTIPERS_WINDOWS_RUNTIME_DEP_DIRECTORIES
      "${_multipers_conda_prefix}/Library/bin"
      "${_multipers_conda_prefix}/bin"
    )
  endif()
endif()

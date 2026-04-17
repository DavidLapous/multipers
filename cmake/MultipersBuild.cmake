include_guard(GLOBAL)

set(MULTIPERS_DISABLE_MPFREE_INTERFACE OFF)
message(STATUS "[mpfree] Set MULTIPERS_DISABLE_MPFREE_INTERFACE=${MULTIPERS_DISABLE_MPFREE_INTERFACE}")
set(MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE OFF)
message(STATUS "[function_delaunay] Set MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE=${MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE}")
set(MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE OFF)
message(STATUS "[multi_critical] Set MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE=${MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE}")
set(MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE OFF)
message(STATUS "[rhomboid] Set MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE=${MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE}")
set(MULTIPERS_DISABLE_HERA_INTERFACE OFF)
message(STATUS "[hera] Set MULTIPERS_DISABLE_HERA_INTERFACE=${MULTIPERS_DISABLE_HERA_INTERFACE}")

if(WIN32)
  set(MULTIPERS_DISABLE_MPFREE_INTERFACE ON)
  message(STATUS "[mpfree] Forced MULTIPERS_DISABLE_MPFREE_INTERFACE=${MULTIPERS_DISABLE_MPFREE_INTERFACE} on WIN32")
  set(MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE ON)
  message(STATUS "[function_delaunay] Forced MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE=${MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE} on WIN32")
  set(MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE ON)
  message(STATUS "[multi_critical] Forced MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE=${MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE} on WIN32")
  set(MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE ON)
  message(STATUS "[rhomboid] Forced MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE=${MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE} on WIN32")
  set(MULTIPERS_DISABLE_HERA_INTERFACE ON)
  message(STATUS "[hera] Forced MULTIPERS_DISABLE_HERA_INTERFACE=${MULTIPERS_DISABLE_HERA_INTERFACE} on WIN32")
endif()

if(NOT MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER)
  set(MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE ON)
  message(STATUS "[multi_critical] Set MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE=${MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE} because MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER=${MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER}")
endif()

if(NOT CGAL_FOUND)
  set(MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE ON)
  message(STATUS "[rhomboid] Set MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE=${MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE} because CGAL_FOUND=${CGAL_FOUND}")
endif()

if(NOT TARGET multipers_2pac_static)
  set(MULTIPERS_DISABLE_2PAC_INTERFACE ON)
  message(STATUS "[2pac] Set MULTIPERS_DISABLE_2PAC_INTERFACE=${MULTIPERS_DISABLE_2PAC_INTERFACE} because multipers_2pac_static is missing")
else()
  message(STATUS "[2pac] multipers_2pac_static target is available")
endif()

if(NOT TARGET multipers_aida_static)
  set(MULTIPERS_DISABLE_AIDA_INTERFACE ON)
  message(STATUS "[aida] Set MULTIPERS_DISABLE_AIDA_INTERFACE=${MULTIPERS_DISABLE_AIDA_INTERFACE} because multipers_aida_static is missing")
else()
  message(STATUS "[aida] multipers_aida_static target is available")
endif()

set(MULTIPERS_INTERFACE_DISABLE_FLAGS
  MULTIPERS_DISABLE_MPFREE_INTERFACE
  MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE
  MULTIPERS_DISABLE_2PAC_INTERFACE
  MULTIPERS_DISABLE_AIDA_INTERFACE
  MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE
  MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE
  MULTIPERS_DISABLE_HERA_INTERFACE
)

set(MULTIPERS_INTERFACE_DISABLE_DEFINITIONS "")
foreach(_flag IN LISTS MULTIPERS_INTERFACE_DISABLE_FLAGS)
  if(${_flag})
    set(_flag_value 1)
  else()
    set(_flag_value 0)
  endif()
  list(APPEND MULTIPERS_INTERFACE_DISABLE_DEFINITIONS "${_flag}=${_flag_value}")
endforeach()

function(multipers_apply_common_build_flags target_name)
  target_compile_definitions(
    ${target_name}
    PRIVATE
      NPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION
      GUDHI_USE_TBB
      WITH_TBB=ON
      ${MULTIPERS_INTERFACE_DISABLE_DEFINITIONS}
  )

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /O2 /DNDEBUG /W1 /WX- /openmp)
  else()
    target_compile_options(
      ${target_name}
      PRIVATE
        -O3
        -fno-associative-math
        -fno-unsafe-math-optimizations
        -DNDEBUG
        -Wall
        -Wextra
        -Wno-assume
        -Wno-deprecated-declarations
    )
  endif()
endfunction()

function(multipers_link_openmp target_name)
  target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
endfunction()

function(multipers_link_tbb target_name)
  target_link_libraries(${target_name} PRIVATE TBB::tbb)
endfunction()

function(multipers_link_cgal target_name)
  target_link_libraries(${target_name} PRIVATE CGAL::CGAL)
  # CGAL_Core is optional (provides exact arithmetic via GMP/MPFR)
  if(TARGET CGAL::CGAL_Core)
    target_link_libraries(${target_name} PRIVATE CGAL::CGAL_Core)
  endif()
endfunction()

set(MULTIPERS_GENERATED_INCLUDE_DIRS
  "${MULTIPERS_GENERATED_ROOT}/multipers"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi"
  "${MULTIPERS_GENERATED_ROOT}/tools/core"
)

set(MULTIPERS_COMPILED_MODULES_DIR "${CMAKE_BINARY_DIR}/compiled_modules/multipers")

if(WIN32 AND NOT SKBUILD AND NOT DEFINED ENV{MULTIPERS_INTERNAL_WHEEL_BUILD})
  # Only collect runtime deps for local installs, not wheel builds
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

function(multipers_add_core_object_library target_name source_file)
  add_library(${target_name} OBJECT "${source_file}")
  add_dependencies(${target_name} multipers_codegen ${ARGN})
  target_include_directories(
    ${target_name}
    PRIVATE
      ${MULTIPERS_GENERATED_INCLUDE_DIRS}
      ${MULTIPERS_BASE_INCLUDE_DIRS}
      ${MULTIPERS_PHAT_INCLUDE_DIRS}
  )
  multipers_apply_common_build_flags(${target_name})
endfunction()

function(multipers_add_nanobind_object_library target_name source_file)
  add_library(${target_name} OBJECT "${source_file}")
  add_dependencies(${target_name} multipers_codegen ${ARGN})
  target_include_directories(
    ${target_name}
    PRIVATE
      ${MULTIPERS_GENERATED_INCLUDE_DIRS}
      ${MULTIPERS_BASE_INCLUDE_DIRS}
      ${MULTIPERS_NANOBIND_INCLUDE_DIR}
  )
  target_include_directories(${target_name} SYSTEM PRIVATE ${MULTIPERS_NANOBIND_ROBIN_MAP_INCLUDE_DIR})
  target_include_directories(${target_name} BEFORE PRIVATE ${MULTIPERS_PHAT_INCLUDE_DIRS})
  target_compile_definitions(${target_name} PRIVATE NB_COMPACT_ASSERTIONS)
  multipers_apply_common_build_flags(${target_name})
  if(NOT MSVC)
    target_compile_options(${target_name} PRIVATE -fvisibility=hidden)
    set_target_properties(${target_name} PROPERTIES COMPILE_FLAGS "--no-warnings")
  endif()
endfunction()

multipers_add_core_object_library(
  multipers_core_backend_log_policy_obj
  "${CMAKE_SOURCE_DIR}/tools/core/backend_log_policy_core.cc"
)

multipers_add_core_object_library(
  multipers_core_filtrations_obj
  "${CMAKE_SOURCE_DIR}/tools/core/filtrations_core.cc"
)

multipers_add_core_object_library(
  multipers_core_simplextree_obj
  "${CMAKE_SOURCE_DIR}/tools/core/simplextree_core.cc"
  multipers_core_filtrations_obj
)

multipers_add_core_object_library(
  multipers_core_slicer_obj
  "${CMAKE_SOURCE_DIR}/tools/core/slicer_core.cc"
  multipers_core_filtrations_obj
)

multipers_add_core_object_library(
  multipers_core_hera_obj
  "${CMAKE_SOURCE_DIR}/tools/core/hera_monte_carlo_core.cc"
)
target_include_directories(
  multipers_core_hera_obj
  BEFORE
  PRIVATE
    ${MULTIPERS_HERA_PHAT_INCLUDE_DIRS}
    ${MULTIPERS_HERA_INCLUDE_DIRS}
)
multipers_link_openmp(multipers_core_hera_obj)

add_library(
  multipers_core_shared
  SHARED
  $<TARGET_OBJECTS:multipers_core_backend_log_policy_obj>
  $<TARGET_OBJECTS:multipers_core_filtrations_obj>
  $<TARGET_OBJECTS:multipers_core_simplextree_obj>
  $<TARGET_OBJECTS:multipers_core_slicer_obj>
  $<TARGET_OBJECTS:multipers_core_hera_obj>
)
add_dependencies(multipers_core_shared multipers_codegen)
multipers_link_tbb(multipers_core_shared)

set(MULTIPERS_LOCAL_RPATH "")
if(APPLE)
  set(MULTIPERS_LOCAL_RPATH "@loader_path")
elseif(UNIX)
  set(MULTIPERS_LOCAL_RPATH "$ORIGIN")
endif()

set_target_properties(multipers_core_shared PROPERTIES OUTPUT_NAME "multipers_core")
set_target_properties(
  multipers_core_shared
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${MULTIPERS_COMPILED_MODULES_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${MULTIPERS_COMPILED_MODULES_DIR}"
)
if(WIN32)
  set_target_properties(multipers_core_shared PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(MULTIPERS_LOCAL_RPATH)
  set_target_properties(
    multipers_core_shared
    PROPERTIES
      BUILD_RPATH "${MULTIPERS_LOCAL_RPATH}"
      INSTALL_RPATH "${MULTIPERS_LOCAL_RPATH}"
  )
endif()
if(APPLE)
  set_target_properties(
    multipers_core_shared
    PROPERTIES
      INSTALL_NAME_DIR "@rpath"
      SUFFIX ".so"
  )
endif()

find_program(MULTIPERS_PATCH_EXECUTABLE patch REQUIRED)

set(MULTIPERS_EXT_PATCH_DIR "${CMAKE_SOURCE_DIR}/ext/patches")
set(MULTIPERS_GENERATED_EXT_PATCH_DIR "${CMAKE_BINARY_DIR}/generated_ext_patches")
set(MULTIPERS_EXT_PATCH_GENERATOR "${MULTIPERS_EXT_PATCH_DIR}/generate_log_patch.py")

function(multipers_add_generated_patch_file target_name library_name output_path output_var)
  get_filename_component(_patch_dir "${output_path}" DIRECTORY)
  add_custom_command(
    OUTPUT "${output_path}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_patch_dir}"
    COMMAND
      "${Python3_EXECUTABLE}"
      "${MULTIPERS_EXT_PATCH_GENERATOR}"
      "${library_name}"
      --output
      "${output_path}"
    DEPENDS
      "${MULTIPERS_EXT_PATCH_GENERATOR}"
      ${ARGN}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VERBATIM
  )
  add_custom_target(${target_name} DEPENDS "${output_path}")
  set(${output_var} "${output_path}" PARENT_SCOPE)
endfunction()

function(multipers_add_refresh_patch_target target_name library_name output_path)
  get_filename_component(_patch_dir "${output_path}" DIRECTORY)
  add_custom_target(
    ${target_name}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_patch_dir}"
    COMMAND
      "${Python3_EXECUTABLE}"
      "${MULTIPERS_EXT_PATCH_GENERATOR}"
      "${library_name}"
      --output
      "${output_path}"
    BYPRODUCTS "${output_path}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VERBATIM
  )
endfunction()

function(multipers_add_generated_patch_overlay target_name library_name patch_path library_relative_root overlay_root_var)
  set(_overlay_root "${CMAKE_BINARY_DIR}/patched_ext/${library_name}")
  set(_stamp_file "${_overlay_root}/${library_name}_runtime_logs.stamp")
  add_custom_command(
    OUTPUT "${_stamp_file}"
    COMMAND "${CMAKE_COMMAND}"
            -DREPO_ROOT=${CMAKE_SOURCE_DIR}
            -DOVERLAY_ROOT=${_overlay_root}
            -DLIBRARY_NAME=${library_name}
            -DLIBRARY_RELATIVE_ROOT=${library_relative_root}
            "-DSUBDIRS=${ARGN}"
            -DPATCH_FILE=${patch_path}
            -DPATCH_EXECUTABLE=${MULTIPERS_PATCH_EXECUTABLE}
            -DSTAMP_FILE=${_stamp_file}
            -P "${CMAKE_SOURCE_DIR}/cmake/ApplyExtPatchOverlay.cmake"
    DEPENDS
      "${CMAKE_SOURCE_DIR}/cmake/ApplyExtPatchOverlay.cmake"
      "${patch_path}"
      "${MULTIPERS_EXT_PATCH_GENERATOR}"
    VERBATIM
  )
  add_custom_target(${target_name} DEPENDS "${_stamp_file}")
  set(${overlay_root_var} "${_overlay_root}" PARENT_SCOPE)
endfunction()

file(
  GLOB MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_INPUTS
  CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/include/function_delaunay/*.h"
)

multipers_add_generated_patch_file(
  multipers_generate_mpfree_log_patch
  mpfree
  "${MULTIPERS_GENERATED_EXT_PATCH_DIR}/mpfree_runtime_logs.patch"
  MULTIPERS_MPFREE_LOG_PATCH_FILE
  "${CMAKE_SOURCE_DIR}/ext/mpfree/include/mpfree/global.h"
)

multipers_add_generated_patch_file(
  multipers_generate_function_delaunay_log_patch
  function_delaunay
  "${MULTIPERS_GENERATED_EXT_PATCH_DIR}/function_delaunay_runtime_logs.patch"
  MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_FILE
  ${MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_INPUTS}
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/mpfree_mod/include/mpfree/global.h"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/multi_chunk_mod/include/multi_chunk/basic.h"
)

multipers_add_generated_patch_file(
  multipers_generate_multi_critical_log_patch
  multi_critical
  "${MULTIPERS_GENERATED_EXT_PATCH_DIR}/multi_critical_runtime_logs.patch"
  MULTIPERS_MULTI_CRITICAL_LOG_PATCH_FILE
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/include/multi_critical/basic.h"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/mpfree_mod/include/mpfree/global.h"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/multi_chunk_mod/include/multi_chunk/basic.h"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/scc_mod/include/scc/basic.h"
)

set(MULTIPERS_TRACKED_MPFREE_LOG_PATCH_FILE "${MULTIPERS_EXT_PATCH_DIR}/mpfree_runtime_logs.patch")
set(MULTIPERS_TRACKED_FUNCTION_DELAUNAY_LOG_PATCH_FILE "${MULTIPERS_EXT_PATCH_DIR}/function_delaunay_runtime_logs.patch")
set(MULTIPERS_TRACKED_MULTI_CRITICAL_LOG_PATCH_FILE "${MULTIPERS_EXT_PATCH_DIR}/multi_critical_runtime_logs.patch")

multipers_add_refresh_patch_target(
  multipers_refresh_mpfree_log_patch
  mpfree
  "${MULTIPERS_TRACKED_MPFREE_LOG_PATCH_FILE}"
)

multipers_add_refresh_patch_target(
  multipers_refresh_function_delaunay_log_patch
  function_delaunay
  "${MULTIPERS_TRACKED_FUNCTION_DELAUNAY_LOG_PATCH_FILE}"
)

multipers_add_refresh_patch_target(
  multipers_refresh_multi_critical_log_patch
  multi_critical
  "${MULTIPERS_TRACKED_MULTI_CRITICAL_LOG_PATCH_FILE}"
)

add_custom_target(multipers_refresh_ext_patches)
add_dependencies(
  multipers_refresh_ext_patches
  multipers_refresh_mpfree_log_patch
  multipers_refresh_function_delaunay_log_patch
  multipers_refresh_multi_critical_log_patch
)

add_custom_target(multipers_generate_ext_patches)
add_dependencies(multipers_generate_ext_patches multipers_refresh_ext_patches)

add_custom_target(
  multipers_check_ext_patches
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${MULTIPERS_MPFREE_LOG_PATCH_FILE}" "${MULTIPERS_TRACKED_MPFREE_LOG_PATCH_FILE}"
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_FILE}" "${MULTIPERS_TRACKED_FUNCTION_DELAUNAY_LOG_PATCH_FILE}"
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${MULTIPERS_MULTI_CRITICAL_LOG_PATCH_FILE}" "${MULTIPERS_TRACKED_MULTI_CRITICAL_LOG_PATCH_FILE}"
  VERBATIM
)
add_dependencies(
  multipers_check_ext_patches
  multipers_generate_mpfree_log_patch
  multipers_generate_function_delaunay_log_patch
  multipers_generate_multi_critical_log_patch
)

multipers_add_generated_patch_overlay(
  multipers_mpfree_log_overlay
  mpfree
  "${MULTIPERS_MPFREE_LOG_PATCH_FILE}"
  ext/mpfree
  MULTIPERS_MPFREE_PATCH_OVERLAY_ROOT
  include
)

multipers_add_generated_patch_overlay(
  multipers_function_delaunay_log_overlay
  function_delaunay
  "${MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_FILE}"
  ext/function_delaunay
  MULTIPERS_FUNCTION_DELAUNAY_PATCH_OVERLAY_ROOT
  include
  mpfree_mod/include
  multi_chunk_mod/include
)

multipers_add_generated_patch_overlay(
  multipers_multi_critical_log_overlay
  multi_critical
  "${MULTIPERS_MULTI_CRITICAL_LOG_PATCH_FILE}"
  ext/multi_critical
  MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT
  include
  mpfree_mod/include
  multi_chunk_mod/include
  scc_mod/include
)

set(MULTIPERS_MPFREE_INCLUDE_DIRS
  "${MULTIPERS_MPFREE_PATCH_OVERLAY_ROOT}/ext/mpfree/include"
  "${CMAKE_SOURCE_DIR}/ext/mpfree/mpp_utils_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/mpfree/phat_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/mpfree/scc_mod/include"
)

set(MULTIPERS_FUNCTION_DELAUNAY_INCLUDE_DIRS
  "${MULTIPERS_FUNCTION_DELAUNAY_PATCH_OVERLAY_ROOT}/ext/function_delaunay/include"
  "${MULTIPERS_FUNCTION_DELAUNAY_PATCH_OVERLAY_ROOT}/ext/function_delaunay/mpfree_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/mpp_utils_mod/include"
  "${MULTIPERS_FUNCTION_DELAUNAY_PATCH_OVERLAY_ROOT}/ext/function_delaunay/multi_chunk_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/phat/include"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/scc_mod/include"
)

set(MULTIPERS_MULTI_CRITICAL_INCLUDE_DIRS
  "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/include"
  "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/mpfree_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/mpp_utils_mod/include"
  "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/multi_chunk_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/phat_mod/include"
  "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/scc_mod/include"
)

function(multipers_link_shared_core target_name)
  target_link_libraries(${target_name} PRIVATE multipers_core_shared)
  if(MULTIPERS_LOCAL_RPATH)
    set_property(TARGET ${target_name} APPEND PROPERTY BUILD_RPATH "${MULTIPERS_LOCAL_RPATH}")
    set_property(TARGET ${target_name} APPEND PROPERTY INSTALL_RPATH "${MULTIPERS_LOCAL_RPATH}")
  endif()
endfunction()

multipers_add_nanobind_object_library(
  multipers_nanobind_runtime_obj
  "${CMAKE_SOURCE_DIR}/multipers/ext_interface/nanobind_registry_runtime.cpp"
)

# clangd infers commands for ext_interface headers from this shared TU, so it
# must see the optional backend headers that gate those interfaces.
target_include_directories(
  multipers_nanobind_runtime_obj
  PRIVATE
    ${MULTIPERS_AIDA_INCLUDE_DIRS}
    ${MULTIPERS_2PAC_INCLUDE_DIRS}
    ${MULTIPERS_MPFREE_INCLUDE_DIRS}
    ${MULTIPERS_MULTI_CRITICAL_INCLUDE_DIRS}
    ${MULTIPERS_FUNCTION_DELAUNAY_INCLUDE_DIRS}
    ${MULTIPERS_RHOMBOID_TILING_INCLUDE_DIRS}
    ${MULTIPERS_HERA_INCLUDE_DIRS}
)
if(TARGET multipers_2pac_static)
  target_compile_definitions(multipers_nanobind_runtime_obj PRIVATE MULTIPERS_HAS_2PAC_INTERFACE=1)
endif()
add_dependencies(
  multipers_nanobind_runtime_obj
  multipers_mpfree_log_overlay
  multipers_function_delaunay_log_overlay
  multipers_multi_critical_log_overlay
)

function(multipers_link_nanobind_runtime target_name)
  add_dependencies(${target_name} multipers_nanobind_runtime_obj)
  target_sources(${target_name} PRIVATE $<TARGET_OBJECTS:multipers_nanobind_runtime_obj>)
endfunction()

# =============================================================================
# Per-module configuration
# =============================================================================
# Each module explicitly declares its link requirements; optional backends are
# compiled as runtime stubs when their disable flag is active.

function(multipers_configure_module module_name target_name)
  # Default: add standard phat includes
  set(_use_phat_includes TRUE)

  if(module_name STREQUAL "_simplex_tree_multi_nanobind")
    multipers_link_shared_core(${target_name})
    multipers_link_tbb(${target_name})

  elseif(module_name STREQUAL "_slicer_nanobind")
    multipers_link_shared_core(${target_name})
    multipers_link_tbb(${target_name})
    # Build slicer templates in-module so import does not depend on ELF shared-core exports.
    target_compile_definitions(${target_name} PRIVATE MULTIPERS_BUILD_CORE_TEMPLATES=1)

  elseif(module_name STREQUAL "_mma_nanobind")
    multipers_link_tbb(${target_name})

  elseif(module_name STREQUAL "_function_rips_nanobind")
    multipers_link_tbb(${target_name})

  elseif(module_name STREQUAL "_mpfree_interface")
    if(NOT MULTIPERS_DISABLE_MPFREE_INTERFACE)
      add_dependencies(${target_name} multipers_mpfree_log_overlay)
      multipers_link_shared_core(${target_name})
      multipers_link_nanobind_runtime(${target_name})
      target_link_libraries(${target_name} PRIVATE Boost::system Boost::timer Boost::chrono)
      target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
      multipers_link_openmp(${target_name})
      multipers_link_tbb(${target_name})
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_MPFREE_INCLUDE_DIRS})
    endif()
    set(_use_phat_includes FALSE)

  elseif(module_name STREQUAL "_function_delaunay_interface")
    if(NOT MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE)
      add_dependencies(${target_name} multipers_function_delaunay_log_overlay)
      multipers_link_shared_core(${target_name})
      multipers_link_nanobind_runtime(${target_name})
      target_link_libraries(${target_name} PRIVATE Boost::system Boost::timer Boost::chrono)
      target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
      multipers_link_openmp(${target_name})
      multipers_link_tbb(${target_name})
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_FUNCTION_DELAUNAY_INCLUDE_DIRS})
    endif()
    set(_use_phat_includes FALSE)

  elseif(module_name STREQUAL "_multi_critical_interface")
    if(NOT MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE)
      add_dependencies(${target_name} multipers_multi_critical_log_overlay)
      multipers_link_shared_core(${target_name})
      multipers_link_nanobind_runtime(${target_name})
      target_link_libraries(${target_name} PRIVATE Boost::system Boost::timer Boost::chrono)
      target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
      multipers_link_openmp(${target_name})
      multipers_link_tbb(${target_name})
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_MULTI_CRITICAL_INCLUDE_DIRS})
    endif()
    set(_use_phat_includes FALSE)

  elseif(module_name STREQUAL "_rhomboid_tiling_interface")
    if(NOT MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE)
      multipers_link_shared_core(${target_name})
      multipers_link_nanobind_runtime(${target_name})
      target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
      multipers_link_tbb(${target_name})
      multipers_link_cgal(${target_name})
      target_link_libraries(${target_name} PRIVATE multipers_rhomboid_tiling_static)
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_RHOMBOID_TILING_INCLUDE_DIRS})
    endif()

  elseif(module_name STREQUAL "_2pac_interface")
    if(NOT MULTIPERS_DISABLE_2PAC_INTERFACE)
      multipers_link_shared_core(${target_name})
      multipers_link_nanobind_runtime(${target_name})
      target_link_libraries(${target_name} PRIVATE multipers_2pac_static)
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_2PAC_INCLUDE_DIRS})
      multipers_link_openmp(${target_name})
    endif()

  elseif(module_name STREQUAL "_aida_interface")
    if(NOT MULTIPERS_DISABLE_AIDA_INTERFACE)
      multipers_link_shared_core(${target_name})
      multipers_link_nanobind_runtime(${target_name})
      target_link_libraries(${target_name} PRIVATE Boost::system Boost::timer Boost::chrono)
      target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
      multipers_link_openmp(${target_name})
      multipers_link_tbb(${target_name})
      target_link_libraries(${target_name} PRIVATE multipers_aida_static)
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_AIDA_INCLUDE_DIRS})
    endif()

  elseif(module_name STREQUAL "_hera_interface")
    if(NOT MULTIPERS_DISABLE_HERA_INTERFACE)
      multipers_link_shared_core(${target_name})
      multipers_link_nanobind_runtime(${target_name})
      multipers_link_openmp(${target_name})
      multipers_link_tbb(${target_name})
      target_include_directories(
        ${target_name}
        BEFORE
        PRIVATE
          ${MULTIPERS_HERA_PHAT_INCLUDE_DIRS}
          ${MULTIPERS_HERA_INCLUDE_DIRS}
      )
    endif()
    set(_use_phat_includes FALSE)

  endif()

  # Add standard phat includes unless module uses its own forked version
  if(_use_phat_includes)
    target_include_directories(${target_name} BEFORE PRIVATE ${MULTIPERS_PHAT_INCLUDE_DIRS})
  endif()

  if(NOT MSVC)
    set_target_properties(${target_name} PROPERTIES COMPILE_FLAGS "--no-warnings")
  endif()
endfunction()

function(multipers_add_nanobind_module module_name)
  string(REPLACE "." "/" module_path "${module_name}")
  set(source_cpp_file "${CMAKE_SOURCE_DIR}/multipers/${module_path}.cpp")
  set(generated_cpp_file "${MULTIPERS_GENERATED_ROOT}/multipers/${module_path}.cpp")
  set(cpp_template "${CMAKE_SOURCE_DIR}/multipers/${module_path}.cpp.tp")
  if(EXISTS "${source_cpp_file}")
    set(cpp_file "${source_cpp_file}")
  elseif(EXISTS "${cpp_template}")
    set(cpp_file "${generated_cpp_file}")
  else()
    message(FATAL_ERROR "Missing C++ source/template: ${source_cpp_file}")
  endif()

  string(REPLACE "." "_" target_name "multipers_${module_name}")
  nanobind_add_module(${target_name} NB_STATIC "${cpp_file}")
  add_dependencies(${target_name} multipers_codegen)
  set_target_properties(${target_name} PROPERTIES PREFIX "")

  target_include_directories(
    ${target_name}
    PRIVATE
      ${MULTIPERS_GENERATED_INCLUDE_DIRS}
      ${MULTIPERS_BASE_INCLUDE_DIRS}
  )
  multipers_apply_common_build_flags(${target_name})
  multipers_configure_module("${module_name}" ${target_name})

  set_target_properties(
    ${target_name}
    PROPERTIES
      OUTPUT_NAME "${module_name}"
      LIBRARY_OUTPUT_DIRECTORY "${MULTIPERS_COMPILED_MODULES_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${MULTIPERS_COMPILED_MODULES_DIR}"
  )

  if(WIN32)
    if(DEFINED MULTIPERS_WINDOWS_RUNTIME_DEP_SET)
      install(TARGETS ${target_name}
        RUNTIME_DEPENDENCY_SET ${MULTIPERS_WINDOWS_RUNTIME_DEP_SET}
        LIBRARY DESTINATION "multipers"
        RUNTIME DESTINATION "multipers"
        ARCHIVE DESTINATION "multipers"
      )
    else()
      install(TARGETS ${target_name}
        LIBRARY DESTINATION "multipers"
        RUNTIME DESTINATION "multipers"
        ARCHIVE DESTINATION "multipers"
      )
    endif()
  else()
    install(TARGETS ${target_name}
      LIBRARY DESTINATION "multipers"
    )
  endif()
endfunction()

set(MULTIPERS_NANOBIND_MODULES
  _slicer_nanobind
  _mma_nanobind
  _simplex_tree_multi_nanobind
  _function_rips_nanobind
  _grid_helper_nanobind
  _mpfree_interface
  _function_delaunay_interface
  _2pac_interface
  _hera_interface
  _multi_critical_interface
  _rhomboid_tiling_interface
  _aida_interface
)

foreach(module_name IN LISTS MULTIPERS_NANOBIND_MODULES)
  multipers_add_nanobind_module(${module_name})
  string(REPLACE "." "_" module_target_name "multipers_${module_name}")
  list(APPEND MULTIPERS_EXTENSION_TARGETS ${module_target_name})
endforeach()

if(WIN32 AND DEFINED MULTIPERS_WINDOWS_RUNTIME_DEP_SET)
  set(_multipers_runtime_dependency_install_args
    DESTINATION "multipers"
    PRE_EXCLUDE_REGEXES
      [=[python[0-9]+\.dll]=]
      [=[vcruntime.*\.dll]=]
      [=[msvcp.*\.dll]=]
      [=[ucrtbase\.dll]=]
      [=[concrt.*\.dll]=]
    POST_EXCLUDE_REGEXES
      [=[.*[Ww]indows[/\\][Ss]ystem32[/\\]]=]
      [=[api-ms-win-.*]=]
      [=[ext-ms-.*]=]
  )
  if(MULTIPERS_WINDOWS_RUNTIME_DEP_DIRECTORIES)
    list(APPEND _multipers_runtime_dependency_install_args
      DIRECTORIES ${MULTIPERS_WINDOWS_RUNTIME_DEP_DIRECTORIES}
    )
  endif()

  install(
    RUNTIME_DEPENDENCY_SET ${MULTIPERS_WINDOWS_RUNTIME_DEP_SET}
    ${_multipers_runtime_dependency_install_args}
  )
endif()

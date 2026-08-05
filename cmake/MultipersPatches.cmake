include_guard(GLOBAL)

set(MULTIPERS_TRACKED_MPFREE_LOG_PATCH_FILE "${MULTIPERS_EXT_PATCH_DIR}/mpfree_runtime_logs.patch")
set(MULTIPERS_TRACKED_MUPHASA_LOG_PATCH_FILE "${MULTIPERS_EXT_PATCH_DIR}/muphasa_runtime_logs.patch")
set(MULTIPERS_TRACKED_FUNCTION_DELAUNAY_LOG_PATCH_FILE "${MULTIPERS_EXT_PATCH_DIR}/function_delaunay_runtime_logs.patch")
set(MULTIPERS_TRACKED_MULTI_CRITICAL_LOG_PATCH_FILE "${MULTIPERS_EXT_PATCH_DIR}/multi_critical_runtime_logs.patch")
set(MULTIPERS_TRACKED_MULTI_CRITICAL_FEATURES_PATCH_FILE "${MULTIPERS_EXT_PATCH_DIR}/multi_critical_features.patch")
set(MULTIPERS_TRACKED_DEG_RIPS_OPTIMIZATION_PATCH_FILE "${MULTIPERS_EXT_PATCH_DIR}/deg_rips_edge_copy_reducer.patch")

set(_multipers_patch_overlay_needed OFF)
foreach(_feature
    MULTIPERS_FEATURE_MPFREE
    MULTIPERS_FEATURE_MUPHASA
    MULTIPERS_FEATURE_FUNCTION_DELAUNAY
    MULTIPERS_FEATURE_MULTI_CRITICAL
    MULTIPERS_FEATURE_DEG_RIPS
)
  if(${_feature})
    set(_multipers_patch_overlay_needed ON)
  endif()
endforeach()
if(_multipers_patch_overlay_needed)
  find_program(MULTIPERS_PATCH_EXECUTABLE patch REQUIRED)
endif()

function(multipers_add_generated_patch_file target_name library_name output_path output_var)
  get_filename_component(_patch_dir "${output_path}" DIRECTORY)
  add_custom_command(
    OUTPUT "${output_path}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_patch_dir}"
    COMMAND
      "${Python_EXECUTABLE}"
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

function(multipers_add_passthrough_patch_file target_name source_path output_path output_var)
  get_filename_component(_patch_dir "${output_path}" DIRECTORY)
  add_custom_command(
    OUTPUT "${output_path}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_patch_dir}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${source_path}" "${output_path}"
    DEPENDS "${source_path}"
    VERBATIM
  )
  add_custom_target(${target_name} DEPENDS "${output_path}")
  set(${output_var} "${output_path}" PARENT_SCOPE)
endfunction()

function(multipers_add_optional_generated_patch enabled_var target_name library_name output_filename output_var tracked_patch_file)
  set(_output_path "${MULTIPERS_GENERATED_EXT_PATCH_DIR}/${output_filename}")
  if(${enabled_var})
    multipers_add_generated_patch_file(
      ${target_name}
      ${library_name}
      "${_output_path}"
      ${output_var}
      ${ARGN}
    )
  else()
    multipers_add_passthrough_patch_file(
      ${target_name}
      "${tracked_patch_file}"
      "${_output_path}"
      ${output_var}
    )
  endif()
  set(${output_var} "${${output_var}}" PARENT_SCOPE)
endfunction()

function(multipers_add_refresh_patch_target target_name library_name output_path)
  get_filename_component(_patch_dir "${output_path}" DIRECTORY)
  add_custom_target(
    ${target_name}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_patch_dir}"
    COMMAND
      "${Python_EXECUTABLE}"
      "${MULTIPERS_EXT_PATCH_GENERATOR}"
      "${library_name}"
      --output
      "${output_path}"
    BYPRODUCTS "${output_path}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VERBATIM
  )
endfunction()

function(multipers_add_generated_patch_overlay target_name library_name patch_paths library_relative_root overlay_root_var)
  set(_overlay_root "${CMAKE_BINARY_DIR}/patched_ext/${library_name}")
  set(_stamp_file "${_overlay_root}/${library_name}_runtime_logs.stamp")
  set(_depfile "${_stamp_file}.d")
  string(REPLACE ";" "::" _patch_paths_joined "${patch_paths}")
  add_custom_command(
    OUTPUT "${_stamp_file}"
    COMMAND "${CMAKE_COMMAND}"
      "-DREPO_ROOT=${CMAKE_SOURCE_DIR}"
      "-DOVERLAY_ROOT=${_overlay_root}"
      "-DLIBRARY_NAME=${library_name}"
      "-DLIBRARY_RELATIVE_ROOT=${library_relative_root}"
      "-DSUBDIRS=${ARGN}"
      "-DPATCH_FILES=${_patch_paths_joined}"
      "-DPATCH_EXECUTABLE=${MULTIPERS_PATCH_EXECUTABLE}"
      "-DSTAMP_FILE=${_stamp_file}"
      "-DDEPFILE=${_depfile}"
      -P "${MULTIPERS_APPLY_PATCH_SCRIPT}"
    DEPENDS
      "${MULTIPERS_APPLY_PATCH_SCRIPT}"
      ${patch_paths}
      "${MULTIPERS_EXT_PATCH_GENERATOR}"
    DEPFILE "${_depfile}"
    VERBATIM
  )
  add_custom_target(${target_name} DEPENDS "${_stamp_file}")
  set(${overlay_root_var} "${_overlay_root}" PARENT_SCOPE)
endfunction()

function(multipers_add_optional_patch_overlay enabled_var target_name library_name patch_paths library_relative_root overlay_root_var)
  if(${enabled_var})
    multipers_add_generated_patch_overlay(
      ${target_name}
      ${library_name}
      "${patch_paths}"
      ${library_relative_root}
      ${overlay_root_var}
      ${ARGN}
    )
    set(${overlay_root_var} "${${overlay_root_var}}" PARENT_SCOPE)
  else()
    add_custom_target(${target_name})
    set(${overlay_root_var} "" PARENT_SCOPE)
  endif()
endfunction()

file(GLOB MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_INPUTS CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/include/function_delaunay/*.h"
)
file(GLOB MULTIPERS_MUPHASA_LOG_PATCH_INPUTS CONFIGURE_DEPENDS
  "${MULTIPERS_MUPHASA_SOURCE_DIR}/mph/*.cpp"
  "${MULTIPERS_MUPHASA_SOURCE_DIR}/mph/*.h"
)

multipers_add_optional_generated_patch(
  MULTIPERS_FEATURE_MPFREE
  multipers_generate_mpfree_log_patch
  mpfree
  mpfree_runtime_logs.patch
  MULTIPERS_MPFREE_LOG_PATCH_FILE
  "${MULTIPERS_TRACKED_MPFREE_LOG_PATCH_FILE}"
  "${CMAKE_SOURCE_DIR}/ext/mpfree/include/mpfree/global.h"
)
multipers_add_optional_generated_patch(
  MULTIPERS_FEATURE_MUPHASA
  multipers_generate_muphasa_log_patch
  muphasa
  muphasa_runtime_logs.patch
  MULTIPERS_MUPHASA_LOG_PATCH_FILE
  "${MULTIPERS_TRACKED_MUPHASA_LOG_PATCH_FILE}"
  ${MULTIPERS_MUPHASA_LOG_PATCH_INPUTS}
)
multipers_add_optional_generated_patch(
  MULTIPERS_FEATURE_FUNCTION_DELAUNAY
  multipers_generate_function_delaunay_log_patch
  function_delaunay
  function_delaunay_runtime_logs.patch
  MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_FILE
  "${MULTIPERS_TRACKED_FUNCTION_DELAUNAY_LOG_PATCH_FILE}"
  ${MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_INPUTS}
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/mpfree_mod/include/mpfree/global.h"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/multi_chunk_mod/include/multi_chunk/basic.h"
)
multipers_add_optional_generated_patch(
  MULTIPERS_FEATURE_MULTI_CRITICAL
  multipers_generate_multi_critical_log_patch
  multi_critical_logs
  multi_critical_runtime_logs.patch
  MULTIPERS_MULTI_CRITICAL_LOG_PATCH_FILE
  "${MULTIPERS_TRACKED_MULTI_CRITICAL_LOG_PATCH_FILE}"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/include/multi_critical/basic.h"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/mpfree_mod/include/mpfree/global.h"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/scc_mod/include/scc/basic.h"
)
multipers_add_optional_generated_patch(
  MULTIPERS_FEATURE_MULTI_CRITICAL
  multipers_generate_multi_critical_features_patch
  multi_critical_features
  multi_critical_features.patch
  MULTIPERS_MULTI_CRITICAL_FEATURES_PATCH_FILE
  "${MULTIPERS_TRACKED_MULTI_CRITICAL_FEATURES_PATCH_FILE}"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/include/multi_critical/free_resolution.h"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/mpp_utils_mod/include/mpp_utils/Graded_matrix.h"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/mpp_utils_mod/include/mpp_utils/create_graded_matrices_from_pre_column_struct.h"
)
multipers_add_optional_generated_patch(
  MULTIPERS_FEATURE_DEG_RIPS
  multipers_generate_deg_rips_optimization_patch
  deg_rips
  deg_rips_edge_copy_reducer.patch
  MULTIPERS_DEG_RIPS_OPTIMIZATION_PATCH_FILE
  "${MULTIPERS_TRACKED_DEG_RIPS_OPTIMIZATION_PATCH_FILE}"
  "${CMAKE_SOURCE_DIR}/ext/deg_rips/include/deg_rips/Edge_domination_checker.h"
)

multipers_add_refresh_patch_target(multipers_refresh_mpfree_log_patch mpfree "${MULTIPERS_TRACKED_MPFREE_LOG_PATCH_FILE}")
multipers_add_refresh_patch_target(multipers_refresh_muphasa_log_patch muphasa "${MULTIPERS_TRACKED_MUPHASA_LOG_PATCH_FILE}")
multipers_add_refresh_patch_target(multipers_refresh_function_delaunay_log_patch function_delaunay "${MULTIPERS_TRACKED_FUNCTION_DELAUNAY_LOG_PATCH_FILE}")
multipers_add_refresh_patch_target(multipers_refresh_multi_critical_log_patch multi_critical_logs "${MULTIPERS_TRACKED_MULTI_CRITICAL_LOG_PATCH_FILE}")
multipers_add_refresh_patch_target(multipers_refresh_multi_critical_features_patch multi_critical_features "${MULTIPERS_TRACKED_MULTI_CRITICAL_FEATURES_PATCH_FILE}")
multipers_add_refresh_patch_target(multipers_refresh_deg_rips_optimization_patch deg_rips "${MULTIPERS_TRACKED_DEG_RIPS_OPTIMIZATION_PATCH_FILE}")

add_custom_target(multipers_generate_ext_patches)
add_custom_target(
  multipers_check_ext_patches
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${MULTIPERS_MPFREE_LOG_PATCH_FILE}" "${MULTIPERS_TRACKED_MPFREE_LOG_PATCH_FILE}"
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${MULTIPERS_MUPHASA_LOG_PATCH_FILE}" "${MULTIPERS_TRACKED_MUPHASA_LOG_PATCH_FILE}"
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_FILE}" "${MULTIPERS_TRACKED_FUNCTION_DELAUNAY_LOG_PATCH_FILE}"
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${MULTIPERS_MULTI_CRITICAL_LOG_PATCH_FILE}" "${MULTIPERS_TRACKED_MULTI_CRITICAL_LOG_PATCH_FILE}"
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${MULTIPERS_MULTI_CRITICAL_FEATURES_PATCH_FILE}" "${MULTIPERS_TRACKED_MULTI_CRITICAL_FEATURES_PATCH_FILE}"
  COMMAND "${CMAKE_COMMAND}" -E compare_files "${MULTIPERS_DEG_RIPS_OPTIMIZATION_PATCH_FILE}" "${MULTIPERS_TRACKED_DEG_RIPS_OPTIMIZATION_PATCH_FILE}"
  VERBATIM
)
set(_multipers_patch_targets
  multipers_generate_mpfree_log_patch
  multipers_generate_muphasa_log_patch
  multipers_generate_function_delaunay_log_patch
  multipers_generate_multi_critical_log_patch
  multipers_generate_multi_critical_features_patch
  multipers_generate_deg_rips_optimization_patch
)
add_dependencies(multipers_generate_ext_patches ${_multipers_patch_targets})
add_dependencies(multipers_check_ext_patches ${_multipers_patch_targets})

set(_multi_critical_patches "${MULTIPERS_MULTI_CRITICAL_LOG_PATCH_FILE}" "${MULTIPERS_MULTI_CRITICAL_FEATURES_PATCH_FILE}")
multipers_add_optional_patch_overlay(
  MULTIPERS_FEATURE_MPFREE multipers_mpfree_log_overlay mpfree
  "${MULTIPERS_MPFREE_LOG_PATCH_FILE}" ext/mpfree MULTIPERS_MPFREE_PATCH_OVERLAY_ROOT
  include
)
multipers_add_optional_patch_overlay(
  MULTIPERS_FEATURE_MUPHASA multipers_muphasa_log_overlay muphasa
  "${MULTIPERS_MUPHASA_LOG_PATCH_FILE}" ext/muphasa MULTIPERS_MUPHASA_PATCH_OVERLAY_ROOT
  mph
)
multipers_add_optional_patch_overlay(
  MULTIPERS_FEATURE_FUNCTION_DELAUNAY multipers_function_delaunay_log_overlay function_delaunay
  "${MULTIPERS_FUNCTION_DELAUNAY_LOG_PATCH_FILE}" ext/function_delaunay MULTIPERS_FUNCTION_DELAUNAY_PATCH_OVERLAY_ROOT
  include mpfree_mod/include multi_chunk_mod/include
)
multipers_add_optional_patch_overlay(
  MULTIPERS_FEATURE_MULTI_CRITICAL multipers_multi_critical_log_overlay multi_critical
  "${_multi_critical_patches}" ext/multi_critical MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT
  include mpfree_mod/include mpp_utils_mod/include multi_chunk_mod/include phat_mod/include scc_mod/include
)
multipers_add_optional_patch_overlay(
  MULTIPERS_FEATURE_DEG_RIPS multipers_deg_rips_optimization_overlay deg_rips
  "${MULTIPERS_DEG_RIPS_OPTIMIZATION_PATCH_FILE}" ext/deg_rips MULTIPERS_DEG_RIPS_PATCH_OVERLAY_ROOT
  include
)

include_guard(GLOBAL)

set(MULTIPERS_CORE_GENERATED_FILES
  "${MULTIPERS_GENERATED_ROOT}/tools/core/filtrations_instantiations.inc"
  "${MULTIPERS_GENERATED_ROOT}/tools/core/simplextree_instantiations.inc"
  "${MULTIPERS_GENERATED_ROOT}/tools/core/simplextree_conversion_instantiations.inc"
  "${MULTIPERS_GENERATED_ROOT}/tools/core/slicer_instantiations1.inc"
  "${MULTIPERS_GENERATED_ROOT}/tools/core/slicer_instantiations2.inc"
  "${MULTIPERS_GENERATED_ROOT}/tools/core/slicer_instantiations3.inc"
  "${MULTIPERS_GENERATED_ROOT}/multipers/_slicer_nanobind_registry.inc"
  "${MULTIPERS_GENERATED_ROOT}/multipers/_mma_nanobind_registry.inc"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi/filtrations_extern_templates.h"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi/simplextree_multi_extern_templates.h"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi/simplextree_conversion_extern_templates.h"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi/slicer_extern_templates.h"
)

set(MULTIPERS_CODEGEN_DRIVER "${CMAKE_SOURCE_DIR}/tools/tempita_grid_gen.py")
set(MULTIPERS_CODEGEN_QUERY "${CMAKE_SOURCE_DIR}/tools/codegen/query_config.py")

execute_process(
  COMMAND
    "${Python_EXECUTABLE}"
    "${MULTIPERS_CODEGEN_QUERY}"
    --format cmake
  RESULT_VARIABLE MULTIPERS_FLAT_CONTAINER_DETECT_RESULT
  OUTPUT_VARIABLE MULTIPERS_HAS_FLAT_CONTAINER_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE MULTIPERS_FLAT_CONTAINER_DETECT_ERROR
)
if(NOT MULTIPERS_FLAT_CONTAINER_DETECT_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to detect FILTRATION_CONTAINERS from options.py: ${MULTIPERS_FLAT_CONTAINER_DETECT_ERROR}")
endif()

set(MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER OFF)
if(MULTIPERS_HAS_FLAT_CONTAINER_RAW STREQUAL "1")
  set(MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER ON)
endif()
message(STATUS "MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER=${MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER}")

option(MULTIPERS_CODEGEN_VERBOSE "Enable verbose generated-registry/codegen logs" OFF)
if(MULTIPERS_CODEGEN_VERBOSE)
  set(MULTIPERS_TEMPITA_GRID_VERBOSE_VALUE "1")
else()
  set(MULTIPERS_TEMPITA_GRID_VERBOSE_VALUE "0")
endif()

target_include_directories(
  multipers_project_options
  BEFORE INTERFACE
    "${MULTIPERS_GENERATED_ROOT}/multipers"
    "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi"
    "${MULTIPERS_GENERATED_ROOT}/tools/core"
)

add_custom_command(
  OUTPUT ${MULTIPERS_CORE_GENERATED_FILES}
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MULTIPERS_GENERATED_ROOT}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MULTIPERS_CODEGEN_CACHE_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "MULTIPERS_TEMPITA_GRID_VERBOSE=${MULTIPERS_TEMPITA_GRID_VERBOSE_VALUE}"
    "MULTIPERS_TEMPITA_GRID_OUTPUT_ROOT=${MULTIPERS_GENERATED_ROOT}"
    "MULTIPERS_TEMPITA_CACHE_DIR=${MULTIPERS_CODEGEN_CACHE_DIR}"
    "${Python_EXECUTABLE}" "${MULTIPERS_CODEGEN_DRIVER}"
  DEPENDS
    "${MULTIPERS_CODEGEN_DRIVER}"
    "${MULTIPERS_CODEGEN_QUERY}"
    "${CMAKE_SOURCE_DIR}/options.py"
    "${CMAKE_SOURCE_DIR}/tools/codegen/_registry.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  VERBATIM
)

add_custom_target(multipers_codegen DEPENDS ${MULTIPERS_CORE_GENERATED_FILES})

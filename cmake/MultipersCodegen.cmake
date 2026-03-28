include_guard(GLOBAL)

set(MULTIPERS_GENERATED_ROOT "${CMAKE_BINARY_DIR}/generated")
set(MULTIPERS_TEMPITA_CACHE_DIR "${CMAKE_BINARY_DIR}/tmp")

set(MULTIPERS_CORE_GENERATED_FILES
  "${MULTIPERS_GENERATED_ROOT}/tools/core/filtrations_instantiations.inc"
  "${MULTIPERS_GENERATED_ROOT}/tools/core/simplextree_instantiations.inc"
  "${MULTIPERS_GENERATED_ROOT}/tools/core/slicer_instantiations.inc"
  "${MULTIPERS_GENERATED_ROOT}/multipers/_slicer_nanobind_registry.inc"
  "${MULTIPERS_GENERATED_ROOT}/multipers/_mma_nanobind_registry.inc"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi/filtrations_extern_templates.h"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi/simplextree_multi_extern_templates.h"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi/slicer_extern_templates.h"
)

set(MULTIPERS_TEMPITA_GRID_GEN "${CMAKE_SOURCE_DIR}/tools/tempita_grid_gen.py")

execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" "-c"
    "import importlib.util, pathlib; spec = importlib.util.spec_from_file_location('multipers_user_options', pathlib.Path(r'${CMAKE_SOURCE_DIR}') / 'options.py'); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); print('1' if 'Degree_rips_bifiltration' in getattr(mod, 'FILTRATION_CONTAINERS', ()) else '0')"
  RESULT_VARIABLE MULTIPERS_FLAT_CONTAINER_DETECT_RESULT
  OUTPUT_VARIABLE MULTIPERS_HAS_FLAT_CONTAINER_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE MULTIPERS_FLAT_CONTAINER_DETECT_ERROR
)

if(NOT MULTIPERS_FLAT_CONTAINER_DETECT_RESULT EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to detect FILTRATION_CONTAINERS from options.py: ${MULTIPERS_FLAT_CONTAINER_DETECT_ERROR}"
  )
endif()

set(MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER OFF)
if(MULTIPERS_HAS_FLAT_CONTAINER_RAW STREQUAL "1")
  set(MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER ON)
endif()

message(
  STATUS
    "MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER=${MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER}"
)

option(
  MULTIPERS_TEMPITA_GRID_VERBOSE
  "Enable verbose Tempita grid generation logs"
  OFF
)

if(MULTIPERS_TEMPITA_GRID_VERBOSE)
  set(MULTIPERS_TEMPITA_GRID_VERBOSE_VALUE "1")
else()
  set(MULTIPERS_TEMPITA_GRID_VERBOSE_VALUE "0")
endif()

add_custom_command(
  OUTPUT ${MULTIPERS_CORE_GENERATED_FILES}
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MULTIPERS_GENERATED_ROOT}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MULTIPERS_TEMPITA_CACHE_DIR}"
  COMMAND
    "${CMAKE_COMMAND}" -E env
    "MULTIPERS_TEMPITA_GRID_VERBOSE=${MULTIPERS_TEMPITA_GRID_VERBOSE_VALUE}"
    "MULTIPERS_TEMPITA_GRID_OUTPUT_ROOT=${MULTIPERS_GENERATED_ROOT}"
    "MULTIPERS_TEMPITA_CACHE_DIR=${MULTIPERS_TEMPITA_CACHE_DIR}"
    "${Python3_EXECUTABLE}" "${MULTIPERS_TEMPITA_GRID_GEN}"
  DEPENDS
    "${MULTIPERS_TEMPITA_GRID_GEN}"
    "${CMAKE_SOURCE_DIR}/options.py"
    "${CMAKE_SOURCE_DIR}/tools/codegen/_registry.py"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  VERBATIM
)

add_custom_target(multipers_tempita DEPENDS ${MULTIPERS_CORE_GENERATED_FILES})

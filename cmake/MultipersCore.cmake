include_guard(GLOBAL)

set(MULTIPERS_GENERATED_INCLUDE_DIRS
  "${MULTIPERS_GENERATED_ROOT}/multipers"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi"
  "${MULTIPERS_GENERATED_ROOT}/tools/core"
)

function(multipers_add_core_object_library target_name source_file)
  add_library(${target_name} OBJECT "${source_file}")
  add_dependencies(${target_name} multipers_codegen ${ARGN})
  target_link_libraries(
    ${target_name}
    PRIVATE
      multipers::gudhi
      multipers::phat
      multipers::project_options
      multipers::project_warnings
  )
  multipers_apply_common_build_flags(${target_name})
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
  multipers_core_slicer_obj1
  "${CMAKE_SOURCE_DIR}/tools/core/slicer_core1.cc"
  multipers_core_filtrations_obj
)
multipers_add_core_object_library(
  multipers_core_slicer_obj2
  "${CMAKE_SOURCE_DIR}/tools/core/slicer_core2.cc"
  multipers_core_filtrations_obj
)
multipers_add_core_object_library(
  multipers_core_slicer_obj3
  "${CMAKE_SOURCE_DIR}/tools/core/slicer_core3.cc"
  multipers_core_filtrations_obj
)
multipers_add_core_object_library(
  multipers_core_hera_obj
  "${CMAKE_SOURCE_DIR}/tools/core/hera_monte_carlo_core.cc"
)
multipers_add_core_object_library(
  multipers_core_graph_mph0_obj
  "${CMAKE_SOURCE_DIR}/multipers/graph_mph0/graph_mph0.cpp"
)

target_link_libraries(multipers_core_hera_obj PRIVATE multipers::backend_hera multipers::openmp)
target_include_directories(
  multipers_core_hera_obj
  BEFORE
  PRIVATE
    "${MULTIPERS_HERA_SOURCE_DIR}/extern"
    "${MULTIPERS_HERA_SOURCE_DIR}/include"
)

option(MULTIPERS_BUILD_GRAPH_MPH0_BENCHMARK "Build the Graph MPH0 benchmark" OFF)
if(MULTIPERS_BUILD_GRAPH_MPH0_BENCHMARK)
  add_executable(
    multipers_benchmark_graph_mph0
    "${CMAKE_SOURCE_DIR}/benchmarks/benchmark_graph_mph0.cpp"
    "${CMAKE_SOURCE_DIR}/multipers/graph_mph0/graph_mph0.cpp"
  )
  target_link_libraries(
    multipers_benchmark_graph_mph0
    PRIVATE multipers::gudhi multipers::project_options multipers::project_warnings
  )
  multipers_apply_common_build_flags(multipers_benchmark_graph_mph0)
endif()

add_library(
  multipers_core_shared
  SHARED
  $<TARGET_OBJECTS:multipers_core_backend_log_policy_obj>
  $<TARGET_OBJECTS:multipers_core_filtrations_obj>
  $<TARGET_OBJECTS:multipers_core_simplextree_obj>
  $<TARGET_OBJECTS:multipers_core_slicer_obj1>
  $<TARGET_OBJECTS:multipers_core_slicer_obj2>
  $<TARGET_OBJECTS:multipers_core_slicer_obj3>
  $<TARGET_OBJECTS:multipers_core_hera_obj>
  $<TARGET_OBJECTS:multipers_core_graph_mph0_obj>
)
add_dependencies(multipers_core_shared multipers_codegen)
target_link_libraries(
  multipers_core_shared
  PRIVATE
    multipers::gudhi
    multipers::phat
    multipers::project_options
    multipers::backend_hera
    multipers::tbb
    multipers::openmp
)
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

include_guard(GLOBAL)

add_library(multipers_nanobind_runtime_obj OBJECT
  "${CMAKE_SOURCE_DIR}/multipers/ext_interface/nanobind_registry_runtime.cpp"
)
add_dependencies(
  multipers_nanobind_runtime_obj
  multipers_codegen
  multipers_mpfree_log_overlay
  multipers_muphasa_log_overlay
  multipers_function_delaunay_log_overlay
  multipers_multi_critical_log_overlay
  multipers_deg_rips_optimization_overlay
)
target_link_libraries(
  multipers_nanobind_runtime_obj
  PRIVATE
    multipers::gudhi
    multipers::phat
    multipers::project_options
    multipers::project_warnings
    multipers::python
    multipers::backend_mpfree
    multipers::backend_muphasa
    multipers::backend_function_delaunay
    multipers::backend_graphcode
    multipers::backend_deg_rips
    multipers::backend_multi_critical
    multipers::backend_rhomboid_tiling
    multipers::backend_2pac
    multipers::backend_aida
    multipers::backend_hera
    multipers::backend_persistence_algebra
    multipers::backend_skyscraper
)
target_include_directories(
  multipers_nanobind_runtime_obj
  PRIVATE
    "${MULTIPERS_NANOBIND_INCLUDE_DIR}"
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${MULTIPERS_PHAT_INCLUDE_DIRS}
)
target_compile_definitions(multipers_nanobind_runtime_obj PRIVATE NB_COMPACT_ASSERTIONS)
if(NOT MSVC)
  target_compile_options(multipers_nanobind_runtime_obj PRIVATE -fvisibility=hidden -w)
endif()

function(multipers_link_nanobind_runtime target_name)
  add_dependencies(${target_name} multipers_nanobind_runtime_obj)
  target_sources(${target_name} PRIVATE $<TARGET_OBJECTS:multipers_nanobind_runtime_obj>)
endfunction()

function(multipers_add_extension)
  set(_options USE_CORE USE_NANOBIND_RUNTIME)
  set(_one_value NAME SOURCE FEATURE PHAT_MODE)
  set(_multi_value LINK)
  cmake_parse_arguments(ARG "${_options}" "${_one_value}" "${_multi_value}" ${ARGN})
  if(NOT ARG_NAME OR NOT ARG_SOURCE)
    message(FATAL_ERROR "multipers_add_extension requires NAME and SOURCE")
  endif()

  set(_enabled ON)
  if(ARG_FEATURE AND NOT ${ARG_FEATURE})
    set(_enabled OFF)
  endif()

  string(REPLACE "." "_" _target_name "multipers_${ARG_NAME}")
  nanobind_add_module(${_target_name} NB_STATIC "${ARG_SOURCE}")
  add_dependencies(${_target_name} multipers_codegen)
  target_link_libraries(
    ${_target_name}
    PRIVATE
      multipers::gudhi
      multipers::project_options
      multipers::project_warnings
      multipers::python
  )
  if(NOT ARG_PHAT_MODE OR NOT ARG_PHAT_MODE STREQUAL "NONE")
    target_link_libraries(${_target_name} PRIVATE multipers::phat)
  endif()
  if(_enabled)
    if(ARG_USE_CORE)
      multipers_link_shared_core(${_target_name})
    endif()
    if(ARG_USE_NANOBIND_RUNTIME)
      multipers_link_nanobind_runtime(${_target_name})
    endif()
    if(ARG_LINK)
      target_link_libraries(${_target_name} PRIVATE ${ARG_LINK})
    endif()
  endif()
  multipers_apply_common_build_flags(${_target_name})

  set_target_properties(
    ${_target_name}
    PROPERTIES
      PREFIX ""
      OUTPUT_NAME "${ARG_NAME}"
      LIBRARY_OUTPUT_DIRECTORY "${MULTIPERS_COMPILED_MODULES_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${MULTIPERS_COMPILED_MODULES_DIR}"
  )
  set_property(GLOBAL APPEND PROPERTY MULTIPERS_EXTENSION_TARGETS ${_target_name})
endfunction()

multipers_add_extension(
  NAME _slicer_nanobind
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_slicer_nanobind.cpp"
  USE_CORE
  LINK multipers::tbb
)
target_sources(
  multipers__slicer_nanobind
  PRIVATE "${CMAKE_SOURCE_DIR}/multipers/graph_mph0/nanobind_interface.cpp"
)
target_compile_definitions(multipers__slicer_nanobind PRIVATE MULTIPERS_BUILD_CORE_TEMPLATES=1)

multipers_add_extension(
  NAME _mma_nanobind
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_mma_nanobind.cpp"
  USE_CORE
  LINK multipers::tbb
)
multipers_add_extension(
  NAME _simplex_tree_multi_nanobind
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_simplex_tree_multi_nanobind.cpp"
  USE_CORE
  LINK multipers::tbb
)
multipers_add_extension(
  NAME _function_rips_nanobind
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_function_rips_nanobind.cpp"
  USE_CORE
  LINK multipers::tbb
)
multipers_add_extension(
  NAME _mcbif_nanobind
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_mcbif_nanobind.cpp"
  USE_CORE
  LINK multipers::tbb
)
if(CGAL_FOUND)
  multipers_add_extension(
    NAME _core_delaunay_nanobind
    SOURCE "${CMAKE_SOURCE_DIR}/multipers/_core_delaunay_nanobind.cpp"
    USE_CORE
    LINK multipers::tbb multipers::cgal
  )
endif()
multipers_add_extension(
  NAME _grid_helper_nanobind
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_grid_helper_nanobind.cpp"
  USE_CORE
)

multipers_add_extension(
  NAME _mpfree_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_mpfree_interface.cpp"
  FEATURE MULTIPERS_FEATURE_MPFREE
  PHAT_MODE NONE
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_mpfree
)
multipers_add_extension(
  NAME _muphasa_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_muphasa_interface.cpp"
  FEATURE MULTIPERS_FEATURE_MUPHASA
  PHAT_MODE NONE
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_muphasa
)
multipers_add_extension(
  NAME _function_delaunay_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_function_delaunay_interface.cpp"
  FEATURE MULTIPERS_FEATURE_FUNCTION_DELAUNAY
  PHAT_MODE NONE
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_function_delaunay
)
multipers_add_extension(
  NAME _graphcode_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_graphcode_interface.cpp"
  FEATURE MULTIPERS_FEATURE_GRAPHCODE
  PHAT_MODE NONE
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_graphcode
)
multipers_add_extension(
  NAME _deg_rips_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_deg_rips_interface.cpp"
  FEATURE MULTIPERS_FEATURE_DEG_RIPS
  PHAT_MODE NONE
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_deg_rips
)
multipers_add_extension(
  NAME _2pac_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_2pac_interface.cpp"
  FEATURE MULTIPERS_FEATURE_2PAC
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_2pac
)
multipers_add_extension(
  NAME _hera_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_hera_interface.cpp"
  FEATURE MULTIPERS_FEATURE_HERA
  PHAT_MODE NONE
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_hera
)
target_compile_definitions(multipers__hera_interface PRIVATE MD_USE_TBB=1)
multipers_add_extension(
  NAME _multi_critical_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_multi_critical_interface.cpp"
  FEATURE MULTIPERS_FEATURE_MULTI_CRITICAL
  PHAT_MODE NONE
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_multi_critical
)
multipers_add_extension(
  NAME _rhomboid_tiling_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_rhomboid_tiling_interface.cpp"
  FEATURE MULTIPERS_FEATURE_RHOMBOID_TILING
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_rhomboid_tiling
)
multipers_add_extension(
  NAME _aida_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_aida_interface.cpp"
  FEATURE MULTIPERS_FEATURE_AIDA
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_aida
)
multipers_add_extension(
  NAME _end_curves_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_end_curves_interface.cpp"
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_aida multipers::backend_persistence_algebra
)
multipers_add_extension(
  NAME _persistence_algebra_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_persistence_algebra_interface.cpp"
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_persistence_algebra
)
multipers_add_extension(
  NAME _skyscraper_interface
  SOURCE "${CMAKE_SOURCE_DIR}/multipers/_skyscraper_interface.cpp"
  FEATURE MULTIPERS_FEATURE_SKYSCRAPER
  USE_CORE
  USE_NANOBIND_RUNTIME
  LINK multipers::backend_skyscraper
)

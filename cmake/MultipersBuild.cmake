include_guard(GLOBAL)

function(multipers_apply_common_build_flags target_name)
  target_compile_definitions(
    ${target_name}
    PRIVATE
      NPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION
      GUDHI_USE_TBB
      WITH_TBB=ON
  )

  if(WIN32)
    target_compile_definitions(
      ${target_name}
      PRIVATE
        MULTIPERS_DISABLE_MPFREE_INTERFACE=1
        MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE=1
        MULTIPERS_DISABLE_2PAC_INTERFACE=1
        MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE=1
        MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE=1
        MULTIPERS_DISABLE_HERA_INTERFACE=1
    )
  endif()

  if(NOT MULTIPERS_HAS_FLAT_FILTRATION_CONTAINER)
    target_compile_definitions(
      ${target_name}
      PRIVATE
        MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE=1
    )
  endif()

  if(NOT CGAL_FOUND)
    target_compile_definitions(
      ${target_name}
      PRIVATE
        MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE=1
    )
  endif()

  if(NOT TARGET multipers_2pac_static)
    target_compile_definitions(
      ${target_name}
      PRIVATE
        MULTIPERS_DISABLE_2PAC_INTERFACE=1
    )
  endif()

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
  add_dependencies(${target_name} multipers_tempita ${ARGN})
  target_include_directories(
    ${target_name}
    PRIVATE
      ${MULTIPERS_GENERATED_INCLUDE_DIRS}
      ${MULTIPERS_BASE_INCLUDE_DIRS}
      ${MULTIPERS_PHAT_INCLUDE_DIRS}
  )
  multipers_apply_common_build_flags(${target_name})
endfunction()

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

add_library(
  multipers_core_shared
  SHARED
  $<TARGET_OBJECTS:multipers_core_filtrations_obj>
  $<TARGET_OBJECTS:multipers_core_simplextree_obj>
  $<TARGET_OBJECTS:multipers_core_slicer_obj>
)
add_dependencies(multipers_core_shared multipers_tempita)
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

function(multipers_link_shared_core target_name)
  target_link_libraries(${target_name} PRIVATE multipers_core_shared)
  if(MULTIPERS_LOCAL_RPATH)
    set_property(TARGET ${target_name} APPEND PROPERTY BUILD_RPATH "${MULTIPERS_LOCAL_RPATH}")
    set_property(TARGET ${target_name} APPEND PROPERTY INSTALL_RPATH "${MULTIPERS_LOCAL_RPATH}")
  endif()
endfunction()

# =============================================================================
# Per-module configuration
# =============================================================================
# Each module explicitly declares its dependencies. No conditional checks -
# if a dependency is missing, CMake will fail with a clear error.

function(multipers_configure_module module_name target_name)
  # Default: add standard phat includes
  set(_use_phat_includes TRUE)

  if(module_name STREQUAL "_simplex_tree_multi_nanobind")
    multipers_link_shared_core(${target_name})
    multipers_link_tbb(${target_name})

  elseif(module_name STREQUAL "slicer")
    multipers_link_shared_core(${target_name})
    multipers_link_tbb(${target_name})

  elseif(module_name STREQUAL "_slicer_nanobind")
    multipers_link_shared_core(${target_name})
    multipers_link_tbb(${target_name})

  elseif(module_name STREQUAL "_mma_nanobind")
    multipers_link_shared_core(${target_name})
    multipers_link_tbb(${target_name})

  elseif(module_name STREQUAL "_function_rips_nanobind")
    multipers_link_tbb(${target_name})

  elseif(module_name STREQUAL "_mpfree_interface")
    multipers_link_shared_core(${target_name})
    if(NOT WIN32)
      target_link_libraries(${target_name} PRIVATE Boost::system Boost::timer Boost::chrono)
      target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
      multipers_link_openmp(${target_name})
      multipers_link_tbb(${target_name})
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_MPFREE_INCLUDE_DIRS})
    endif()
    set(_use_phat_includes FALSE)

  elseif(module_name STREQUAL "_function_delaunay_interface")
    multipers_link_shared_core(${target_name})
    if(NOT WIN32)
      target_link_libraries(${target_name} PRIVATE Boost::system Boost::timer Boost::chrono)
      target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
      multipers_link_openmp(${target_name})
      multipers_link_tbb(${target_name})
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_FUNCTION_DELAUNAY_INCLUDE_DIRS})
    endif()
    set(_use_phat_includes FALSE)

  elseif(module_name STREQUAL "_multi_critical_interface")
    multipers_link_shared_core(${target_name})
    if(NOT WIN32)
      target_link_libraries(${target_name} PRIVATE Boost::system Boost::timer Boost::chrono)
      target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
      multipers_link_openmp(${target_name})
      multipers_link_tbb(${target_name})
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_MULTI_CRITICAL_INCLUDE_DIRS})
    endif()
    set(_use_phat_includes FALSE)

  elseif(module_name STREQUAL "_rhomboid_tiling_interface")
    multipers_link_shared_core(${target_name})
    if(TARGET multipers_rhomboid_tiling_static)
      target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
      multipers_link_tbb(${target_name})
      multipers_link_cgal(${target_name})
      target_link_libraries(${target_name} PRIVATE multipers_rhomboid_tiling_static)
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_RHOMBOID_TILING_INCLUDE_DIRS})
    endif()

  elseif(module_name STREQUAL "_2pac_interface")
    multipers_link_shared_core(${target_name})
    if(TARGET multipers_2pac_static)
      target_link_libraries(${target_name} PRIVATE multipers_2pac_static)
      target_include_directories(${target_name} PRIVATE ${MULTIPERS_2PAC_INCLUDE_DIRS})
      multipers_link_openmp(${target_name})
    endif()

  elseif(module_name STREQUAL "_aida_interface")
    target_link_libraries(${target_name} PRIVATE Boost::system Boost::timer Boost::chrono)
    target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
    multipers_link_openmp(${target_name})
    target_link_libraries(${target_name} PRIVATE multipers_aida_static)
    target_include_directories(${target_name} PRIVATE ${MULTIPERS_AIDA_INCLUDE_DIRS})
    set(_use_phat_includes FALSE)

  elseif(module_name STREQUAL "_hera_interface")
    if(NOT WIN32)
      multipers_link_openmp(${target_name})
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
  add_dependencies(${target_name} multipers_tempita)
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
)

if(TARGET multipers_aida_static)
  list(APPEND MULTIPERS_NANOBIND_MODULES _aida_interface)
endif()

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

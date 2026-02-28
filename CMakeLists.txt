cmake_minimum_required(VERSION 3.24)
project(multipers LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:-fdiagnostics-color=always>
    $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>
  )
endif()

if(MSVC)
  add_compile_options(/bigobj)
endif()

if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(Boost REQUIRED COMPONENTS system timer chrono)
find_package(OpenMP REQUIRED COMPONENTS CXX)

find_package(TBB CONFIG REQUIRED COMPONENTS tbb)

find_library(MULTIPERS_GMP_LIBRARY REQUIRED NAMES gmp)

set(MULTIPERS_BASE_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/multipers/gudhi"
  "${CMAKE_SOURCE_DIR}/multipers"
)
list(APPEND MULTIPERS_BASE_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
list(APPEND MULTIPERS_BASE_INCLUDE_DIRS ${Python3_NumPy_INCLUDE_DIRS})

if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND MULTIPERS_BASE_INCLUDE_DIRS
    "$ENV{CONDA_PREFIX}/include"
    "$ENV{CONDA_PREFIX}/include/eigen3"
    "$ENV{CONDA_PREFIX}/Library/include"
    "$ENV{CONDA_PREFIX}/Library/include/eigen3"
  )
endif()
list(REMOVE_DUPLICATES MULTIPERS_BASE_INCLUDE_DIRS)

set(MULTIPERS_TEMPLATE_FILES
  "${CMAKE_SOURCE_DIR}/multipers/filtrations.pxd.tp"
  "${CMAKE_SOURCE_DIR}/multipers/filtration_conversions.pxd.tp"
  "${CMAKE_SOURCE_DIR}/multipers/slicer.pxd.tp"
  "${CMAKE_SOURCE_DIR}/multipers/mma_structures.pyx.tp"
  "${CMAKE_SOURCE_DIR}/multipers/simplex_tree_multi.pyx.tp"
  "${CMAKE_SOURCE_DIR}/multipers/slicer.pyx.tp"
)

set(MULTIPERS_GENERATED_FILES
  "${CMAKE_SOURCE_DIR}/multipers/filtrations.pxd"
  "${CMAKE_SOURCE_DIR}/multipers/filtration_conversions.pxd"
  "${CMAKE_SOURCE_DIR}/multipers/slicer.pxd"
  "${CMAKE_SOURCE_DIR}/multipers/mma_structures.pyx"
  "${CMAKE_SOURCE_DIR}/multipers/simplex_tree_multi.pyx"
  "${CMAKE_SOURCE_DIR}/multipers/slicer.pyx"
)

add_custom_command(
  OUTPUT ${MULTIPERS_GENERATED_FILES}
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/_tempita_grid_gen.py"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/process_tempita.py" ${MULTIPERS_TEMPLATE_FILES}
  DEPENDS
    "${CMAKE_SOURCE_DIR}/_tempita_grid_gen.py"
    "${CMAKE_SOURCE_DIR}/tools/process_tempita.py"
    ${MULTIPERS_TEMPLATE_FILES}
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  VERBATIM
)

add_custom_target(multipers_tempita DEPENDS ${MULTIPERS_GENERATED_FILES})

set(MULTIPERS_CYTHON_COMMON_ARGS
  --cplus
  --fast-fail
  --directive language_level=3
  --directive embedsignature=True
  --directive embedsignature.format=python
  --directive binding=True
  --directive infer_types=True
  --directive boundscheck=False
  --directive wraparound=True
  --directive iterable_coroutine=True
  --directive annotation_typing=True
  --directive emit_code_comments=True
  --directive initializedcheck=False
  --directive cdivision=True
  --directive profile=False
  -I "${CMAKE_SOURCE_DIR}/multipers"
)

function(multipers_collect_include_dirs output_var)
  file(GLOB_RECURSE discovered_include_dirs CONFIGURE_DEPENDS LIST_DIRECTORIES true ${ARGN})
  list(FILTER discovered_include_dirs INCLUDE REGEX "/include$")
  list(REMOVE_DUPLICATES discovered_include_dirs)
  list(SORT discovered_include_dirs)
  set(${output_var} ${discovered_include_dirs} PARENT_SCOPE)
endfunction()

multipers_collect_include_dirs(
  MULTIPERS_AIDA_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/AIDA/*"
  "${CMAKE_SOURCE_DIR}/ext/Persistence-Algebra/*"
)
list(APPEND MULTIPERS_AIDA_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/ext/AIDA/src")
list(REMOVE_DUPLICATES MULTIPERS_AIDA_INCLUDE_DIRS)

multipers_collect_include_dirs(
  MULTIPERS_MPFREE_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/mpfree/*"
)

multipers_collect_include_dirs(
  MULTIPERS_MULTI_CRITICAL_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/*"
)

multipers_collect_include_dirs(
  MULTIPERS_FUNCTION_DELAUNAY_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/*"
)

if(NOT WIN32)
  add_library(
    multipers_aida_static
    STATIC
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/aida_decompose.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/aida_functions.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/aida_helpers.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/aida_interface.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/config.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/option_parser.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/block.cpp"
  )
  target_include_directories(multipers_aida_static PUBLIC ${MULTIPERS_AIDA_INCLUDE_DIRS})
  target_link_libraries(multipers_aida_static PUBLIC Boost::timer Boost::chrono)
endif()

function(multipers_apply_common_build_flags target_name)
  target_compile_definitions(
    ${target_name}
    PRIVATE
      NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
      GUDHI_USE_TBB
      WITH_TBB=ON
  )

  if(WIN32)
    target_compile_definitions(
      ${target_name}
      PRIVATE
        MULTIPERS_DISABLE_MPFREE_INTERFACE=1
        MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE=1
        MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE=1
    )
  endif()

  if(MSVC)
    target_compile_options(${target_name} PRIVATE /O2 /DNDEBUG /W1 /WX- /openmp)
  else()
    target_compile_options(
      ${target_name}
      PRIVATE
        -O3
        -fassociative-math
        -funsafe-math-optimizations
        -DNDEBUG
        -Wall
        -Wextra
        -Wno-assume
    )
  endif()
endfunction()

function(multipers_link_openmp target_name)
  target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
endfunction()

function(multipers_link_tbb target_name)
  if(TARGET TBB::tbb)
    target_link_libraries(${target_name} PRIVATE TBB::tbb)
  elseif(TARGET TBB::tbb_static)
    target_link_libraries(${target_name} PRIVATE TBB::tbb_static)
  else()
    message(FATAL_ERROR "TBB target not found")
  endif()
endfunction()

function(multipers_add_extension module_name)
  string(REPLACE "." "/" module_path "${module_name}")
  set(pyx_file "${CMAKE_SOURCE_DIR}/multipers/${module_path}.pyx")
  set(pyx_template "${CMAKE_SOURCE_DIR}/multipers/${module_path}.pyx.tp")
  if(NOT EXISTS "${pyx_file}" AND NOT EXISTS "${pyx_template}")
    message(FATAL_ERROR "Missing Cython source/template: ${pyx_file}")
  endif()

  set(cpp_file "${CMAKE_BINARY_DIR}/cython/${module_path}.cpp")
  set(dep_file "${cpp_file}.dep")
  get_filename_component(cpp_dir "${cpp_file}" DIRECTORY)

  add_custom_command(
    OUTPUT "${cpp_file}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${cpp_dir}"
    COMMAND "${Python3_EXECUTABLE}" -m cython ${MULTIPERS_CYTHON_COMMON_ARGS} -M "${pyx_file}" -o "${cpp_file}"
    DEPENDS "${pyx_file}" ${MULTIPERS_GENERATED_FILES}
    DEPFILE "${dep_file}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    VERBATIM
  )

  string(REPLACE "." "_" target_name "multipers_${module_name}")
  Python3_add_library(${target_name} MODULE WITH_SOABI "${cpp_file}")
  add_dependencies(${target_name} multipers_tempita)
  set_target_properties(${target_name} PROPERTIES PREFIX "")

  target_include_directories(${target_name} PRIVATE ${MULTIPERS_BASE_INCLUDE_DIRS})
  multipers_apply_common_build_flags(${target_name})

  if(module_name STREQUAL "_aida_interface")
    if(WIN32)
      message(FATAL_ERROR "AIDA backend is not supported on Windows")
    endif()
    target_include_directories(${target_name} PRIVATE ${MULTIPERS_AIDA_INCLUDE_DIRS})
    target_link_libraries(
      ${target_name}
      PRIVATE
        multipers_aida_static
        Boost::system
        Boost::timer
        Boost::chrono
        "${MULTIPERS_GMP_LIBRARY}"
    )
    multipers_link_openmp(${target_name})
  elseif(module_name STREQUAL "_mpfree_interface")
    target_include_directories(${target_name} PRIVATE ${MULTIPERS_MPFREE_INCLUDE_DIRS})
    target_link_libraries(
      ${target_name}
      PRIVATE
        Boost::system
        Boost::timer
        Boost::chrono
        "${MULTIPERS_GMP_LIBRARY}"
    )
    multipers_link_openmp(${target_name})
  elseif(module_name STREQUAL "_multi_critical_interface")
    target_include_directories(${target_name} PRIVATE ${MULTIPERS_MULTI_CRITICAL_INCLUDE_DIRS})
    target_link_libraries(
      ${target_name}
      PRIVATE
        Boost::system
        Boost::timer
        Boost::chrono
        "${MULTIPERS_GMP_LIBRARY}"
    )
    multipers_link_openmp(${target_name})
  elseif(module_name STREQUAL "_function_delaunay_interface")
    target_include_directories(${target_name} PRIVATE ${MULTIPERS_FUNCTION_DELAUNAY_INCLUDE_DIRS})
    target_link_libraries(
      ${target_name}
      PRIVATE
        Boost::system
        Boost::timer
        Boost::chrono
        "${MULTIPERS_GMP_LIBRARY}"
    )
    multipers_link_openmp(${target_name})
    multipers_link_tbb(${target_name})
  elseif(module_name STREQUAL "io" OR module_name STREQUAL "ops")
    # No extra native libs
  else()
    multipers_link_tbb(${target_name})
  endif()

  set(output_name "${module_name}")
  set(package_dir "multipers")

  set_target_properties(${target_name} PROPERTIES OUTPUT_NAME "${output_name}")
  if(WIN32)
    install(TARGETS ${target_name}
    LIBRARY DESTINATION "${package_dir}"
    RUNTIME DESTINATION "${package_dir}"
    ARCHIVE DESTINATION "${package_dir}"
  )

    install(CODE "
      message(STATUS \"Bundling runtime deps for: $<TARGET_FILE:${target_name}>\")
      file(GET_RUNTIME_DEPENDENCIES
        RESOLVED_DEPENDENCIES_VAR deps
        UNRESOLVED_DEPENDENCIES_VAR unresolved
        CONFLICTING_DEPENDENCIES_PREFIX conflict
        MODULES \"$<TARGET_FILE:${target_name}>\"
        DIRECTORIES
          \"\$ENV{CONDA_PREFIX}/Library/bin\"
          \"\$ENV{CONDA_PREFIX}/bin\"
        POST_EXCLUDE_REGEXES
          \".*[Ww]indows[/\\\\][Ss]ystem32[/\\\\].*\"
          \"api-ms-win-.*\"
          \"ext-ms-.*\"
      )

      foreach(dep IN LISTS deps)
        file(INSTALL
          DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${package_dir}\"
          TYPE SHARED_LIBRARY
          FILES \"\${dep}\"
        )
      endforeach()

      foreach(dep IN LISTS unresolved)
        message(WARNING \"Unresolved dependency: \${dep}\")
      endforeach()
    ")
  else()
    install(TARGETS ${target_name}
    LIBRARY DESTINATION "${package_dir}"
  )
  endif()
endfunction()

set(MULTIPERS_MODULES
  simplex_tree_multi
  io
  function_rips
  mma_structures
  multiparameter_module_approximation
  point_measure
  grids
  slicer
  ops
  _mpfree_interface
  _aida_interface
  _function_delaunay_interface
  _multi_critical_interface
)

if(WIN32)
  list(REMOVE_ITEM MULTIPERS_MODULES
    _aida_interface
    # _mpfree_interface
    # _function_delaunay_interface
    # _multi_critical_interface
  )
endif()

foreach(module_name IN LISTS MULTIPERS_MODULES)
  multipers_add_extension(${module_name})
endforeach()

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/multipers/"
  DESTINATION multipers
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.pyo" EXCLUDE
  PATTERN "*.pyx" EXCLUDE
  PATTERN "*.pxd" EXCLUDE
  PATTERN "*.tp" EXCLUDE
  PATTERN "*.cpp" EXCLUDE
  PATTERN "*.h" EXCLUDE
  PATTERN "*.hpp" EXCLUDE
)

include_guard(GLOBAL)

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
        MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE=1
    )
  endif()

  if(NOT CGAL_FOUND)
    target_compile_definitions(
      ${target_name}
      PRIVATE
        MULTIPERS_DISABLE_RHOMBOID_TILING_INTERFACE=1
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

set(MULTIPERS_SHARED_CORE_MODULES
  simplex_tree_multi
  slicer
  multiparameter_module_approximation
  _mpfree_interface
  _function_delaunay_interface
  _multi_critical_interface
  _rhomboid_tiling_interface
)

set(MULTIPERS_BOOST_MODULES
  _aida_interface
  _mpfree_interface
  _multi_critical_interface
  _function_delaunay_interface
)

set(MULTIPERS_GMP_MODULE
  _aida_interface
  _mpfree_interface
  _multi_critical_interface
  _function_delaunay_interface
  _rhomboid_tiling_interface
)

set(MULTIPERS_GMP_MODULES ${MULTIPERS_GMP_MODULE})

set(MULTIPERS_OPENMP_MODULES
  _aida_interface
  _mpfree_interface
  _multi_critical_interface
  _function_delaunay_interface
)

set(MULTIPERS_TBB_MODULES
  simplex_tree_multi
  function_rips
  mma_structures
  multiparameter_module_approximation
  point_measure
  grids
  slicer
  _function_delaunay_interface
)

set(MULTIPERS_MODULE_INCLUDE_DIR_MAP
  _aida_interface MULTIPERS_AIDA_INCLUDE_DIRS
  _mpfree_interface MULTIPERS_MPFREE_INCLUDE_DIRS
  _multi_critical_interface MULTIPERS_MULTI_CRITICAL_INCLUDE_DIRS
  _function_delaunay_interface MULTIPERS_FUNCTION_DELAUNAY_INCLUDE_DIRS
  _rhomboid_tiling_interface MULTIPERS_RHOMBOID_TILING_INCLUDE_DIRS
)

set(MULTIPERS_FORKED_PHAT_MODULES
  _aida_interface
  _mpfree_interface
  _multi_critical_interface
  _function_delaunay_interface
)

set(MULTIPERS_GENERATED_INCLUDE_DIRS
  "${MULTIPERS_GENERATED_ROOT}/multipers"
  "${MULTIPERS_GENERATED_ROOT}/multipers/gudhi"
  "${MULTIPERS_GENERATED_ROOT}/tools/core"
)

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

function(multipers_configure_extension_include_dirs module_name target_name)
  list(LENGTH MULTIPERS_MODULE_INCLUDE_DIR_MAP include_dir_map_length)
  if(include_dir_map_length GREATER 1)
    math(EXPR include_dir_map_last "${include_dir_map_length} - 2")
    foreach(i RANGE 0 ${include_dir_map_last} 2)
      math(EXPR j "${i} + 1")
      list(GET MULTIPERS_MODULE_INCLUDE_DIR_MAP ${i} include_module)
      list(GET MULTIPERS_MODULE_INCLUDE_DIR_MAP ${j} include_var)
      if("${module_name}" STREQUAL "${include_module}")
        target_include_directories(${target_name} PRIVATE ${${include_var}})
        break()
      endif()
    endforeach()
  endif()

  list(FIND MULTIPERS_FORKED_PHAT_MODULES "${module_name}" uses_forked_phat)
  if(uses_forked_phat EQUAL -1)
    target_include_directories(${target_name} BEFORE PRIVATE ${MULTIPERS_PHAT_INCLUDE_DIRS})
  endif()
endfunction()

function(multipers_configure_extension_backend module_name target_name)
  if(module_name STREQUAL "_aida_interface" AND WIN32)
    message(FATAL_ERROR "AIDA backend is not supported on Windows")
  endif()

  list(FIND MULTIPERS_BOOST_MODULES "${module_name}" uses_boost)
  if(NOT uses_boost EQUAL -1)
    target_link_libraries(${target_name} PRIVATE Boost::system Boost::timer Boost::chrono)
  endif()

  list(FIND MULTIPERS_GMP_MODULE "${module_name}" uses_gmp)
  if(NOT uses_gmp EQUAL -1)
    target_link_libraries(${target_name} PRIVATE "${MULTIPERS_GMP_LIBRARY}")
  endif()

  if(module_name STREQUAL "_aida_interface")
    target_link_libraries(${target_name} PRIVATE multipers_aida_static)
  endif()

  if(module_name STREQUAL "_rhomboid_tiling_interface")
    if(NOT MSVC)
      target_compile_options(
        ${target_name}
        PRIVATE
          -fno-associative-math
          -fno-unsafe-math-optimizations
      )
    endif()
    if(TARGET multipers_rhomboid_tiling_static)
      target_link_libraries(${target_name} PRIVATE multipers_rhomboid_tiling_static)
    endif()
    if(TARGET CGAL::CGAL)
      target_link_libraries(${target_name} PRIVATE CGAL::CGAL)
    endif()
    if(TARGET CGAL::CGAL_Core)
      target_link_libraries(${target_name} PRIVATE CGAL::CGAL_Core)
    endif()
  endif()

  list(FIND MULTIPERS_OPENMP_MODULES "${module_name}" uses_openmp)
  if(NOT uses_openmp EQUAL -1)
    multipers_link_openmp(${target_name})
  endif()

  list(FIND MULTIPERS_TBB_MODULES "${module_name}" uses_tbb)
  if(NOT uses_tbb EQUAL -1)
    multipers_link_tbb(${target_name})
  endif()
endfunction()

function(multipers_add_extension module_name)
  string(REPLACE "." "/" module_path "${module_name}")
  set(source_pyx_file "${CMAKE_SOURCE_DIR}/multipers/${module_path}.pyx")
  set(generated_pyx_file "${MULTIPERS_GENERATED_ROOT}/multipers/${module_path}.pyx")
  set(pyx_template "${CMAKE_SOURCE_DIR}/multipers/${module_path}.pyx.tp")
  if(EXISTS "${pyx_template}")
    set(pyx_file "${generated_pyx_file}")
  elseif(EXISTS "${source_pyx_file}")
    set(pyx_file "${source_pyx_file}")
  else()
    message(FATAL_ERROR "Missing Cython source/template: ${source_pyx_file}")
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

  target_include_directories(
    ${target_name}
    PRIVATE
      ${MULTIPERS_GENERATED_INCLUDE_DIRS}
      ${MULTIPERS_BASE_INCLUDE_DIRS}
  )
  multipers_apply_common_build_flags(${target_name})

  list(FIND MULTIPERS_SHARED_CORE_MODULES "${module_name}" uses_shared_core)
  if(NOT uses_shared_core EQUAL -1)
    multipers_link_shared_core(${target_name})
  endif()

  multipers_configure_extension_include_dirs("${module_name}" ${target_name})
  multipers_configure_extension_backend("${module_name}" ${target_name})

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
  _rhomboid_tiling_interface
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

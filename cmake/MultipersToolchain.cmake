include_guard(GLOBAL)

add_library(multipers_project_options INTERFACE)
add_library(multipers::project_options ALIAS multipers_project_options)
target_compile_features(multipers_project_options INTERFACE cxx_std_20)
set_property(TARGET multipers_project_options PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
target_include_directories(
  multipers_project_options
  INTERFACE
    "${CMAKE_SOURCE_DIR}/multipers"
    "${CMAKE_SOURCE_DIR}/multipers/gudhi"
    "${MULTIPERS_GENERATED_ROOT}"
)
if(DEFINED ENV{CONDA_PREFIX})
  foreach(_multipers_conda_include
      "$ENV{CONDA_PREFIX}/include"
      "$ENV{CONDA_PREFIX}/include/eigen3"
      "$ENV{CONDA_PREFIX}/Library/include"
      "$ENV{CONDA_PREFIX}/Library/include/eigen3"
  )
    if(IS_DIRECTORY "${_multipers_conda_include}")
      target_include_directories(multipers_project_options INTERFACE "${_multipers_conda_include}")
    endif()
  endforeach()
endif()
target_compile_definitions(
  multipers_project_options
  INTERFACE
    NPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION
    GUDHI_USE_TBB
)

add_library(multipers_project_warnings INTERFACE)
add_library(multipers::project_warnings ALIAS multipers_project_warnings)

if(MSVC)
  target_compile_options(multipers_project_options INTERFACE /bigobj)
  target_compile_options(multipers_project_warnings INTERFACE /W1 /WX-)
else()
  target_compile_options(
    multipers_project_options
    INTERFACE
      $<$<NOT:$<CONFIG:Debug>>:-g0>
      -fno-associative-math
      -fno-unsafe-math-optimizations
  )
  target_compile_options(
    multipers_project_warnings
    INTERFACE
      -Wall
      -Wextra
      -Wno-deprecated-declarations
  )
endif()

function(multipers_apply_common_build_flags target_name)
  target_link_libraries(
    ${target_name}
    PRIVATE
      multipers::project_options
      multipers::project_warnings
  )
endfunction()

function(multipers_link_openmp target_name)
  target_link_libraries(${target_name} PRIVATE multipers::openmp)
endfunction()

function(multipers_link_tbb target_name)
  target_link_libraries(${target_name} PRIVATE multipers::tbb)
endfunction()

function(multipers_link_cgal target_name)
  target_link_libraries(${target_name} PRIVATE multipers::cgal)
endfunction()

set(MULTIPERS_LOCAL_RPATH "")
if(APPLE)
  set(MULTIPERS_LOCAL_RPATH "@loader_path")
elseif(UNIX)
  set(MULTIPERS_LOCAL_RPATH "$ORIGIN")
endif()

function(multipers_link_shared_core target_name)
  target_link_libraries(${target_name} PRIVATE multipers_core_shared)
  if(MULTIPERS_LOCAL_RPATH)
    set_property(TARGET ${target_name} APPEND PROPERTY BUILD_RPATH "${MULTIPERS_LOCAL_RPATH}")
    set_property(TARGET ${target_name} APPEND PROPERTY INSTALL_RPATH "${MULTIPERS_LOCAL_RPATH}")
  endif()
endfunction()

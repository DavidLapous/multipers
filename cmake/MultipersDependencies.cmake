include_guard(GLOBAL)

if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
  RESULT_VARIABLE MULTIPERS_NANOBIND_CMAKE_DIR_RESULT
  OUTPUT_VARIABLE MULTIPERS_NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE MULTIPERS_NANOBIND_CMAKE_DIR_ERROR
)

if(NOT MULTIPERS_NANOBIND_CMAKE_DIR_RESULT EQUAL 0)
  message(
    FATAL_ERROR
      "Failed to locate nanobind CMake files: ${MULTIPERS_NANOBIND_CMAKE_DIR_ERROR}"
  )
endif()

list(PREPEND CMAKE_PREFIX_PATH "${MULTIPERS_NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

find_package(Boost REQUIRED COMPONENTS system timer chrono)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(TBB CONFIG REQUIRED COMPONENTS tbb)
if(POLICY CMP0167)
  set(CMAKE_POLICY_DEFAULT_CMP0167 NEW)
endif()
find_package(CGAL QUIET COMPONENTS Core)
if(POLICY CMP0167)
  unset(CMAKE_POLICY_DEFAULT_CMP0167)
endif()

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

set(MULTIPERS_AIDA_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/AIDA/include"
  "${CMAKE_SOURCE_DIR}/ext/AIDA/src"
  "${CMAKE_SOURCE_DIR}/ext/Persistence-Algebra/include"
)

set(MULTIPERS_MPFREE_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/mpfree/include"
  "${CMAKE_SOURCE_DIR}/ext/mpfree/mpp_utils_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/mpfree/phat_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/mpfree/scc_mod/include"
)

set(MULTIPERS_PHAT_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/phat/include"
)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/ext/phat/include/phat/representations/bit_tree_pivot_column.h")
  message(FATAL_ERROR "Missing vanilla PHAT headers under ext/phat/include")
endif()

set(MULTIPERS_MULTI_CRITICAL_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/include"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/mpfree_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/mpp_utils_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/multi_chunk_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/phat_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/multi_critical/scc_mod/include"
)

set(MULTIPERS_FUNCTION_DELAUNAY_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/include"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/mpfree_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/mpp_utils_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/multi_chunk_mod/include"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/phat/include"
  "${CMAKE_SOURCE_DIR}/ext/function_delaunay/scc_mod/include"
)

set(MULTIPERS_RHOMBOID_TILING_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/ext/rhomboidtiling_newer_cgal_version/src"
)

set(MULTIPERS_HERA_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/hera" CACHE PATH "Path to a Hera source checkout")
set(MULTIPERS_HERA_INCLUDE_DIRS "")
set(MULTIPERS_HERA_PHAT_INCLUDE_DIRS "")
if(EXISTS "${MULTIPERS_HERA_SOURCE_DIR}/include/hera/matching_distance.h")
  list(APPEND MULTIPERS_HERA_INCLUDE_DIRS "${MULTIPERS_HERA_SOURCE_DIR}/include")
endif()
if(EXISTS "${MULTIPERS_HERA_SOURCE_DIR}/extern/phat/boundary_matrix.h")
  list(APPEND MULTIPERS_HERA_PHAT_INCLUDE_DIRS "${MULTIPERS_HERA_SOURCE_DIR}/extern")
endif()

if(CGAL_FOUND)
  list(APPEND MULTIPERS_RHOMBOID_TILING_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS})
endif()

set(MULTIPERS_2PAC_SOURCE_DIR "${CMAKE_SOURCE_DIR}/ext/2pac" CACHE PATH "Path to a 2pac source tree")
set(MULTIPERS_2PAC_INCLUDE_DIRS "")
if(EXISTS "${MULTIPERS_2PAC_SOURCE_DIR}/matrices.hpp" AND EXISTS "${MULTIPERS_2PAC_SOURCE_DIR}/lw.cpp")
  set(MULTIPERS_2PAC_INCLUDE_DIRS "${MULTIPERS_2PAC_SOURCE_DIR}")
  add_library(
    multipers_2pac_static
    STATIC
    "${MULTIPERS_2PAC_SOURCE_DIR}/minimize.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/factor.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/chunk.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/lw.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/matrices.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/ArrayColumn.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/HeapColumn.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/time_measurement.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/block_column_matrix.cpp"
  )
  target_include_directories(multipers_2pac_static PUBLIC ${MULTIPERS_2PAC_INCLUDE_DIRS})
  target_link_libraries(multipers_2pac_static PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(multipers_2pac_static PUBLIC MULTIPERS_HAS_2PAC_INTERFACE=1)
  set_target_properties(
    multipers_2pac_static
    PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
  )
endif()

if(CGAL_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/ext/rhomboidtiling_newer_cgal_version/src/rhomboid.cpp" AND EXISTS "${CMAKE_SOURCE_DIR}/ext/rhomboidtiling_newer_cgal_version/src/utils.cpp")
  add_library(
    multipers_rhomboid_tiling_static
    STATIC
    "${CMAKE_SOURCE_DIR}/ext/rhomboidtiling_newer_cgal_version/src/rhomboid.cpp"
    "${CMAKE_SOURCE_DIR}/ext/rhomboidtiling_newer_cgal_version/src/utils.cpp"
  )
  target_include_directories(multipers_rhomboid_tiling_static PUBLIC ${MULTIPERS_RHOMBOID_TILING_INCLUDE_DIRS})
  target_link_libraries(multipers_rhomboid_tiling_static PUBLIC CGAL::CGAL)
  if(TARGET CGAL::CGAL_Core)
    target_link_libraries(multipers_rhomboid_tiling_static PUBLIC CGAL::CGAL_Core)
  endif()
endif()

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
  set_target_properties(
    multipers_aida_static
    PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
  )
  if(NOT MSVC)
    set_target_properties(multipers_aida_static PROPERTIES COMPILE_FLAGS "--no-warnings")
  endif()
endif()

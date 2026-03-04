include_guard(GLOBAL)

if(DEFINED ENV{CONDA_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(Boost REQUIRED COMPONENTS system timer chrono)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(TBB CONFIG REQUIRED COMPONENTS tbb)
find_package(CGAL QUIET COMPONENTS Core)

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

if(CGAL_FOUND)
  list(APPEND MULTIPERS_RHOMBOID_TILING_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS})
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
  set_target_properties(multipers_aida_static PROPERTIES COMPILE_FLAGS "--no-warnings")
endif()


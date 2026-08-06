include_guard(GLOBAL)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(TBB CONFIG REQUIRED COMPONENTS tbb)

set(_multipers_need_boost OFF)
foreach(_feature
    MULTIPERS_ENABLE_MPFREE
    MULTIPERS_ENABLE_MUPHASA
    MULTIPERS_ENABLE_FUNCTION_DELAUNAY
    MULTIPERS_ENABLE_GRAPHCODE
    MULTIPERS_ENABLE_DEG_RIPS
    MULTIPERS_ENABLE_MULTI_CRITICAL
    MULTIPERS_ENABLE_AIDA
    MULTIPERS_ENABLE_2PAC
    MULTIPERS_ENABLE_SKYSCRAPER
)
  if(${_feature})
    set(_multipers_need_boost ON)
  endif()
endforeach()
if(_multipers_need_boost)
  find_package(Boost REQUIRED COMPONENTS timer chrono)
else()
  find_package(Boost QUIET)
endif()

find_package(CGAL QUIET COMPONENTS Core)

set(_multipers_need_gmp OFF)
foreach(_feature
    MULTIPERS_ENABLE_MPFREE
    MULTIPERS_ENABLE_FUNCTION_DELAUNAY
    MULTIPERS_ENABLE_MULTI_CRITICAL
    MULTIPERS_ENABLE_RHOMBOID_TILING
    MULTIPERS_ENABLE_AIDA
)
  if(${_feature})
    set(_multipers_need_gmp ON)
  endif()
endforeach()
if(_multipers_need_gmp)
  find_library(MULTIPERS_GMP_LIBRARY REQUIRED NAMES gmp)
else()
  find_library(MULTIPERS_GMP_LIBRARY NAMES gmp)
endif()

file(GLOB _multipers_gudhi_module_include_dirs CONFIGURE_DEPENDS LIST_DIRECTORIES TRUE
  "${MULTIPERS_GUDHI_SOURCE_DIR}/src/*/include"
)
set(MULTIPERS_GUDHI_INCLUDE_DIRS "")
foreach(_include_dir IN LISTS _multipers_gudhi_module_include_dirs)
  if(IS_DIRECTORY "${_include_dir}")
    list(APPEND MULTIPERS_GUDHI_INCLUDE_DIRS "${_include_dir}")
  endif()
endforeach()
if(NOT EXISTS "${MULTIPERS_GUDHI_SOURCE_DIR}/src/Simplex_tree/include/gudhi/Simplex_tree.h")
  message(FATAL_ERROR "Missing Gudhi headers under ${MULTIPERS_GUDHI_SOURCE_DIR}. Run git submodule update --init ext/gudhi-devel")
endif()

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/ext/phat/include/phat/representations/bit_tree_pivot_column.h")
  message(FATAL_ERROR "Missing vanilla PHAT headers under ext/phat/include")
endif()
set(MULTIPERS_PHAT_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/ext/phat/include")

add_library(multipers_gudhi INTERFACE)
add_library(multipers::gudhi ALIAS multipers_gudhi)
target_include_directories(multipers_gudhi INTERFACE ${MULTIPERS_GUDHI_INCLUDE_DIRS})

add_library(multipers_phat INTERFACE)
add_library(multipers::phat ALIAS multipers_phat)
target_include_directories(multipers_phat SYSTEM INTERFACE ${MULTIPERS_PHAT_INCLUDE_DIRS})

add_library(multipers_openmp INTERFACE)
add_library(multipers::openmp ALIAS multipers_openmp)
target_link_libraries(multipers_openmp INTERFACE OpenMP::OpenMP_CXX)

add_library(multipers_tbb INTERFACE)
add_library(multipers::tbb ALIAS multipers_tbb)
target_link_libraries(multipers_tbb INTERFACE TBB::tbb)

add_library(multipers_boost INTERFACE)
add_library(multipers::boost ALIAS multipers_boost)
if(TARGET Boost::headers)
  target_link_libraries(multipers_boost INTERFACE Boost::headers)
elseif(TARGET Boost::boost)
  target_link_libraries(multipers_boost INTERFACE Boost::boost)
elseif(Boost_INCLUDE_DIRS)
  target_include_directories(multipers_boost SYSTEM INTERFACE ${Boost_INCLUDE_DIRS})
endif()
if(TARGET Boost::timer)
  target_link_libraries(multipers_boost INTERFACE Boost::timer)
endif()
if(TARGET Boost::chrono)
  target_link_libraries(multipers_boost INTERFACE Boost::chrono)
endif()

add_library(multipers_gmp INTERFACE)
add_library(multipers::gmp ALIAS multipers_gmp)
if(MULTIPERS_GMP_LIBRARY)
  target_link_libraries(multipers_gmp INTERFACE "${MULTIPERS_GMP_LIBRARY}")
endif()

add_library(multipers_cgal INTERFACE)
add_library(multipers::cgal ALIAS multipers_cgal)
if(CGAL_FOUND)
  target_link_libraries(multipers_cgal INTERFACE CGAL::CGAL)
  if(TARGET CGAL::CGAL_Core)
    target_link_libraries(multipers_cgal INTERFACE CGAL::CGAL_Core)
  endif()
endif()

cmake_minimum_required(VERSION 3.24)
project(multipers LANGUAGES C CXX)

if(CMAKE_GENERATOR STREQUAL "Unix Makefiles")
  find_program(MULTIPERS_MAKE_PROGRAM NAMES make gmake)
  if(MULTIPERS_MAKE_PROGRAM)
    set(CMAKE_MAKE_PROGRAM "${MULTIPERS_MAKE_PROGRAM}" CACHE FILEPATH "Make program for Unix Makefiles" FORCE)
  endif()
endif()

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:-fdiagnostics-color=always>
    $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>
    -g0
  )
endif()

if(MSVC)
  add_compile_options(/bigobj)
endif()

include("${CMAKE_SOURCE_DIR}/cmake/MultipersDependencies.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersCodegen.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersBuild.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersInstall.cmake")

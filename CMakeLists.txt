cmake_minimum_required(VERSION 3.27)
project(multipers LANGUAGES C CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  file(REMOVE "${CMAKE_SOURCE_DIR}/compile_commands.json")
  file(
    CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    SYMBOLIC
    COPY_ON_ERROR
  )
endif()

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

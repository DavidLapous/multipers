cmake_minimum_required(VERSION 3.27)
project(multipers LANGUAGES CXX)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

option(
  MULTIPERS_DEVELOPER_COMPILE_COMMANDS_LINK
  "Create a compile_commands.json symlink in the source tree"
  OFF
)
if(MULTIPERS_DEVELOPER_COMPILE_COMMANDS_LINK AND NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  file(REMOVE "${CMAKE_SOURCE_DIR}/compile_commands.json")
  file(
    CREATE_LINK
    "${CMAKE_BINARY_DIR}/compile_commands.json"
    "${CMAKE_SOURCE_DIR}/compile_commands.json"
    SYMBOLIC
    COPY_ON_ERROR
  )
endif()

include("${CMAKE_SOURCE_DIR}/cmake/MultipersOptions.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersToolchain.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersPython.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersPackages.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersCodegen.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersFeatureSummary.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersPatches.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersBackends.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersCore.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersExtensions.cmake")
include("${CMAKE_SOURCE_DIR}/cmake/MultipersInstall.cmake")

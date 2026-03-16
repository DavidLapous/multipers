include_guard(GLOBAL)

if(WIN32)
  if(DEFINED MULTIPERS_WINDOWS_RUNTIME_DEP_SET)
    install(
      TARGETS multipers_core_shared
      RUNTIME_DEPENDENCY_SET ${MULTIPERS_WINDOWS_RUNTIME_DEP_SET}
      LIBRARY DESTINATION multipers
      RUNTIME DESTINATION multipers
      ARCHIVE DESTINATION multipers
    )
  else()
    install(
      TARGETS multipers_core_shared
      LIBRARY DESTINATION multipers
      RUNTIME DESTINATION multipers
      ARCHIVE DESTINATION multipers
    )
  endif()
else()
  install(
    TARGETS multipers_core_shared
    LIBRARY DESTINATION multipers
  )
endif()

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/multipers/"
  DESTINATION multipers
  PATTERN ".DS_Store" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN "*.pyo" EXCLUDE
  PATTERN "*.dep" EXCLUDE
  PATTERN "*.pyx" EXCLUDE
  PATTERN "*.pxd" EXCLUDE
  PATTERN "*.tp" EXCLUDE
  PATTERN "*.cpp" EXCLUDE
  PATTERN "*.h" EXCLUDE
  PATTERN "*.hpp" EXCLUDE
)

set(MULTIPERS_PYTHON_EXECUTABLE "${Python3_EXECUTABLE}")
set(MULTIPERS_CMAKE_GENERATOR "${CMAKE_GENERATOR}")
configure_file("${CMAKE_SOURCE_DIR}/cmake/InstallWheel.cmake" "${CMAKE_BINARY_DIR}/InstallWheel.cmake" @ONLY)

# Only install the wheel-building script if we are NOT already building a wheel
# (avoids recursive loops with scikit-build-core/build/pip)
if(NOT SKBUILD AND NOT DEFINED ENV{MULTIPERS_INTERNAL_WHEEL_BUILD})
  install(SCRIPT "${CMAKE_BINARY_DIR}/InstallWheel.cmake")
endif()

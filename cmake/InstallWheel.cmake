set(_multipers_source_dir "@CMAKE_SOURCE_DIR@")
set(_multipers_python "@MULTIPERS_PYTHON_EXECUTABLE@")
set(_multipers_cmake "@CMAKE_COMMAND@")
set(_multipers_generator "@MULTIPERS_CMAKE_GENERATOR@")

message(STATUS "Building wheel before install.")
execute_process(
  COMMAND "${_multipers_cmake}" -E env MULTIPERS_INTERNAL_WHEEL_BUILD=1 "CMAKE_GENERATOR=${_multipers_generator}" "${_multipers_python}" -m build --wheel -n -Cbuild-dir=build/wheel .
  WORKING_DIRECTORY "${_multipers_source_dir}"
  RESULT_VARIABLE _multipers_build_result
)
if(NOT _multipers_build_result EQUAL 0)
  message(FATAL_ERROR "Failed to build wheel during install (exit code: ${_multipers_build_result}).")
endif()

file(GLOB _multipers_wheels "${_multipers_source_dir}/dist/multipers-*.whl")

if(NOT _multipers_wheels)
  message(FATAL_ERROR "No multipers wheel found in ${_multipers_source_dir}/dist after build.")
endif()

list(SORT _multipers_wheels)
list(LENGTH _multipers_wheels _multipers_wheel_count)
math(EXPR _multipers_wheel_index "${_multipers_wheel_count} - 1")
list(GET _multipers_wheels ${_multipers_wheel_index} _multipers_wheel)

message(STATUS "Installing wheel into current environment: ${_multipers_wheel}")
execute_process(
  COMMAND "${_multipers_python}" -m pip install --force-reinstall "${_multipers_wheel}"
  WORKING_DIRECTORY "${_multipers_source_dir}"
  RESULT_VARIABLE _multipers_pip_result
)
if(NOT _multipers_pip_result EQUAL 0)
  message(FATAL_ERROR "Failed to install wheel with pip (exit code: ${_multipers_pip_result}).")
endif()

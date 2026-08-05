include_guard(GLOBAL)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  RESULT_VARIABLE MULTIPERS_NANOBIND_CMAKE_DIR_RESULT
  OUTPUT_VARIABLE MULTIPERS_NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_VARIABLE MULTIPERS_NANOBIND_CMAKE_DIR_ERROR
)
if(NOT MULTIPERS_NANOBIND_CMAKE_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to locate nanobind CMake files: ${MULTIPERS_NANOBIND_CMAKE_DIR_ERROR}")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${MULTIPERS_NANOBIND_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)
set(MULTIPERS_NANOBIND_INCLUDE_DIR "${NB_DIR}/include")

add_library(multipers_python INTERFACE)
add_library(multipers::python ALIAS multipers_python)
if(TARGET Python::NumPy)
  target_link_libraries(multipers_python INTERFACE Python::NumPy)
else()
  target_include_directories(multipers_python INTERFACE ${Python_NumPy_INCLUDE_DIRS})
endif()
target_include_directories(multipers_python INTERFACE ${Python_INCLUDE_DIRS})

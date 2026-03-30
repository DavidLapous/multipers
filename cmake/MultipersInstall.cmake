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

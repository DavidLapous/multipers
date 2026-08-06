include_guard(GLOBAL)

get_property(MULTIPERS_EXTENSION_TARGETS GLOBAL PROPERTY MULTIPERS_EXTENSION_TARGETS)
set(_multipers_install_targets multipers_core_shared ${MULTIPERS_EXTENSION_TARGETS})

if(WIN32 AND DEFINED MULTIPERS_WINDOWS_RUNTIME_DEP_SET)
  install(
    TARGETS ${_multipers_install_targets}
    RUNTIME_DEPENDENCY_SET ${MULTIPERS_WINDOWS_RUNTIME_DEP_SET}
    LIBRARY DESTINATION multipers
    RUNTIME DESTINATION multipers
    ARCHIVE DESTINATION multipers
  )
else()
  install(
    TARGETS ${_multipers_install_targets}
    LIBRARY DESTINATION multipers
    RUNTIME DESTINATION multipers
    ARCHIVE DESTINATION multipers
  )
endif()

if(WIN32 AND DEFINED MULTIPERS_WINDOWS_RUNTIME_DEP_SET)
  set(_multipers_runtime_dependency_install_args
    DESTINATION "multipers"
    PRE_EXCLUDE_REGEXES
      [=[python[0-9]+\.dll]=]
      [=[vcruntime.*\.dll]=]
      [=[msvcp.*\.dll]=]
      [=[ucrtbase\.dll]=]
      [=[concrt.*\.dll]=]
    POST_EXCLUDE_REGEXES
      [=[.*[Ww]indows[/\\][Ss]ystem32[/\\]]=]
      [=[api-ms-win-.*]=]
      [=[ext-ms-.*]=]
  )
  if(MULTIPERS_WINDOWS_RUNTIME_DEP_DIRECTORIES)
    list(APPEND _multipers_runtime_dependency_install_args
      DIRECTORIES ${MULTIPERS_WINDOWS_RUNTIME_DEP_DIRECTORIES}
    )
  endif()
  install(
    RUNTIME_DEPENDENCY_SET ${MULTIPERS_WINDOWS_RUNTIME_DEP_SET}
    ${_multipers_runtime_dependency_install_args}
  )
endif()

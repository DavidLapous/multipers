if(NOT DEFINED REPO_ROOT)
  message(FATAL_ERROR "ApplyExtPatchOverlay.cmake needs REPO_ROOT")
endif()
if(NOT DEFINED OVERLAY_ROOT)
  message(FATAL_ERROR "ApplyExtPatchOverlay.cmake needs OVERLAY_ROOT")
endif()
if(NOT DEFINED LIBRARY_NAME)
  message(FATAL_ERROR "ApplyExtPatchOverlay.cmake needs LIBRARY_NAME")
endif()
if(NOT DEFINED LIBRARY_RELATIVE_ROOT)
  message(FATAL_ERROR "ApplyExtPatchOverlay.cmake needs LIBRARY_RELATIVE_ROOT")
endif()
if(NOT DEFINED SUBDIRS)
  message(FATAL_ERROR "ApplyExtPatchOverlay.cmake needs SUBDIRS")
endif()
if(NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "ApplyExtPatchOverlay.cmake needs PATCH_FILE")
endif()
if(NOT DEFINED PATCH_EXECUTABLE)
  message(FATAL_ERROR "ApplyExtPatchOverlay.cmake needs PATCH_EXECUTABLE")
endif()
if(NOT DEFINED STAMP_FILE)
  message(FATAL_ERROR "ApplyExtPatchOverlay.cmake needs STAMP_FILE")
endif()

file(REMOVE_RECURSE "${OVERLAY_ROOT}")

foreach(subdir IN LISTS SUBDIRS)
  set(_src "${REPO_ROOT}/${LIBRARY_RELATIVE_ROOT}/${subdir}")
  if(NOT EXISTS "${_src}")
    message(FATAL_ERROR "Failed to copy ${LIBRARY_NAME} subtree ${subdir} into patch overlay: source path ${_src} is missing")
  endif()

  get_filename_component(_subdir_parent "${subdir}" DIRECTORY)
  set(_dst_parent "${OVERLAY_ROOT}/${LIBRARY_RELATIVE_ROOT}")
  if(NOT "${_subdir_parent}" STREQUAL "")
    set(_dst_parent "${_dst_parent}/${_subdir_parent}")
  endif()
  file(MAKE_DIRECTORY "${_dst_parent}")
  file(COPY "${_src}" DESTINATION "${_dst_parent}")
endforeach()

execute_process(
  COMMAND "${PATCH_EXECUTABLE}" -p1 -i "${PATCH_FILE}"
  WORKING_DIRECTORY "${OVERLAY_ROOT}"
  RESULT_VARIABLE patch_result
  OUTPUT_VARIABLE patch_stdout
  ERROR_VARIABLE patch_stderr
)
if(NOT patch_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to apply ${LIBRARY_NAME} runtime log patch.\nstdout:\n${patch_stdout}\nstderr:\n${patch_stderr}")
endif()

file(WRITE "${STAMP_FILE}" "${LIBRARY_NAME} runtime log overlay ready\n")

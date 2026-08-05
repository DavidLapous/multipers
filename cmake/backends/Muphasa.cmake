include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(muphasa)
if(MULTIPERS_FEATURE_MUPHASA)
  multipers_backend_include(
    ${MULTIPERS_BACKEND_TARGET}
    "${MULTIPERS_MUPHASA_PATCH_OVERLAY_ROOT}/ext/muphasa/mph"
    "${MULTIPERS_MUPHASA_SOURCE_DIR}/mph"
  )
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers::boost)
  multipers_backend_depends(${MULTIPERS_BACKEND_TARGET} multipers_muphasa_log_overlay)
endif()

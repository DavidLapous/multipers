include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(deg_rips)
if(MULTIPERS_FEATURE_DEG_RIPS)
  multipers_backend_include(
    ${MULTIPERS_BACKEND_TARGET}
    "${MULTIPERS_DEG_RIPS_PATCH_OVERLAY_ROOT}/ext/deg_rips/include"
    "${CMAKE_SOURCE_DIR}/ext/deg_rips/include"
  )
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers::boost multipers::tbb)
  multipers_backend_depends(${MULTIPERS_BACKEND_TARGET} multipers_deg_rips_optimization_overlay)
endif()

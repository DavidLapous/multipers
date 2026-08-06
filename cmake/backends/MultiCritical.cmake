include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(multi_critical)
if(MULTIPERS_FEATURE_MULTI_CRITICAL)
  multipers_backend_include(
    ${MULTIPERS_BACKEND_TARGET}
    "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/include"
    "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/mpfree_mod/include"
    "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/mpp_utils_mod/include"
    "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/multi_chunk_mod/include"
    "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/phat_mod/include"
    "${MULTIPERS_MULTI_CRITICAL_PATCH_OVERLAY_ROOT}/ext/multi_critical/scc_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/multi_critical/include"
    "${CMAKE_SOURCE_DIR}/ext/multi_critical/mpfree_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/multi_critical/mpp_utils_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/multi_critical/multi_chunk_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/multi_critical/phat_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/multi_critical/scc_mod/include"
  )
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers::boost multipers::gmp multipers::openmp multipers::tbb)
  multipers_backend_depends(${MULTIPERS_BACKEND_TARGET} multipers_multi_critical_log_overlay)
endif()

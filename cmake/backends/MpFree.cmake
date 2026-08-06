include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(mpfree)
if(MULTIPERS_FEATURE_MPFREE)
  multipers_backend_include(
    ${MULTIPERS_BACKEND_TARGET}
    "${MULTIPERS_MPFREE_PATCH_OVERLAY_ROOT}/ext/mpfree/include"
    "${CMAKE_SOURCE_DIR}/ext/mpfree/include"
    "${CMAKE_SOURCE_DIR}/ext/mpfree/mpp_utils_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/mpfree/phat_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/mpfree/scc_mod/include"
  )
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers::boost multipers::gmp multipers::openmp multipers::tbb)
  multipers_backend_depends(${MULTIPERS_BACKEND_TARGET} multipers_mpfree_log_overlay)
endif()

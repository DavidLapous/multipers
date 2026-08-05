include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(function_delaunay)
if(MULTIPERS_FEATURE_FUNCTION_DELAUNAY)
  multipers_backend_include(
    ${MULTIPERS_BACKEND_TARGET}
    "${MULTIPERS_FUNCTION_DELAUNAY_PATCH_OVERLAY_ROOT}/ext/function_delaunay/include"
    "${CMAKE_SOURCE_DIR}/ext/function_delaunay/include"
    "${MULTIPERS_FUNCTION_DELAUNAY_PATCH_OVERLAY_ROOT}/ext/function_delaunay/mpfree_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/function_delaunay/mpfree_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/function_delaunay/mpp_utils_mod/include"
    "${MULTIPERS_FUNCTION_DELAUNAY_PATCH_OVERLAY_ROOT}/ext/function_delaunay/multi_chunk_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/function_delaunay/multi_chunk_mod/include"
    "${CMAKE_SOURCE_DIR}/ext/function_delaunay/phat/include"
    "${CMAKE_SOURCE_DIR}/ext/function_delaunay/scc_mod/include"
  )
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers::boost multipers::gmp multipers::openmp multipers::tbb)
  multipers_backend_depends(${MULTIPERS_BACKEND_TARGET} multipers_function_delaunay_log_overlay)
endif()

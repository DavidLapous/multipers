include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(graphcode)
if(MULTIPERS_FEATURE_GRAPHCODE)
  multipers_backend_include(
    ${MULTIPERS_BACKEND_TARGET}
    "${MULTIPERS_GRAPHCODE_SOURCE_DIR}/include"
    "${MULTIPERS_GRAPHCODE_SOURCE_DIR}/mpp_utils_mod/include"
    "${MULTIPERS_GRAPHCODE_SOURCE_DIR}/phat_mod/include"
  )
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers::boost)
endif()

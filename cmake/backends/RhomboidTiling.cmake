include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(rhomboid_tiling)
if(MULTIPERS_FEATURE_RHOMBOID_TILING)
  set(_multipers_rhomboid_root "${CMAKE_SOURCE_DIR}/ext/rhomboidtiling_newer_cgal_version")
  add_library(
    multipers_rhomboid_tiling_static
    STATIC
    "${_multipers_rhomboid_root}/src/rhomboid.cpp"
    "${_multipers_rhomboid_root}/src/utils.cpp"
  )
  target_include_directories(multipers_rhomboid_tiling_static PUBLIC "${_multipers_rhomboid_root}/src")
  target_link_libraries(multipers_rhomboid_tiling_static PUBLIC multipers::cgal)
  multipers_apply_common_build_flags(multipers_rhomboid_tiling_static)
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers_rhomboid_tiling_static multipers::gmp multipers::tbb multipers::cgal)
  multipers_backend_include(${MULTIPERS_BACKEND_TARGET} "${_multipers_rhomboid_root}/src")
endif()

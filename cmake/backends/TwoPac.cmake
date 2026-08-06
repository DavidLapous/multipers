include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(2pac)
if(MULTIPERS_FEATURE_2PAC)
  set(_multipers_2pac_sources
    "${MULTIPERS_2PAC_SOURCE_DIR}/minimize.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/factor.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/chunk.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/Cone.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/complexes.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/computation.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/bireductions.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/lw.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/matrices.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/reductions.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/relative_cohomology.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/ArrayColumn.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/HeapColumn.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/time_measurement.cpp"
    "${MULTIPERS_2PAC_SOURCE_DIR}/block_column_matrix.cpp"
  )
  add_library(multipers_2pac_static STATIC ${_multipers_2pac_sources})
  target_include_directories(multipers_2pac_static PUBLIC "${MULTIPERS_2PAC_SOURCE_DIR}")
  target_link_libraries(multipers_2pac_static PUBLIC multipers::boost multipers::openmp)
  multipers_apply_common_build_flags(multipers_2pac_static)
  set_target_properties(multipers_2pac_static PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers_2pac_static)
  multipers_backend_include(${MULTIPERS_BACKEND_TARGET} "${MULTIPERS_2PAC_SOURCE_DIR}")
endif()

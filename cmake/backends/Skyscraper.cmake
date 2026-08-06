include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(skyscraper)
if(MULTIPERS_FEATURE_SKYSCRAPER)
  add_library(
    multipers_skyscraper_core
    STATIC
    "${MULTIPERS_SKYSCRAPER_SOURCE_DIR}/src/skyscraper_core.cpp"
    "${MULTIPERS_SKYSCRAPER_SOURCE_DIR}/src/uni_b1.cpp"
    "${MULTIPERS_SKYSCRAPER_SOURCE_DIR}/src/hnf_at.cpp"
  )
  target_include_directories(
    multipers_skyscraper_core
    PUBLIC
      "${MULTIPERS_SKYSCRAPER_SOURCE_DIR}/include"
      "${CMAKE_SOURCE_DIR}/ext/Persistence-Algebra/include"
  )
  target_link_libraries(multipers_skyscraper_core PUBLIC multipers::boost)
  multipers_apply_common_build_flags(multipers_skyscraper_core)
  set_target_properties(multipers_skyscraper_core PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers_skyscraper_core)
  multipers_backend_include(
    ${MULTIPERS_BACKEND_TARGET}
    "${MULTIPERS_SKYSCRAPER_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/ext/Persistence-Algebra/include"
  )
endif()

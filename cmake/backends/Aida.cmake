include("${CMAKE_CURRENT_LIST_DIR}/BackendHelpers.cmake")
multipers_create_backend(aida)
if(MULTIPERS_FEATURE_AIDA)
  add_library(
    multipers_aida_static
    STATIC
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/aida_decompose.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/aida_functions.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/aida_helpers.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/aida_interface.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/config.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/option_parser.cpp"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src/block.cpp"
  )
  target_include_directories(
    multipers_aida_static
    PUBLIC
      "${CMAKE_SOURCE_DIR}/ext/AIDA/include"
      "${CMAKE_SOURCE_DIR}/ext/AIDA/src"
      "${CMAKE_SOURCE_DIR}/ext/Persistence-Algebra/include"
  )
  target_link_libraries(multipers_aida_static PUBLIC multipers::boost multipers::gmp multipers::openmp multipers::tbb)
  multipers_apply_common_build_flags(multipers_aida_static)
  if(NOT MSVC)
    target_compile_options(multipers_aida_static PRIVATE -w)
  endif()
  set_target_properties(multipers_aida_static PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  target_link_libraries(${MULTIPERS_BACKEND_TARGET} INTERFACE multipers_aida_static)
  multipers_backend_include(
    ${MULTIPERS_BACKEND_TARGET}
    "${CMAKE_SOURCE_DIR}/ext/AIDA/include"
    "${CMAKE_SOURCE_DIR}/ext/AIDA/src"
    "${CMAKE_SOURCE_DIR}/ext/Persistence-Algebra/include"
  )
endif()

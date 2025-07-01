add_library(unit_tests SHARED)

target_include_directories(unit_tests PUBLIC ../)

target_sources(unit_tests
  PRIVATE    
    box_overlap.cpp
    #occlusion_detection_test.cpp
    pn_differ_test.cpp
    pn_reasoning_test.cpp
    pn_util.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    pn_util.hpp
)

target_link_libraries(unit_tests
    PRIVATE
        state_observation
        hand-pose-estimation
)

target_compile_features(unit_tests PUBLIC cxx_std_20)

LinkAssets(unit_tests)
CopyRuntimeDeps(unit_tests)

install(
  TARGETS 
    unit_tests
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/unit_tests
)
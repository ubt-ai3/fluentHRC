add_executable(sample_data_registration)

target_include_directories(sample_data_registration PUBLIC ../)

target_sources(sample_data_registration
  PRIVATE    
    #ColorBasics.cpp
    hand_evaluation.cpp
    ImageRenderer.cpp
    object_evaluation.cpp
    projection_matrix.cpp
    #registration_tester.cpp
    main.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    #ColorBasics.h
    hand_evaluation.hpp
    ImageRenderer.h
    object_evaluation.hpp
    projection_matrix.hpp
    #registration_tester.hpp
    resource.h
    stdafx.h
)

find_package(Boost REQUIRED COMPONENTS date_time)

target_link_libraries(sample_data_registration
    PRIVATE
        Boost::date_time
        hand-pose-estimation
        state_observation
        csv_reader
)

target_compile_features(sample_data_registration PUBLIC cxx_std_20)

LinkAssets(sample_data_registration)
CopyRuntimeDeps(sample_data_registration)
FixExecutionContext(sample_data_registration)

install(
  TARGETS 
    sample_data_registration
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/sample_data_registration
)
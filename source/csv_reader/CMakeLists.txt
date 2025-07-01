add_library(csv_reader)

target_include_directories(csv_reader PUBLIC ../)

target_sources(csv_reader
  PRIVATE    
    csvworker.cpp
    tracker.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    csvworker.h
    tracker.hpp
)

find_package(Eigen3 CONFIG REQUIRED)

target_link_libraries(csv_reader
    PUBLIC
        Eigen3::Eigen
)

target_compile_features(csv_reader PUBLIC cxx_std_20)

install(
  TARGETS 
    csv_reader
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/csv_reader
)
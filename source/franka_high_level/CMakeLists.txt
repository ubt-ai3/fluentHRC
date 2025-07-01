add_library(franka_high_level)

target_include_directories(franka_high_level PUBLIC ../)

target_sources(franka_high_level
  PRIVATE    
    franka_actor.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    franka_actor.h
#    ${INCLUDE_DIR}/ImportExport.h
)

find_package(franka-proxy CONFIG REQUIRED)

target_compile_features(franka_high_level PUBLIC cxx_std_20)
target_link_libraries(franka_high_level
  PRIVATE
    state_observation
    franka_proxy_client
    franka_proxy_share
    eigen_serialization
)

install(
  TARGETS 
    franka_high_level
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/franka_high_level
)
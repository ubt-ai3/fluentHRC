add_library(franka_voxel)

target_include_directories(franka_voxel PUBLIC ../)

target_sources(franka_voxel
  PRIVATE    
    motion_generator_joint_max_accel.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    motion_generator_joint_max_accel.h
)

find_package(franka-proxy CONFIG REQUIRED)

target_link_libraries(franka_voxel
    PRIVATE
        franka_proxy_share
)

target_compile_features(franka_voxel PUBLIC cxx_std_20)

install(
  TARGETS 
    franka_voxel
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/franka_voxel
)
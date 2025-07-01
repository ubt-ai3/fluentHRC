add_library(grpc_server)

target_include_directories(grpc_server PUBLIC ../)

target_sources(grpc_server
  PRIVATE      
    point_cloud_processing.cpp
    proto_plugin.cpp
    server_module.cpp
    service_impl.cpp
    util.cpp
    #visualizer.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    global.hpp
    point_cloud_processing.h
    proto_plugin.h
    server_module.h
    service_impl.h
    util.h
    #visualizer.h
    wrapper.hpp
)

find_package(base-transformation CONFIG REQUIRED)
find_package(PCL COMPONENTS registration)

target_link_libraries(grpc_server
    PUBLIC
        hand-pose-estimation
        state_observation
        proto-objects
        base-transformation::base-transformation
        base-transformation::plugins
        ${PCL_LIBRARIES}
)

target_compile_features(grpc_server PUBLIC cxx_std_20)

install(
  TARGETS 
    grpc_server
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/grpc_server
)
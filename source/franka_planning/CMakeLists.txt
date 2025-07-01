add_library(franka_planning)

target_include_directories(franka_planning PUBLIC ../)
target_compile_definitions(franka_planning PUBLIC GLOG_NO_ABBREVIATED_SEVERITIES)

target_sources(franka_planning
  PRIVATE    
    franka_actor.cpp
    robot_agent.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    franka_actor.hpp
    robot_agent.hpp
)

find_package(enact REQUIRED)
find_package(Boost REQUIRED COMPONENTS date_time)

target_link_libraries(franka_planning
  PRIVATE
    Boost::date_time
    intention_prediction
  PUBLIC
    state_observation
)

target_compile_features(franka_planning PUBLIC cxx_std_20)

install(
  TARGETS 
    franka_planning
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/franka_planning
)
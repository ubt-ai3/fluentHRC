add_library(simulation SHARED)

target_include_directories(simulation PUBLIC ../)
target_compile_definitions(simulation PRIVATE SIMULATION_EXPORTS)

target_sources(simulation
  PRIVATE    
    baraglia17.cpp
    behavior_test.cpp
    build_test.cpp
    #building.cpp
    franka_actor_sim.cpp
    hoellerich22.cpp
    benchmark.cpp
    mogaze.cpp
    rendering.cpp
    riedelbauch17.cpp
    scene.cpp
    #simulation_io.cpp
    task.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    baraglia17.hpp
    behavior_test.hpp
    build_test.hpp
    benchmark.hpp
    #building.hpp
    franka_actor_sim.h
    hoellerich22.hpp
    mogaze.hpp
    rendering.hpp
    riedelbauch17.hpp
    scene.hpp
    #simulation_io.hpp
    task.hpp
)

find_package(OpenGL REQUIRED)
find_package(FreeGLUT CONFIG REQUIRED)
find_package(franka-proxy CONFIG REQUIRED)
find_package(PCL COMPONENTS io simulation)

target_link_libraries(simulation
    PUBLIC
        state_observation
        franka_planning
        OpenGL::GL
        $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static>
        ${PCL_LIBRARIES}
    PRIVATE
        franka_proxy_share
        franka_voxel
)

target_compile_features(simulation PUBLIC cxx_std_20)

install(
  TARGETS 
    simulation
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/simulation
)
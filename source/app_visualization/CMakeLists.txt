add_executable(app_visualization)

target_include_directories(app_visualization PUBLIC ../)

target_sources(app_visualization
  PRIVATE
    franka_actor.cpp
    franka_visualization.cpp
    franka_visualization_gpu.cu
    intention_visualizer.cpp
    module_manager.cpp
    mogaze.cpp
    presenter.cpp
    #registration.cpp
    simulated_hand_tracking.cpp
    task_description.cpp
    task_progress_visualizer.cpp
    util.cpp
    viewer.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    franka_actor.h
    franka_visualization.h
    franka_visualization_gpu.h
    intention_visualizer.hpp
    module_manager.hpp
    mogaze.hpp
    presenter.hpp
    signal_types.hpp
    simulated_hand_tracking.hpp
    task_description.hpp
    task_progress_visualizer.hpp
    util.hpp
    viewer.hpp
)

find_package(gpu_voxels CONFIG REQUIRED)
find_package(icl_core CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Boost COMPONENTS timer)
find_package(tinyobjloader CONFIG REQUIRED)

set_property(TARGET app_visualization PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET app_visualization PROPERTY CUDA_ARCHITECTURES all)
set_property(TARGET app_visualization PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_compile_options(app_visualization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                       -g
#                       -G
#                       >)

target_link_libraries(app_visualization
    PRIVATE
        gpu_voxels
    PUBLIC
        state_observation
        simulation
        franka_planning
        franka_voxel
        hand-pose-estimation
        intention_prediction
        franka_high_level
        grpc_server
        Boost::timer
        tinyobjloader::tinyobjloader
)

target_compile_features(app_visualization PUBLIC cxx_std_20)

LinkAssets(app_visualization)
CopyRuntimeDeps(app_visualization)
FixExecutionContext(app_visualization)

#CopyCudnn(app_visualization)

#set_target_properties(app_visualization PROPERTIES VS_DEBUGGER_ENVIRONMENT "GPU_VOXELS_MODEL_PATH=./assets/models")

install(
  TARGETS 
    app_visualization
)
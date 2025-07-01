add_library(state_observation SHARED)

target_include_directories(state_observation PUBLIC ../)
target_compile_definitions(state_observation PRIVATE STATEOBSERVATION_EXPORTS)

target_sources(state_observation
  PRIVATE    
    building_estimation.cpp
    calibration.cpp
    classification_handler.cpp
    classification_new.cpp
    classification.cpp
    color_space_conversion.cpp
    #module_manager.cpp
    object_detection.cpp
    object_prototype_loader.cpp
    object_tracking.cpp
    parameter_set.cpp
    pn_model_extension.cpp
    pn_model.cpp
    pn_reasoning.cpp
    pn_world_traceability.cpp
    pointcloud_util.cpp
    workspace_calibration.cpp
    workspace_objects.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    building_estimation.hpp
    calibration.hpp
    classification_handler.hpp
    classification_new.hpp
    classification.hpp
    color_space_conversion.hpp
    framework.hpp
    object_detection.hpp
    object_prototype_loader.hpp
    object_tracking.hpp
    parameter_set.hpp
    pn_model_extension.hpp
    pn_model.hpp
    pn_reasoning.hpp
    pn_world_traceability.hpp
    pointcloud_util.hpp
    workspace_calibration.h
    workspace_objects_forward.hpp
    workspace_objects.hpp
#    ${INCLUDE_DIR}/ImportExport.h
)

find_package(Eigen3 CONFIG REQUIRED)
find_package(enact CONFIG REQUIRED)
find_package(PCL COMPONENTS common kdtree visualization filters segmentation surface features)
find_package(OpenCV CONFIG REQUIRED)
find_package(unofficial-kinectsdk2 CONFIG REQUIRED)
find_package(unofficial-kinectsdk1 CONFIG REQUIRED)
find_package(VTK CONFIG REQUIRED)

target_link_libraries(state_observation
  PRIVATE 
    
  PUBLIC
    unofficial::kinectsdk2::kinectsdk2
    eigen_serialization 
    enact::enact_core
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    enact::enact_priority
    ${OpenCV_LIBS}
    unofficial::kinectsdk1::kinectsdk1
)

target_compile_features(state_observation PUBLIC cxx_std_20)

install(
  TARGETS 
    state_observation
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/state_observation
)
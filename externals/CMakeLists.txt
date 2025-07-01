add_library(eigen_serialization INTERFACE)
target_sources(eigen_serialization
    INTERFACE
        eigen_serialization/eigen_serialization.hpp
)
target_include_directories(eigen_serialization INTERFACE
    ./
)

#### hand-pose-estimation

add_library(hand-pose-estimation SHARED)

set(INC_DIR hand-pose-estimation/source/hand_pose_estimation)

target_include_directories(hand-pose-estimation PUBLIC hand-pose-estimation/source)
target_compile_definitions(hand-pose-estimation 
    PRIVATE HANDPOSEESTIMATION_EXPORTS
    PUBLIC GLOG_NO_ABBREVIATED_SEVERITIES
    )

target_sources(hand-pose-estimation
  PRIVATE    
    ${INC_DIR}/bounding_box_tracking.cpp
    ${INC_DIR}/classification_handler.cpp
    ${INC_DIR}/classification.cpp
    ${INC_DIR}/color_space_conversion.cpp
    ${INC_DIR}/finger_detection.cpp
    ${INC_DIR}/gradient_decent.cpp
    ${INC_DIR}/hand_model.cpp
    ${INC_DIR}/hand_pose_estimation.cpp
    ${INC_DIR}/hand_tracker_enact.cpp
    ${INC_DIR}/hand_tracker.cpp
    ${INC_DIR}/parameter_set.cpp
    #${INC_DIR}/particle_swarm_filter.cpp
    ${INC_DIR}/ra_point_cloud_classifier.cpp
    ${INC_DIR}/ra_skin_color_detector.cpp
    ${INC_DIR}/skin_detection.cpp
    ${INC_DIR}/utility.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    ${INC_DIR}/bounding_box_tracking.hpp
    ${INC_DIR}/classification_handler.hpp
    ${INC_DIR}/classification.hpp
    ${INC_DIR}/color_space_conversion.hpp
    ${INC_DIR}/finger_detection.hpp
    ${INC_DIR}/framework.h
    ${INC_DIR}/gradient_decent.hpp
    ${INC_DIR}/hand_model.hpp
    ${INC_DIR}/hand_pose_estimation.h
    ${INC_DIR}/hand_tracker_enact.hpp
    ${INC_DIR}/hand_tracker.hpp
    ${INC_DIR}/hololens_hand_data.hpp
    ${INC_DIR}/parameter_set.hpp
    #${INC_DIR}/particle_swarm_filter.hpp
    ${INC_DIR}/ra_point_cloud_classifier.hpp
    ${INC_DIR}/ra_skin_color_detector.hpp
    ${INC_DIR}/skin_detection.hpp
    ${INC_DIR}/utility.hpp
)

#find_package(Eigen3 REQUIRED)
find_package(Caffe CONFIG REQUIRED)
find_package(enact REQUIRED)
find_package(PCL COMPONENTS common search filters segmentation)
find_package(OpenCV REQUIRED)
#find_package(unofficial-kinectsdk2 CONFIG REQUIRED)
#find_package(unofficial-kinectsdk1 CONFIG REQUIRED)

target_link_libraries(hand-pose-estimation
  #PRIVATE 
  #  Eigen3::Eigen enact::enact_priority
  #  ${PCL_LIBRARIES}
  #  ${OpenCV_LIBS}
  #  unofficial::kinectsdk2::kinectsdk2
  #  unofficial::kinectsdk1::kinectsdk1
  PUBLIC
  #  eigen_serialization 
    enact::enact_core enact::enact_priority caffe ${OpenCV_LIBS} ${PCL_LIBRARIES}
)

target_compile_features(hand-pose-estimation PUBLIC cxx_std_20)
#target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_CORE_)

install(
  TARGETS 
    hand-pose-estimation
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/hand-pose-estimation
)
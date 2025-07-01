#pragma once

#ifndef SAMPLE_DATA_REGISTRATION__OBJECT_EVALUATION
#define SAMPLE_DATA_REGISTRATION__OBJECT_EVALUATION

#include <fstream>

#include <pcl/point_types.h>

#include "state_observation/workspace_objects.hpp"
#include "state_observation/pointcloud_util.hpp"
#include "state_observation/object_detection.hpp"
#include "state_observation/classification_new.hpp"
//#include "state_observation/classification.hpp"
#include "state_observation/calibration.hpp"

class object_evaluation
{
public:
	typedef pcl::PointXYZRGBA PointT;
	typedef std::shared_ptr<enact_core::entity_id> entity_id;
	typedef enact_core::lockable_data_typed<state_observation::object_instance> object_instance_data;

	const state_observation::classifier classifier_;

	object_evaluation(state_observation::pointcloud_preprocessing& pc_prepro,
	                  state_observation::segment_detector& obj_detect,
	                  state_observation::kinect2_parameters& kinect2_params);

	void update(const pcl::PointCloud<PointT>::ConstPtr& cloud,
				const cv::Mat& img);



private:
	state_observation::pointcloud_preprocessing& pc_prepro;
	state_observation::segment_detector& obj_detect;
	state_observation::kinect2_parameters& kinect2_params;
	std::string file_name;
	std::fstream output;

};

#endif // !SAMPLE_DATA_REGISTRATION__OBJECT_EVALUATION

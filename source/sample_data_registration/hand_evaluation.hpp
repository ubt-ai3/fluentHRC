#pragma once

#ifndef SAMPLE_DATA_REGISTRATION__EVALUATION
#define SAMPLE_DATA_REGISTRATION__EVALUATION

#include <fstream>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "hand_pose_estimation/hand_model.hpp"
#include "hand_pose_estimation/classification_handler.hpp"

#include "csv_reader/tracker.hpp"

class hand_evaluator
{
public:
	typedef pcl::PointXYZRGBA PointT;

	hand_evaluator(tracker& track,
		const hand_pose_estimation::classifier_set& gesture_classifiers,
		std::string output_path = std::string(),
		float certainty_threshold = 0.8f,
		float distance_threshold = 0.15f);

	~hand_evaluator();

	void update(const hand_pose_estimation::visual_input::ConstPtr& input, 
		float time);

	void close() noexcept;

private:
	tracker opti_track;
	const hand_pose_estimation::classifier_set& gesture_classifiers;

	std::fstream output;

	float certainty_threshold;
	float distance_threshold;

	float latest_time;

	std::map<hand_type, size_t> mapping;
	std::map<hand_type, float> mapping_start;
};

#endif // !SAMPLE_DATA_REGISTRATION__EVALUATION

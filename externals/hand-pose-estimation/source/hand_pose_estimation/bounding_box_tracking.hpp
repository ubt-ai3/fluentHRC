#pragma once

#include "framework.h"

#include <string>

#include <opencv2/opencv.hpp>

#include "hand_model.hpp"

namespace hand_pose_estimation
{

/**
 * @class bounding_box_tracker
 * @brief Multi-object tracking system with occlusion handling
 *
 * Implements a robust tracking system for multiple objects that can handle
 * occlusions, disappearances, and reappearances. Uses bounding box tracking
 * with position and shape comparison to maintain object correspondences.
 *
 * Features:
 * - Multi-object tracking
 * - Occlusion handling
 * - Object correspondence management
 * - Position extrapolation
 * - Shape comparison
 * - Background plane handling
 * - Thread-safe operation
 * 
 * Key Functionality:
 * - Updates correspondences between previous and new bounding boxes
 * - Handles object spawning and purging
 * - Performs position and shape-based matching
 * - Supports background plane constraints
 * - Provides timestamp tracking
 */
class bounding_box_tracker
{
public:
	HANDPOSEESTIMATION_API bounding_box_tracker(float certainty_threshold = 0.3f,
				std::chrono::duration<float> purge_duration = std::chrono::duration<float>(3.f),
				float smoothing_factor = 2.f,
				bool spawn_hands = true);

	HANDPOSEESTIMATION_API ~bounding_box_tracker();

	/**
	* Updates the correspondences of previous and new bounding boxes.
	*/
	HANDPOSEESTIMATION_API void update(const visual_input& input,
			    const std::vector<img_segment::Ptr>& objects);

	HANDPOSEESTIMATION_API void add_hand(const std::shared_ptr<hand_instance>& hand);


	/**
	* Deletes all tracking information and starts from scratch.
	*/
	HANDPOSEESTIMATION_API void reset();


	HANDPOSEESTIMATION_API std::vector<hand_instance::Ptr> get_hands() const;

	HANDPOSEESTIMATION_API std::chrono::duration<float> get_latest_timestamp() const;

	/*
	 * Assumes that @param{plan} normal points towards camera
	 * All cloud points below the plane are excluded from hand segments
	 */
	HANDPOSEESTIMATION_API void set_background_plane(Eigen::Hyperplane<float, 3> plane);

	const hand_kinematic_parameters hand_kin_params;

private:
	std::list<hand_instance::Ptr> hands;

	float certainty_threshold;
	std::chrono::duration<float> purge_duration;
	float smoothing_factor;
	std::chrono::duration<float> timestamp;

	bool spawn_hands;

	std::thread helper;
	std::mutex helper_mutex;
	std::condition_variable helper_wake_up;
	const visual_input* helper_visual_input;
	std::vector<img_segment::Ptr> helper_input;
	std::vector<img_segment::Ptr> helper_output;
	std::atomic_bool helper_sleep;
	std::atomic_bool helper_terminate;
	

	Eigen::Hyperplane<float, 3> plane = Eigen::Hyperplane<float, 3>(Eigen::Vector3f::Zero(), 0.f);

	/*
	* Turns an image segment into one or more segments based on depth differences
	*/
	std::vector<img_segment::Ptr> subsegment(const visual_input& input, 
		const img_segment::Ptr seg, 
		int max_clusters);

	/*
	* Linear extrapolition of the bounding box of the hand using the last 
	* two observations
	*/
	cv::Rect2i extrapolate_pose(const visual_input& input, const hand_instance& hand) const;

	/**
	* Returns the likelihood that @ref{hand} is at the position of @ref{seg},
	* assuming linear motion and @ref{seg} is from the next frame.
	*/
	float compare_position(const visual_input& input, const hand_instance& hand, const img_segment& seg) const;
	/**
	* Returns the similarity of the last snapshot of hand and seg.
	* Uses Hu moments.
	*/
	float compare_shape(const hand_instance& hand, const img_segment& seg) const;


	float bell_curve(float x, float stdev) const;
};

} /* hand_pose_estimation */



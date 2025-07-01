#pragma once

#include "framework.h"

#include <thread>

#include "hand_model.hpp"
#include "ra_skin_color_detector.hpp"
#include "bounding_box_tracking.hpp"
#include "gradient_decent.hpp"

namespace  hand_pose_estimation
{
/**
 * @class hand_tracker
 * @brief Main API for asynchronous hand tracking and pose estimation
 *
 * Provides a comprehensive interface for tracking hands and estimating their poses
 * in real-time. Processes new images asynchronously and maintains a list of tracked
 * hands with their pose histories.
 *
 * Features:
 * - Asynchronous hand tracking
 * - Real-time pose estimation
 * - Background plane handling
 * - Hand orientation preferences
 * - Thread-safe hand access
 * - Configurable processing parameters
 * - Skin region visualization
 * 
 * Usage:
 * - New images are provided via the update method
 * - Tracked hands can be retrieved using get_hands
 * - For thread-safe access, lock update_mutex before accessing hand data
 * - The last pose in the list is the latest but may be subject to refinement
 * - It is recommended to use the second latest pose for stable results
 */
class hand_tracker
{
public:
	/*
	 * purge_duration - if hand is not detected for more than purge_duration seconds, it is removed
	 * min_hand_area \in [0,1] - chunks smaller than min_hand_area are not further processed, relative to full image
	 *
	 * max_threads:
	 *	0  - update() is synchronous
	 *	1-2  - update() is async but each frame is processed until the last step achieves its best result
	 *	3+ - full anytime capabilities
	 *
	 *	backgound - a solid surface which hands cannot pierce, see set_background
	 *	back_palm_orientation - default hand orientation, see set_back_palm_orientation 
	 */
	HANDPOSEESTIMATION_API hand_tracker(std::chrono::duration<float> purge_duration = std::chrono::duration<float>(3.f),
		int max_threads = std::thread::hardware_concurrency(), 
		bool spawn_hands = true,
		Eigen::Vector3f back_palm_orientation = Eigen::Vector3f(0.f,0.f,0.f),
		Eigen::Hyperplane<float, 3> background = Eigen::Hyperplane<float, 3>(Eigen::Vector3f::Zero(), 0.f));

	HANDPOSEESTIMATION_API virtual ~hand_tracker();

	// async
	HANDPOSEESTIMATION_API void update(const visual_input::ConstPtr& input);
	
	HANDPOSEESTIMATION_API std::vector<hand_instance::Ptr> get_hands() const;
	HANDPOSEESTIMATION_API std::vector<hand_instance::Ptr> wait_for_new_hands();
	HANDPOSEESTIMATION_API void wait_for_batch_completed();

	HANDPOSEESTIMATION_API const hand_kinematic_parameters& get_hand_kinematic_parameters() const;

	HANDPOSEESTIMATION_API float get_hand_certainty_threshold() const;

	/*
	 * Defines a plane which hands cannot pierce. Points behind this plane are removed from hand candidates,
	 *				normal must point towards origin (= camera position)
	 *
	 */
	HANDPOSEESTIMATION_API void set_background(const Eigen::Hyperplane<float, 3>& plane);

	/*
	 * Sets the preferable direction of the back hand palm normal vector.
	 * 	Eigen::Vector3f(0.f,0.f,-1.f) - prefer back hand facing camera
	 *	Eigen::Vector3f(0.f,0.f,1.f) - prefer front hand facing camera
	 *	Eigen::Vector3f(0.f,0.f,0.f) - all orientations equally feasible
	 */
	HANDPOSEESTIMATION_API void set_back_palm_orientation(const Eigen::Vector3f& normal);

	HANDPOSEESTIMATION_API void show_skin_regions();
	
protected:
	skin_detector skin_detector;	
	bounding_box_tracker bb_tracker;
	gradient_decent_scheduler gd_scheduler;

	visual_input::ConstPtr input;
	std::set<hand_instance::Ptr> hands;
	std::vector<hand_instance::Ptr> new_hands;

	mutable std::mutex update_mutex;
	std::thread internal_thread;
	std::atomic_bool terminate_flag;
	std::condition_variable new_input_condition;
	std::condition_variable new_hands_condition;
	std::condition_variable batch_completion_condition;

	virtual void process_input();
};
}

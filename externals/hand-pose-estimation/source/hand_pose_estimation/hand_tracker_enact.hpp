#pragma once

#include <chrono>

#include "framework.h"

#include "hand_model.hpp"
#include "hand_tracker.hpp"

#include "enact_core/data.hpp"
#include "enact_core/world.hpp"
#include "enact_core/id.hpp"
#include "enact_priority/signaling_actor.hpp"

namespace hand_pose_estimation
{

/**
 * @class hand_trajectory
 * @brief A lightweight representation of hand movement over time
 *
 * Tracks the trajectory of a hand's movement by storing a sequence of poses
 * with timestamps. Provides functionality for updating and transforming hand poses
 * while maintaining certainty scores and hand identification.
 *
 * Features:
 * - Pose sequence tracking
 * - Timestamp management
 * - Certainty scoring
 * - Hand identification (left/right)
 * - Pose transformation support
 */
class HANDPOSEESTIMATION_API hand_trajectory
{
public:
	typedef std::shared_ptr<hand_trajectory> Ptr;
	typedef std::shared_ptr<const hand_trajectory> ConstPtr;
	typedef std::weak_ptr<enact_core::entity_id> id;

	static const std::shared_ptr<enact_core::aspect_id> aspect_id;
	
	float certainty_score = 0.5f;
	float right_hand = 0.5f;

	// (timestamp, pose)
	std::list<std::pair<std::chrono::duration<float>, hand_pose_18DoF>> poses;

	hand_trajectory(const hand_instance& source, const Eigen::Affine3f& transformation);

	hand_trajectory(float certainty_score,
	float right_hand,
	std::chrono::duration<float> timestamp,
	const hand_pose_18DoF& poses);

	bool update(const hand_instance& source, const Eigen::Affine3f& transformation);
};

namespace hololens {
	struct hand_data;
}

/**
 * @class hand_tracker_enact
 * @brief Hand tracking system with enactment capabilities
 *
 * Extends the basic hand tracker with enactment functionality, allowing for
 * real-time tracking and processing of hand movements. Integrates with the
 * Hololens system for enhanced hand tracking and provides trajectory management.
 *
 * Features:
 * - Real-time hand tracking
 * - Hololens integration
 * - Trajectory management
 * - Pose transformation
 * - Thread-safe operation
 * - Background plane detection
 * - Delay compensation
 */
class hand_tracker_enact :
	public hand_tracker,
	public enact_priority::signaling_actor < std::pair<std::shared_ptr<enact_core::entity_id>, img_segment::ConstPtr>>
{
public:

	typedef std::shared_ptr<enact_core::entity_id> entity_id;
	typedef enact_core::lockable_data_typed<hand_trajectory> hand_trajectory_data;

	using duration_t = std::chrono::file_clock::duration;
	
	HANDPOSEESTIMATION_API hand_tracker_enact(
		enact_core::world_context& world,
		Eigen::Affine3f world_transformation = Eigen::Affine3f::Identity(),
		std::chrono::duration<float> purge_duration = std::chrono::duration<float>(3.f),
		std::chrono::duration<float> frame_duration = std::chrono::duration<float>(0.03f),
		int max_threads = std::thread::hardware_concurrency(),
		bool spawn_hands = true,
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now(),
		Eigen::Vector3f back_palm_orientation = Eigen::Vector3f(0.f, 0.f, 0.f),
		Eigen::Hyperplane<float, 3> background = Eigen::Hyperplane<float, 3>(Eigen::Vector3f::Zero(), 0.f));
	
	HANDPOSEESTIMATION_API virtual ~hand_tracker_enact();

	HANDPOSEESTIMATION_API virtual void add_hand_pose(const std::shared_ptr<const hololens::hand_data>& hand_data, enact_priority::operation op);

	HANDPOSEESTIMATION_API virtual void reset(std::chrono::high_resolution_clock::time_point start_time);

protected:
	struct hand_info {
		// transmission delay + time offset
		duration_t delay;
		unsigned int received_messages = 0;

		entity_id id;
		hand_instance::Ptr kinect_tracked_hand = nullptr;
	};

	void process_input() override;

	img_segment::Ptr get_segment(const visual_input& input, const hand_pose_18DoF& pose, std::chrono::duration<float> timestamp) const;

	enact_core::world_context& world;
	Eigen::Affine3f transformation;
	Eigen::Affine3f inv_transformation;

	std::chrono::duration<float> frame_duration;

	std::atomic<duration_t> cloud_stamp_offset;
	std::atomic<std::chrono::duration<float>> last_input_stamp;
	std::atomic<duration_t> last_hololens_stamp;
	
	std::map<hand_instance::Ptr, entity_id> hand_to_ids;

	// hands tracked by Hololens
	hand_info left_hand;
	hand_info right_hand;

	std::ofstream file;
	std::chrono::high_resolution_clock::time_point start_time;
	Eigen::IOFormat csv_format = Eigen::IOFormat(4,0,",",",","","","","");
};

}
#pragma once

#ifndef STATE_OBSERVATION__OBJECT_TRACKING__HPP
#define STATE_OBSERVATION__OBJECT_TRACKING__HPP

#include "framework.hpp"

#include <string>
#include <atomic>

#include <opencv2/opencv.hpp>

#include "workspace_objects.hpp"

#include "enact_core/actor.hpp"
#include "enact_core/world.hpp"
#include "enact_core/id.hpp"
#include "enact_priority/signaling_actor.hpp"
#include "enact_priority/priority_actor.hpp"

namespace state_observation
{

/**
 * @class object_tracker
 * @brief Tracks multiple objects with occlusion handling
 * 
 * Implements object tracking with support for occlusion, disappearance,
 * and reappearance. Uses a combination of position, size, and color
 * comparison for robust tracking.
 * 
 * Features:
 * - Multi-object tracking
 * - Occlusion handling
 * - Object lifecycle management
 * - Position comparison
 * - Size and color matching
 * - Priority-based updates
 * - Signal-based communication
 * - Thread-safe operation
 */
class STATEOBSERVATION_API object_tracker
	: public enact_priority::priority_actor, 
	  public enact_priority::signaling_actor<std::shared_ptr<enact_core::entity_id>>
{
public:
	typedef pcl::PointXYZRGBA PointT;
	// this class determines the lifetime of objects, that's why we use strong ownership here
	typedef std::shared_ptr<enact_core::entity_id> entity_id; 
	typedef enact_core::lockable_data_typed<object_instance> object_instance_data;



	object_tracker(enact_core::world_context& world,
				 const pointcloud_preprocessing& pc_prepro,
				 const object_parameters& object_params,
				 float certainty_threshold = 0.5f,
				 std::chrono::duration<float> purge_duration = 
					std::chrono::duration<float>(100000));

	object_tracker(enact_core::world_context& world,
		occlusion_detector::Ptr occlusion_detect,
		const object_parameters& object_params,
		float certainty_threshold = 0.5f,
		std::chrono::duration<float> purge_duration =
		std::chrono::duration<float>(100000));

	virtual ~object_tracker();

	/**
	* Updates the correspondences of previous and new bounding boxes.
	*/
	void update(const std::vector<pc_segment::Ptr>& objects);

	/*
	 * Adds point cloud to test for occlusions
	 */
	void update(const pcl::PointCloud<PointT>::ConstPtr& cloud);

	/**
	* Deletes all tracking information and starts from scratch.
	*/
	void reset();

	std::chrono::duration<float> get_latest_timestamp() const;

private:

	std::list<entity_id> live_objects;
	std::queue<pcl::PointCloud<PointT>::ConstPtr> clouds;
	std::mutex clouds_mutex;

	enact_core::world_context& world;
	const object_parameters& object_params;
	float certainty_threshold;
	std::chrono::duration<float> purge_duration;
	std::atomic<std::chrono::duration<float>> timestamp;
	occlusion_detector::Ptr occlusion_detect;

	/**
	* Returns the likelihood that @ref{hand} is at the position of @ref{seg},
	* assuming linear motion and @ref{seg} is from the next frame.
	*/
	float compare_position(const object_instance& hand, const pc_segment& seg) const;
	/**
	* Returns the similarity of the last snapshot of hand and seg.
	*/
	float compare_size(const object_instance& hand, const pc_segment& seg) const;

	float compare_color(const object_instance& hand, const pc_segment& seg) const;


	static float bell_curve(float x, float stdev);
};

} // namespace state_observation

#endif

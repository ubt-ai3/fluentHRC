#pragma once

#include <mutex>

#include "enact_core/id.hpp"
#include "enact_core/world.hpp"

#include <hand_pose_estimation/hand_model.hpp>

#include <state_observation/pointcloud_util.hpp>

namespace state = state_observation;

class actor_occlusion_detector : public state::occlusion_detector
{
public:
	using entity_id = std::shared_ptr<enact_core::entity_id>;
	using Ptr = std::shared_ptr<actor_occlusion_detector>;

	actor_occlusion_detector(
			float min_object_height,
			const Eigen::Matrix<float,3,4>& camera_projection);


	void update_hand(const entity_id& id, hand_pose_estimation::img_segment::ConstPtr);

	static Eigen::Matrix4f add_z(const Eigen::Matrix<float, 3, 4>& projection);

protected:
	enact_core::world_context world;

	mutable std::mutex update_mutex;
	std::map < entity_id, hand_pose_estimation::img_segment::ConstPtr> hands;

	bool occluded_by_actor(const pcl::PointXY& p) const override;


};
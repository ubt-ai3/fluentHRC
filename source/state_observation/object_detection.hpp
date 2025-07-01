#pragma once

#ifndef STATE_OBSERVATION__OBJECT_DETECTION__HPP
#define STATE_OBSERVATION__OBJECT_DETECTION__HPP

#include "framework.hpp"

#include <set>

#include "enact_core/world.hpp"
#include "enact_core/id.hpp"
#include "enact_priority/signaling_actor.hpp"
#include "enact_priority/priority_actor.hpp"

#include "workspace_objects.hpp"
#include "pointcloud_util.hpp"

namespace state_observation
{

/**
 *************************************************************************
 *
 * @class segment_detector
 *
 * Extracts all regions from a point cloud which have a consistent color 
 * and do not have major depth jumps.
 *
 ************************************************************************/
class STATEOBSERVATION_API segment_detector
	: public enact_priority::priority_actor,
	  public enact_priority::signaling_actor<std::vector<pc_segment::Ptr>&>
{
public:
	typedef pcl::PointXYZRGBA PointT;
	typedef std::weak_ptr<enact_core::entity_id> strong_id;
	typedef std::pair<enact_priority::operation, std::shared_ptr<enact_core::entity_id>> hand_event;

	segment_detector(enact_core::world_context& world,
		pointcloud_preprocessing& pc_prepro);

	virtual ~segment_detector() noexcept;

	/**
	* Updates the correspondences of previous and new bounding boxes.
	*/
	void update(const pcl::PointCloud<PointT>::ConstPtr& cloud);

	std::vector<pc_segment::Ptr> segment_pc  (const pcl::PointCloud<PointT>::ConstPtr& cloud) const;

private:
//	std::list<strong_id> live_objects;
	std::mutex live_objects_mutex;

	enact_core::world_context& world;
	pointcloud_preprocessing& pc_prepro;
	uint64_t cloud_stamp;

};

} // namespace state_observation
	
#endif
#include "util.hpp"

actor_occlusion_detector::actor_occlusion_detector(float min_object_height, const Eigen::Matrix<float, 3, 4>& camera_projection)
	:
	occlusion_detector(min_object_height, add_z(camera_projection))
{}

void actor_occlusion_detector::update_hand(const entity_id& id, hand_pose_estimation::img_segment::ConstPtr seg)
{
	std::lock_guard<std::mutex> lock(update_mutex);
	
	auto iter = hands.find(id);
	if (iter == hands.end())
		hands.emplace(id, std::move(seg));
	else
		iter->second = std::move(seg);
}

Eigen::Matrix4f actor_occlusion_detector::add_z(const Eigen::Matrix<float, 3, 4>& projection)
{
	Eigen::Matrix4f M = Eigen::Matrix4f::Zero();
	M.row(0) = projection.row(0);
	M.row(1) = projection.row(1);
	M(2, 2) = 1.f;
	M.row(3) = projection.row(2);

	return M;
}

bool actor_occlusion_detector::occluded_by_actor(const pcl::PointXY& p) const
{
	std::unique_lock<std::mutex> lock(update_mutex);
	auto local_hands = hands;
	lock.unlock();

	for (const auto& entry : local_hands)
	{
		if (!entry.second)
			continue;
	
		const auto& seg = *entry.second;
		cv::Point cp(p.x, p.y);
		if(seg.bounding_box.contains(cp - seg.bounding_box.tl()) &&
			seg.mask.at<uchar>(cp - seg.bounding_box.tl()))
			return true;
	}

	return false;
}




#pragma once

#include <pcl/visualization/pcl_visualizer.h>

#include "point_cloud_processing.h"

class visualizer
{
public:

	visualizer();

	void set_pcl(incremental_point_cloud::Ptr pcl);
	void set_hand(const hand_pose_estimation::hololens::hand_data::ConstPtr& hand);
	
	void start();

private:

	bool is_stopped() const;

	void remove_hand(hand_pose_estimation::hololens::hand_index hand);
	void update_hand(const hand_pose_estimation::hololens::hand_data& hand_data);
	
	std::mutex mtx;
	mutable std::mutex viewer_mtx;
	std::atomic_bool changed = false;

	std::mutex hand_mtx;
	std::queue<hand_pose_estimation::hololens::hand_data::ConstPtr> hand_queue;
	
	pcl::visualization::PCLVisualizer::Ptr viewer = nullptr;
	incremental_point_cloud::Ptr pcl;
	boost::signals2::connection connection;
};

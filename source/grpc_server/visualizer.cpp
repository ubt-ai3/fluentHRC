#include "visualizer.h"

#include "service_impl.h"
#include <hand_pose_estimation/hololens_hand_data.hpp>

visualizer::visualizer()
{
	viewer = pcl::make_shared<pcl::visualization::PCLVisualizer>("Hello");
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	server::hand_tracking_service::on_tracked_hand.connect(
		[this](const hand_pose_estimation::hololens::hand_data::ConstPtr& hand)
		{
			set_hand(hand);
		});
}

void visualizer::set_pcl(incremental_point_cloud::Ptr pcl)
{
	std::unique_lock lock(mtx);

	if (connection.connected())
		changed = true;
	
	this->pcl = pcl;
	connection.disconnect();
	connection = this->pcl->changed.connect([this]()
		{
			changed = true;
		});
}

void visualizer::set_hand(const hand_pose_estimation::hololens::hand_data::ConstPtr& hand)
{
	std::unique_lock lock(hand_mtx);
	hand_queue.push(hand);
}

void visualizer::start()
{	
	while (!viewer->wasStopped())
	{
		{
			std::unique_lock lock(mtx);
			if (changed.exchange(false) && pcl)
			{
				std::scoped_lock sub_lock(pcl->mtx);
				const auto& pcl_ref = pcl->get_pcl();
				std::string id = std::string((char*)pcl_ref.get());

				if (!viewer->updatePointCloud(pcl_ref, id))
					viewer->addPointCloud(pcl_ref, id);
			}
		}
		
		std::queue<hand_pose_estimation::hololens::hand_data::ConstPtr> hand_swap;
		{
			std::unique_lock lock(hand_mtx);
			std::swap(hand_swap, hand_queue);
		}
		for (const auto& hand : hand_swap._Get_container())
		{			
			if (!hand->valid)
				remove_hand(hand->hand);
			else	
				update_hand(*hand);
		}
		viewer->spinOnce();
	}
}

bool visualizer::is_stopped() const
{
	std::unique_lock lock(viewer_mtx);
	return (!viewer || viewer->wasStopped());
}

void visualizer::remove_hand(hand_pose_estimation::hololens::hand_index hand)
{
	std::string hand_string = hand == hand_pose_estimation::hololens::hand_index::LEFT ? "left_" : "right_";
	
	for (size_t i = 0; i < (size_t)hand_pose_estimation::hololens::hand_key_point::SIZE; ++i)
	{
		std::string id = hand_string +
			hand_pose_estimation::hololens::hr_hand_key_points.at((hand_pose_estimation::hololens::hand_key_point)i);

		viewer->removeCoordinateSystem(id);
	}

	std::string shape_str;
	size_t i = 0, j = 2, k = 5;
	while (i < 5)
	{
		shape_str = hand_string + std::string("wrist_") + 
			hand_pose_estimation::hololens::hr_hand_key_points.at((hand_pose_estimation::hololens::hand_key_point)j);
		
		viewer->removeShape(shape_str);

		for (; j < k; ++j)
		{
			shape_str = hand_string + 
				hand_pose_estimation::hololens::hr_hand_key_points.at((hand_pose_estimation::hololens::hand_key_point)j) + 
				hand_pose_estimation::hololens::hr_hand_key_points.at((hand_pose_estimation::hololens::hand_key_point)(j + 1));
			viewer->removeShape(shape_str);
		}
		++i;
		j = k + 1;
		k += 5;
	}
}

void visualizer::update_hand(const hand_pose_estimation::hololens::hand_data& hand_data)
{
	std::string hand_string = hand_data.hand == hand_pose_estimation::hololens::hand_index::LEFT ? "left_" : "right_";
	const auto& key_data = hand_data.key_data;
	
	for (size_t i = 0; i < (size_t)hand_pose_estimation::hololens::hand_key_point::SIZE; ++i)
	{
		const auto& [position, rotation, radius] = key_data[i];

		Eigen::Affine3f frame = Eigen::Affine3f(Eigen::Translation3f(position) * rotation);

		std::string id = hand_string +
			hand_pose_estimation::hololens::hr_hand_key_points.at((hand_pose_estimation::hololens::hand_key_point)i);

		/*
		 * updateCoordinateSystemPose applies affine transform to existing -> Bug?
		 */
		viewer->removeCoordinateSystem(id);
		viewer->addCoordinateSystem((double)radius, frame, id);
	}
	std::string shape_str;
	auto wrist_pos = server::convert<pcl::PointXYZ>(key_data[1].position);

	size_t i = 0, j = 2, k = 5;
	while (i < 5)
	{
		shape_str = hand_string + std::string("wrist_") + 
			hand_pose_estimation::hololens::hr_hand_key_points.at((hand_pose_estimation::hololens::hand_key_point)j);
		
		viewer->removeShape(shape_str);
		viewer->addLine(wrist_pos, 
			server::convert<pcl::PointXYZ>(key_data[j].position), shape_str);
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.f, 1.f, 0.f, shape_str);

		for (; j < k; ++j)
		{
			shape_str = hand_string +
				hand_pose_estimation::hololens::hr_hand_key_points.at((hand_pose_estimation::hololens::hand_key_point)j) +
				hand_pose_estimation::hololens::hr_hand_key_points.at((hand_pose_estimation::hololens::hand_key_point)(j + 1));

			viewer->removeShape(shape_str);
			viewer->addLine(server::convert<pcl::PointXYZ>(key_data[j].position),
				server::convert<pcl::PointXYZ>(key_data[j + 1].position), shape_str);
			viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.f, 1.f, 0.f, shape_str);
		}
		++i;
		j = k + 1;
		k += 5;
	}
}
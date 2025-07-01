#include "module_manager.hpp"

#include "KinectGrabber/kinect2_grabber.h"

#include <iostream>
namespace state_observation
{

	module_manager::module_manager()
		:
		grabber(new pcl::Kinect2Grabber),
		object_params(new object_parameters)

	{
		std::cout << "started Program";
		pc_prepro = std::make_unique<pointcloud_preprocessing>(object_params);
		auto cloud_signal = std::make_shared<cloud_signal_t>();
		std::weak_ptr<cloud_signal_t> weak_cloud_signal = cloud_signal;

		//TODO test

		std::function<void(const pcl::PointCloud<PointT>::ConstPtr&)> pc_grabber =
			[weak_cloud_signal, this](const pcl::PointCloud<PointT>::ConstPtr& input) {
			auto cloud_signal = weak_cloud_signal.lock();
			if (cloud_signal)
			{
				static std::queue<pcl::PointCloud<PointT>::ConstPtr> cloud_queue;
				if (cloud_queue.size() < 200)
					cloud_queue.emplace(input);
				static int skipped{0};
				if ( skipped>5 &&!cloud_queue.empty())
				{
					skipped = 0;
					std::cout << "Processing cloud: " <<cloud_queue.size() <<" in queue\n ";// skipped "<<current_skipped_clouds <<" before \n";
					cloud_signal->operator()(pc_prepro->remove_table(cloud_queue.front()));
					cloud_queue.pop();

					current_skipped_clouds = 0;
				}
				else skipped++;
			}
		};
		obj_detect = std::make_unique<segment_detector>(world, *pc_prepro);

		obj_track = std::make_unique<object_tracker>(world, *object_params);

		obj_classify = std::make_unique<classification_handler>(world, object_params);

		view = std::make_unique<viewer>(world, *pc_prepro);
	
	cloud_signal->connect([&](const pcl::PointCloud<PointT>::ConstPtr& cloud) {
		obj_detect->update(cloud);
		view->update_cloud(cloud);
	});

	auto sig1 = obj_detect->get_signal(enact_priority::operation::CREATE);
	sig1->connect([&](std::vector<pc_segment::Ptr> segments) {
		//std::cout << "Signal: start tracking\n";
		obj_track->update(segments);
		});

	auto sig2 = obj_track->get_signal(enact_priority::operation::CREATE);
	sig2->connect([&](const entity_id& id) {
	//std::cout << "Signal: start classifying\n";
		obj_classify->update(id);
	});

	auto sig3 = obj_track->get_signal();
	sig3->connect([&](const entity_id& id, enact_priority::operation op) {
		view->update_object(id, op);
	});

	auto sig4 = obj_classify->get_signal();
	sig4->connect([&](const entity_id& id, enact_priority::operation op) {
		view->update_object(id, op);
	});
	
		// Kinect2Grabber
	std::shared_ptr<pcl::Grabber> grabber = std::make_shared<pcl::Kinect2Grabber>();

	// Register Callback Function
	std::signals2::connection connection_pc = grabber->registerCallback(pc_grabber);
	grabber->start();

	view->wait_for_close();


}

module_manager::~module_manager()
{
	grabber->stop();
}


} // namespace state_observation

int main(int argc, char* argv[])
{
	state_observation::module_manager manager;
}

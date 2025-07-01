// Disable Error C4996 that occur when using pcl_io

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <memory>
#include <numbers>

#include "KinectGrabber/kinect2_grabber.h"

#include <csv_reader/tracker.hpp>

#include <pcl/visualization/pcl_visualizer.h>

#include <boost/timer/timer.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/join.hpp>

#include <enact_core/world.hpp>

#include <hand_pose_estimation/hand_pose_estimation.h>
// #include "hand_pose_estimation/hand_tracker_enact.hpp"

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <state_observation/pointcloud_util.hpp>


#include "hand_evaluation.hpp"
#include "hand_pose_estimation/hand_tracker.hpp"
#include "state_observation/calibration.hpp"

#include "projection_matrix.hpp"

#define HAND_TRACKING_EVALUATION

#ifdef HAND_TRACKING_EVALUATION
std::string opti_track_file("W:\\DB_Forschung\\FlexCobot\\11.Unterprojekte\\DS_Nico_Höllerich\\05.Rohdaten\\MMK.Bausteine\\Motion Tracking.no_backup\\Team.10.IDs.19.20.Trial.3.csv"); 
#endif

using namespace state_observation;

typedef pcl::PointXYZRGBA PointT;

class recording_timer
{
private:
	float start_time = -1.f;

public:
	float update(uint64_t stamp)
	{
		float time = stamp * 1e-6;
		if (start_time == -1.f || time < start_time)
			start_time = time;

		return time - start_time;
	}

	float get_relative_time() const
	{
		if (start_time == -1.f)
			return 0.f;
		return start_time;
	}
};

void show_opti_track(const std::shared_ptr<pcl::visualization::PCLVisualizer>& pcl_viewer,
	tracker& opti_track,
	float time)
{
	auto hand_keypoints = std::make_shared<pcl::PointCloud<PointT>>();

	for (int hand_id : {0, 1, 2, 3})
	{
		hand_type hand = (hand_type)hand_id;
		Eigen::Vector3f pos(opti_track.get_position_3d(hand, time));
		if (isnan(pos(0)))
			continue;

		PointT p;
		p.getArray3fMap() = pos;

		const int hand_num = static_cast<int>(hand) + 1;
		p.r = 255 * (hand_num & 1);
		p.g = 255 * (hand_num & 2);
		p.b = 255 * (hand_num & 4);

		hand_keypoints->push_back(p);
	}


	if (!pcl_viewer->updatePointCloud(hand_keypoints, "opti hands"))
		pcl_viewer->addPointCloud(hand_keypoints, "opti hands");

	pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "opti hands");
}


void show_camera_track(const std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const std::vector<hand_pose_estimation::hand_instance::Ptr>& hands)
{
	auto hand_keypoints = std::make_shared<pcl::PointCloud<PointT>>();

	for (const hand_pose_estimation::hand_instance::Ptr& hand : hands)
	{
		if (hand->certainty_score > 0.5f)
		{
			size_t id = std::hash<hand_pose_estimation::hand_instance*>{}(&*hand);
			cv::Scalar color = cv::Vec3b(127 * (id / 16 % 4), 127 * (id / 4 % 4), 127 * (id % 4));
			//std::cout << id << " ";

			// rectangle
			const hand_pose_estimation::img_segment& seg = *hand->observation_history.back();

			PointT center;
			//center.getArray3fMap() = seg.palm_center_3d;
			center.r = color(2);
			center.g = color(1);
			center.b = color(0);
			hand_keypoints->push_back(center);

		}
	}



	if (!viewer->updatePointCloud(hand_keypoints, "camera hands"))
		viewer->addPointCloud(hand_keypoints, "camera hands");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "camera hands");
}

#ifdef HAND_TRACKING_EVALUATION

void test_3d_transformation()
{
	const Eigen::Matrix4f transformation(compute_transformation_opti_track_to_camera_space());

	std::cout << "Reading CSV" << std::endl;

	tracker track(
		opti_track_file,
		transformation
	);

	std::cout << "Starting Grabber" << std::endl;

	auto pcl_viewer = std::make_shared<pcl::visualization::PCLVisualizer>("Point Cloud Viewer");

	pcl_viewer->addCoordinateSystem();

	// Point Cloud
	pcl::PointCloud<PointT>::ConstPtr cloud;


	// Retrieved Point Cloud Callback Function
	std::mutex mutex;
	std::function<void(const pcl::PointCloud<PointT>::ConstPtr&)> pc_grabber =
		[&cloud, &mutex](const pcl::PointCloud<PointT>::ConstPtr& ptr) {
		std::lock_guard<std::mutex> lock(mutex);

		if (!ptr || ptr->header.stamp == 0)
			return;

		cloud = ptr->makeShared();

	};


	// Kinect2Grabber
	std::shared_ptr<pcl::Grabber> grabber = std::make_shared<pcl::Kinect2Grabber>();

	// Register Callback Function
	boost::signals2::connection connection_pc = grabber->registerCallback(pc_grabber);


	// Start Grabber
	grabber->start();


	std::cout << "Preperation done. Start the recording!" << std::endl;
	recording_timer timer;

	while (!pcl_viewer->wasStopped()) {
		// Update Viewer
		try {
			pcl_viewer->spinOnce();
		}
		catch (...) {
			continue;
		}

		pcl::PointCloud<PointT>::ConstPtr temp_cloud = nullptr;

		{
			if (mutex.try_lock() && cloud && !cloud->empty()) {
				temp_cloud = cloud;
				mutex.unlock();
			}

		}

		if (temp_cloud)
		{
			float time = timer.update(temp_cloud->header.stamp);

			show_opti_track(pcl_viewer, track, time);

			if (!pcl_viewer->updatePointCloud(temp_cloud))
				pcl_viewer->addPointCloud(temp_cloud);

		}


	}

	// Stop Grabber
	grabber->stop();

	// Disconnect Callback Function
	if (connection_pc.connected()) {
		connection_pc.disconnect();
	}
}

#endif

//void show_hand_tracking(boost::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
//	const state_observation::kinect2_parameters& kinect2_params,
//	hand_pose_estimation::hand_pose_estimation& hand_pose_est,
//	hand_pose_estimation::hand_tracker& hand_track,
//	const cv::Mat4b& img,
//	pcl::PointCloud<PointT>::ConstPtr cloud)
//{
//	std::vector< std::vector<cv::Point>> hand_regions;
//
//
//	viewer->removeAllShapes();
//	viewer->removeAllPointClouds();
//
//
//	for (const hand_pose_estimation::hand_instance::Ptr& hand : hand_track.get_hands())
//	{
//		hand_regions.push_back(hand->observation_history.back()->contour);
//	}
//
//	std::vector<hand_pose_estimation::img_segment::Ptr> hand_segments = hand_pose_est.detect_hands(cloud, hand_regions, cloud->header.stamp, img, kinect2_params.rgb_projection);
//	hand_track.update(img, hand_segments, cloud->header.stamp);
//	std::vector<hand_pose_estimation::hand_instance::Ptr> hands = hand_track.get_hands();
//
//
//	show_camera_track(viewer, hands);
//
//	if (!viewer->updatePointCloud(cloud))
//		viewer->addPointCloud(cloud);
//}

int main(int argc, char* argv[])
{
#ifdef	HAND_TRACKING_EVALUATION
//	test_3d_transformation();

	if (argc > 1)
		opti_track_file = std::string(argv[1]);
#endif



	// objects needed for both evaluations
	state_observation::kinect2_parameters kinect2_params;
	std::shared_ptr<state_observation::object_parameters> object_params(new state_observation::object_parameters);
	enact_core::world_context world;


#ifdef HAND_TRACKING_EVALUATION
	//hand_pose_estimation::hand_pose_estimation hand_pose_est;
	hand_pose_estimation::hand_tracker hand_track(std::chrono::milliseconds(200));
	hand_pose_estimation::classifier_set classifiers;
	tracker opti_track(opti_track_file, compute_transformation_opti_track_to_camera_space());

	//generate output file name
	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
	std::vector<std::string> split_string;
	boost::algorithm::split(split_string, opti_track_file, [](char c) {return c == '.'; });
	split_string.pop_back();
	split_string.emplace_back("eval");

	std::stringstream stream;
	stream << time.date().year()
		<< "." << time.date().month().as_number()
		<< "." << time.date().day()
		<< "." << time.time_of_day().hours()
		<< "." << time.time_of_day().minutes();
	split_string.push_back(stream.str());
	split_string.emplace_back("csv");
	std::string output_file_name = boost::algorithm::join(split_string, ".");


	hand_evaluator hand_eval(opti_track, classifiers, output_file_name,0.5f);
	std::vector< std::vector<cv::Point>> hand_regions;
	recording_timer timer;
#else
	pointcloud_preprocessing pc_prepro(object_params);
	segment_detector obj_detect(world, pc_prepro);
	object_evaluation obj_eval(pc_prepro, obj_detect, kinect2_params);
#endif

	auto pcl_viewer = std::make_shared<pcl::visualization::PCLVisualizer>("Point Cloud Viewer");
	pcl_viewer->addCoordinateSystem();


	// Point Cloud
	pcl::PointCloud<PointT>::ConstPtr cloud;
	pcl::PointCloud<PointT>::ConstPtr prev_cloud;
	std::shared_ptr<cv::Mat4b> img;
	bool updated = false;

	// Retrieved Point Cloud Callback Function
	std::mutex mutex;
	std::function pc_grabber =
		[&cloud, &prev_cloud, &mutex, &updated](const pcl::PointCloud<PointT>::ConstPtr& ptr) {
		std::lock_guard<std::mutex> lock(mutex);

		if (!ptr || ptr->header.stamp == 0)
			return;

		if (!cloud || cloud->header.stamp != ptr->header.stamp)
		{
			prev_cloud = cloud;
			cloud = ptr->makeShared();

			updated = true;
		}

		if (updated && cloud->header.stamp < ptr->header.stamp)
		{ // a new recording is playing
			updated = false;
			prev_cloud = nullptr;
		}

	};

	std::function img_grabber =
		[&img, &mutex](const std::shared_ptr<cv::Mat4b>& input) {
		std::lock_guard<std::mutex> lock(mutex);

		img = input;
	};

	// Kinect2Grabber
	std::shared_ptr<pcl::Grabber> grabber = std::make_shared<pcl::Kinect2Grabber>();

	// Register Callback Function
	boost::signals2::connection connection_pc = grabber->registerCallback(pc_grabber);
	boost::signals2::connection connection_img = grabber->registerCallback(img_grabber);






	// Start Grabber
	grabber->start();

	// Create a window for display.



	std::cout << "Preparation done. Start the recording!" << std::endl;

	while (!pcl_viewer->wasStopped()) {
		// Update Viewer
		try {
			pcl_viewer->spinOnce();
		}
		catch (...) {
			continue;
		}

		pcl::PointCloud<PointT>::ConstPtr temp_cloud = nullptr;
		bool temp_updated = false;
		std::shared_ptr<cv::Mat4b> temp_img = nullptr;

		{
			std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);
			if (lock.owns_lock() && updated && !cloud->empty()) {
				temp_cloud = cloud;
				temp_updated = true;
			}

			if (lock.owns_lock() && img && !img->empty()) {
				temp_img = img;
			}
		}



		if (temp_img && temp_cloud && !temp_cloud->empty())
		{
			const auto& image = *temp_img;
			const auto& cloud = temp_cloud;


#ifdef HAND_TRACKING_EVALUATION
			float time = timer.update(cloud->header.stamp);

			Eigen::Translation2f center(image.size[1] * 0.5f, image.size[0] * 0.5f);
			Eigen::Matrix<float, 3, 4> transform = center * Eigen::Affine2f(Eigen::Rotation2D<float>(std::numbers::pi)) * center.inverse() * kinect2_params.rgb_projection;

			//std::vector<hand_pose_estimation::img_segment::Ptr> hand_segments = hand_pose_est.detect_hands(cloud, hand_regions, cloud->header.stamp, *img, transform);
			////hand_track.update(*img, hand_segments, cloud->header.stamp);
			//std::vector<hand_pose_estimation::hand_instance::Ptr> hands = hand_track.get_hands();

			//hand_eval.update(cloud, hands, time);
			//hand_regions.clear();

			//for (const hand_pose_estimation::hand_instance::Ptr& hand : hands)
			//{
			//	hand_regions.push_back(hand->observation_history.back()->contour);
			//}

			//show_camera_track(pcl_viewer, hands);
			show_opti_track(pcl_viewer, opti_track, time);

			if (!pcl_viewer->updatePointCloud(temp_cloud))
				pcl_viewer->addPointCloud(temp_cloud);

#else
			obj_eval.update(cloud, image);
//			object_recognition_test obj_test;
//			obj_test.compute_and_show_classified_objects(pcl_viewer, pc_prepro, obj_eval.classifiers, cloud);
			break;
#endif

		}

	}

	// Stop Grabber
	grabber->stop();

	// Disconnect Callback Function
	if (connection_pc.connected()) {
		connection_pc.disconnect();
	}

	if (connection_img.connected()) {
		connection_img.disconnect();
	}

	return 0;
}


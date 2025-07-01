#include "viewer.hpp"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

#include <boost/timer/timer.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "enact_core/access.hpp"

#include <state_observation/building_estimation.hpp>

#include "tests/object_recognition.hpp"
#include "tests/hand_pose.hpp"

#include <iostream>

using namespace state_observation;

typedef pcl::PointXYZRGBA PointT;

// Disable Error C4996 that occur when using Boost.Signals2.
#ifdef _DEBUG
#define _SCL_SECURE_NO_WARNINGS
#endif


viewer::viewer(enact_core::world_context& world,
	std::string opencv_viewer_title,
	std::string pcl_viewer_title)
	:
	close_signal(std::make_shared< boost::signals2::signal<void(void)>>()),
	world(world),
	cv_window_title(std::move(opencv_viewer_title)),
	pc_window_title(std::move(pcl_viewer_title)),
	redraw_scheduled(false)
{

}

viewer::~viewer() noexcept
{
}



void viewer::run_sync()
{
	{
		std::scoped_lock lock(queue_mutex, viewer_mutex);
		pcl_viewer = std::make_shared<pcl::visualization::PCLVisualizer>(pc_window_title);
		//	pcl_viewer->setCameraPosition(0.0, 0.0, 2.0, 1.0, 0.0, 0.0);
		//	pcl_viewer->setCameraPosition(0.0, 0.0, 2.0, 0.0, 1.0, 0.0);

		pcl_viewer->createViewPort(0, 0, 1, 1, viewport_full_window);
		pcl_viewer->createViewPortCamera(viewport_full_window);
		pcl_viewer->setCameraPosition(0.0, 0.0, 3.5, 0.0, 1.0, 0.0, viewport_full_window);

		//DEBUG remove overlay
		pcl_viewer->createViewPort(0.66, 0.66, 1, 1, viewport_overlay);
		pcl_viewer->createViewPortCamera(viewport_overlay);
		pcl_viewer->setCameraPosition(0.0, 0.0, 2.0, 0.0, 1.0, 0.0, viewport_overlay);

		pcl_viewer->addCoordinateSystem(1.0, "reference", viewport_full_window);
		pcl_viewer->addCoordinateSystem(1.0, "reference", viewport_overlay);
	}
	
	refresh(); // adds another refresh call to queue

	while (true)
	{
		try {
			std::function<void()> f;
			{
				std::unique_lock<std::mutex> lock(queue_mutex);

				if (pcl_viewer->wasStopped())
				{
					(*close_signal)();
					return;
				}


				queue_condition_variable.wait(lock, [&] {return !queue.empty() || pcl_viewer->wasStopped(); });

				if (pcl_viewer->wasStopped())
				{
					(*close_signal)();
					return;
				}

				f = queue.front();
				queue.pop_front();
			}

			f(); // process data
		}
		catch (const std::exception & e)
		{
			std::cout << e.what() << std::endl;
		}

	}
}

void viewer::refresh()
{
	{
		std::lock_guard<std::mutex> guard(viewer_mutex);
		pcl_viewer->spinOnce();
	}

	bool queue_empty;
	{
		std::lock_guard<std::mutex> guard(queue_mutex);
		queue_empty = queue.empty();
	}

	if(queue_empty)
		std::this_thread::sleep_for(std::chrono::milliseconds(15));

	schedule([this]() {refresh(); });

}


void viewer::run_async()
{
	internal_thread =
		std::make_unique<std::thread>([this]() -> void { this->run_sync(); });
}

std::string viewer::to_string(const strong_id& id)
{
	return std::to_string(std::hash<enact_core::entity_id*>{}(&*id));
}

int viewer::viewport_full_window_id() const
{
	return viewport_full_window;
}

int viewer::viewport_overlay_id() const
{
	return viewport_overlay;
}

void viewer::redraw()
{
	if (redraw_scheduled)
		return;

	redraw_scheduled = true;

	schedule([this]() {
		redraw_scheduled = false;


		std::lock_guard<std::mutex> guard(viewer_mutex);

		if (pcl_viewer->wasStopped())
		{
			(*close_signal)();
			return;
		}

//		pcl_viewer->removeAllPointClouds();
		pcl_viewer->removeAllShapes();

		pcl_viewer->addPointCloud(cloud, "cloud", viewport_full_window);


		//for (auto iter = live_objects.begin(); iter != live_objects.end();)
		//{
		//	strong_id id(iter->lock());
		//	if (!id)
		//	{
		//		iter = live_objects.erase(iter);
		//		continue;
		//	}

		//	draw_object(id);

		//	++iter;


		//}
	});
}

void viewer::add_bounding_box(const obb& obox, const std::string& name, float r,float g,float b)
{
	{
		std::lock_guard<std::mutex> box_lock(box_mutex);
		box_names.emplace(name);
		//if (!box_names.emplace(name).second)
			//return;
	}
	schedule([this, obox, name,r,g,b]() {

		std::lock_guard<std::mutex> lock(viewer_mutex);
		if (pcl_viewer->contains(name))
			pcl_viewer->removeShape(name);
		draw_obb(obox, name, r, g, b);
		});//schedule
}

void viewer::remove_bounding_box(const std::string& name)
{
	std::unique_lock<std::mutex> box_lock(box_mutex);
	if (!box_names.erase(name))
		return;

	schedule([this, name]() {

		std::lock_guard<std::mutex> lock(viewer_mutex);
		remove_obb(name);
	});//schedule
}

std::shared_ptr<boost::signals2::signal<void(void)>> viewer::get_close_signal()
{
	return close_signal;
}

void viewer::update_cloud(const pcl::PointCloud<PointT>::ConstPtr& cloud)
{

	cloud_stamp = cloud->header.stamp * 1e-6;
	
	schedule([this, cloud]() {

		//DEPRECATED TEST
		//auto new_cloud = pcl::make_shared<pcl::PointCloud<PointT>>();
		//static Eigen::Vector3f a(0, 0, 0);
		//static Eigen::Vector3f axis(0, 0, 1);
		//axis.normalize();
		//static float angle = 0.13;
		//Eigen::Quaternionf b = Eigen::Quaternionf(Eigen::AngleAxisf(angle, axis));
		////pcl::transformPointCloud<PointT>(*cloud, *new_cloud,a , b);
		//pcl::transformPointCloud<PointT>(*cloud, *new_cloud, Eigen::Vector3f(0,0,0),Eigen::Quaternionf::Identity());
		//new_cloud->sensor_orientation_.setIdentity();
		//new_cloud->sensor_origin_ = Eigen::Vector4f(0, 0, 0,1);
		//this->cloud = new_cloud;

		std::lock_guard<std::mutex> guard(viewer_mutex);
		
		if (pcl_viewer->wasStopped())
		{
			(*close_signal)();
			return;
		}


		if (!pcl_viewer->updatePointCloud(cloud))
			pcl_viewer->addPointCloud(cloud, "cloud", viewport_full_window);

//		pcl_viewer->spinOnce();

		//if (!cloud->empty() && cloud->isOrganized())
		//{
		//	img = cv::Mat(cloud->height, cloud->width, CV_8UC4);

		//	if (!cloud->empty()) {

		//		for (int h = 0; h < img.rows; h++) {
		//			for (int w = 0; w < img.cols; w++) {
		//				pcl::PointXYZRGBA point = cloud->at(w, h);

		//				Eigen::Vector3i rgb = point.getRGBVector3i();

		//				img.at<cv::Vec4b>(h, w)[0] = rgb[2];
		//				img.at<cv::Vec4b>(h, w)[1] = rgb[1];
		//				img.at<cv::Vec4b>(h, w)[2] = rgb[0];
		//			}
		//		}
		//	}

		//	update_image(img);
		//}
		});
}

void viewer::update_object(const strong_id& id, enact_priority::operation op)
{
	weak_id w_id(id);
	schedule([this, w_id, str_id = std::string(to_string(id)), op]() {

		{
			std::lock_guard<std::mutex> guard(viewer_mutex);
			if (pcl_viewer->wasStopped())
			{
				(*close_signal)();
				return;
			}
		}

		if (op == enact_priority::operation::DELETED)
		{
			const auto iter = live_objects.find(str_id);
			if (iter != live_objects.end())
			{
				{
					std::lock_guard<std::mutex> guard(viewer_mutex);
					pcl_viewer->removeShape(str_id);
					pcl_viewer->removePolygonMesh(str_id);
					live_objects.erase(iter);
				}

			}
			//else
			//	std::cerr << "Object not found" << std::endl;
			
			return;
			

		}

		strong_id id(w_id.lock());
		if (!id)
			return;

		if (!live_objects.contains(str_id))
		draw_object(id);
		});
}

void viewer::update_building(const strong_id& id, enact_priority::operation op)
{
	weak_id w_id(id);
	schedule([this, w_id, op]() {

		{
			std::lock_guard<std::mutex> guard(viewer_mutex);
			if (pcl_viewer->wasStopped())
			{
				(*close_signal)();
				return;
			}
		}

		if (op == enact_priority::operation::DELETED)
		{
			const auto iter = live_buildings.find(w_id);
			if (iter != live_buildings.end())
			{
				std::lock_guard<std::mutex> guard(viewer_mutex);

				for (const auto& element : iter->second)
				{
					pcl_viewer->removePolygonMesh(element);
				}
				live_buildings.erase(iter);
			}
			return;
		}

		strong_id id(w_id.lock());
		if (!id)
			return;

		draw_building(id);
	});
}

void viewer::update_hand(const strong_id& id, enact_priority::operation op)
{
	weak_id w_id(id);
	schedule([this, w_id, str_id = std::string(to_string(id)), op]() {

		{
			std::lock_guard<std::mutex> guard(viewer_mutex);
			if (pcl_viewer->wasStopped())
			{
				(*close_signal)();
				return;
			}

			if (op == enact_priority::operation::DELETED)
			{
				const auto iter = live_hands.find(str_id);
				if (iter != live_hands.end())
				{
					pcl_viewer->removePolygonMesh("hand_" + str_id);
					for (int i = 1; i < 21; i++)
						pcl_viewer->removeShape("line_" + std::to_string(i) + str_id);
					live_hands.erase(iter);
				}
				return;
			}
		}

		if (op == enact_priority::operation::CREATE)
			live_hands.emplace(str_id);

		draw_hand_on_image(w_id);

		strong_id id(w_id.lock());
		if (!id)
			return;

		draw_hand(id);
		});
}

void viewer::update_image(const cv::Mat& img)
{
	schedule([this, img]() {
		img.copyTo(this->img);
		show_image();
	});
}

void viewer::update_overlay(const cv::Mat& overlay)
{
	schedule([this, overlay]() {
		overlay.copyTo(this->overlay);
		show_image();
	});

}

void viewer::update_segment(const std::shared_ptr<pc_segment>& seg)
{
	schedule([this, seg]() {
		const obb& box = seg->bounding_box;
		std::string id(std::to_string(std::hash<std::shared_ptr<pc_segment>>{}(seg)));
			{
				std::lock_guard<std::mutex> guard(viewer_mutex);
				draw_obb(box, id, 0., 0., 1., viewport_overlay);
			}
		});
}

void viewer::update_element(std::function<void(pcl::visualization::PCLVisualizer&)>&& f)
{
	schedule([this, ff = std::move(f)]()
		{
			std::lock_guard<std::mutex> lock(viewer_mutex);
			ff(*pcl_viewer);
		});
}


void viewer::draw_hand_on_image(const weak_id& w_id)
{
	/*
	strong_id id = w_id.lock();
	if (!id)
		return;
		
	enact_core::lock l(
	, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::write));
	enact_core::const_access<hand_instance_data> access_hand(l.at(id, hand_pose_estimation::hand_instance::aspect_id));
	const hand_pose_estimation::hand_instance& hand = access_hand->payload;

	if (hand.certainty_score > 0.5f)// && hand->observation_history.back()->timestamp == hand_track.get_latest_timestamp())
	{

		auto int_id = std::hash<enact_core::entity_id*>{}(&*id);
		cv::Scalar color = cv::Vec3b(127 * (int_id / 16 % 4), 127 * (int_id / 4 % 4), 127 * (int_id % 4));
		//std::cout << id << " ";

		// rectangle
		const hand_pose_estimation::img_segment& seg = *hand.observation_history.back();
		cv::rectangle(img, seg.bounding_box, color, 2);
		cv::circle(img, seg.palm_center_2d, 8, cv::Scalar(0, 0, 255), -1);


		float certainty = 0.f;


		cv::drawContours(img, std::vector<std::vector<cv::Point2i>>({ seg.contour }), -1, color, certainty > 0.5f ? -1 : 2);
		cv::putText(img, cv::format("%.2f", certainty), hand.observation_history.back()->palm_center_2d, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 0), 2);

		for (const cv::Point2i& p : seg.finger_tips_2d)
			cv::circle(img, p, 8, cv::Scalar(0, 0, 255), -1);

		PointT center;
		center.getArray3fMap() = seg.palm_center_3d;
		center.r = color(2);
		center.g = color(1);
		center.b = color(0);
		hand_keypoints->push_back(center);
	}
	*/
}

void viewer::draw_obb(const obb& box, const std::string& id, double r, double g, double b, int viewport)
{
	pcl_viewer->addCube(box.translation, box.rotation, box.diagonal.x(), box.diagonal.y(), box.diagonal.z(), id, viewport);
	pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
	pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r, g, b, id);
}

void viewer::remove_obb(const std::string& id)
{
	pcl_viewer->removeShape(id);	
}
	
void viewer::draw_object(const strong_id& id)
{
	enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::read));
	const enact_core::const_access<object_instance_data> access_object(l.at(id, object_instance::aspect_id));
	const object_instance& obj = access_object->payload;

	std::string str_id = to_string(id);
	
	auto draw_bb = [&]()
	{
		const obb& box = obj.observation_history.back()->bounding_box;
		remove_obb(str_id);
		draw_obb(box, str_id, 0., 1., 0., viewport_overlay);
	};
	
	if (obj.observation_history.empty())
		return;

	const pc_segment::Ptr seg = obj.get_classified_segment();

	if(!seg)
	{
		//draw_bb();
		return;
	}

	
	const classification_result& result = seg->classification_results.front();
	//debug uncomment
	if (result.local_certainty_score > 0.5f && result.prototype)
	{
		const object_prototype& prototype = *result.prototype;

		if (prototype.has_mesh())
		{
			const Eigen::Affine3f proto_transform = calc_proto_transform(*seg);

			const Eigen::Affine3f transform = proto_transform *
				Eigen::Scaling(0.5f * prototype.get_bounding_box().diagonal);
				//Scaling with half the diagonal -> Scaling by 0.5 in both negative/positive direction -> Effectively scaling by 1.0
			{
				std::lock_guard<std::mutex> guard(viewer_mutex);
				if (!pcl_viewer->updatePointCloudPose(str_id, transform))
				{
					pcl_viewer->addPolygonMesh(*result.prototype->load_mesh(), str_id, viewport_overlay);
					pcl_viewer->updatePointCloudPose(str_id, transform);
				}
			}
			live_objects.emplace(str_id);
		}
	}
		//else
		//{
		//draw_bb();
		//}

	
}

Eigen::Affine3f viewer::calc_proto_transform(const pc_segment& seg)
{
	const classification_result& result = seg.classification_results.front();
	const object_prototype& prototype = *result.prototype;
	
	//align with top - for stacked objects
	const float top_z = std::max(prototype.get_bounding_box().diagonal.z(), seg.bounding_box.top_z());
	Eigen::Vector3f translation = seg.bounding_box.translation;
	translation.z() = top_z - 0.5f * prototype.get_bounding_box().diagonal.z();

	return Eigen::Translation3f(translation) *
		Eigen::Affine3f(seg.bounding_box.rotation) * Eigen::Affine3f(result.prototype_rotation);
}

void viewer::draw_building(const strong_id& id)
{
	enact_core::lock l(world, enact_core::lock_request(id, building::aspect_id, enact_core::lock_request::read));
	const enact_core::const_access<building_instance_data> access_object(l.at(id, building::aspect_id));
	const building& building = access_object->payload;

	const auto& elements = building.visualize();

	auto to_string = [](const std::shared_ptr<single_building_element>& element) 
	{
		return std::to_string(std::hash<single_building_element*>{}(&*element));
	};

	const auto it = live_buildings.emplace(id, std::set<std::string>());

	for (const auto& element : elements)
	{
		Eigen::Affine3f transform = Eigen::Translation3f(element->obbox.translation) *
			Eigen::Affine3f(element->obbox.rotation) *
			Eigen::Scaling(0.5f * element->obbox.diagonal);

		{
			if (!pcl_viewer->updatePointCloudPose(to_string(element), transform))
			{
				pcl_viewer->addPolygonMesh(*element->token->object->load_mesh(), to_string(element));
				pcl_viewer->updatePointCloudPose(to_string(element), transform);
			}
		}
		it.first->second.emplace(to_string(element));
}
}


void viewer::draw_hand(const strong_id& id)
{
	enact_core::lock l(world, enact_core::lock_request(id, hand_pose_estimation::hand_trajectory::aspect_id, enact_core::lock_request::read));
	const enact_core::const_access<hand_instance_data> access_object(l.at(id, hand_pose_estimation::hand_trajectory::aspect_id));
	const hand_pose_estimation::hand_trajectory& obj = access_object->payload;

	auto key_points = obj.poses.back().second.get_key_points();
	const int int_id = std::hash<enact_core::entity_id*>{}(&*id);
	cv::Scalar color = cv::Vec3b(63 * (int_id / 16 % 4), 63 * (int_id / 4 % 4), 63 * (int_id % 4));

	

	std::vector<PointT> points;
	const auto skeleton_cloud = pcl::make_shared<pcl::PointCloud<PointT>>();
	for (int i = 0; i < key_points.cols(); i++)
	{
		Eigen::Vector3f key_point = key_points.col(i);

		PointT p;
		p.x = key_point.x();
		p.y = key_point.y();
		p.z = key_point.z();
		if (i == 4)
		{
			p.r = p.g = p.b = p.a = 255;
		}
		else
		{
			p.r = color(2); p.g = color(1); p.b = color(0); p.a = 255;
		}

		skeleton_cloud->push_back(p);


		
		if (!points.empty())
		{
			try {
				std::string str_id = std::string("line_") + std::to_string(i) + to_string(id);
				pcl_viewer->removeShape(str_id);
				if (i % 4 == 1) {

					pcl_viewer->addLine(points.front(), p, p.r, p.g, p.b, str_id);
				}
				else
					pcl_viewer->addLine(points.back(), p, p.r, p.g, p.b, str_id);
			}
			catch(const std::exception&){}
		}
		points.push_back(p);
	}

	if (!pcl_viewer->updatePointCloud(skeleton_cloud, "hand_" + to_string(id)))
		pcl_viewer->addPointCloud(skeleton_cloud, "hand_" + to_string(id));
	pcl_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "hand_" + to_string(id));
}

	
void viewer::show_image()
{
	if(img_timer.elapsed().wall < 10000000) // 10fps
	{
		return;
	}
	
	if (!img.empty())
	{
		if (overlay.empty())
		{
			try
			{
				cv::imshow(cv_window_title, img);
				cv::waitKey(1);
			}
			catch (const std::exception & )
			{
			}
		}
		else {
			auto zeros = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));

			std::vector<cv::Mat> channels;
			cv::split(overlay, channels);
			cv::Mat alpha;
			cv::merge(std::vector<cv::Mat>({ channels[3],channels[3],channels[3],zeros }), alpha);

			const cv::Mat combined = img.mul(cv::Scalar(255, 255, 255, 255) - alpha, 1. / 255) + overlay.mul(alpha, 1. / 255);

			try
			{
				cv::imshow(cv_window_title, combined);
				cv::waitKey(1);
			}
			catch (const std::exception & )
			{
			}
		}

	}

	img_timer.stop();
	img_timer.start();
}
	


/*
int main(int argc, char* argv[])
{


	std::shared_ptr<const object_parameters> object_params(new object_parameters);
	// Pointcloud preprocessing
	pointcloud_preprocessing prepro(object_params);
	classifier_set classifiers(object_params);
	kinect2_parameters kinect2_params;
	hand_pose_estimation::hand_pose_estimation hand_pose_est;
	hand_pose_estimation::hand_tracker hand_track(0.1f, 5000000);
	hand_pose_estimation::classifier_set gesture_classifiers;
//	std::cout << kinect2_params.rgb_projection << std::endl;

	object_recognition_test obj_rec_test;
	hand_pose_test hand_pos_test;

	auto viewer = boost::make_shared<pcl::visualization::PCLVisualizer>("Viewer");

	// Point Cloud
	pcl::PointCloud<PointT>::ConstPtr cloud;
	pcl::PointCloud<PointT>::ConstPtr prev_cloud;
	std::shared_ptr<cv::Mat4b> img;
	bool updated = false;

	// Retrieved Point Cloud Callback Function
	boost::mutex mutex;
	boost::function<void(const pcl::PointCloud<PointT>::ConstPtr&)> pc_grabber =
		[&cloud, &prev_cloud, &mutex, &updated](const pcl::PointCloud<PointT>::ConstPtr& ptr) {
		boost::mutex::scoped_lock lock(mutex);

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

	boost::function<void(const std::shared_ptr<cv::Mat4b> & input)> img_grabber =
		[&img, &mutex](const std::shared_ptr<cv::Mat4b>& input) {
		boost::mutex::scoped_lock lock(mutex);

		img = input;
	};

	// Kinect2Grabber
	boost::shared_ptr<pcl::Grabber> grabber = boost::make_shared<pcl::Kinect2Grabber>();

	// Register Callback Function
	boost::signals2::connection connection_pc = grabber->registerCallback(pc_grabber);
	boost::signals2::connection connection_img = grabber->registerCallback(img_grabber);


	// Start Grabber
	grabber->start();

	// Create a window for display.

	boost::timer::cpu_timer timer;
	boost::timer::nanosecond_type total_time = 0;
	int image_counter = 0;

	while (!viewer->wasStopped()) {
		// Update Viewer
		try {
			viewer->spinOnce();
		}
		catch (...) {
			continue;
		}
		
		pcl::PointCloud<PointT>::ConstPtr temp_cloud = nullptr;
		pcl::PointCloud<PointT>::ConstPtr temp_prev_cloud = nullptr;
		bool temp_updated = false;
		std::shared_ptr<cv::Mat4b> temp_img = nullptr;

		{
			boost::mutex::scoped_try_lock lock(mutex);
			if (lock.owns_lock() && updated && cloud->size()) {
				temp_cloud = cloud;
				temp_prev_cloud = prev_cloud;
				temp_updated = true;
			}

			if (lock.owns_lock() && img && !img->empty()) {
				temp_img = img;
			}
		}	



		if (temp_cloud)
		{
//			obj_rec_test.compute_and_show_clusters(viewer, prepro, temp_cloud);
		}

		if(temp_updated && temp_prev_cloud) 
		{
//			obj_rec_test.compute_and_show_classified_objects(viewer, prepro, classifiers, prepro.fuse(temp_prev_cloud,  temp_cloud));
		}

		
		if (temp_img && temp_cloud && temp_updated)
		{
			timer.start();

//			cv::imshow("Display window", obj_rec_test.find_shapes(temp_img));
			hand_pos_test.show_hand_tracking(viewer, prepro, kinect2_params, hand_pose_est, hand_track, *temp_img, temp_cloud);
		
			if (++image_counter >= 30) {
				std::cout << image_counter / (double)total_time * 1e9 << " FPS" << std::endl;
				image_counter = 0;
				total_time = 0;
			}
			else {
				total_time += timer.elapsed().wall;
			}
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
*/

void viewer::schedule(std::function<void()>&& f, unsigned int priority)
{
	std::lock_guard<std::mutex> lock(queue_mutex);
	queue.emplace_back( f);
	queue_condition_variable.notify_all();
}
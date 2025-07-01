#include "hand_pose.hpp"

#include <iostream>
#include <vector>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "opencv-serialization/opencv_serialization.hpp"

#include "hand_pose_estimation/hand_pose_estimation.h"
#include "hand_pose_estimation/skin_detection.hpp"



using namespace hand_pose_estimation;

void hand_pose_test::show_hand_keypoints(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const state_observation::kinect2_parameters& kinect2_params,
	::hand_pose_estimation::hand_pose_estimation& hand_pose_est,
	const visual_input& input) const
{

	viewer->removeAllShapes();
	viewer->removeAllPointClouds();



	std::vector<img_segment::Ptr> hands = hand_pose_est.detect_hands(input, std::vector<hand_instance::Ptr>());

	/// Draw contours + rotated rects + ellipses
	cv::RNG rng(12345);
	cv::Mat drawing = input.img.clone();
	for (const auto& hand : hands)
	{
		if (hand_pose_est.estimate_keypoints(input.img, *hand))
		{

			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			// rectangle
			cv::Point2f rect_points[4];
			const auto& box = hand->net_evals[0]->input_box;

			cv::rectangle(drawing, box, color, 2, 8);

			draw_keypoints_2d(drawing, hand->net_evals[0]->key_points_2d);
		}
	}


	cv::imshow("Hand box", drawing);

	//viewer->removePointCloud("normals");
	//viewer->addPointCloudNormals<pcl::PointNormal>(normals, 100, 0.02f, "normals");

	if (!viewer->updatePointCloud(input.cloud))
		viewer->addPointCloud(input.cloud);
}





void hand_pose_test::show_hand_tracking(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const state_observation::kinect2_parameters& kinect2_params,
	::hand_pose_estimation::hand_pose_estimation& hand_pose_est,
	bounding_box_tracker& hand_track,
	gradient_decent_scheduler& optimizer,
	const visual_input::ConstPtr& input) const
{

	quality_key_points_below_surface criterion_below_surface(1.f);

	auto start = std::chrono::high_resolution_clock::now();
	std::vector<img_segment::Ptr> hand_segments = hand_pose_est.detect_hands(*input, hand_track.get_hands());
	std::cout << "skin detection  " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;

	start = std::chrono::high_resolution_clock::now();
	hand_track.update(*input, hand_segments);
	std::cout << "2d tracking  " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;
	std::vector<hand_instance::Ptr> hands = hand_track.get_hands();



	start = std::chrono::high_resolution_clock::now();
	optimizer.update(input, hands);
	std::cout << "optimization prep  " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;


	show_hands(viewer, input, hands);
}

void hand_pose_test::show_hands(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const visual_input::ConstPtr& input,
	const std::vector<hand_instance::Ptr>& hands) const
{
	auto start = std::chrono::high_resolution_clock::now();
	viewer->removeAllShapes();
	viewer->removeAllPointClouds();

	/// Draw contours + rects
	cv::RNG rng(12345);
	cv::Mat drawing = input->img.clone();

	auto hand_keypoints = pcl::make_shared<pcl::PointCloud<visual_input::PointT>>();

	for (const hand_instance::Ptr& hand : hands)
	{

		if (hand->certainty_score > 0.3f && hand->poses.size() >= 2)
		{

			auto& seg = *(++hand->poses.rbegin());

			size_t id = std::hash<hand_instance*>{}(&*hand);
			cv::Scalar color = cv::Vec3b(63 * (id / 16 % 4), 63 * (id / 4 % 4), 63 * (id % 4));
		

			// display the bounding box
			auto box = seg.get_box(*input);
			cv::rectangle(drawing, box, color, 2);

			// display certainty of being a hand and whether it's a left or right hand
			cv::Point2i center(box.x + box.width / 2, box.y + box.height / 2);
			cv::putText(drawing, cv::format(hand->right_hand > 0.5f ? "r%.2f" : "l%.2f", hand->certainty_score.load()), center, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 0), 2);

			// display the 2D key points estimated by the neural network
			if (seg.net_eval)
				draw_keypoints_2d(drawing, seg.net_eval->key_points_2d, color);

			// display the skeleton model
			// the kinematic parameters are stored in seg.pose
			// whereas the derived vector of key points is cached in seg.key_points
			// see assets/handpose-demo-keypoints for identification (applies for left and right hands)
			// e.g. the tip of the index finger is always 8
			draw_keypoints_3d(viewer,
				hand->poses.back().key_points,
				id, color, hand_keypoints);
		}
	}

	cv::imshow("Hand box", drawing);

	if (!viewer->updatePointCloud(input->cloud))
		viewer->addPointCloud(input->cloud);

	if (!viewer->updatePointCloud(hand_keypoints, "hands"))
		viewer->addPointCloud(hand_keypoints, "hands");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "hands");
	std::cout << "rendering  " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;
}


void hand_pose_test::show_gestures(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const state_observation::kinect2_parameters& kinect2_params,
	::hand_pose_estimation::hand_pose_estimation& hand_pose_est,
	bounding_box_tracker& hand_track,
	classifier_set& gesture_classifiers,
	const visual_input& input) const
{
	pcl::PointCloud<PointT>::Ptr hand_keypoints(new pcl::PointCloud<PointT>);

	viewer->removeAllShapes();
	viewer->removeAllPointClouds();


	std::vector<img_segment::Ptr> hand_segments = hand_pose_est.detect_hands(input, hand_track.get_hands());
	hand_track.update(input.img, hand_segments);
	std::vector<hand_instance::Ptr> hands = hand_track.get_hands();

	cv::RNG rng(12345);
	cv::Mat drawing = input.img.clone();
	for (const hand_instance::Ptr& hand : hands)

		for (auto& hand : hands)
		{
			if (hand->certainty_score > 0.5f)// && hand->observation_history.back()->timestamp == hand_track.get_latest_timestamp())
			{

				size_t id = std::hash<hand_instance*>{}(&*hand);
				cv::Scalar color = cv::Vec3b(127 * (id / 16 % 4), 127 * (id / 4 % 4), 127 * (id % 4));
				//std::cout << id << " ";

				// rectangle
				const img_segment& seg = *hand->observation_history.back();
				cv::rectangle(drawing, seg.bounding_box, color, 2);
				cv::circle(drawing, seg.palm_center_2d, 8, cv::Scalar(0, 0, 255), -1);


				float certainty = 0.f;

				for (auto& classifier : gesture_classifiers)
				{

					certainty = classifier->classify(*hand).certainty_score;
					if (certainty > 0.5f)
					{
						break;
					}
				}


				cv::drawContours(drawing, std::vector<std::vector<cv::Point2i>>({ seg.contour }), -1, color, certainty > 0.5f ? -1 : 2);
				cv::putText(drawing, cv::format("%.2f", certainty), hand->observation_history.back()->palm_center_2d, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 0), 2);

				const Eigen::Matrix3Xf finger_tips = hand->poses.back().pose.get_tips();
				for (int i = 0; i < finger_tips.cols(); i++)
					cv::circle(drawing, input.to_img_coordinates(finger_tips.col(i)), 8, cv::Scalar(0, 0, 255), -1);

			}
		}


	//std::cout << std::endl;


	cv::imshow("Hand box", drawing);

	//viewer->removePointCloud("normals");
	//viewer->addPointCloudNormals<pcl::PointNormal>(normals, 100, 0.02f, "normals");

	if (!viewer->updatePointCloud(input.cloud))
		viewer->addPointCloud(input.cloud);

	if (!viewer->updatePointCloud(hand_keypoints, "hands"))
		viewer->addPointCloud(hand_keypoints, "hands");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 8, "hands");
}

Eigen::Matrix<float, 3, 4> hand_pose_test::projection_matrix(const state_observation::kinect2_parameters& kinect2_params,
	const cv::MatSize& size) const
{
	// kinect2_params.rgb_projection has its origin in the bottom right of the image, we need it in the top left
	Eigen::Translation2f center(size[1] * 0.5f, size[0] * 0.5f);
	return  center * Eigen::Affine2f(Eigen::Rotation2D<float>(M_PI)) * center.inverse() * kinect2_params.rgb_projection;
}

void hand_pose_test::show_projection(state_observation::kinect2_parameters& kinect2_params, const visual_input& input) const
{
	//	cloud = prepro.remove_table(cloud);
	const auto& img = input.img;
	const auto& cloud = input.cloud;

	//	Eigen::Matrix<float, 3, 4> projection = projection_matrix(prepro, kinect2_params, img.size);
	//Eigen::Translation2f center(img.size[1] * 0.5f, img.size[0] * 0.5f);
	//Eigen::Matrix<float, 3, 4> projection = center * Eigen::Affine2f(Eigen::Rotation2D<float>(M_PI)) * center.inverse() * kinect2_params.rgb_projection;
	cv::Mat output_image = img.clone();

	float min_x, min_y, max_x, max_y;
	min_x = min_y = -0.5f;
	max_x = output_image.cols - 0.5f;
	max_y = output_image.rows - 0.5f;

	for (const PointT& p : *cloud)
	{
		Eigen::Vector2f p_img = (kinect2_params.rgb_projection * Eigen::Vector3f(p.data).homogeneous()).hnormalized();
		if (p_img.x() > min_x && p_img.x() < max_x && p_img.y() > min_y && p_img.y() < max_y)
			output_image.at<cv::Vec4b>(static_cast<int>(std::round(p_img.y())), static_cast<int>(std::round(p_img.x()))) = cv::Vec4b(p.getBGRAVector4cMap().data());
	}

	cv::imshow("Projection", output_image);
}

void hand_pose_test::save_contours(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const state_observation::kinect2_parameters& kinect2_params,
	::hand_pose_estimation::hand_pose_estimation& hand_pose_est,
	bounding_box_tracker& hand_track,
	const visual_input& input) const
{


	std::vector<std::pair<cv::Vec4b, std::vector<cv::Point>>> colors_contours;

	std::vector<img_segment::Ptr> hand_segments = hand_pose_est.detect_hands(input, std::vector<hand_instance::Ptr>());
	hand_track.update(input, hand_segments);

	cv::Mat drawing = input.img.clone();


#undef RGB
	for (const hand_instance::Ptr& hand : hand_track.get_hands()) {
		if (hand->certainty_score > 0.25f)
		{
			size_t id = std::hash<size_t>{}(hand->get_id());
			cv::Vec4b color = cv::Vec4b(85 * (id / 16 % 4), 85 * (id / 4 % 4), 85 * (id % 4), 255);

			colors_contours.push_back(std::make_pair(
				color, hand->observation_history.back()->contour
			));

			cv::drawContours(drawing, std::vector<std::vector<cv::Point2i>>({ hand->observation_history.back()->contour }), -1, color, 2);
		}
	}

	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();

	std::stringstream stream;

	stream << "hand_contours_" << time.date().year()
		<< "-" << time.date().month()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes();

	std::string name(stream.str());

	std::ofstream file(name + ".xml");
	boost::archive::xml_oarchive oa{ file };
	oa << BOOST_SERIALIZATION_NVP(colors_contours);

	cv::imwrite(name + ".png", drawing, std::vector<int>({ cv::IMWRITE_PNG_COMPRESSION }));
	cv::imshow("Hand detection", drawing);
}

void hand_pose_test::show_templates(classifier_set& gesture_classifiers) const
{

	for (auto& classifier : gesture_classifiers)
	{
		cv::RNG rng(12345);
		cv::Mat drawing(1080, 1920, CV_8UC3);
		int i = 0;
		for (auto& templ : classifier->get_object_prototype()->get_templates())
		{
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			cv::drawContours(drawing, std::vector<std::vector<cv::Point>>({ templ }), 0, color, 2);

			cv::Moments hu_moments(cv::moments(templ));
			cv::Point2i center_2d(hu_moments.m10 / hu_moments.m00, hu_moments.m01 / hu_moments.m00);
			cv::putText(drawing, cv::format("%d", i++), center_2d, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);
		}

		cv::imshow(classifier->get_object_prototype()->get_name(), drawing);
	}
}

void hand_pose_test::draw_keypoints_2d(cv::Mat& canvas,
	const std::vector<cv::Point2i>& key_points,
	const cv::Scalar& color) const
{
	for (int i = 0; i < key_points.size(); i++)
	{

		if (key_points[i].x == -1)
			continue;

		cv::circle(canvas, key_points[i], 4, color, cv::FILLED);

		if (i)
		{
			if (i % 4 == 1)
			{
				if (key_points[0].x != -1)
					cv::line(canvas, key_points.front(), key_points[i], color, 2);
			}
			else if (key_points[i - 1].x != -1)
				cv::line(canvas, key_points[i - 1], key_points[i], color, 2);
		}
	}
}


void hand_pose_test::draw_keypoints_3d(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const Eigen::Matrix3Xf& key_points,
	size_t id,
	const cv::Scalar& color,
	const pcl::PointCloud<PointT>::Ptr& skeleton_cloud) const
{


	std::vector<PointT> points;
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

		if (skeleton_cloud)
			skeleton_cloud->push_back(p);

		if (points.size())
		{
			if (i % 4 == 1)
			{
				viewer->addLine(points.front(), p, p.r, p.g, p.b, std::string("line") + std::to_string(i) + std::to_string(id));
			}
			else
				viewer->addLine(points.back(), p, p.r, p.g, p.b, std::string("line") + std::to_string(i) + std::to_string(id));
		}

		points.push_back(p);
	}


}

void hand_pose_test::draw_pose_estimation_points(const visual_input& input,
	::hand_pose_estimation::hand_pose_estimation& hand_pose_est,
	std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const net_evaluation& net_eval,
	size_t id,
	const cv::Scalar& color,
	const pcl::PointCloud<PointT>::Ptr& skeleton_cloud) const
{
	std::vector<PointT> points;
	for (int joint : {0, 2, 5, 9, 17})
	{
		if (net_eval.key_points_2d.at(joint).x == -1)
			continue;

		cv::Point2i cloud_p = input.to_cloud_pixel_coordinates(net_eval.key_points_2d.at(joint));
		if (!input.is_valid_point(cloud_p))
			continue;

		Eigen::Vector3f key_point = input.get_point(cloud_p).getVector3fMap();

		PointT p;
		p.x = key_point.x();
		p.y = key_point.y();
		p.z = key_point.z();
		p.r = color(2); p.g = color(1); p.b = color(0); p.a = 255;

		if (skeleton_cloud)
			skeleton_cloud->push_back(p);

		if (points.size())
		{
			viewer->addLine(points.front(), p, p.r, p.g, p.b, std::string("pose_est_line0") + std::to_string(joint) + std::to_string(id));
			viewer->addLine(points.back(), p, p.r, p.g, p.b, std::string("pose_est_line1") + std::to_string(joint) + std::to_string(id));
		}

		points.push_back(p);
	}


}

void hand_pose_test::show_fused_heatmap(net_evaluation& seg) const
{

	cv::Mat fuse(seg.maps.at(0).rows, seg.maps.at(0).cols, CV_32FC1);
	for (const auto map : seg.maps)
		fuse = cv::max(fuse, map);

	cv::Mat fuse_gray(seg.maps.at(0).rows, seg.maps.at(0).cols, CV_8UC1);
	fuse.convertTo(fuse_gray, CV_8UC1, 255);
	cv::imshow("Max heatmaps", fuse_gray);
}

void hand_pose_test::show_demo_hand(hand_kinematic_parameters& hand_kin_params,
	std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const cv::Scalar& color) const
{
	auto pc = pcl::make_shared<pcl::PointCloud<PointT>>();

	for (int i = 0; i < 5; i++)
	{
		std::vector<PointT> points;

		for (int j = 0; j < 4; j++)
		{
			std::vector<float> angles(j + 1, 0);
			if (i != 0)
			{
				angles[0] = M_PI_4;
				if (j >= 1)
					angles[1] = -M_PI_4;
				if (j >= 2)
					angles[2] = -M_PI_2 / 3;
				if (j >= 3)
					angles[3] = -M_PI_4 / 3;
			}
			else
			{
				angles[0] = -M_PI_4;
				if (j >= 1)
					angles[1] = -M_PI_4;
				if (j >= 2)
					angles[2] = -M_PI_2 / 3;
				if (j >= 3)
					angles[3] = -M_PI_4 / 3;
			}

			auto key_point =
				hand_kin_params.fingers[i].transformation_to_base(angles) * Eigen::Vector3f::Zero();

			PointT p;
			p.x = key_point.x();
			p.y = key_point.y();
			p.z = key_point.z();
			p.r = color(2); p.g = color(1); p.b = color(0);

			if (points.size())
			{
				viewer->addLine(points.back(), p, 255, 255, 255, std::string("line") + std::to_string(j + 1) + std::to_string(i));
			}

			pc->push_back(p);
			points.push_back(p);
		}

		PointT p;
		p.x = p.y = p.z = 0.f;
		p.r = p.g = p.b = p.a = 255;
		viewer->addLine(points.front(), p, 255, 255, 255, std::string("line0") + std::to_string(i));
	}
	viewer->addPointCloud(pc);
	//viewer->addCoordinateSystem();
}

void hand_pose_test::show_normals(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const pcl::PointCloud<PointT>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>& normals) const
{
	auto combined = pcl::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();

	auto n = normals.begin();
	for (auto p = cloud->begin(); p != cloud->end() && n != normals.end(); ++p, ++n)
	{
		pcl::PointXYZRGBNormal c;
		c.x = p->x;
		c.y = p->y;
		c.z = p->z;
		c.r = p->r;
		c.g = p->g;
		c.b = p->b;
		c.normal_x = n->normal_x;
		c.normal_y = n->normal_y;
		c.normal_z = n->normal_z;

		combined->push_back(c);
	}

	if (!viewer->updatePointCloud(cloud))
		viewer->addPointCloud(cloud);
	viewer->addPointCloudNormals<pcl::PointXYZRGBNormal>(combined, 2, 0.02, "normals");
}

void hand_pose_test::show_index_finger_tip_3d(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const visual_input& input,
	const img_segment& seg) const
{
	if (seg.net_evals[1]->certainties[8] < 0.1f)
		return;

	auto cloud = std::make_shared<pcl::PointCloud<PointT>>();


	PointT p;
	seg.prop_3d->get_surface_point_img(seg.net_evals[1]->key_points_2d[8], p);
	p.g = p.a = 255;

	cloud->push_back(p);
	const std::string id("index finger tip");
	if (!viewer->updatePointCloud(cloud, id))
	{
		viewer->addPointCloud(cloud, id);

	}
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, id);

}

void hand_pose_test::show_surface_distances(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
	const visual_input& input,
	const img_segment& seg, const Eigen::Matrix3Xf& key_points, size_t id,
	const pcl::PointCloud<PointT>::Ptr& skeleton_cloud) const
{
	float radius = quality_key_points_below_surface::finger_radius(seg.particles.front() ? seg.particles.front()->pose : seg.particles.back()->pose);

	for (int i = 0; i < 21; i++) {
		Eigen::Vector3f key_point = key_points.col(i);

		visual_input::PointT skeleton_p;
		skeleton_p.x = key_point.x();
		skeleton_p.y = key_point.y();
		skeleton_p.z = key_point.z();

		pcl::Normal normal;
		visual_input::PointT p;
		visual_input::PointT best_p;
		float best_dist = std::numeric_limits<float>::infinity();

		seg.prop_3d->get_surface_point(input, key_point, p, &normal);
		Eigen::Vector3f n = normal.getNormalVector3fMap();

		float dist = n.dot(key_point) - n.dot(p.getVector3fMap());
		if (!std::isfinite(dist))
			dist = (key_point - p.getVector3fMap()).norm();

		viewer->addLine(skeleton_p, p, 255, 255, 255, std::string("distance_1_") + std::to_string(id) + std::to_string(i));

		if (std::abs(dist) < best_dist)
		{
			best_dist = std::abs(dist);
			best_p = p;
		}

		Eigen::Vector3f key_point_n = key_point.normalized();
		Eigen::Vector3f orthogonal = (n - n.dot(key_point_n) * key_point_n).normalized();


		seg.prop_3d->get_surface_point(input, 0.5f * radius * orthogonal + key_point, p, &normal);
		n = normal.getNormalVector3fMap();

		dist = n.dot(key_point) - n.dot(p.getVector3fMap());
		if (!std::isfinite(dist))
			dist = (key_point - p.getVector3fMap()).norm();

		if (std::abs(dist) < best_dist)
		{
			best_dist = std::abs(dist);
			best_p = p;
		}

		seg.prop_3d->get_surface_point(input, radius * orthogonal + key_point, p, &normal);
		n = normal.getNormalVector3fMap().normalized();

		dist = n.dot(key_point) - n.dot(p.getVector3fMap());
		if (!std::isfinite(dist))
			dist = (key_point - p.getVector3fMap()).norm();

		if (std::abs(dist) < best_dist)
		{
			best_p = p;
		}


		viewer->addLine(skeleton_p, best_p, 255, 255, 0, std::string("distance_b_") + std::to_string(id) + std::to_string(i));

	}
}

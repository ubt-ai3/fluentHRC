#pragma once

#include <memory>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include "state_observation/calibration.hpp"

#include "hand_pose_estimation/hand_pose_estimation.h"
#include "hand_pose_estimation/bounding_box_tracking.hpp"
#include "hand_pose_estimation/gradient_decent.hpp"
#include "hand_pose_estimation/classification_handler.hpp"



class hand_pose_test
{
public:
	typedef pcl::PointXYZRGBA PointT;

	void show_hand_keypoints(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const state_observation::kinect2_parameters& kinect2_params,
		hand_pose_estimation::hand_pose_estimation& hand_pose_est,
		const hand_pose_estimation::visual_input& input) const;

	void show_hand_tracking(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const state_observation::kinect2_parameters& kinect2_params,
		hand_pose_estimation::hand_pose_estimation& hand_pose_est,
		hand_pose_estimation::bounding_box_tracker& hand_track,
		hand_pose_estimation::gradient_decent_scheduler& optimizer,
		const hand_pose_estimation::visual_input::ConstPtr& input) const;

	void show_hands(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const hand_pose_estimation::visual_input::ConstPtr& input,
		const std::vector<hand_pose_estimation::hand_instance::Ptr>& hands) const;
	
	void show_gestures(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const state_observation::kinect2_parameters& kinect2_params,
		hand_pose_estimation::hand_pose_estimation& hand_pose_est,
		hand_pose_estimation::bounding_box_tracker& hand_track,
		hand_pose_estimation::classifier_set& gesture_classifiers,
		const hand_pose_estimation::visual_input& input) const;

	Eigen::Matrix<float, 3, 4> projection_matrix(const state_observation::kinect2_parameters & kinect2_params,
		const cv::MatSize& size) const;

	void show_projection(state_observation::kinect2_parameters& kinect2_params,
	                     const hand_pose_estimation::visual_input& input) const;

	void save_contours(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const state_observation::kinect2_parameters& kinect2_params,
		hand_pose_estimation::hand_pose_estimation& hand_pose_est,
		hand_pose_estimation::bounding_box_tracker& hand_track,
		const hand_pose_estimation::visual_input& input) const;

	void show_templates(hand_pose_estimation::classifier_set& gesture_classifiers) const;

	void draw_keypoints_2d(cv::Mat& canvas, 
		const std::vector<cv::Point2i>& key_points, 
		const cv::Scalar& color = cv::Scalar(0,255,0)) const;

	void draw_keypoints_3d(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const Eigen::Matrix3Xf& key_points,
		size_t id = 0,
		const cv::Scalar& color = cv::Scalar(0, 255, 0),
		const pcl::PointCloud<PointT>::Ptr& skeleton_cloud = nullptr) const;

	void draw_pose_estimation_points(const hand_pose_estimation::visual_input& input,
		hand_pose_estimation::hand_pose_estimation& hand_pose_est,
		std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const hand_pose_estimation::net_evaluation& net_eval,
		size_t id = 0,
		const cv::Scalar& color = cv::Scalar(0, 255, 0),
		const pcl::PointCloud<PointT>::Ptr& skeleton_cloud = nullptr) const;

	void show_fused_heatmap(hand_pose_estimation::net_evaluation& seg) const;

	void show_demo_hand(hand_pose_estimation::hand_kinematic_parameters& hand_kin_params,
		std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const cv::Scalar& color = cv::Scalar(255, 255, 255)) const;

	void show_normals(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const pcl::PointCloud<hand_pose_estimation::visual_input::PointT>::Ptr& cloud,
		const pcl::PointCloud<pcl::Normal>& normals) const;

	void show_index_finger_tip_3d(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const hand_pose_estimation::visual_input& input,
		const hand_pose_estimation::img_segment& seg) const;

	void show_surface_distances(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		const hand_pose_estimation::visual_input& input,
		const hand_pose_estimation::img_segment& seg,
		const Eigen::Matrix3Xf& key_points,
		size_t id,
		const pcl::PointCloud<PointT>::Ptr& skeleton_cloud) const;
};


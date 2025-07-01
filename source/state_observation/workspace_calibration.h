#pragma once

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <state_observation/pn_model_extension.hpp>

namespace state_observation
{

	struct STATEOBSERVATION_API PlaneResult
	{
		Eigen::Affine3f transform;
		pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud;
	};

	STATEOBSERVATION_API Eigen::Affine3f kinect_2_robot();

	STATEOBSERVATION_API PlaneResult plane_estimation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud);

	STATEOBSERVATION_API Eigen::Matrix<float, 3, Eigen::Dynamic> convert(const std::vector<Eigen::Vector3f>& in);

	STATEOBSERVATION_API PlaneResult coarse_calibration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, const Eigen::Affine3f& initial_transform = Eigen::Affine3f::Identity());

	STATEOBSERVATION_API PlaneResult coarse_calibration(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& point_cloud, const Eigen::Affine3f& initial_transform = Eigen::Affine3f::Identity());

	STATEOBSERVATION_API Eigen::Affine3f coarse_calibration_with_robot(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, const Eigen::Affine3f& initial_transform = Eigen::Affine3f::Identity());

	STATEOBSERVATION_API Eigen::Affine3f calibrate_by_points(const std::vector<Eigen::Vector3f>& should, const std::vector<Eigen::Vector3f>& is);

	STATEOBSERVATION_API Eigen::Affine3f fine_calibration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, std::vector<state_observation::pn_boxed_place::Ptr>& box_places);

	STATEOBSERVATION_API Eigen::Affine3f full_calibration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, std::vector<state_observation::pn_boxed_place::Ptr>& box_places, state_observation::pointcloud_preprocessing& pc_prepro);
}
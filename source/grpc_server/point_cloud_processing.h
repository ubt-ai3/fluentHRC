#pragma once
#include <ranges>

#include <boost/asio/thread_pool.hpp>
#include <boost/signals2/signal.hpp>
#include <pcl/point_cloud.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include <pcl/filters/passthrough.h>

#include "util.h"
#include <state_observation/workspace_calibration.h>

class incremental_point_cloud
{
	typedef pcl::PointCloud<pcl::PointXYZ> point_cloud;
public:

	typedef std::shared_ptr<incremental_point_cloud> Ptr;
	
	incremental_point_cloud(size_t threads = std::thread::hardware_concurrency() - 1);
	~incremental_point_cloud();
	
	void insert(server::holo_pointcloud::Ptr&& pcl_data);
	void save(const std::string& filename) const;
	
	/*
	 * mtx must be locked prior to access
	 */
	pcl::PointCloud<pcl::PointXYZ>::ConstPtr get_pcl() const;
	
	template<class PointType>
	Eigen::Affine3f register_pcl(typename pcl::PointCloud<PointType>::ConstPtr cloud)
	{
		if (!cloud)
			return Eigen::Affine3f::Identity();
		
		std::unique_lock lock(mtx);
		auto kinect_preprocessed = 
			pcl::make_shared<pcl::PointCloud<PointType>>(*cloud);
		
		auto cloud_hololens_preprocessed = 
			pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*pcl);
		
		auto cloud_registered =
			pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

		{
			pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
			voxel_grid.setInputCloud(kinect_preprocessed);
			voxel_grid.setLeafSize(0.007f, 0.007f, 0.007f);
			voxel_grid.filter(*kinect_preprocessed);
		}
		{
			pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
			sor.setInputCloud(kinect_preprocessed);
			sor.setMeanK(50);
			sor.setStddevMulThresh(1.0);
			sor.filter(*kinect_preprocessed);
		}
		{
			pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
			sor.setInputCloud(cloud_hololens_preprocessed);
			sor.setMeanK(50);
			sor.setStddevMulThresh(1.0);
			sor.filter(*cloud_hololens_preprocessed);
		}
		if (cloud_hololens_preprocessed->points.empty())
			return Eigen::Affine3f::Identity();
		
		/*Eigen::Vector4f centroid;
		compute3DCentroid(*cloud_hololens_preprocessed, centroid);*/

		auto max_z_holo = std::ranges::max_element(
			cloud_hololens_preprocessed->points, 
			std::ranges::less{}, [](const pcl::PointXYZ& p)
			{
				return p.z;
			})->z;

		auto max_z_kinect = std::ranges::max_element(kinect_preprocessed->points, 
			std::ranges::less{}, [](const pcl::PointXYZ& p)
			{
				return p.z;
			})->z;

		/*
		* Align with top of robot
		*/
		auto diff = max_z_kinect - max_z_holo;

		pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
		feature_extractor.setInputCloud(cloud_hololens_preprocessed);
		feature_extractor.compute();

		pcl::PointXYZ min_point;
		pcl::PointXYZ max_point;
		pcl::PointXYZ position;
		Eigen::Matrix3f rotation;
		feature_extractor.getOBB(min_point, max_point, position, rotation);

		float yaw = rotation.eulerAngles(0, 1, 2).z();

		if (max_point.x - min_point.x >
			max_point.y - min_point.y)
			yaw += M_PI_2;

		auto inv = Eigen::Affine3f(Eigen::Quaternionf(Eigen::AngleAxisf(-yaw, Eigen::Vector3f::UnitZ()))) * 
			Eigen::Affine3f(Eigen::Translation3f(-position.x, -position.y, diff));
		pcl::transformPointCloud(*cloud_hololens_preprocessed, *cloud_hololens_preprocessed, inv);

		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		// Set the input source and target
		icp.setInputSource(cloud_hololens_preprocessed);
		icp.setInputTarget(kinect_preprocessed);

		// Set the max correspondence distance to 5cm (e.g., correspondences with higher
		// distances will be ignored)
		icp.setMaxCorrespondenceDistance(0.6);
		// Set the maximum number of iterations (criterion 1)
		icp.setMaximumIterations(100);
		// Set the transformation epsilon (criterion 2)
		icp.setTransformationEpsilon(1e-9);
		// Set the euclidean distance difference epsilon (criterion 3)
		icp.setEuclideanFitnessEpsilon(1e-7);
		// Perform the alignment
		icp.align(*cloud_registered);
		
		return Eigen::Affine3f(icp.getFinalTransformation()) * inv;
	}
	
	template<class PointType>
	Eigen::Affine3f register_pcl(
		typename pcl::PointCloud<PointType>::ConstPtr cloud,
		const state_observation::obb& bb)
	{
		if (!cloud)
			return Eigen::Affine3f::Identity();

		std::unique_lock lock(mtx);
		auto kinect_preprocessed =
			pcl::make_shared<pcl::PointCloud<PointType>>(*cloud);

		auto cloud_hololens_preprocessed =
			pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>(*pcl);

		auto cloud_registered =
			pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

		{
			pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
			voxel_grid.setInputCloud(kinect_preprocessed);
			voxel_grid.setLeafSize(0.007f, 0.007f, 0.007f);
			voxel_grid.filter(*kinect_preprocessed);
		}
		{
			pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
			sor.setInputCloud(kinect_preprocessed);
			sor.setMeanK(50);
			sor.setStddevMulThresh(1.0);
			sor.filter(*kinect_preprocessed);
		}
		{
			pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
			sor.setInputCloud(cloud_hololens_preprocessed);
			sor.setMeanK(50);
			sor.setStddevMulThresh(1.0);
			sor.filter(*cloud_hololens_preprocessed);
		}
		if (cloud_hololens_preprocessed->points.empty())
			return Eigen::Affine3f::Identity();

		auto max_z_holo = std::ranges::max_element(
			cloud_hololens_preprocessed->points,
			std::ranges::less{}, [](const pcl::PointXYZ& p)
			{
				return p.z;
			})->z;

		auto max_z_kinect = std::ranges::max_element(
			kinect_preprocessed->points,
			std::ranges::less{}, [](const pcl::PointXYZ& p)
			{
				return p.z;
			})->z;

		/*
		* Align with top of robot
		*/
		auto diff = max_z_kinect - max_z_holo;

		Eigen::Affine3f obb =
			Eigen::Translation3f(bb.translation.x(), bb.translation.y(), 0.f) *
			Eigen::Affine3f(bb.rotation.normalized());

		auto inv = Eigen::Translation3f(0.35f, 0.f, diff) * obb.inverse();
		pcl::transformPointCloud(*cloud_hololens_preprocessed, *cloud_hololens_preprocessed, inv);

		pcl::PassThrough<pcl::PointXYZ> pass;
		pass.setInputCloud(cloud_hololens_preprocessed);
		pass.setFilterFieldName("z");
		pass.setFilterLimits(-0.05, 0.1f);
		pass.filter(*cloud_hololens_preprocessed);


		pcl::PassThrough<pcl::PointXYZ> pass2;
		pass.setInputCloud(kinect_preprocessed);
		pass.setFilterFieldName("z");
		pass.setFilterLimits(-0.05, 0.1f);
		pass.filter(*kinect_preprocessed);

		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		// Set the input source and target
		icp.setInputSource(cloud_hololens_preprocessed);
		icp.setInputTarget(kinect_preprocessed);

		// Set the max correspondence distance to 5cm (e.g., correspondences with higher
		// distances will be ignored)
		icp.setMaxCorrespondenceDistance(0.6);
		// Set the maximum number of iterations (criterion 1)
		icp.setMaximumIterations(1000);
		// Set the transformation epsilon (criterion 2)
		icp.setTransformationEpsilon(1e-11);
		// Set the euclidean distance difference epsilon (criterion 3)
		icp.setEuclideanFitnessEpsilon(1e-5);
		// Perform the alignment
		icp.align(*cloud_registered);

		return Eigen::Affine3f(icp.getFinalTransformation()) * inv;
	}


	template<class PointType>
	Eigen::Affine3f register_pcl_2(
		typename pcl::PointCloud<PointType>::ConstPtr kinect_cloud,
		const state_observation::obb& bb, const state_observation::computed_workspace_parameters& workspace_parameters)
	{
		if (!kinect_cloud)
			return Eigen::Affine3f::Identity();

		std::unique_lock lock(mtx);

		auto cloud_registered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

		Eigen::Affine3f obb =
			Eigen::Translation3f(bb.translation.x(), bb.translation.y(), 0.f) *
			Eigen::Affine3f(bb.rotation.normalized());

		auto hololens_prepro = state_observation::coarse_calibration(pcl, obb);
		auto kinect_prepro = state_observation::coarse_calibration(kinect_cloud, state_observation::kinect_2_robot());

		pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
		// Set the input source and target
		icp.setInputSource(hololens_prepro.point_cloud);
		icp.setInputTarget(kinect_prepro.point_cloud);

		// Set the max correspondence distance to 5cm (e.g., correspondences with higher
		// distances will be ignored)
		icp.setMaxCorrespondenceDistance(0.05);
		// Set the maximum number of iterations (criterion 1)
		icp.setMaximumIterations(1000);
		// Set the transformation epsilon (criterion 2)
		icp.setTransformationEpsilon(1e-11);
		// Set the euclidean distance difference epsilon (criterion 3)
		icp.setEuclideanFitnessEpsilon(1e-5);
		// Perform the alignment
		icp.align(*cloud_registered);

		//kinect_2_workspace <- kinect-prelocation_2_kinect-location <- hl-prelocation_2_kinect-prelocation <- hl_2_hl-prelocation
		return workspace_parameters.get_cloud_transformation() * kinect_prepro.transform.inverse() * Eigen::Affine3f{ icp.getFinalTransformation() } * hololens_prepro.transform;
	}


	boost::signals2::signal<void()> changed;
	mutable std::mutex mtx;

private:
	
	boost::asio::thread_pool thread_pool;
	point_cloud::Ptr pcl = pcl::make_shared<point_cloud>();
};

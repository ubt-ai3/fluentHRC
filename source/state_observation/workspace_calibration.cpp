#include "workspace_calibration.h"

#include <numbers>
#include <ranges>

#include <Eigen/Geometry>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>

namespace state_observation
{
const float min = -std::numeric_limits<float>::infinity();
const float max = std::numeric_limits<float>::infinity();

struct extrema {
	Eigen::Vector3f min_x = { max, max, max };
	Eigen::Vector3f min_y = { max, max, max };
	Eigen::Vector3f max_x = { min, min, min };
	Eigen::Vector3f max_y = { min, min, min };

	float width() const
	{
		return max_x.x() - min_x.x();
	}

	float height() const
	{
		return max_y.y() - min_y.y();
	}
};

extrema get_extrema(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	extrema extr;

	for (const auto& p : cloud->points)
	{
		if (p.x < extr.min_x.x())
			extr.min_x = { p.x, p.y, p.z };
		else if (p.x > extr.max_x.x())
			extr.max_x = { p.x, p.y, p.z };

		if (p.y < extr.min_y.y())
			extr.min_y = { p.x, p.y, p.z };
		else if (p.y > extr.max_y.y())
			extr.max_y = { p.x, p.y, p.z };
	}
	return extr;
}

Eigen::Affine3f kinect_2_robot()
{
	return Eigen::Affine3f{ Eigen::Matrix4f{
		{0, 1,  0, 0},
		{1, 0,  0, 0},
		{0, 0, -1, 0},
		{0, 0,  0, 1}
	} };
}

PlaneResult plane_estimation(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud)
{
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	{
		pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
		voxel_grid.setInputCloud(point_cloud);
		voxel_grid.setLeafSize(0.007f, 0.007f, 0.007f);
		voxel_grid.filter(*cloud);
	}
	{
		//get coarse plane segmentation

		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients(false);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setDistanceThreshold(0.1);
		seg.setMaxIterations(20000);

		seg.setInputCloud(cloud);
		seg.segment(*inliers, *coefficients);
	}
	{
		//move points on plane into new cloud
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud);
		extract.setIndices(inliers);
		extract.filter(*cloud_filtered);
	}
	{
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

		tree->setInputCloud(cloud_filtered);
		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
		ec.setClusterTolerance(0.01); // 2cm
		ec.setMinClusterSize(5000);
		ec.setSearchMethod(tree);
		ec.setInputCloud(cloud_filtered);
		ec.extract(cluster_indices);

		if (cluster_indices.empty())
			return { Eigen::Affine3f::Identity(), nullptr };

		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud_filtered); //TODO:: range test
		auto sth = std::make_shared<pcl::PointIndices>(cluster_indices[0]);
		extract.setIndices(sth);
		extract.filter(*cloud_filtered);
		//pcl::copyPointCloud(*cloud_filtered, *cloud_temp);
	}
	/*{
		//remove points which are very distant
		pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
		sor.setInputCloud(cloud_filtered);
		sor.setMeanK(50);
		sor.setStddevMulThresh(1.0);
		sor.filter(*cloud_filtered);
	}*/
	{
		//create final plane segmentation
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setDistanceThreshold(0.01);
		seg.setMaxIterations(1000);

		seg.setInputCloud(cloud_filtered);
		seg.segment(*inliers, *coefficients);
	}
	{
		//final extraction
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(cloud_filtered);
		extract.setIndices(inliers);
		extract.filter(*cloud_filtered);
	}

	//get center point for offset
	pcl::PointXYZ center_p;
	pcl::CentroidPoint<pcl::PointXYZ> center;
	for (const auto& p : cloud_filtered->points)
		center.add(p);
	center.get(center_p);

	Eigen::Vector3f xy_plane_normal = { 0.f, 0.f, 1.f };
	Eigen::Vector3f floor_normal = { coefficients->values[0], coefficients->values[1], coefficients->values[2] };
	floor_normal.normalize();
	Eigen::Vector3f rotation_vec;

	//calculate rotation from normal vector of plane to xy-plane normal
	rotation_vec = xy_plane_normal.cross(floor_normal);
	rotation_vec.normalize();

	auto rotation_angle = -acos(floor_normal.dot(xy_plane_normal));
	Eigen::Affine3f transform = Eigen::AngleAxisf(rotation_angle, rotation_vec) * Eigen::Affine3f(Eigen::Translation3f(-center_p.x, -center_p.y, -center_p.z));

	pcl::transformPointCloud(*cloud_filtered, *cloud_filtered, transform);

	return { transform, cloud_filtered };
}

Eigen::Matrix<float, 3, Eigen::Dynamic> convert(const std::vector<Eigen::Vector3f>& in)
{
	Eigen::Matrix<float, 3, Eigen::Dynamic> out;
	out.conservativeResize(Eigen::NoChange, in.size());

	for (size_t i = 0; i < in.size(); ++i)
		out.col(i) = in[i];

	return out;
}

PlaneResult coarse_calibration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, const Eigen::Affine3f& initial_transform)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::transformPointCloud(*point_cloud, *cloud, initial_transform);

	PlaneResult result = plane_estimation(cloud);
	if (!result.point_cloud)
		return result;

	result.transform = result.transform * initial_transform;

	return result;
}

PlaneResult coarse_calibration(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& point_cloud, const Eigen::Affine3f& initial_transform)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::transformPointCloud(*point_cloud, *cloud, initial_transform);

	PlaneResult result = plane_estimation(cloud);
	if (!result.point_cloud)
		return result;

	result.transform = result.transform * initial_transform;

	return result;
}

Eigen::Affine3f coarse_calibration_with_robot(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, const Eigen::Affine3f& initial_transform)
{
	PlaneResult coarse = coarse_calibration(point_cloud, initial_transform);
	if (!coarse.point_cloud)
		return Eigen::Affine3f::Identity();

	extrema res = get_extrema(coarse.point_cloud);

	const Eigen::Affine3f robot_offset = Eigen::Affine3f(Eigen::Translation3f(-0.205f + (res.width() / 2.f), 0, 0));
	return robot_offset * coarse.transform;
}

Eigen::Affine3f calibrate_by_points(const std::vector<Eigen::Vector3f>& should, const std::vector<Eigen::Vector3f>& is)
{
	//make z-axis align with size of blocks and use it here!!!
	/*Eigen::Matrix3f i = Eigen::umeyama(convert(should), convert(is), false);
	Eigen::Matrix2f rotation = i.block<2, 2>(0, 0);
	Eigen::Vector2f translation = i.block<2, 1>(0, 2);

	Eigen::Matrix4f res = Eigen::Matrix4f::Identity();
	res.block<2, 2>(0, 0) = rotation;
	res.block<2, 1>(0, 3) = translation;*/

	Eigen::Matrix4f res = Eigen::umeyama(convert(should), convert(is), false);
	return Eigen::Affine3f{ res };
}

Eigen::Affine3f fine_calibration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, std::vector<state_observation::pn_boxed_place::Ptr>& box_places)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(point_cloud);

	std::vector<pcl::PointIndices> cluster_indices;

	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

	ec.setClusterTolerance(0.02); // 2cm
	ec.setMinClusterSize(10);
	ec.setSearchMethod(tree);
	ec.setInputCloud(point_cloud);
	ec.extract(cluster_indices);

	std::vector<Eigen::Vector3f> is;
	int i = 0;
	for (const auto& cluster : cluster_indices)
	{
		pcl::CentroidPoint<pcl::PointXYZ> centroid;
		float z = min;

		for (auto idx : cluster.indices)
		{
			centroid.add((*point_cloud)[idx]);
			z = std::max(z, (*point_cloud)[idx].z);
		}

		pcl::PointXYZ center;
		centroid.get(center);

		//Eigen::Affine3f pose = Eigen::Affine3f(Eigen::Translation3f(center.x, center.y, z / 2.f));
		is.emplace_back(center.x, center.y, z / 2.f);
	}

	auto places_point_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
	std::vector<Eigen::Vector3f> should;

	for (const auto& box : box_places)
	{
		const Eigen::Vector3f& t = box->box.translation;
		places_point_cloud->emplace_back(pcl::PointXYZ{ t.x(), t.y(), t.z() });
	}

	pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
	std::vector<int> pointIdxKNNSearch(1);
	std::vector<float> pointKNNSquaredDistance(1);

	kd_tree.setInputCloud(places_point_cloud);
	for (const auto& p : is)
	{
		kd_tree.nearestKSearch(pcl::PointXYZ{ p.x(), p.y(), p.z() }, 1, pointIdxKNNSearch, pointKNNSquaredDistance);
		auto f_p = (*places_point_cloud)[pointIdxKNNSearch[0]];
		should.emplace_back(f_p.x, f_p.y, f_p.z);
	}
	return calibrate_by_points(should, is).inverse();
}

Eigen::Affine3f full_calibration(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, 
	std::vector<state_observation::pn_boxed_place::Ptr>& box_places, state_observation::pointcloud_preprocessing& pc_prepro)
{
	Eigen::Affine3f coarse = coarse_calibration_with_robot(point_cloud, kinect_2_robot());
	if (coarse.matrix() == Eigen::Affine3f::Identity().matrix())
	{
		std::cerr << "too good to be true match in full_calibration" << std::endl;
	}

	auto temp_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
	//pcl::transformPointCloud(*point_cloud, *temp_cloud, coarse);

	const pcl::detail::Transformer<float> tf(coarse.matrix());
	auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

	cloud_filtered->header = point_cloud->header;
	cloud_filtered->is_dense = point_cloud->is_dense;
	cloud_filtered->height = 1;
	// trade-off between avoiding copies and an over-sized vector 
	cloud_filtered->points.reserve(0.3 * point_cloud->size());
	cloud_filtered->sensor_orientation_ = point_cloud->sensor_orientation_;
	cloud_filtered->sensor_origin_ = point_cloud->sensor_origin_;

	auto& points_out = cloud_filtered->points;
	float threshold = pc_prepro.object_params->min_object_height;
	auto min = pc_prepro.workspace_params.crop_box_min;
	auto max = pc_prepro.workspace_params.crop_box_max;
		
	//speedup 2x
	auto result = point_cloud->points
		| std::views::filter([](const pcl::PointXYZ& pIn)
			{
				return std::isfinite(pIn.x) &&
					std::isfinite(pIn.y) &&
					std::isfinite(pIn.z) &&
					pIn.z != 0.f;
			})
		| std::views::transform([&tf](const pcl::PointXYZ& pIn)
			{
				pcl::PointXYZ p;
				tf.se3(pIn.data, p.data);
				return p;
			})
		| std::views::filter([&threshold](const pcl::PointXYZ& p)
			{
				return std::abs(p.z) >= threshold;
			})
		| std::views::filter([&min, &max](const pcl::PointXYZ& p)
					{
						return
							p.x >= min.x() && p.x <= max.x() &&
							p.y >= min.y() && p.y <= max.y() &&
							p.z >= min.z() && p.z <= 0.3;
					});

	for (const auto res : result)
		points_out.emplace_back(res);

	return fine_calibration(cloud_filtered, box_places) * coarse;
}
}
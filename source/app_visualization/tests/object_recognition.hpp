#pragma once

#ifndef TESTS__OBJECT_RECOGNITION
#define TESTS__OBJECT_RECOGNITION

#include <boost/predef/other/endian.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>



#include <state_observation/pointcloud_util.hpp>
#include <state_observation/classification_handler.hpp>

namespace state_observation
{
	
class object_recognition_test
{
public:
	typedef pcl::PointXYZRGBA PointT;

	void compute_and_show_clusters(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		pointcloud_preprocessing& prepro,
		pcl::PointCloud<PointT>::ConstPtr input) const;

	void compute_and_show_bounding_boxes(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		pointcloud_preprocessing& prepro,
		pcl::PointCloud<PointT>::ConstPtr input) const;

	void display(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		pointcloud_preprocessing& prepro,
		const pc_segment& seg,
		const classification_result& classification,
		int index) const;

	void compute_and_show_classified_objects(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		pointcloud_preprocessing& prepro,
		const std::vector<classifier::classifier_aspect>& classifiers,
		pcl::PointCloud<PointT>::ConstPtr input) const;

	void test_pca(pointcloud_preprocessing& prepro) const;

	void show_mesh(pointcloud_preprocessing& prepro) const;

	cv::Mat4b find_shapes(const std::shared_ptr<cv::Mat4b>& src) const;

	cv::Mat4b draw_contoures(const std::vector<std::vector<cv::Point> >& contours, const cv::Size& size) const;

	void test_shape_classifier() const;

	void show_preprocessed_cloud(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer,
		pointcloud_preprocessing& prepro,
		const pcl::PointCloud<PointT>::ConstPtr& input) const;

	void test_rgb_hsv_conversion() const;

	void test_cloud_transformation(pointcloud_preprocessing& prepro) const;
};

} // namespace state_observation

#endif // !TESTS__OBJECT_RECOGNITION

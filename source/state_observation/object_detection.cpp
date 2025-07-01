#include "object_detection.hpp"

#include <pcl/filters/extract_indices.h>
#include <iostream>

namespace state_observation
{

segment_detector::segment_detector(enact_core::world_context& world,
	pointcloud_preprocessing& pc_prepro)
	:
	world(world),
	pc_prepro(pc_prepro),
	cloud_stamp(0)
{
}

segment_detector::~segment_detector() noexcept
{
	stop_thread();
}

void segment_detector::update(const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
	cloud_stamp = cloud->header.stamp;
	
	schedule([this, cloud]() {
		if (cloud_stamp != cloud->header.stamp)
			return;
		
		auto start = std::chrono::high_resolution_clock::now();

		const pcl::IndicesClustersPtr clusters = pc_prepro.conditional_euclidean_clustering(cloud);
		std::vector<pc_segment::Ptr> segments;

		pcl::ExtractIndices<PointT> points_extract;
		points_extract.setInputCloud(cloud);
		points_extract.setNegative(false);

		for (pcl::PointIndices& cluster : *clusters)
		{
			if (cluster.indices.size() < 20)
				continue;

			pc_segment::Ptr seg = std::make_shared<pc_segment>();
			seg->indices = pcl::make_shared<pcl::PointIndices>(std::move(cluster));
			seg->timestamp = std::chrono::microseconds(cloud->header.stamp);

			pcl::PointCloud<PointT>::Ptr object_cloud(new pcl::PointCloud<PointT>());
			points_extract.setIndices(seg->indices);
			points_extract.filter(*object_cloud);

			seg->reference_frame = cloud;
			seg->bounding_box = pc_prepro.oriented_bounding_box_for_standing_objects(object_cloud);
			const auto& bb = seg->bounding_box.diagonal;

			pcl::CentroidPoint<PointT> centroid_computation;
			for (const auto& point : *object_cloud)
				centroid_computation.add(point);

			centroid_computation.get(seg->centroid);
			seg->points = object_cloud;
			seg->compute_mean_color();

			segments.push_back(seg);
		}

		(*emitter)(segments, enact_priority::operation::CREATE);

		//std::cout << clusters->size()<<" clusters detected within " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;
	});

}


} //namespace state_observation
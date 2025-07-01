#include "point_cloud_processing.h"

#include <boost/asio/post.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include "service_impl.h"

incremental_point_cloud::incremental_point_cloud(size_t threads)
	: thread_pool(threads)
{}

incremental_point_cloud::~incremental_point_cloud()
{
	thread_pool.stop();
}

void incremental_point_cloud::insert(
	server::holo_pointcloud::Ptr&& pcl_data)
{
	post(thread_pool, [this, pcl_data = pcl_data]()
		{
			using namespace server;

			auto& new_pcl = pcl_data->pcl;

			pcl::PassThrough<pcl::PointXYZ> pass;
			pass.setInputCloud(new_pcl);
			pass.setFilterFieldName("z");
			pass.setFilterLimits(-1e20, 1e20);
			pass.filter(*new_pcl);

			pcl::PointXYZ min, max;
			getMinMax3D(*new_pcl, min, max);

			pcl::CropBox<pcl::PointXYZ> box(true);
			box.setMin(convert<Eigen::Vector4f>(min));
			box.setMax(convert<Eigen::Vector4f>(max));
			box.setInputCloud(pcl);

			std::unique_lock lock(mtx);
			pcl::PointCloud<pcl::PointXYZ> extract;
			box.filter(extract);
			box.setNegative(true);
			box.filter(*pcl);

			new_pcl->reserve(new_pcl->size() + extract.size());
			for (const auto& p : extract.points)
				new_pcl->emplace_back(p);

			pcl::VoxelGrid<pcl::PointXYZ> sor;
			sor.setInputCloud(new_pcl);
			sor.setLeafSize(0.007f, 0.007f, 0.007f);
			sor.filter(*new_pcl);

			pcl->reserve(pcl->size() + new_pcl->size());
			for (const auto& p : new_pcl->points)
				pcl->emplace_back(p);

			changed();
		});
}

void incremental_point_cloud::save(const std::string& filename) const
{
	std::unique_lock lock(mtx);
	pcl::io::savePLYFileBinary(filename + ".ply", *pcl);
}

pcl::PointCloud<pcl::PointXYZ>::ConstPtr incremental_point_cloud::get_pcl() const
{
	return pcl;
}
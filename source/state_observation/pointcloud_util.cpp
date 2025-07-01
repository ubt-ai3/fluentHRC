#include "pointcloud_util.hpp"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ranges>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <eigen_serialization/eigen_serialization.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/common/common.h>

#include <state_observation/workspace_objects.hpp>


namespace state_observation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: object_parameters
//
//
/////////////////////////////////////////////////////////////

object_parameters::object_parameters()
{
	filename_ = std::string("object_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else 
	{
		min_object_height = 0.01f;		
		max_height = 1.f;
		construction_base_dimensions = Eigen::Vector3f(0.15f, 0.03f, 0.03f);
	}
}

object_parameters::~object_parameters()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const object_parameters& object_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(object_params);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: computed_workspace_parameters
//
//
/////////////////////////////////////////////////////////////

computed_workspace_parameters::computed_workspace_parameters(bool simulation)
	:
	crop_box_min(-std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones()),
	crop_box_max(std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones()),
	construction_area_min(-std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones()),
	construction_area_max(std::numeric_limits<float>::infinity()* Eigen::Vector3f::Ones()),
	transformation(Eigen::Affine3f::Identity()),
	simulation_transformation(Eigen::Affine3f::Identity()),
	simulation(simulation),
	max_object_dimension(0.07f)
{
	filename_ = std::string("computed_workspace_parameters.xml");

	std::ifstream file( folder_ + filename_ );
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
}

computed_workspace_parameters::~computed_workspace_parameters()
{
	std::ofstream file{ folder_ + filename_ };
	boost::archive::xml_oarchive oa{ file };
	const computed_workspace_parameters& computed_workspace_params = *this;
	oa << BOOST_SERIALIZATION_NVP(computed_workspace_params);
}




Eigen::Affine3f computed_workspace_parameters::get_cloud_transformation() const
{
	if (simulation)
		return simulation_transformation;
	return transformation;
}

Eigen::Affine3f computed_workspace_parameters::get_inv_cloud_transformation() const 
{
	if (simulation)
		return simulation_transformation.inverse();
	return transformation.inverse();
}

Eigen::Hyperplane<float, 3> computed_workspace_parameters::get_plane() const
{
	const auto& trafo = get_cloud_transformation();

	Eigen::Vector3f normal = trafo.rotation().inverse().col(2);

	return { normal, -normal.dot(trafo.translation()) };
}

/////////////////////////////////////////////////////////////
//
//
//  Class: pointcloud_preprocessing
//
//
/////////////////////////////////////////////////////////////

pointcloud_preprocessing::pointcloud_preprocessing(const std::shared_ptr<const object_parameters>& object_params, bool simulation)
	:	
	workspace_params(simulation),
	object_params(object_params)
{
	if (!object_params)
		throw std::invalid_argument("object_params is null");
}

pcl::PointCloud<pcl::PointNormal>::ConstPtr pointcloud_preprocessing::normals(const pcl::PointCloud<PointT>::ConstPtr& input) const 
{
	auto normals = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();

		pcl::NormalEstimation<PointT, pcl::PointNormal> ne;
	ne.setRadiusSearch(0.01);
		ne.setInputCloud(input);
		ne.compute(*normals);
	
	auto normals_p = normals->begin();
	for (auto input_p = input->begin();
			normals_p != normals->end() && input_p != input->end();
			++input_p, ++normals_p) 
		{
			normals_p->x = input_p->x;
			normals_p->y = input_p->y;
			normals_p->z = input_p->z;
		}
	return normals;
}

void pointcloud_preprocessing::add_point_size_sample(size_t point_size)
{
	if (point_size == 0)
		return;

	const float a = 1.f / (1.f + samples);
	average_points = static_cast<size_t>((1.f - a) * average_points + point_size * a);

	if (samples < MAX_SAMPLES)
		samples += 1;
}

pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr pointcloud_preprocessing::remove_table(const pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr& input) 
{
	auto start = std::chrono::high_resolution_clock::now();

	// get by value or there is the risk that the temp object may go out of scope
	// pcl::detail::Transformer only captures matrix by ref!
	const auto matrix = workspace_params.get_cloud_transformation().matrix();
	const pcl::detail::Transformer<float> tf(matrix);
	auto cloud_filtered = std::make_shared<pcl::PointCloud<PointT>>();

	cloud_filtered->header = input->header;
	cloud_filtered->is_dense = input->is_dense;
	cloud_filtered->height = 1;
	// trade-off between avoiding copies and an over-sized vector 
	cloud_filtered->points.reserve(average_points * 1.3f);
	cloud_filtered->sensor_orientation_ = input->sensor_orientation_;
	cloud_filtered->sensor_origin_ = input->sensor_origin_;

	auto& points_out = cloud_filtered->points;
	float threshold = object_params->min_object_height;
	auto min = workspace_params.crop_box_min;
	auto max = workspace_params.crop_box_max;

	auto timestart = std::chrono::high_resolution_clock::now();
	{
		//speedup 2x
		auto result = input->points
			| std::views::filter(AA)
			| std::views::transform([&tf](const PointT& pIn)
				{
					PointT p;
					tf.se3(pIn.data, p.data);
					p.rgba = pIn.rgba;
					return p;
				})
			| std::views::filter([&threshold](const PointT& p)
				{
					return std::abs(p.z) >= threshold;
				})
			| std::views::filter([&min, &max](const PointT& p)
				{
					return
						p.x >= min.x() && p.x <= max.x() &&
						p.y >= min.y() && p.y <= max.y() &&
						p.z >= min.z() && p.z <= max.z();
				});

		for (const auto res : result)
			points_out.emplace_back(res);
	}
	/*{
		for (const auto& pIn : input->points)
		{
			if (!std::isfinite(pIn.x) ||
				!std::isfinite(pIn.y) ||
				!std::isfinite(pIn.z) ||
				pIn.z == 0.f)
				continue;

			PointT p;
			tf.se3(pIn.data, p.data);

			if (std::abs(p.z) < threshold)
				continue;

			if (p.x < min.x() || p.y < min.y() || p.z < min.z() ||
				p.x > max.x() || p.y > max.y() || p.z > max.z())
				continue;

			p.rgba = pIn.rgba;
			points_out.emplace_back(p);
		}
	}*/
	//auto duration = std::chrono::high_resolution_clock::now() - timestart;
	//std::cout << (float)duration.count() / std::chrono::high_resolution_clock::period::den << std::endl;

	cloud_filtered->width = points_out.size();
	add_point_size_sample(points_out.size());


	//if (!points_out.empty())
	//	std::cout << "Points: " << points_out.size() << "; average: " << average_points << std::endl;


	//std::cout << "preprocessing took " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;


	return cloud_filtered;
}

Eigen::Affine3f pointcloud_preprocessing::get_cloud_transformation() const
{
	return workspace_params.get_cloud_transformation();
}


Eigen::Affine3f pointcloud_preprocessing::get_inv_cloud_transformation() const
{
	return workspace_params.get_inv_cloud_transformation();
}

pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr pointcloud_preprocessing::filter_nan(const pcl::PointCloud<PointT>::ConstPtr& input) const
{
	pcl::PassThrough<PointT> pass;
	pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);

	// Build a passthrough filter to remove spurious NaNs
	pass.setInputCloud(input);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(-1e20f, 1e20f);
	pass.filter(*cloud_filtered);

	return cloud_filtered;
}

pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr pointcloud_preprocessing::crop(const pcl::PointCloud<PointT>::ConstPtr& input) const
{
	pcl::CropBox<PointT> cropper;
	pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);

	cropper.setKeepOrganized(true);
	cropper.setMin(workspace_params.crop_box_min.homogeneous());
	cropper.setMax(workspace_params.crop_box_max.homogeneous());

	cropper.setInputCloud(input);
	cropper.filter(*cloud_filtered);

	return cloud_filtered;
}



pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr pointcloud_preprocessing::super_voxel_clustering(const pcl::PointCloud<PointT>::ConstPtr& input) const
{
	typedef pcl::PointXYZRGBA PointT;
	typedef pcl::PointCloud<PointT> PointCloudT;
	typedef pcl::PointNormal PointNT;
	typedef pcl::PointCloud<PointNT> PointNCloudT;
	typedef pcl::PointXYZL PointLT;
	typedef pcl::PointCloud<PointLT> PointLCloudT;

	pcl::SupervoxelClustering<PointT> clusterer(0.008f, 0.1f);

	float voxel_resolution = 0.008f;
	float seed_resolution = 0.1f;
	float color_importance = 1.0f;
	float spatial_importance = 0.4f;
	float normal_importance = 0.2f;


	pcl::SupervoxelClustering<PointT> super(voxel_resolution, seed_resolution);

	super.setInputCloud(input);
	super.setColorImportance(color_importance);
	super.setSpatialImportance(spatial_importance);
	super.setNormalImportance(normal_importance);

	std::map <uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;

	super.extract(supervoxel_clusters);
	std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
	super.getSupervoxelAdjacency(supervoxel_adjacency);




	return super.getVoxelCentroidCloud();
	//return std::make_pair(supervoxel_clusters, supervoxel_adjacency);
}

pcl::IndicesClustersPtr  pointcloud_preprocessing::conditional_euclidean_clustering(const pcl::PointCloud<PointT>::ConstPtr& input) const
{

	pcl::IndicesClustersPtr clusters(new pcl::IndicesClusters);
	pcl::search::KdTree<PointT>::Ptr search_tree(new pcl::search::KdTree<PointT>);

	//pcl::NormalEstimation<PointT, pcl::Normal> ne;
	//ne.setInputCloud(input);
	//ne.setSearchMethod(search_tree);
	//ne.setRadiusSearch(0.02f);
	//ne.compute(*normals);

	pcl::ConditionalEuclideanClustering<PointT> cec(false);
	cec.setInputCloud(input);
	cec.setConditionFunction([&](const PointT& p1, const PointT& p2, float squared_distance) 
		{
			return std::abs(p2.r-p1.r) < 20 && std::abs(p2.g - p1.g) < 20 && std::abs(p2.b - p1.b) < 20 
				&& std::abs(p1.z - p2.z) < object_params->min_object_height;
		});
	cec.setClusterTolerance(object_params->min_object_height);
	cec.setMinClusterSize(15);
	cec.setMaxClusterSize(std::max(300, static_cast<int>(input->points.size() / 3)));
	cec.segment(*clusters);

	return clusters;

}

std::vector<pc_segment::Ptr> pointcloud_preprocessing::extract_segments(const pcl::PointCloud<PointT>::ConstPtr& input) const
{
	const pcl::IndicesClustersPtr clusters = conditional_euclidean_clustering(input);
	std::vector<pc_segment::Ptr> segments;

	pcl::ExtractIndices<PointT> points_extract;
	points_extract.setInputCloud(input);
	points_extract.setNegative(false);

	for (const pcl::PointIndices& cluster : *clusters) 
	{
		if (cluster.indices.size() < min_points_per_object)
			continue;

		auto seg = std::make_shared<pc_segment>();
		seg->indices = std::make_shared<const pcl::PointIndices>(cluster);
		//seg->reference_frame = input;

		pcl::PointCloud<PointT>::Ptr object_cloud(new pcl::PointCloud<PointT>());
		points_extract.setIndices(seg->indices);
		points_extract.filter(*object_cloud);

		seg->bounding_box = oriented_bounding_box_for_standing_objects(object_cloud);

		pcl::CentroidPoint<PointT> centroid_computation;
		for (const auto& iter : *object_cloud)
			centroid_computation.add(iter);

		centroid_computation.get(seg->centroid);
		seg->points = object_cloud;

		segments.push_back(seg);
	}
	return segments;
}

pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr pointcloud_preprocessing::project_on_table(const pcl::PointCloud<PointT>::ConstPtr& input) const
{
	pcl::PointCloud<pointcloud_preprocessing::PointT>::Ptr output(new pcl::PointCloud<pointcloud_preprocessing::PointT>);
	Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
	transformation(2, 2) = 0;

	pcl::transformPointCloud(*input, *output, transformation);

	return output;
}

std::vector<cv::Point2f> pointcloud_preprocessing::project_on_table_cv(const pcl::PointCloud<PointT>::ConstPtr& input) const
{
	std::vector<cv::Point2f> output;
	output.reserve(input->size());
	
	for (const PointT& p : *input)
		output.emplace_back(p.x, p.y);

	return output;
}

obb pointcloud_preprocessing::oriented_bounding_box_for_standing_objects(const pcl::PointCloud<PointT>::ConstPtr& input) const
{
	float max_z = 0.f;
	for (PointT p : *input)
		max_z = std::max(max_z, p.z);

	cv::RotatedRect rect(cv::minAreaRect(project_on_table_cv(input)));

	if (rect.angle < -45.f || rect.angle > 45.f)
	{
		std::swap(rect.size.width, rect.size.height);
		rect.angle -= std::copysignf(90.f, rect.angle);
	}

	return {
		Eigen::Vector3f(rect.size.width,rect.size.height, max_z),
		Eigen::Vector3f(rect.center.x, rect.center.y, 0.5f * max_z),
		Eigen::Quaternionf(Eigen::AngleAxisf(rect.angle * M_PI / 180.f, Eigen::Vector3f::UnitZ()))
	};
}

pcl::PolygonMesh::Ptr pointcloud_preprocessing::color(const pcl::PolygonMesh::ConstPtr& input, const pcl::RGB& color)
{
	pcl::PolygonMesh::Ptr output;

	if (std::strcmp(input->cloud.fields.back().name.c_str(), "rgb") != 0) 
	{
		// no rgb values present - expand data
		output = std::make_shared<pcl::PolygonMesh>();
		output->header = input->header;
		output->polygons = input->polygons;
		output->cloud.fields = input->cloud.fields;

		pcl::PCLPointField rgbField;
		rgbField.count = 1;
		rgbField.datatype = rgbField.FLOAT32;
		rgbField.name = "rgb";
		rgbField.offset = input->cloud.point_step;

		output->cloud.fields.push_back(rgbField);
		output->cloud.header = input->cloud.header;
		output->cloud.height = input->cloud.height;
		output->cloud.is_bigendian = input->cloud.is_bigendian;
		output->cloud.is_dense = input->cloud.is_dense;
		output->cloud.point_step = input->cloud.point_step + 4;
		output->cloud.width = input->cloud.width;

		int size = output->cloud.width * output->cloud.height;
		int out_pstep = output->cloud.point_step;
		int in_pstep = input->cloud.point_step;
		output->cloud.data.resize(size * out_pstep);
		for (int p_idx = 0; p_idx < size; ++p_idx) {
			for (int field_offset = 0; field_offset < input->cloud.point_step; ++field_offset) {
				output->cloud.data[p_idx * out_pstep + field_offset] = input->cloud.data[p_idx * in_pstep + field_offset];
			}
			output->cloud.data[p_idx * out_pstep + rgbField.offset] = color.b;
			output->cloud.data[p_idx * out_pstep + rgbField.offset + 1] = color.g;
			output->cloud.data[p_idx * out_pstep + rgbField.offset + 2] = color.r;
			output->cloud.data[p_idx * out_pstep + rgbField.offset + 3] = color.a;
		}
	}
	else 
	{
		output = std::make_shared<pcl::PolygonMesh>(*input);
		const pcl::PCLPointField& rgbField = input->cloud.fields.back();
		int size = output->cloud.width * output->cloud.height;
		int pstep = input->cloud.point_step;
		for (int point_offset = 0; point_offset < size; point_offset += pstep) {
			output->cloud.data[point_offset + rgbField.offset] = color.b;
			output->cloud.data[point_offset + rgbField.offset + 1] = color.g;
			output->cloud.data[point_offset + rgbField.offset + 2] = color.r;
			output->cloud.data[point_offset + rgbField.offset + 3] = color.a;
		}
	}
	
	return output;
}

pcl::PolygonMesh::Ptr pointcloud_preprocessing::transform(const pcl::PolygonMesh::ConstPtr& input, const obb& bounding_box)
{
	Eigen::Affine3f matrix =
		Eigen::Translation3f(bounding_box.translation) *
		bounding_box.rotation *
		Eigen::Scaling(0.5f * bounding_box.diagonal);

	return transform(input, matrix);
}

pcl::PolygonMesh::Ptr pointcloud_preprocessing::transform(const pcl::PolygonMesh::ConstPtr& input, const Eigen::Affine3f& matrix)
{
	pcl::PolygonMesh::Ptr output(new pcl::PolygonMesh(*input));

	int size = input->cloud.width * input->cloud.height;
	int pstep = input->cloud.point_step;
	for (int point_offset = 0; point_offset < size; ++point_offset)
	{
		auto point = reinterpret_cast<float*>(&output->cloud.data[point_offset * pstep]);
		memcpy(point, (matrix * Eigen::Vector3f(point)).data(), 3 * sizeof(float));
	}

	return output;
}

pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr pointcloud_preprocessing::fuse(const pcl::PointCloud<PointT>::ConstPtr& cloud1, const pcl::PointCloud<PointT>::ConstPtr& cloud2) const
{
	if (cloud1->size() != cloud2->size())
		throw std::exception("Clouds differ in size.");

	pcl::PointCloud<PointT>::Ptr output(new pcl::PointCloud<PointT>);

	output->header = cloud2->header;
	output->height = cloud2->height;
	output->is_dense = cloud2->is_dense;
	output->sensor_orientation_ = cloud2->sensor_orientation_;
	output->sensor_origin_ = cloud2->sensor_origin_;
	output->width = cloud2->width;

	output->points.resize(cloud2->size());
	auto out = output->begin();
	for (auto in1 = cloud1->begin(), in2 = cloud2->begin();
		in1 != cloud1->end() && in2 != cloud2->end() && out != output->end();
		++in1, ++in2, ++out)
	{

		if (std::isnan(in1->z))
			*out = *in2;
		else if (std::isnan(in2->z))
			*out = *in1;
		else
		{
			out->x = in2->x;
			out->y = in2->y;
			out->z = 0.5f * (in1->z + in2->z);
			out->r = (in1->r + in2->r) / 2;
			out->g = (in1->g + in2->g) / 2;
			out->b = (in1->b + in2->b) / 2;
		}
	}

	return output;
}

pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr pointcloud_preprocessing::to_pc_rgba(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& pc_rgb)
{
	auto pc_rgba = std::make_shared<pcl::PointCloud<PointT>>();
	pc_rgba->header = pc_rgb->header;

	pc_rgba->height = pc_rgb->height;
	pc_rgba->is_dense = pc_rgb->is_dense;
	pc_rgba->sensor_orientation_ = pc_rgb->sensor_orientation_;
	pc_rgba->sensor_origin_ = pc_rgb->sensor_origin_;
	pc_rgba->width = pc_rgb->width;

	pc_rgba->points.reserve(pc_rgba->height * pc_rgba->width);
	for (const pcl::PointXYZRGB& p : pc_rgb->points)
	{
		PointT pa;
		pa.x = p.x; pa.y = p.y; pa.z = p.z;
		pa.r = p.r; pa.g = p.g; pa.b = p.b; pa.a = 255;
		pc_rgba->points.emplace_back(pa);
	}
	return pc_rgba;
}

std::pair<pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr, pcl::PointCloud<pcl::PointNormal>::Ptr>
pointcloud_preprocessing::smooth(const pcl::PointCloud<PointT>::ConstPtr& input) const
{
	pcl::PointCloud<PointT>::Ptr output(new pcl::PointCloud<PointT>);

	// Create a KD-Tree
	const auto tree = std::make_shared<pcl::search::KdTree<PointT>>();

	// Output has the PointNormal type in order to store the normals calculated by MLS
	auto mls_points = std::make_shared<pcl::PointCloud<pcl::PointNormal>>();

	// Init object (second point type is for the normals, even if unused)
	pcl::MovingLeastSquares<PointT, pcl::PointNormal> mls;

	mls.setComputeNormals(true);

	// Set parameters
	mls.setInputCloud(input);
	mls.setPolynomialOrder(2);
	mls.setSearchMethod(tree);
	mls.setSearchRadius(0.01);

	// Reconstruct
	mls.process(*mls_points);

	const pcl::PointIndicesPtr correspondences(mls.getCorrespondingIndices());
	output->resize(mls_points->size());
	for (int i = 0; i < mls_points->size(); ++i)
	{
		const PointT& input_p = input->operator[](correspondences->indices[i]);
		const pcl::PointNormal& mls_p = mls_points->operator[](i);
		PointT& output_p = output->operator[](i);

		output_p = input_p;
		output_p.x = mls_p.x;
		output_p.y = mls_p.y;
		output_p.z = mls_p.z;
	}

	return std::make_pair(output, mls_points);
}

pcl::PointIndices::Ptr pointcloud_preprocessing::remove_depth_ramps(const pcl::PointCloud<PointT> cloud,
	const pcl::PointIndices& object_indices,
	bool indices_to_indices) const
{
	std::map<unsigned int, std::vector<signed int>> rows;
	std::map<unsigned int, std::vector<signed int>> columns;

	const int w = cloud.width;

	for (int index : object_indices.indices) 
	{
		unsigned int row =  index / w;
		unsigned int column = index % w;

		auto row_iter = rows.find(row);
		if (row_iter == rows.end())
			rows.emplace(row, std::vector<signed int>({ index }));
		else
			row_iter->second.push_back(index);

		auto column_iter = columns.find(column);
		if (column_iter == columns.end())
			columns.emplace(column, std::vector<signed int>({ index }));
		else
			column_iter->second.push_back(index);
	}

	std::map<signed int, signed int> result;
	for (signed int i = 0; i < object_indices.indices.size(); i++)
	{
		result.emplace(object_indices.indices[i], i);
	}
	
	float delta_z = 2 * object_params->min_object_height;
	float delta_xy = 3 * object_params->min_object_height;

	for (auto& indices : rows | std::views::values) 
	{
		std::ranges::sort(indices);

		/*
		* Procedure: Determines the maximum z value within delta_xy from the border of the object
		* Erases all points below max_z - delta_z within the margin
		*/

		// process left margin
		{
			auto iter = indices.begin();
			float max_z = cloud.at(*iter).z;
			float prev_z = cloud.at(*iter).z;
			float start_x = cloud.at(*iter).x;
			float prev_x = cloud.at(*iter).x;

			while (iter != indices.end() &&
				(std::abs(cloud.at(*iter).x - start_x) < delta_xy ||
					cloud.at(*iter).x - prev_x < cloud.at(*iter).z - prev_z )) 
			{
				max_z = std::max(max_z, cloud.at(*iter).z);
				prev_z = cloud.at(*iter).z;
				prev_x = cloud.at(*iter).x;
				++iter;
			}

			iter = indices.begin();
			while (cloud.at(*iter).z + delta_z < max_z) 
			{
				result.erase(*iter);
				++iter;
			}
		}


		// process right margin
		{
			auto iter = indices.rbegin();
			float max_z = cloud.at(*iter).z;
			float prev_z = cloud.at(*iter).z;
			float end_x = cloud.at(*iter).x;
			float prev_x = cloud.at(*iter).x;

			while (iter != indices.rend() &&
				(std::abs(cloud.at(*iter).x - end_x) < delta_xy ||
				cloud.at(*iter).x - prev_x < cloud.at(*iter).z - prev_z)) 
			{
				max_z = std::max(max_z, cloud.at(*iter).z);
				prev_z = cloud.at(*iter).z;
				prev_x = cloud.at(*iter).x;
				++iter;
			}

			iter = indices.rbegin();
			while (cloud.at(*iter).z + delta_z < max_z) 
			{
				result.erase(*iter);
				++iter;
			}
		}
	}

	for (auto& indices : columns | std::views::values) 
	{
		std::ranges::sort(indices);
		// process bottom margin
		{
			auto iter = indices.begin();
			float max_z = cloud.at(*iter).z;
			float prev_z = cloud.at(*iter).z;
			float start_y = cloud.at(*iter).y;
			float prev_y = cloud.at(*iter).y;

			while (iter != indices.end() &&
				(std::abs(cloud.at(*iter).y - start_y) < delta_xy ||
				cloud.at(*iter).y - prev_y < cloud.at(*iter).z - prev_z)) 
			{
				max_z = std::max(max_z, cloud.at(*iter).z);
				prev_z = cloud.at(*iter).z;
				prev_y = cloud.at(*iter).y;
				++iter;
			}

			iter = indices.begin();
			while (cloud.at(*iter).z + delta_z < max_z) 
			{
				result.erase(*iter);
				++iter;
			}
		}


		// process top margin
		{
			auto iter = indices.rbegin();
			float max_z = cloud.at(*iter).z;
			float prev_z = cloud.at(*iter).z;
			float end_y = cloud.at(*iter).y;
			float prev_y = cloud.at(*iter).y;

			while (iter != indices.rend() &&
				(std::abs(cloud.at(*iter).y - end_y) < delta_xy ||
				cloud.at(*iter).y - prev_y < cloud.at(*iter).z - prev_z)) 
			{
				max_z = std::max(max_z, cloud.at(*iter).z);
				prev_z = cloud.at(*iter).z;
				prev_y = cloud.at(*iter).y;
				++iter;
			}

			iter = indices.rbegin();
			while (cloud.at(*iter).z + delta_z < max_z) 
			{
				result.erase(*iter);
				++iter;
			}
		}
	}

	auto result_indices = std::make_shared<pcl::PointIndices>();
	result_indices->header.frame_id = cloud.header.frame_id;
	result_indices->header.seq = cloud.header.seq;
	result_indices->header.stamp = cloud.header.stamp;

	result_indices->indices.reserve(result.size());

	for(const auto& entry : result)
		result_indices->indices.push_back(indices_to_indices ? entry.second : entry.first);

	return result_indices;
}

occlusion_detector::occlusion_detector(
	float min_object_height,
	const Eigen::Matrix4f& camera_projection)
:
	height_tolerance(min_object_height),
	camera_projection_(camera_projection),
	reference_cloud_2D_(pcl::make_shared< pcl::PointCloud<pcl::PointXY>>()),
	reference_cloud_()
{}
	
bool occlusion_detector::has_reference_cloud() const
{
	return !!reference_cloud_;
}

bool occlusion_detector::has_valid_reference_cloud() const
{
	return has_reference_cloud() && !reference_cloud_->empty();
}

Eigen::Matrix4f occlusion_detector::compute_view_projection(const Eigen::Vector3f& cam_pos, const Eigen::Vector3f& look_at)
{
	/*
		fovy and window size don't matter for occlusion detection
	*/
	pcl::visualization::Camera cam;
	cam.focal[0] = look_at.x(), cam.focal[1] = look_at.y(), cam.focal[2] = look_at.z();
	cam.fovy = 90. / 360. * M_PI; 
	cam.pos[0] = cam_pos.x();
	cam.pos[1] = cam_pos.y();
	cam.pos[2] = cam_pos.z();
	cam.clip[0] = 0.01;
	cam.clip[1] = 100; //clip has to be large enough
	//any view direction vec works which is perpendicular to cam_pos-look_at
	Eigen::Vector3f view_dir = cam_pos - look_at;
	Eigen::Vector3f e1(1, 0, 0);
	Eigen::Vector3f e2(0, 1, 0);
	Eigen::Vector3f up;
	if (std::abs(e1.dot(view_dir)) < std::abs(e2.dot(view_dir))) //prohibit view_dir parallel to used e
		up = e1.cross(view_dir);
	else
		up = e2.cross(view_dir);
	up.normalize();

	cam.view[0] = up.x();
	cam.view[1] = up.y();
	cam.view[2] = up.z();
	cam.window_size[0] = 1000;
	cam.window_size[1] = 1000;
	cam.window_pos[0] = 0;
	cam.window_pos[1] = 0;
	Eigen::Matrix4d projection;
	Eigen::Matrix4d view;
	cam.computeProjectionMatrix(projection);
	cam.computeViewMatrix(view);

	//Eigen::Affine3d world_to_camera(Eigen::Translation3d(-1. * cam_pos));

	
	//view = world_to_camera.matrix();
	return (projection* view).cast<float>();
}

//occlusion_detector::occlusion_detector(const std::shared_ptr<const object_parameters>& object_params, const Eigen::Matrix4f& camera_projection)
//	:
//	object_params(object_params),
//	view_projection(camera_projection)
//{
//
//}



void occlusion_detector::set_reference_cloud(const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
	if (!cloud || cloud == reference_cloud_)
		return;//nothing changes

	reference_cloud_ = cloud;
	auto temp = std::make_shared<pcl::PointCloud<PointT>>();
	pcl::copyPointCloud(*cloud, *temp);
	for (auto& p : *temp)
	{
		//transform points into camera image plane
		Eigen::Vector4f hom = camera_projection_ * Eigen::Vector4f(p.x, p.y, p.z, 1);
		Eigen::Vector3f div(hom.x() / hom.w(), hom.y() / hom.w(), std::abs(hom.z()));
		p = PointT(div.x(), div.y(), div.z(), 0, 0, 0, 0);
	}
	pcl::copyPointCloud(*temp, *reference_cloud_2D_);

	std::lock_guard<std::mutex> lock(kd_tree_mutex_);
	kd_tree_.setInputCloud(reference_cloud_2D_);

	//compute average closest point distance
	const int k = 2;
	pcl::Indices indices(k);
	std::vector<float> sqr_distances(k);
	float max_distance = -1;
	for (const auto& point : reference_cloud_2D_->points)
	{
		kd_tree_.nearestKSearch(point, k, indices, sqr_distances);
		for (auto dist : sqr_distances)
			if (max_distance < dist)
				max_distance = dist;
	}
	max_squared_distance = max_distance;
}

occlusion_detector::result occlusion_detector::perform_detection(const obb& obox) const
{
	auto cloud = occlusion_detector::convert_to_point_cloud_surface(obox, camera_projection_);
	auto casted_cloud = std::make_shared < pcl::PointCloud<occlusion_detector::PointT>>();
	pcl::copyPointCloud(*cloud, *casted_cloud);
	return perform_detection(*casted_cloud);
}


//float occlusion_detector::covered(const pc_segment& seg) const
//{
//	return perform_detection(*seg.points).occluded_pct;
//}
//
//float occlusion_detector::disappeared(const pc_segment& seg) const
//{
//	return perform_detection(*seg.points).disappeared_pct;
//}
//
//float occlusion_detector::present(const pc_segment& seg) const
//{
//	return perform_detection(*seg.points).present_pct;
//}
//
//float occlusion_detector::covered(const obb& bb) const
//{
//	auto cloud = convert_to_point_cloud_surface(bb, camera_projection_);
//	auto casted_cloud = std::make_shared < pcl::PointCloud<occlusion_detector::PointT>>();
//	pcl::copyPointCloud(*cloud, *casted_cloud);
//	return perform_detection(*casted_cloud).occluded_pct;
//}
//
//float occlusion_detector::disappeared(const obb& bb) const
//{
//	auto cloud = convert_to_point_cloud_surface(bb, camera_projection_);
//	auto casted_cloud = std::make_shared < pcl::PointCloud<occlusion_detector::PointT>>();
//	pcl::copyPointCloud(*cloud, *casted_cloud);
//	return perform_detection(*casted_cloud).disappeared_pct;
//}
//
//float occlusion_detector::present(const obb& bb) const
//{
//	auto cloud = convert_to_point_cloud_surface(bb, camera_projection_);
//	auto casted_cloud = std::make_shared < pcl::PointCloud<occlusion_detector::PointT>>();
//	pcl::copyPointCloud(*cloud, *casted_cloud);
//	return perform_detection(*casted_cloud).disappeared_pct;
//}

occlusion_detector::result occlusion_detector::perform_detection(const pcl::PointCloud<occlusion_detector::PointT>& cloud) const
{
	CV_DbgAssert(cloud.size());
	CV_DbgAssert(reference_cloud_ && reference_cloud_->size());
	CV_DbgAssert(height_tolerance >= 0);
	//using cloud_ptr = decltype(present_points);
	using PointT = state_observation::occlusion_detector::PointT;

	//Plan:
	//Project reference_cloud in  a plane with camera projection
	//project seg into plane
	//perform nearest neighbor search
	//if z-values agree -> visible
	//if seg z-values greater -> disappeared
	//-f seg z-values smaller -> occluded
	//pcl::kdTreeFLANN<pcl::PointT>
	//float resolution = 0.02;//distance of two points to be still in the same location
	//0,0,2 -> 0,0,2

	//std::vector<Eigen::Vector3f> test = { {0,0,-1},{0,0,-2},{1,0,-1},{0,-1,-2},{1,1,-2} };
	//test.push_back(Eigen::Vector3f(seg.front().x, seg.front().y, seg.front().z));
	//std::vector<Eigen::Vector3f> res;
	//for (auto vec : test)
	//{
	//	Eigen::Vector4f hom = camera_projection_* Eigen::Vector4f(vec.x(), vec.y(), vec.z(), 1);
	//	Eigen::Vector3f div(hom.x() / hom.w(), hom.y() / hom.w(), std::abs(hom.z()));
	//	res.push_back(div);
	//}

	//pcl::PointCloud<occlusion_detector::PointT> transformed_seg;
	//pcl::transformPointCloud(seg, transformed_seg, camera_projection_);
	
	std::vector<int> nearest_indices(1);
	std::vector<float> squared_distances(1);
	std::vector<float> closest_point_dist;

	float max_squared_dist_for_proximity = max_squared_distance;//std::pow(0.001, 2.0);
	int no_near_point_count = 0;

	int disappeared_count = 0;
	int occluded_count = 0;
	int present_count = 0;

	std::size_t seg_index = -1;
	std::lock_guard<std::mutex> lock(kd_tree_mutex_);
	for (const PointT& p : cloud)
	{
		seg_index++;
		//perspective division
		Eigen::Vector4f hom = camera_projection_ * Eigen::Vector4f(p.x, p.y, p.z, 1);
		Eigen::Vector3f div(hom.x() / hom.w(), hom.y() / hom.w(), std::abs(hom.z()));
		pcl::PointXY p2d{ div.x(),div.y() };
		//float compare_z = div.z();
		//Todo manual transform of points and perspective division

		bool found = kd_tree_.nearestKSearch(p2d, 1, nearest_indices, squared_distances);
		closest_point_dist.push_back(std::sqrt(squared_distances.front()));

		//evaluate search results
		auto reference_p = (*reference_cloud_)[nearest_indices[0]];
		if (occluded_by_actor(p2d))
		{
			occluded_count++;
			continue;
		}
			

		if (!found || isnan(reference_p.z) || squared_distances[0] > max_squared_dist_for_proximity)
		{
			//assumes only table is here
			no_near_point_count++;
			//assume only table at 0 is here -> disappeared
			//Attention could lead to errors if seg contains points belonging to the table 
			disappeared_count++;
			continue;
		}

		CV_DbgAssert(p.z > 0);//points need to be on top of the table
		//compares z values in world coordinates higher z values are above lower z values
		if (p.z < reference_p.z - height_tolerance)
		{
			//occluded_points->push_back(p);
			occluded_count++;
		}
		else if (p.z > reference_p.z + height_tolerance) 
		{
			//disappeared_points->push_back(p);
			disappeared_count++;
		}
		else //object still here
		{
			//if the hand is above and next to the box the camera interpolates points between hand
			// and object or table below. These points can by chance fall into the bounding box of a place
			kd_tree_.radiusSearch(p2d, max_shadow_extent, nearest_indices, squared_distances);
			bool covered = false;
			for (auto neighbor : nearest_indices)
			{
				float z = (*reference_cloud_)[neighbor].z;

				if (z > min_hand_height && p.z < z - height_tolerance)
				{
					occluded_count++;
					covered = true;
					break;
				}
			}

			if (covered)
				continue;

			present_count++;
		}
	}
	float point_count = cloud.size();
	float disappeared_pct = disappeared_count / point_count;
	float occluded_pct = occluded_count / point_count;
	float present_pct = present_count / point_count;

	float points_matched_pct = (point_count - no_near_point_count) / cloud.size();
	//debug_data

	if (occluded_pct > covered_threshold)
		return result::COVERED;
	if (present_pct > present_threshold)
		return result::PRESENT;
	else
		return result::DISAPPEARED;
}
//code from 
//https://pharr.org/matt/blog/2019/02/27/triangle-sampling-1.html

std::array<float, 3> LowDiscrepancySampleTriangle(float u)
{
	uint32_t uf = u * (1ull << 32);

	Eigen::Vector2f A(1, 0), B(0, 1), C(0, 0);        // Barycentrics
	for (int i = 0; i < 16; ++i) {            // For each base-4 digit
		int d = (uf >> (2 * (15 - i))) & 0x3; // Get the digit
		Eigen::Vector2f An, Bn, Cn;
		switch (d) {
		case 0:
			An = (B + C) / 2;
			Bn = (A + C) / 2;
			Cn = (A + B) / 2;
			break;
		case 1:
			An = A;
			Bn = (A + B) / 2;
			Cn = (A + C) / 2;
			break;
		case 2:
			An = (B + A) / 2;
			Bn = B;
			Cn = (B + C) / 2;
			break;
		case 3:
			An = (C + A) / 2;
			Bn = (C + B) / 2;
			Cn = C;
			break;
		}
		A = An;
		B = Bn;
		C = Cn;
	}

	Eigen::Vector2f r = (A + B + C) / 3;
	return { r.x(), r.y(), 1 - r.x() - r.y() };
}

std::vector<std::array<Eigen::Vector3f,3>> unit_cube_to_triangles()
{
	using Eigen::Vector3f;
	std::vector<std::array<Vector3f,3>> triangles;
	//front
	Vector3f flu{ -1,1,-1 };
	Vector3f fru{ 1,1,-1 };
	Vector3f fro{ 1,1,1 };
	Vector3f flo{ -1,1,1 };
	//back
	Vector3f blu{ -1,1,-1 };
	Vector3f bru{ 1,1,-1 };
	Vector3f bro{ 1,1,1 };
	Vector3f blo{ -1,1,1 };

	auto add_triangulated_quad = [&](Vector3f p1, Vector3f p2, Vector3f p1opp, Vector3f p2opp)
	{
		triangles.push_back({ p1,p2,p1opp });
		triangles.push_back({ p1,p2opp,p1opp });
	};

	add_triangulated_quad(flu, fru, fro, flo);//front
	add_triangulated_quad(blu, bru, bro, blo);//back
	add_triangulated_quad(flu, flo, blo, blu);//left
	add_triangulated_quad(flu, fru, bru, blu);//below
	add_triangulated_quad(fru, fro, bro, bru);//right
	add_triangulated_quad(flo, fro, bro, blo);//above
	return triangles;
}

//from
//https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm

bool RayIntersectsTriangle(const Eigen::Vector3f& rayOrigin,
	const Eigen::Vector3f& rayVector,
	const std::array<Eigen::Vector3f,3>& inTriangle,
	Eigen::Vector3f& outIntersectionPoint)
{
	using Eigen::Vector3f;
	constexpr float EPSILON = 0.0000001f;
	const Vector3f& vertex0 = inTriangle[0];
	const Vector3f& vertex1 = inTriangle[1];
	const Vector3f& vertex2 = inTriangle[2];
	Vector3f edge1, edge2, h, s, q;
	float a, f, u, v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;
	h = rayVector.cross(edge2);
	a = edge1.dot(h);
	if (a > -EPSILON && a < EPSILON)
		return false;    // This ray is parallel to this triangle.
	f = 1.0f / a;
	s = rayOrigin - vertex0;
	u = f * s.dot(h);
	if (u < 0.0 || u > 1.0)
		return false;
	q = s.cross(edge1);
	v = f * rayVector.dot(q);
	if (v < 0.0 || u + v > 1.0)
		return false;
	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = f * edge2.dot(q);
	if (t > EPSILON) // ray intersection
	{
		outIntersectionPoint = rayOrigin + rayVector * t;
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
		return false;
}

occlusion_detector::Ptr occlusion_detector::construct_from(const pointcloud_preprocessing& prepro_for_camera, const object_parameters& params_for_tolerance)
{
	Eigen::Vector3f cam_pos = prepro_for_camera.get_cloud_transformation().translation();
	Eigen::Vector3f cam_pos_look_at = cam_pos + prepro_for_camera.get_cloud_transformation() * Eigen::Vector3f(0, 0, -1);
	return std::make_shared<occlusion_detector>(params_for_tolerance.min_object_height,occlusion_detector::compute_view_projection(cam_pos,cam_pos_look_at));
}

bool occlusion_detector::intersect(Eigen::Vector3f p1, Eigen::Vector3f p2, const std::array<Eigen::Vector3f, 3>& triangle)
{
	auto rayVector = p2 - p1;
	auto midpoint = (p1 + p2) / 2;
	auto half_length_sqr = (rayVector / 2).squaredNorm();
	Eigen::Vector3f intersectionPoint;
	return (RayIntersectsTriangle(p1, rayVector, triangle, intersectionPoint)
		&& (midpoint - intersectionPoint).squaredNorm() <= half_length_sqr - 0.0001f);//intersection is on triangle and on the line
			//if the line ends on the triangle it shouldn't count as intersection
}

bool occlusion_detector::intersect(Eigen::Vector3f p1, Eigen::Vector3f p2, const std::array<Eigen::Vector3f, 4>& quad)
{
	std::array<Eigen::Vector3f, 3> triangle{ quad[0],quad[1],quad[2] };
	if (intersect(p1, p2, triangle))
		return true;
	triangle[1] = quad[3];
	if (intersect(p1, p2, triangle))
		return true;
	return false;
}


bool occlusion_detector::occluded_by_actor(const pcl::PointXY& p) const
{
	return false;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr occlusion_detector::convert_to_point_cloud_surface(
	const obb& obox,const Eigen::Matrix4f& camera_projection)
{

	//Plan: generate quads of obox
	// sample points on the quads
	// project them onto camera plane
	// find and remove hidden points (points behind other triangles)
		//points are hidden if the ray between camera position and point intersects a quad
	//TODO rotation of obox
	using Eigen::Vector3f;
	Vector3f scale = obox.diagonal;
	Eigen::Affine3f obox_trafo = Eigen::Affine3f(Eigen::Translation3f(obox.translation)) * Eigen::Affine3f(obox.rotation) * Eigen::Affine3f(Eigen::Scaling(obox.diagonal * 0.5));
	//unit cube corners
	//front
	Vector3f flu{ -1, 1,-1 };
	Vector3f fru{ 1 , 1,-1 };
	Vector3f fro{ 1 , 1, 1 };
	Vector3f flo{ -1, 1, 1 };
	//back		
	Vector3f blu{ -1,-1,-1 };
	Vector3f bru{  1,-1,-1 };
	Vector3f bro{  1,-1, 1 };
	Vector3f blo{ -1,-1, 1 };

	std::vector<std::array<Vector3f,4>> quads;
	auto add_quad = [&quads,&obox_trafo](Vector3f p1, Vector3f p2, Vector3f p1opp, Vector3f p2opp)mutable
	{
		quads.push_back({obox_trafo* p1,obox_trafo * p2,obox_trafo * p1opp,obox_trafo * p2opp });
	};

	//we only need the top quad
	//quads are listed either counter or clockwise, but not diagonally wise
	//add_quad(flu, fru, fro, flo);//front
	//add_quad(blu, bru, bro, blo);//back
	//add_quad(flu, flo, blo, blu);//left
	//add_quad(flu, fru, bru, blu);//below
	//add_quad(fru, fro, bro, bru);//right
	add_quad(flo, fro, bro, blo);//above
	
	//sample quads
	const int num_samples_per_axis = 10;
	std::vector<Vector3f> samples;
	for (const auto& quad : quads)
	{
		//TODO add explicit edge sampling
		Vector3f step_x = (quad[1] - quad[0]) / num_samples_per_axis;
		Vector3f step_y = (quad[2] - quad[1]) / num_samples_per_axis;
		Vector3f start = quad[0];
		for (int x = 0; x < num_samples_per_axis; x++)
		{
			auto cur_sample = start;
			for (int y = 0; y < num_samples_per_axis; y++)
			{
				samples.push_back(cur_sample);
				cur_sample += step_y;
			}
			start += step_x;
		}
	}

	auto out = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
	Vector3f camera_pos = { camera_projection(0,3),camera_projection(1,3),camera_projection(2,3) };

	auto view_result = samples
		| std::ranges::views::filter([&](const Vector3f& p)
			{
				return !std::ranges::any_of(quads, [&](const std::array<Vector3f, 4>& quad)
					{
						return intersect(camera_pos, p, quad);
					});
			})
		| std::ranges::views::transform([](const Vector3f& p) { return pcl::PointXYZ{ p.x(),p.y(),p.z() }; });
	out->points = { view_result.begin(), view_result.end() };
	out->width = out->size();
	out->height = 1;
	//setting width & height according to existing documentation
	//TODO::[Check] do we want height to be 1, as mentioned in the documentation?

	/*
	for (const auto& p : samples)
	{
		bool occluded = false;
		for (const auto& quad : quads)
		{
			std::array<Vector3f, 3> triangle0{ quad[0],quad[1],quad[2] };
			std::array<Vector3f, 3> triangle1{ quad[0],quad[3],quad[2] };



			if (intersect(camera_pos, p, triangle))
			{
				occluded = true;
				break;
			}
			triangle[1] = quad[3];
			if (intersect(camera_pos, p, triangle))
			{
				occluded = true;
				break;
			}
		}
		if (!occluded)
			out->push_back(pcl::PointXYZ{ p.x(),p.y(),p.z() });
	}*/

	return out;
}

} //namespace state_observation
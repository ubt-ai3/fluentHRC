#include <WinSock2.h> // solve incompatibility issue of Windows.h and WinSock2.h (which are both included by libraries)

#include <KinectGrabber/kinect2_grabber.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <simulation/rendering.hpp>

#include "state_observation/object_prototype_loader.hpp"
#include "state_observation/pointcloud_util.hpp"
#include "module_manager.hpp"

using namespace state_observation;

using PointT = pcl::PointXYZRGBA;

/**
* computes parameters describing the plane of the table (normal vector + translation)
* @throws if input has not enough points
*/
Eigen::Vector4f compute_plane(const pcl::PointCloud<PointT>::ConstPtr& input, const object_parameters& obj_params, pcl::PointCloud<PointT>::Ptr& out_plane, pcl::PointCloud<PointT>::Ptr& out_non_plane)
{

	if (!input || input->size() < 3)
		throw std::exception("Pointcloud has insufficient points");



	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices);
	pcl::ModelCoefficients table_coefficients;

	// Estimate point normals
	ne.setSearchMethod(tree);
	ne.setInputCloud(input);
	ne.setKSearch(50);
	ne.compute(*cloud_normals);

	// Create the segmentation object for the planar model and set all the parameters
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
	seg.setNormalDistanceWeight(0.05);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations(100);
	seg.setDistanceThreshold(5.f * obj_params.min_object_height);
	seg.setInputCloud(input);
	seg.setInputNormals(cloud_normals);
	// Obtain the plane inliers and coefficients
	seg.segment(*inliers_plane, table_coefficients);

	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud(input);
	extract.setIndices(inliers_plane);
	extract.setKeepOrganized(false);


	if (out_plane)
	{
		extract.setNegative(false);
		extract.filter(*out_plane);
	}

	if (out_non_plane)
	{
		extract.setNegative(true);
		extract.filter(*out_non_plane);
	}

	return Eigen::Map<Eigen::Vector4f>(table_coefficients.values.data());
}

void compute_crop_box_and_construction_area(const pcl::PointCloud<PointT>& table, computed_workspace_parameters& workspace_params, float max_height)
{
	PointT min, max;
	pcl::getMinMax3D(table, min, max);


	workspace_params.crop_box_min = Eigen::Vector3f(min.x, min.y, min.z - max_height);
	workspace_params.crop_box_max = Eigen::Vector3f(max.x, max.y, max.z + max_height);

	Eigen::Vector3f transformed_crop_box_min = workspace_params.crop_box_min;
	Eigen::Vector3f transformed_crop_box_max = workspace_params.crop_box_max;

	float width = std::abs(transformed_crop_box_max.x() - transformed_crop_box_min.x());
	float breadth = std::abs(transformed_crop_box_max.y() - transformed_crop_box_min.y());
	workspace_params.construction_area_min = Eigen::Vector3f(std::min(transformed_crop_box_max.x(), transformed_crop_box_min.x()) + 0.05f * width,
		std::min(transformed_crop_box_max.y(), transformed_crop_box_min.y()) + 0.1f * breadth,
		-0.01f
	);

	std::cout << width << " " << std::max(transformed_crop_box_max.x(), transformed_crop_box_min.x()) << 0.45f * width;
	workspace_params.construction_area_max = Eigen::Vector3f(std::max(transformed_crop_box_max.x(), transformed_crop_box_min.x()) - 0.45f * width,
		std::max(transformed_crop_box_max.y(), transformed_crop_box_min.y()) - 0.1f * breadth,
		0.5f
	);
}

pcl::PointCloud<PointT>::ConstPtr render_registration_scene(int argc, char* argv[], const object_parameters& obj_params, const Eigen::Vector3f& origin, const Eigen::Vector3f& other)
{
	auto env = std::make_shared<simulation::environment>(obj_params);
	env->additional_scene_objects.push_back(std::make_shared<simulation::simulated_table>());
	state_observation::object_prototype_loader loader;
	auto cube = loader.get("wooden cube");
	auto block = loader.get("red block");

	env->add_object(block, state_observation::aabb(block->get_bounding_box().diagonal, Eigen::Vector3f(origin.x(), origin.y(), 0.5 * block->get_bounding_box().diagonal.z())));
	env->add_object(cube, state_observation::aabb(cube->get_bounding_box().diagonal, Eigen::Vector3f(other.x(), other.y(), 0.5 * cube->get_bounding_box().diagonal.z())));

	simulation::pc_renderer renderer(argc, argv, env, 640, 720, false);
	return pointcloud_preprocessing::to_pc_rgba(renderer.render(std::chrono::duration<float>(0)));
}

// Returns face aligend bounding boxes order from biggest to smallest of the color col
std::vector<obb> get_boxes(pointcloud_preprocessing& pc_prepro, const pcl::PointCloud<PointT>::ConstPtr& cloud, cv::Vec3f ref_color/*color in rgb in range 0-1*/,double color_sensitivity = 15)
{
	auto clusters = pc_prepro.conditional_euclidean_clustering(cloud);

	pcl::ExtractIndices<PointT> points_extract;
	points_extract.setInputCloud(cloud);
	points_extract.setNegative(false);

	std::vector<obb> boxes;
	float min = 200;

	//convert ref color from rgb to bgr to cielab
	ref_color = cv::Vec3f(ref_color.val[2]/*b*/, ref_color.val[1]/*g*/, ref_color.val[0]/*r*/);
	cv::Mat m_refcol = cv::Mat(1, 1, CV_32FC3, &ref_color);
	cv::Mat m_ref_cielab;
	cv::cvtColor(m_refcol, m_ref_cielab, cv::COLOR_BGR2Lab);
	const cv::Vec3f& lab2 = m_ref_cielab.at<cv::Vec3f>(0, 0);

	for (const auto& cluster : *clusters)
	{
		pcl::PointCloud<PointT>::Ptr object_cloud(new pcl::PointCloud<PointT>());
		if (clusters->size())
			points_extract.setIndices(pcl::make_shared<pcl::PointIndices>(cluster));
		else
			std::cout << "no clusters found\n";
		points_extract.filter(*object_cloud);

		double r = 0;
		double g = 0;
		double b = 0;
		double size = object_cloud->size();
		for (const auto& p : *object_cloud)
		{
			r += p.r;
			g += p.g;
			b += p.b;
		}
		r /= size;
		g /= size;
		b /= size;
		//compare mean cluster color to re colr using cielab color space
		//covnert from range 0-255 to 0-1 to cielab

		try {
			cv::Vec3f bgr(b, g, r);
			cv::Mat m_bgr = cv::Mat(1,1,CV_32FC3,&bgr);
			m_bgr /= 255.;
			cv::Mat m_cielab;
			
			volatile int test = m_refcol.type();
			volatile int test2 = m_refcol.depth();

			cv::cvtColor(m_bgr,    m_cielab,	 cv::COLOR_BGR2Lab);

			//from http://zschuessler.github.io/DeltaE/learn/ deltaE 76

			const cv::Vec3f& lab1 = m_cielab.at<cv::Vec3f>(0, 0);

			std::cout << lab1 << "\n";
			float delta_e76 = std::sqrt(
				std::pow(lab2.val[0] - lab1.val[0], 2) +
				std::pow(lab2.val[1] - lab1.val[1], 2) +
				std::pow(lab2.val[2] - lab1.val[2], 2));
			min = std::min(min,delta_e76);
			if (delta_e76 < color_sensitivity)
				boxes.push_back(pc_prepro.oriented_bounding_box_for_standing_objects(object_cloud));
		}
		catch (const std::exception& e)
		{
			std::cout << e.what()<<"res"<<min;
		}
	}
	std::cout << "ref cielab " << lab2;
	std::sort(boxes.begin(), boxes.end(), [](const obb& lhs, const obb& rhs) {return lhs.diagonal.z() > rhs.diagonal.z(); });

	return boxes;
}

void show_result(const computed_workspace_parameters& workspace_params, const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
	std::thread t([&]() {
		pcl::PointCloud<PointT>::Ptr transformed(new pcl::PointCloud<PointT>());
		pcl::transformPointCloud(*cloud, *transformed, workspace_params.transformation);
		transformed->sensor_orientation_ = Eigen::Quaternionf::Identity();
		transformed->sensor_origin_ = Eigen::Vector4f::Zero();
		
		auto pcl_viewer = pcl::make_shared<pcl::visualization::PCLVisualizer>();
		pcl_viewer->addCoordinateSystem();
		pcl_viewer->addPointCloud(transformed);

		while (!pcl_viewer->wasStopped())
			pcl_viewer->spinOnce();
		});

	if (t.joinable())
		t.join();
}


int main(int argc, char* argv[])
{
	camera_type camera = camera_type::KINECT_V2;
	auto pc_prepro = std::make_unique<state_observation::pointcloud_preprocessing>(std::make_shared<state_observation::object_parameters>(), false);

	Eigen::Vector3f origin = Eigen::Vector3f::Zero();
	Eigen::Vector3f other(0.5f, 0.f, 0.f);
	pcl::PointCloud<PointT>::ConstPtr cloud;

	switch (camera)
	{
	case camera_type::SIMULATION:
	{
		std::thread t([&]()
			{
				cloud = render_registration_scene(argc, argv, *pc_prepro->object_params, origin, other);
			});
		if (t.joinable())
			t.join();
		break;
	}
	case camera_type::KINECT_V2:
	{
		auto grabber = std::make_shared<pcl::Kinect2Grabber>();
		std::mutex mutex;
		std::condition_variable signal;
		boost::signals2::connection connection_pc = grabber->registerCallback(
			std::function<void(const pcl::PointCloud<PointT>::ConstPtr&)>(
			[&cloud, &mutex,&signal](const pcl::PointCloud<PointT>::ConstPtr& input)
			{
				cloud = input;
				signal.notify_all();
			}));
		grabber->start();

		std::unique_lock<std::mutex> lock(mutex);
		signal.wait(lock, [&cloud]() {return !!cloud; });
		break;
	}
		
	}
	

	auto& params = pc_prepro->workspace_params;
	pcl::PointCloud<PointT>::Ptr cloud_roi(new pcl::PointCloud<PointT>);
	pcl::CropBox<PointT> region(true);
	region.setMin(Eigen::Vector4f(-0.45, -0.45, 1, 1));
	region.setMax(Eigen::Vector4f(0.45,0.45,3, 1));
	region.setInputCloud(cloud);
	region.filter(*cloud_roi);

	show_result(params, cloud_roi);
	
	pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());
	pcl::PointCloud<PointT>::Ptr cloud_non_plane(new pcl::PointCloud<PointT>());
	//Debug TODO remove
	Eigen::Vector4f plane = compute_plane(cloud_roi, *pc_prepro->object_params, cloud_plane, cloud_non_plane);

	show_result(params, cloud_non_plane);
	Eigen::Vector3f normal = -plane.head(3).normalized();

	Eigen::Vector3f table_translation = -plane(3) * normal;
	Eigen::Quaternionf table_rotation = Eigen::Quaternionf().setFromTwoVectors(normal, Eigen::Vector3f(0, 0, 1));
	params.transformation = table_rotation * Eigen::Translation3f(table_translation);

	pcl::transformPointCloud(*cloud_plane, *cloud_plane, params.transformation);
	pcl::transformPointCloud(*cloud_non_plane, *cloud_non_plane, params.transformation);
	//auto no_table  = pc_prepro->remove_table(cloud);


	auto boxes = get_boxes(*pc_prepro, /*no_table*/ cloud_non_plane, cv::Vec3f{ 240./255,100./255,100./255 }/*orange red*/);
	std::cout << "found " << boxes.size() << " blocks" << std::endl;
	if (boxes.size() == 2)
	{
		/*IMPORTANT modify Box Parameter here for registration 
		* for some reason the cube is bigger than the quader
		*/
		Eigen::Vector2f big_object_position = { 0.50f,-0.195f }/*cube*/;
		Eigen::Vector2f small_object_position = { 0.50f,0.245f }/*quader*/;

		std::vector < Eigen::Vector2f> centers_2d;
		for (const auto& box : boxes)
			centers_2d.push_back(Eigen::Vector2f(box.translation.x(), box.translation.y()));

		Eigen::Vector2f centroid_destination = 0.5f * (big_object_position + small_object_position);
		Eigen::Vector2f centroid_observed = 0.5f * (centers_2d[0] + centers_2d[1]);

		const auto to_vec3 = [](const Eigen::Vector2f& v) {
			return Eigen::Vector3f(v.x(), v.y(), 0);
		};

		Eigen::Affine3f origin_trafo = Eigen::Translation3f(centroid_destination.x(), centroid_destination.y(), 0.f) * 
			Eigen::Quaternionf::FromTwoVectors(to_vec3(centers_2d[0] - centers_2d[1]).normalized(), to_vec3(big_object_position - small_object_position).normalized()) *
			Eigen::Translation3f(-centroid_observed.x(), -centroid_observed.y(),0.f);

		pcl::transformPointCloud(*cloud_plane, *cloud_plane, origin_trafo);
		pcl::transformPointCloud(*cloud_non_plane, *cloud_non_plane, origin_trafo);

		params.transformation =
			
			origin_trafo *
			params.transformation;
	}

	compute_crop_box_and_construction_area(*cloud_plane, params, pc_prepro->object_params->max_height);
	std::cout << params.get_plane().coeffs() << std::endl;

	show_result(params, cloud);

}

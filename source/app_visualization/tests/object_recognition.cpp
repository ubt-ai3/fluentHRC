#include "object_recognition.hpp"

#include <pcl/geometry/planar_polygon.h>

#include "hand_pose_estimation/color_space_conversion.hpp"
#include "hand_pose_estimation/hand_pose_estimation.h"

#ifdef _DEBUG
#define _SCL_SECURE_NO_WARNINGS
#endif

namespace state_observation
{

void object_recognition_test::compute_and_show_clusters(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer, pointcloud_preprocessing& prepro, pcl::PointCloud<PointT>::ConstPtr input) const
{
	input = prepro.remove_table(input);
	//			auto normals = prepro.normals(temp_cloud);
	auto segments = prepro.extract_segments(input);
	//			temp_cloud = prepro.super_voxel_clustering(temp_cloud);

				//show_clusters(viewer, pair.first, pair.second);

	//			std::cout << "Clusters: " << clusters->size() << std::endl;

	pcl::PointCloud<PointT>::Ptr colored_clusters(new pcl::PointCloud<PointT>);
	for (const pc_segment::Ptr& seg : segments)
	{
		float r, g, b;
		r = g = b = 0.f;
		float weight = 1.f / seg->points->size();
		for (const PointT& p : *seg->points)
		{
			r += weight * p.r;
			g += weight * p.g;
			b += weight * p.b;
		}

		for (const PointT& p : *seg->points)
		{
			PointT p2(p);
			p2.r = static_cast<uint8_t>(r);
			p2.g = static_cast<uint8_t>(g);
			p2.b = static_cast<uint8_t>(b);

			colored_clusters->push_back(p2);
		}

	}


	if (viewer->contains("cloud"))
		viewer->updatePointCloud(colored_clusters, "cloud");
	else
		viewer->addPointCloud(colored_clusters, "cloud");

	//			viewer->addPointCloudNormals<PointT, pcl::Normal>(temp_cloud, normals, 25, 0.01f, "normals");
}

void object_recognition_test::compute_and_show_bounding_boxes(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer, pointcloud_preprocessing& prepro, pcl::PointCloud<PointT>::ConstPtr input) const
{
	input = prepro.remove_table(input);
	auto segments = prepro.extract_segments(input);

	viewer->removeAllShapes();

	pcl::PointCloud<PointT>::Ptr colored_clusters(new pcl::PointCloud<PointT>);

	int i = 0;
	for (const pc_segment::Ptr& seg : segments)
	{
		float r, g, b;
		r = g = b = 0.f;
		float weight = 1.f / seg->points->size();
		for (const PointT& p : *seg->points)
		{
			r += weight * p.r;
			g += weight * p.g;
			b += weight * p.b;
		}

		for (const PointT& p : *seg->points)
		{
			PointT p2(p);
			p2.r = static_cast<uint8_t>(r);
			p2.g = static_cast<uint8_t>(g);
			p2.b = static_cast<uint8_t>(b);

			colored_clusters->push_back(p2);
		}

		std::string id("cube " + std::to_string(i++));
		const obb& box = seg->bounding_box;
		viewer->addCube(box.translation, box.rotation, box.diagonal(0), box.diagonal(1), box.diagonal(2), id);
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, r / 255.f, g / 255.f, b / 255.f, id);
	}


	if (!viewer->updatePointCloud(colored_clusters))
		viewer->addPointCloud(colored_clusters);

	//			viewer->addPointCloudNormals<PointT, pcl::Normal>(temp_cloud, normals, 25, 0.01f, "normals");
}

void object_recognition_test::display(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer, pointcloud_preprocessing& prepro, const pc_segment& seg, const classification_result& classification, int index) const
{


	if (classification.prototype->has_mesh())
	{
		obb transformation_box(
			classification.prototype->get_bounding_box().diagonal,
			seg.bounding_box.translation,
			seg.bounding_box.rotation * classification.prototype_rotation);

		pcl::PolygonMesh::ConstPtr transformed_mesh(prepro.transform(classification.prototype->load_mesh(), transformation_box));

		std::string id("mesh " + std::to_string(index));
		viewer->addPolygonMesh(*transformed_mesh, id);


		id = "cube " + std::to_string(index);
		const obb& box = seg.bounding_box;
		viewer->addCube(box.translation, box.rotation, box.diagonal(0), box.diagonal(1), box.diagonal(2), id);
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, seg.centroid.r / 255.f, seg.centroid.g / 255.f, seg.centroid.b / 255.f, id);
	}
	else
	{

		std::string id("cube " + std::to_string(index));
		const obb& box = seg.bounding_box;
		viewer->addCube(box.translation, box.rotation, box.diagonal(0), box.diagonal(1), box.diagonal(2), id);
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
		viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.f, 1.f, 0.f, id);
	}
}

void object_recognition_test::compute_and_show_classified_objects(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer, 
	pointcloud_preprocessing& prepro, 
	const std::vector<classifier::classifier_aspect>& classifiers, 
	pcl::PointCloud<PointT>::ConstPtr input) const
{
	input = prepro.remove_table(input);
	auto segments = prepro.extract_segments(input);

	viewer->removeAllShapes();
	viewer->removeAllPointClouds();

	pcl::PointCloud<PointT>::Ptr colored_clusters(new pcl::PointCloud<PointT>);
	pcl::PointCloud<pcl::PointNormal>::Ptr normals(new pcl::PointCloud<pcl::PointNormal>);

	int i = 0;
	int count_background_segments = 0;
	for (const pc_segment::Ptr& seg : segments)
	{
		float max_similarity = 0.f;
		int best_classification_result = 0;
		state_observation::object_parameters params;
		state_observation::classifier clas(std::make_shared<object_parameters> (),classifiers);
		seg->classification_results = clas.classify_all(*seg);

		std::vector<classification_result>& results = seg->classification_results;
		std::sort(results.begin(), results.end(), [](const classification_result& lhs, const classification_result& rhs) {
			return lhs.local_certainty_score > rhs.local_certainty_score;
			});

		const classification_result& result = seg->classification_results.front();
		if (result.local_certainty_score > 0.01f
			&& result.prototype)
		{
			const object_prototype& prototype = *result.prototype;

			if (prototype.has_mesh())
			{
				display(viewer, prepro, *seg, result, i++);
			}
			else
			{
				count_background_segments++;

				std::string id("cube " + std::to_string(i++));
				const obb& box = seg->bounding_box;
				viewer->addCube(box.translation, box.rotation, box.diagonal(0), box.diagonal(1), box.diagonal(2), id);
				viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, id);
				viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.f, 1.f, 0.f, id);
			}

		}

		colored_clusters->insert(colored_clusters->end(), seg->points->begin(), seg->points->end());
//		normals->insert(normals->end(), seg->normals->begin(), seg->normals->end());



		pcl::PlanarPolygon<PointT> outline;
		outline.setCoefficients(Eigen::Vector4f(0.f, 0.f, 1.f, -seg->bounding_box.diagonal(2)));
		outline.setContour(*seg->get_outline());
		viewer->addPolygon(outline, seg->centroid.r, seg->centroid.g, seg->centroid.b, "polygon " + std::to_string(i++));
	}

//	std::cout << "Background segments: " << count_background_segments << std::endl;

	viewer->removePointCloud("normals");
	viewer->addPointCloudNormals<pcl::PointNormal>(normals, 100, 0.02f, "normals");

	if (!viewer->updatePointCloud(colored_clusters))
		viewer->addPointCloud(colored_clusters);
}

void object_recognition_test::test_pca(pointcloud_preprocessing& prepro) const
{
	std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.5, 0.0, 0.0, 0.0);
	viewer->addCoordinateSystem();

	pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());

	for (float x = 0.f; x < 0.05f; x += 0.005f)
	{
		for (float y = 0.f; y < 0.1f; y += 0.005f)
		{
			PointT p;
			p.x = x * std::sinf(0.75f) + y * std::cosf(0.75f);
			p.y = x * std::cosf(0.75f) - y * std::sinf(0.75f);
			p.z = 0.05f;
			p.r = 255;
			p.g = 255;

			cloud->push_back(p);
		}
	}

	obb box(prepro.oriented_bounding_box_for_standing_objects(cloud));

	viewer->addPointCloud(cloud);
	viewer->addCube(box.translation, box.rotation, box.diagonal(0), box.diagonal(1), box.diagonal(2));

	while (!viewer->wasStopped())
		viewer->spinOnce();
}

void object_recognition_test::show_mesh(pointcloud_preprocessing& prepro) const
{
	std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
		new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.5, 0.0, 0.0, 0.0);
	viewer->addCoordinateSystem();

	pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh);


	pcl::io::load("assets/object_meshes/cylinder.obj", *mesh); // ignore warnings

	pcl::RGB color(255, 0, 0);
	obb box(Eigen::Vector3f(0.25f, 0.25f, 0.5f), Eigen::Vector3f(1.f, 0.f, 0.f),
		Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)));
	viewer->addPolygonMesh(*prepro.transform(prepro.color(mesh, color), box));



	while (!viewer->wasStopped())
		viewer->spinOnce();
}

cv::Mat4b object_recognition_test::find_shapes(const std::shared_ptr<cv::Mat4b>& src) const
{
	cv::Mat src_gray;
	int thresh = 100;
	int max_thresh = 255;


	cv::Mat threshold_output;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::cvtColor(*src, src_gray, cv::COLOR_BGRA2GRAY);
	/// Detect edges using Threshold
	threshold(src_gray, threshold_output, thresh, 255, cv::THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	return draw_contoures(contours, threshold_output.size());
}



cv::Mat4b object_recognition_test::draw_contoures(const std::vector<std::vector<cv::Point>>& contours, const cv::Size& size) const
{
	cv::RNG rng(12345);

	/// Find the rotated rectangles and ellipses for each contour
	std::vector<cv::RotatedRect> minRect(contours.size());
	std::vector<cv::RotatedRect> minEllipse(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		minRect[i] = cv::minAreaRect(cv::Mat(contours[i]));
		if (contours[i].size() > 5)
		{
			minEllipse[i] = cv::fitEllipse(cv::Mat(contours[i]));
		}
	}

	/// Draw contours + rotated rects + ellipses
	cv::Mat drawing = cv::Mat::zeros(size, CV_8UC4);
	for (int i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		// contour
		drawContours(drawing, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
		// ellipse
		ellipse(drawing, minEllipse[i], color, 2, 8);
		// rotated rectangle
		cv::Point2f rect_points[4]; minRect[i].points(rect_points);
		for (int j = 0; j < 4; j++)
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
	}

	return drawing;
}

//
//void object_recognition_test::test_shape_classifier() const
//{
//	pcl::RGB wooden(200, 190, 180);
//	classifier::shape_classifier clas(object_prototype::Ptr(new object_prototype(
//		Eigen::Vector3f(0.028f, 0.028f, 0.058f),
//		wooden,
//		mesh_wrapper::Ptr(new mesh_wrapper("assets/object_meshes/bridge.obj")))));
//
//	Eigen::Vector3f zero(Eigen::Vector3f::Zero());
//	Eigen::Vector3f x1(1.f, 0.f, 0.f);
//	Eigen::Vector3f y1(0.f, 1.f, 0.f);
//	Eigen::Vector3f z1(0.f, 0.f, 1.f);
//
//	std::cout << "shall\tis" << std::endl;
//	std::cout << 0 << "\t" << clas.distance(
//		Eigen::Vector3f(0.25f, 0.25f, 0.f),
//		zero, x1, y1)
//		<< std::endl;
//
//	std::cout << 1.f << "\t" << clas.distance(
//		Eigen::Vector3f(0.5f, 0.5f, 1.f),
//		zero, x1, y1)
//		<< std::endl;
//
//	std::cout << 0.707107f << "\t" << clas.distance(
//		Eigen::Vector3f(1.f, 1.f, 0.f),
//		zero, x1, y1)
//		<< std::endl;
//
//	std::cout << 1.f << "\t" << clas.distance(
//		Eigen::Vector3f(2.f, 0.f, 0.f),
//		zero, x1, y1)
//		<< std::endl;
//
//	std::cout << 0.707107f << "\t" << clas.distance(
//		Eigen::Vector3f(0.f, 0.f, 0.f),
//		z1, x1, y1)
//		<< std::endl;
//
//	std::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
//		new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
//	viewer->addCoordinateSystem();
//
//	const obb box(clas.get_object_prototype()->get_bounding_box().diagonal,
//		Eigen::Vector3f(0.1f, 0.1f, 0.014f),
//		Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0.f, 0.f, 1.f), Eigen::Vector3f(1.f, 0.f, 0.f)));
//
//	Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
//	transform.block<3, 3>(0, 0) = box.rotation
//		* Eigen::Quaternionf::Identity()
//		* (0.5f * clas.get_object_prototype()->get_bounding_box().diagonal.asDiagonal());
//	transform.block<3, 1>(0, 3) = box.translation;
//	transform(2, 3) = std::abs(transform(2, 2));
//
//	pcl::PolygonMesh mesh = pcl::PolygonMesh(*clas.get_object_prototype()->load_mesh());
//	float similarity = 0.f;
//
//	std::function<void(int)> transform_point =
//		[&mesh, &transform](int index)
//	{
//		float* point = reinterpret_cast<float*>(&mesh.cloud.data.at(mesh.cloud.point_step * index));
//		Eigen::Vector3f vec(point);
//		memcpy(point, Eigen::Vector3f((transform * vec.homogeneous()).hnormalized()).data(), 3 * sizeof(float));
//	};
//
//	for (int i = 0; i < mesh.cloud.width; ++i)
//	{
//		transform_point(i);
//	}
//
//	viewer->addPolygonMesh(mesh);
//
//	while (!viewer->wasStopped()) {
//		// Update Viewer
//		try {
//			viewer->spinOnce();
//		}
//		catch (...) {
//			continue;
//		}
//	}
//
//	std::cout << "done" << std::endl;
//}

void object_recognition_test::show_preprocessed_cloud(std::shared_ptr<pcl::visualization::PCLVisualizer>& viewer, pointcloud_preprocessing& prepro, const pcl::PointCloud<PointT>::ConstPtr& input) const
{
	pcl::PointCloud<PointT>::ConstPtr processed_cloud = prepro.remove_table(input);

	if (!viewer->updatePointCloud(processed_cloud))
		viewer->addPointCloud(processed_cloud);
}

void object_recognition_test::test_rgb_hsv_conversion() const
{
	auto print = [](const hsv& is, const hsv& shall)
	{
		std::printf("(% .2f, % .2f, % .2f) ?= (% .2f, % .2f, % .2f)\n", is.h, is.s, is.v, shall.h, shall.s, shall.v);
	};

	rgb black_rgb(0., 0., 0.); hsv black_hsv(0., 0., 0.);
	rgb white_rgb(1., 1., 1.); hsv white_hsv(0., 0., 1.);
	rgb red_rgb(1., 0., 0.); hsv red_hsv(0., 1., 1.);
	rgb lime_rgb(0., 1., 0.); hsv lime_hsv(120., 1., 1.);
	rgb blue_rgb(0., 0., 1.); hsv blue_hsv(240., 1., 1.);
	rgb yellow_rgb(1., 1., 0.); hsv yellow_hsv(60., 1., 1.);
	rgb cyan_rgb(0., 1., 1.); hsv cyan_hsv(180., 1., 1.);
	rgb magenta_rgb(1., 0., 1.); hsv magenta_hsv(300., 1., 1.);
	rgb silver_rgb(0.5, 0.5, 0.5); hsv silver_hsv(0., 0., 0.5);

	print(hsv(black_rgb), black_hsv);
	print(hsv(white_rgb), white_hsv);
	print(hsv(red_rgb), red_hsv);
	print(hsv(lime_rgb), lime_hsv);
	print(hsv(blue_rgb), blue_hsv);
	print(hsv(yellow_rgb), yellow_hsv);
	print(hsv(cyan_rgb), cyan_hsv);
	print(hsv(magenta_rgb), magenta_hsv);
	print(hsv(silver_rgb), silver_hsv);

	std::cout << std::endl;

	auto print2 = [](const rgb& is, const rgb& shall)
	{
		std::printf("(% .2f, % .2f, % .2f) ?= (% .2f, % .2f, % .2f)\n", is.r, is.g, is.b, shall.r, shall.g, shall.b);
	};

	print2(rgb(black_hsv), black_rgb);
	print2(rgb(white_hsv), white_rgb);
	print2(rgb(red_hsv), red_rgb);
	print2(rgb(lime_hsv), lime_rgb);
	print2(rgb(blue_hsv), blue_rgb);
	print2(rgb(yellow_hsv), yellow_rgb);
	print2(rgb(cyan_hsv), cyan_rgb);
	print2(rgb(magenta_hsv), magenta_rgb);
	print2(rgb(silver_hsv), silver_rgb);
}

void object_recognition_test::test_cloud_transformation(pointcloud_preprocessing& prepro) const
{
	std::cout << "forward transformation:" << std::endl << prepro.get_cloud_transformation().matrix() << std::endl;
	std::cout << "backward transformation:" << std::endl << prepro.get_inv_cloud_transformation().matrix() << std::endl;
	std::cout << "backward transformation (numerical):" << std::endl << prepro.get_cloud_transformation().inverse().matrix() << std::endl;
	std::cout << "joint transformation (must be identity):" << std::endl << (prepro.get_inv_cloud_transformation() * prepro.get_cloud_transformation()).matrix() << std::endl;
}

} // namespace state_observation

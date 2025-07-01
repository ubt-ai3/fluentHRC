#include "CppUnitTest.h"
#include "app_visualization/module_manager.hpp"
#include "state_observation/object_prototype_loader.hpp"

#include <depthcameragrabber/include/depth_camera_grabber/grabber.hpp>
#include <state_observation/pointcloud_util.hpp>
#include <simulation/rendering.hpp>

namespace unittests
{

	TEST_CLASS(occlusion_detector_test)
	{
	public:

		TEST_METHOD(test_occlusion)
		{

			using namespace state_observation;
			object_parameters object_params;
			object_params.min_object_height = 0.01;
			object_prototype_loader loader;
			auto prototype = loader.get("wooden cube");

			//add objects to scene here

			const Eigen::Vector3f diag(0, 0, 0);//diag is ignored
			//Eigen::Vector3f trans1{ 0.1,0.1,0.1 };
			//Eigen::Vector3f trans2{ 0.1,0.1,0.2 };
			struct test_case
			{
				std::string name;
				Eigen::Vector3f reference;
				Eigen::Vector3f compare;
				occlusion_detector::occlusion_result expected_output;
			};
			std::vector<test_case> test_cases;
			//disappeared,occluded, present,near_point_count
			test_cases.push_back({ "fully occluded",{0.1,0.1,0.2},{0.1,0.1,0.1},{0,1,0,1} });
			test_cases.push_back({ "unchanged",{0.1,0.1,0.1},{0.1,0.1,0.1},{0,0,1,1} });
			test_cases.push_back({ "neglegible change",{0.1,0.1,0.1},{0.1,0.1,0.105},{0,0,1,1} });
			test_cases.push_back({ "smallest relevant change to disappear",{0.1,0.1,0.1},{0.1,0.1,0.115},{1,0,0,1} });
			test_cases.push_back({ "half disappeared",{0.1,0.1,0.1},{0.114,0.1,0.1},{0.5,0.0,0.5,0.5} });
			test_cases.push_back({ "half occluded half disappeared ",{0.1,0.1,0.2},{0.114,0.1,0.1},{0.5,0.5,0.0,0.5} });
			//test_cases.push_back({ "disappeared",{0.1,0.1,0.1},{1,1,1},{1,0,0,0} });
			//obb bb1(diag,trans1);
			//obb bb2(diag, trans2);
			//bb1.
		//	env1->add_object(prototype, bb1,true);
			//env2->add_object(prototype, bb2,true);
			pcl::visualization::Camera cam;
			cam.focal[0] = 0, cam.focal[1] = 0, cam.focal[2] = 0;
			cam.fovy = 90. / 360. * M_PI;
			cam.pos[0] = 0;
			cam.pos[1] = 0;
			cam.pos[2] = 2;
			cam.clip[0] = 0.1;
			cam.clip[1] = 10;
			cam.view[0] = 0.f;
			cam.view[1] = 1.f;
			cam.view[2] = 0.f;
			cam.window_size[0] = 1000;
			cam.window_size[1] = 1000;
			cam.window_pos[0] = 0;
			cam.window_pos[1] = 0;
			Eigen::Matrix4d projection;
			Eigen::Matrix4d view;
			cam.computeProjectionMatrix(projection);
			cam.computeViewMatrix(view);

			Eigen::Affine3d world_to_camera(Eigen::Translation3d(0, 0, -2));

			projection = Eigen::Matrix4d::Identity();
			projection(3, 2) = -1 * 1;//view direction is negative z, display plane is 1 unit apart
			projection(3, 3) = 0;
			view = world_to_camera.matrix();


			simulation::pc_renderer renderer1(1, nullptr, nullptr, 800, 600, false);


			for (auto& test : test_cases)
			{
				auto pc1 = pcl::make_shared<pcl::PointCloud<occlusion_detector::PointT>>();
				auto pc2 = pcl::make_shared<pcl::PointCloud<occlusion_detector::PointT>>();
				auto env1 = std::make_shared<simulation::environment>(object_params);
				auto env2 = std::make_shared <simulation::environment>(object_params);

				env1->additional_scene_objects.push_back(std::make_shared<simulation::simulated_table>());
				env2->additional_scene_objects.push_back(std::make_shared<simulation::simulated_table>());

				env1->add_object(prototype, obb{ diag, test.reference }, true);
				env2->add_object(prototype, obb{ diag,test.compare }, true);

				pointcloud_preprocessing prepro1(std::make_shared<std::decay_t<decltype(object_params)>>(object_params));
				pointcloud_preprocessing prepro2(std::make_shared<std::decay_t<decltype(object_params)>>(object_params));

				//render
				renderer1.env = env1;
				pcl::copyPointCloud(*renderer1.render(0), *pc1);
				renderer1.env = env2;
				pcl::copyPointCloud(*renderer1.render(0), *pc2);

				//remove table
				auto pc1_transformed = prepro1.remove_table(pc1);
				auto pc2_transformed = prepro2.remove_table(pc2);
				pc1 = nullptr;
				pc2 = nullptr;
				//perform detection
				occlusion_detector detect(pc1_transformed, object_params, (projection * view).cast<float>());
				pc_segment seg;
				seg.points = pc2_transformed;
				auto f = detect.perform_detection(*pc2_transformed);
				float disappeared_diff = test.expected_output.disappeared_pct - f.disappeared_pct;
				float occluded_diff = test.expected_output.occluded_pct - f.occluded_pct;
				float present_diff = test.expected_output.present_pct - f.present_pct;
				float near_point_diff = test.expected_output.near_point_pct - f.near_point_pct;
				float total_error = std::abs(disappeared_diff) + std::abs(occluded_diff) + std::abs(present_diff) + std::abs(near_point_diff);
				std::cout << "Error in Testcase " << test.name << " " << total_error << "\n";
				//using namespace Microsoft::VisualStudio::CppUnitTestFramework;
				using Microsoft::VisualStudio::CppUnitTestFramework::Assert;
				Assert::IsTrue(total_error < 1);
			}

		}
	};
}
#include "module_manager.hpp"

#include <filesystem>

#include <pcl/io/grabber.h>
#include "KinectGrabber/kinect2_grabber.h"
#include <icl_core_logging/icl_core_logging.h>
//#include <state_observation/calibration.hpp>
#include "viewer.hpp"

#include "state_observation/object_prototype_loader.hpp"
#include <state_observation/workspace_objects.hpp>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "benchmark.hpp"
#include <simulation/baraglia17.hpp>
#include <simulation/behavior_test.hpp>
#include <simulation/riedelbauch17.hpp>
#include <simulation/hoellerich22.hpp>
#include <simulation/rendering.hpp>


#include "intention_visualizer.hpp"
#include "simulated_hand_tracking.hpp"
#include <app_visualization/mogaze.hpp>
#include <app_visualization/util.hpp>

#include <state_observation/workspace_calibration.h>
#include <franka_proxy_share/franka_proxy_util.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

using namespace prediction;
using namespace hand_pose_estimation;
using namespace state_observation;
using namespace std::chrono;
using namespace std::literals::chrono_literals;

#define DEBUG
#ifdef USE_HOLOLENS

void server_callback::PreSynchronousRequest(grpc::ServerContext* context)
{
	if (first) return;
	
	first = true;
	on_first_connect();
}

void server_callback::PostSynchronousRequest(grpc::ServerContext* context)
{}

bool server_callback::get_first() const
{
	return first;
}
/*
simulation::sim_task::Ptr create_simulation_and_task_for_testing(const object_parameters& object_params)
{
	using namespace simulation;
	//object_prototype_loader loader;

	auto env = std::make_shared<environment>(object_params);
	env->additional_scene_objects.push_back(std::make_shared<simulated_table>());

	return std::make_shared<sim_task>("default_building", env, std::vector<agent::Ptr>({}), );
}*/
#endif

module_manager::module_manager(int argc, char* argv[], camera_type camera)
	: object_params(std::make_shared<object_parameters>()),
	kinect_params(std::make_shared<kinect2_parameters>())
{
	auto start = high_resolution_clock::now();

	/*
	* The task that is executed
	*/
	//DEBUG
	object_prototype_loader loader;
	int count_agents =
#ifdef USE_ROBOT
		3; // robot + 2 hands
#else
		2;
#endif

	simulation::sim_task::Ptr task;
	std::unique_ptr<task_manager> task_manage;
	if(camera == camera_type::SIMULATION)
	{
		//auto task = simulation::baraglia17::task_a_1(*object_params, loader);
		//auto task = simulation::riedelbauch17::pack_and_stack(*object_params, loader);
		task = simulation::hoellerich22::structure_1(*object_params, loader);
		task_manage = std::make_unique<task_manager>(*task);

		count_agents = task->agents.size();
		//auto build_and_simulation = create_building_simulation_task(*object_params, loader);
		//auto task = build_and_simulation->task;
		
	}
	else
	{
		task_manage = std::make_unique<task_manager>(*object_params, loader, count_agents, start);
	}

	auto net = task_manage->net;
	auto init_marking = task_manage->initial_marking;
	std::vector<pn_place::Ptr> agent_places(net->get_places().begin(), net->get_places().begin() + count_agents);

	pc_prepro = std::make_unique<pointcloud_preprocessing>(object_params, camera == camera_type::SIMULATION);
	std::vector<actor_occlusion_detector::Ptr> occlusion_detectors;
	auto get_detector = [&]()
	{
		occlusion_detectors.emplace_back(std::make_shared<actor_occlusion_detector>(object_params->min_object_height, kinect_params->depth_projection * pc_prepro->get_inv_cloud_transformation().matrix()));
		return occlusion_detectors.back();
	};

	// Recognition pipeline - steps
	std::vector<object_prototype::ConstPtr> prototypes/*({ loader.get("wooden block"), loader.get("red cube") })*/;
	std::map<object_prototype::ConstPtr, pn_object_token::Ptr> token_traces;
	for (const auto& token : net->get_tokens())
	{
		if (auto obj_token = std::dynamic_pointer_cast<pn_object_token>(token)) 
		{
			prototypes.push_back(obj_token->object);
			token_traces.emplace(obj_token->object, obj_token);
		}
	}

	//obj_detect = std::make_unique<segment_detector>(world, *pc_prepro);

	//obj_track = std::make_unique<object_tracker>(world, get_detector(), *object_params, 0.5, 1e10);


	obj_classify = std::make_unique<place_classification_handler>(world, *pc_prepro, object_params, get_detector(), prototypes, net, token_traces, false);

	Eigen::Hyperplane<float, 3> table = pc_prepro->workspace_params.get_plane();
	if (camera != camera_type::SIMULATION) {
		hand_track = std::make_unique<hand_tracker_enact>(world, 
			pc_prepro->workspace_params.get_cloud_transformation(), 
			std::chrono::duration<float>(1.f), 
			std::chrono::duration<float>(0.2f), 
			3, // threads
			false, // only track HoloLens hands	
			start,
			Eigen::Vector3f(0.f, 0.f, -1.f), table);
		//hand_track->show_skin_regions();
	}

#ifdef USE_HOLOLENS
	/**
	 * Documentaion of grpc::Server::SetGlobalCallbacks is wrong
	 * Function DOES take ownership
	 * https://github.com/grpc/grpc/issues/23204
	 */
	grpc::Server::SetGlobalCallbacks(new server_callback());
	server = std::make_unique<server::server_module>(pc_prepro->workspace_params);

	boost::signals2::signal<void(const hololens::hand_data::ConstPtr&, enact_priority::operation)> hand_signal;
	auto hand_callback = [&hand_signal](const hololens::hand_data::ConstPtr& hand_data)
	{
		enact_priority::operation op = hand_data->valid
			? enact_priority::operation::UPDATE
			: enact_priority::operation::MISSING;

		hand_signal(hand_data, op);
	};
	server::hand_tracking_service::on_tracked_hand.connect(std::move(hand_callback));
#endif	
	view = std::make_unique<viewer>(world);


	auto initial_marking = std::make_shared<pn_belief_marking>(init_marking);

	agent_manage = std::make_shared<agent_manager>(world, net, pc_prepro->workspace_params, std::vector<pn_place::Ptr>(
#ifdef USE_ROBOT
		std::next(agent_places.begin()),
#else
		agent_places.begin(),
#endif
		agent_places.end()));

	progress_viewer = std::make_unique<task_progress_visualizer>(world, *obj_classify, initial_marking, agent_manage, start);
	//progress_viewer = std::make_unique<task_progress_visualizer>(world, *tracer, initial_marking);

	//intention_view = std::make_unique<intention_visualizer>(world, pc_prepro->workspace_params, *agent_manage, *view, initial_marking);
	//build_estimation = std::make_unique<building_estimation>(world, prototypes);

#ifdef USE_ROBOT

	std::shared_ptr<Controller> controller;

	if (camera != camera_type::SIMULATION)
	{
		while (true)
		{
			try
			{
				controller = std::make_shared<remote_controller_wrapper>(start);
				break;
			}
			catch (...)
			{
				std::cout << "Retrying connection to robot\n";
			}
		}
	}
	else
		controller = std::make_shared<simulation::simulation_controller_wrapper>(task->env);

	robot = std::make_unique<robot::agent>(net, std::make_unique<robot::null_planner>(*agent_places.begin()), controller);

#if defined(USE_HOLOLENS) && defined(USE_ROBOT)
	/*robot->get_joint_signal().connect([this](const franka_proxy::robot_config_7dof& joints)
		{
			server->voxel_service.joint_slot(joints);
		});*/
#endif
#endif

#ifdef USE_HOLOLENS
	present = std::make_unique<presenter>(franka_visualizations{ true, true, true }, server, get_detector());
	present->update_net(net);
#endif
	std::atomic_bool termination_flag = false;

	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector3f> recorded_positions;
	int current_pos = 0;
	// show all places
	int i = 7352890;
	size_t k = 0;
	for (const auto& entry : net->get_places())
	{
		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(entry);
		if (boxed_place)
		{
			view->add_bounding_box(boxed_place->box, std::to_string(i++), 1, 0);
			if (k == 32)
				continue;

			positions.emplace_back(boxed_place->box.translation);
			positions.back().z() += boxed_place->box.diagonal.z() * 0.5;
			view->update_element([&](pcl::visualization::PCLVisualizer& viewer)
				{
					Eigen::Affine3f pos = Eigen::Affine3f(Eigen::Translation3f(positions[current_pos]));
					viewer.addCoordinateSystem(0.05, pos,  "box" + std::to_string(current_pos), view->viewport_full_window_id());
				});

			++k;
		}
	}
	recorded_positions.reserve(positions.size());
	/*
	recorded_positions = {
		{0.207435533, -0.401944935, 0.0311098546},
{0.307141572, -0.402615845, 0.0325025879},
{0.408115923, -0.400116414, 0.0348668732},
{0.506821275, -0.399778932, 0.0373118520},
{0.605638444, -0.398767054, 0.0400879048},
{0.206987500, -0.302525908, 0.0301033854},
{0.307435453, -0.300875038, 0.0315368660},
{0.407982528, -0.301074117, 0.0335349143},
{0.509288549, -0.299264967, 0.0367717780},
{0.608200431, -0.298482895, 0.0383846164},
{0.306751579, -0.201326326, 0.0298052542},
{0.407206923, -0.200159848, 0.0321140550},
{0.508275032, -0.198744670, 0.0353800505},
{0.607536852, -0.198159695, 0.0376678891},
{0.305040389, 0.203177869, 0.0302464068},
{0.405461401, 0.204190552, 0.0315950625},
{0.504463732, 0.204236850, 0.0340634435},
{0.606147468, 0.204291031, 0.0370394550},
{0.204826385, 0.304436654, 0.0293540712},
{0.198974580, 0.405220687, 0.0298711695},
{0.200133070, 0.505676091, 0.0305139087},
{0.304427058, 0.304359406, 0.0305043217},
{0.404810041, 0.304569662, 0.0324864797},
{0.507228732, 0.303699642, 0.0345343053},
{0.606152892, 0.303896517, 0.0370615162},
{0.302917480, 0.405035585, 0.0320522748},
{0.404609263, 0.405552447, 0.0335530750},
{0.507383347, 0.405417800, 0.0361543596},
{0.604718447, 0.403808296, 0.0384977050},
{0.302539498, 0.503919065, 0.0629585162},
{0.402949631, 0.502329886, 0.0641657189},
{0.505771756, 0.502892077, 0.0673794895}
	};*/

	//Eigen::Affine3f out = calibrate_by_points(std::vector<Eigen::Vector3f>(positions.begin(), positions.begin() + 32), recorded_positions) * pc_prepro->workspace_params.get_cloud_transformation();
	//Eigen::Affine3f out_2 = calibrate_by_points(std::vector<Eigen::Vector3f>(positions.begin(), positions.begin() + 32), recorded_positions).inverse() * pc_prepro->workspace_params.get_cloud_transformation();
	//std::cout << out.matrix() << std::endl << std::endl;
	//std::cout << out_2.matrix() << std::endl << std::endl;

	// Grabber functions	

	auto cloud_signal = std::make_shared<cloud_signal_t>();
	std::weak_ptr<cloud_signal_t> weak_cloud_signal = cloud_signal;

	auto image_signal = std::make_shared<image_signal_t>();
	std::weak_ptr<image_signal_t> weak_image_signal = image_signal;

	//Eigen::Translation2f center(1920 * 0.5f, 1080 * 0.5f);
	//Eigen::Matrix<float, 3,4> projection =  center * Eigen::Affine2f(Eigen::Rotation2D<float>(M_PI)) * center.inverse() * kinect_params->rgb_projection * pc_prepro->get_inv_cloud_transformation().matrix();

	bool first_0 = true;

	std::function pc_grabber =
		[&weak_cloud_signal, this, &first_0, &net](const pcl::PointCloud<PointT>::ConstPtr& input) {
		
#ifdef USE_HOLOLENS

		auto pre_process = [this](const pcl::PointCloud<PointT>::ConstPtr& input)
			-> pcl::PointCloud<pcl::PointXYZ>::Ptr
			{
				const auto matrix = pc_prepro->get_cloud_transformation().matrix();
				const pcl::detail::Transformer tf(matrix);
				auto processed = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
				processed->header = input->header;
				processed->is_dense = input->is_dense;
				processed->height = 1;
				// trade-off between avoiding copies and an over-sized vector 
				processed->points.reserve(1000);
				processed->sensor_orientation_ = input->sensor_orientation_;
				processed->sensor_origin_ = input->sensor_origin_;

				const auto& points_in = input->points;
				auto& points_out = processed->points;

				for (const auto& p_in : points_in)
				{
					if (!std::isfinite(p_in.x) ||
						!std::isfinite(p_in.y) ||
						!std::isfinite(p_in.z))
						continue;

					pcl::PointXYZ p;
					tf.se3(p_in.data, p.data);

					if (p.z >= -0.05f && p.y > -0.7)
						points_out.emplace_back(p);
				}
				processed->width = points_out.size();

				return processed;
			};
		
			/*auto calibrate = [this, &net](const pcl::PointCloud<PointT>::ConstPtr& input)
			{
				std::vector<pn_boxed_place::Ptr> boxed_places;
				size_t k = 0;
				for (const auto& entry : net->get_places())
				{
					if (k == 32)
						break;
					auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(entry);
					if (!boxed_place)
						continue;
					boxed_places.emplace_back(boxed_place);
					++k;
				}
				auto point_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
				point_cloud->header = input->header;
				point_cloud->is_dense = input->is_dense;
				point_cloud->height = 1;
				// trade-off between avoiding copies and an over-sized vector 
				point_cloud->points.reserve(input->points.size());
				point_cloud->sensor_orientation_ = input->sensor_orientation_;
				point_cloud->sensor_origin_ = input->sensor_origin_;
				point_cloud->width = input->size();

				for (const auto& p : input->points)
					point_cloud->points.emplace_back(p.x, p.y, p.z);

				auto result = state_observation::full_calibration(point_cloud, boxed_places, *pc_prepro).matrix();
				//pc_prepro->workspace_params.transformation = result;
				std::cout << std::endl << result << std::endl;
			};*/
			auto copy = [](const pcl::PointCloud<PointT>::ConstPtr& input)
				{
					auto point_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
					point_cloud->header = input->header;
					point_cloud->is_dense = input->is_dense;
					point_cloud->height = 1;
					// trade-off between avoiding copies and an over-sized vector 
					point_cloud->points.reserve(input->points.size());
					point_cloud->sensor_orientation_ = input->sensor_orientation_;
					point_cloud->sensor_origin_ = input->sensor_origin_;
					point_cloud->width = input->size();

					for (const auto& p : input->points)
						point_cloud->points.emplace_back(p.x, p.y, p.z);

					return point_cloud;
				};

			/*auto post_calibration = [&](const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& input)
				{
					pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

					*cloud = *input;
					pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
					{
						pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
						voxel_grid.setInputCloud(cloud);
						voxel_grid.setLeafSize(0.007f, 0.007f, 0.007f);
						voxel_grid.filter(*cloud);
					}

					pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
					pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
					{
						// Create the segmentation object
						pcl::SACSegmentation<pcl::PointXYZ> seg;
						// Optional
						seg.setOptimizeCoefficients(false);
						// Mandatory
						seg.setModelType(pcl::SACMODEL_PLANE);
						seg.setMethodType(pcl::SAC_RANSAC);
						seg.setDistanceThreshold(0.1);
						seg.setMaxIterations(20000);

						seg.setInputCloud(cloud);
						seg.segment(*inliers, *coefficients);
					}
					{
						pcl::ExtractIndices<pcl::PointXYZ> extract;
						extract.setInputCloud(cloud);
						extract.setIndices(inliers);
						extract.filter(*cloud_filtered);
						//pcl::copyPointCloud(*cloud_filtered, *cloud_temp);
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

						pcl::ExtractIndices<pcl::PointXYZ> extract;
						extract.setInputCloud(cloud_filtered); //TODO:: range test
						auto sth = std::make_shared<pcl::PointIndices>(cluster_indices[0]);
						extract.setIndices(sth);
						extract.filter(*cloud_filtered);
					}
					{
						// Create the segmentation object
						pcl::SACSegmentation<pcl::PointXYZ> seg;
						// Optional
						seg.setOptimizeCoefficients(true);
						// Mandatory
						seg.setModelType(pcl::SACMODEL_PLANE);
						seg.setMethodType(pcl::SAC_RANSAC);
						seg.setDistanceThreshold(0.01);
						seg.setMaxIterations(1000);

						seg.setInputCloud(cloud_filtered);
						seg.segment(*inliers, *coefficients);
					}
					{
						pcl::ExtractIndices<pcl::PointXYZ> extract;
						extract.setInputCloud(cloud_filtered);
						extract.setIndices(inliers);
						extract.filter(*cloud_filtered);
					}
					Eigen::Vector3f floor_normal = { coefficients->values[0], 0, coefficients->values[2] };
					floor_normal.normalize();
					Eigen::Vector3f rotation_vec;

					Eigen::Vector3f xy_plane_normal = { 0.f, 0.f, 1.f };

					rotation_vec = xy_plane_normal.cross(floor_normal);
					rotation_vec.normalize();

					auto meh = -acos(floor_normal.dot(xy_plane_normal));

					std::cout << "Theta: " << meh << ", " << meh * 180.f / std::numbers::pi << "�" << std::endl;

					Eigen::Affine3f transform{ Eigen::AngleAxisf(meh, rotation_vec) };


					pcl::transformPointCloud(*cloud_filtered, *cloud_filtered, transform);

					float max_z = 0.f;
					for (const auto& p : cloud_filtered->points)
						max_z = std::max(max_z, p.z);

					return Eigen::Translation3f(0, 0, -max_z) * transform;
				};*/

		server->service_pcl.set_ref(pre_process(input), copy(input));


#endif
		last_cloud = input;
		if (const auto cloud_signal = weak_cloud_signal.lock())
		{
			//cloud_signal->operator()(pre_process_2());
			cloud_signal->operator()(pc_prepro->remove_table(input));
			if (first_0)
			{
				
				//pcl::io::savePLYFileBinary("meh2.ply", *pre_process(input));
				first_0 = false;
				//calibrate(input);
				//pc_prepro->workspace_params.transformation = post_calibration(pre_process(input)) * pc_prepro->workspace_params.transformation;
			}
		}
	};

	int frame_count = 0;
	Eigen::Vector3f sensor_origin = pc_prepro->get_cloud_transformation().translation();
	std::function img_grabber =
		[&weak_image_signal, this, &frame_count, &start, &sensor_origin](const cv::Mat4b& img) {
		//if (frame_count++ >= 30)
		//{
		//	std::cout << "Input: " << frame_count / (double)std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - start).count() << " FPS" << std::endl;
		//	frame_count = 0;
		//	start = std::chrono::high_resolution_clock::now();
		//}

		if (last_cloud)
		{
			auto input = std::make_shared<visual_input>(last_cloud, img, kinect_params->rgb_projection, kinect_params->depth_projection, sensor_origin);
			if (hand_track)
				hand_track->update(input);
			progress_viewer->update(input->timestamp_seconds);
		}

		if (const auto image_signal = weak_image_signal.lock())
			image_signal->operator()(img);
	};


	// Recognition pipeline - interwire steps
	cloud_signal->connect([&](const pcl::PointCloud<PointT>::ConstPtr& cloud) {
		//obj_detect->update(cloud);
		//obj_track->update(cloud);
		obj_classify->update(cloud);
		/*([&](const pcl::PointCloud<PointT>::ConstPtr& cloud)
			{
				if (!first_0)
					return;
				first_0 = false;

				pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
				tree->setInputCloud(cloud);

				std::vector<pcl::PointIndices> cluster_indices;

				pcl::EuclideanClusterExtraction<PointT> ec;

				ec.setClusterTolerance(0.02); // 2cm
				ec.setMinClusterSize(10);
				ec.setSearchMethod(tree);
				ec.setInputCloud(cloud);
				ec.extract(cluster_indices);

				std::vector<Eigen::Vector2f> is;
				int i = 0;
				for (const auto& cluster : cluster_indices)
				{
					pcl::CentroidPoint<PointT> centroid;
					for (auto idx : cluster.indices)
						centroid.add((*cloud)[idx]);

					PointT center;
					centroid.get(center);

					Eigen::Affine3f pose = Eigen::Affine3f(Eigen::Translation3f(center.x, center.y, 0.f));
					is.emplace_back(center.x, center.y);
					view->update_element([&, pose, viewport = view->viewport_full_window_id()](pcl::visualization::PCLVisualizer& viewer)
						{

							viewer.addCoordinateSystem(.02, pose, std::string("coord") + std::to_string(i++), viewport);
						});
				}

				auto places_point_cloud = std::make_shared<pcl::PointCloud<pcl::PointXY>>();
				std::vector<Eigen::Vector2f> should;

				for (const auto& entry : net->get_places())
				{
					auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(entry);
					if (boxed_place)
					{
						auto t = boxed_place->box.translation;
						places_point_cloud->emplace_back(pcl::PointXY{ t.x(), t.y() });
					}
				}
				pcl::KdTreeFLANN<pcl::PointXY> kd_tree;
				std::vector<int> pointIdxKNNSearch(1);
				std::vector<float> pointKNNSquaredDistance(1);

				kd_tree.setInputCloud(places_point_cloud);
				for (const auto& p : is)
				{
					kd_tree.nearestKSearch(pcl::PointXY{ p.x(), p.y() }, 1, pointIdxKNNSearch, pointKNNSquaredDistance);
					auto f_p = (*places_point_cloud)[pointIdxKNNSearch[0]];
					should.emplace_back(f_p.x, f_p.y);
				}
				Eigen::Affine3f refinement = calibrate_by_points(should, is);
				std::cout << "refined: " << std::endl << (refinement.inverse() * pc_prepro->workspace_params.get_cloud_transformation()).matrix() << std::endl;

			})(cloud);*/
		view->update_cloud(cloud);
			// header stamp is in microseconds but update expects seconds
			progress_viewer->update(std::chrono::microseconds(cloud->header.stamp));
			//tracer->update(cloud);
#ifdef USE_HOLOLENS
			present->update_cloud(cloud);
#endif
		});

	image_signal->connect([&](const cv::Mat& img) {
		view->update_image(img);
		});

	//object_prototype::ConstPtr ref_obj;
	//for(const auto& proto : prototypes)
	//	if(proto->get_name().compare("purple bridge") == 0)
	//	{
	//		ref_obj = proto;
	//		break;
	//	}

	//auto sig1 = obj_detect->get_signal(enact_priority::operation::CREATE);
	//sig1->connect([&](const std::vector<pc_segment::Ptr>& segments) {
	//	obj_track->update(segments);
	//	//task->evaluate_net(obj_track->get_latest_timestamp());
	//	/*for(const auto& segment : segments)
	//	{
	//		auto results = obj_classify->classify_segment(segment);
	//		if(results.front().local_certainty_score > 0.5 && results.front().prototype == ref_obj)
	//		{
	//			std::stringstream stream;
	//			boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();

	//			stream << ref_obj->get_name() << time.date().year()
	//				<< "-" << time.date().month()
	//				<< "-" << time.date().day()
	//				<< "-" << time.time_of_day().hours()
	//				<< "-" << time.time_of_day().minutes()
	//				<< ".pcd";

	//			std::string name(stream.str());

	//			pcl::io::savePCDFileASCII(name, *segment->points);
	//		}
	//	}*/
	//	});

	//auto sig6 = obj_track->get_signal();
	//sig6->connect([&](const entity_id& id, enact_priority::operation op) {
	//	view->update_object(id, op);
	//	});

	//auto sig2 = obj_track->get_signal(enact_priority::operation::UPDATE);
	//sig2->connect([&](const entity_id& id) {
	//	obj_classify->update(id);
	//	progress_viewer->update(id, enact_priority::operation::UPDATE);
	//	view->update_object(id, enact_priority::operation::CREATE);
	//	});

	//auto sig3 = obj_track->get_signal(enact_priority::operation::DELETED);
	//sig3->connect([&](const entity_id& id) {
	//	view->update_object(id, enact_priority::operation::DELETED);
	//	progress_viewer->update(id, enact_priority::operation::DELETED);
	//	//build_estimation->update(id, enact_priority::operation::DELETED);
	//	});

	auto sig4 = obj_classify->get_signal();
	std::atomic_bool executing = false;

	Eigen::Vector3f drop_pos = {0.62f,0,0};
	sig4->connect([&](const entity_id& id, enact_priority::operation op) {
		view->update_object(id, op);
		//DEBUG robot stack cubes
		//  let robot grip objects and filter objects of a certain color 
		//{


		//enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::read));
		//enact_core::const_access<classification_handler::object_instance_data> access_object(l.at(id, object_instance::aspect_id));
		//const object_instance& obj = access_object->payload;
		//auto seg = obj.get_classified_segment();

		//if (!seg)
		//	return;

		//if(seg->classification_results.front().local_certainty_score > 0.25 && loader.get("red cube") == seg->classification_results.front().prototype && seg->bounding_box.top_z < 0.05){

		//	if (!executing.exchange(true))
		//	{
		//		std::cout << "gripping " << obj.observation_history.back()->bounding_box.translation << std::endl;

		//		obb pick = aabb({ 0.03,0.03,0.03 }, { 0.6,0.0,0.015 });
		//		robot->picknplace(obj.observation_history.back()->bounding_box, aabb(obj.observation_history.back()->bounding_box.diagonal, drop_pos));
		//		//robot->picknplace(pick, aabb(obj.observation_history.back()->bounding_box.diagonal, drop_pos));
		//		drop_pos.z() += 0.03;

		//		//obb place = aabb({ 0.04,0.04,0.04 }, { 0.1,-0.5,0.0 });
		//		//robot->picknplace(pick, place);
		//		std::cout << "gripped\n";
		//		executing = false;
		//	}
		//}

		//}

		
		//build_estimation->update(id, op);
		//test_function(id);
		});

	auto sig5 = progress_viewer->get_signal();
	sig5->connect([&](const std::pair<state_observation::pn_belief_marking::ConstPtr, std::map<state_observation::pn_transition::Ptr, double>>& marking_transitions, enact_priority::operation op) {
		//intention_view->update(marking_transitions.first);
#ifdef USE_ROBOT
		robot->update_marking(marking_transitions.first);
#endif
#ifdef USE_HOLOLENS
		present->update_marking(marking_transitions.first);
#endif
		if(task_manage)
			task_manage->update_marking(marking_transitions.first);
	});

	decltype(hand_track->get_signal()) sig6; 
	if (camera != camera_type::SIMULATION && hand_track)
	{
#ifdef USE_HOLOLENS
		hand_signal.connect([&](const hololens::hand_data::ConstPtr& hand_data, enact_priority::operation op) {
			hand_track->add_hand_pose(hand_data, op);
		});
#endif
		sig6 = hand_track->get_signal();
		sig6->connect([&](const std::pair<entity_id, img_segment::ConstPtr>& entry, enact_priority::operation op) {
			agent_manage->update(entry.first, op);
			view->update_hand(entry.first, op);

			for (const auto& detector : occlusion_detectors)
				detector->update_hand(entry.first, entry.second);
			});
	}

#ifdef USE_ROBOT
	auto sig7 = robot->get_signal();
	sig7->connect([&](const auto& t, auto op)
		{
#ifdef USE_HOLOLENS
			present->update_robot_action(t, op);
#endif
			if (op == enact_priority::operation::DELETED)
				progress_viewer->update(std::get<0>(t));
		});
#endif

	std::shared_ptr<boost::signals2::signal<void(pn_belief_marking::ConstPtr)>> sig8;
	if (task_manage) {
		sig8 = task_manage->get_signal(enact_priority::operation::UPDATE);
		sig8->connect([&](const auto& t)
			{
				progress_viewer->update_goal(t);
#ifdef USE_ROBOT
				robot->update_goal(t);
#endif
#ifdef USE_HOLOLENS
				present->update_goal(t);
#endif
			});
	}

	// Register grabber based on camera type and show result
	std::thread puller;
	//std::thread scene_renderer;

	pcl::shared_ptr<pcl::Grabber> grabber;
#ifdef USE_HOLOLENS
	server->load_mesh_data(loader);
	
	/* {
		std::mutex mtx;
		std::condition_variable cv;

		callback.on_first_connect.connect([&cv]()
			{
				cv.notify_all();
			});
		std::unique_lock lock(mtx);
		cv.wait(lock, [this]()
		{
			return callback.get_first();
		});
	}*/

	std::thread server_thread([this]()
		{
			server->get_instance().Wait();
		});
#endif
	
	switch (camera)
	{
	case camera_type::SIMULATION:
	{
		puller = std::thread([&]() {
			try
			{
					auto time = std::chrono::duration<float>(0);

					double fps = 25;
					auto duration = std::chrono::milliseconds(static_cast<int>(1000 / fps));
					
					simulation::task_execution execution(task, fps);
				//simulation::repeated_task_execution execution(task, fps);

					// pc_renderer must NOT be constructed and run in the same thread as a PCLVisualizer
					simulation::pc_renderer renderer(argc, argv, task->env, 640, 720, false);

#ifdef USE_ROBOT
					auto ref_time = std::chrono::high_resolution_clock::now();
					auto robo_controller = std::static_pointer_cast<simulation::simulation_controller_wrapper>(controller);
					robo_controller->reset(ref_time);
					robot->reset(ref_time);
					task->env->additional_scene_objects.emplace_back(robo_controller);

					auto rob_sig = robot->get_signal();
					rob_sig->connect([&](const robot::agent_signal& signal, enact_priority::operation op)
						{
							auto action = get<0>(signal);
							robo_controller->set_action(action);
						});

#endif
					
					auto hand_kin_params = std::make_unique<hand_kinematic_parameters>();
					simulated_hand_tracker sim_hand_tracking(world,
						{
#ifdef USE_ROBOT
							++task->agents.begin(),
#else
							task->agents.begin(),
#endif
							task->agents.end()
						},
						*agent_manage, *hand_kin_params);

					auto sig9 = sim_hand_tracking.get_signal();
					sig9->connect([&](const entity_id& id, enact_priority::operation op) {
						agent_manage->update(id, op);
						view->update_hand(id, op);
						});

					auto prev_marking = task->env->get_marking();

					// wait for 1s to detect initial scene
					pc_grabber(pointcloud_preprocessing::to_pc_rgba(renderer.render(time)));
					std::this_thread::sleep_for(500ms);
					pc_grabber(pointcloud_preprocessing::to_pc_rgba(renderer.render(time + std::chrono::duration<float>(1.f))));
					std::this_thread::sleep_for(500ms);
					while (!progress_viewer->is_initial_recognition_done()) {
						std::cerr << "Initial workspace state not detected, retrying..." << std::endl;
						std::this_thread::sleep_for(std::chrono::seconds(2));
					}

					task_manage->next();


					bool finished = false;

					while (!termination_flag)
					{
						auto frame_start = std::chrono::high_resolution_clock::now();
						execution.step();
						auto pc_rgb = renderer.render(time);

#ifdef USE_ROBOT
						auto wrapper = std::static_pointer_cast<simulation::simulation_controller_wrapper>(controller);
						view->update_element([&wrapper, tp = time, viewport = view->viewport_overlay_id()](pcl::visualization::PCLVisualizer& visualizer)
							{
								wrapper->render(visualizer, tp, viewport);
							});						
#endif

						//Debug
						/*

						auto marking = task->env->get_marking();
						if (!(*marking == *prev_marking))
						{
							ground_truth_renderer.update(marking);
							prev_marking = marking;
						}
						*/

						pc_grabber(pointcloud_preprocessing::to_pc_rgba(pc_rgb));
						progress_viewer->update(time);

						sim_hand_tracking.generate_emissions(time);

						std::this_thread::sleep_until(frame_start + duration);
						time = time + std::chrono::duration<float>(1.f / fps);

						if (finished)
							continue;

						finished = true;

						for (const auto& agent : task->agents)
						{
							if (!agent->is_idle(time))
							{
								finished = false;
								break;
							}
						}
#ifdef USE_ROBOT
						//TODO::
						//if (robo_controller->)
						finished = false;
#endif

						if (finished) {
							std::cout << "finished" << std::endl;

							for (const auto& agent : task->agents)
							{
								if (auto human_agent = std::dynamic_pointer_cast<simulation::human_agent>(agent); human_agent)
									human_agent->print_actions();
							}
						}
#ifdef USE_ROBOT
						ref_time += duration;
						robo_controller->set_time(ref_time);
#endif
					}
#ifdef USE_ROBOT
					robot.reset();
#endif
					progress_viewer.reset();
				}
				catch (const std::exception& e)
				{
					std::cout << e.what() << std::endl;
				}
			});

		//scene_renderer = std::thread([&]()
		//	{
		//		simulation::model_renderer renderer(task->env);
		//		float prev_time = 0;
		//		while (!termination_flag)
		//		{
		//			if (prev_time != time)
		//			{
		//				renderer.update(time);
		//				prev_time = time;
		//			}
		//			renderer.show();
		//		}
		//	});

		break;
	}

	case camera_type::KINECT_V2:
	{
		// Kinect2Grabber
		grabber = pcl::make_shared<pcl::Kinect2Grabber>();

		// Register Callback Function
		boost::signals2::connection connection_pc = grabber->registerCallback(pc_grabber);
		boost::signals2::connection connection_img = grabber->registerCallback(img_grabber);

		grabber->start();

		break;
	}

	case camera_type::REALSENSE:
	{
		puller = std::thread([&]()
			{
				try {
					//TODO Merge
					/*auto grabber = depth_camera_grabber::realsense_grabber::create({ 1280, 720 },
						{ 1280, 720 },
					15);

					pcl::PointCloud<PointT>::Ptr prev_cloud;
					grabber->enable_decimation_filter();
					grabber->set_decimation_filter_magnitude(2);

					grabber->enable_spatial_edge_perserving_filter();
					grabber->set_spatial_edge_perserving_filter_magnitude(1);
					grabber->set_spatial_edge_perserving_filter_smooth_alpha(0.66);
					grabber->set_spatial_edge_perserving_filter_smooth_delta(50);

					grabber->enable_temporal_filter();
					grabber->set_temporal_filter_smooth_alpha(0.22);
					grabber->set_temporal_filter_smooth_delta(40);*/

					while (!termination_flag)
					{
						/*if (!prev_cloud)
							prev_cloud = grabber->grab_colored_cloud();
						else
						{
							pc_grabber(pc_prepro->fuse(prev_cloud, grabber->grab_colored_cloud()));
							prev_cloud = nullptr;
						}*/
						//pc_grabber(grabber->grab_colored_cloud());
					}
				}
				catch (const std::exception& e)
				{
					std::cout << e.what() << std::endl;
				}
			});

		break;
	}
	}

	std::ofstream timestamp_file;
	{
		boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
		std::stringstream stream;
		stream << time.date().year()
			<< "-" << time.date().month().as_number()
			<< "-" << time.date().day()
			<< "-" << time.time_of_day().hours()
			<< "-" << time.time_of_day().minutes();

		std::filesystem::create_directory(stream.str());

		timestamp_file.open(stream.str() + "/timestamps.csv");
		timestamp_file << "time (ms)" << std::endl;
	}

	std::thread cmd_input_handler{ [&]()
	{
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		while (!termination_flag)
		{
			char a;
			std::cin.get(a);
			if (termination_flag)
				return;

			switch (a)
			{
			case '\n':
			{
				auto timestamp = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
				timestamp_file << timestamp << std::endl;
				std::cout << "Timestamp: " << timestamp << std::endl;
				break;
			}
			case 'n':
				task_manage->next();
				break;
			case 'm':
				task_manage->decompose();
				break;
#ifdef USE_ROBOT
			case '0':
				robot->update_behaviour(std::make_unique<robot::null_planner>(*agent_places.begin()));
				break;
			case '1':
				robot->update_behaviour(std::make_unique<robot::planner_layerwise_rtl>(*agent_places.begin()));
				break;
			case '2':
				robot->update_behaviour(std::make_unique<robot::planner_adaptive>(*agent_manage, *agent_places.begin()));
				break;
			case '3':
				robot->update_behaviour(std::make_unique<robot::planner_adversarial>(*agent_manage, *agent_places.begin()));
				break;
#endif
			case 'r':
				start = high_resolution_clock::now();
#ifdef USE_ROBOT
				robot->reset(start);
#endif
				if (camera != camera_type::SIMULATION)
					hand_track->reset(start);

				agent_manage->reset(start);
				progress_viewer->reset(start);
				//intention_view->reset();
				task_manage->reset(start);
				break;
			case 'v':
			{
				auto vis = present->franka_visualizer_.visualizations();
				vis.voxels = !vis.voxels;
				present->franka_visualizer_.set_visual_generators(vis);
				break;
			}
			case 'j':
			{
				auto vis = present->franka_visualizer_.visualizations();
				vis.shadow_robot = !vis.shadow_robot;
				present->franka_visualizer_.set_visual_generators(vis);
				break;
			}
			case 't':
			{
				auto vis = present->franka_visualizer_.visualizations();
				vis.tcps = !vis.tcps;
				present->franka_visualizer_.set_visual_generators(vis);
				break;
			}
#ifdef USE_ROBOT
			case 'l': //callibrate by positioning the robot at known positions
			{
				auto config = controller->current_config();
				std::cout << "Current joints: ";
				for (const auto& q : config)
				{
					std::cout << q << ", ";
				}
				std::cout << std::endl;
				auto pos = (franka_proxy::franka_proxy_util::fk(config).back()* Eigen::Translation3d(0., 0., 0.22)).translation().cast<float>();
				std::cout << "Current position: " << pos.x() << ", " << pos.y() << ", " << pos.z() << std::endl << std::endl;

				if (current_pos < positions.size())
				{
					recorded_positions.emplace_back(pos);
					view->update_element([&](pcl::visualization::PCLVisualizer& viewer)
						{
							viewer.removeCoordinateSystem("box" + std::to_string(current_pos), view->viewport_full_window_id());
							++current_pos;
							if (current_pos == positions.size())
							{
								Eigen::Affine3f out = calibrate_by_points(positions, recorded_positions) * pc_prepro->workspace_params.get_cloud_transformation();
								Eigen::Affine3f out_2 = calibrate_by_points(positions, recorded_positions).inverse() * pc_prepro->workspace_params.get_cloud_transformation();

								//pc_prepro->workspace_params.transformation = out;
								std::cout << out.matrix() << std::endl << std::endl;
								std::cout << out_2.matrix() << std::endl << std::endl;
							}


							Eigen::Affine3f pos = Eigen::Affine3f(Eigen::Translation3f(positions[current_pos]));
							viewer.addCoordinateSystem(0.05, pos, "box" + std::to_string(current_pos), view->viewport_full_window_id());
						});
				}
				break;
			}
#endif
			default:
				continue;
			}
		}
	} };

	view->run_sync();
#ifdef USE_HOLOLENS
	server->get_instance().Shutdown(
		std::chrono::system_clock::now() +
		std::chrono::milliseconds(500));

	server_thread.join();
#endif
	
	termination_flag = true;
	if (puller.joinable())
		puller.join();

	//if (scene_renderer.joinable())
		//scene_renderer.join();

	std::cout << std::endl << "You can now close the console by either putting in anything and pressing enter or pressing X" << std::endl;

	if (cmd_input_handler.joinable())
		cmd_input_handler.join();
}

module_manager::~module_manager() = default;


void test_pick_place()
{
#ifdef USE_ROBOT
	state_observation::franka_agent robot{ std::make_shared<remote_controller_wrapper>() };


	obb box = obb(aabb({ 9.064823389e-03f ,9.662224352e-02f ,5.904299021e-02f }, { -4.725716412e-01f,3.087965846e-01f,2.952149510e-02f }));

	obb goal_place = obb(aabb(box.diagonal, Eigen::Vector3f{ -0.27f,0.85f,0.02f }));
		robot.picknplace(box, goal_place);
#endif
}

int main(int argc, char* argv[])
{
	icl_core::logging::initialize(argc, argv);
	// Disable verbose output when loading neural networks
	google::InitGoogleLogging("XXX");
	google::SetCommandLineOption("GLOG_minloglevel", "2");

	//mogaze_predictor(2);
	module_manager manager(argc, argv, camera_type::KINECT_V2);
}



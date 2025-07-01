#include "franka_actor_sim.h"

#include <franka_proxy_share/franka_proxy_util.hpp>
#include <franka_voxel/motion_generator_joint_max_accel.h>

namespace simulation
{
	simulation_controller_wrapper::simulation_controller_wrapper(const simulation::environment::Ptr& environment, bool is_logging)
		: is_logging(is_logging), env(environment)
	{
		link_meshes.reserve(links.size());

		for (const auto& path : links)
		{
			auto mesh = std::make_shared<pcl::PolygonMesh>();
			
			if (pcl::io::loadOBJFile(path.string(), *mesh))
				throw std::runtime_error("Could not load polygon file");

			Eigen::Matrix3f conv;
			conv <<
				1, 0,  0,
				0, 0, -1,
				0, 1,  0;
			
			pcl::PointCloud<pcl::PointXYZ> cloud;
			pcl::fromPCLPointCloud2(mesh->cloud, cloud);
			pcl::transformPointCloud(cloud, cloud, Eigen::Affine3f(conv));
			pcl::toPCLPointCloud2(cloud, mesh->cloud);

			link_meshes.emplace_back(mesh);
		}
	}

	bool simulation_controller_wrapper::do_logging()
	{
		return is_logging;
	}

	std::chrono::high_resolution_clock::time_point simulation_controller_wrapper::start_time()
	{
		return start_time_;
	}

	bool simulation_controller_wrapper::needs_update_loop() const
	{
		return true;
	}

	void simulation_controller_wrapper::move_to(const franka_proxy::robot_config_7dof& target)
	{
		generator = std::make_unique<franka_proxy::Visualize::franka_joint_motion_generator>(
			speed_factor,
			current_config_, target);

		start_time_ = current_time_;
		const auto q_start = generator->q_start();

		while (true)
		{
			std::unique_lock lock(mtx);
			cv.wait(lock);

			const auto dt = std::chrono::duration<double>(current_time_ - start_time_).count();
			const auto intermediate = q_start + generator->calculateDesiredValues(dt).delta_q_d;

			for (int i = 0; i < 7; ++i)
				current_config_[i] = intermediate(i, 0);
			if (dt >= generator->end_time())
			{
				return;
			}
		}
	}

	franka_proxy::robot_config_7dof simulation_controller_wrapper::current_config() const
	{
		return current_config_;
	}

	bool simulation_controller_wrapper::vacuum_gripper_vacuum(std::uint8_t vacuum_strength, std::chrono::milliseconds timeout)
	{
		assert(this->action && "Vaccum was issued without action");

		if (const auto& action = std::dynamic_pointer_cast<state_observation::pick_action>(this->action))
		{
			auto it = env->object_traces.find(*action->inputs.begin());
			if (it == env->object_traces.end())
				return false;
			carried_object = it->second->prototype;
		}
		else if (const auto& action = std::dynamic_pointer_cast<state_observation::reverse_stack_action>(this->action))
		{
			auto it = env->object_traces.find(action->from);
			if (it == env->object_traces.end())
				return false;
			carried_object = it->second->prototype;
		}
		//vacuum_gripper_state_.actual_power_ = vacuum_strength;
		//vacuum_gripper_state_.vacuum_level = vacuum_strength;
		//vacuum_gripper_state_.
		env->update(action);

		return true;
	}

	bool simulation_controller_wrapper::vacuum_gripper_stop()
	{
		if (carried_object)
		{
			env->update(action);
			carried_object = nullptr;
		}
		return true;
	}

	bool simulation_controller_wrapper::vacuum_gripper_drop(std::chrono::milliseconds timeout)
	{
		if (carried_object)
		{
			env->update(action);
			carried_object = nullptr;
		}
		return true;
	}

	void simulation_controller_wrapper::set_speed_factor(double speed_factor)
	{
		this->speed_factor = speed_factor;
	}

	void simulation_controller_wrapper::update()
	{}

	void simulation_controller_wrapper::set_object(const state_observation::object_prototype::ConstPtr& prototype)
	{
		carried_object = prototype;
	}

	void simulation_controller_wrapper::set_action(const state_observation::pn_transition::Ptr& action)
	{
		this->action = action;
	}

	void simulation_controller_wrapper::render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp)
	{
		std::vector<Eigen::Affine3d> poses;
		{
			std::unique_lock lock(mtx);
			poses = franka_proxy::franka_proxy_util::fk(current_config_);
			poses.emplace_back(poses.back() * Eigen::Translation3d(0., 0., 0.105));
		}
		int i = 0;
		for (const auto& mesh : link_meshes)
		{
			const Eigen::Affine3f matrix = poses[i].cast<float>();
			scene.add(std::make_shared<pcl::simulation::TriangleMeshModel>(
				state_observation::pointcloud_preprocessing::color(
					state_observation::pointcloud_preprocessing::transform(mesh, matrix),
					pcl::RGB{ 211, 211, 211 }
				)
			));
			++i;
		}
		if (carried_object)
		{
			Eigen::Affine3f matrix = poses.back().cast<float>() * Eigen::Translation3f(0, 0, 0.107f) *
				Eigen::Translation3f(0.f, 0.f, 0.5f * carried_object->get_bounding_box().diagonal.z()) * Eigen::Scaling(0.5f * carried_object->get_bounding_box().diagonal);

			scene.add(std::make_shared<pcl::simulation::TriangleMeshModel>(
				state_observation::pointcloud_preprocessing::color(
					state_observation::pointcloud_preprocessing::transform(carried_object->load_mesh(), matrix),
					carried_object->get_mean_color()
				)
			));
		}
	}
	void simulation_controller_wrapper::render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport)
	{
		std::vector<Eigen::Affine3d> poses;
		{
			std::unique_lock lock(mtx);
			poses = franka_proxy::franka_proxy_util::fk(current_config_);
			poses.emplace_back(poses.back() * Eigen::Translation3d(0., 0., 0.105));
		}

		int i = 0;
		for (const auto& mesh : link_meshes)
		{
			std::string id = std::to_string(std::hash<simulation_controller_wrapper*>{}(this)) + std::to_string(std::hash<const pcl::PolygonMesh*>{}(&*mesh));
			const Eigen::Affine3f matrix = poses[i].cast<float>();

			if (!viewer.updatePointCloudPose(id, matrix))
			{
				viewer.addPolygonMesh(*mesh, id, viewport);
				viewer.updatePointCloudPose(id, matrix);
			}
			++i;
		}
		std::string id = std::to_string(std::hash<simulation_controller_wrapper*>{}(this)) + "_carried_object";
		if (carried_object)
		{
			Eigen::Affine3f matrix = poses.back().cast<float>() * Eigen::Translation3f(0, 0, 0.107f) * 
				Eigen::Translation3f(0.f, 0.f, 0.5f * carried_object->get_bounding_box().diagonal.z()) * Eigen::Scaling(0.5f * carried_object->get_bounding_box().diagonal);

			if (!viewer.updatePointCloudPose(id, matrix))
			{
				viewer.addPolygonMesh(*carried_object->load_mesh(), id, viewport);
				viewer.updatePointCloudPose(id, matrix);
			}
		}
		else
			viewer.removePolygonMesh(id, viewport);
	}
	void simulation_controller_wrapper::reset(const std::chrono::high_resolution_clock::time_point& tp)
	{
		start_time_ = tp;
	}
	void simulation_controller_wrapper::set_time(const std::chrono::high_resolution_clock::time_point& tp)
	{
		std::unique_lock lock(mtx);
		current_time_ = tp;
		cv.notify_one();
	}
}

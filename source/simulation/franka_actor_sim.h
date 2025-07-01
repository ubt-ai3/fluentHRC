#pragma once

#include <filesystem>

#include <franka_planning/franka_actor.hpp>

#include "scene.hpp"

namespace franka_proxy
{
	namespace Visualize
	{
		class franka_joint_motion_generator;
	}
}

namespace simulation
{
	class SIMULATION_API simulation_controller_wrapper : public state_observation::Controller, public scene_object
	{
	public:

		explicit simulation_controller_wrapper(const simulation::environment::Ptr& environment, bool is_logging = true);
		~simulation_controller_wrapper() override = default;

		bool do_logging() override;
		std::chrono::high_resolution_clock::time_point start_time() override;
		[[nodiscard]] bool needs_update_loop() const override;

		void move_to(const franka_proxy::robot_config_7dof& target) override;
		[[nodiscard]] franka_proxy::robot_config_7dof current_config() const override;
		bool vacuum_gripper_vacuum(std::uint8_t vacuum_strength, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) override;
		bool vacuum_gripper_stop() override;
		bool vacuum_gripper_drop(std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) override;
		void set_speed_factor(double speed_factor) override;
		void update() override;

		void set_object(const state_observation::object_prototype::ConstPtr& prototype);
		void set_action(const state_observation::pn_transition::Ptr& action);

		
		inline static const std::vector<std::filesystem::path> links =
		{
			std::filesystem::path{"assets/models/franka/link0.obj"},
			std::filesystem::path{"assets/models/franka/link1.obj"},
			std::filesystem::path{"assets/models/franka/link2.obj"},
			std::filesystem::path{"assets/models/franka/link3.obj"},
			std::filesystem::path{"assets/models/franka/link4.obj"},
			std::filesystem::path{"assets/models/franka/link5.obj"},
			std::filesystem::path{"assets/models/franka/link6.obj"},
			std::filesystem::path{"assets/models/franka/link7.obj"},
			std::filesystem::path{"assets/models/franka/cobot_pump.obj"},
		};

		void render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp) override;
		void render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport = 0) override;

		void reset(const std::chrono::high_resolution_clock::time_point& tp);
		void set_time(const std::chrono::high_resolution_clock::time_point& tp);

	private:

		franka_proxy::robot_config_7dof current_config_ = {
			0.0108909,
			-0.483135,
			-0.0079431,
			-2.81406,
			-0.0382218,
			2.3383,
			0.877388 };

		int current_gripper_pos_;
		int max_gripper_pos_;
		bool gripper_grasped_;

		//franka_proxy::vacuum_gripper_state vacuum_gripper_state_;

		double speed_factor = 1.;
		std::chrono::high_resolution_clock::time_point start_time_;
		std::chrono::high_resolution_clock::time_point current_time_;
		bool is_logging;

		std::condition_variable cv;
		std::mutex mtx;

		//rendering related
		std::vector<pcl::PolygonMesh::ConstPtr> link_meshes;
		std::chrono::duration<float> start;
		franka_proxy::robot_config_7dof target;
		std::unique_ptr<franka_proxy::Visualize::franka_joint_motion_generator> generator;

		state_observation::object_prototype::ConstPtr carried_object;

		state_observation::pn_transition::Ptr action;
		simulation::environment::Ptr env;
	};
}

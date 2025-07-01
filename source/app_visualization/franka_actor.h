#pragma once

#include <franka_planning/franka_actor.hpp>

namespace state_observation
{
	/**
	 * @typedef visual_action
	 * @brief Tuple type representing a robot action with configuration and speed
	 * 
	 * Contains:
	 * - franka_proxy::robot_config_7dof: The target robot configuration
	 * - double: The speed factor for the movement
	 */
	typedef std::tuple<franka_proxy::robot_config_7dof, double> visual_action;

	/**
	 * @class visual_controller_wrapper
	 * @brief A controller wrapper for recording Franka robot actions
	 * 
	 * This class implements a pseudo-controller that records robot actions
	 * for visualization purposes. It maintains a sequence of actions that can
	 * be used to visualize robot movements.
	 * 
	 * Features:
	 * - Action sequence recording
	 * - Speed control
	 * - Configuration tracking
	 * - Action sequence playback support
	 */
	class visual_controller_wrapper : public state_observation::Controller
	{
	public:

		visual_controller_wrapper() = default;
		~visual_controller_wrapper() override = default;

		void move_to(const franka_proxy::robot_config_7dof& target) override;
		[[nodiscard]] franka_proxy::robot_config_7dof current_config() const override;
		void set_speed_factor(double speed_factor) override;


		void action_reset(const franka_proxy::robot_config_7dof& config);
		[[nodiscard]] const std::list<visual_action>& action_sequence() const;

	private:

		double speed_factor = 1.;
		std::list<visual_action> action_sequence_;
	};
}
#include "franka_actor.h"

namespace state_observation
{
	void visual_controller_wrapper::move_to(const franka_proxy::robot_config_7dof& target)
	{
		action_sequence_.emplace_back(target, speed_factor);
	}

	franka_proxy::robot_config_7dof visual_controller_wrapper::current_config() const
	{
		return std::get<0>(action_sequence_.back());
	}

	void visual_controller_wrapper::set_speed_factor(double speed_factor)
	{
		this->speed_factor = speed_factor;
	}

	void visual_controller_wrapper::action_reset(const franka_proxy::robot_config_7dof& config)
	{
		action_sequence_.clear();
		action_sequence_.emplace_back(config, speed_factor);
	}

	const std::list<visual_action>& visual_controller_wrapper::action_sequence() const
	{
		return action_sequence_;
	}
}
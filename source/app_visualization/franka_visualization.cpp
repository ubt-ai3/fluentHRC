#include "franka_visualization.h"

#include <numbers>

#include "robot.pb.h"

#include <state_observation/pn_model_extension.hpp>

static_assert(generated::Visual_Change::ENABLED == 0);
static_assert(generated::Visual_Change::DISABLED == 1);
static_assert(generated::Visual_Change::REVOKED == 2);


franka_visualizer::franka_visualizer(franka_visualizations visualizations)
	: visualizations_(visualizations),
	controller_(std::make_shared<state_observation::visual_controller_wrapper>()),
	vis_agent_(controller_)
{}

void franka_visualizer::update_robot_action(const std::tuple<state_observation::pn_transition::Ptr, state_observation::pn_transition::Ptr, franka_proxy::robot_config_7dof> payload, enact_priority::operation op)
{
	using namespace franka_proxy::Visualize;

	//early out if no visualization method active
	if (!visualizations_.voxels && !visualizations_.shadow_robot && !visualizations_.tcps)
		return;

	std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();

	const auto& [action, next_action, joints] = payload;

	switch (op)
	{
	case enact_priority::operation::CREATE:
	{
		/*
		 * ignore creation of caused by next_action becoming action
		 * while being part of consecutive execution
		 */
		if (action && action == visual_transitions.second)
			return;

		visual_transitions = { std::get<0>(payload), std::get<1>(payload) };

		//execute and extract actions with pseudo robot controller
		controller_->action_reset(joints);
		if (!action && !next_action)
		{
			vis_agent_.rest();
		}
		else
		{
			vis_agent_.approach(*action);
			vis_agent_.execute_transition(*action);

			if (are_transitions_consecutive(action, next_action))
			{
				vis_agent_.approach(*next_action);
				vis_agent_.execute_transition(*next_action);
			}
		}
		const auto& sequence = controller_->action_sequence();

		std::list<std::shared_ptr<franka_joint_motion_generator>> generators;

		//create a generator for each robot action path
		//there are at least two elements in the sequence (start, next_point, ...)
		//first speed does not have any information for us
		auto it_prev = sequence.begin();
		auto it = it_prev;
		++it;
		for (; it != sequence.end(); ++it)
		{
			auto prev_config = std::get<0>(*it_prev);
			const auto& [config, current_speed] = *it;

			generators.emplace_back(std::make_shared<franka_joint_motion_generator>(
				current_speed,
				prev_config, config));

			it_prev = it;
		}

		//sample all active visualizations
		if (visualizations_.shadow_robot)
		{
			const franka_joint_motion_sampler sampler{ 0.005, 5000, generators };
			const franka_joint_motion_sync joint_vis{ start_time };
			joints_progress_signal(joint_vis.discretize_path(sampler));
		}
		if (visualizations_.tcps)
		{
			const franka_joint_motion_sampler sampler{ 0.005, 500, generators };
			tcps_signal(franka_joint_motion_tcps::discretize_path(sampler));
		}
		if (visualizations_.voxels)
		{
			const franka_joint_motion_sampler sampler{ 0.05, 300, generators };
			const franka_joint_motion_voxelizer voxelizer{ m_voxel_instance };

			voxel_signal(voxelizer.discretize_path(sampler));
		}
		break;
	}
	//retract visualizations for erroneous or done action sequences
	case enact_priority::operation::DELETED:
		if (action && action == visual_transitions.first)
			return;
		[[fallthrough]];
	case enact_priority::operation::MISSING:
	{
		if (visualizations_.shadow_robot)
			joints_progress_signal(Visual_Change::REVOKED);

		if (visualizations_.tcps)
			tcps_signal(Visual_Change::REVOKED);

		if (visualizations_.voxels)
			voxel_signal(Visual_Change::REVOKED);

		break;
	}
	case enact_priority::operation::UPDATE:
	{
		throw std::exception("Cannot handle update in this context!");
		break;
	}
	}
}

franka_visualizations franka_visualizer::visualizations() const
{
	return visualizations_;
}


void franka_visualizer::set_visual_generators(franka_visualizations visualizations)
{
	if (visualizations_.shadow_robot != visualizations.shadow_robot)
		joints_progress_signal(visualizations.shadow_robot ? Visual_Change::ENABLED : DISABLED);

	if (visualizations_.tcps != visualizations.tcps)
		tcps_signal(visualizations.tcps ? Visual_Change::ENABLED : DISABLED);

	if (visualizations_.voxels != visualizations.voxels)
		voxel_signal(visualizations.voxels ? Visual_Change::ENABLED : DISABLED);

	visualizations_ = visualizations;
}

bool franka_visualizer::are_transitions_consecutive(const state_observation::pn_transition::Ptr& t0, const state_observation::pn_transition::Ptr& t1)
{
	if (!t0 || !t1)
		return false;
	{
		auto object0 = get_picked_object(t0);
		if (object0.first)
		{
			auto object1 = get_placed_object(t1);
			if (!object1.first)
				return false;

			return object0.second == object1.second;
		}
	}
	{
		auto object0 = get_placed_object(t0);
		if (object0.first)
		{
			auto object1 = get_picked_object(t1);
			if (!object1.first)
				return false;

			return object0.second == object1.second;
		}
	}
	return false;
}

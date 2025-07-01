#pragma once

#include "framework.hpp"

#include "task.hpp"

#include "state_observation/object_prototype_loader.hpp"

namespace state = state_observation;

namespace state_observation {
	class object_prototype_loader;
}

namespace simulation
{

	/**
	* Tasks used in study for human-robot fluency

	*/
	class SIMULATION_API hoellerich22
	{
	public:
		const static state::obb structure_pose;
		const static Eigen::Quaternionf quarter_rot_z;

		static state_observation::pn_instance create_goal(state::pn_net& net, const std::vector<state::pn_object_instance>& instances);

		static std::vector<state::pn_object_instance> init_resource_pool(const environment::Ptr& net, const state::object_prototype_loader& loader);

		static sim_task::Ptr structure_1(const state_observation::object_parameters& object_params,
			state_observation::object_prototype_loader loader);
		static sim_task::Ptr structure_2(const state_observation::object_parameters& object_params,
			state_observation::object_prototype_loader loader);

	};

}

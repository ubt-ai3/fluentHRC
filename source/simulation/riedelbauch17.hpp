#pragma once

#include "framework.hpp"

#include "task.hpp"
#include "state_observation/object_prototype_loader.hpp"

namespace state_observation {
	class object_prototype_loader;
}

namespace simulation
{

class SIMULATION_API riedelbauch17
{
public:

	static state_observation::pn_instance create_goal(state_observation::pn_net& net,
		std::vector<state_observation::pn_instance>&& instances);

	static state_observation::pn_instance create_goal(state_observation::pn_net& net,
		const std::set<state_observation::pn_instance>& instances);


	static state_observation::pn_instance create_goal(state_observation::pn_net& net,
		std::initializer_list<std::vector<state_observation::pn_object_instance>> instances);

	static sim_task::Ptr pack_and_stack(const state_observation::object_parameters& object_params,
	                                    state_observation::object_prototype_loader loader);
	static sim_task::Ptr pack_and_stack_and_swap(const state_observation::object_parameters& object_params,
	                                             state_observation::object_prototype_loader loader);
	static sim_task::Ptr pack_and_stack_and_swap_small(const state_observation::object_parameters& object_params,
	                                                   state_observation::object_prototype_loader loader);
};

}

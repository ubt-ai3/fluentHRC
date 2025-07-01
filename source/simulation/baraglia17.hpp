#pragma once

#include "framework.hpp"

#include "task.hpp"

#include "state_observation/object_prototype_loader.hpp"

namespace state_observation {
	class object_prototype_loader;
}

namespace simulation
{

/**
* Tasks used in
 Baraglia, J.; Cakmak, M.; Nagai, Y.; Rao, R. P. & Asada, M.

 Efficient human-robot collaboration: when should a robot take initiative? 

 The International Journal of Robotics Research, 



 SAGE Publications Sage UK: London, England, 
2017, 36, 563-579 

*/
class SIMULATION_API baraglia17
{
	static state_observation::pn_instance create_goal(state_observation::pn_net& net,
		std::vector<state_observation::pn_instance>&& instances);

	static state_observation::pn_instance create_goal(state_observation::pn_net& net,
		const std::set<state_observation::pn_instance>& instances);


	static state_observation::pn_instance create_goal(state_observation::pn_net& net,
		std::initializer_list<state_observation::pn_object_instance> instances);

public:

	static sim_task::Ptr practice(const state_observation::object_parameters& object_params,
	                              state_observation::object_prototype_loader loader);
	static sim_task::Ptr task_a_1(const state_observation::object_parameters& object_params,
	                              state_observation::object_prototype_loader loader);
	static sim_task::Ptr task_b_1(const state_observation::object_parameters& object_params,
	                              state_observation::object_prototype_loader loader);
};

}

#pragma once

#include "framework.hpp"

#include "task.hpp"
#include "state_observation/object_prototype_loader.hpp"

namespace state_observation {
class object_prototype_loader;
}

namespace simulation
{

class SIMULATION_API behavior_test
{
public:
	static sim_task::Ptr row_ltr(const state_observation::object_parameters& object_params,
		state_observation::object_prototype_loader loader);
	static sim_task::Ptr row_rtl(const state_observation::object_parameters& object_params,
		state_observation::object_prototype_loader loader);
	static sim_task::Ptr mix_ltr(const state_observation::object_parameters& object_params,
		state_observation::object_prototype_loader loader);
	static sim_task::Ptr mix_rtl(const state_observation::object_parameters& object_params,
		state_observation::object_prototype_loader loader);
};

}

#pragma once
#include "framework.hpp"

#include "task.hpp"

#include <state_observation/building_estimation.hpp>


namespace simulation
{
	using namespace state_observation;

	struct SIMULATION_API building_simulation_test
	{
		sim_task::Ptr task;
		building::Ptr building;
	};

	struct SIMULATION_API sth
	{
		static std::shared_ptr<building_simulation_test> create_building_simulation_task(
			const object_parameters& object_params,
			object_prototype_loader loader);
	};
}
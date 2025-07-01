#pragma once

#include "framework.hpp"
#include "task.hpp"


/*
class SIMULATION_API Test
{
	inline int saySth()
	{
		return 0;
	}
};*/

#include "state_observation/building_estimation.hpp"

/*
namespace state_observation
{
	class object_parameters;
	class object_prototype_loader;
	class building;
}
*/
namespace simulation
{
	struct SIMULATION_API building_simulation_test
	{
		sim_task::Ptr task;
		state_observation::building::Ptr building;
	};

	/*class SIMULATION_API sth
	{
	public:*/
	SIMULATION_API std::shared_ptr<building_simulation_test> create_building_simulation_task(
			const state_observation::object_parameters& object_params,
			state_observation::object_prototype_loader loader);
	//};

	class SIMULATION_API building_benchmark
	{
	public:

		const static state_observation::obb structure_pose;
		const static Eigen::Quaternionf quarter_rot_z;

		static std::vector<state_observation::pn_object_instance> init_resource_pool(const state_observation::pn_net::Ptr& net, const state_observation::object_prototype_loader& loader);

		static std::map<std::string, state_observation::pn_object_token::Ptr> named_tokens(const std::vector<state_observation::pn_object_instance>& resource_pool);
		static std::map<state_observation::object_prototype::ConstPtr, state_observation::pn_object_token::Ptr> get_token_traces(const std::vector<state_observation::pn_object_instance>& resource_pool);

		static state_observation::pn_instance decompose(const state_observation::pn_net::Ptr& net, const std::vector<state_observation::pn_object_instance>& resource_pool);
		static std::map<state_observation::pn_place::Ptr, state_observation::pn_token::Ptr> to_marking(const state_observation::pn_net::Ptr& net, const std::vector<state_observation::pn_object_instance>& resource_pool,
			const state_observation::building& building);

		static sim_task::Ptr building_1(const state_observation::object_parameters& object_params, state_observation::object_prototype_loader loader);
		//static sim_task::Ptr building_2(const state_observation::object_parameters& object_params, state_observation::object_prototype_loader loader);
		//static sim_task::Ptr building_3(const state_observation::object_parameters& object_params, state_observation::object_prototype_loader loader);
		//static sim_task::Ptr building_4(const state_observation::object_parameters& object_params, state_observation::object_prototype_loader loader);

		//static std::shared_ptr<state::building> flag_denmark(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token);
	};
}
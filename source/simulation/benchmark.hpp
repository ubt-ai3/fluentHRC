#pragma once

#include <random>

#include <enact_priority/signaling_actor.hpp>

#include <state_observation/pn_model_extension.hpp>
#include <state_observation/object_prototype_loader.hpp>

#include <simulation/scene.hpp>

namespace state = state_observation;

namespace state_observation {
class building;
};

namespace simulation
{
	class sim_task;
}

class SIMULATION_API benchmark {
public:
	const static state::obb structure_pose;
	const static Eigen::Quaternionf quarter_rot_z;

	static state::pn_net::Ptr init_net(const state::object_parameters& object_params, int count_agents);
	static std::vector<state::pn_object_instance> init_resource_pool(const state::pn_net::Ptr& net, const state::object_prototype_loader& loader);

	static state::pn_binary_marking::Ptr to_marking(const state::pn_net::Ptr& net, const std::vector<state::pn_object_instance>& resource_pool, 
		const std::map<std::string, std::shared_ptr<state::building>>& buildings = {});
	static std::map<std::string, state::pn_object_token::Ptr> named_tokens(const std::vector<state::pn_object_instance>& resource_pool);
	
	static state::pn_instance decompose(const state::pn_net::Ptr& net, const std::vector<state::pn_object_instance>& resource_pool);
	
	static state::pn_instance pyramid(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token);

	static std::shared_ptr<state::building> flag_denmark(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token);
	
	static std::shared_ptr<state::building> building_1(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token);
	static std::shared_ptr<state::building> building_2(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token);
	static std::shared_ptr<state::building> building_3(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token);
	static std::shared_ptr<state::building> building_4(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token);

};

class SIMULATION_API task_manager : public enact_priority::signaling_actor<state::pn_belief_marking::ConstPtr>
{
public:
	task_manager(const state::object_parameters& object_params, const state::object_prototype_loader& loader, int count_agents = 3, 
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());

	task_manager(const simulation::sim_task& task,
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());

	state::pn_belief_marking::ConstPtr get_marking();

	void update_marking(const state::pn_belief_marking::ConstPtr& marking);

	void next();
	void decompose();
	void reset(std::chrono::high_resolution_clock::time_point start_time);

	const state::pn_net::Ptr net;
	const std::vector<state::pn_object_instance> resource_pool;
	const std::map<std::string, state::pn_object_token::Ptr> name_to_token;

	const state::pn_instance decomposition_goal;
	const std::map<std::string, std::shared_ptr<state::building>> buildings;

private:
	std::map<std::string, state::pn_instance> tasks;

public:
	const state::pn_binary_marking::ConstPtr initial_marking; // initial marking must be created after all tasks are created

private:
	std::mutex m;
	std::mt19937 rand;

	state::pn_belief_marking::ConstPtr marking;



	state::pn_instance next_task();
	void erase_goal_tokens();

	std::ofstream file;
	std::chrono::high_resolution_clock::time_point start_time;

	static std::vector<state_observation::pn_object_instance> to_object_instances(const simulation::environment::ObjectTraces& objects);

	void log_places(const std::string& path) const;
};
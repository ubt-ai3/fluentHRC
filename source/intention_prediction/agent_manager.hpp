#pragma once

#include "framework.hpp"
#include "observed_agent.hpp"
#include "enact_core/data.hpp"
#include "enact_priority/priority_actor.hpp"
#include "hand_pose_estimation/hand_tracker_enact.hpp"
#include "state_observation/pn_reasoning.hpp"

namespace prediction
{
/**
 * @class instance_manipulation
 * @brief Bookkeeping information for agent_manager
 *
 * Tracks and manages transition probabilities, timestamps, and summed probabilities
 * for instances in the agent_manager. Used for maintaining state and history of
 * agent transitions and manipulations.
 *
 * Features:
 * - Transition probability tracking
 * - Timestamp management
 * - Summed probability calculation
 * - Agent retrieval from Petri net
 */
class instance_manipulation
{
public:
	double summed_probability;
	std::chrono::duration<float> timestamp;
	std::map<state_observation::pn_transition::Ptr, double> probabilities;

	instance_manipulation(const std::pair< state_observation::pn_transition::Ptr, double>& transition, std::chrono::duration<float> timestamp);

	void add(const std::pair< state_observation::pn_transition::Ptr, double>& transition, std::chrono::duration<float> timestamp);

	static state_observation::pn_place::Ptr get_agent(const state_observation::pn_net& net, const state_observation::pn_transition&);

};

/**
 * @class agent_manager
 * @brief Manages agent creation, updates, and deletion based on observed hands
 *
 * Creates, updates, and deletes agents based on observed hand trajectories and
 * Petri net transitions. Provides functionality to track and manage agent states,
 * transition probabilities, and blocked transitions.
 *
 * Features:
 * - Agent lifecycle management
 * - Transition probability updates
 * - Blocked transition tracking
 * - Agent association with hand trajectories
 * - Task completion signaling
 */
class INTENTIONPREDICTION_API agent_manager
{
public:
	using Ptr = std::shared_ptr<agent_manager>;
	using entity_id = std::shared_ptr<enact_core::entity_id>;
	using hand_trajectory_data = enact_core::lockable_data_typed<::hand_pose_estimation::hand_trajectory>;

	inline static const float certainties_threshold = 0.35f;
	
	const unsigned int max_agents;
	
	agent_manager(enact_core::world_context& world,
	              state_observation::pn_net::Ptr net,
		const state_observation::computed_workspace_parameters& workspace_params,
		unsigned int max_agents = 4,
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());

	agent_manager(enact_core::world_context& world,
		state_observation::pn_net::Ptr net,
		const state_observation::computed_workspace_parameters& workspace_params,
		const std::vector<state_observation::pn_place::Ptr>& agent_places,
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());

	~agent_manager();

	void update(const state_observation::pn_emission::ConstPtr& emission);
	void update(const entity_id& id, enact_priority::operation op);
	void update_agents(std::map<observed_agent::Ptr, std::set<state_observation::pn_transition::Ptr>> executed_transitions, const
	                   state_observation::pn_belief_marking::ConstPtr& prev_marking, const state_observation::pn_belief_marking::ConstPtr&
	                   current_marking, std::chrono::duration<float> timestamp);
	void update(const std::map<state_observation::pn_transition::Ptr, double>& transition_probabilities,
	            const state_observation::pn_belief_marking::ConstPtr& prev_marking,
	            const state_observation::pn_belief_marking::ConstPtr& current_marking,
			    std::chrono::duration<float> time_seconds);

	/*
	 * Used when association of hand trajectory and agent place is known. If no association
	 * is established prior to calling to update, then update searches for a fitting agent place.
	 */
	observed_agent::Ptr add(const entity_id& id, const state_observation::pn_place::Ptr& place);

	/*
	 * Signals that the task is finished, for debugging.
	 */
	void finished(std::chrono::high_resolution_clock::time_point start);
	
	std::set<state_observation::pn_transition::Ptr> get_blocked_transitions(bool use_tracking_data = true);

	std::set<observed_agent::Ptr> get_agents() const;

	void reset(std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now());

private:
	enact_core::world_context& world;
	state_observation::pn_net::Ptr net;
	const state_observation::computed_workspace_parameters& workspace_params;

	mutable std::mutex update_mutex;
	mutable std::mutex agent_mutex;

	std::set<state_observation::pn_place::Ptr> unused_agent_places;
	std::map<entity_id, observed_agent::Ptr> agents;

	std::vector<std::pair<entity_id, enact_priority::operation>> pending_updates;

	state_observation::pn_emission::ConstPtr emission;

	std::map<state_observation::pn_instance, instance_manipulation> consumed_instances;
	std::map<state_observation::pn_instance, instance_manipulation> produced_instances;
	std::map<state_observation::pn_place::Ptr, std::chrono::duration<float>> last_seen;

	std::chrono::high_resolution_clock::time_point start_time;

	void store_transition(const std::pair<state_observation::pn_transition::Ptr, double>& entry, std::chrono::duration<float> time_seconds);
	void process_instances(std::map<state_observation::pn_instance, instance_manipulation>& container, 
		std::map<observed_agent::Ptr, std::set<state_observation::pn_transition::Ptr>>& executed_transitions, 
		std::chrono::duration<float> timestamp);

};

}

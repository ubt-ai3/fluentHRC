#pragma once

#include "enact_priority/signaling_actor.hpp"

#include "state_observation/pn_world_traceability.hpp"
#include <state_observation/pn_reasoning.hpp>
#include <state_observation/classification_handler.hpp>

namespace prediction {
	class agent_manager;
}

/**
 * @class task_progress_visualizer
 * @brief Visualizes and tracks task progress using Petri nets
 * 
 * A threaded actor that monitors and visualizes the progress of tasks
 * using Petri net state transitions and belief markings. Integrates with
 * agent management and classification systems.
 * 
 * Features:
 * - Real-time task progress tracking
 * - Petri net state visualization
 * - Belief marking updates
 * - Agent action monitoring
 * - Transition execution tracking
 * - Goal state monitoring
 * - Logging and debugging support
 * - Thread-safe operation
 */
class task_progress_visualizer : public enact_core::threaded_actor,
                                 public enact_priority::signaling_actor<std::pair<state_observation::pn_belief_marking::ConstPtr, std::map<state_observation::pn_transition::Ptr, double>>>
{
public:
	typedef std::shared_ptr<enact_core::entity_id> strong_id;
	typedef std::weak_ptr<enact_core::entity_id> weak_id;
	typedef enact_core::lockable_data_typed<state_observation::object_instance> object_instance_data;

	task_progress_visualizer(enact_core::world_context& world,
		state_observation::place_classification_handler& tracing,
		state_observation::pn_belief_marking::ConstPtr initial_marking,
		std::shared_ptr<prediction::agent_manager> agent_manage = nullptr,
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());
	
	//task_progress_visualizer(enact_core::world_context& world,
	//	state_observation::place_classification_handler& tracing,
	//	state_observation::pn_belief_marking::ConstPtr initial_marking,
	//	const state_observation::pn_transition_extractor::Ptr& extractor,
	//	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());



	virtual ~task_progress_visualizer();

	bool is_initial_recognition_done() const
	{
		return initial_recognition_done;
	}

	void update(std::chrono::duration<float> timestamp);
	void update(const strong_id& id, enact_priority::operation op);
	void update(const state_observation::pn_transition::Ptr& executed);
	void update_goal(const state_observation::pn_belief_marking::ConstPtr& marking);

	void reset(std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());

protected:
	virtual void update() override;

private:
	void evaluate_net(std::chrono::duration<float> timestamp, bool use_tracking_data = true);

	void print(const state_observation::pn_emission& emission, const std::map<state_observation::pn_place::Ptr, int>& mismatches = {}) const;
	void log(const state_observation::pn_emission& emission) const;
	void log(const state_observation::pn_belief_marking& marking) const;

	enact_core::world_context& world;
	state_observation::place_classification_handler& tracing;

	std::atomic<std::chrono::duration<float>> timestamp;
	state_observation::pn_belief_marking::ConstPtr marking;
	state_observation::pn_transition_extractor::Ptr differ;
	std::shared_ptr<prediction::agent_manager> agent_manage;

	bool initial_recognition_done = false;
	state_observation::pn_belief_marking::ConstPtr marking_update = nullptr;

	std::mutex update_mutex;
	std::vector<std::pair<strong_id, enact_priority::operation>> pending_instance_updates;
	std::vector<state_observation::pn_transition::Ptr> executed_transitions;

	/* for debugging */
	int consecutive_mismatches = 0;
	std::chrono::high_resolution_clock::time_point last_successful_evaluation = std::chrono::high_resolution_clock::now();

	mutable std::ofstream file;
	mutable std::ofstream file_markings;
	std::chrono::high_resolution_clock::time_point start_time;
};

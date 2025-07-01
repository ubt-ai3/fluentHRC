#pragma once

#include "state_observation/pn_model.hpp"

#include "simulation/mogaze.hpp"
#include <enact_core/world.hpp>
#include <app_visualization/viewer.hpp>
#include <intention_prediction/observed_agent.hpp>

/**
 * @class mogaze_predictor
 * @brief Predicts human actions using the MoGaze framework
 * 
 * A threaded actor that predicts human actions based on observed behavior
 * and Petri net state transitions. Uses a combination of task execution
 * tracking and intention prediction.
 * 
 * Features:
 * - Action prediction with lookahead
 * - Task execution tracking
 * - Transition probability mapping
 * - Multi-task prediction support
 * - Logging and visualization
 * - Thread-safe operation
 */
class mogaze_predictor : public enact_core::threaded_actor {
public:
	using transition_probability = std::map<state::pn_transition::Ptr, double>;

	static constexpr int lookahead = 2;

	/**
	 * @struct prediction_context
	 * @brief Context for action prediction
	 * 
	 * Holds the current state and possible actions for prediction.
	 * Includes marking state, action sequence, and prediction weight.
	 */
	struct prediction_context
	{
		state_observation::pn_binary_marking::ConstPtr marking;
		std::vector<state_observation::pn_transition::Ptr> actions;
		double weight;

		prediction_context(state_observation::pn_binary_marking::ConstPtr marking,
			std::vector<state_observation::pn_transition::Ptr> actions = std::vector<state_observation::pn_transition::Ptr>(),
			double weight = 1.)
			:
			marking(std::move(marking)),
			actions(std::move(actions)),
			weight(weight)
		{}
	};

	/**
	 * @struct prediction_entry
	 * @brief Stores prediction results
	 * 
	 * Tracks prediction state including previous actions,
	 * candidate counts, and transition probabilities.
	 */
	struct prediction_entry
	{
		int prev_action_index = -1;
		int count_candidates = 0;
		transition_probability probabilities;
	};

	mogaze_predictor(int person, bool single_predictor = false);

	mogaze_predictor() = delete;
	mogaze_predictor(const mogaze_predictor&) = delete;
	mogaze_predictor(mogaze_predictor&&) = delete;
	mogaze_predictor& operator=(const mogaze_predictor&) = delete;

	virtual ~mogaze_predictor();

	static bool reverses(const state_observation::pn_transition& t1, const state_observation::pn_transition& t2);

protected:
	void update() override;
	void init_logging();
	void log_prediction(const prediction::observed_agent& agent, const prediction_entry& prediction);

	// locks m
	prediction::observed_agent::Ptr get_predictor(int task_id, const state_observation::pn_binary_marking::ConstPtr& marking);



	std::mutex m;

	state_observation::computed_workspace_parameters workspace_params;
	enact_core::world_context world;
	state_observation::pointcloud_preprocessing pc_prepro;
	std::unique_ptr< pcl::visualization::PCLVisualizer> pcl_viewer;

	// ensure m is locked before accessing execution
	simulation::mogaze::task_execution execution;

	// one predictor for a certain subset of tasks
	std::vector<state_observation::pn_place::Ptr> goals;
	prediction::observed_agent::Ptr agent;
	prediction_entry prediction;

	state_observation::pn_transition::Ptr prev_action = nullptr;
	std::vector<std::string> box_names;

	std::ofstream file;
};
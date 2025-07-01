#pragma once

#include <enact_core/actor.hpp>

#include "state_observation/pn_model.hpp"

namespace enact_core {
class world_context;
}

namespace prediction {
class agent_manager;
class observed_agent;
}

namespace state = state_observation;
class viewer;

/**
 * @class intention_visualizer
 * @brief Visualizes predicted agent intentions and actions
 * 
 * A threaded actor that visualizes the predicted intentions and actions
 * of observed agents in the workspace. Integrates with the viewer system
 * to display predictions and agent states.
 * 
 * Features:
 * - Agent intention visualization
 * - Action prediction display
 * - Color-coded agent tracking
 * - Belief marking updates
 * - Prediction logging
 * - Thread-safe operation
 */
class intention_visualizer : public enact_core::threaded_actor
{
public:
	using transition_probability = std::map<state::pn_transition::Ptr, double>;

	static constexpr int lookahead = 2;

	/**
	 * @struct prediction_context
	 * @brief Context for intention prediction
	 * 
	 * Holds the current state and possible actions for predicting
	 * agent intentions. Includes belief marking, action sequence,
	 * and prediction weight.
	 */
	struct prediction_context
	{
		state::pn_belief_marking::ConstPtr marking;
		std::vector<state::pn_transition::Ptr> actions;
		double weight;

		prediction_context(state::pn_belief_marking::ConstPtr marking,
			std::vector<state::pn_transition::Ptr> actions = std::vector<state::pn_transition::Ptr>(),
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

	intention_visualizer(enact_core::world_context& world,
		const state::computed_workspace_parameters& workspace_params,
		prediction::agent_manager& agents,
		viewer& view,
		state::pn_belief_marking::ConstPtr initial_marking = nullptr);

	intention_visualizer() = delete;
	intention_visualizer(const intention_visualizer&) = delete;
	intention_visualizer(intention_visualizer&&) = delete;
	intention_visualizer& operator=(const intention_visualizer&) = delete;

	virtual ~intention_visualizer();

	cv::Vec3b get_color(const std::shared_ptr<enact_core::entity_id>& id) const;

	void update(const state::pn_belief_marking::ConstPtr& marking);

	void reset();

protected:
	void update() override;

private:
	enact_core::world_context& world;
	const state_observation::computed_workspace_parameters& workspace_params;
	prediction::agent_manager& agents;
	viewer& view;

	std::vector<std::string> box_names;
	state::pn_belief_marking::ConstPtr marking = nullptr;

	std::map<std::shared_ptr<prediction::observed_agent>, prediction_entry> predictions;
	std::ofstream file;

	std::chrono::duration<float> start;

	void log_prediction(const prediction::observed_agent& agent, const prediction_entry& predict);
};

#pragma once

//#include <boost/bimap.hpp>

#include <enact_core/actor.hpp>

#include <state_observation/pn_model_extension.hpp>

#include "franka_actor.hpp"

namespace state_observation
{
class building;
};

namespace prediction
{
class agent_manager;
}

namespace robot
{

namespace state = state_observation;

/*
* Plans the action sequence for the robot
* Used to change robot behaviour during run-time
*/
class action_planner
{
public:
	const state::pn_place::Ptr robot_place;

	action_planner(const state::pn_place::Ptr& robot_place);

	[[nodiscard]] virtual std::string get_class_name() const;

	virtual void update_marking(const state::franka_agent& franka, state::pn_belief_marking::ConstPtr marking) noexcept;

	[[nodiscard]] bool gripper_collides(const state::pn_boxed_place::Ptr& target_place) const noexcept;
	[[nodiscard]] state::pn_boxed_place::Ptr get_target_place(const state::pn_transition::Ptr& transition) const noexcept;
	
	virtual std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> next(const state::franka_agent& franka, const state::pn_transition::Ptr& excluded = nullptr) noexcept;
	virtual void compute_forward_transitions(const state::franka_agent& franka);

protected:
	state::pn_belief_marking::ConstPtr marking;
	state::pn_net::Ptr net;
	std::set<state::pn_transition::Ptr> forward_transitions;
	std::vector<state::pn_boxed_place::Ptr> occupied_places;

	[[nodiscard]] std::vector<state::pn_transition::Ptr> get_feasible_actions(const state::pn_transition::Ptr& excluded = nullptr) const noexcept;

	[[nodiscard]] std::vector<state::pn_transition::Ptr> get_future_feasible_actions(const std::vector<state::pn_transition::Ptr>& next_actions) const noexcept;

	[[nodiscard]] std::vector<state::pn_transition::Ptr> get_future_feasible_actions(const std::vector<state::pn_belief_marking::ConstPtr>& next_markings) const noexcept;

	/*
	* feasible_actions must not be empty
	*/
	virtual std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> select_action(const std::vector<state::pn_transition::Ptr>& feasible_actions, const state::franka_agent& franka) noexcept;

};

/*
* Robot does nothing 
*/
class null_planner : public action_planner
{
public:
	null_planner(const state::pn_place::Ptr& robot_place);

	[[nodiscard]] std::string get_class_name() const override;

protected:
	std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> next(const state::franka_agent& franka, const state::pn_transition::Ptr& excluded = nullptr) noexcept override;

};

/*
* Builds (and dismantles) target building from right to left (from user perspective), left to right (from its perspective)
*/
class planner_layerwise_rtl : public action_planner
{
public:
	planner_layerwise_rtl(state::pn_place::Ptr robot_place);

	[[nodiscard]] std::string get_class_name() const override;

	virtual ~planner_layerwise_rtl() noexcept = default;

	void compute_forward_transitions(const state::franka_agent& franka) override;

protected:
	std::vector<std::pair<state::pn_transition::Ptr, Eigen::Vector3f>> place_order;

	std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> select_action(const std::vector<state::pn_transition::Ptr>& feasible_actions, const state::franka_agent& franka) noexcept override;
};

/*
* Executes the action (pick + place) the user will least likely execute next
*/
class planner_adaptive : public planner_layerwise_rtl
{
public:
	using transition_probability = std::map<state::pn_transition::Ptr, double>;
	using transition_map = std::map < state::pn_transition::Ptr, state::pn_transition::Ptr>;

	static constexpr int lookahead = 2;

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

	struct prediction_entry
	{
		int prev_action_index = -1;
		int count_candidates = 0;
		transition_probability probabilities;
	};

	planner_adaptive(prediction::agent_manager& agents, state::pn_place::Ptr robot_place);

	~planner_adaptive() noexcept override = default;

	[[nodiscard]] std::string get_class_name() const override;

	void update_marking(const state::franka_agent& franka, state::pn_belief_marking::ConstPtr marking) noexcept override;


protected:
	prediction::agent_manager& agents;
	// human to robot
	std::map<state::pn_place::Ptr, transition_map> action_dictionaries_h2r;
	// robot to human
	std::map<state::pn_place::Ptr, transition_map> action_dictionaries_r2h;

	std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> select_action(const std::vector<state::pn_transition::Ptr>&feasible_actions, const state::franka_agent& franka) noexcept override;

	// actions must not be empty
	[[nodiscard]] bool prioritize_place_target(const std::vector<state::pn_transition::Ptr>& actions) const noexcept;

	// predicts the joint probabilities for the next action and the one after the next
	[[nodiscard]] std::vector<transition_probability> predict_consecutive(const std::vector<state::pn_transition::Ptr>& feasible_actions, const state::franka_agent& franka) const;
	// predicts the probability for the action after the next and conditions the next one on those
	[[nodiscard]] std::map<state::pn_transition::Ptr, std::pair<double, planner_adaptive::transition_probability>> predict_pick_based_on_place(const std::vector<state::pn_transition::Ptr>& feasible_actions, const state::franka_agent& franka) const;

};

/*
* Executes the action (pick + place) the user will most likely execute next
*/
class planner_adversarial : public planner_adaptive
{
public:

	planner_adversarial(prediction::agent_manager& agents, state::pn_place::Ptr robot_place);

	~planner_adversarial() noexcept override = default;

	[[nodiscard]] std::string get_class_name() const override;

protected:
	std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> select_action(const std::vector<state::pn_transition::Ptr>& feasible_actions, const state::franka_agent& franka) noexcept override;
};

typedef std::tuple<state::pn_transition::Ptr, state::pn_transition::Ptr, franka_proxy::robot_config_7dof> agent_signal;
/*
* Processes input, plans, and executes commands on the robot
*/
class agent :
	public enact_priority::signaling_actor<agent_signal>,
	public enact_core::threaded_actor
{
public:

	/*
	 * creates ip based agent
	 */
	/*agent(state::pn_net::Ptr net,
		std::unique_ptr<action_planner>&& behaviour,
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now(),
		std::string_view ip_addr = "132.180.194.120");*/

	/*
	 * creates simulation based agent
	 */
	/*agent(state::pn_net::Ptr net,
		std::unique_ptr<action_planner>&& behaviour,
		bool do_logging,
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());*/

	agent(state::pn_net::Ptr net,
		std::unique_ptr<action_planner>&& behaviour,
		std::shared_ptr<state_observation::Controller> controller);

	~agent() noexcept override;

	void update_marking(state::pn_belief_marking::ConstPtr marking) noexcept;
	void update_goal(state::pn_belief_marking::ConstPtr marking) noexcept;
	void update_behaviour(std::unique_ptr<action_planner>&& behaviour) noexcept;

	void reset(std::chrono::high_resolution_clock::time_point start);

	//decltype(state::franka_agent::joint_signal)& get_joint_signal();

protected:
	mutable std::mutex update_mutex;

	state::franka_agent franka;
	state::pn_net::Ptr net;
	state::pn_place::Ptr robot_place;
	state::pn_belief_marking::ConstPtr marking;
	bool goal_update;

	std::unique_ptr<action_planner> behaviour;

	state::pn_transition::Ptr next_action = nullptr;
	state::pn_transition::Ptr failed_action = nullptr;
	state::pn_transition::Ptr last_action = nullptr;

	void update() override;

	state::pn_transition::Ptr get_put_back_action() const noexcept;

};
}
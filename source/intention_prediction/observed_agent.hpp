#pragma once

#include <chrono>
#include <memory>
#include <filesystem>

#include "framework.hpp"
#pragma warning( push )
#pragma warning( disable : 4996 )
#include <caffe/solver.hpp>
#pragma warning( pop )
#include "hand_pose_estimation/hand_model.hpp"
#include "hand_pose_estimation/hand_tracker_enact.hpp"
#include "state_observation/pn_model.hpp"
#include "enact_priority/priority_actor.hpp"
#include "state_observation/pn_model_extension.hpp"

namespace prediction
{

class agent_manager;

/**
 * @struct transition_context
 * @brief Represents the context of a Petri net transition
 *
 * Stores information about a transition's context including workspace parameters,
 * transition details, neighbors, center coordinates, and various similarity metrics.
 * Used for comparing and predicting transitions in the agent system.
 *
 * Features:
 * - Transition state tracking
 * - Neighbor comparison
 * - Feature vector generation
 * - Color and volume similarity metrics
 * - Action type classification
 */
struct INTENTIONPREDICTION_API transition_context
{
	inline static const float neighbor_distance_threshold = 0.05f;
	inline static const float angle_identity = 0.98f;
	inline static const float color_similarity = 20.f / 255.f;
	inline static const float volume_similarity = 0.1f;

	inline static const size_t feature_vector_size = 33;

	static float bell_curve(float x, float stdev);

	static std::vector<float> get_neighbors(const state_observation::pn_belief_marking& marking,
		const state_observation::pn_boxed_place::Ptr& place,
		const std::set<state_observation::pn_place::Ptr>& excluded_places = std::set<state_observation::pn_place::Ptr>());

	static float compare_neighbors(const std::vector<float>&, const std::vector<float>&);

	const state_observation::computed_workspace_parameters& workspace_params;
	state_observation::pn_transition::Ptr transition;
	std::vector<float> neighbors;
	Eigen::Vector3f center;
	Eigen::Vector2f center_xy;
	state_observation::obb box;

	/*
	 *  \in [0,1]
	 */
	float color;
	float volume;
	/*
	 * 1 = pick
	 * -1 = place
	 * 0 = hand is side condition or not involved
	 */
	float action_type;

	std::chrono::duration<float> timestamp;

	transition_context(const transition_context& other);
	transition_context(transition_context&& other);

	transition_context(const state_observation::computed_workspace_parameters& workspace_params, 
		const state_observation::pn_transition::Ptr& transition,
		const state_observation::pn_belief_marking& marking,
		const state_observation::pn_place::Ptr& hand = nullptr,
		std::chrono::duration<float> timestamp = std::chrono::duration<float>(std::numeric_limits<float>::quiet_NaN()));

	transition_context& operator=(const transition_context& other);

	[[nodiscard]] float compare(const transition_context& other) const;

	[[nodiscard]] std::vector<float> to_feature_vector() const;
};

/**
 * @class net_predictor
 * @brief Neural network-based predictor for transition probabilities
 *
 * Implements a priority actor that uses a neural network to predict transition
 * probabilities based on historical data and current context. Manages training
 * and prediction using Caffe-based neural networks.
 *
 * Features:
 * - Neural network-based prediction
 * - Training data management
 * - Feature vector processing
 * - Transition probability calculation
 * - Discriminative feature enforcement
 */
class INTENTIONPREDICTION_API net_predictor : public enact_priority::priority_actor
{
public:
	using transition_probability = std::map<::state_observation::pn_transition::Ptr, double>;

	inline static const float epsilon = 0.000001f;
	inline static const unsigned int time_horizon = 4;

	inline static const std::string solver_path = "assets/prediction/action_prediction_solver.prototxt";
	inline static const std::string net_path = "assets/prediction/action_prediction_model.prototxt";

	const state_observation::computed_workspace_parameters& workspace_params;
	
	net_predictor(const state_observation::computed_workspace_parameters& workspace_params,
		const std::string& path,
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());

	virtual ~net_predictor() override;

	void add_data(const transition_context& executed, const std::vector<transition_context>& non_executed);
	void replace_last(const transition_context& executed);

	void train();
	void finished();

	transition_probability predict(const std::vector<transition_context>& candidates);
	transition_probability predict(const std::vector<transition_context>& history, const std::vector<transition_context>& candidates);
	std::vector<float> predict(const float* data_history, const float* data_candidates, size_t count_candidates);

private:
	// initialize the nets one after the other since we need to change the working directory
	static std::mutex net_read_mutex;


	std::mutex testnet_mutex;
	std::mutex local_data_mutex;

	unsigned int epoch;
	static constexpr unsigned int max_epoch = 3;
	
	std::vector<float> data_history;
	std::vector<float> indices_history;
	std::vector<float> data_candidates;
	std::vector<float> indices_candidates;
	std::vector<float> labels;
	// sum of label * data_history for all training samples
	// entry is zero iff that feature value is identical for a positive and all its corresponding negative candidates (must hold for all positive samples)
	std::vector<float> discriminative_features;

	std::unique_ptr<caffe::Solver<float> > solver;
	std::unique_ptr<caffe::Net<float> > testnet;

	std::ofstream file;
	const std::string path;
	std::chrono::high_resolution_clock::time_point start_time;


	caffe::NetParameter readParams(const std::string& param_file, caffe::Phase phase,
		const int level = 0, const std::vector<std::string>* stages = (std::vector<std::string>*)0);
	void init_weights();
	void log_shape(const caffe::Net<float>& net);
	void print_weights();
	void enforce_discriminative_features();
};

/**
 * @class observed_agent
 * @brief Manages and predicts agent behavior based on observed hand movements
 *
 * Tracks and predicts agent behavior by monitoring hand movements and their
 * interactions with objects in the workspace. Uses Petri nets and neural networks
 * to predict likely transitions and actions.
 *
 * Features:
 * - Hand trajectory tracking
 * - Action prediction
 * - Transition probability calculation
 * - Object interaction detection
 * - Behavior-based prediction rules
 * - Training data management
 */
class INTENTIONPREDICTION_API observed_agent
{
public:
	using Ptr = std::shared_ptr<observed_agent>;
	using entity_id = std::shared_ptr<enact_core::entity_id>;
	using hand_trajectory_data = enact_core::lockable_data_typed<::hand_pose_estimation::hand_trajectory>;
	using transition_probability = std::map<::state_observation::pn_transition::Ptr, double>;

	inline static const float max_hand_distance = 0.06f;
	inline static const std::chrono::duration<float> proximity_forget_duration = std::chrono::duration<float>(5.f);
	inline static const double min_probability = 0.01f;
	inline static const float enable_threshold = 0.1f;

	static void normalize(transition_probability& distribution);



	observed_agent(enact_core::world_context& world,
		const state_observation::pn_net::Ptr& net,
		const state_observation::computed_workspace_parameters& workspace_params,
		entity_id tracked_hand,
		state_observation::pn_place::Ptr model_hand,
		std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now());

#ifdef DEBUG_PN_ID
	~observed_agent();
#endif

	observed_agent(const observed_agent&) = delete;

	float certainty;

	const state_observation::pn_net::Ptr& net;
	const entity_id tracked_hand;
	const state_observation::pn_place::Ptr model_hand;


	void update(const state_observation::pn_belief_marking::ConstPtr& marking);
	void update_action_candidates(const std::set<state_observation::pn_place::Ptr>& occluded_places);

	/*
	 * Returns all actions this agent may have executed within @param{proximity_forget_duration}
	 * Updates the internal data structure
	 */
	std::set<state_observation::pn_transition::Ptr> get_action_candidates();

	/**
	 * Returns all actions this agent can execute given the marking
	 */
	std::vector<transition_context> get_executable_actions(const state_observation::pn_belief_marking& marking);

	[[nodiscard]] const std::vector<transition_context>& get_executed_actions() const;

	[[nodiscard]] bool has_object_grabbed() const noexcept;

	/*
	* Retruns true if the action contributes towards the goal,
	* i.e. pick from a non-goal instance and placing at a goal instance
	*/
	[[nodiscard]] bool is_forward_transition(const state_observation::pn_transition::Ptr& action) const;

	[[nodiscard]] float get_distance(const ::hand_pose_estimation::hand_pose_18DoF& pose,
	                                 const state_observation::pn_boxed_place& place) const;

	void add_transition(const state_observation::pn_transition::Ptr&,
		const state_observation::pn_belief_marking& prev_marking,
		std::chrono::duration<float> timestamp);
	void add_transition(const transition_context& executed,
		const std::vector<transition_context>& non_executed);
	void add_transitions(std::set<state_observation::pn_transition::Ptr> transitions,
		const state_observation::pn_belief_marking& prev_marking,
		const state_observation::pn_belief_marking& current_marking,
		std::chrono::duration<float> timestamp);
	
	void start_training();
	void finished(std::chrono::high_resolution_clock::time_point start);

	/*
 * Returns the token that must be in hand_model before @param{transition} is fired
 */
	[[nodiscard]] state_observation::pn_token::Ptr get_precondition(const state_observation::pn_transition& transition) const noexcept;
	/*
	 * Returns the token that must be in hand_model after @param{transition} is fired
	 */
	[[nodiscard]] state_observation::pn_token::Ptr get_postcondition(const state_observation::pn_transition& transition) const noexcept;

	[[nodiscard]] transition_probability predict(const std::vector<transition_context>& candidates,
	                                             const std::vector< transition_context>& future_transitions = std::vector< transition_context>()) const;
	[[nodiscard]] transition_probability predict(const std::vector<transition_context>& candidates,
	                                             const std::vector<state_observation::pn_transition::Ptr>& future_transitions,
	                                             const state_observation::pn_belief_marking& marking) const;

	[[nodiscard]] const transition_context& get_top_left(const std::vector<transition_context>& contexts) const;

	void remove_double_detections(std::set<state_observation::pn_transition::Ptr>& transitions) const;

private:
	enact_core::world_context& world;
public:
	const state_observation::computed_workspace_parameters& workspace_params;
private:

	std::map<state_observation::pn_place::Ptr, std::chrono::duration<float>> proxy_places_and_timestamp;

	std::unique_ptr<std::pair<transition_context, std::vector<transition_context>>> pending_action = nullptr;
	std::vector<transition_context> executed_actions;
	// one per task
	std::map< state_observation::pn_place::Ptr, std::unique_ptr<net_predictor>> predictors;

	std::chrono::duration<float> latest_update = std::chrono::duration<float>(0.f);
	Eigen::Vector3f front;

	state_observation::pn_place::Ptr goal = nullptr;
	std::set<state_observation::pn_transition::Ptr> forward_transitions;

	std::string path;
	std::chrono::high_resolution_clock::time_point start_time;

	[[nodiscard]] transition_probability get_similars(const state_observation::pn_transition::Ptr& transition,
	                                                  const state_observation::pn_belief_marking& marking) const;

	[[nodiscard]] transition_probability apply_general_prediction_rules(const std::vector<transition_context>& contexts) const;

	[[nodiscard]] transition_probability apply_general_prediction_rules(const std::vector<transition_context>& contexts,
	                                                                    const transition_context& prev_action) const;

	[[nodiscard]] observed_agent::transition_probability apply_behavior_based_prediction_rules(const std::vector<transition_context>& candidates,
		const std::vector<transition_context>& time_horizon_history) const;

	net_predictor& get_predictor();
};

}
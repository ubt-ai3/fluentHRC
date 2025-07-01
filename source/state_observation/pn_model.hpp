#pragma once

#ifndef STATE_OBSERVATION__PN_MODEL_HPP
#define STATE_OBSERVATION__PN_MODEL_HPP

#include "framework.hpp"
#include "workspace_objects.hpp"

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

namespace state_observation
{

#define DEBUG_PN_ID

class pn_transition;
class pn_place;
class pn_net;
class pn_token;
class pn_marking;
class pn_binary_marking;
class pn_belief_marking;
class pn_emission;
typedef std::pair<std::shared_ptr<pn_place>, std::shared_ptr<pn_token>> pn_instance;



/**
 * @class pn_place
 * @brief Represents a place in a Petri net model
 * 
 * A place is a fundamental element in a Petri net that can hold tokens.
 * It maintains connections to incoming and outgoing transitions through arcs.
 * Places are created through the pn_net class to ensure proper network construction.
 * 
 * Features:
 * - Tracks incoming and outgoing transitions
 * - Supports token distribution
 * - Maintains network connectivity
 */
class STATEOBSERVATION_API pn_place
{
public:
	friend class pn_net;
	
	typedef std::shared_ptr<pn_place> Ptr;

#ifdef DEBUG_PN_ID
	static int id_counter;
	int id;

	pn_place();
#else

	pn_place() = default;
#endif
	pn_place(pn_place&&) = default;
	pn_place(const pn_place&) = default;
	pn_place& operator=(const pn_place&) = default;
	pn_place& operator=(pn_place&&) = default;

	const std::vector<std::weak_ptr<pn_transition>>& get_incoming_transitions() const;
	const std::vector<std::weak_ptr<pn_transition>>& get_outgoing_transitions() const;
	
	virtual ~pn_place() = default;

private:
	std::vector<std::weak_ptr<pn_transition>> incoming_transitions;
	std::vector<std::weak_ptr<pn_transition>> outgoing_transitions;

};

	
/**
 * @class pn_token
 * @brief Represents a token in a Petri net model
 * 
 * Tokens are the dynamic elements that move through the Petri net.
 * Multiple instances of a token can exist across different places,
 * representing the state of the system.
 * 
 * Features:
 * - Unique identification
 * - Distribution tracking
 * - State representation
 */
class STATEOBSERVATION_API pn_token
{
public:
	typedef std::shared_ptr<pn_token> Ptr;

#ifdef DEBUG_PN_ID
	static int id_counter;
	int id;
	pn_token();
#else

	pn_token() = default;
#endif
	pn_token(pn_token&&) = default;
	pn_token(const pn_token&) = default;
	pn_token& operator=(const pn_token&) = default;
	pn_token& operator=(pn_token&&) = default;
	
	virtual ~pn_token() = default;
	
};

	
/**
 * @class pn_transition
 * @brief Represents a transition in a Petri net model
 * 
 * Transitions are the active elements that move tokens between places.
 * They have input and output arcs connecting to places, and can have
 * side conditions that must be satisfied for firing.
 * 
 * Features:
 * - Input and output arc management
 * - Side condition handling
 * - Transition firing logic
 * - Reversibility checking
 */
class STATEOBSERVATION_API pn_transition
{
public:
	typedef std::shared_ptr<pn_transition> Ptr;
	typedef pn_instance pn_arc;

#ifdef DEBUG_PN_ID
	static int id_counter;
	int id;
#endif

	const std::set<pn_arc> inputs;
	const std::set<pn_arc> outputs;

	
	pn_transition(std::vector<pn_arc>&& inputs,
		std::vector<pn_arc>&& outputs);
	virtual ~pn_transition() = default;

	pn_transition(pn_transition&&) = default;
	pn_transition(const pn_transition&) = default;
	pn_transition& operator=(const pn_transition&) = default;
	pn_transition& operator=(pn_transition&&) = default;


	[[nodiscard]] std::vector<pn_place::Ptr> get_inputs(
		const std::set<pn_token::Ptr>& filter = {}) const;
	
	[[nodiscard]] std::vector<pn_place::Ptr> get_outputs(
		const std::set<pn_token::Ptr>& filter = {}) const;

	[[nodiscard]] std::set<pn_transition::pn_arc> get_pure_input_arcs() const;

	[[nodiscard]] std::set<pn_arc> get_pure_output_arcs() const;

	[[nodiscard]] bool has_input_arc(const pn_arc& arc) const;
	[[nodiscard]] bool has_output_arc(const pn_arc& arc) const;

	[[nodiscard]] bool is_side_condition(const pn_arc& arc) const;

	/**
	* Returns true if starting from a marking m, firing this and afterwards other ends up in m.
	*/
	[[nodiscard]] bool reverses(const pn_transition& other) const;
	
	/*
	 * A side condition is an arc that appears in input and output arcs
	 */
	[[nodiscard]] const std::set<pn_transition::pn_arc> get_side_conditions() const;

	virtual std::string to_string() const;

private:

	std::set<pn_arc> side_conditions;

	void compute_side_conditions();
};

	
/**
 * @class pn_marking
 * @brief Represents the state of a Petri net through token distribution
 * 
 * A marking represents the current state of the Petri net by tracking
 * the distribution of tokens across places. It supports probabilistic
 * distributions and provides methods for state evolution through
 * transition firing.
 * 
 * Features:
 * - Probabilistic token distribution
 * - Transition enabling checking
 * - State evolution through firing
 * - Distribution querying
 */
class STATEOBSERVATION_API pn_marking
{
public:
	typedef std::shared_ptr<pn_marking> Ptr;
	typedef std::shared_ptr<const pn_marking> ConstPtr;

	std::weak_ptr<pn_net> net;
	std::map<pn_instance, double> distribution;

	static constexpr double EPSILON = 0.0001;

	pn_marking(std::weak_ptr<pn_net> net,
		std::map<pn_instance, double>&& distribution);

	pn_marking(pn_marking&&) noexcept = default;
	pn_marking(const pn_marking&) = default;
	pn_marking& operator=(const pn_marking&) = default;
	pn_marking& operator=(pn_marking&&) noexcept = default;

	[[nodiscard]] double get_probability(const pn_place::Ptr& place, const pn_token::Ptr& token) const;
	[[nodiscard]] double get_probability(const pn_token::Ptr& token, const pn_place::Ptr& place) const;
	[[nodiscard]] double get_probability(const pn_instance& instance) const;

	[[nodiscard]] std::map<pn_token::Ptr, double> get_distribution(const pn_place::Ptr& place) const;
	[[nodiscard]] std::map<pn_place::Ptr, double> get_distribution(const pn_token::Ptr& token) const;

	[[nodiscard]] double get_summed_probability(const pn_place::Ptr& place) const;
	[[nodiscard]] double get_summed_probability(const pn_token::Ptr& token) const;

	/**
	 * @param{threshold} only if the certainty of a token in some place exceeds this value, the token is considered to be present
	 * @returns this transition can fire in @param{marking}
	 **/
	[[nodiscard]] double is_enabled(const pn_transition::Ptr& transition) const;

	[[nodiscard]] double would_enable(
		const pn_transition::Ptr& transition, 
		const std::map<pn_instance, double>& additional_distribution) const;

	[[nodiscard]] Ptr fire(const pn_transition::Ptr& transition, double threshold = std::numeric_limits<double>::infinity()) const;

	/*
	 * removes zero entries from the maps
	 */
	void compact();

};



/**
 * @class pn_binary_marking
 * @brief Represents a binary state marking in a Petri net
 * 
 * A binary marking represents the state of a Petri net where tokens are either
 * present or absent in places, without probabilistic distributions.
 * 
 * Features:
 * - Binary token presence tracking
 * - Transition enabling checking
 * - State evolution through firing
 * - Distribution querying
 * - Hash-based comparison
 */
class STATEOBSERVATION_API pn_binary_marking
{
public:
	typedef std::shared_ptr<pn_binary_marking> Ptr;
	typedef std::shared_ptr<const pn_binary_marking> ConstPtr;

	static const std::function<size_t(const pn_binary_marking::Ptr&)> hasher;
	static const std::function<bool(const pn_binary_marking::Ptr&, const pn_binary_marking::Ptr&)> eq;

	std::weak_ptr<pn_net> net;
	std::set<pn_instance> distribution;

	pn_binary_marking(std::weak_ptr<pn_net> net,
		std::set<pn_instance> distribution);

	pn_binary_marking(pn_binary_marking&&) = default;
	pn_binary_marking(const pn_binary_marking&) = default;
	pn_binary_marking& operator=(const pn_binary_marking&) = default;
	pn_binary_marking& operator=(pn_binary_marking&&) = default;

	bool operator==(const pn_binary_marking&) const;

	bool has(const pn_place::Ptr& place, const pn_token::Ptr& token) const;
	bool has(const pn_token::Ptr& token, const pn_place::Ptr& place) const;
	bool has(const pn_instance& instance) const;

	std::set<pn_token::Ptr> get_distribution(const pn_place::Ptr& place) const;
	std::set<pn_place::Ptr> get_distribution(const pn_token::Ptr& token) const;
	bool is_occupied(const pn_place::Ptr& place) const;
	bool is_output_occupied(const pn_transition::Ptr& transition) const;

	bool is_enabled(const pn_transition::Ptr& transition) const;

	bool would_enable(
		const pn_transition::Ptr& transition,
		const std::set<pn_instance>& additional_distribution) const;

	Ptr fire(const pn_transition::Ptr& transition) const;

	size_t hash() const;

private:
	size_t cached_hash;


};

struct hash_pn_binary_marking_ptr {
	size_t operator()(const pn_binary_marking::Ptr& marking) const
	{
		return marking->hash();
	}
};

struct eq_pn_binary_marking_ptr {
	bool operator()(const pn_binary_marking::Ptr& lhs, const pn_binary_marking::Ptr& rhs) const
	{
		return lhs->distribution == rhs->distribution;
	}
};


/**
 * @class pn_belief_marking
 * @brief Represents a belief state marking in a Petri net
 * 
 * A belief marking extends the probabilistic marking concept to handle
 * uncertain states and observations in the Petri net.
 * 
 * Features:
 * - Probabilistic state representation
 * - Belief state management
 * - Transition enabling with uncertainty
 * - State evolution through firing
 * - Conversion to standard marking
 */
class STATEOBSERVATION_API pn_belief_marking
{
public:
	typedef std::shared_ptr<pn_belief_marking> Ptr;
	typedef std::shared_ptr<const pn_belief_marking> ConstPtr;

	// compile errors for copy assignment of hash function when using pn_binary_marking::ConstPtr
	typedef std::unordered_map<pn_binary_marking::Ptr, double, hash_pn_binary_marking_ptr, eq_pn_binary_marking_ptr> marking_dist_t;

	std::weak_ptr<pn_net> net;
	marking_dist_t distribution;

	static constexpr double EPSILON = 0.01;

	pn_belief_marking(std::weak_ptr<pn_net> net,
		marking_dist_t&& distribution);

	pn_belief_marking(pn_belief_marking&&)noexcept = default;
	pn_belief_marking(const pn_binary_marking::Ptr&);
	pn_belief_marking(const pn_binary_marking::ConstPtr&);
	pn_belief_marking(const pn_belief_marking&) = default;
	pn_belief_marking& operator=(const pn_belief_marking&) = default;
	pn_belief_marking& operator=(pn_belief_marking&&) = default;

	double get_probability(const pn_place::Ptr& place, const pn_token::Ptr& token) const;
	double get_probability(const pn_token::Ptr& token, const pn_place::Ptr& place) const;
	double get_probability(const pn_instance& instance) const;

	std::map<pn_token::Ptr, double> get_distribution(const pn_place::Ptr& place) const;
	std::map<pn_place::Ptr, double> get_distribution(const pn_token::Ptr& token) const;

	double get_summed_probability(const pn_place::Ptr& place) const;
	double get_summed_probability(const pn_token::Ptr& token) const;

	/**
	 * @param{threshold} only if the certainty of a token in some place exceeds this value, the token is considered to be present
	 * @returns this transition can fire in @param{marking}
	 **/
	double is_enabled(const pn_transition::Ptr& transition) const;

	double would_enable(
		const pn_transition::Ptr& transition,
		const std::set<pn_instance>& additional_distribution) const;

	Ptr fire(const pn_transition::Ptr& transition) const;

	pn_marking::Ptr to_marking() const;
};


class pn_boxed_place;


/**
 * @class pn_net
 * @brief Represents a complete Petri net model
 * 
 * The Petri net class manages the overall structure of the net, including
 * places, transitions, tokens, and their relationships. It provides methods
 * for creating and managing the network components.
 * 
 * Features:
 * - Place and transition creation
 * - Token management
 * - Network integration
 * - Goal state tracking
 * - Meta-transition handling
 * - Forward transition computation
 */
class STATEOBSERVATION_API pn_net
{
public:
	typedef std::shared_ptr<pn_net> Ptr;

	const object_parameters* object_params;

	//Must be locked while iterating over transitions or places, or when the net is changed
	mutable std::mutex mutex;

	pn_net();
	pn_net(const object_parameters& params);
	pn_net(pn_net&&) = default;
	pn_net(const pn_net&) = default;
	pn_net& operator=(const pn_net&) = default;
	pn_net& operator=(pn_net&&) = default;

	pn_place::Ptr create_place(bool agent = false);
	
	pn_transition::Ptr create_transition(std::vector<pn_transition::pn_arc>&& inputs,
		std::vector<pn_transition::pn_arc>&& outputs);
	void add_transition(const pn_transition::Ptr& transition);

	void add_place(const pn_place::Ptr& place, bool agent = false);

	void add_token(const pn_token::Ptr& token);

	/**
	* Returns a place that has almost the same pose as box (except for differences of +/- min_object_dimension),
	* or none
	*/
	std::shared_ptr<pn_boxed_place> get_place(const obb& box) const noexcept;

	// Ensure that mutex is locked before iterating over places
	const std::vector<pn_place::Ptr>& get_places() const;

	// Ensure that mutex is locked before iterating over places
	const std::set<pn_place::Ptr>& get_agent_places() const;
	
	// Ensure that mutex is locked before iterating over transitions
	const std::vector<pn_transition::Ptr>& get_transitions() const;

	// Mutex must not be locked before calling
	const std::vector<pn_transition::Ptr> get_meta_transitions() const;

	// Ensure that mutex is locked before iterating over tokens
	const std::vector<pn_token::Ptr>& get_tokens() const;

	// Mutex must not be locked before calling
	// Returns all actions the given agent can execute that contribute to the goal
	const std::set<pn_transition::Ptr> get_forward_transitions(const pn_place::Ptr& agent) const;

	const pn_place::Ptr& get_place(size_t index) const;
	const pn_transition::Ptr& get_transition(size_t index) const;
	const pn_token::Ptr& get_token(size_t index) const;

	void set_goal(const pn_place::Ptr& place);
	pn_place::Ptr get_goal() const;

	std::set<pn_instance> get_goal_instances() const;

	/*
	 * Merges @param{other} into this, reuses places of other
	 * Agent places are mapped and transitions recreated
	 */
	void integrate(pn_net&& other, bool optional_goal = false);

	 /*Prints a representation of emission and marking to the console
	 * The representation follows the spatial alignment and uses the following symbols:
	 * '_' - empty place
	 * 'X' - observed, occupied place
	 * 'x' - potentially occupied place
	 * '0' - unobserved place
	 */
	static void print_benchmark_state(const pn_emission& emissions, const pn_belief_marking& marking);

private:
	std::vector<pn_place::Ptr> places;
	std::set<pn_place::Ptr> agent_places;
	std::vector<pn_transition::Ptr> transitions;
	std::set<pn_token::Ptr> tokens;
	std::vector<pn_token::Ptr> tokens_vec;
	pn_place::Ptr goal;
};
	

/**
 * @class pn_emission
 * @brief Represents emission probabilities in a Petri net
 * 
 * Handles the emission probabilities for observations in the Petri net,
 * supporting both observed and unobserved states.
 * 
 * Features:
 * - Emission probability tracking
 * - Observation state management
 * - Place and instance state checking
 * - Probability distribution handling
 */
class STATEOBSERVATION_API pn_emission
{
public:
	typedef std::shared_ptr<const pn_emission> ConstPtr;

	std::set<pn_place::Ptr> empty_places;
	std::set<pn_place::Ptr> unobserved_places;
	std::map<pn_instance, double> token_distribution;
	// the maximum over token_distribution for a given place
	std::map<pn_place::Ptr, double> max_probabilities;

	pn_emission(std::set<pn_place::Ptr>&& empty_places,
		std::set<pn_place::Ptr>&& unobserved_places,
		std::map<pn_instance, double>&& token_distribution,
		std::map<pn_place::Ptr, double>&& max_probabilities);

	bool is_empty(const pn_place::Ptr& place) const;
	bool is_empty(const pn_instance& instance) const;

	bool is_unobserved(const pn_place::Ptr& place) const;
	bool is_unobserved(const pn_instance& instance) const;

	double get_probability(const pn_instance& instance) const;
};


/**
 * @class transition_sequence
 * @brief Represents a sequence of transitions in a Petri net
 * 
 * Tracks a sequence of transitions that transform an initial marking
 * into a final marking in the Petri net.
 * 
 * Features:
 * - Transition sequence storage
 * - Initial and final marking tracking
 * - Sequence validation
 */
class STATEOBSERVATION_API transition_sequence
{
public:
	const pn_binary_marking::Ptr initial_marking;
	const pn_binary_marking::Ptr final_marking;
	const std::vector<pn_transition::Ptr> sequence;

	transition_sequence(pn_binary_marking::Ptr initial_marking,
		pn_binary_marking::Ptr final_marking,
		std::vector<pn_transition::Ptr> sequence);
};

}

#endif // !STATE_OBSERVATION__PN_MODEL_HPP
#pragma once

#ifndef STATE_OBSERVATION__PN_REASONING_HPP
#define STATE_OBSERVATION__PN_REASONING_HPP

#include "framework.hpp"

#include <random>
#include <list>
#include <unordered_set>

#include "pn_model.hpp"

namespace state_observation
{
class dbn_node;
class dbn_instance_node;
class dbn_transition_node;
class dbn_net;
class dbn_marking;
class pn_updater;
class pn_simple_transition_extractor;
class sampling_optimizer_belief;
class constraint_optimizer_belief;
typedef unsigned int slice_t;

/**
 * @class pn_transition_extractor
 * @brief Interface for transition extractors in Petri nets
 * 
 * Abstract base class defining the interface for classes that extract
 * transitions from Petri net markings and emissions.
 * 
 * Features:
 * - Marking update handling
 * - Emission update handling
 * - Transition extraction
 * - Threshold-based certainty checks
 */
class STATEOBSERVATION_API pn_transition_extractor
{
public:
	typedef std::shared_ptr<pn_transition_extractor> Ptr;

	virtual ~pn_transition_extractor() = default;

	virtual void update(pn_marking::ConstPtr marking, bool update_emission = true) = 0;
	virtual void update(const pn_emission::ConstPtr& emission) = 0;

	virtual std::set<pn_transition::Ptr> extract() = 0;

	constexpr static double certainty_equal_threshold = 0.01;
	constexpr static double token_changed_threshold = 0.85;
};

/**
 * @class pn_enabled_transition_extractor
 * @brief Extracts currently enabled transitions from a Petri net
 * 
 * Analyzes a Petri net to identify transitions that are likely
 * to be enabled based on current markings.
 * 
 * Features:
 * - Network-based analysis
 * - Probability threshold filtering
 * - Emission-based extraction
 */
class STATEOBSERVATION_API pn_enabled_transition_extractor
{
public:
	typedef std::shared_ptr<pn_enabled_transition_extractor> Ptr;
	
	pn_enabled_transition_extractor(pn_net::Ptr net);

	std::set<pn_transition::Ptr> extract(const pn_emission& emission) const;
	
	double probability_threshold = 0.5;
private:
	pn_net::Ptr net;

};

/**
 * @class pn_feasible_transition_extractor
 * @brief Extracts feasible transitions that may have fired
 * 
 * Analyzes Petri net state changes to identify transitions
 * that could have caused observed changes in markings.
 * 
 * Features:
 * - Blocked transition tracking
 * - Instance generation tracking
 * - Instance consumption tracking
 * - Source-target relationship tracking
 * - Place change monitoring
 */
class STATEOBSERVATION_API pn_feasible_transition_extractor : public pn_transition_extractor
{
public:
	typedef std::shared_ptr<pn_feasible_transition_extractor> Ptr;
	
	pn_feasible_transition_extractor(pn_net::Ptr net, pn_marking::ConstPtr initial_marking);

	~pn_feasible_transition_extractor() override = default;
	
	void update(pn_marking::ConstPtr marking, bool update_emission = true) override;
	void update(const pn_emission::ConstPtr& emission) override;
	
	std::set<pn_transition::Ptr> extract() override;

	/**
	 * Transitions blocked according to additional information (e.g. movment of agents)
	 * Transitions that are blocked based on information derived from emissions do not need to be included
	 */
	void set_blocked_transitions(std::set<pn_transition::Ptr> transitions);

	const std::set<pn_transition::Ptr>& get_blocked_transitions() const;
	
protected:
	pn_net::Ptr net;
	pn_marking::ConstPtr marking;
	pn_emission::ConstPtr emission;
	pn_emission::ConstPtr prev_emission;

	std::set<pn_instance> generated_instances;
	std::set<pn_instance> consumed_instances;

	std::set<pn_transition::Ptr> feasible_transitions;

	std::set<pn_place::Ptr> unchanged_places;
	std::map<pn_instance, std::set<pn_instance>> sources;
	std::list<pn_instance> open_places;
	std::map<pn_transition::Ptr, std::set<pn_instance>> open_transitions;

	std::set<pn_transition::Ptr> blocked_transitions;

	/*
	 * Helper methods for extraction
	 */
	void add_sources(const pn_instance& source, const pn_instance& target);

	bool is_param_setminus_sources_empty(const std::set<pn_instance>& places) const;

	virtual void handle_generated_instance(const pn_instance& instances);
	
	virtual bool handle_consumed_instance(const pn_instance& pair);
	
	virtual void propagate_consumption();

	virtual bool is_blocked(const pn_transition::Ptr& transition) const;
};

/**
 * @class pn_simple_transition_extractor
 * @brief Extracts transitions connected to changed places
 * 
 * A simplified transition extractor that identifies transitions
 * connected to places that have changed state.
 * 
 * Features:
 * - Network-based analysis
 * - Marking tracking
 * - Emission tracking
 * - Changed place tracking
 */
class STATEOBSERVATION_API pn_simple_transition_extractor : public pn_transition_extractor
{
public:
	typedef std::shared_ptr<pn_simple_transition_extractor> Ptr;

	pn_simple_transition_extractor(pn_net::Ptr net, pn_marking::ConstPtr initial_marking);

	void update(pn_marking::ConstPtr marking, bool update_emission = true) override;
	void update(const pn_emission::ConstPtr& emission) override;

	std::set<pn_transition::Ptr> extract() override;

private:
	pn_net::Ptr net;
	pn_marking::ConstPtr marking;
	pn_emission::ConstPtr emission;
	pn_emission::ConstPtr prev_emission;

	std::set<pn_transition::Ptr> feasible_transitions;

	std::set<pn_place::Ptr> unchanged_places;

};

/**
 * @class constraint_optimizer_belief
 * @brief Optimizes transition sequences using constraint satisfaction
 * 
 * Analyzes and optimizes sequences of transitions in a Petri net
 * based on constraints and observations.
 * 
 * Features:
 * - Transition sequence optimization
 * - Belief marking updates
 * - Emission consistency checking
 * - Prior probability tracking
 * - Observation probability tracking
 */
class STATEOBSERVATION_API constraint_optimizer_belief
{
public:
	typedef std::shared_ptr<constraint_optimizer_belief> Ptr;
	typedef std::vector<std::pair<pn_transition::Ptr, double>> transition_sequence;

	constraint_optimizer_belief(const std::set<pn_transition::Ptr>& feasible_transitions,
	                            pn_belief_marking::ConstPtr marking,
		const pn_emission::ConstPtr& emission);

	/*
	 * use @ref{combine} to compute a marking update
	 */
	pn_belief_marking::Ptr optimal_transition_sequence();

	/*
	 * searches best solution among all transition sequences
	 */
	void optimize_transition_backtracking(const pn_binary_marking::Ptr& source_marking,
		const pn_binary_marking::Ptr& marking, 
		double prior_probability,
		std::vector<pn_transition::Ptr> initial_transitions = std::vector<pn_transition::Ptr>());


	/**
 * The returned marking is the same as in apply, but addtinally a consistency value is returned
 * that increases when a transition fires stronger than the inputs allow or the output tokens of a place
 * sum up to more than one.
 *
 */
	double emission_consistency(const pn_binary_marking::Ptr& marking) const;



private:

	std::vector<pn_transition::Ptr> all_transitions;
	pn_belief_marking::ConstPtr initial_marking;
	pn_emission::ConstPtr emission;
	std::set<pn_place::Ptr> marked_places;

	pn_belief_marking::marking_dist_t prior_probabilities;
	pn_belief_marking::marking_dist_t observation_probabilities;
	std::unordered_map<pn_binary_marking::Ptr,
		std::unordered_set<pn_binary_marking::Ptr, hash_pn_binary_marking_ptr, eq_pn_binary_marking_ptr>,
		hash_pn_binary_marking_ptr, eq_pn_binary_marking_ptr>
		sources;
};

/**
 * @class sampling_optimizer_belief
 * @brief Optimizes transition sequences using sampling
 * 
 * Uses sampling techniques to find optimal transition sequences
 * in a Petri net based on observations and beliefs.
 * 
 * Features:
 * - Transition sequence sampling
 * - Belief marking updates
 * - Emission consistency checking
 * - Fired transition tracking
 * - Sequence storage
 */
class STATEOBSERVATION_API sampling_optimizer_belief
{
public:
	typedef std::shared_ptr<sampling_optimizer_belief> Ptr;


	sampling_optimizer_belief(const std::set<pn_transition::Ptr>& feasible_transitions,
		const pn_belief_marking::ConstPtr& marking,
		const pn_emission::ConstPtr& emission);

	/*
	 * use @ref{combine} to compute a marking update
	 */
	pn_belief_marking::Ptr update(int iterations);

	/*
	 * probabilistically search for transition sequences that match the emission
	 */
	void sample_transition_sequence(const pn_binary_marking::Ptr& source_marking,
		const pn_binary_marking::Ptr& marking,
		double prior_probability);


	/**
 * The returned marking is the same as in apply, but addtinally a consistency value is returned
 * that increases when a transition fires stronger than the inputs allow or the output tokens of a place
 * sum up to more than one.
 *
 */
	double emission_consistency(const pn_binary_marking::Ptr& marking) const;

	const pn_belief_marking::marking_dist_t& get_tested_markings() const;

	std::map< pn_transition::Ptr, double> get_fired_transitions() const;

private:
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen; //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> uniform_norm_distribution;

	std::set<pn_transition::Ptr> all_transitions;
	pn_belief_marking::ConstPtr initial_marking;
	pn_emission::ConstPtr emission;
	std::set<pn_place::Ptr> marked_places;

	std::vector<pn_transition::Ptr> used_transitions;

	pn_belief_marking::marking_dist_t prior_probabilities;
	pn_belief_marking::marking_dist_t observation_probabilities;
	std::map<pn_transition::Ptr, unsigned int> transition_counter;

	std::set<pn_transition::Ptr> transitions_observed_output;
	std::set<pn_transition::Ptr> transitions_other;

	std::set<pn_transition::Ptr> unused_transitions_observed_output;
	std::set<pn_transition::Ptr> unused_transitions_other;
	std::vector<pn_transition::Ptr> fired_transitions;

	unsigned int count_successful_samples;
	
	void store_transition_sequence();
};
	
}

#endif /* !STATE_OBSERVATION__PN_REASONING_HPP */
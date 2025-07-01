#include "pn_reasoning.hpp"

#include <ranges>

#include "pn_model_extension.hpp"

#include <Eigen/Core>

namespace state_observation
{
/////////////////////////////////////////////////////////////
//
//
//  Class: pn_feasible_transition_extractor
//
//
/////////////////////////////////////////////////////////////

pn_feasible_transition_extractor::pn_feasible_transition_extractor(pn_net::Ptr net, pn_marking::ConstPtr initial_marking)
	: net(std::move(net)),
	marking(std::move(initial_marking))
{
	std::lock_guard<std::mutex> lock(this->net->mutex);

	std::set<pn_place::Ptr> empty_places(this->net->get_places().begin(), this->net->get_places().end());
	std::map<pn_instance, double> token_distribution;
	std::map<pn_place::Ptr, double> max_probabilities;

	for (const auto& entry : marking->distribution)
	{
		auto p_iter = empty_places.find(entry.first.first);
		auto peak = max_probabilities.find(entry.first.first);

		if (peak == max_probabilities.end())
			max_probabilities.emplace(entry.first.first, entry.second);
		else
			peak->second = std::max(peak->second, entry.second);

		if (p_iter == empty_places.end())
		{
			token_distribution.insert_or_assign(entry.first, entry.second);
		}
		else
		{
			empty_places.erase(p_iter);

			for (const pn_token::Ptr& token : this->net->get_tokens())
			{
				token_distribution.emplace(std::make_pair(entry.first.first, token), 0);
			}

			token_distribution.insert_or_assign(entry.first, entry.second);
		}
	}

	prev_emission = std::make_shared<pn_emission>(
		std::move(empty_places),
		std::set<pn_place::Ptr>(),
		std::move(token_distribution),
		std::move(max_probabilities));
}

void pn_feasible_transition_extractor::update(pn_marking::ConstPtr marking, bool update_emission)
{
	std::swap(marking, this->marking);

	if(update_emission)
		prev_emission = emission;

	generated_instances.clear();
	consumed_instances.clear();

	feasible_transitions.clear();

	sources.clear();
	open_places.clear();
	open_transitions.clear();
}

void pn_feasible_transition_extractor::update(const pn_emission::ConstPtr& emission)
{
	std::lock_guard<std::mutex> lock(net->mutex);
	this->emission = emission;

	generated_instances.clear();
	consumed_instances.clear();

	feasible_transitions.clear();

	unchanged_places.clear();

	sources.clear();
	open_places.clear();
	open_transitions.clear();

	for (const pn_place::Ptr& place : prev_emission->empty_places)
	{
		if (emission->is_empty(place))
			unchanged_places.emplace(place);
	}

	for (const auto& entry : prev_emission->token_distribution)
	{
		if (emission->is_empty(entry.first) || emission->is_unobserved(entry.first))
			continue;

		auto iter = emission->max_probabilities.find(entry.first.first);
		const double peak = iter == emission->max_probabilities.end() ? 1. : iter->second;
		if (std::abs(entry.second - emission->get_probability(entry.first)) / peak < token_changed_threshold)
			unchanged_places.emplace(entry.first.first);
	}

	for (const std::pair<pn_instance, double>& entry : emission->token_distribution)
	{
		auto iter = emission->max_probabilities.find(entry.first.first);
		const double peak = iter == emission->max_probabilities.end() ? 1. : iter->second;
		double certainty_diff = entry.second / peak - marking->get_probability(entry.first);
		if (certainty_diff > token_changed_threshold || marking->get_summed_probability(entry.first.first) < 1 - certainty_equal_threshold)
		{
			generated_instances.emplace(entry.first);

			// in rare cases prev_emission is not valid for marking
			unchanged_places.erase(entry.first.first);
		}
		else if (certainty_diff < -0.5)
		{
			consumed_instances.emplace(entry.first);
			unchanged_places.erase(entry.first.first);
		}
	}

	for (const pn_place::Ptr& place : emission->empty_places)
	{
		if (marking->get_summed_probability(place) > token_changed_threshold)
		{
			for (const std::pair<pn_token::Ptr, double>& entry : marking->get_distribution(place))
			{
				if (entry.second > certainty_equal_threshold)
				{
					auto instance = std::make_pair(place, entry.first);
					consumed_instances.emplace(instance);
					unchanged_places.erase(place);
					add_sources(instance, instance);
				}
			}
		}
	}
}

std::set<pn_transition::Ptr> pn_feasible_transition_extractor::extract()
{

	for (const pn_place::Ptr& empty_place : emission->empty_places)
		if (marking->get_summed_probability(empty_place) > certainty_equal_threshold)
			for (const auto& entry : marking->get_distribution(empty_place))
				if (entry.second > certainty_equal_threshold)
				{
					pn_instance inst = std::make_pair(empty_place, entry.first);
					add_sources(inst, { inst });
				}

	propagate_consumption();

	for (const auto& entry : generated_instances)
	{
		auto iter = sources.find(entry);
		if (iter == sources.end())
			handle_generated_instance(entry);
	}

	open_transitions.clear();

	for (const auto& entry : consumed_instances)
	{

		handle_consumed_instance(entry);
	}

	return feasible_transitions;
}

void pn_feasible_transition_extractor::set_blocked_transitions(std::set<pn_transition::Ptr> transitions)
{
	blocked_transitions = std::move(transitions);
}

const std::set<pn_transition::Ptr>& pn_feasible_transition_extractor::get_blocked_transitions() const
{
	return blocked_transitions;
}


void pn_feasible_transition_extractor::add_sources(const pn_instance& source, const pn_instance& target)
{
	if (std::dynamic_pointer_cast<pn_empty_token>(source.second))
		return;

	auto t_iter = sources.find(target);
	if (t_iter == sources.end())
	{
		auto s_iter = sources.find(source);
		if (s_iter == sources.end())
			sources.emplace(target, std::set<pn_instance>({ source }));
		else
		{
			sources.emplace(target, s_iter->second);
			sources[target].emplace(target);
		}

		open_places.emplace_back(target);
	}
	else
	{
		unsigned int size = t_iter->second.size();
		const auto& s_sources = sources[source];
		t_iter->second.insert(s_sources.begin(), s_sources.end());

		if (size != t_iter->second.size())
		{ // new sources added to target, propagate changes
			for (auto& entry : sources)
				if (entry.second.contains(t_iter->first))
					entry.second.insert(t_iter->second.begin(), t_iter->second.end());
		}
	}
}

bool pn_feasible_transition_extractor::is_param_setminus_sources_empty(const std::set<pn_instance>& places) const
{
	for (const auto& in_place : places)
	{
		if (!std::dynamic_pointer_cast<pn_empty_token>(in_place.second) && !sources.contains(in_place))
			return false;
	}

	return true;
}

void pn_feasible_transition_extractor::handle_generated_instance(const pn_instance& instance)
{
	if (std::dynamic_pointer_cast<pn_empty_token>(instance.second))
		return;

	// skip place if it definitely produces a token
	if (marking->get_probability(instance) > 1 - certainty_equal_threshold &&
		(emission->is_unobserved(instance) || emission->is_empty(instance)))
	{
		return;
	}

	if (this->sources.contains(instance))
		return;

	for (const auto& w_transition : instance.first->get_incoming_transitions())
	{
		const pn_transition::Ptr transition = w_transition.lock();
		auto side_conditions = transition->get_side_conditions();

		if (!transition || !transition->has_output_arc(instance) || side_conditions.contains(instance))
			continue;

		if (feasible_transitions.contains(transition))
			continue;

		// check that transition is feasible to fire (i.e. destination unobserved or has token)
		if (is_blocked(transition))
			continue;

		feasible_transitions.emplace(transition);

		if (!is_param_setminus_sources_empty(transition->inputs))
		{
			for (const auto& in_instance : transition->inputs)
			{
				if (marking->get_probability(in_instance) < 1 || emission->get_probability(in_instance) > 0)
					handle_generated_instance(in_instance);
			}
		}
	}
}

bool pn_feasible_transition_extractor::handle_consumed_instance(const pn_instance& instance)
{
	if (std::dynamic_pointer_cast<pn_empty_token>(instance.second))
		return true;

	std::list<pn_transition::Ptr> open_transitions;

	for (const auto& w_transition : instance.first->get_outgoing_transitions())
	{
		const pn_transition::Ptr transition = w_transition.lock();
		if (!transition)
			continue;

		auto side_conditions = transition->get_side_conditions();

		if (!transition->has_input_arc(instance) || side_conditions.contains(instance))
			continue;

		if (feasible_transitions.contains(transition)) // we stop when we find any transition that consumes the token
			return true;

		if (!is_param_setminus_sources_empty(transition->inputs)) // we do not follow the transition if we have found no way to generate all input instances
			continue;

		if (is_blocked(transition))
			continue;

		if (!feasible_transitions.contains(transition))
		{
			open_transitions.emplace_back(transition);
		}
	}

	bool found_sink = false;
	for (const auto& transition : open_transitions)
	{
		bool is_enabled = true;
		for (const auto& out_instance : transition->outputs)
		{
			if (emission->get_probability(out_instance) > token_changed_threshold)
			{
				is_enabled = false;
				break;
			}
		}

		if (is_enabled)
		{
			found_sink = true;
			feasible_transitions.emplace(transition);
		}
	}

	return found_sink;
}

void pn_feasible_transition_extractor::propagate_consumption()
{
	/*
		 * build closure over consumed instances following outgoing transitions
		 * */

	while (!open_places.empty())
	{
		const auto instance = open_places.front();
		open_places.pop_front();

		// skip place if it definitely consumes a token
		if (marking->get_probability(instance) < 1 &&
			(emission->is_unobserved(instance) || emission->get_probability(instance) > certainty_equal_threshold))
		{
			continue;
		}

		for (const std::weak_ptr<pn_transition>& w_transition : instance.first->get_outgoing_transitions())
		{
			auto transition = w_transition.lock();
			if (!transition)
				continue;

			auto t_iter = open_transitions.find(transition);
			if (t_iter == open_transitions.end())
			{
				t_iter = open_transitions.emplace(transition, transition->inputs).first;

				for (const auto& side : transition->get_side_conditions())
					if (marking->get_probability(side) > certainty_equal_threshold ||
						sources.contains(side))
						t_iter->second.erase(side);
			}
			t_iter->second.erase(instance);
			if (t_iter->second.empty()) // we only follow transitions which input instances are generated by the propage_consumption process
			{
				if (is_blocked(t_iter->first))
					continue;

				feasible_transitions.emplace(transition);

				for (const auto& out_place : t_iter->first->outputs)
					if (!t_iter->first->is_side_condition(out_place))
						add_sources(instance, out_place);


				open_transitions.erase(t_iter);
			}
		}
	}
}

bool pn_feasible_transition_extractor::is_blocked(const pn_transition::Ptr& transition) const
{
	for (const auto& in_instance : transition->inputs)
	{
		if (std::dynamic_pointer_cast<pn_empty_token>(in_instance.second))
		{
			if (emission->is_empty(in_instance) || prev_emission->is_empty(in_instance))
				continue;
			else if (!emission->is_unobserved(in_instance) && !prev_emission->is_unobserved(in_instance))
				return true;
			else
				continue;
		}

		if (transition->is_side_condition(in_instance))
		{
			if (emission->is_empty(in_instance) && prev_emission->is_empty(in_instance))
			{
				return true;
			}
		}
		else if (unchanged_places.contains(in_instance.first) ||
			prev_emission->is_empty(in_instance) && emission->get_probability(in_instance) > 0)
		{
			return true;
		}
	}

	for (const auto& out_instance : transition->outputs)
	{
		if (!transition->is_side_condition(out_instance) && unchanged_places.contains(out_instance.first) ||
			emission->is_empty(out_instance) && prev_emission->get_probability(out_instance) > 0)
		{
			return true;
		}
	}

	return blocked_transitions.contains(transition);
}

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_simple_transition_extractor
//
//
/////////////////////////////////////////////////////////////

pn_simple_transition_extractor::pn_simple_transition_extractor(pn_net::Ptr net, pn_marking::ConstPtr initial_marking)
	: net(std::move(net)),
	marking(std::move(initial_marking))
{
	std::lock_guard<std::mutex> lock(this->net->mutex);

	std::set<pn_place::Ptr> empty_places(this->net->get_places().begin(), this->net->get_places().end());
	std::map<pn_instance, double> token_distribution;
	std::map<pn_place::Ptr, double> max_probabilities;

	for (const auto& entry : marking->distribution)
	{
		auto peak = max_probabilities.find(entry.first.first);

		if (peak == max_probabilities.end())
			max_probabilities.emplace(entry.first.first, entry.second);
		else
			peak->second = std::max(peak->second, entry.second);

		auto p_iter = empty_places.find(entry.first.first);

		if (p_iter == empty_places.end())
		{
			token_distribution.insert_or_assign(entry.first, entry.second);
		}
		else
		{
			empty_places.erase(p_iter);

			for (const pn_token::Ptr& token : this->net->get_tokens())
			{
				token_distribution.emplace(std::make_pair(entry.first.first, token), 0);
			}

			token_distribution.insert_or_assign(entry.first, entry.second);
		}
	}

	prev_emission = std::make_shared<pn_emission>(
		std::move(empty_places),
		std::set<pn_place::Ptr>(),
		std::move(token_distribution),
		std::move(max_probabilities));
}

void pn_simple_transition_extractor::update(pn_marking::ConstPtr marking, bool update_emission)
{
	std::swap(marking, this->marking);

	if(update_emission)
		prev_emission = emission;

	feasible_transitions.clear();
}

void pn_simple_transition_extractor::update(const pn_emission::ConstPtr& emission)
{
	this->emission = emission;

	feasible_transitions.clear();

	unchanged_places.clear();

	for (const pn_place::Ptr& place : emission->empty_places)
	{
		if (prev_emission->is_empty(place) ||
			marking->get_summed_probability(place) < certainty_equal_threshold)
			unchanged_places.emplace(place);
	}

	for (const auto& instance : prev_emission->token_distribution | std::views::keys)
	{
		if (emission->get_probability(instance) > 0)
			unchanged_places.emplace(instance.first);
	}
}

std::set<pn_transition::Ptr> pn_simple_transition_extractor::extract()
{
	for (const pn_transition::Ptr& transition : net->get_transitions())
	{
		bool skip = false;

		for (const auto& out_instance : transition->outputs)
		{
			if (transition->is_side_condition(out_instance))
			{
				if (emission->is_empty(out_instance) && prev_emission->is_empty(out_instance))
				{
					skip = true;
					break;
				}
			}
			else if (unchanged_places.contains(out_instance.first))
			{
				skip = true;
				break;
			}
		}

		if (skip)
			continue;

		for (const auto& in_instance : transition->inputs)
		{
			if (!transition->is_side_condition(in_instance) && unchanged_places.contains(in_instance.first))
			{
				skip = true;
				break;
			}
		}

		if (skip)
			continue;

		feasible_transitions.emplace(transition);
	}
	return feasible_transitions;
}

/////////////////////////////////////////////////////////////
//
//
//  Class: constraint_optimizer_belief
//
//
/////////////////////////////////////////////////////////////

constraint_optimizer_belief::constraint_optimizer_belief(const std::set<pn_transition::Ptr>& feasible_transitions,
                                                         pn_belief_marking::ConstPtr marking, const pn_emission::ConstPtr& emission)
	: all_transitions(feasible_transitions.begin(), feasible_transitions.end()),
	initial_marking(std::move(marking)),
	emission(emission)
{
	for (const auto& instance : emission->token_distribution | std::views::keys)
		if(!std::dynamic_pointer_cast<pn_empty_token>(instance.second))
			marked_places.emplace(instance.first);
}

pn_belief_marking::Ptr  constraint_optimizer_belief::optimal_transition_sequence()
{
	for (const auto& marking : initial_marking->distribution)
	{
		prior_probabilities.emplace(marking.first, marking.second);
		observation_probabilities.emplace(marking.first, emission_consistency(marking.first));
		sources.emplace(marking.first, std::unordered_set<pn_binary_marking::Ptr, hash_pn_binary_marking_ptr, eq_pn_binary_marking_ptr>({ marking.first }));
	}

	for (const auto& marking : initial_marking->distribution)
	{

		optimize_transition_backtracking(marking.first, marking.first, marking.second);
	}

	double sum = 0;

	double peak_product = 1;
	for (const auto& probability : emission->max_probabilities | std::views::values)
		peak_product *= probability;

	for (auto iter = prior_probabilities.begin(); iter != prior_probabilities.end();)
	{
		iter->second *= observation_probabilities.at(iter->first);


		if (iter->second < peak_product * pn_belief_marking::EPSILON)
			iter = prior_probabilities.erase(iter);
		else
		{
			sum += iter->second;
			++iter;
		}
	}

	// no hypothesis found, return old one
	if (prior_probabilities.empty())
		return std::make_shared<pn_belief_marking>(initial_marking->net, pn_belief_marking::marking_dist_t(initial_marking->distribution));

	for (auto& probability : prior_probabilities | std::views::values)
		probability /= sum;

	return std::make_shared<pn_belief_marking>(initial_marking->net, std::move(prior_probabilities));
}

void constraint_optimizer_belief::optimize_transition_backtracking(const pn_binary_marking::Ptr& source_marking,
	const pn_binary_marking::Ptr& marking,
	double prior_probability,
	std::vector<pn_transition::Ptr> initial_transitions)
{
	bool recursion_end = initial_transitions.size() == all_transitions.size();

	const std::set< pn_transition::Ptr> used_transitions(initial_transitions.begin(), initial_transitions.end());

	for (const pn_transition::Ptr& next_transition : all_transitions)
	{
		if (used_transitions.contains(next_transition))
			continue;

		if (!marking->is_enabled(next_transition))
			continue;

		initial_transitions.push_back(next_transition);
		const auto next_marking = marking->fire(next_transition);

		auto iter = prior_probabilities.find(next_marking);
		if (iter == prior_probabilities.end())
		{
			double prob = emission_consistency(next_marking);
			prior_probabilities.emplace(next_marking, prior_probability);
			observation_probabilities.emplace(next_marking, prob);
			sources.emplace(next_marking, std::unordered_set<pn_binary_marking::Ptr, hash_pn_binary_marking_ptr, eq_pn_binary_marking_ptr>({ source_marking }));
		}
		else
		{
			auto src = sources.at(next_marking);
			if (!src.contains(source_marking))
			{
				iter->second += prior_probability;
				src.emplace(source_marking);
			}
			else
			{
				continue;
			}
		}

		optimize_transition_backtracking(source_marking, next_marking, prior_probability, initial_transitions);
		initial_transitions.pop_back();
	}
}

double constraint_optimizer_belief::emission_consistency(const pn_binary_marking::Ptr& marking) const
{
	auto m = initial_marking;
	double consistency = 1.;

	// empty places can either have no token or the empty token
	std::set<pn_place::Ptr> remaining_marked_places(marked_places);

	for (const auto& instance : marking->distribution)
	{
		if (std::dynamic_pointer_cast<pn_empty_token>(instance.second))
			continue;

		if (emission->is_empty(instance))
			return 0;

		if (emission->is_unobserved(instance))
			continue;

		remaining_marked_places.erase(instance.first);

		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);

		if (boxed_place)
			for (const auto& overlap : boxed_place->overlapping_places)
				if (marking->is_occupied(overlap.lock()))
					return 0;
				else
					// if we observe objects at two overlapping places, only one of them must have an object
					remaining_marked_places.erase(overlap.lock()); 

		consistency *= emission->get_probability(instance);
	}

	if (!remaining_marked_places.empty())
		return 0;

	return consistency;
}

/////////////////////////////////////////////////////////////
//
//
//  Class: sampling_optimizer_belief
//
//
/////////////////////////////////////////////////////////////

sampling_optimizer_belief::sampling_optimizer_belief(const std::set<pn_transition::Ptr>& feasible_transitions,
	const pn_belief_marking::ConstPtr& marking,
	const pn_emission::ConstPtr& emission)
	:
	gen(rd()),
	uniform_norm_distribution(0., 1.),
	initial_marking(marking),
	emission(emission),
	count_successful_samples(0)
{
	std::lock_guard<std::mutex> lock(marking->net.lock()->mutex);
	for (const auto& transition : feasible_transitions)
	{
		bool observed_output = false;
		for (const auto& output : transition->outputs)
			if (!emission->is_unobserved(output) && !transition->is_side_condition(output) && !std::dynamic_pointer_cast<pn_empty_token>(output.second))
			{
				observed_output = true;
				transitions_observed_output.emplace(transition);
				break;
			}
		if (!observed_output)
			transitions_other.emplace(transition);
	}

	for (const auto& instance : emission->token_distribution | std::views::keys)
		marked_places.emplace(instance.first);
}

pn_belief_marking::Ptr  sampling_optimizer_belief::update(int iterations)
{
	for (const auto& marking : initial_marking->distribution)
	{
		prior_probabilities.emplace(marking.first, marking.second);
		observation_probabilities.emplace(marking.first, emission_consistency(marking.first));
	}

	for (int i = 0; i < iterations; i++)
	{
		double rand = uniform_norm_distribution(gen);
		double sum = 0;

		fired_transitions.clear();

		for (const auto& marking : initial_marking->distribution)
		{
			sum += marking.second;
			used_transitions.clear();

			if (sum > rand)
			{
				unused_transitions_observed_output = transitions_observed_output;
				unused_transitions_other = transitions_other;
				sample_transition_sequence(marking.first, marking.first, 1. / iterations);

				break;
			}
		}
	}

	double sum = 0;
	double peak_product = 1;
	for (const auto& probability : emission->max_probabilities | std::views::values)
		peak_product *= probability;

	for (auto iter = prior_probabilities.begin(); iter != prior_probabilities.end();)
	{
		iter->second *= observation_probabilities.at(iter->first);


		if (iter->second < peak_product * pn_belief_marking::EPSILON)
			iter = prior_probabilities.erase(iter);
		else
		{
			sum += iter->second;
			++iter;
		}
	}

	// no hypothesis found, return old one
	if (prior_probabilities.empty())
		throw std::exception();
	//		return std::make_shared<pn_belief_marking>(initial_marking->net, pn_belief_marking::marking_dist_t(initial_marking->distribution));

	for (auto& probability : prior_probabilities | std::views::values)
		probability /= sum;

	return std::make_shared<pn_belief_marking>(initial_marking->net, std::move(prior_probabilities));
}

void sampling_optimizer_belief::sample_transition_sequence(const pn_binary_marking::Ptr& source_marking,
	const pn_binary_marking::Ptr& marking,
	double prior_probability)
{
	std::vector<pn_transition::Ptr> enabled_transitions;
	std::set<pn_transition::Ptr>* reference_set = &unused_transitions_observed_output;

	for (const auto& transition : unused_transitions_observed_output)
		if (marking->is_enabled(transition))
			enabled_transitions.push_back(transition);

	if (enabled_transitions.empty())
	{
		reference_set = &unused_transitions_other;

		for (const auto& transition : unused_transitions_other)
			if (marking->is_enabled(transition))
				enabled_transitions.push_back(transition);
	}

	if (enabled_transitions.empty())
		return;

	std::uniform_int_distribution<> dis(0., enabled_transitions.size() - 1);
	auto next_transition = enabled_transitions[dis(gen)];

	fired_transitions.emplace_back(next_transition);

	reference_set->erase(next_transition);

	const auto next_marking = marking->fire(next_transition);

	auto iter = prior_probabilities.find(next_marking);
	if (iter == prior_probabilities.end())
	{
		double prob = emission_consistency(next_marking);
		iter = prior_probabilities.emplace(next_marking, prior_probability).first;
		observation_probabilities.emplace(next_marking, prob);

		//if(prob > 0)
		//	transition_sequences.at(next_marking).push_back(used_transitions);
	}
	else
	{
		iter->second += prior_probability;
	}

	if (observation_probabilities.at(next_marking) > 0)
	{
		store_transition_sequence();
		return;
	}

	sample_transition_sequence(source_marking, next_marking, prior_probability);

}

double sampling_optimizer_belief::emission_consistency(const pn_binary_marking::Ptr& marking) const
{
	auto m = initial_marking;
	double consistency = 1.;

	std::set<pn_place::Ptr> remaining_marked_places(marked_places);

	for (const auto& instance : marking->distribution)
	{
		if (std::dynamic_pointer_cast<pn_empty_token>(instance.second))
			continue;

		if (emission->is_empty(instance))
			return 0;

		remaining_marked_places.erase(instance.first);

		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);

		if (boxed_place)
			for (const auto& overlap : boxed_place->overlapping_places)
				if (marking->is_occupied(overlap.lock()))
					return 0;
				else
					// if we observe objects at two overlapping places, only one of them must have an object
					remaining_marked_places.erase(overlap.lock());

		if (emission->is_unobserved(instance))
			continue;

		consistency *= emission->get_probability(instance);

		//early-out
		if (consistency == 0.)
			return 0;
	}

	if (!remaining_marked_places.empty())
		return 0;

	return consistency;
}

const pn_belief_marking::marking_dist_t& sampling_optimizer_belief::get_tested_markings() const
{
	return observation_probabilities;
}

std::map< pn_transition::Ptr, double> sampling_optimizer_belief::get_fired_transitions() const
{
	std::map< pn_transition::Ptr, double> result;

	for (const auto& entry : transition_counter)
		result.emplace(entry.first, static_cast<double>(entry.second) / static_cast<double>(count_successful_samples));

	return result;
}

void sampling_optimizer_belief::store_transition_sequence()
{
	count_successful_samples++;

	for (const auto& transition : fired_transitions)
	{
		auto iter = transition_counter.find(transition);

		if (iter == transition_counter.end())
			transition_counter.emplace(transition, 1);
		else
			iter->second++;
	}
}

pn_enabled_transition_extractor::pn_enabled_transition_extractor(pn_net::Ptr net)
	:
	net(std::move(net))
{}

std::set<pn_transition::Ptr> pn_enabled_transition_extractor::extract(const pn_emission& emission) const
{
	std::set<pn_transition::Ptr> out;

	for (const auto& transition : net->get_transitions())
	{
		bool enabled = true;

		//all places on incoming arcs need to be occupied
		for (const auto& requirement : transition->inputs)
		{
			if (emission.get_probability(requirement) < probability_threshold)
			{
				enabled = false;
				break;
			}
		}

		//all places on outgoing arcs need to be free except for side conditions
		for (const auto& output : transition->outputs)
		{
			if (!transition->is_side_condition(output) &&
				emission.get_probability(output) > probability_threshold)
			{
				enabled = false;
				break;
			}
		}

		if (enabled)
			out.insert(transition);
	}

	return out;
}

} // namespace state_observation
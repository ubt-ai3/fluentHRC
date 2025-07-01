#include "pn_model.hpp"
#include "pn_model_extension.hpp"

#include <boost/container_hash/hash.hpp>

#include <stdexcept>
#include <set>
#include <stack>
#include <algorithm>
#include <ranges>

#include "workspace_objects.hpp"

namespace state_observation
{

#ifdef DEBUG_PN_ID
int pn_place::id_counter = 0;
int pn_token::id_counter = 0;
int pn_transition::id_counter = 0;
#endif

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_place
//
//
/////////////////////////////////////////////////////////////

#ifdef DEBUG_PN_ID
pn_place::pn_place()
	: id(id_counter++)
{}
#endif

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_token
//
//
/////////////////////////////////////////////////////////////
#ifdef DEBUG_PN_ID
pn_token::pn_token()
	: id(id_counter++)
{}
#endif

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_transition
//
//
/////////////////////////////////////////////////////////////

pn_transition::pn_transition(std::vector<pn_arc>&& inputs,
	std::vector<pn_arc>&& outputs)
	:
#ifdef DEBUG_PN_ID
	id(id_counter++),
#endif
	inputs(inputs.begin(), inputs.end()),
	outputs(outputs.begin(), outputs.end())
{
	compute_side_conditions();
}

std::vector<pn_place::Ptr> pn_transition::get_inputs(const std::set<pn_token::Ptr>& filter) const
{
	if (filter.empty())
	{
		const auto& keys = std::views::keys(inputs);
		return std::vector<pn_place::Ptr>{ keys.begin(), keys.end() };
	}

	std::vector<pn_place::Ptr> result;
	for (const pn_transition::pn_arc& arc : inputs)
	{
		if (filter.contains(arc.second))
			result.push_back(arc.first);
	}
	return result;
}

std::vector<pn_place::Ptr> pn_transition::get_outputs(const std::set<pn_token::Ptr>& filter) const
{
	if (filter.empty())
	{
		const auto& keys = std::views::keys(outputs);
		return std::vector<pn_place::Ptr>{ keys.begin(), keys.end() };
	}

	std::vector<pn_place::Ptr> result;
	for (const pn_transition::pn_arc& arc : outputs)
	{
		if (filter.contains(arc.second))
			result.push_back(arc.first);
	}

	return result;
}

std::set<pn_transition::pn_arc> pn_transition::get_pure_input_arcs() const
{
	std::set<pn_arc> output;
	std::ranges::set_difference(inputs, side_conditions, std::inserter(output, output.begin()));

	return output;
}

std::set<pn_transition::pn_arc> pn_transition::get_pure_output_arcs() const
{
	std::set<pn_arc> output;
	std::ranges::set_difference(outputs, side_conditions, std::inserter(output, output.begin()));

	return output;
}

bool pn_transition::has_input_arc(const pn_arc& arc) const
{
	return inputs.contains(arc);
}

bool pn_transition::has_output_arc(const pn_arc& arc) const
{
	return outputs.contains(arc);
}

bool pn_transition::is_side_condition(const pn_arc& arc) const
{
	//return has_input_arc(arc) && has_output_arc(arc);
	return side_conditions.contains(arc);
}

bool pn_transition::reverses(const pn_transition& t2) const
{
	const auto& t1 = *this;
	
	for (const auto& input : t1.get_pure_input_arcs())
	{
		if (!t2.has_output_arc(input))
			return false;
	}
	for (const auto& input : t2.get_pure_input_arcs())
	{
		if (!t1.has_output_arc(input))
			return false;
	}

	for (const auto& output : t1.get_pure_output_arcs())
	{
		if (!t2.has_input_arc(output))
			return false;
	}
	for (const auto& output : t2.get_pure_output_arcs())
	{
		if (!t1.has_input_arc(output))
			return false;
	}
	return true;
}

const std::set<pn_transition::pn_arc> pn_transition::get_side_conditions() const
{
	return side_conditions;
}

std::string pn_transition::to_string() const
{
	return std::string();
}

void pn_transition::compute_side_conditions()
{
	side_conditions.clear();
	std::ranges::set_intersection(inputs, outputs, std::inserter(side_conditions, side_conditions.begin()));
	/*
	side_conditions = {};
	const std::set<pn_arc> *vec1 = &inputs, *vec2 = &outputs;
	if (vec1->size() > vec2->size())
		std::swap(vec1, vec2);

	std::set<pn_arc> arcs_set(vec1->begin(), vec1->end());
	for (const pn_arc& arc : *vec2)
		if (arcs_set.contains(arc))
			side_conditions.emplace(arc);
	*/
}

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_marking
//
//
/////////////////////////////////////////////////////////////

pn_marking::pn_marking(std::weak_ptr<pn_net> net, std::map<pn_instance, double>&& distribution)
	: net(std::move(net)),
	distribution(distribution)
{}

double pn_marking::get_probability(const pn_place::Ptr& place, const pn_token::Ptr& token) const
{
	return get_probability(std::make_pair(place, token));
}

double pn_marking::get_probability(const pn_token::Ptr& token, const pn_place::Ptr& place) const
{
	return get_probability(std::make_pair(place, token));
}

double pn_marking::get_probability(const pn_instance& instance) const
{
	if (auto iter = distribution.find(instance); iter != distribution.end())
		return iter->second;

	return 0;
}

std::map<pn_token::Ptr, double> pn_marking::get_distribution(const pn_place::Ptr& place) const
{
	std::map<pn_token::Ptr, double> result;
	for (const auto& entry : distribution) // entry: ((place,token),double)
		if (entry.first.first == place && !std::dynamic_pointer_cast<pn_empty_token>(entry.first.second))
			result.emplace(entry.first.second, entry.second);

	return result;
}

std::map<pn_place::Ptr, double> pn_marking::get_distribution(const pn_token::Ptr& token) const
{
	std::map<pn_place::Ptr, double> result;
	for (const auto& entry : distribution) // entry: ((place,token),double)
		if (entry.first.second == token)
			result.emplace(entry.first.first, entry.second);

	return result;
}

double pn_marking::get_summed_probability(const pn_place::Ptr& place) const
{
	double prob = 0.0;
	for (const auto& entry : distribution) // entry: ((place,token),double)
		if (entry.first.first == place && !std::dynamic_pointer_cast<pn_empty_token>(entry.first.second))
			prob += entry.second;

	return prob;
}

double pn_marking::get_summed_probability(const pn_token::Ptr& token) const
{
	double prob = 0.0;
	for (const auto& entry : distribution) // entry: ((place,token),double)
		if (entry.first.second == token)
			prob += entry.second;

	return prob;
}

double pn_marking::is_enabled(const pn_transition::Ptr& transition) const
{
	double certainty = 1.0;
	for (const pn_transition::pn_arc& arc : transition->inputs)
	{
		certainty = std::min(certainty, get_probability(arc));

		if (certainty < EPSILON)
			return 0.0;
	}
	return certainty;
}

double pn_marking::would_enable(
	const pn_transition::Ptr& transition,
	const std::map<pn_instance, double>& additional_distribution) const
{
	double certainty = 1.0;
	for (pn_transition::pn_arc arc : transition->inputs)
	{
		double probability = get_probability(arc);

		if (auto it = additional_distribution.find(arc); it != additional_distribution.end())
			probability += it->second;

		certainty = std::min(certainty, probability);

		if (certainty < EPSILON)
			return 0.;
	}
	return certainty;
}

pn_marking::Ptr pn_marking::fire(const pn_transition::Ptr& transition, double threshold) const
{
	if (threshold < 0)
		throw std::invalid_argument("Threshold must be greater than 0.");

	double certainty = is_enabled(transition);

	if (certainty < EPSILON)
		throw std::invalid_argument("Transition not enabled");

	certainty = std::min(certainty, threshold);

	auto new_distribution(distribution);

	for (const pn_transition::pn_arc& arc : transition->inputs)
	{
		if (!transition->is_side_condition(arc))
			new_distribution[arc] -= certainty;
	}

	for (const pn_transition::pn_arc& arc : transition->outputs)
	{
		if (transition->is_side_condition(arc))
			continue;

		const auto iter = new_distribution.try_emplace(arc, certainty);
		if (!iter.second)
			iter.first->second += certainty;
	}
	return std::make_shared<pn_marking>(net, std::move(new_distribution));
}

void pn_marking::compact()
{
	std::erase_if(distribution, [](const std::pair<pn_instance, double>& val)
		{
			return val.second < EPSILON;
		});
}

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_binary_marking
//
//
/////////////////////////////////////////////////////////////

const std::function<size_t(const pn_binary_marking::Ptr&)> pn_binary_marking::hasher = [](const pn_binary_marking::Ptr& m)
{
	return m->hash();
};

const std::function<bool(const pn_binary_marking::Ptr&, const pn_binary_marking::Ptr&)> pn_binary_marking::eq = [](const pn_binary_marking::Ptr& lhs, const pn_binary_marking::Ptr& rhs)
{
	return lhs && rhs && *lhs == *rhs;
};

pn_binary_marking::pn_binary_marking(std::weak_ptr<pn_net> net,
	std::set<pn_instance> distribution)
	: net(std::move(net)),
	distribution(std::move(distribution)),
	cached_hash(0)
{
	for (const auto& entry : this->distribution)
	{
		size_t instance_hash = 0;
		boost::hash_combine(instance_hash, entry.first);
		boost::hash_combine(instance_hash, entry.second);
		cached_hash = cached_hash ^ instance_hash; // ensure commutativity w.r.t instance ordering
	}
}

bool pn_binary_marking::operator==(const pn_binary_marking& other) const
{
	return distribution == other.distribution;
}

bool pn_binary_marking::has(const pn_place::Ptr& place, const pn_token::Ptr& token) const
{
	return has(std::make_pair(place, token));
}

bool pn_binary_marking::has(const pn_token::Ptr& token, const pn_place::Ptr& place) const
{
	return has(std::make_pair(place, token));
}

bool pn_binary_marking::has(const pn_instance& instance) const
{
	return distribution.contains(instance);
}

std::set<pn_token::Ptr> pn_binary_marking::get_distribution(const pn_place::Ptr& place) const
{
	std::set<pn_token::Ptr> result;
	for (const auto& entry : distribution) // entry: (place,token)
		if (entry.first == place)
			result.emplace(entry.second);

	return result;
}

bool pn_binary_marking::is_occupied(const pn_place::Ptr& place) const
{
	return std::ranges::any_of(distribution, [&place](const auto& val)
		{
			return val.first == place && !std::dynamic_pointer_cast<pn_empty_token>(val.second);
		});
}

bool pn_binary_marking::is_output_occupied(const pn_transition::Ptr& transition) const
{
	for (const pn_transition::pn_arc& arc : transition->outputs)
	{
		// object is side condition or removed and replaced by empty token
		if (transition->is_side_condition(arc) || std::dynamic_pointer_cast<pn_empty_token>(arc.second))
			continue;

		/**
		 * Check if place already has a token
		 */
		if (is_occupied(arc.first))
			return true;

		/**
		 * Check if spatial place is already occupied
		 */
		const auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(arc.first);
		if (boxed_place && !boxed_place->overlapping_places.empty())
			for (const auto& overlapping : boxed_place->overlapping_places)
				if (is_occupied(overlapping.lock()))
					return true;
	}
	return false;
}

std::set<pn_place::Ptr> pn_binary_marking::get_distribution(const pn_token::Ptr& token) const
{
	std::set<pn_place::Ptr> result;
	for (const auto& entry : distribution) // entry: (place,token)
		if (entry.second == token)
			result.emplace(entry.first);

	return result;
}

bool pn_binary_marking::is_enabled(const pn_transition::Ptr& transition) const
{
	/**
	* Check if all inputs are present in the distribution
	*/
	for (const pn_transition::pn_arc& arc : transition->inputs)
	{
		if (!distribution.contains(arc))
			return false;
	}
	return !is_output_occupied(transition);
}

bool pn_binary_marking::would_enable(const pn_transition::Ptr& transition, const std::set<pn_instance>& additional_distribution) const
{
	/**
	 * Check if all inputs are present in the distributions
	 */
	for (const pn_transition::pn_arc& arc : transition->inputs)
	{
		if (!distribution.contains(arc)
			&& !additional_distribution.contains(arc))
			return false;
	}
	return !is_output_occupied(transition);
}

pn_binary_marking::Ptr pn_binary_marking::fire(const pn_transition::Ptr& transition) const
{
	if (!is_enabled(transition))
		throw std::invalid_argument("Transition not enabled");

	auto new_distribution(distribution);

	for (const pn_transition::pn_arc& arc : transition->inputs)
	{
		new_distribution.erase(arc);
	}

	new_distribution.insert(transition->outputs.cbegin(), transition->outputs.cend());

	return std::make_shared<pn_binary_marking>(net, std::move(new_distribution));
}

size_t pn_binary_marking::hash() const
{
	return cached_hash;
}

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_belief_marking
//
//
/////////////////////////////////////////////////////////////

pn_belief_marking::pn_belief_marking(std::weak_ptr<pn_net> net, marking_dist_t&& distribution)
	: net(std::move(net)),
	distribution(distribution)
{}

pn_belief_marking::pn_belief_marking(const pn_binary_marking::Ptr& marking)
	: net(marking->net),
	distribution({ {marking, 1.} })
{}

pn_belief_marking::pn_belief_marking(const pn_binary_marking::ConstPtr& marking)
	: pn_belief_marking(std::make_shared<pn_binary_marking>(marking->net, marking->distribution))
{}

double pn_belief_marking::get_probability(const pn_place::Ptr& place, const pn_token::Ptr& token) const
{
	return get_probability(std::make_pair(place, token));
}

double pn_belief_marking::get_probability(const pn_token::Ptr& token, const pn_place::Ptr& place) const
{
	return get_probability(std::make_pair(place, token));
}

double pn_belief_marking::get_probability(const pn_instance& instance) const
{
	double weight = 0;
	for (const auto& marking : distribution)
		if (marking.first->has(instance))
			weight += marking.second;

	return weight;
}

std::map<pn_token::Ptr, double> pn_belief_marking::get_distribution(const pn_place::Ptr& place) const
{
	std::map<pn_token::Ptr, double> result;
	for (const auto& marking : distribution)
		for (const auto& token : marking.first->get_distribution(place))
		{
			if (std::dynamic_pointer_cast<pn_empty_token>(token))
				continue; //TODO:: set to 1.0 before loop?

			const auto result_pair = result.try_emplace(token, marking.second);
			if (!result_pair.second)
				result_pair.first->second += marking.second;
		}
	return result;
}

std::map<pn_place::Ptr, double> pn_belief_marking::get_distribution(const pn_token::Ptr& token) const
{
	std::map<pn_place::Ptr, double> result;
	for (const auto& marking : distribution)
		for (const auto& place : marking.first->get_distribution(token))
		{
			const auto result_pair = result.try_emplace(place, marking.second);
			if (!result_pair.second)
				result_pair.first->second += marking.second;
		}
	return result;
}

double pn_belief_marking::get_summed_probability(const pn_place::Ptr& place) const
{
	double prob = 0.0;
	for (const auto& entry : get_distribution(place))
		prob += entry.second;

	return prob;
}

double pn_belief_marking::get_summed_probability(const pn_token::Ptr& token) const
{
	double prob = 0.0;
	for (const auto& entry : get_distribution(token))
		prob += entry.second;

	return prob;
}

double pn_belief_marking::is_enabled(const pn_transition::Ptr& transition) const
{
	double weight = 0;
	for (const auto& marking : distribution)
		if (marking.first->is_enabled(transition))
			weight += marking.second;

	return weight;
}

double pn_belief_marking::would_enable(const pn_transition::Ptr& transition, const std::set<pn_instance>& additional_distribution) const
{
	double weight = 0;
	for (const auto& marking : distribution)
		if (marking.first->would_enable(transition, additional_distribution))
			weight += marking.second;

	return weight;
}

pn_belief_marking::Ptr pn_belief_marking::fire(const pn_transition::Ptr& transition) const
{
	marking_dist_t new_distribution;

	double weight = 0;
	for (const auto& marking : distribution)
	{
		if (!marking.first->is_enabled(transition))
			continue;

		try
		{
			auto new_marking = marking.first->fire(transition);
			const auto it = new_distribution.try_emplace(new_marking, marking.second);
			if (!it.second)
				it.first->second += marking.second;

			weight += marking.second;
		}
		catch (const std::exception&)
		{
			// skip hypothesis that cannot fire transition
		}
	}

	if (new_distribution.empty())
		throw std::invalid_argument("transition not enabled");

	for (auto& val : new_distribution | std::views::values)
		val /= weight;

	return std::make_shared<pn_belief_marking>(net, std::move(new_distribution));
}

pn_marking::Ptr pn_belief_marking::to_marking() const
{
	std::map<pn_instance, double> new_distribution;
	for (const auto& marking : distribution)
		for (const auto& instance : marking.first->distribution)
		{
			if (std::dynamic_pointer_cast<pn_empty_token>(instance.second))
			{
				//TODO:: check
				//new_distribution.emplace(instance, marking.second);
				continue;
			}
			const auto iter = new_distribution.try_emplace(instance, marking.second);
			if (!iter.second)
				iter.first->second += marking.second;
		}

	return std::make_shared<pn_marking>(net, std::move(new_distribution));
}

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_net
//
//
/////////////////////////////////////////////////////////////

pn_net::pn_net()
	:
	object_params(nullptr)
{}

pn_net::pn_net(const object_parameters& object_params)
	: object_params(&object_params)
{}

pn_place::Ptr pn_net::create_place(bool agent)
{
	places.push_back(std::make_shared<pn_place>());
	if (agent)
		agent_places.emplace(places.back());

	return places.back();
}

pn_transition::Ptr pn_net::create_transition(std::vector<pn_transition::pn_arc>&& inputs, std::vector<pn_transition::pn_arc>&& outputs)
{
	pn_transition::Ptr trans(std::make_shared<pn_transition>(std::move(inputs), std::move(outputs)));

	add_transition(trans);

	return trans;
}

void pn_net::add_transition(const pn_transition::Ptr& trans)
{
	const auto& other_transitions = trans->inputs.empty() ? trans->outputs.begin()->first->get_incoming_transitions() : trans->inputs.begin()->first->get_outgoing_transitions();

	if (std::ranges::any_of(other_transitions, [&trans](const auto& t_weak)
		{
			auto t = t_weak.lock();
			return t->inputs == trans->inputs && t->outputs == trans->outputs;
		}))
		return;

	transitions.push_back(trans);

	for (const auto& entry : trans->inputs)
	{
		entry.first->outgoing_transitions.push_back(trans);
		if (tokens.insert(entry.second).second)
			tokens_vec.push_back(entry.second);
	}

	for (const auto& entry : trans->outputs)
	{
		entry.first->incoming_transitions.push_back(trans);
		if (tokens.insert(entry.second).second)
			tokens_vec.push_back(entry.second);
	}
}

pn_boxed_place::Ptr pn_net::get_place(const obb& box) const noexcept
{
	// create distinct places for every building - even if they are identical
	// otherwise one can get valid markings with holes in the buildings
	//for (auto place : places)
	//{
	//	auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place);

	//	if (!boxed_place)
	//		continue;

	//	const auto& p_box = boxed_place->box;

	//	if ((box.translation - p_box.translation).norm() > object_params->min_object_height)
	//		continue;

	//	if ((box.diagonal - p_box.diagonal).lpNorm<1>() > object_params->min_object_height)
	//		continue;

	//	// ignore rotation differences since boxes of composed buildings are rotated by 90°
	//	//if (box.rotation.angularDistance(p_box.rotation) > 0.1f)
	//	//	continue;

	//	return boxed_place;
	//}

	return nullptr;
}

void pn_net::add_place(const pn_place::Ptr& place, bool agent)
{
	auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place);

	if (!boxed_place)
		return;

	const auto& box = boxed_place->box;
	Eigen::Affine3f transform = Eigen::Translation3f(box.translation) * box.rotation * Eigen::Scaling(box.diagonal);
	std::vector<Eigen::Vector3f> points;
	points.reserve(5 * 5 * 5);
	float step = 1.f / 6.f;

	for (float x = -0.5f + step; x < 0.5f - step / 2; x += step)
		for (float y = -0.5f + step; y < 0.5f - step / 2; y += step)
			for (float z = -0.5f + step; z < 0.5f - step / 2; z += step)
				points.push_back(transform * Eigen::Vector3f(x, y, z));

	Eigen::AlignedBox3f unit_box(Eigen::Vector3f(-0.5f, -0.5f, -0.5f), Eigen::Vector3f(0.5f, 0.5f, 0.5f));

	for (auto other : places)
	{
		auto other_boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(other);

		if (!other_boxed_place)
			continue;

		const auto& other_box = other_boxed_place->box;

		Eigen::Affine3f inv_transform = (Eigen::Translation3f(other_box.translation) * other_box.rotation * Eigen::Scaling(other_box.diagonal)).inverse();

		bool intersects = false;
		for (const auto& point : points)
		{
			auto p = inv_transform * point;
			if (unit_box.contains(p))
			{
				intersects = true;
				break;
			}
		}

		if (!intersects)
			continue;

		/*
#ifdef _DEBUG
		float distance = (boxed_place->box.translation - other_boxed_place->box.translation).norm();
		std::cout
			<< "--------------------------------------------------------------------------------------------------" << std::endl
			<< "[top_z_0]: " << boxed_place->top_z << " \t "
			<< "[]: "
			<< boxed_place->top_z - boxed_place->box.diagonal.z() - other_boxed_place->top_z << std::endl
			<< "[top_z_1]: " << other_boxed_place->top_z << " \t "
			<< "[]: "
			<< other_boxed_place->top_z - other_boxed_place->box.diagonal.z() - boxed_place->top_z << std::endl
			<< "[distance]: " << distance << std::endl
			<< "[min_object_height]: " << object_params->min_object_height << std::endl
			<< "[dist+min]: " << distance + object_params->min_object_height << std::endl
			<< "--------------------------------------------------------------------------------------------------" << std::endl;
#endif
*/
		other_boxed_place->overlapping_places.emplace(pn_boxed_place::WPtr(boxed_place));
		boxed_place->overlapping_places.emplace(pn_boxed_place::WPtr(other_boxed_place));

	}

	places.push_back(place);

	if (agent)
		agent_places.emplace(place);
}

void pn_net::add_token(const pn_token::Ptr& token)
{
	if (tokens.contains(token))
		return;
	tokens.emplace(token);
	tokens_vec.emplace_back(token);
}

const std::vector<pn_place::Ptr>& pn_net::get_places() const
{
	return places;
}

const std::set<pn_place::Ptr>& pn_net::get_agent_places() const
{
	return agent_places;
}

const std::vector<pn_transition::Ptr>& pn_net::get_transitions() const
{
	return transitions;
}

const std::vector<pn_transition::Ptr> pn_net::get_meta_transitions() const
{
	std::lock_guard<std::mutex> lock(mutex);
	std::vector<pn_transition::Ptr> result;

	for (const auto& transition : transitions)
	{
		//check if invalid
		if (std::ranges::any_of(transition->inputs, [&transition](const auto& input)
			{
				return !transition->is_side_condition(input) && std::dynamic_pointer_cast<pn_boxed_place>(input.first);
			}) 
			|| 
			std::ranges::any_of(transition->outputs, [&transition](const auto& output)
				{
					return !transition->is_side_condition(output) && std::dynamic_pointer_cast<pn_boxed_place>(output.first);
				})
			)
			continue;

		//add if valid
		result.push_back(transition);
	}
	return result;
}

const std::vector<pn_token::Ptr>& pn_net::get_tokens() const
{
	return tokens_vec;
}

const std::set<pn_transition::Ptr> pn_net::get_forward_transitions(const pn_place::Ptr& agent) const
{
	if (!agent_places.contains(agent))
		throw std::invalid_argument("Place passed to pn_net::get_forward_transitions must be an agent place.");

	if (!get_goal())
		return {};



	std::lock_guard<std::mutex> lock(mutex);


	pn_token::Ptr empty_token = nullptr;
	for (const auto& token : tokens)
		if (std::dynamic_pointer_cast<pn_empty_token>(token))
		{
			empty_token = token;
			break;
		}

	std::set<pn_transition::Ptr> forward_transitions;
	std::set<pn_token::Ptr> tokens;
	const auto& goal_instances = get_goal_instances();

	for (const auto& instance : goal_instances)
	{
		auto place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);
		auto token = std::dynamic_pointer_cast<pn_object_token>(instance.second);

		if (!place || !token)
			continue;

		for (const auto& transition_w : instance.first->get_incoming_transitions())
		{
			auto transition = transition_w.lock();
			const auto& inputs = transition->get_inputs();

			if (!transition->has_output_arc(instance))
				continue;

			if (std::ranges::find(inputs, agent) == inputs.end())
				continue;

			if (transition->is_side_condition(instance))
				continue;

			forward_transitions.emplace(transition);
			tokens.emplace(token);
		}

	}

	for (const auto& transition2_w : agent->get_incoming_transitions())
	{
		auto transition2 = transition2_w.lock();
		
		bool add = false;

		for (const auto& out : transition2->outputs)
		{
			if (out.first != agent)
				continue;

			if (tokens.contains(out.second))
			{
				add = true;
				break;
			}
		}

		if (!add)
			continue;

		for (const auto& in : transition2->inputs)
		{
			if (transition2->is_side_condition(in) || in.second == empty_token)
				continue;

			if (goal_instances.contains(in))
			{
				add = false;
				break;
			}
		}

		if (add)
			forward_transitions.emplace(transition2);
	}

	return forward_transitions;
}

const pn_place::Ptr& pn_net::get_place(size_t index) const
{
	return places.at(index);
}

const pn_transition::Ptr& pn_net::get_transition(size_t index) const
{
	return transitions.at(index);
}

const pn_token::Ptr& pn_net::get_token(size_t index) const
{
	return tokens_vec.at(index);
}

void pn_net::set_goal(const pn_place::Ptr& place)
{
	goal = place;

	for (const pn_place::Ptr& p : places)
		if (p == place)
			return;

	places.push_back(place);
}

pn_place::Ptr pn_net::get_goal() const
{
	return goal;
}

std::set<pn_instance> pn_net::get_goal_instances() const
{
	//std::lock_guard<std::mutex> lock(mutex);
	//std::set<pn_object_instance> result;

	//for (const auto& instance : get_goal()->get_incoming_transitions().begin()->lock()->get_side_conditions())
	//{
	//	auto p = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);
	//	auto t = std::dynamic_pointer_cast<pn_object_token>(instance.second);

	//	if (p && t)
	//		result.emplace(p, t);
	//}

	//return result;

	return get_goal()->get_incoming_transitions().begin()->lock()->get_side_conditions();
}

void pn_net::integrate(pn_net&& other, bool optional_goal)
{
	const std::vector<pn_transition::Ptr> transition_prototypes = std::move(other.transitions);
	std::set<pn_place::Ptr> new_places;

	for (const auto& place : other.places)
	{
		place->incoming_transitions.clear();
		place->outgoing_transitions.clear();
	}

	// process not agent related transitions
	for (const auto& transition_prototype : transition_prototypes)
	{
		std::vector<pn_instance> inputs, outputs;

		bool skip = false;

		auto process_arc = [&skip, &new_places, &other](const std::set<pn_instance>& arc_prototypes, std::vector<pn_instance>& arcs)
		{
			for (const auto& input : arc_prototypes)
			{
				//TODO:: ask if changes are valid if agent is discovered later on
				// skip place agents
				if (other.agent_places.contains(input.first))
				{
					skip = true;
					break;
				}
				arcs.push_back(input);
				new_places.emplace(input.first);
			}
		};

		process_arc(transition_prototype->inputs, inputs);
		if (skip)
			continue;

		process_arc(transition_prototype->outputs, outputs);
		if (skip)
			continue;

		create_transition(std::move(inputs), std::move(outputs));
	}

	// recreate each transitions by mapping each agent from @param{place} to each agent of this
	for (const auto& agent_prototype : other.agent_places)
		for (const auto& agent : agent_places)
			for (const auto& transition_prototype : transition_prototypes)
			{
				std::vector<pn_instance> inputs, outputs;

				bool skip = false;
				bool agent_included;

				auto process_arc = [&skip, &agent_included, &agent_prototype, &agent, &new_places, &other](const std::set<pn_instance>& arc_prototypes, std::vector<pn_instance>& arcs)
				{
					for (const auto& input : arc_prototypes)
					{
						// skip place agents
						if (input.first == agent_prototype)
						{
							arcs.emplace_back(agent, input.second);
							agent_included = true;
						}
						else if (other.agent_places.contains(input.first))
						{
							skip = true;
							break;
						}
						else
						{
							arcs.push_back(input);
							new_places.emplace(input.first);
						}
					}
				};

				process_arc(transition_prototype->inputs, inputs);
				if (skip)
					continue;

				process_arc(transition_prototype->outputs, outputs);
				if (skip || !agent_included)
					continue;

				create_transition(std::move(inputs), std::move(outputs));
			}

	for (const auto& place : new_places)
		add_place(place);
}

void pn_net::print_benchmark_state(const pn_emission& emissions,
	const pn_belief_marking& marking)
{
	const auto& net = marking.net.lock();

	float prev_height = 0.f;
	const int agent_count = net->get_agent_places().size();
	const auto& places = net->get_places();

	std::vector<std::string> marking_pool(10, "                    ");
	std::vector<std::string> emission_pool(marking_pool);

	std::vector<std::string> marking_structure;
	std::vector<std::string> emission_structure;

	auto get_m_char = [&](const pn_place::Ptr& p)
	{
		double prob = marking.get_summed_probability(p);
		if (prob < 0.1)
			return '_';
		else if (prob < 0.9)
			return 'x';
		else
			return 'X';
	};

	auto get_e_char = [&](const pn_place::Ptr& p)
	{
		if (emissions.empty_places.contains(p))
			return '_';
		if (emissions.unobserved_places.contains(p))
			return '0';
		return 'X';
	};

	// create ASCII representation of pool
	for (int i = agent_count; i < 33 + agent_count; i++)
		if (auto place = std::dynamic_pointer_cast<pn_boxed_place>(places.at(i)))
		{
			int x = std::round(10 * place->box.translation.x());
			int y = std::round(10 + 10 * place->box.translation.y());

			emission_pool.at(x).at(y) = get_e_char(place);
			marking_pool.at(x).at(y) = get_m_char(place);
		}

	// create ASCII representation of structure
	std::string m_line;
	std::string e_line;
	for (int i = 33 + agent_count; i < places.size(); i++)
		if (auto place = std::dynamic_pointer_cast<pn_boxed_place>(places.at(i)))
		{
			if (place->top_z != prev_height)
			{
				prev_height = place->top_z;
				marking_structure.insert(marking_structure.begin(), m_line);
				emission_structure.insert(emission_structure.begin(), e_line);
				m_line = e_line = "";
			}

			m_line += get_m_char(place);
			e_line += get_e_char(place);
		}

	marking_structure.insert(marking_structure.begin(), m_line);
	emission_structure.insert(emission_structure.begin(), e_line);

	auto print = [](const std::vector<std::string>& vec)
	{
		for (const auto& str : vec)
			std::cout << str << std::endl;
	};

	std::cout.clear();
	std::cout << "Current emission" << std::endl;
	print(emission_pool);
	print(emission_structure);
	std::cout << "Agents";
	for (const auto& p : net->get_agent_places())
		std::cout << ", " << get_e_char(p);
	std::cout << "\n\n";

	std::cout << "Current marking" << std::endl;
	print(marking_pool);
	print(marking_structure);
	for (const auto& p : net->get_agent_places())
		std::cout << ", " << get_m_char(p);
	std::cout << std::endl;
}

const std::vector<std::weak_ptr<pn_transition>>& pn_place::get_incoming_transitions() const
{
	return incoming_transitions;
}

const std::vector<std::weak_ptr<pn_transition>>& pn_place::get_outgoing_transitions() const
{
	return outgoing_transitions;
}

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_emission
//
//
/////////////////////////////////////////////////////////////

pn_emission::pn_emission(std::set<pn_place::Ptr>&& empty_places,
	std::set<pn_place::Ptr>&& unobserved_places,
	std::map<pn_instance, double>&& token_distribution,
	std::map<pn_place::Ptr, double>&& max_probabilities)
	: empty_places(empty_places),
	unobserved_places(unobserved_places),
	token_distribution(token_distribution),
	max_probabilities(max_probabilities)
{}

bool pn_emission::is_empty(const pn_place::Ptr& place) const
{
	return empty_places.contains(place);
}

bool pn_emission::is_empty(const pn_instance& instance) const
{
	return is_empty(instance.first);
}

bool pn_emission::is_unobserved(const pn_place::Ptr& place) const
{
	return unobserved_places.contains(place);
}

bool pn_emission::is_unobserved(const pn_instance& instance) const
{
	return is_unobserved(instance.first);
}

double pn_emission::get_probability(const pn_instance& instance) const
{
	if (const auto iter = token_distribution.find(instance); iter != token_distribution.end())
		return iter->second;

	return 0;
}

/////////////////////////////////////////////////////////////
//
//
//  Class: transition_sequence
//
//
/////////////////////////////////////////////////////////////

transition_sequence::transition_sequence(pn_binary_marking::Ptr initial_marking,
	pn_binary_marking::Ptr final_marking,
	std::vector<pn_transition::Ptr> sequence)
	: initial_marking(std::move(initial_marking)),
	final_marking(std::move(final_marking)),
	sequence(std::move(sequence))
{}

} // namespace state_observation

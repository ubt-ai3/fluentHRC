#include "agent_manager.hpp"

#include <ranges>

#include <enact_core/id.hpp>
#include <enact_core/access.hpp>
#include <enact_core/data.hpp>
#include <enact_core/lock.hpp>

#include "enact_priority/signaling_actor.hpp"
#include "simulation/task.hpp"
#include "state_observation/object_tracking.hpp"

using namespace state_observation;
using namespace hand_pose_estimation;

namespace prediction
{

/////////////////////////////////////////////////////////////
//
//
//  Class: instance_manipulation
//
//
/////////////////////////////////////////////////////////////

instance_manipulation::instance_manipulation(const std::pair<state_observation::pn_transition::Ptr, double>& transition,
	std::chrono::duration<float> timestamp)
	:
	summed_probability(transition.second),
	timestamp(timestamp)
{
	probabilities.insert(transition);
}

void instance_manipulation::add(const std::pair<state_observation::pn_transition::Ptr, double>& transition, std::chrono::duration<float> timestamp)
{
	summed_probability += transition.second;
	this->timestamp = std::max(timestamp, this->timestamp);

	const auto iter = probabilities.find(transition.first);
	if (iter == probabilities.end())
		probabilities.insert(transition);
	else
		iter->second += transition.second;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: agent_manager
//
//
/////////////////////////////////////////////////////////////



agent_manager::agent_manager(enact_core::world_context& world,
	pn_net::Ptr net,
	const state_observation::computed_workspace_parameters& workspace_params,
	unsigned int max_agents,
	std::chrono::high_resolution_clock::time_point start_time)
	:
	max_agents(std::max(static_cast<unsigned int>(net->get_agent_places().size()), max_agents)),
	world(world),
	net(std::move(net)),
	workspace_params(workspace_params),
	start_time(start_time)
{
	const unsigned int count_agents = this->net->get_agent_places().size();
	if (max_agents > count_agents)
	{
		pn_net agent_net(*this->net->object_params);
		for (unsigned int i = count_agents; i < max_agents; ++i)
			agent_net.create_place(true);

		this->net->integrate(std::move(agent_net));
	}

	unused_agent_places = this->net->get_agent_places();
}


agent_manager::agent_manager(enact_core::world_context& world,
	pn_net::Ptr net,
	const state_observation::computed_workspace_parameters& workspace_params,
	const std::vector<pn_place::Ptr>& agent_places,
	std::chrono::high_resolution_clock::time_point start_time)
	:
	max_agents(agent_places.size()),
	world(world),
	net(std::move(net)),
	workspace_params(workspace_params),
	unused_agent_places(agent_places.begin(), agent_places.end()),
	start_time(start_time)
{
}

agent_manager::~agent_manager()
{
	//	stop_thread();
}

void agent_manager::update(const state_observation::pn_emission::ConstPtr& emission)
{
	this->emission = emission;
}

void agent_manager::update(const entity_id& id, enact_priority::operation op)
{
	std::lock_guard<std::mutex> lock(update_mutex);
	pending_updates.emplace_back(id, op);
}

void agent_manager::update_agents(
	std::map<observed_agent::Ptr, std::set<pn_transition::Ptr>> executed_transitions,
	const state_observation::pn_belief_marking::ConstPtr& prev_marking,
	const state_observation::pn_belief_marking::ConstPtr& current_marking, std::chrono::duration<float> timestamp)
{
	std::list<pn_belief_marking::ConstPtr> marking_sequence;
	marking_sequence.emplace_back(prev_marking);
	const auto last_entry = marking_sequence.emplace(marking_sequence.end(), current_marking);

	std::list<std::pair<observed_agent::Ptr, pn_transition::Ptr>> all_executed_transitions;

	for (auto& agent_transitions : executed_transitions) {
		agent_transitions.first->remove_double_detections(agent_transitions.second);
		
		for (const auto& transition : agent_transitions.second)
			all_executed_transitions.emplace_back(agent_transitions.first, transition);

	}

	// find the first (i.e. with the least number of other transitions applied) marking that enables the transition
	size_t untested_transitions = all_executed_transitions.size();
	while (!all_executed_transitions.empty() && untested_transitions)
	{
		auto agent_transition = all_executed_transitions.front();
		all_executed_transitions.pop_front();

		bool found_marking = false;
		for (const auto& marking : marking_sequence)
		{
			if (marking->is_enabled(agent_transition.second) > 0)
			{
				marking_sequence.emplace(last_entry, marking->fire(agent_transition.second)); // add intermediate marking state before current_marking

				if (!found_marking) {
					agent_transition.first->add_transition(agent_transition.second, *marking, timestamp);

					untested_transitions = all_executed_transitions.size();

					found_marking = true;
				}
			}
		}

		if (!found_marking)
		{
			untested_transitions--;
			all_executed_transitions.emplace_back(agent_transition);
		}
	}
}

void agent_manager::update(const std::map<state_observation::pn_transition::Ptr, double>& transition_probabilities,
	const pn_belief_marking::ConstPtr& prev_marking,
	const pn_belief_marking::ConstPtr& current_marking,
	std::chrono::duration<float> timestamp)
{
	if (transition_probabilities.empty())
		return;

	std::lock_guard<std::mutex> lock(update_mutex);

	for (const auto& entry : transition_probabilities)
		store_transition(entry, timestamp);

	std::map<observed_agent::Ptr, std::set<pn_transition::Ptr>> executed_transitions;
	{
		std::lock_guard<std::mutex> lock(agent_mutex);
		for (const auto& agent : agents | std::views::values)
			executed_transitions.emplace(agent, std::set<pn_transition::Ptr>());
	}
	process_instances(consumed_instances, executed_transitions, timestamp);
	process_instances(produced_instances, executed_transitions, timestamp);

	update_agents(std::move(executed_transitions), prev_marking, current_marking, timestamp);

	if (!emission)
		return;

	std::lock_guard<std::mutex> net_lock(net->mutex);
	for (const pn_place::Ptr& place : net->get_places())
		if (!emission->is_unobserved(place))
		{
			auto iter = last_seen.find(place);
			if (iter == last_seen.end())
				last_seen.emplace(place, timestamp);
			else
				iter->second = timestamp;
		}
}

void agent_manager::process_instances(std::map<pn_instance, instance_manipulation>& container,
	std::map<observed_agent::Ptr, std::set<pn_transition::Ptr>>& executed_transitions,
	std::chrono::duration<float> timestamp)
{
	std::map<pn_place::Ptr, observed_agent::Ptr> place_to_id;
	{
		std::lock_guard<std::mutex> lock(agent_mutex);
		for (const auto& agent : agents | std::views::values)
			place_to_id.emplace(agent->model_hand, agent);
	}
	
	for (auto iter = container.begin(); iter != container.end();)
	{
		const auto& info = iter->second;

		auto last_seen_iter = last_seen.find(iter->first.first);
		std::chrono::duration<float> place_last_seen = last_seen_iter == last_seen.end()
			? std::chrono::duration<float>(0)
			: last_seen_iter->second;


		if (info.summed_probability > 0.7)
		{
			double max = 0;
			observed_agent::Ptr best_agent;
			pn_transition::Ptr best_transition;
			auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(iter->first.first);

			if (!boxed_place)
				throw std::exception("Boxed place expected");

			for (const auto& entry : info.probabilities)
			{
				float closest_distance = std::numeric_limits<float>::infinity();

				const auto& agent = instance_manipulation::get_agent(*net, *entry.first);
				auto agent_iter = place_to_id.find(agent);
				if (agent_iter == place_to_id.end())
					continue;

				const auto& id = agent_iter->second->tracked_hand;

				enact_core::lock l(world, enact_core::lock_request(id, hand_trajectory::aspect_id, enact_core::lock_request::read));
				const enact_core::const_access<hand_trajectory_data> access_object(l.at(id, hand_trajectory::aspect_id));
				auto& obj = access_object->payload;

				for (const auto& pose : std::ranges::reverse_view(obj.poses))
				{
					if (pose.first + observed_agent::proximity_forget_duration < place_last_seen)
						break;

					closest_distance = std::min(closest_distance, agent_iter->second->get_distance(pose.second, *boxed_place));
				}

				const float certainty = transition_context::bell_curve(closest_distance, boxed_place->box.diagonal.norm()) * entry.second;

				if (certainty > max)
				{
					max = certainty;
					best_agent = agent_iter->second;
					best_transition = entry.first;
				}
			}

			if (best_agent)
			{
				executed_transitions.at(best_agent).emplace(best_transition);
			}

			iter = container.erase(iter);
			continue;
		}

		if (info.timestamp + observed_agent::proximity_forget_duration < place_last_seen)
			iter = container.erase(iter);
		else
			++iter;
	}
}

observed_agent::Ptr agent_manager::add(const entity_id& id, const state_observation::pn_place::Ptr& place)
{
	std::lock_guard<std::mutex> lock(agent_mutex);
	auto agent = std::make_shared<observed_agent>(world, net, workspace_params, id, place, start_time);
	agents.emplace(id, agent);
	unused_agent_places.erase(place);

	return agent;
}

void agent_manager::finished(std::chrono::high_resolution_clock::time_point start)
{
	std::unique_lock<std::mutex> lock(agent_mutex);
	auto agents_temp = agents;
	lock.release();
	
	for (const auto& agent : agents | std::views::values)
		agent->finished(start);
}

std::set<pn_transition::Ptr> agent_manager::get_blocked_transitions(bool use_tracking_data)
{
	if (!emission)
		throw std::exception("Update emission before calling get_blocked_transitions");

	std::vector<std::pair<entity_id, enact_priority::operation>> updates;

	{
		std::lock_guard<std::mutex> lock(update_mutex);
		std::swap(updates, pending_updates);
	}
	
	std::set<observed_agent::Ptr> deleted_agents;
	std::set<entity_id> candidate_ids;

	for (const auto& entry : updates) {
		const entity_id& id = entry.first;
		const enact_priority::operation op = entry.second;

		if (op == enact_priority::operation::DELETED)
		{
			std::lock_guard<std::mutex> lock(agent_mutex);
			auto iter = agents.find(id);

			if (iter != agents.end())
			{
				deleted_agents.emplace(agents.at(id));
				unused_agent_places.emplace(iter->second->model_hand);
				agents.erase(id);
				
			}

			continue;
		}


		if (op == enact_priority::operation::UPDATE)
		{
			std::lock_guard<std::mutex> lock(agent_mutex);
			auto iter = agents.find(id);
			if (iter == agents.end()) {
				candidate_ids.emplace(id);
			}
		}
		else if (op == enact_priority::operation::CREATE)
		{
			candidate_ids.emplace(id);
		}

	}

	{
		std::lock_guard<std::mutex> lock(agent_mutex);
		for (const auto& entry : agents)
			entry.second->update_action_candidates(emission->unobserved_places);
	}
	
	for (const entity_id& id : candidate_ids)
	{
		if (!unused_agent_places.empty()) // we have a spare agent place
		{
			std::unique_lock<std::mutex> lock(agent_mutex);
			auto agent = std::make_shared<observed_agent>(world, net, workspace_params, id, *unused_agent_places.begin());
			agents.emplace(id, agent);
			unused_agent_places.erase(unused_agent_places.begin());

			if (agents.size() == 2)
				std::cout << "Both hands registered." << std::endl;
		}
		else
		{
			std::cerr << "Trying to add more than 2 hands. Restart the program!" << std::endl;
		}
		
	}

	
	std::set<pn_transition::Ptr> blocked_transitions;

	std::unique_lock<std::mutex> lock(agent_mutex);
	// places that must not be changed by reasoning because they are under direct control, e.g. the robot
	std::set<pn_place::Ptr> controlled_agents = net->get_agent_places();

	if(use_tracking_data)
		for (const auto& agent : agents | std::views::values)
		{
			controlled_agents.erase(agent->model_hand);

			auto action_candidates = agent->get_action_candidates();

			for (const auto& transition : agent->model_hand->get_incoming_transitions())
				if (!action_candidates.contains(transition.lock()))
					blocked_transitions.emplace(transition);

			for (const auto& transition : agent->model_hand->get_outgoing_transitions())
				if (!action_candidates.contains(transition.lock()))
					blocked_transitions.emplace(transition);
		}
	else
		for (const auto& agent : agents | std::views::values)
			controlled_agents.erase(agent->model_hand);

	lock.unlock();
	std::lock_guard<std::mutex> net_lock(net->mutex);

	for (const auto& agent : controlled_agents)
	{
		for (const auto& transition : agent->get_incoming_transitions())
			blocked_transitions.emplace(transition);

		for (const auto& transition : agent->get_outgoing_transitions())
			blocked_transitions.emplace(transition);
	}

	return blocked_transitions;
}

std::set<observed_agent::Ptr> agent_manager::get_agents() const
{
	std::lock_guard<std::mutex> lock(agent_mutex);
	
	std::set<observed_agent::Ptr> result;
	for (const auto& agent : agents | std::views::values)
		result.emplace(agent);
	return result;
}

void agent_manager::reset(std::chrono::high_resolution_clock::time_point start)
{
	std::scoped_lock lock(agent_mutex, update_mutex);
	for (const auto& agent : agents | std::views::values)
		agent->finished(start);
}

void agent_manager::store_transition(const std::pair<state_observation::pn_transition::Ptr, double>& entry, std::chrono::duration<float> time_seconds)
{
	auto is_agent = [agent_places(net->get_agent_places())](const pn_place::Ptr& place)
	{
		return agent_places.contains(place);
	};

	const auto& transition = entry.first;
	pn_place::Ptr agent = nullptr;


	for (const auto& place : transition->get_inputs())
		if (is_agent(place))
		{
			agent = place;
			break;
		}

	if (!agent)
		for (const auto& place : transition->get_outputs())
			if (is_agent(place))
			{
				agent = place;
				break;
			}

	if (!agent)
		return;


	for (const auto& instance : transition->inputs)
	{
		if (transition->is_side_condition(instance))
			continue;

		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);

		if (!boxed_place || std::dynamic_pointer_cast<pn_empty_token>(instance.second))
			continue;

		auto iter = consumed_instances.find(instance);
		if (iter == consumed_instances.end())
			consumed_instances.emplace(instance, instance_manipulation(entry, time_seconds));
		else
			iter->second.add(entry, time_seconds);
	}

	for (const auto& instance : transition->outputs)
	{
		if (transition->is_side_condition(instance))
			continue;

		const auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);

		if (!boxed_place || std::dynamic_pointer_cast<pn_empty_token>(instance.second))
			continue;

		auto iter = produced_instances.find(instance);
		if (iter == produced_instances.end())
			produced_instances.emplace(instance, instance_manipulation(entry, time_seconds));
		else
			iter->second.add(entry, time_seconds);
	}

};



pn_place::Ptr instance_manipulation::get_agent(const pn_net& net, const pn_transition& transition)
{
	std::lock_guard<std::mutex> lock(net.mutex);

	auto is_agent = [agent_places(net.get_agent_places())](const pn_place::Ptr& place)
	{
		return agent_places.contains(place);
	};

	for (const auto& place : transition.get_inputs())
		if (is_agent(place))
			return place;

	for (const auto& place : transition.get_outputs())
		if (is_agent(place))
			return place;

	return nullptr;
}

}
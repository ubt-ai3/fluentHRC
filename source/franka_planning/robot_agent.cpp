#include "robot_agent.hpp"

#include <stack>

#include <intention_prediction/agent_manager.hpp>
#include <state_observation/pn_model_extension.hpp>

using namespace state_observation;
using namespace prediction;

namespace robot
{
action_planner::action_planner(const pn_place::Ptr& robot_place)
	:
	robot_place{ robot_place }
{}

std::string action_planner::get_class_name() const
{
	return "action_planner";
}

void action_planner::update_marking(const franka_agent& franka,
	state::pn_belief_marking::ConstPtr marking) noexcept
{
	if (marking == nullptr || marking->net.lock() == nullptr)
		return;

	net = marking->net.lock();
	std::unique_lock<std::mutex> net_lock(net->mutex);
	auto places = marking->net.lock()->get_places();
	net_lock.unlock();

	occupied_places.clear();

	for (const auto& place : places)
	{
		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place);

		if (!boxed_place)
			continue;

		if (marking->get_summed_probability(place) > 0.5)
			occupied_places.push_back(boxed_place);
	}

	this->marking = std::move(marking);
}


bool action_planner::gripper_collides(const pn_boxed_place::Ptr& target_place) const noexcept
{
	float gripper_bottom_z = target_place->box.top_z() + franka_agent::below_suction_cup_height;
	for (const auto& occupied : occupied_places)
	{
		const auto& box = occupied->box;

		if (box.top_z() < gripper_bottom_z)
			continue;

		for (const auto& corner : box.get_corners())
			// distance in x-y-plane
			if ((corner.head<2>() - target_place->box.translation.head<2>()).norm() < franka_agent::gripper_radius)
				return true;
	}

	return false;
}

state::pn_transition::Ptr agent::get_put_back_action() const noexcept
{
	if (marking->get_summed_probability(robot_place) < 0.5)
		return nullptr;

	// transition to resource places must come first in the vector
	for (const auto& action : robot_place->get_outgoing_transitions())
		if (marking->is_enabled(action.lock()) > 0.5)
			return action.lock();

	return nullptr;
}

pn_boxed_place::Ptr action_planner::get_target_place(const pn_transition::Ptr& transition) const noexcept
{
	if (auto action = std::dynamic_pointer_cast<stack_action>(transition))
	{
		return action->to.first;
	}
	else if (auto action = std::dynamic_pointer_cast<pick_action>(transition))
	{
		return action->from;

	}
	else if (auto action = std::dynamic_pointer_cast<place_action>(transition))
	{
		return action->to;
	}
	else if (auto action = std::dynamic_pointer_cast<reverse_stack_action>(transition))
	{
		return action->from.first;
	}


	return nullptr;
}

std::vector<state::pn_transition::Ptr> action_planner::get_feasible_actions(const pn_transition::Ptr& excluded) const noexcept
{
	std::vector<state::pn_transition::Ptr> result;

	auto goal_instances = net->get_goal_instances();

	for (const auto& transition : forward_transitions)
	{
		if (transition == excluded)
			continue;

		if (marking->is_enabled(transition) < 0.5)
			continue;

		pn_boxed_place::Ptr target_place = get_target_place(transition);

		if (!target_place)
			continue;

		if (gripper_collides(target_place))
			continue;

		result.emplace_back(transition);
	}

	return result;
}

std::vector<state::pn_transition::Ptr> action_planner::get_future_feasible_actions(const std::vector<state::pn_transition::Ptr>& next_actions) const noexcept
{
	std::vector<state::pn_transition::Ptr> result;

	auto goal_instances = net->get_goal_instances();

	std::set<pn_instance> postconditions;
	std::set<pn_transition::Ptr> next_actions_set;

	for (const auto& a : next_actions)
	{
		next_actions_set.emplace(a);
		for (const auto& inst : a->outputs)
			postconditions.emplace(inst);
	}

	for (const auto& transition : forward_transitions)
	{
		if (next_actions_set.contains(transition))
			continue;

		if (marking->would_enable(transition, postconditions) < 0.5)
			continue;

		pn_boxed_place::Ptr target_place = get_target_place(transition);

		if (!target_place)
			continue;

		if (gripper_collides(target_place))
			continue;


		result.emplace_back(transition);
	}

	return result;
}

std::vector<state::pn_transition::Ptr> action_planner::get_future_feasible_actions(const std::vector<pn_belief_marking::ConstPtr>& next_markings) const noexcept
{
	std::vector<state::pn_transition::Ptr> result;

	for (const auto& transition : forward_transitions)
	{
		pn_boxed_place::Ptr target_place = get_target_place(transition);

		if (!target_place)
			continue;

		if (gripper_collides(target_place))
			continue;

		bool is_enabled = false;
		for (const auto& m : next_markings)
			if (m->is_enabled(transition) > 0.5)
			{
				is_enabled = true;
				break;
			}

		if (is_enabled)
			result.emplace_back(transition);
	}

	return result;
}

std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> action_planner::next(const franka_agent& franka, const pn_transition::Ptr& excluded) noexcept
{
	const auto& actions = get_feasible_actions(excluded);
	if (actions.empty())
		return { nullptr, nullptr };
	else if (actions.size() == 1)
		return { actions.front(), nullptr };

	return select_action(actions, franka);
}

std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> action_planner::select_action(const std::vector<state::pn_transition::Ptr>& feasible_actions,
	const franka_agent& franka) noexcept
{
	for (const auto& action : feasible_actions)
	{
		auto next_marking = marking->fire(action);

		for (const auto& next_action : forward_transitions)
		{
			if (next_marking->is_enabled(next_action) > 0.5 && !gripper_collides(get_target_place(next_action)))
				return { action, next_action };
		}
	}

	return { feasible_actions.front(), nullptr };
}

void action_planner::compute_forward_transitions(const franka_agent& franka)
{
	forward_transitions.clear();

	for (const auto& transition : net->get_forward_transitions(robot_place))
		if (franka.can_execute_transition(*transition))
			forward_transitions.emplace(transition);
}



/////////////////////////////////////////////////////////////
//
//
//  Class: null_planner
//
//
/////////////////////////////////////////////////////////////

null_planner::null_planner(const pn_place::Ptr& robot_place)
	:
	action_planner(robot_place)
{}

std::string null_planner::get_class_name() const
{
	return "null_planner";
}
std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> null_planner::next(const state::franka_agent& franka, const state::pn_transition::Ptr& excluded) noexcept
{
	return { nullptr, nullptr };
}


/////////////////////////////////////////////////////////////
//
//
//  Class: planner_layerwise_rtl
//
//
/////////////////////////////////////////////////////////////

planner_layerwise_rtl::planner_layerwise_rtl(state::pn_place::Ptr robot_place)
	:
	action_planner(robot_place)
{}

std::string planner_layerwise_rtl::get_class_name() const
{
	return "planner_layerwise_rtl";
}

void planner_layerwise_rtl::compute_forward_transitions(const franka_agent& franka)
{
	action_planner::compute_forward_transitions(franka);

	place_order.clear();

	for (const auto& t : forward_transitions)
	{
		Eigen::Vector3f bottom;

		auto action = std::dynamic_pointer_cast<place_action>(t);
		if (action)
		{
			bottom = action->to->box.translation;
			bottom.z() = action->to->box.bottom_z();
		}
		else
		{
			auto action = std::dynamic_pointer_cast<stack_action>(t);

			if (!action)
				continue;

			bottom = action->to.first->box.translation;
			bottom.z() = action->to.first->box.bottom_z();
		}

		place_order.emplace_back(t, bottom);
	}


	auto blocks = [&](const pn_boxed_place::Ptr& blocker, const pn_boxed_place::Ptr& blocked)
	{

		float gripper_bottom_z = blocked->box.top_z() + franka.below_suction_cup_height;
		const auto& box = blocker->box;

		if (box.top_z() > gripper_bottom_z)
		{
			for (const auto& corner : box.get_corners())
				// distance in x-y-plane
				if ((corner.head<2>() - blocked->box.translation.head<2>()).norm() < franka.gripper_radius)
					return true;
		}

		return false;
	};

	std::ranges::sort(place_order, [&](auto& lhs, auto& rhs)
	{
		// check if rhs blocks lhs, then lhs must be first
		if (blocks(get_target_place(rhs.first), get_target_place(lhs.first)))
			return true;

		if (blocks(get_target_place(lhs.first), get_target_place(rhs.first)))
			return false;

		if (lhs.second.z() != rhs.second.z())
			return lhs.second.z() < rhs.second.z();

		if (lhs.second.y() != rhs.second.y())
			return lhs.second.y() > rhs.second.y();

		return lhs.second.x() < rhs.second.x();
	});

}

std::pair<pn_transition::Ptr, pn_transition::Ptr> planner_layerwise_rtl::select_action(const std::vector<state::pn_transition::Ptr>& feasible_actions, const franka_agent& franka) noexcept
{
	for (const auto& reference : place_order)
	{
		const auto& target = get_target_place(reference.first);
		if (gripper_collides(target) || marking->get_summed_probability(target) > 0.5)
			continue;		

		if (std::ranges::find(feasible_actions, reference.first) != feasible_actions.end())
			return { reference.first, nullptr };

		std::vector<std::tuple<Eigen::Vector3f, pn_transition::Ptr, pn_transition::Ptr>> candidates;


		for (const auto& t : feasible_actions)
		{
			if (!marking->would_enable(reference.first, t->outputs) || gripper_collides(get_target_place(t)))
				continue;

			Eigen::Vector3f center;

			auto action = std::dynamic_pointer_cast<pick_action>(t);
			if (action)
				center = action->from->box.translation;
			else
			{
				auto action = std::dynamic_pointer_cast<reverse_stack_action>(t);

				if (!action)
					continue;

				center = action->from.first->box.translation;
			}

			candidates.emplace_back(center, t, reference.first);
		}

		if (candidates.empty())
			continue;

		std::ranges::sort(candidates, [](auto& lhs, auto& rhs)
		{
			const auto& lbox = std::get<0>(lhs);
			const auto& rbox = std::get<0>(rhs);

			if (lbox.z() != rbox.z())
				return lbox.z() < rbox.z();

			if (lbox.y() != rbox.y())
				return lbox.y() > rbox.y();

			return lbox.x() < rbox.x();
		});

		const auto& [_, t0, t1] = candidates.front();
		return { t0, t1 };
	}

	return { nullptr, nullptr };
}

/////////////////////////////////////////////////////////////
//
//
//  Class: planner_adaptive
//
//
/////////////////////////////////////////////////////////////

planner_adaptive::planner_adaptive(prediction::agent_manager& agents,
	state::pn_place::Ptr robot_place)
	:
	planner_layerwise_rtl(std::move(robot_place)),
	agents(agents)
{}

std::string planner_adaptive::get_class_name() const
{
	return "planner_adaptive";
}

void planner_adaptive::update_marking(const state::franka_agent& franka, state::pn_belief_marking::ConstPtr marking) noexcept
{
	bool initialized = !!net;
	action_planner::update_marking(franka, marking);

	if (!net || initialized)
		return;

	try
	{
		// create dictionary for robot to human action
		auto agents = net->get_agent_places();

		for (const auto& agent : agents)
		{
			if (agent == robot_place)
				continue;

			action_dictionaries_h2r.emplace(agent, transition_map{});
			action_dictionaries_r2h.emplace(agent, transition_map{});
		}

		// handle incoming transitions
		{
			auto get_target = [](const pn_transition::Ptr& t)
			{
				if (auto a = std::dynamic_pointer_cast<pick_action>(t))
					return a->from;
				else if (auto a = std::dynamic_pointer_cast<reverse_stack_action>(t))
					return a->from.first;
				else
					return pn_boxed_place::Ptr{};
			};

			std::map<pn_boxed_place::Ptr, pn_transition::Ptr> targets;
			for (const auto& t : robot_place->get_incoming_transitions())
			{
				if (auto target = get_target(t.lock()))
					targets.emplace(target, t.lock());
			}

			for (const auto& agent : agents)
			{
				auto h2r_iter = action_dictionaries_h2r.find(agent);
				auto r2h_iter = action_dictionaries_r2h.find(agent);

				if (h2r_iter == action_dictionaries_h2r.end())
					continue;

				for (const auto& t : agent->get_incoming_transitions())
				{
					if (auto target = get_target(t.lock()))
					{
						auto iter = targets.find(target);
						if (iter != targets.end())
						{
							r2h_iter->second.emplace(iter->second, t);
							h2r_iter->second.emplace(t, iter->second);
						}
					}
				}
			}
		}


		// handle outgoing transitions
		{
			auto get_target = [](const pn_transition::Ptr& t)
			{
				if (auto a = std::dynamic_pointer_cast<place_action>(t))
					return a->to;
				else if (auto a = std::dynamic_pointer_cast<stack_action>(t))
					return a->to.first;
				else
					return pn_boxed_place::Ptr{};
			};

			std::map<pn_boxed_place::Ptr, pn_transition::Ptr> targets;
			for (const auto& t : robot_place->get_outgoing_transitions())
			{
				if (auto target = get_target(t.lock()))
					targets.emplace(target, t.lock());
			}

			for (const auto& agent : agents)
			{
				auto h2r_iter = action_dictionaries_h2r.find(agent);
				auto r2h_iter = action_dictionaries_r2h.find(agent);

				if (h2r_iter == action_dictionaries_h2r.end())
					continue;

				for (const auto& t : agent->get_outgoing_transitions())
				{
					if (auto target = get_target(t.lock()))
					{
						auto iter = targets.find(target);
						if (iter != targets.end())
						{
							r2h_iter->second.emplace(iter->second, t);
							h2r_iter->second.emplace(t, iter->second);
						}
					}
				}
			}
		}

	}
	catch (...) {}
}


bool planner_adaptive::prioritize_place_target(const std::vector<state::pn_transition::Ptr>& actions) const noexcept
{
	const auto& goals = net->get_goal_instances();
	auto a = std::dynamic_pointer_cast<pick_action>(actions.front());
	return a != nullptr && !goals.contains(*a->inputs.begin());
}

std::vector<planner_adaptive::transition_probability> planner_adaptive::predict_consecutive(const std::vector<state::pn_transition::Ptr>& robot_actions, const franka_agent& franka) const
{
	using queue_entry_t = std::pair<std::vector<pn_transition::Ptr>, double>;
	std::vector<transition_probability> feasible_action_weight{ 2 };

	auto local_marking = marking; // avoid race conditions
	const std::vector<pn_transition::Ptr> meta_transitions = local_marking->net.lock()->get_meta_transitions();

	for (const auto& agent : agents.get_agents())
	{
		auto r2h_iter = action_dictionaries_r2h.find(agent->model_hand);

		int lookahead = 2;
		std::vector<pn_transition::Ptr> fake_history;
		if (agent->has_object_grabbed() != franka.has_object_gripped())
		{
			const auto& candidates = agent->get_executable_actions(*marking);
			// ensure that the most recent action is a pick in case we want to evaluate place actions - and vice versa. 
			// Other mismatches originating from choosing some actions as a predecessor are not so relevant
			if (!candidates.empty())
			{
				fake_history.emplace_back(candidates.front().transition);
				lookahead++;
			}
		}

		try
		{
			double total_weight = 1.;
			constexpr double weight_threshold_to_abort = 0.9;
			double added_weight = 0.;
			static constexpr double infeasible_multiplier = 0.001;
			double max_weight = 0.;

			std::vector<std::map<pn_transition::Ptr, double>> future_transitions(lookahead);
			auto add = [&](const std::vector<pn_transition::Ptr>& transitions, double w)
			{
				for (int i = 0; i < transitions.size(); i++)
				{
					auto iter = future_transitions.at(i).find(transitions.at(i));
					if (iter == future_transitions.at(i).end())
						future_transitions.at(i).emplace(transitions.at(i), w);
					else
						iter->second += w;
				}
				added_weight += w;
				max_weight = std::max(max_weight, w);
			};

			auto comp = [](const prediction_context& lhs, const prediction_context& rhs) -> bool { return lhs.weight < rhs.weight; };
			std::priority_queue<prediction_context,
				std::vector<prediction_context>,
				decltype(comp)>
				queue(comp);


			queue.emplace(local_marking, fake_history);

			while (!queue.empty() && added_weight < weight_threshold_to_abort * total_weight)
			{
				auto pred_ctx = queue.top(); queue.pop();
				const auto& actions = pred_ctx.actions;

				if (actions.size() >= lookahead )
				{
					if (actions.size() >= 2 && actions[lookahead - 1]->reverses(*actions[lookahead - 2]))
					{
						total_weight -= (1 - infeasible_multiplier) * pred_ctx.weight;
						add(actions, infeasible_multiplier * pred_ctx.weight);
					}
					else
					{
						add(actions, pred_ctx.weight);
					}

					continue;
				}

				auto transition_candidates = lookahead - pred_ctx.actions.size() == 2 ?
					robot_actions :
					get_future_feasible_actions(std::vector<pn_belief_marking::ConstPtr>(1, pred_ctx.marking));

				std::vector<transition_context> candidates;
				candidates.reserve(transition_candidates.size());
				for (const auto& t : transition_candidates)
					candidates.emplace_back(agent->workspace_params, t, *local_marking, robot_place);

				if (candidates.empty())
				{
					bool feasible = false;
					for (const auto& t : meta_transitions)
					{
						if (pred_ctx.marking->is_enabled(t) > observed_agent::enable_threshold)
						{
							feasible = true;
							break;
						}
					}

					if (feasible)
					{
						add(pred_ctx.actions, pred_ctx.weight);
					}
					else
					{
						total_weight -= (1 - infeasible_multiplier) * pred_ctx.weight;
						add(pred_ctx.actions, infeasible_multiplier * pred_ctx.weight);
					}

					continue;
				}

				try
				{
					auto prediction = agent->predict(candidates, pred_ctx.actions, *pred_ctx.marking);


					for (const auto& pred : prediction)
					{
						std::vector<pn_transition::Ptr> new_actions(pred_ctx.actions);
						new_actions.push_back(pred.first);
						queue.emplace(pred_ctx.marking->fire(pred.first), std::move(new_actions), pred_ctx.weight * pred.second);
					}
				}
				catch (...) {}
			}

			//auto h2r_iter = action_dictionaries_h2r.find(agent->model_hand);
			auto process = [&](const decltype(future_transitions)::value_type& future_transition, auto& feasible_actions_w)
			{
				for (const auto& entry : future_transition)
				{
					if (entry.second < observed_agent::min_probability && entry.second < max_weight)
						continue;

					auto iter = feasible_actions_w.find(entry.first);

					if (iter == feasible_actions_w.end())
						iter = feasible_actions_w.emplace(entry.first, 0.).first;

					iter->second += entry.second / added_weight * (agent->get_executed_actions().size() + 1);
				}
			};

			process(*(++future_transitions.rbegin()), feasible_action_weight.front());
			process(future_transitions.back(), feasible_action_weight.back());

		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << std::endl;
		}
	}

	for (auto& entry : feasible_action_weight)
		observed_agent::normalize(entry);

	return feasible_action_weight;
}

// returns two nested transition probability maps - the outer one for the place action, the inner one for the next pick action
std::map<pn_transition::Ptr, std::pair<double, planner_adaptive::transition_probability>> planner_adaptive::predict_pick_based_on_place(const std::vector<state::pn_transition::Ptr>& robot_actions, const franka_agent& franka) const
{
	std::map<pn_transition::Ptr, std::pair<double, planner_adaptive::transition_probability>> result;
	std::map<pn_token::Ptr, std::vector<pn_transition::Ptr>> picked;
	std::map<pn_token::Ptr, std::vector<pn_transition::Ptr>> placed;

	const auto local_marking = marking; // avoid race conditions

	std::vector<pn_belief_marking::ConstPtr> future_markings;
	future_markings.reserve(robot_actions.size());
	for (const auto& a : robot_actions)
	{
		try
		{
			future_markings.emplace_back(local_marking->fire(a));
		}
		catch (...) {}
	}

	auto future_actions = get_future_feasible_actions(future_markings);
	if (future_actions.empty())
		return {};

	for (const auto& action : future_actions)
	{
		for (const auto& instance : action->inputs)
			if (instance.first == robot_place)
			{
				auto iter = placed.find(instance.second);
				if (iter == placed.end())
					iter = placed.emplace(instance.second, std::vector<pn_transition::Ptr>()).first;

				iter->second.push_back(action);

				break;
			}
	}

	for (const auto& action : robot_actions)
	{

		for (const auto& instance : action->outputs)
			if (instance.first == robot_place)
			{
				if (!placed.contains(instance.second))
					break;

				auto iter = picked.find(instance.second);
				if (iter == picked.end())
					iter = picked.emplace(instance.second, std::vector<pn_transition::Ptr>()).first;

				iter->second.push_back(action);

				break;
			}
	}


	for (const auto& agent : agents.get_agents())
	{
		//auto r2h_iter = action_dictionaries_r2h.find(agent->model_hand);

		transition_probability prediction;
		/*
		auto process = [&](const auto& future_transition, auto& feasible_actions_w, double weight = 1)
		{
			for (const auto& entry : future_transition)
			{
				auto iter = feasible_actions_w.find(entry.first);

				if (iter == feasible_actions_w.end())
					continue;

				iter->second += weight * entry.second * (agent->get_executed_actions().size() + 1);
			}
		};
		*/

		try
		{
			if (future_actions.size() == 1)
			{
				prediction.emplace(future_actions.front(), 1.);
			}
			else
			{

				std::vector<pn_transition::Ptr> fake_history;
				// since we skip an action the current state of grabbed objects must not match
				if (agent->has_object_grabbed() == franka.has_object_gripped())
				{
					const auto& candidates = agent->get_executable_actions(*marking);
					// ensure that the most recent action is a pick in case we want to evaluate place actions - and vice versa. 
					// Other mismatches originating from choosing some actions as a predecessor are not so relevant
					if (!candidates.empty())
						fake_history.emplace_back(candidates.front().transition);
				}

				std::vector<transition_context> candidates;
				candidates.reserve(future_actions.size());
				for (const auto& t : future_actions)
					candidates.emplace_back(agent->workspace_params, t, *local_marking, robot_place);

				prediction = agent->predict(candidates, fake_history, *local_marking);
				observed_agent::normalize(prediction);

				for (const auto& entry : prediction)
					result.emplace(entry.first, std::make_pair(entry.second, transition_probability{}));
			}


			std::vector<pn_transition::Ptr> fake_history;
			if (agent->has_object_grabbed() != franka.has_object_gripped())
			{
				const auto& candidates = agent->get_executable_actions(*marking);
				if (!candidates.empty())
					fake_history.emplace_back(candidates.front().transition);
			}

			for (const auto& entry : placed)
			{

				std::vector<transition_context> candidates;
				for (const auto& t : picked.at(entry.first))
					candidates.emplace_back(agent->workspace_params, t, *local_marking, robot_place);

				auto conditioned_prediction = agent->predict(candidates, fake_history, *local_marking);
				observed_agent::normalize(conditioned_prediction);
				
				for (const auto& t : entry.second)
				{
					auto iter = result.find(t);

					if (iter != result.end())
						iter->second.second = conditioned_prediction;
				}
			}

		}
		catch (...) {}
	}

	return result;
}

std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> planner_adaptive::select_action(const std::vector<state::pn_transition::Ptr>& feasible_actions, const franka_agent& franka) noexcept
{
	if (agents.get_agents().empty())
		return planner_layerwise_rtl::select_action(feasible_actions, franka);

	auto get_minimizer = [&](const transition_probability& map, pn_token::Ptr precondition = nullptr)
	{
		std::vector<pn_transition::Ptr> least_likely_actions;
		double  likelihood = std::numeric_limits<double>::infinity();

		for (const auto& entry : map)
		{
			bool valid = true;

			if (precondition)
				for (const auto& instance : entry.first->inputs)
					if (instance.first == robot_place)
					{
						valid = instance.second == precondition;
						break;
					}

			if (!valid)
				continue;

			if (entry.second < likelihood)
			{
				least_likely_actions = {};
				likelihood = entry.second;
			}

			if (entry.second == likelihood)
				least_likely_actions.push_back(entry.first);
		}

		if (least_likely_actions.empty())
			return pn_transition::Ptr();

		if (least_likely_actions.size() > 1)
			planner_layerwise_rtl::select_action(least_likely_actions, franka).first;

		return least_likely_actions.front();
	};


	try
	{
		if (prioritize_place_target(feasible_actions))
		{
			auto prediction = predict_pick_based_on_place(feasible_actions, franka);

			if (prediction.empty())
				return { nullptr, nullptr };

			auto least_likely_action = prediction.begin();


			for (auto iter = prediction.begin(); iter != prediction.end(); ++iter)
			{
				auto prob = iter->second.first;
				if (iter->second.first < least_likely_action->second.first)
					least_likely_action = iter;
			}

			return { get_minimizer(least_likely_action->second.second), least_likely_action->first };
		}

		const auto feasible_action_weight = predict_consecutive(feasible_actions, franka);

		auto least_likely_action = get_minimizer(feasible_action_weight.front());

		if (least_likely_action == nullptr)
			return planner_layerwise_rtl::select_action(feasible_actions, franka);

		pn_token::Ptr postcondition = nullptr;
		for (const auto& instance : least_likely_action->outputs)
			if (instance.first == robot_place)
			{
				postcondition = instance.second;
				break;
			}

		return { least_likely_action, get_minimizer(feasible_action_weight.back(), postcondition) };

	}
	catch (...)
	{
		return { nullptr, nullptr };
	}
}


/////////////////////////////////////////////////////////////
//
//
//  Class: planner_adversarial
//
//
/////////////////////////////////////////////////////////////

planner_adversarial::planner_adversarial(prediction::agent_manager& agents,
	state::pn_place::Ptr robot_place)
	:
	planner_adaptive(agents, std::move(robot_place))
{}

std::string planner_adversarial::get_class_name() const
{
	return "planner_adversarial";
}

std::pair<state::pn_transition::Ptr, state::pn_transition::Ptr> planner_adversarial::select_action(const std::vector<state::pn_transition::Ptr>& feasible_actions, const franka_agent& franka) noexcept
{
	if (agents.get_agents().empty())
		return planner_layerwise_rtl::select_action(feasible_actions, franka);

	auto get_maximizer = [&](const transition_probability& map, pn_token::Ptr precondition = nullptr)
	{
		std::vector<pn_transition::Ptr> most_likely_actions;
		double likelihood = -1;

		for (const auto& entry : map)
		{
			bool valid = true;

			if (precondition)
				for (const auto& instance : entry.first->inputs)
					if (instance.first == robot_place)
					{
						valid = instance.second == precondition;
						break;
					}

			if (!valid)
				continue;

			if (entry.second > likelihood)
			{
				most_likely_actions = {};
				likelihood = entry.second;
			}

			if (entry.second == likelihood)
				most_likely_actions.push_back(entry.first);
		}

		if (most_likely_actions.empty())
			return pn_transition::Ptr();

		if (most_likely_actions.size() > 1)
			planner_layerwise_rtl::select_action(most_likely_actions, franka).first;

		return most_likely_actions.front();
	};


	try
	{
		if (prioritize_place_target(feasible_actions))
		{
			auto prediction = predict_pick_based_on_place(feasible_actions, franka);

			if (prediction.empty())
				return { nullptr, nullptr };

			auto most_likely_action = prediction.begin();


			for (auto iter = prediction.begin(); iter != prediction.end(); ++iter)
			{
				auto prob = iter->second.first;
				if (iter->second.first > most_likely_action->second.first)
					most_likely_action = iter;
			}

			return { get_maximizer(most_likely_action->second.second), most_likely_action->first };
		}

		const auto feasible_action_weight = predict_consecutive(feasible_actions, franka);


		auto most_likely_actions = get_maximizer(feasible_action_weight.front());

		if (most_likely_actions == nullptr)
			return planner_layerwise_rtl::select_action(feasible_actions, franka);


		pn_token::Ptr postcondition = nullptr;
		for (const auto& instance : most_likely_actions->outputs)
			if (instance.first == robot_place)
			{
				postcondition = instance.second;
				break;
			}

		return { most_likely_actions, get_maximizer(feasible_action_weight.back(), postcondition) };

	}
	catch (...)
	{
		return { nullptr, nullptr };
	}
}


/////////////////////////////////////////////////////////////
//
//
//  Class: agent
//
//
/////////////////////////////////////////////////////////////
/*
agent::agent(pn_net::Ptr net,
	std::unique_ptr<action_planner>&& behaviour,
	std::chrono::high_resolution_clock::time_point start_time,
	std::string_view ip_addr)
	:
	franka(std::make_shared<remote_controller_wrapper>(start_time, ip_addr)),
	robot_place(behaviour->robot_place),
	marking(nullptr), // we get initial marking once initial detection is completed
	goal_update(true), // for initial computation of forward transitions
	behaviour(std::move(behaviour))
{
	start_thread();
}

agent::agent(state::pn_net::Ptr net, std::unique_ptr<action_planner>&& behaviour, bool do_logging, std::chrono::high_resolution_clock::time_point start_time)
	:
	franka(std::make_shared<simulation_controller_wrapper>(start_time, do_logging)),
	robot_place(behaviour->robot_place),
	marking(nullptr), // we get initial marking once initial detection is completed
	goal_update(true), // for initial computation of forward transitions
	behaviour(std::move(behaviour))
{
	start_thread();
}
*/
agent::agent(state::pn_net::Ptr net, std::unique_ptr<action_planner>&& behaviour, std::shared_ptr<state_observation::Controller> controller)
	:
	franka(std::move(controller)),
	robot_place(behaviour->robot_place),
	marking(nullptr), // we get initial marking once initial detection is completed
	goal_update(true), // for initial computation of forward transitions
	behaviour(std::move(behaviour))
{
	start_thread();
}

agent::~agent() noexcept
{
	stop_thread();
}

void agent::update_marking(state::pn_belief_marking::ConstPtr marking) noexcept
{
	if (marking == nullptr)
		return;

	auto tmp = marking->net.lock();
	if (!tmp)
		return;

	if (!net)
		net = std::move(tmp);

	if (franka.has_object_gripped() && marking->get_summed_probability(robot_place) < 0.9)
	{
		if (last_action)
			(*emitter)({ last_action, nullptr, franka.get_config() }, enact_priority::operation::DELETED);
		// std::cout << "Robot is gripping an object but marking doesn't represent this. Ignore marking update." << std::endl;
		return;
	}

	std::lock_guard<std::mutex> lock(update_mutex);
	behaviour->update_marking(franka, marking);

	this->marking = std::move(marking);
}

void agent::update_goal(state::pn_belief_marking::ConstPtr marking) noexcept
{
	update_marking(marking);
	goal_update = true;
}

void agent::update_behaviour(std::unique_ptr<action_planner>&& behaviour) noexcept
{
	this->behaviour = std::move(behaviour);
	std::cout << "robot behaviour " << this->behaviour->get_class_name() << std::endl;

	try
	{
		franka.log(",behaviour," + this->behaviour->get_class_name());
	}
	catch (...) {}

	if (this->marking)
	{
		this->behaviour->update_marking(franka, marking);
		this->behaviour->compute_forward_transitions(franka);
	}
}

void agent::reset(std::chrono::high_resolution_clock::time_point start)
{
	franka.reset_log(start);
	update_behaviour(std::make_unique<null_planner>(behaviour->robot_place));
}
/*
decltype(state::franka_agent::joint_signal)& agent::get_joint_signal()
{
	return franka.joint_signal;
}
*/
void agent::update()
{
	//no markings -> return to base pose
	if (marking == nullptr)
	{
		if(!franka.rest())
			(*emitter)({ nullptr, nullptr, franka.get_config() }, enact_priority::operation::CREATE);
		return;
	}

	//goal got updated and exists
	if (goal_update && this->net->get_goal())
	{
		behaviour->compute_forward_transitions(franka);
		goal_update = false;
		//remove any queued actions related to previous goal
		if(next_action)
			(*emitter)({ next_action, nullptr, franka.get_config() }, enact_priority::operation::MISSING);
		next_action = nullptr;
	}

	//next action exists but is not enabled
	if (next_action && marking->is_enabled(next_action) < 0.5)
	{
		(*emitter)({ next_action, nullptr, franka.get_config() }, enact_priority::operation::MISSING);
		next_action = nullptr;
	}

	//remove action from queue and work on it
	auto action = next_action;
	next_action = nullptr;

	{
		std::unique_lock<std::mutex> lock(update_mutex);
		if (!action)
		{
			//if there is no action available (e.g. after abort)
			//look for other actions excluding failed action
			const auto& [new_action, new_next_action] = behaviour->next(franka, failed_action);
			action = new_action;
			if (new_next_action != new_action)
				next_action = new_next_action;
		}
	}

	if (!action)
	{
		if (!std::dynamic_pointer_cast<pick_action>(last_action))
		{
			if(!franka.rest())
				(*emitter)({ nullptr, nullptr, franka.get_config() }, enact_priority::operation::CREATE);
		}

		{
			std::unique_lock<std::mutex> lock(update_mutex);
			action = get_put_back_action();
		}

		if (!action)
		{
			failed_action = nullptr;
			return;
		}
	}

	(*emitter)({ action, next_action, franka.get_config() }, enact_priority::operation::CREATE);

	auto abort = [&]()
	{
		//revoke and remove next_action from queue
		auto backup_next_action = next_action;
		next_action = nullptr;

		if (action)
		{
			//revoke action and enable backup operations
			failed_action = action;
		}
		if (action || next_action)
			(*emitter)({ action, next_action, franka.get_config() }, enact_priority::operation::MISSING);
	};

	try
	{
		if (!franka.approach(*action))
		{
			//error during approach -> abort
			abort();
			return;
		}

		auto local_marking = marking;
		if (local_marking->is_enabled(action) < 0.5)
		{
			//current action not possible -> abort
			abort();
			return;
		}

		const auto next_marking = local_marking->fire(action);
		if (next_action && next_marking->is_enabled(next_action) < 0.5)
		{
			/*//next_action not possible after executing action -> revoke and abort next_action
			next_action = nullptr;
			(*emitter)(action, enact_priority::operation::MISSING);*/
			abort();
			return;
		}

		if (!franka.execute_transition(*action) &&
			!(franka.has_object_gripped() && marking->get_summed_probability(robot_place) < 0.9f))
		{
			//action not executable and franka doesn't grip the object with enough probability
			abort();
			action = nullptr;
			return;
		}

		{
			std::unique_lock<std::mutex> lock(update_mutex);
			last_action = action;

			// only use newer marking if the successfully executed
			// action can be fired
			// the reason is that miss-detections can let the reasoner think that the human must have picked the object
			// and the robot's internal marking would represent no attached object despite it has.
			if (this->marking->is_enabled(action) > 0.1)
				local_marking = this->marking;

			this->marking = local_marking->fire(action);
			behaviour->update_marking(franka, marking);
			failed_action = nullptr;
		}

		//action got executed successfully
		(*emitter)({action, nullptr, franka.get_config() }, enact_priority::operation::DELETED);
	}
	catch (...)
	{
		//retry (maybe other action) later if something went wrong
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}
}

}
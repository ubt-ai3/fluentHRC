#include "task.hpp"

#include <memory>

#include <state_observation/pn_model_extension.hpp>

using namespace state_observation;

namespace simulation
{
const float human_agent::transfer_height = 0.2f;
const float human_agent::shoulder_height = 0.4f;

agent::agent(const environment::Ptr& env,
	pn_place::Ptr place,
	std::function<double(const pn_transition::Ptr&, agent&)> capabilities)
	:
	place(std::move(place)),
	env(env),
	capabilities(std::move(capabilities)),
	grabbed_object(nullptr)
{
	//env->additional_scene_objects.push_back(arm);
}

human_agent::human_agent(const environment::Ptr& env,
	Eigen::Vector3f base,
	Eigen::Vector3f front,
	pn_place::Ptr place,
	std::function<double(const pn_transition::Ptr&, agent&)> capabilities)
	:
	agent(env, place, std::move(capabilities)),
	home_pose(Eigen::Vector3f((base + front).x(), (base + front).y(), transfer_height)),
	arm(std::make_shared<simulated_arm>(
		Eigen::Vector3f(base + Eigen::Vector3f(0, 0, shoulder_height) + 0.25f * (front.cross(Eigen::Vector3f(0, 0, shoulder_height).normalized()))),
		home_pose)),
	state(execution_state::APPROACH_HORIZONTAL)
{
	env->additional_scene_objects.push_back(arm);
}

std::shared_ptr<agent> human_agent::clone() const
{
	return std::make_shared<human_agent>(*this);
}

bool human_agent::is_idle(std::chrono::duration<float> timestamp) const
{
	if (grabbed_object)
		return false;

	if (executing_action)
		return false;

	if (arm->is_moving(timestamp))
		return false;

	return arm->get_tcp(timestamp).isApprox(home_pose);
}

object_prototype::ConstPtr agent::get_grabbed_object() const
{
	return grabbed_object ? grabbed_object->prototype : nullptr;
}

pn_transition::Ptr human_agent::get_put_back_action(const pn_binary_marking::ConstPtr& marking) const noexcept
{
	if (!grabbed_object)
		return nullptr;


	pn_transition::Ptr candidate = nullptr;

	// transition to resource places must come first in the vector
	for (const auto& action : place->get_outgoing_transitions())
		if (marking->is_enabled(action.lock()))
		{
			candidate = action.lock();

			if (!executed_actions.empty() && grabbed_object && candidate->reverses(*executed_actions.back()))
				return candidate;
		}

	return candidate;
}

std::vector<state_observation::pn_transition::Ptr> human_agent::generate_feasible_actions()
{
	std::vector<state_observation::pn_transition::Ptr> result;
	const auto marking = env->get_marking();

	const auto& forward_transitions = env->net->get_forward_transitions(place);

	for (const pn_transition::Ptr& transition : forward_transitions)
	{

		if (!marking->is_enabled(transition) ||
			capabilities(transition, *this) < 0.5)
			continue;

		if (auto picking = std::dynamic_pointer_cast<pick_action>(transition))
		{
			const auto next_marking = marking->fire(transition);

			bool is_progress = false;
			for (const auto& next : forward_transitions) {
				if (next_marking->is_enabled(next)) {
					is_progress = true;
					break;
				}
			}

			if (!is_progress)
				continue;
		}

		
		result.push_back(transition);
	}
	//std::cout << "Amount: " << result.size() << std::endl << std::endl << std::endl;
	//if (result.empty())
		//std::cout << "No feasible actions generated" << std::endl;

	return result;
}

void human_agent::execute_action(const state_observation::pn_transition::Ptr& action, std::chrono::duration<float> timestamp)
{
	constexpr bool PRINT_ACTION = false;

	executing_action = action;
	state = execution_state::APPROACH_HORIZONTAL;

	executed_actions.push_back(action);

	auto print_pos = [](const Eigen::Vector3f& vec)
	{
		std::cout << "\t\t" << std::setprecision(3) << std::setfill('0') << "{ " << vec.x() << ", " << vec.y() << ", " << vec.z() << " }" << std::endl;
	};

	Eigen::Vector3f destination;
	if (const auto dynPIAction = std::dynamic_pointer_cast<pick_action>(executing_action))
	{
		pick_location = destination = dynPIAction->from->box.translation;

		if (PRINT_ACTION) {
			std::cout << "Picking " << dynPIAction->token->object->get_name() << " from ";
			print_pos(pick_location);
		}
	}
	else if (const auto dynPAction = std::dynamic_pointer_cast<place_action>(executing_action); dynPAction)
	{
		destination = dynPAction->to->box.translation;
		if (PRINT_ACTION) {
			std::cout << "Placing " << dynPAction->token->object->get_name() << " at ";
			print_pos(destination);
		}
	}
	else if (const auto dynSAction = std::dynamic_pointer_cast<stack_action>(executing_action); dynSAction)
	{
		//TODO:: test
		//TODO:: breaks possible assumption that there is a single movement for every action
		destination = dynSAction->center;
		if (PRINT_ACTION) {
			std::cout << "Stacking " << dynSAction->top_object->get_name() << " at "; // << dynSAction->to.first->box.translation << std::endl;
			print_pos(destination);
		}
	}
	else if (const auto dynRSAction = std::dynamic_pointer_cast<reverse_stack_action>(executing_action); dynRSAction)
	{
		//TODO:: test
		//TODO:: breaks possible assumption that there is a single movement for every action
		//--->	maybe take into account during fea in step method -> potential downside: sequential movements for one single agent
		destination = dynRSAction->from.first->box.translation;
		if (PRINT_ACTION) {
			std::cout << "Picking from stack " << dynRSAction->top_objects.front()->get_name() << " at ";
			print_pos(destination);
		}
	}
	else
	{
		throw std::runtime_error("Cannot execute this action");
	}

	destination.z() = transfer_height;
	arm->move(destination, timestamp);
}

void human_agent::to_home_pose(std::chrono::duration<float> timestamp)
{
	//std::cout << "Going home" << std::endl;
	arm->move(home_pose, timestamp);
}


void human_agent::step(std::chrono::duration<float> timestamp)
{
	if (arm->is_moving(timestamp))
	{
		if (grabbed_object)
			grabbed_object->center = arm->get_tcp(timestamp) + Eigen::Vector3f(0, 0, -simulated_arm::hand_radius);

		return;
	}

	if (!executing_action)
		return;

	if (const auto exPIAction = std::dynamic_pointer_cast<pick_action>(executing_action); exPIAction)
	{
		switch (state)
		{
		case execution_state::APPROACH_HORIZONTAL:
		{
			state = execution_state::APPROACH_VERTICAL;
			Eigen::Vector3f destination = exPIAction->from->box.translation;
			destination.z() += arm->hand_radius;
			arm->move(destination, timestamp);
			break;
		}

		case execution_state::APPROACH_VERTICAL:
		{
			state = execution_state::LIFT;
			auto iter = env->object_traces.find(*executing_action->inputs.begin());
			if (iter == env->object_traces.end())
			{
				executing_action = nullptr;
				break;
			}
			else 
			{
				grabbed_object = iter->second;
			}

			try {
				env->update(executing_action);
			}
			catch (const std::exception&)
			{
				executing_action = nullptr;
			}

			break;
		}

		case execution_state::LIFT:
		{
			executing_action = nullptr;
			break;
		}

		}
	}
	else if (const auto exPAction = std::dynamic_pointer_cast<place_action>(executing_action); exPAction)
	{
		switch (state)
		{
		case execution_state::APPROACH_HORIZONTAL:
		{
			state = execution_state::APPROACH_VERTICAL;
			Eigen::Vector3f destination = exPAction->to->box.translation;
			destination.z() += arm->hand_radius;
			arm->move(destination, timestamp);
			break;
		}

		case execution_state::APPROACH_VERTICAL:
		{
			state = execution_state::LIFT;

			if (!env->object_traces.contains(*executing_action->outputs.begin()))
			{
				try {
					env->update(executing_action); // position object on table
					grabbed_object = nullptr;
				}
				catch (const std::exception&)
				{
					executing_action = nullptr;
				}
			}
			else {
				executing_action = nullptr;
			}
			break;
		}

		case execution_state::LIFT:
		{
			executing_action = nullptr;
			break;
		}
		}
	}
	else if (const auto exSAction = std::dynamic_pointer_cast<stack_action>(executing_action); exSAction)
	{
		switch (state)
		{
		case execution_state::APPROACH_HORIZONTAL:
		{
			state = execution_state::APPROACH_VERTICAL;
			const Eigen::Vector3f destination = exSAction->center + Eigen::Vector3f(0, 0, simulated_arm::hand_radius);
			arm->move(destination, timestamp);
			break;
		}

		case execution_state::APPROACH_VERTICAL:
		{
			state = execution_state::LIFT;
			
			if (!env->object_traces.contains(exSAction->to))
			{
				try {
					env->update(executing_action); // position object in scene
					grabbed_object = nullptr;
				}
				catch (const std::exception&)
				{
					executing_action = nullptr;
				}
			}
			else 
			{
				executing_action = nullptr;
			}
			break;
		}

		case execution_state::LIFT:
		{
			executing_action = nullptr;
			break;
		}
		}
	}
	else if (const auto exRSAction = std::dynamic_pointer_cast<reverse_stack_action>(executing_action); exRSAction)
	{
		switch (state)
		{
		case execution_state::APPROACH_HORIZONTAL:
		{
			state = execution_state::APPROACH_VERTICAL;
			Eigen::Vector3f destination = exRSAction->from.first->box.translation;
			destination.z() += arm->hand_radius;
			arm->move(destination, timestamp);
			break;
		}

		case execution_state::APPROACH_VERTICAL:
		{
			state = execution_state::LIFT;
			auto iter = env->object_traces.find(exRSAction->from);
			if (iter == env->object_traces.end())
			{
				executing_action = nullptr;
				break;
			}
			else
			{
				grabbed_object = iter->second;
			}

			try {
				env->update(executing_action);
			}
			catch (const std::exception&)
			{
				executing_action = nullptr;
			}

			break;
		}

		case execution_state::LIFT:
		{
			executing_action = nullptr;
			break;
		}

		}
	} else
	{
		throw std::runtime_error("Cannot execute this action");
	}
}

bool human_agent::is_connected_to_agent_place(const pn_transition& transition) const
{
	return 
	std::ranges::any_of(transition.inputs, [this](const auto& p) { return p.first == place; }) ||
	std::ranges::any_of(transition.outputs, [this](const auto& p) { return p.first == place; });
}

void human_agent::print_actions() const
{
#ifdef DEBUG_PN_ID
	std::cout << place->id << ": ";

	for (const auto& entry : executed_actions)
	{
		const auto& input = entry->get_pure_input_arcs();
		if (!input.empty())
		{
			const auto& instance = *input.begin();
			std::cout << "(" << instance.first->id << ", " << instance.second->id << ")";
		}

		std::cout << " -> ";

		const auto& output = entry->get_pure_input_arcs();
		if (!output.empty())
		{
			const auto& instance = *output.begin();
			std::cout << "(" << instance.first->id << ", " << instance.second->id << ")";
		}
	}
	std::cout << std::endl;
#endif
}


sim_task::sim_task(const std::string& name, 
	const environment::Ptr& env, 
	const std::vector<agent::Ptr>& agents,
	const pn_instance& init_goal,
	const pn_instance& task_goal)
	:
	name(name),
	env(env),
	agents(agents),
	init_goal(init_goal),
	task_goal(task_goal)
{
	env->net->set_goal(init_goal.first);
}

task_execution::task_execution(const sim_task::Ptr& task,
	double fps)
	:
	task(task),
	duration(1 / fps),
	timestamp(-duration)
{
}

std::chrono::duration<float> task_execution::step()
{
	timestamp += duration;
	std::set<pn_place::Ptr> destinations;

	auto add_overlapping_destinations = [&destinations](const std::vector<pn_place::Ptr>& places) {
		for (const auto& place : places)
		{
			destinations.emplace(place);

			if (auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place); boxed_place)
				for (const auto& overlap : boxed_place->overlapping_places)
					destinations.emplace(overlap);
		}
	};

	/**
	 * Add places of actions which are already beeing executed for later exclusivity check
	 */
	for (const auto& agent : task->agents)
	{
		agent->step(timestamp);
		if (!agent->executing_action)
			continue;

		add_overlapping_destinations(agent->executing_action->get_outputs());
		destinations.emplace(agent->executing_action->get_inputs().at(0));
	}

	auto marking = task->env->get_marking();
	for (const auto& agent : task->agents)
	{
		if (agent->executing_action)
			continue;

		if (auto human_agent = std::dynamic_pointer_cast<simulation::human_agent>(agent); human_agent)
		{
			/**
			 * Check if agent is currently not doing anything
			 * Then get available actions which are not already being executed
			 * if no action is available return to base
			 */
			auto actions = human_agent->generate_feasible_actions();
			std::erase_if(actions, [&destinations](const pn_transition::Ptr& action)
				{
					return destinations.contains(action->get_outputs().at(0)) ||
						destinations.contains(action->get_inputs().at(0));
				});

			if (agent->get_grabbed_object() && actions.empty())
				if (auto action = human_agent->get_put_back_action(marking))
					actions.emplace_back(action);

			if (!actions.empty())
			{
				std::uniform_int_distribution<int> dist(0, static_cast<int>(actions.size() - 1));
				//std::cout << "Picking 1 action out of " << actions.size() << std::endl;
				//Error:: picks even actions where no consecutive action is possible
				human_agent->execute_action(actions[dist(rd)], timestamp);

				add_overlapping_destinations(agent->executing_action->get_outputs());
				destinations.emplace(agent->executing_action->get_inputs().at(0));
			}
			else
			{
				//std::cout << "No feasible actions which are not already executed" << std::endl;
				human_agent->to_home_pose(timestamp);
			}
		}
	}

	
	for (const auto& trans : task->env->net->get_meta_transitions())
	{
		if (!std::dynamic_pointer_cast<pick_action>(trans) &&
			!std::dynamic_pointer_cast<place_action>(trans) &&
			!std::dynamic_pointer_cast<stack_action>(trans) &&
			marking->is_enabled(trans))
		{
			marking = task->env->update(trans);
		}
	}

	return timestamp;
}

repeated_task_execution::repeated_task_execution(const sim_task::Ptr& task, double fps, unsigned count)
	:
	task_execution(task, fps),
	remaining(count),
	initial_objects(task->env->object_traces)
{
	for (const auto& agent : task->agents)
		initial_agents.emplace_back(agent);

	// modify net by adding cleanup and initialization transitions
	auto& net = *task->env->net;
	if (!net.get_goal())
		return;

	std::lock_guard<std::mutex> lock(net.mutex);
	std::set<pn_instance> cleanable_instances;
	pn_instance goal_condition;

	const auto& goal_t = *net.get_goal()->get_incoming_transitions()[0].lock();

	if (const auto& output = goal_t.get_pure_output_arcs(); !output.empty())
		goal_condition = *output.begin();

	for (const auto& transition : std::vector<pn_transition::Ptr>(net.get_transitions()))
	{
		// iterate over copy to avoid invalid iterator when adding new transition
		for(const auto& instance : std::set<pn_instance>(transition->outputs))
		{
			if (!std::dynamic_pointer_cast<pn_boxed_place>(instance.first))
				continue;

			if (cleanable_instances.insert(instance).second)
				cleanup.emplace_back(net.create_transition({ instance, goal_condition }, { goal_condition }));
		}
	}	
	init = net.create_transition({goal_condition}, std::vector<pn_instance>(task->env->distribution.begin(), task->env->distribution.end()));
}
	
std::chrono::duration<float> repeated_task_execution::step()
{
	auto& env = *task->env;
	auto& agents = task->agents;

	if(cleaned && timestamp >= init_time)
	{
		env.update(init);
		env.object_traces = initial_objects;
		for (auto& entry : env.object_traces)
			entry.second->center = std::dynamic_pointer_cast<pn_boxed_place>(entry.first.first)->box.translation;

		agents.clear();
		for (const auto& a : initial_agents)
			agents.emplace_back(a->clone());

		cleaned = false;

		return task_execution::step();
	}
	
	bool finished = env.get_marking()->is_occupied(env.net->get_goal());

	for (const auto& agent : agents)
		if (!agent->is_idle(timestamp))
		{
			finished = false;
			break;
		}

	if (finished && !cleaned) {
		if (!--remaining)
			return task_execution::step();

		for(const auto& transition : cleanup)
		{
			if (env.get_marking()->is_enabled(transition))
				env.update(transition);			
		}
		
		// do initialization in next step so that 
		// pn reasoning can catch up
		cleaned = true;
		init_time = timestamp + std::chrono::duration<float>(1);
	}

	return task_execution::step();
}
robot_agent::robot_agent(const environment::Ptr& env, state_observation::pn_place::Ptr place, std::function<double(const state_observation::pn_transition::Ptr&, agent&)> capabilities)
	: agent(env, std::move(place), std::move(capabilities))
{
}

std::shared_ptr<agent> robot_agent::clone() const
{
	return std::make_shared<robot_agent>(*this);
}

bool robot_agent::is_idle(std::chrono::duration<float> timestamp) const
{
	return true;
}

void robot_agent::step(std::chrono::duration<float> timestamp)
{

}

}
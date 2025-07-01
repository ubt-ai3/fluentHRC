#include "riedelbauch17.hpp"

#include <state_observation/object_prototype_loader.hpp>
#include <state_observation/pn_model_extension.hpp>

using namespace state_observation;

namespace simulation
{

pn_instance riedelbauch17::create_goal(pn_net& net,
	std::vector<pn_instance>&& instances)
{
	auto goal = net.create_place();

	auto goal_token = std::make_shared<pn_token>();
	std::vector<pn_instance> output_instances(instances.begin(), instances.end());
	output_instances.push_back(std::make_pair(goal, goal_token));

	net.create_transition(std::move(instances),
		std::move(output_instances));

	return std::make_pair(goal, goal_token);
}

pn_instance riedelbauch17::create_goal(pn_net& net,
	const std::set<pn_instance>& instances)
{
	return create_goal(net, std::vector<pn_instance>(instances.begin(), instances.end()));
}

pn_instance riedelbauch17::create_goal(pn_net& net,
	std::initializer_list<std::vector<pn_object_instance>> vectors)
{
	std::vector<pn_instance> instances;
	for (const auto& vec : vectors)
		instances.insert(instances.end(), vec.begin(), vec.end());

	return create_goal(net, std::move(instances));
}

sim_task::Ptr simulation::riedelbauch17::pack_and_stack(const object_parameters& object_params,
	object_prototype_loader loader)
{
	auto env = std::make_shared<environment>(object_params);
	env->additional_scene_objects.push_back(std::make_shared<simulated_table>());

	auto get_box = [](int x, int y, const object_prototype::ConstPtr& obj) {
		return aabb(obj->get_bounding_box().diagonal,
			Eigen::Vector3f(x * simulated_table::width / 8, y * simulated_table::breadth / 6, obj->get_bounding_box().diagonal.z() / 2));
	};

	// create initial objects
	std::vector<pn_object_instance> red_blocks;
	auto rb = loader.get("red block");

	for (int i = 1; i < 5; i++)
	{
		red_blocks.push_back(env->add_object(rb, get_box(i, -1, rb)));
		red_blocks.push_back(env->add_object(rb, get_box(i, 2, rb)));
	}

	std::vector<pn_object_instance> wooden_cubes;
	auto wc = loader.get("wooden cube");

	for (int i = -5; i < -1; i++)
	{
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -2, wc)));
		wooden_cubes.push_back(env->add_object(wc, get_box(i, 3, wc)));
	}

	auto init_goal = create_goal(*env->net, { wooden_cubes,  red_blocks });

	// create target locations
	std::vector<pn_token::Ptr> tokens({ red_blocks.front().second, wooden_cubes.front().second });

	auto h_hand = env->net->create_place(true);
	auto r_hand = env->net->create_place(true);

	std::vector<pn_boxed_place::Ptr> target_locations;
	for (int i = -5; i < -1; i++)
	{
		target_locations.push_back(env->add_location(get_box(i, 0, rb)));
		target_locations.push_back(env->add_location(get_box(i, 1, rb)));
	}


	std::set<pn_instance> goal_instances;
	for (const auto& place : target_locations)
		goal_instances.emplace(place, tokens.front());

	// create actions
	for (const auto& [boxPlace, token] : wooden_cubes)
	{
		if (boxPlace->box.translation.y() <= 3. / 10 * simulated_table::breadth)
		{
			//human reach zone
			env->net->add_transition(std::make_shared<pick_action>(token, boxPlace, h_hand));
			env->net->add_transition(std::make_shared<place_action>(token, h_hand, boxPlace));
		}

		if (boxPlace->box.translation.y() >= -3. / 10 * simulated_table::breadth)
		{
			//robot reach zone
			env->net->add_transition(std::make_shared<pick_action>(token, boxPlace, r_hand));
			env->net->add_transition(std::make_shared<place_action>(token, r_hand, boxPlace));
		}
	}

	for (const auto& [boxPlace, token] : red_blocks)
	{
		env->net->add_transition(std::make_shared<pick_action>(token, boxPlace, h_hand));
		env->net->add_transition(std::make_shared<pick_action>(token, boxPlace, r_hand));
	}

	for (const pn_boxed_place::Ptr& place : target_locations)
	{
		// place red blocks
		env->net->add_transition(std::make_shared<place_action>(red_blocks.front().second, h_hand, place));
		env->net->add_transition(std::make_shared<place_action>(red_blocks.front().second, r_hand, place));


		// stack cubes
		goal_instances.emplace(stack_action::create(env->net, env->token_traces, h_hand, place, loader.get("red block"), loader.get("wooden cube"))->to);
		stack_action::create(env->net, env->token_traces, r_hand, place, loader.get("red block"), loader.get("wooden cube")); // goal instance already added
	}


	// create agents
	auto robot = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0, simulated_table::width / 2 + 0.1f, 0),
		Eigen::Vector3f(0, -2. / 10 * simulated_table::width, 0),
		r_hand);


	auto human = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		h_hand);


	return std::make_shared<sim_task>("Riedelbauch17_Stack", env, std::vector<agent::Ptr>({ robot, human }), init_goal, create_goal(*env->net, goal_instances));
}

sim_task::Ptr simulation::riedelbauch17::pack_and_stack_and_swap(const object_parameters& object_params,
	object_prototype_loader loader)
{
	auto env = std::make_shared<environment>(object_params);
	env->additional_scene_objects.push_back(std::make_shared<simulated_table>());

	auto get_box = [](int x, int y, const object_prototype::ConstPtr& obj) {
		return aabb(obj->get_bounding_box().diagonal,
			Eigen::Vector3f(x * simulated_table::width / 8, y * simulated_table::breadth / 6, obj->get_bounding_box().diagonal.z() / 2));
	};

	// create initial objects
	std::vector<pn_object_instance> red_blocks;
	auto rb = loader.get("red block");

	for (int i = 1; i < 5; i++)
	{
		for(int j = -1; j < 3; j++)
			red_blocks.push_back(env->add_object(rb, get_box(i, j, rb)));
	}

	std::vector<pn_object_instance> wooden_cubes;
	auto wc = loader.get("wooden cube");

	std::vector<pn_boxed_place::Ptr> target_locations;

	for (int i = -5; i < -1; i++)
	{
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -2, wc)));
		wooden_cubes.push_back(env->add_object(wc, get_box(i, 3, wc)));

		wooden_cubes.push_back(env->add_object(wc, get_box(i, -1, wc)));
 		wooden_cubes.push_back(env->add_object(wc, get_box(i, 2, wc)));
	}

	auto init_goal = create_goal(*env->net, { wooden_cubes,  red_blocks });

	// create target locations
	std::vector<pn_token::Ptr> tokens({ red_blocks.front().second, wooden_cubes.front().second });

	auto h0_hand = env->net->create_place(true);
	auto r0_hand = env->net->create_place(true);
	auto h1_hand = env->net->create_place(true);
	auto r1_hand = env->net->create_place(true);


	for (int i = -5; i < -1; i++)
	{
		for (int j = -1; j < 3; j++)
		{
			target_locations.push_back(env->add_location(get_box(i, j, rb)));
		}
	}


	std::set<pn_instance> goal_instances;
	for (const auto& place : target_locations)
		goal_instances.emplace(place, tokens.front());

	// create actions
	for (const pn_object_instance& instance : wooden_cubes)
	{
		if (instance.first->box.translation.y() <= 3. / 10 * simulated_table::breadth)
		{
			for (const auto& h_hand : { h0_hand, h1_hand })
			{
				//human reach zone
				env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, h_hand));
				env->net->add_transition(std::make_shared<place_action>(instance.second, h_hand, instance.first));
			}
		}

		if (instance.first->box.translation.y() >= -3. / 10 * simulated_table::breadth)
		{
			for (const auto& r_hand : { r0_hand, r1_hand })
			{
				//robot reach zone
				env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, r_hand));
				env->net->add_transition(std::make_shared<place_action>(instance.second, r_hand, instance.first));
			}
		}
	}

	for (const pn_object_instance& instance : red_blocks)
	{
		for (const auto& hand : { h0_hand, h1_hand, r0_hand, r1_hand })
			env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));
	}

	for (const pn_boxed_place::Ptr& place : target_locations)
	{
		for (const auto& hand : { h0_hand, h1_hand, r0_hand, r1_hand })
		{
			// place red blocks
			env->net->add_transition(std::make_shared<place_action>(red_blocks.front().second, hand, place));

			// stack cubes
			goal_instances.emplace(stack_action::create(env->net, env->token_traces, hand, place, loader.get("red block"), loader.get("wooden cube"))->to);
		}
	}


	// create agents
	auto robot0 = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0.15f, simulated_table::width / 2 + 0.1f, 0),
		Eigen::Vector3f(0, -2. / 10 * simulated_table::width, 0),
		r0_hand);

	auto robot1 = std::make_shared<human_agent>(env,
		Eigen::Vector3f(-0.15f, simulated_table::width / 2 + 0.1f, 0),
		Eigen::Vector3f(0, -2. / 10 * simulated_table::width, 0),
		r1_hand);


	auto human0 = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		h0_hand);

	auto human1 = std::make_shared<human_agent>(env,
		Eigen::Vector3f(-0.15f, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		h1_hand);


	return std::make_shared<sim_task>("Riedelbauch17_StackAndSwap", env, std::vector<agent::Ptr>({ robot0, robot1, human0, human1 }), init_goal, create_goal(*env->net, goal_instances));
}

sim_task::Ptr simulation::riedelbauch17::pack_and_stack_and_swap_small(const object_parameters& object_params,
	object_prototype_loader loader)
{
	auto env = std::make_shared<environment>(object_params);
	env->additional_scene_objects.push_back(std::make_shared<simulated_table>());

	auto get_box = [](int x, int y, const object_prototype::ConstPtr& obj) {
		return aabb(obj->get_bounding_box().diagonal,
			Eigen::Vector3f(x * simulated_table::width / 8, y * simulated_table::breadth / 6, obj->get_bounding_box().diagonal.z() / 2));
	};

	// create initial objects
	std::vector<pn_object_instance> red_blocks;
	auto rb = loader.get("red block");

	for (int i = 1; i < 3; i++)
	{
		for (int j = -1; j < 3; j++)
			red_blocks.push_back(env->add_object(rb, get_box(i, j, rb)));
	}

	std::vector<pn_object_instance> wooden_cubes;
	auto wc = loader.get("wooden cube");

	std::vector<pn_boxed_place::Ptr> target_locations;

	for (int i = -3; i < -1; i++)
	{
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -2, wc)));
		wooden_cubes.push_back(env->add_object(wc, get_box(i, 3, wc)));

		wooden_cubes.push_back(env->add_object(wc, get_box(i, -1, wc)));
		wooden_cubes.push_back(env->add_object(wc, get_box(i, 2, wc)));
	}

	auto init_goal = create_goal(*env->net, { wooden_cubes,  red_blocks });

	// create target locations
	std::vector<pn_token::Ptr> tokens({ red_blocks.front().second, wooden_cubes.front().second });

	auto h0_hand = env->net->create_place(true);
	auto r0_hand = env->net->create_place(true);
	auto h1_hand = env->net->create_place(true);
	auto r1_hand = env->net->create_place(true);


	for (int i = -3; i < -1; i++)
	{
		for (int j = -1; j < 3; j++)
		{
			target_locations.push_back(env->add_location(get_box(i, j, rb)));
		}
	}


	std::set<pn_instance> goal_instances;
	for (const auto& place : target_locations)
		goal_instances.emplace(place, tokens.front());

	// create actions
	for (const pn_object_instance& instance : wooden_cubes)
	{
		if (instance.first->box.translation.y() <= 3. / 10 * simulated_table::breadth)
		{
			for (const auto& h_hand : { h0_hand, h1_hand })
			{
				//human reach zone
				env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, h_hand));
				env->net->add_transition(std::make_shared<place_action>(instance.second, h_hand, instance.first));
			}
		}

		if (instance.first->box.translation.y() >= -3. / 10 * simulated_table::breadth)
		{
			for (const auto& r_hand : { r0_hand, r1_hand })
			{
				//robot reach zone
				env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, r_hand));
				env->net->add_transition(std::make_shared<place_action>(instance.second, r_hand, instance.first));
			}
		}
	}

	for (const pn_object_instance& instance : red_blocks)
	{
		for (const auto& hand : { h0_hand, h1_hand, r0_hand, r1_hand })
			env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));
	}

	for (const pn_boxed_place::Ptr& place : target_locations)
	{
		for (const auto& hand : { h0_hand, h1_hand, r0_hand, r1_hand })
		{
			// place red blocks
			env->net->add_transition(std::make_shared<place_action>(red_blocks.front().second, hand, place));

			// stack cubes
			goal_instances.emplace(stack_action::create(env->net, env->token_traces, hand, place, loader.get("red block"), loader.get("wooden cube"))->to);
		}
	}


	// create agents
	auto robot0 = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0.15f, simulated_table::width / 2 + 0.1f, 0),
		Eigen::Vector3f(0, -2. / 10 * simulated_table::width, 0),
		r0_hand);

	auto robot1 = std::make_shared<human_agent>(env,
		Eigen::Vector3f(-0.15f, simulated_table::width / 2 + 0.1f, 0),
		Eigen::Vector3f(0, -2. / 10 * simulated_table::width, 0),
		r1_hand);


	auto human0 = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		h0_hand);

	auto human1 = std::make_shared<human_agent>(env,
		Eigen::Vector3f(-0.15f, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		h1_hand);


	return std::make_shared<sim_task>("Riedelbauch17_StackAndSwap_Small", env, std::vector<agent::Ptr>({ robot0, robot1, human0, human1 }), init_goal, create_goal(*env->net, goal_instances));
}

} // namespace simulation
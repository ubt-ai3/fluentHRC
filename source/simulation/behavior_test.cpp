#include "behavior_test.hpp"

#include "riedelbauch17.hpp"

using namespace simulation;
using namespace state_observation;

sim_task::Ptr behavior_test::row_ltr(const object_parameters& object_params,
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
		red_blocks.push_back(env->add_object(rb, get_box(i, 0, rb)));
		red_blocks.push_back(env->add_object(rb, get_box(i, -1, rb)));
	}

	std::vector<pn_object_instance> wooden_cubes;
	auto wc = loader.get("wooden cube");

	std::vector<pn_boxed_place::Ptr> target_locations;

	for (int i = -5; i < -1; i++)
	{
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -1, wc)));
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -2, wc)));
	}

	auto init_goal = riedelbauch17::create_goal(*env->net, { wooden_cubes,  red_blocks });

	// create target locations
	std::vector<pn_token::Ptr> tokens({ red_blocks.front().second, wooden_cubes.front().second });

	auto hand = env->net->create_place(true);


	for (int i = -5; i < -1; i++)
	{
		target_locations.push_back(env->add_location(get_box(i, 0, rb)));
		target_locations.push_back(env->add_location(get_box(i, -1, rb)));
	}


	std::set<pn_instance> goal_instances;

	// create actions in the order in which they will be executed
	auto pick = [&](const pn_object_instance& instance)
	{
		env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));
	};
	auto put = [&](const pn_boxed_place::Ptr& place)
	{
		auto pick = env->net->get_transitions().back();

		env->net->add_transition(std::make_shared<place_action>(std::dynamic_pointer_cast<pn_object_token>((*pick->outputs.begin()).second), hand, place));
	};
	auto stack = [&](const pn_boxed_place::Ptr& place)
	{
		goal_instances.emplace(stack_action::create(env->net, env->token_traces, hand, place, loader.get("red block"), loader.get("wooden cube"))->to);
	};

	for (int i = 0; i < 4; i++)
	{
		pick(red_blocks[0 + 2 * i]);
		put(target_locations[0 + 2 * i]);
	}

	for (int i = 0; i < 4; i++)
	{
		pick(wooden_cubes[0 + 2 * i]);
		stack(target_locations[0 + 2 * i]);
	}

	for (int i = 0; i < 4; i++)
	{
		pick(red_blocks[1 + 2 * i]);
		put(target_locations[1 + 2 * i]);
	}

	for (int i = 0; i < 4; i++)
	{
		pick(wooden_cubes[1 + 2 * i]);
		stack(target_locations[1 + 2 * i]);
	}



	// create agent, execute actions in order of creation
	// requires that capabilities is called at most once per step and transition
	auto human = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		hand,
		[sequence(env->net->get_transitions())](const pn_transition::Ptr& t, agent&) mutable
	{
		if (!sequence.empty() && sequence.front() == t)
		{
			sequence.erase(sequence.begin());
			return 1.;
		}

		return 0.;
	});




	return std::make_shared<sim_task>("Riedelbauch17_Stack_RowLTR", env, std::vector<agent::Ptr>({ human}), init_goal, riedelbauch17::create_goal(*env->net, goal_instances));
}

sim_task::Ptr simulation::behavior_test::row_rtl(const state_observation::object_parameters& object_params, state_observation::object_prototype_loader loader)
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
		red_blocks.push_back(env->add_object(rb, get_box(i, 0, rb)));
		red_blocks.push_back(env->add_object(rb, get_box(i, -1, rb)));
	}

	std::vector<pn_object_instance> wooden_cubes;
	auto wc = loader.get("wooden cube");

	std::vector<pn_boxed_place::Ptr> target_locations;

	for (int i = -5; i < -1; i++)
	{
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -1, wc)));
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -2, wc)));
	}

	auto init_goal = riedelbauch17::create_goal(*env->net, { wooden_cubes,  red_blocks });

	// create target locations
	std::vector<pn_token::Ptr> tokens({ red_blocks.front().second, wooden_cubes.front().second });

	auto hand = env->net->create_place(true);


	for (int i = -5; i < -1; i++)
	{
		target_locations.push_back(env->add_location(get_box(i, 0, rb)));
		target_locations.push_back(env->add_location(get_box(i, -1, rb)));
	}


	std::set<pn_instance> goal_instances;

	// create actions in the order in which they will be executed
	auto pick = [&](const pn_object_instance& instance)
	{
		env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));
	};
	auto put = [&](const pn_boxed_place::Ptr& place)
	{
		auto pick = env->net->get_transitions().back();

		env->net->add_transition(std::make_shared<place_action>(std::dynamic_pointer_cast<pn_object_token>((*pick->outputs.begin()).second), hand, place));
	};
	auto stack = [&](const pn_boxed_place::Ptr& place)
	{
		goal_instances.emplace(stack_action::create(env->net, env->token_traces, hand, place, loader.get("red block"), loader.get("wooden cube"))->to);
	};

	for (int i = 3; i >= 0; i--)
	{
		pick(red_blocks[0 + 2 * i]);
		put(target_locations[0 + 2 * i]);
	}

	for (int i = 3; i >= 0; i--)
	{
		pick(wooden_cubes[0 + 2 * i]);
		stack(target_locations[0 + 2 * i]);
	}

	for (int i = 3; i >= 0; i--)
	{
		pick(red_blocks[1 + 2 * i]);
		put(target_locations[1 + 2 * i]);
	}

	for (int i = 3; i >= 0; i--)
	{
		pick(wooden_cubes[1 + 2 * i]);
		stack(target_locations[1 + 2 * i]);
	}



	// create agent, execute actions in order of creation
	// requires that capabilities is called at most once per step and transition
	auto human = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		hand,
		[sequence(env->net->get_transitions())](const pn_transition::Ptr& t, agent&) mutable
	{
		if (!sequence.empty() && sequence.front() == t)
		{
			sequence.erase(sequence.begin());
			return 1.;
		}

		return 0.;
	});


	return std::make_shared<sim_task>("Riedelbauch17_Stack_RowRTL", env, std::vector<agent::Ptr>({ human }), init_goal, riedelbauch17::create_goal(*env->net, goal_instances));
}

sim_task::Ptr simulation::behavior_test::mix_ltr(const state_observation::object_parameters& object_params, state_observation::object_prototype_loader loader)
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
		red_blocks.push_back(env->add_object(rb, get_box(i, 0, rb)));
		red_blocks.push_back(env->add_object(rb, get_box(i, -1, rb)));
	}

	std::vector<pn_object_instance> wooden_cubes;
	auto wc = loader.get("wooden cube");

	std::vector<pn_boxed_place::Ptr> target_locations;

	for (int i = -5; i < -1; i++)
	{
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -1, wc)));
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -2, wc)));
	}

	auto init_goal = riedelbauch17::create_goal(*env->net, { wooden_cubes,  red_blocks });

	// create target locations
	std::vector<pn_token::Ptr> tokens({ red_blocks.front().second, wooden_cubes.front().second });

	auto hand = env->net->create_place(true);


	for (int i = -5; i < -1; i++)
	{
		target_locations.push_back(env->add_location(get_box(i, 0, rb)));
		target_locations.push_back(env->add_location(get_box(i, -1, rb)));
	}


	std::set<pn_instance> goal_instances;

	// create actions in the order in which they will be executed
	auto pick = [&](const pn_object_instance& instance)
	{
		env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));
	};
	auto put = [&](const pn_boxed_place::Ptr& place)
	{
		auto pick = env->net->get_transitions().back();

		env->net->add_transition(std::make_shared<place_action>(std::dynamic_pointer_cast<pn_object_token>((*pick->outputs.begin()).second), hand, place));
	};
	auto stack = [&](const pn_boxed_place::Ptr& place)
	{
		goal_instances.emplace(stack_action::create(env->net, env->token_traces, hand, place, loader.get("red block"), loader.get("wooden cube"))->to);
	};

	for (int i = 0; i < 4; i++)
	{
		pick(red_blocks[0 + 2 * i]);
		put(target_locations[0 + 2 * i]);
		pick(wooden_cubes[0 + 2 * i]);
		stack(target_locations[0 + 2 * i]);
	}

	for (int i = 0; i < 4; i++)
	{
		pick(red_blocks[1 + 2 * i]);
		put(target_locations[1 + 2 * i]);
		pick(wooden_cubes[1 + 2 * i]);
		stack(target_locations[1 + 2 * i]);
	}

	// create agent, execute actions in order of creation
	// requires that capabilities is called at most once per step and transition
	auto human = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		hand,
		[sequence(env->net->get_transitions())](const pn_transition::Ptr& t, agent&) mutable
	{
		if (!sequence.empty() && sequence.front() == t)
		{
			sequence.erase(sequence.begin());
			return 1.;
		}

		return 0.;
	});


	return std::make_shared<sim_task>("Riedelbauch17_Stack_MixLTR", env, std::vector<agent::Ptr>({ human }), init_goal, riedelbauch17::create_goal(*env->net, goal_instances));
}

sim_task::Ptr simulation::behavior_test::mix_rtl(const state_observation::object_parameters& object_params, state_observation::object_prototype_loader loader)
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
		red_blocks.push_back(env->add_object(rb, get_box(i, 0, rb)));
		red_blocks.push_back(env->add_object(rb, get_box(i, -1, rb)));
	}

	std::vector<pn_object_instance> wooden_cubes;
	auto wc = loader.get("wooden cube");

	std::vector<pn_boxed_place::Ptr> target_locations;

	for (int i = -5; i < -1; i++)
	{
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -1, wc)));
		wooden_cubes.push_back(env->add_object(wc, get_box(i, -2, wc)));
	}

	auto init_goal = riedelbauch17::create_goal(*env->net, { wooden_cubes,  red_blocks });

	// create target locations
	std::vector<pn_token::Ptr> tokens({ red_blocks.front().second, wooden_cubes.front().second });

	auto hand = env->net->create_place(true);


	for (int i = -5; i < -1; i++)
	{
		target_locations.push_back(env->add_location(get_box(i, 0, rb)));
		target_locations.push_back(env->add_location(get_box(i, -1, rb)));
	}


	std::set<pn_instance> goal_instances;

	// create actions in the order in which they will be executed
	auto pick = [&](const pn_object_instance& instance)
	{
		env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));
	};
	auto put = [&](const pn_boxed_place::Ptr& place)
	{
		auto pick = env->net->get_transitions().back();

		env->net->add_transition(std::make_shared<place_action>(std::dynamic_pointer_cast<pn_object_token>((*pick->outputs.begin()).second), hand, place));
	};
	auto stack = [&](const pn_boxed_place::Ptr& place)
	{
		goal_instances.emplace(stack_action::create(env->net, env->token_traces, hand, place, loader.get("red block"), loader.get("wooden cube"))->to);
	};

	for (int i = 3; i >= 0; i--)
	{
		pick(red_blocks[0 + 2 * i]);
		put(target_locations[0 + 2 * i]);
		pick(wooden_cubes[0 + 2 * i]);
		stack(target_locations[0 + 2 * i]);
	}

	for (int i = 3; i >= 0; i--)
	{
		pick(red_blocks[1 + 2 * i]);
		put(target_locations[1 + 2 * i]);
		pick(wooden_cubes[1 + 2 * i]);
		stack(target_locations[1 + 2 * i]);
	}

	// create agent, execute actions in order of creation
	// requires that capabilities is called at most once per step and transition
	auto human = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		hand,
		[sequence(env->net->get_transitions())](const pn_transition::Ptr& t, agent&) mutable
	{
		if (!sequence.empty() && sequence.front() == t)
		{
			sequence.erase(sequence.begin());
			return 1.;
		}

		return 0.;
	});


	return std::make_shared<sim_task>("Riedelbauch17_Stack_MixRTL", env, std::vector<agent::Ptr>({ human }), init_goal, riedelbauch17::create_goal(*env->net, goal_instances));
}

#include "build_test.hpp"

#include <ranges>

#include "riedelbauch17.hpp"

namespace simulation
{
	using namespace state_observation;

	std::shared_ptr<building_simulation_test> create_building_simulation_task(
		const object_parameters& object_params,
		object_prototype_loader loader)
	{
		building_simulation_test test;
		const auto& prototypes = loader.get_prototypes();

		std::vector<pn_object_token::Ptr> tokens;
		tokens.reserve(prototypes.size());

		for (const auto& proto : prototypes)
			tokens.emplace_back(std::make_shared<pn_object_token>(proto));

		
		building_element::load_building_elements(tokens);
		composed_building_element::load_type_diags();
		state_observation::building::load_building_elements(prototypes);

		std::string wb = "wooden block";
		std::string wp = "wooden plank";
		std::string wcu = "wooden cube";
		std::string wcy = "wooden cylinder";
		std::string mcy = "magenta cylinder";
		std::string bb = "blue block";
		std::string pfb = "purple flat block";
		std::string yfb = "yellow flat block";

		const auto& rotation = Eigen::Quaternionf(std::cosf(std::numbers::pi_v<float> / 4.f), 0, std::sinf(std::numbers::pi_v<float> / 4.f) * 1.f, 0);
		//const auto& rotation1 = Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 4.f), 0, 0, std::sin(std::numbers::pi_v<float> / 4.f) * 1.f);

		test.building = builder()
			.add_element(single_building_element(wb), Position(0, 0))
			.add_element(single_building_element(wb), Position(0, 2))
			.add_element(single_building_element(wp), Position(0, 1))
			.add_element(single_building_element(wp), Position(1, 1))
			.add_element(single_building_element(wcu), Position(2, 1))
			.add_element(single_building_element(wcu), Position(2, 2))
			.add_element(single_building_element(wcu), Position(2, 3))
			.add_element(single_building_element(wb), Position(3, 4))
			.add_element(single_building_element(wcy), Position(3, 0))
			.add_element(single_building_element(wcy), Position(3, 1))
			.add_element(single_building_element(wcy), Position(3, 2))
			.add_element(single_building_element(wcy), Position(3, 3))
			.add_element(single_building_element(mcy), Position(4, 2))
			.add_element(single_building_element(bb, rotation), Position(4, 1))
			.add_element(single_building_element(pfb), Position(5, 1))
			.add_element(single_building_element(yfb), Position(6, 1))
			.add_element(single_building_element(bb, rotation), Position(4, 0))
			.add_element(single_building_element(pfb), Position(5, 0))
			.add_element(single_building_element("red flat block"), Position(6, 0))
			.create_building(Eigen::Vector3f::Zero(), Eigen::Quaternionf::Identity());

		/*.add_element(composed_building_element(
{ "wooden bridge", "purple semicylinder",
  "purple bridge", "wooden semicylinder" }), Position(4, 0))*/

		using namespace simulation;

		auto hand = test.building->get_network()->create_place(true);

		auto token_traces = building_element::get_token_traces();
		auto env = std::make_shared<environment>(
			test.building->get_network(),
			test.building->get_distribution(),
			token_traces);

		env->additional_scene_objects.push_back(std::make_shared<simulated_table>());
		auto init_goal = std::make_pair(env->net->create_place(), std::make_shared<pn_token>());

		/*auto robot0 = std::make_shared<agent>(env,
			Eigen::Vector3f(0.15f, simulated_table::width / 2 + 0.1f, 0),
			Eigen::Vector3f(0, -2. / 10 * simulated_table::width, 0),
			[r0_hand](pn_transition::Ptr trans, agent& robot) {
				for (const auto& in : trans->inputs)
					if (in.first == r0_hand)
						return 1;

				for (const auto& out : trans->outputs)
					if (out.first == r0_hand)
						return 1;

				return 0;
		});*/


		//auto abc = agent(env->net);
		//auto constructive = abc.get_constructive_transitions();

		float startX = 0.1f;
		float startY = 0.1f;

		int temp = 0;

		int i = 0;

		for (const auto& building_element : test.building->visualize())
		{
			obb tempObb(building_element->obbox.diagonal, Eigen::Vector3f(startX, startY, building_element->obbox.diagonal.z() * 0.5),
				building_element->obbox.rotation);
			//obb& tempObb = building_element->obbox;
			auto instance = env->add_object(building_element->token->object, tempObb);
			env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));

			//const auto& agent_places = env->net->get_agent_places();



			//env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));
			//Transition to dummy agent place
			/*([&]()
			{
				for (auto s = constructive.begin(); s < constructive.end(); ++s)
				{
					const auto& filtered = (*s)->get_inputs({ instance.second });
					if (filtered.size() == 0)
						continue;
					for (const auto& filter_place : filtered) {
						if (std::dynamic_pointer_cast<pn_agent_place>(filter_place)) {
							env->net->add_transition(std::make_shared<pn_transition>(std::vector<pn_instance>({ {hand, instance.second } }), std::vector<pn_instance>({ {filter_place, instance.second} })));
							constructive.erase(s);
							return;
						}
					}
				}
			})();*/

			startX += 0.08;
			if (startX > 0.5)
			{
				startX = 0.1f;
				startY += 0.05;
			}
			++i;
		};

		std::vector<place_action::Ptr> ground_actions;

		for (const auto& transition : env->net->get_transitions())
		{
			if (auto action = std::dynamic_pointer_cast<place_action>(transition))
			{
				ground_actions.push_back(action);
			}
		}

		/*auto human = std::make_shared<simulation::agent>(env,
			Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
			Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
			hand,
			[ground_actions, token_traces](const pn_transition::Ptr& t, agent&) mutable
		{
			if (!ground_actions.empty())
			{
				if (ground_actions.front() == t)
				{
					ground_actions.erase(ground_actions.begin());
					return 1.;
				}

				if (std::dynamic_pointer_cast<pick_action>(t))
				{
					const auto token = t->outputs.begin()->second;

					for (const auto& action : ground_actions)
					{
						if (!action->get_inputs({ token }).empty())
						{
							return 1.;
						}
					}
				}
			}

			return 0.;
		});*/

		auto human = std::make_shared<simulation::human_agent>(env,
			Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
			Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
			hand,
[actions(env->net->get_transitions()), env, next_action(pn_transition::Ptr())](const pn_transition::Ptr& t, agent&) mutable
			{
				if (!actions.empty())
				{
					if (next_action == t)
					{
						actions.erase(std::ranges::find(actions, next_action));
						next_action = nullptr;
						return 1.;
					}
					if (next_action)
						return 0.;

					if (auto action = std::dynamic_pointer_cast<pick_action>(t))
					{
						auto token = t->outputs.begin()->second;
						auto marking = env->get_marking();

						for (const auto& action : actions)
						{
							if (auto placing = std::dynamic_pointer_cast<place_action>(action))
							{
								if (token == placing->token && marking->would_enable(placing, { std::pair(placing->from, placing->token) }))
								{
									next_action = placing;
									return 1.;
								}
							}
							else if (auto stacking = std::dynamic_pointer_cast<stack_action>(action))
							{
								const auto& instance = stacking->from;
								if (token == instance.second && marking->would_enable(stacking, { instance }))
								{
									next_action = stacking;
									return 1.;
								}
							}
							else
							{
								//throw std::exception("action not supported");
							}
						}
					}
				}

		return 0.;
			});

		test.task = std::make_shared<sim_task>("Shah10_Structure1", env, std::vector<simulation::agent::Ptr>({human}), init_goal, test.building->get_goal());


		return std::make_shared<building_simulation_test>(std::move(test));

		//auto build = builder()
		/*	.add_element(single_building_element(bb), Position(0, 0))
			.add_element(single_building_element("yellow plank"), Position(0, 1))
			.add_element(single_building_element("yellow block"), Position(0, 2))
			.add_element(single_building_element("cyan plank"), Position(1, 1))
			.add_element(single_building_element("red small cylinder", rotation1), Position(2, 1))
			.add_element(single_building_element("red small cylinder", rotation1), Position(2, 2))
			.add_element(single_building_element("red small cylinder", rotation1), Position(2, 3))
			.add_element(single_building_element(yfb), Position(3, 0))
			.add_element(single_building_element("red plank"), Position(3, 1))*/
			/*.add_element(single_building_element(bb, rotation), Position(0, 0))
				.add_element(composed_building_element(
					{ "wooden bridge", "purple semicylinder",
					  "purple bridge", "wooden semicylinder" }), Position(1, 0))
				.add_element(single_building_element(bb, rotation), Position(2, 0))
				.create_building();*/


				/*for (const auto& constructive : agent_ge->get_constructive_transitions())
					{
						const auto& inputs = constructive->inputs;
						initial_marking->distribution.insert({ std::make_shared<pn_binary_marking>(net, inputs), 1.0 });
						// update(*initial_marking);
					}*/

					/*std::unique_ptr<enact_core::lockable_data_typed<building>> a = std::make_unique<enact_core::lockable_data_typed<building>>(building({
							{std::make_shared<single_building_element>("wooden block"),	Position(0,0)},
							{std::make_shared<single_building_element>("wooden block"), Position(0, 2)},
							{std::make_shared<single_building_element>("wooden block"), Position(3, 4)},
							{std::make_shared<single_building_element>("wooden plank"), Position(0, 1)},
							{std::make_shared<single_building_element>("wooden plank"), Position(1, 1)},
							{std::make_shared<single_building_element>("wooden cube"), Position(2, 1)},
							{std::make_shared<single_building_element>("wooden cube"), Position(2, 2)},
							{std::make_shared<single_building_element>("wooden cube"), Position(2, 3)},
							{std::make_shared<single_building_element>("wooden cylinder"), Position(3, 0)},
							{std::make_shared<single_building_element>("wooden cylinder"), Position(3, 1)},
							{std::make_shared<single_building_element>("wooden cylinder"), Position(3, 2)},
							{std::make_shared<single_building_element>("wooden cylinder"), Position(3, 3)},
							{std::make_shared<single_building_element>("magenta cylinder"), Position(4, 2)},
							{std::make_shared<single_building_element>("blue block", Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 4.f), 0, std::sin(std::numbers::pi_v<float> / 4.f) * 1.f, 0)), Position(4, 1)},
							{std::make_shared<single_building_element>("purple flat block"), Position(5, 1)},
							{std::make_shared<single_building_element>("yellow flat block"), Position(6, 1)},
							{std::make_shared<composed_building_element>(composed_building_element(
								{single_building_element("wooden bridge"), single_building_element("purple semicylinder"),
								 single_building_element("purple bridge"), single_building_element("wooden semicylinder")})), Position(4, 0)}
							}));*/
	}

	const obb building_benchmark::structure_pose(
		Eigen::Vector3f::Ones(),
		Eigen::Vector3f(0.6f + +0.015f, -0.075f * (1.0 + building::spacing), 0.f),
		Eigen::Quaternionf(Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f(0.f, 0.f, 1.f))));

	const Eigen::Quaternionf building_benchmark::quarter_rot_z(Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitZ()));

	std::vector<pn_object_instance> building_benchmark::init_resource_pool(const pn_net::Ptr& net, const object_prototype_loader& loader)
	{
		const auto& prototypes = loader.get_prototypes();

		std::map<std::string, pn_object_token::Ptr> tokens;
		std::vector<pn_object_instance> pool;

		// x and y position in decimeter
		auto add = [&](int x, int y, const pn_object_token::Ptr& token)
		{
			const auto& obj_box = token->object->get_bounding_box();

			auto place = std::make_shared<pn_boxed_place>(
				aabb(
					obj_box.diagonal,
					Eigen::Vector3f(x * 0.1f, y * 0.101f, 0.5f * obj_box.diagonal.z())
				));

			net->add_place(place);
			pool.emplace_back(place, token);

			for (auto& agent : net->get_agent_places())
			{
				// picking / placing from / to the resource pool must not use empty tokens
				net->add_transition(std::make_shared<pick_action>(token, place, agent));
				net->add_transition(std::make_shared<place_action>(token, agent, place));
			}
		};

		auto add_row = [&](const std::string& type, int y)
		{
			auto token_iter = tokens.find(type);
			if (token_iter == tokens.end())
				token_iter = tokens.emplace(type, std::make_shared<pn_object_token>(loader.get(type))).first;


			for (int x = 2; x <= 6; x++)
				add(x, y, token_iter->second);
		};

		tokens.emplace("wooden block", std::make_shared<pn_object_token>(loader.get("wooden block")));
		tokens.emplace("wooden cube", std::make_shared<pn_object_token>(loader.get("wooden cube")));

		add(3, -2, tokens.at("wooden block"));
		add(5, -2, tokens.at("wooden cube"));

		//add_row("wooden block", -5);
		//add_row("wooden block horizontal", -4);
		//add_row("wooden cube", -3);

		/*
		for (int x = 3; x <= 6; x++)
			add(x, -2, tokens.at("wooden cube"));*/

		/*
		auto rc = tokens.emplace("red cube", std::make_shared<pn_object_token>(loader.get("red cube"))).first;
		for (int x = 3; x <= 6; x++)
			add(x, 2, rc->second);

		auto rb = tokens.emplace("red block horizontal", std::make_shared<pn_object_token>(loader.get("red block horizontal"))).first;*
		for (int y = 3; y <= 5; y++)
		{
			const auto& obj_box = rb->second->object->get_bounding_box();

			auto place = std::make_shared<pn_boxed_place>(
				aabb(
					obj_box.diagonal,
					Eigen::Vector3f(0.201f, y * 0.101f, 0.5f * obj_box.diagonal.z())
				));

			net->add_place(place);
			pool.emplace_back(place, rb->second);

			for (auto& agent : net->get_agent_places())
			{
				// picking / placing from / to the resource pool must not use empty tokens
				net->add_transition(std::make_shared<pick_action>(rb->second, place, agent));
				net->add_transition(std::make_shared<place_action>(rb->second, agent, place));
			}
		}
		
		auto gc = tokens.emplace("green cube", std::make_shared<pn_object_token>(loader.get("green cube"))).first;
		for (int x = 3; x <= 6; x++)
			add(x, 3, gc->second);

		auto pc = tokens.emplace("purple cube", std::make_shared<pn_object_token>(loader.get("purple cube"))).first;
		for (int x = 3; x <= 6; x++)
			add(x, 4, pc->second);

		auto bb = tokens.emplace("blue block", std::make_shared<pn_object_token>(loader.get("blue block"))).first;
		for (int x = 3; x <= 5; x++)
			add(x, 5, bb->second);
			*/

		building_element::load_building_elements(pool);

		return pool;
	}

	std::map<std::string, pn_object_token::Ptr> building_benchmark::named_tokens(const std::vector<pn_object_instance>& resource_pool)
	{
		std::map<std::string, pn_object_token::Ptr> lookup;
		for (const auto& objToken : resource_pool | std::views::values)
			lookup.try_emplace(objToken->object->get_name(), objToken);

		building::set_building_elements(lookup);

		return lookup;
	}

	std::map<state_observation::object_prototype::ConstPtr, state_observation::pn_object_token::Ptr> building_benchmark::get_token_traces(const std::vector<pn_object_instance>& resource_pool)
	{
		std::map<state_observation::object_prototype::ConstPtr, state_observation::pn_object_token::Ptr> trace;
		for (const auto& objToken : resource_pool | std::views::values)
			trace.try_emplace(objToken->object, objToken);

		return trace;
	}

	pn_instance building_benchmark::decompose(const pn_net::Ptr& net, const std::vector<pn_object_instance>& resource_pool)
	{
		auto goal = net->create_place();

		auto goal_instance = std::make_pair(goal, std::make_shared<pn_token>());

		std::vector<pn_instance> output_instances(resource_pool.begin(), resource_pool.end());
		output_instances.push_back(goal_instance);

		net->create_transition(std::vector<pn_instance>(resource_pool.begin(), resource_pool.end()),
			std::move(output_instances));

		return goal_instance;
	}

	std::map<state_observation::pn_place::Ptr, state_observation::pn_token::Ptr> building_benchmark::to_marking(
		const state_observation::pn_net::Ptr& net, 
		const std::vector<state_observation::pn_object_instance>& resource_pool, 
		const state_observation::building& building)
	{
		std::map<state_observation::pn_place::Ptr, state_observation::pn_token::Ptr> distribution;

		for (const auto& entry : building.get_distribution())
			distribution.emplace(entry);

		for (const auto& instance : resource_pool)
			distribution.emplace(instance.first, instance.second);

		return distribution;
		//return std::make_shared<pn_binary_marking>(net, std::move(distribution));
	}

	sim_task::Ptr building_benchmark::building_1(const object_parameters& object_params, object_prototype_loader loader)
	{
		auto net = std::make_shared<state_observation::pn_net>(object_params);
		auto r_hand = net->create_place(true);
		/*auto r0_hand = net->create_place(true);
		auto h1_hand = net->create_place(true);
		auto r1_hand = net->create_place(true);*/


		//
		//auto ressource_pool = init_resource_pool(net, loader);
		//auto name_to_token = named_tokens(ressource_pool);
		//auto decomposition_goal = decompose(net, ressource_pool);



		std::string wb = "wooden block";
		std::string wbh = "wooden block horizontal";
		std::string wcu = "wooden cube";
		std::string rb = "red block horizontal";
		std::string bb = "blue block";

		building_element::load_building_elements(loader);
		building::load_building_elements(loader.get_prototypes());

		builder b;
		b
			.add_element(single_building_element(wb), Position(0, 0))
			.add_element(single_building_element(wcu), Position(0, 1))
			.add_element(single_building_element(wbh, quarter_rot_z), Position(0, 2))
			.add_element(single_building_element(wb), Position(0, 3))

			.add_element(single_building_element(wcu), Position(1, 1))
			.add_element(single_building_element(wcu), Position(1, 2))
			.add_element(single_building_element(wcu), Position(1, 3))

			.add_element(single_building_element(bb), Position(2, 0))
			.add_element(single_building_element(wb), Position(2, 1))
			.add_element(single_building_element(wb), Position(2, 2))
			.add_element(single_building_element(wb), Position(2, 3))
			.add_element(single_building_element(bb), Position(2, 4))

			.add_element(single_building_element(wbh, quarter_rot_z), Position(3, 0))
			.add_element(single_building_element(rb, quarter_rot_z), Position(3, 1))
			.add_element(single_building_element(bb), Position(3, 2))

			.add_element(single_building_element(rb, quarter_rot_z), Position(4, 0))
			.add_element(single_building_element("purple cube"), Position(4, 1))
			.add_element(single_building_element("green cube"), Position(4, 2));


		const auto build = b.create_building(structure_pose.translation, structure_pose.rotation, net);

		//auto init_marking = std::make_shared<pn_belief_marking>(to_marking(net, ressource_pool, *build));
		//net->set_goal(decomposition_goal.first);

		//auto env = std::make_shared<environment>(net, to_marking(net, ressource_pool, *build), get_token_traces(ressource_pool));


		auto env = std::make_shared<environment>(net, build->get_distribution(), building_element::get_token_traces());

		/*
		int a = 0;
		for (const auto& d : build->get_distribution())
		{
			if (typeid(*d.second) == typeid(pn_empty_token))
				++a;
		}
		std::cout << "There are: " << a << " empty tokens!" << std::endl;
		*/
		env->additional_scene_objects.push_back(std::make_shared<simulated_table>());
		std::vector<pn_instance> resources;

		auto addBlockInstance = [&](int x, int y, const std::string& protoName)
		{
			auto proto = loader.get(protoName);
			auto boxDiag = proto->get_bounding_box().diagonal;
			auto inst = env->add_object(proto, obb{ boxDiag, Eigen::Vector3f(x * 0.1f, y * 0.101f, 0.5f * boxDiag.z()) });
			resources.push_back(inst);
			for (auto& agent : env->net->get_agent_places())
			{
				// picking / placing from / to the resource pool must not use empty tokens
				env->net->add_transition(std::make_shared<pick_action>(inst.second, inst.first, agent));
				env->net->add_transition(std::make_shared<place_action>(inst.second, agent, inst.first));
			}
		};

		addBlockInstance(2, -3, wb);
		addBlockInstance(4, -3, wb);
		addBlockInstance(3, -3, wcu);
		addBlockInstance(6, -3, wbh);

		addBlockInstance(2, 0, wcu);
		addBlockInstance(2, 1, wcu);
		addBlockInstance(2, 2, wcu);

		addBlockInstance(2, 3, bb);
		addBlockInstance(2, 4, bb);
		addBlockInstance(4, 0, wb);
		addBlockInstance(4, 1, wb);
		addBlockInstance(4, 2, wb);

		addBlockInstance(4, 3, wbh);
		addBlockInstance(4, 4, rb);
		addBlockInstance(4, 5, bb);

		addBlockInstance(5, 0, rb);
		addBlockInstance(5, 1, "purple cube");
		addBlockInstance(5, 2, "green cube");



		//env->add

		// create agents
		auto robot = std::make_shared<human_agent>(env,
			Eigen::Vector3f(0, simulated_table::width / 2.f + 0.1f, 0),
			Eigen::Vector3f(0, -2.f / 10.f * simulated_table::width, 0),
			r_hand);

		return std::make_shared<sim_task>("Shah10_Structure4", env, std::vector<agent::Ptr>({ robot }), riedelbauch17::create_goal(*env->net, std::move(resources)), build->get_goal());
	}

}

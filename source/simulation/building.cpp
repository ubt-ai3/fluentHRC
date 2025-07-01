#include "building.hpp"

namespace simulation
{
	std::shared_ptr<building_simulation_test> create_building_simulation_task(
		const object_parameters& object_params,
		object_prototype_loader loader)
	{
		building_simulation_test test;
		const auto& prototypes = loader.get_prototypes();

		std::vector<pn_object_token::Ptr> tokens;
		for (const auto& proto : prototypes)
			tokens.emplace_back(std::make_shared<pn_object_token>(proto));

		building_element::load_building_elements(tokens);
		composed_building_element::load_type_diags();
		building::load_building_elements(prototypes);

		std::string wb = "wooden block";
		std::string wp = "wooden plank";
		std::string wcu = "wooden cube";
		std::string wcy = "wooden cylinder";
		std::string mcy = "magenta cylinder";
		std::string bb = "blue block";
		std::string pfb = "purple flat block";
		std::string yfb = "yellow flat block";

		const auto& rotation = Eigen::Quaternionf(std::cosf(std::_Pi / 4.f), 0, std::sinf(std::_Pi / 4.f) * 1.f, 0);
		//const auto& rotation1 = Eigen::Quaternionf(std::cos(std::_Pi / 4.f), 0, 0, std::sin(std::_Pi / 4.f) * 1.f);

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

		auto human = std::make_shared<simulation::agent>(env,
			Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
			Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
			hand,
			[actions(env->net->get_transitions()), env, next_action(pn_transition::Ptr())](const pn_transition::Ptr& t, agent&) mutable
			{
				if (!actions.empty())
				{
					if (next_action == t)
					{
						actions.erase(std::find(actions.begin(), actions.end(), next_action));
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

		test.task = std::make_shared<sim_task>(env, std::vector<simulation::agent::Ptr>({ human }));


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
							{std::make_shared<single_building_element>("blue block", Eigen::Quaternionf(std::cos(std::_Pi / 4.f), 0, std::sin(std::_Pi / 4.f) * 1.f, 0)), Position(4, 1)},
							{std::make_shared<single_building_element>("purple flat block"), Position(5, 1)},
							{std::make_shared<single_building_element>("yellow flat block"), Position(6, 1)},
							{std::make_shared<composed_building_element>(composed_building_element(
								{single_building_element("wooden bridge"), single_building_element("purple semicylinder"),
								 single_building_element("purple bridge"), single_building_element("wooden semicylinder")})), Position(4, 0)}
							}));*/
	}
}
#include "baraglia17.hpp"

#include <state_observation/object_prototype_loader.hpp>
#include <state_observation/pn_model_extension.hpp>

using namespace state_observation;

namespace simulation
{

	pn_instance baraglia17::create_goal(pn_net& net,
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

pn_instance baraglia17::create_goal(pn_net& net,
	const std::set<pn_instance>& instances)
{
	return create_goal(net, std::vector<pn_instance>(instances.begin(), instances.end()));
}

pn_instance baraglia17::create_goal(pn_net& net,
	std::initializer_list<pn_object_instance> instances)
{
	return create_goal(net, std::vector<pn_instance>(instances.begin(), instances.end()));
}

sim_task::Ptr simulation::baraglia17::practice(const object_parameters& object_params,
	object_prototype_loader loader)
{

	auto env = std::make_shared<environment>(object_params);
	env->additional_scene_objects.push_back(std::make_shared<simulated_table>());

	auto get_box = [](int x, int y, const object_prototype::ConstPtr& obj) {
		return aabb(obj->get_bounding_box().diagonal,
			Eigen::Vector3f(x * simulated_table::width / 5, y * simulated_table::breadth / 5, obj->get_bounding_box().diagonal.z() / 2));
	};

	// create initial objects
	auto cube = loader.get("red cube");
	auto r0 = env->add_object(loader.get("yellow cube"), get_box(-2, 2, cube));
	auto h0 = env->add_object(loader.get("cyan cube"), get_box(2, -1, cube));

	auto init_goal = create_goal(*env->net, { r0, h0 });

	// create target locations
	std::vector<pn_object_token::Ptr> tokens({ r0.second, h0.second });
	std::vector<pn_boxed_place::Ptr> locations({ r0.first, h0.first });

	auto h_hand = env->net->create_place(true);
	auto r_hand = env->net->create_place(true);

	auto l = env->add_location(get_box(-1, 0, cube));
	auto r = env->add_location(get_box(1, 0, cube));

	locations.push_back(l);
	locations.push_back(r);

	std::set<pn_instance> goal_instances({
			std::make_pair(l, r0.second),
			std::make_pair(r, h0.second)
		});

	// create actions
	for (const pn_boxed_place::Ptr& loc : locations)
	{
		if (loc->box.translation.y() <= 3. / 10 * simulated_table::breadth)
		{
			//human reach zone

			for (const auto& tok : tokens)
			{
				// do not pick objects from the target location
				if (!goal_instances.contains(std::make_pair(loc, tok)))
					env->net->add_transition(std::make_shared<pick_action>(tok, loc, h_hand));

				env->net->add_transition(std::make_shared<place_action>(tok, h_hand, loc));
			}
		}

		if (loc->box.translation.y() >= -3. / 10 * simulated_table::breadth)
		{
			//robot reach zone

			for (const auto& tok : tokens)
			{
				if (!goal_instances.contains(std::make_pair(loc, tok)))
					env->net->add_transition(std::make_shared<pick_action>(tok, loc, r_hand));

				env->net->add_transition(std::make_shared<place_action>(tok, r_hand, loc));
			}
		}
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

	return std::make_shared<sim_task>("Baraglia17_Practice", env, std::vector<agent::Ptr>({robot, human}), init_goal, create_goal(*env->net, goal_instances));
}

sim_task::Ptr simulation::baraglia17::task_a_1(const object_parameters& object_params,
	object_prototype_loader loader)
{
	auto env = std::make_shared<environment>(object_params);
	env->additional_scene_objects.push_back(std::make_shared<simulated_table>());

	auto get_box = [](int x, int y, const object_prototype::ConstPtr& obj) {
		return aabb(obj->get_bounding_box().diagonal,
			Eigen::Vector3f(x * simulated_table::width / 5, y * simulated_table::breadth / 5, obj->get_bounding_box().diagonal.z() / 2));
	};

	// create initial objects
	auto cube = loader.get("red cube");
	auto rb = env->add_object(loader.get("red cube"), get_box(-2, 1, cube));
	auto yb = env->add_object(loader.get("yellow cube"), get_box(-2, -1, cube));
	auto bb = env->add_object(loader.get("cyan cube"), get_box(0, 0, cube));
	auto wb = env->add_object(loader.get("wooden cube"), get_box(2, 2, cube));

	auto init_goal = create_goal(*env->net, { rb, yb, bb, wb });

	// create target locations
	std::vector<pn_object_token::Ptr> tokens({ rb.second, yb.second, bb.second, wb.second });
	std::vector<pn_boxed_place::Ptr> locations({ rb.first, yb.first, bb.first, wb.first });


	auto h_hand = env->net->create_place(true);
	auto r_hand = env->net->create_place(true);

	auto b = env->add_location(get_box(0, 1, cube));
	auto r = env->add_location(get_box(1, 0, cube));
	auto f = env->add_location(get_box(0, -1, cube));
	auto l = env->add_location(get_box(-1, 0, cube));

	locations.push_back(l);
	locations.push_back(r);
	locations.push_back(b);
	locations.push_back(f);

	std::set<pn_instance> goal_instances({
			std::make_pair(b, rb.second),
			std::make_pair(r, bb.second),
			std::make_pair(f, wb.second),
			std::make_pair(l, yb.second)
		});

	// create actions
	for (const pn_boxed_place::Ptr& loc : locations)
	{
		if (loc->box.translation.y() <= 3. / 10 * simulated_table::breadth)
		{
			//human reach zone

			for (const auto& tok : tokens)
			{
				// do not pick objects from the target location
				if (!goal_instances.contains(std::make_pair(loc, tok)))
					env->net->add_transition(std::make_shared<pick_action>(tok, loc, h_hand));

				env->net->add_transition(std::make_shared<place_action>(tok, h_hand, loc));
			}
		}

		if (loc->box.translation.y() >= -3. / 10 * simulated_table::breadth)
		{
			//robot reach zone

			for (const auto& tok : tokens)
			{
				if (!goal_instances.contains(std::make_pair(loc, tok)))
					env->net->add_transition(std::make_shared<pick_action>(tok, loc, r_hand));

				env->net->add_transition(std::make_shared<place_action>(tok, r_hand, loc));
			}
		}
	}

	std::map<pn_token::Ptr, pn_place::Ptr> targets;
	for (const pn_instance& instance : goal_instances)
		targets.emplace(instance.second, instance.first);

	auto closer_to = [targets](const pn_token::Ptr& tok, const Eigen::Vector3f& place_loc, const Eigen::Vector3f& pick_loc)
	{
		const auto target = std::dynamic_pointer_cast<pn_boxed_place>(targets.find(tok)->second);
		const Eigen::Vector3f& dest = target->box.translation;
		return (place_loc - dest).norm() < (pick_loc - dest).norm();
	};

	// create agents
	auto robot = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0, simulated_table::width / 2 + 0.1f, 0),
		Eigen::Vector3f(0, -2. / 10 * simulated_table::width, 0),
		r_hand,
		[closer_to](pn_transition::Ptr trans, agent& robot) {
			if (const auto place = std::dynamic_pointer_cast<place_action>(trans)) {
				if(!closer_to(place->outputs.begin()->second,place->to->box.translation, robot.pick_location))
					return 0;
			}
			return 1;
		});


	auto human = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0, -simulated_table::width / 2 - 0.1f, 0),
		Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
		h_hand,
		[closer_to](pn_transition::Ptr trans, agent& human) {
			if (const auto place = std::dynamic_pointer_cast<place_action>(trans)) {
				if (!closer_to(place->outputs.begin()->second, place->to->box.translation, human.pick_location))
					return 0;
			}

			return 1;
		});


	return std::make_shared<sim_task>("Baraglia17_A1", env, std::vector<agent::Ptr>({ robot, human }), init_goal, create_goal(*env->net, goal_instances));
}


sim_task::Ptr simulation::baraglia17::task_b_1(const object_parameters& object_params,
	object_prototype_loader loader)
{

	auto env = std::make_shared<environment>(object_params);
	env->additional_scene_objects.push_back(std::make_shared<simulated_table>());

	auto get_box = [](int x, int y, const object_prototype::ConstPtr& obj) {
		return aabb(obj->get_bounding_box().diagonal,
			Eigen::Vector3f(x * simulated_table::width / 5, y * simulated_table::breadth / 5, obj->get_bounding_box().diagonal.z() / 2));
	};

	// objects
	auto yc = loader.get("yellow cube");
	auto pc = loader.get("purple cube");
	auto cc = loader.get("cyan cube");
	auto rc = loader.get("red cube");

	// create initial objects
	std::vector<pn_object_instance> picks = { env->add_object(rc, get_box(-2, 0,rc)),
	 env->add_object(yc, get_box(-2, -2,yc)),
	 env->add_object(pc, get_box(2, 2,pc)),
	 env->add_object(yc, get_box(2,1,yc)),
	 env->add_object(cc, get_box(2,-1,cc)),
	 env->add_object(rc, get_box(2, -2, rc))
	};

	auto init_goal = create_goal(*env->net, std::vector<pn_instance>(picks.begin(), picks.end()));

	auto h_hand = env->net->create_place(true);
	auto r_hand = env->net->create_place(true);

	auto r = env->add_location(get_box(1, 0,pc));
	auto l = env->add_location(get_box(-1, 0,rc));

	std::set<pn_instance> goal_instances;

	// create actions
	for (const auto& loc : picks)
	{
		if (loc.first->box.translation.y() <= 3. / 10 * simulated_table::breadth)
		{
			//human reach zone
			env->net->add_transition(std::make_shared<place_action>(loc.second, h_hand, loc.first));
			env->net->add_transition(std::make_shared<pick_action>(loc.second, loc.first, h_hand));
		}

		if (loc.first->box.translation.y() >= -3. / 10 * simulated_table::breadth)
		{
			//robot reach zone
			env->net->add_transition(std::make_shared<place_action>(loc.second, r_hand, loc.first));
			env->net->add_transition(std::make_shared<pick_action>(loc.second, loc.first, r_hand));
		}
	}

	for (auto& agent : { h_hand, r_hand })
	{
		// bowl with object
		env->net->add_transition(std::make_shared<place_action>(env->token_traces.at(loader.get("purple cube")), agent, r));
		goal_instances.emplace(stack_action::create(
			env->net, env->token_traces, agent, r, loader.get("purple cube"), loader.get("yellow cube")
		)->to);

		env->net->add_transition(std::make_shared<place_action>(env->token_traces.at(loader.get("red cube")), agent, l));
		auto top = stack_action::create(
			env->net, env->token_traces, agent, l, loader.get("red cube"), loader.get("yellow cube")
		)->to.first;
		top = stack_action::create(
			env->net, env->token_traces, agent, top, loader.get("yellow cube"), loader.get("cyan cube")
		)->to.first;
		goal_instances.emplace(stack_action::create(
			env->net, env->token_traces, agent, top, loader.get("cyan cube"), loader.get("red cube")
		)->to);
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


	return std::make_shared<sim_task>("Baraglia17_B1", env, std::vector<agent::Ptr>({ robot, human }), init_goal, create_goal(*env->net, goal_instances));
}

}

#include "hoellerich22.hpp"

#include <state_observation/object_prototype_loader.hpp>
#include <state_observation/pn_model_extension.hpp>

#include "baraglia17.hpp"
#include <state_observation/building_estimation.hpp>

using namespace state_observation;

namespace simulation
{

const obb hoellerich22::structure_pose(
	Eigen::Vector3f::Ones(),
	Eigen::Vector3f(0.6f + +0.015f, -0.075f * (1 + building::spacing), 0.f),
	Eigen::Quaternionf(Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f(0.f, 0.f, 1.f))));

const Eigen::Quaternionf hoellerich22::quarter_rot_z(Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitZ()));

pn_instance hoellerich22::create_goal(pn_net& net,
	const std::vector<pn_object_instance>& instances)
{
	auto goal = net.create_place();

	auto goal_token = std::make_shared<pn_token>();
	std::vector<pn_instance> input_instances(instances.begin(), instances.end());
	std::vector<pn_instance> output_instances(instances.begin(), instances.end());
	output_instances.push_back(std::make_pair(goal, goal_token));

	net.create_transition(std::move(input_instances),
		std::move(output_instances));

	return std::make_pair(goal, goal_token);
}

std::vector<pn_object_instance> hoellerich22::init_resource_pool(const environment::Ptr& env, const object_prototype_loader& loader)
{
	const auto& prototypes = loader.get_prototypes();
	const auto& net = env->net;


	std::vector<pn_object_instance> pool;

	// x and y position in decimeter
	auto add = [&](int x, int y, const std::string& type)
	{
		const auto& prototype = loader.get(type);
		const auto& obj_box = prototype->get_bounding_box();

		pool.push_back(env->add_object(prototype, aabb(
			obj_box.diagonal,
			Eigen::Vector3f(x * 0.1f, y * 0.1f, 0.5f * obj_box.diagonal.z())
		)));
		const auto& place = pool.back().first;
		const auto& token = pool.back().second;

		for (auto& agent : net->get_agent_places())
		{
			// picking / placing from / to the resource pool must not use empty tokens
			net->add_transition(std::make_shared<pick_action>(token, place, agent));
			net->add_transition(std::make_shared<place_action>(token, agent, place));
		}
	};

	auto add_row = [&](const std::string& type, int y)
	{

		for (int x = 2; x <= 6; x++)
			add(x, y, type);
	};

	//add_row("wooden block", -5);
	add_row("wooden block horizontal", -4);
	add_row("wooden cube", -3);

	for (int x = 3; x <= 6; x++)
		add(x, -2, "wooden cube");

	for (int x = 3; x <= 6; x++)
		add(x, 2, "red cube");

	for (int y = 3; y <= 5; y++)
	{
		auto rb = loader.get("red block horizontal");
		const auto& obj_box = rb->get_bounding_box();

		pool.push_back(env->add_object(rb, aabb(
			obj_box.diagonal,
			Eigen::Vector3f(0.201f, y * 0.101f, 0.5f * obj_box.diagonal.z())
		)));

		const auto& place = pool.back().first;
		const auto& token = pool.back().second;

		for (auto& agent : net->get_agent_places())
		{
			// picking / placing from / to the resource pool must not use empty tokens
			net->add_transition(std::make_shared<pick_action>(token, place, agent));
			net->add_transition(std::make_shared<place_action>(token, agent, place));
		}
	}

	for (int x = 3; x <= 6; x++)
		add(x, 3, "green cube");

	for (int x = 3; x <= 6; x++)
		add(x, 4, "purple cube");

	for (int x = 3; x <= 5; x++)
		add(x, 5, "blue block");

	std::map<std::string, pn_object_token::Ptr> tokens;
	for (const auto& entry : env->token_traces)
		tokens.emplace(entry.first->get_name(), entry.second);

	building_element::load_building_elements(pool);
	building::set_building_elements(std::move(tokens));

	return pool;
}

sim_task::Ptr hoellerich22::structure_1(const state_observation::object_parameters& object_params, state_observation::object_prototype_loader loader)
{
	auto env = std::make_shared<environment>(object_params);
	env->additional_scene_objects.push_back(std::make_shared<simulated_table>());
	auto r_hand = env->net->create_place(true);
	auto h_hand = env->net->create_place(true);

	auto pool = init_resource_pool(env, loader);


	std::string wc = "wooden cube";
	std::string rc = "red cube";
	std::string rb = "red block horizontal";
	std::string gc = "green cube";
	std::string pc = "purple cube";

	builder b;

	b.add_single_element(rc, Position(0, 0))
		.add_single_element(wc, Position(0, 1))
		.add_single_element(rc, Position(0, 2))
		.add_element(single_building_element(rb, quarter_rot_z), Position(0, 3))

		.add_single_element(wc, Position(1, 0))
		.add_single_element(wc, Position(1, 1))
		.add_single_element(wc, Position(1, 2))
		.add_element(single_building_element("wooden block horizontal", quarter_rot_z), Position(1, 3))

		.add_single_element(rc, Position(2, 0))
		.add_single_element(wc, Position(2, 1))
		.add_single_element(rc, Position(2, 2))
		.add_element(single_building_element(rb, quarter_rot_z), Position(2, 3))

		.add_single_element(gc, Position(3, 0))
		.add_single_element(gc, Position(3, 1))
		.add_single_element(gc, Position(3, 2))
		.add_single_element(pc, Position(3, 3))
		.add_single_element(pc, Position(3, 4));

	auto building = b.create_building(structure_pose.translation, structure_pose.rotation, env->net);
	auto init_goal = create_goal(*env->net, pool); //overwrite with initial goal

	for (const auto& entry : building->get_distribution())
		env->distribution.emplace(entry);

	// create agents
	auto robot = std::make_shared<robot_agent>(env,
		//Eigen::Vector3f(0, 0.f, 0),
//		Eigen::Vector3f( 2. / 10 * simulated_table::width, 0, 0),
		r_hand);


	auto human = std::make_shared<human_agent>(env,
		Eigen::Vector3f(0.9f,0, 0),
		Eigen::Vector3f(-2. / 10 * simulated_table::width, 0, 0),
		h_hand);


	return std::make_shared<sim_task>("Hoellerich22_Structure1", env, std::vector<agent::Ptr>({ robot, human }), init_goal, building->get_goal());
}

}
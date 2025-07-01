#include "benchmark.hpp"

#include <filesystem>
#include <ranges>

#include <boost/date_time/posix_time/posix_time.hpp>

#include <state_observation/pn_model_extension.hpp>
#include <state_observation/object_prototype_loader.hpp>

#include <state_observation/building_estimation.hpp>

#include <simulation/task.hpp>

using namespace state_observation;
using namespace std::chrono;

const obb benchmark::structure_pose(
	Eigen::Vector3f::Ones(),
	Eigen::Vector3f(0.6f + +0.015f, -0.075f * (1 + building::spacing), 0.f),
	Eigen::Quaternionf(Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f(0.f, 0.f, 1.f))));

const Eigen::Quaternionf benchmark::quarter_rot_z(Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitZ()));


pn_net::Ptr benchmark::init_net(const state::object_parameters& object_params, int count_agents)
{
	auto net = std::make_shared<pn_net>(object_params);
	std::vector<pn_place::Ptr> agents;
	agents.reserve(count_agents);

	for (int i = 0; i < count_agents; ++i)
		agents.emplace_back(net->create_place(true));

	return net;
}

std::vector<state::pn_object_instance> benchmark::init_resource_pool(const state::pn_net::Ptr& net, const state::object_prototype_loader& loader)
{
	const auto& prototypes = loader.get_prototypes();

	std::map<std::string, pn_object_token::Ptr> tokens;
	std::vector<state::pn_object_instance> pool;

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

	//add_row("wooden block", -5);
	add_row("wooden block horizontal", -4);
	add_row("wooden cube", -3);
	
	for (int x = 3; x <= 6; x++)
		add(x, -2, tokens.at("wooden cube"));

	auto rc = tokens.emplace("red cube", std::make_shared<pn_object_token>(loader.get("red cube"))).first;
	for (int x = 3; x <= 6; x++)
		add(x, 2, rc->second);

	auto rb = tokens.emplace("red block horizontal", std::make_shared<pn_object_token>(loader.get("red block horizontal"))).first;
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
		add(x,3, gc->second);

	auto pc = tokens.emplace("purple cube", std::make_shared<pn_object_token>(loader.get("purple cube"))).first;
	for (int x = 3; x <= 6; x++)
		add(x, 4, pc->second);

	auto bb = tokens.emplace("blue block", std::make_shared<pn_object_token>(loader.get("blue block"))).first;
	for (int x = 3; x <= 5; x++)
		add(x, 5, bb->second);


	building_element::load_building_elements(pool);

	return pool;
}

state::pn_binary_marking::Ptr benchmark::to_marking(const state::pn_net::Ptr& net, 
	const std::vector<state::pn_object_instance>& resource_pool, 
	const std::map<std::string, std::shared_ptr<state::building>>& buildings)
{
	std::set<pn_instance> distribution;

	if (!buildings.empty())
		for (const auto& entry : buildings.begin()->second->get_distribution())
			distribution.emplace(entry);
	
	for (const auto& instance : resource_pool)
		distribution.emplace(instance.first, instance.second);

	return std::make_shared<pn_binary_marking>(net, std::move(distribution));
}

std::map<std::string, state::pn_object_token::Ptr> benchmark::named_tokens(const std::vector<state::pn_object_instance>& resource_pool)
{
	std::map<std::string, state::pn_object_token::Ptr> lookup;
	for (const auto& instance : resource_pool)
		lookup.try_emplace(instance.second->object->get_name(), instance.second);

	building::set_building_elements(lookup);

	return lookup;
}

state::pn_instance benchmark::decompose(const state::pn_net::Ptr& net, const std::vector<state::pn_object_instance>& resource_pool)
{
	auto goal = net->create_place();

	auto goal_instance = std::make_pair(goal, std::make_shared<pn_token>());

	std::vector<pn_instance> output_instances(resource_pool.begin(), resource_pool.end());
	output_instances.push_back(goal_instance);

	net->create_transition(std::vector<pn_instance>(resource_pool.begin(), resource_pool.end()),
		std::move(output_instances));

	return goal_instance;
}

state::pn_instance benchmark::pyramid(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token)
{
	auto wc = name_to_token.at("wooden cube");
	auto rc = name_to_token.at("red cube");

	const float size = wc->object->get_bounding_box().diagonal.x();
	std::map<pn_object_instance, std::vector<pn_object_instance>> pick_dependencies;

	std::vector<pn_instance> goal_side_conditions;

	auto get_instance = [&](const Eigen::Vector3f& center, const pn_object_token::Ptr& token)
	{
		const auto& obj_box = token->object->get_bounding_box();
		const auto box = aabb(
			obj_box.diagonal,
			center
		);

		auto place = net->get_place(box); 
		
		if (!place)
		{
			place = std::make_shared<pn_boxed_place>(box);
			net->add_place(place);
		}

		auto instance = std::make_pair(place, token);			
		pick_dependencies.emplace(instance, std::vector<pn_object_instance>());
		goal_side_conditions.emplace_back(instance);

		return instance;
	};

	auto get_stack_dependencies = [&](pn_object_instance& instance)
	{
		std::vector<pn_object_instance> dependencies;
		const auto& center = instance.first->box.translation;

		for (auto& entry : pick_dependencies)
		{
			const auto& other_center = entry.first.first->box.translation;

			if ((center.head<2>() - other_center.head<2>()).norm() > size ||
				std::abs(center.z() - other_center.z()) > size)
				continue;

			dependencies.emplace_back(entry.first);
			entry.second.emplace_back(instance);
		}

		return dependencies;
	};

	// first layer - white cubes 3 x 3
	for(int x = -1; x <= 1 ; x++)
		for (int y = -1; y <= 1; y++)
		{
			auto instance = get_instance(Eigen::Vector3f(x * size * 1.1f + 0.6f, y * size * 1.1, size / 2), wc);

			for (auto& agent : net->get_agent_places())
				net->add_transition(std::make_shared<place_action>(net, instance.second, agent, instance.first, true));
		}

	// second layer - red cubes 2 x 2
	for (int x = -1; x <= 1; x+=2)
		for (int y = -1; y <= 1; y += 2)
		{
			auto instance = get_instance(Eigen::Vector3f(0.5f * x * size * 1.1f + 0.6f, 0.5f * y * size * 1.1, 1.5f * size), rc);
			auto dependencies = get_stack_dependencies(instance);

			for (auto& agent : net->get_agent_places())
				stack_action::create(net, agent, dependencies, instance, true);
		}

	// third layer - one red cube
	auto instance = get_instance(Eigen::Vector3f(0.6f, 0.f, 2.5f * size), rc);
	auto dependencies = get_stack_dependencies(instance);

	for (auto& agent : net->get_agent_places())
		stack_action::create(net, agent, dependencies, instance, true);


	// create pick actions
	for (const auto& entry : pick_dependencies)
		if (!entry.second.empty())
			for (auto& agent : net->get_agent_places())
				reverse_stack_action::create(net, agent, entry.second, entry.first);

	std::vector<pn_instance> goal_output_instances(goal_side_conditions);

	auto goal_instance = std::make_pair(net->create_place(), std::make_shared<pn_token>());
	goal_output_instances.push_back(goal_instance);

	net->create_transition(std::move(goal_side_conditions), std::move(goal_output_instances));

	return goal_instance;
}


building::Ptr benchmark::flag_denmark(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token)
{
	std::string wc = "wooden cube";
	std::string rc = "red cube";
	std::string rb = "red block horizontal";
	std::string gc = "green cube";
	std::string pc = "purple cube";
		
	builder b;

	b.add_single_element(rc, Position(0, 0))
	 .add_single_element(wc, Position(0,1))
	 .add_single_element(rc, Position(0,2))
	 .add_element(single_building_element(rb, quarter_rot_z), Position(0,3))
		
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

	return b.create_building(structure_pose.translation, structure_pose.rotation, net);
}

std::shared_ptr<state::building> benchmark::building_1(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token)
{
	std::string wb = "wooden block";
	std::string wbh = "wooden block horizontal";
	std::string wcu = "wooden cube";
	std::string rb = "red block horizontal";
	std::string bb = "blue block";


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


	auto build = b.create_building(structure_pose.translation, structure_pose.rotation, net);

	return build;
}

std::shared_ptr<state::building> benchmark::building_2(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token)
{
	std::string wbh = "wooden block horizontal";
	std::string wc = "wooden cube";
	std::string rb = "red block horizontal";
	std::string bb = "blue block";
	std::string gc = "green cube";
	std::string pc = "purple cube";
	std::string rc = "red cube";

	builder b;
	b
		.add_element(single_building_element(gc), Position(0, 0))
		.add_element(single_building_element(gc), Position(0, 1))
		.add_element(single_building_element(gc), Position(0, 2))
		.add_element(single_building_element(wbh, quarter_rot_z), Position(0, 3))

		.add_element(single_building_element(wbh, quarter_rot_z), Position(1, 0))
		.add_element(single_building_element(pc), Position(1, 1))
		.add_element(single_building_element(rb, quarter_rot_z), Position(1, 2))
		
		.add_element(single_building_element(pc), Position(2, 0))
		.add_element(single_building_element(wbh, quarter_rot_z), Position(2, 1))
		.add_element(single_building_element(wbh, quarter_rot_z), Position(2, 2))

		.add_element(single_building_element(wbh, quarter_rot_z), Position(3, 0))
		.add_element(single_building_element(gc), Position(3, 1))			
		.add_element(single_building_element(rb, quarter_rot_z), Position(3, 2))
		

		.add_element(single_building_element(rc), Position(4, 0))
		.add_element(single_building_element(rc), Position(4, 1))
		.add_element(single_building_element(rc), Position(4, 2))
		.add_element(single_building_element(wc), Position(4, 3))
		.add_element(single_building_element(wc), Position(4, 4));


	auto build = b.create_building(structure_pose.translation, structure_pose.rotation, net);

	return build;
}


std::shared_ptr<state::building> benchmark::building_3(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token)
{
	std::string wbh = "wooden block horizontal";
	std::string wc = "wooden cube";
	std::string rb = "red block horizontal";
	std::string bb = "blue block";
	std::string gc = "green cube";
	std::string pc = "purple cube";
	std::string rc = "red cube";

	builder b;
	b
		.add_element(single_building_element(wbh, quarter_rot_z), Position(0, 0))
		.add_element(single_building_element(gc), Position(0, 1))
		.add_element(single_building_element(gc), Position(0, 2))
		.add_element(single_building_element(gc), Position(0, 3))

		.add_element(single_building_element(wbh, quarter_rot_z), Position(1, 0))
		.add_element(single_building_element(rc), Position(1, 1))
		.add_element(single_building_element(gc), Position(1, 2))
		.add_element(single_building_element(rc), Position(1, 3))

		.add_element(single_building_element(pc), Position(2, 0))
		.add_element(single_building_element(rb, quarter_rot_z), Position(2, 1))		
		.add_element(single_building_element(rb, quarter_rot_z), Position(2, 2))

		.add_element(single_building_element(wbh, quarter_rot_z), Position(3, 0))
		.add_element(single_building_element(bb), Position(3, 1))
		.add_element(single_building_element(wc), Position(3, 2))
		.add_element(single_building_element(wc), Position(3, 3))

		.add_element(single_building_element(rb, quarter_rot_z), Position(4, 0))
		.add_element(single_building_element(wbh, quarter_rot_z), Position(4, 1));


	auto build = b.create_building(structure_pose.translation, structure_pose.rotation, net);

	return build;
}

std::shared_ptr<state::building> benchmark::building_4(const state::pn_net::Ptr& net, const std::map<std::string, state::pn_object_token::Ptr>& name_to_token)
{
	std::string wbh = "wooden block horizontal";
	std::string wc = "wooden cube";
	std::string rb = "red block horizontal";
	std::string bb = "blue block";
	std::string gc = "green cube";
	std::string pc = "purple cube";
	std::string rc = "red cube";

	builder b;
	b
		.add_element(single_building_element(bb), Position(0, 0))
		.add_element(single_building_element(bb), Position(0, 1))
		.add_element(single_building_element(wc), Position(0, 2))
		.add_element(single_building_element(wc), Position(0, 3))
		.add_element(single_building_element(bb), Position(0, 4))

		.add_element(single_building_element(wbh, quarter_rot_z), Position(1, 1))

		.add_element(single_building_element(wbh, quarter_rot_z), Position(2, 0))
		.add_element(single_building_element(wc), Position(2, 1))
		.add_element(single_building_element(rb, quarter_rot_z), Position(2, 2))

		.add_element(single_building_element(rb, quarter_rot_z), Position(3, 0))
		.add_element(single_building_element(pc), Position(3, 2))
		.add_element(single_building_element(wbh, quarter_rot_z), Position(3, 1))
		

		.add_element(single_building_element(wc), Position(4, 0))
		.add_element(single_building_element(wc), Position(4, 1))
		.add_element(single_building_element(gc), Position(4, 2))
		.add_element(single_building_element(gc), Position(4, 3))
		.add_element(single_building_element(rc), Position(4, 4));


	auto build = b.create_building(structure_pose.translation, structure_pose.rotation, net);

	return build;
}


task_manager::task_manager(const state::object_parameters& object_params, const state::object_prototype_loader& loader, int count_agents,
	std::chrono::high_resolution_clock::time_point start_time)
	:
	net(benchmark::init_net(object_params, count_agents)),
	resource_pool(benchmark::init_resource_pool(net, loader)),
	name_to_token(benchmark::named_tokens(resource_pool)),
	decomposition_goal(benchmark::decompose(net, resource_pool)),
	buildings({ 
		{ "1", benchmark::flag_denmark(net, name_to_token)},
		/*{"building 1", benchmark::building_1(net, name_to_token)},*/
		{ "2", benchmark::building_2(net, name_to_token)},
		//{ "3", benchmark::building_3(net, name_to_token) },
		//{ "4", benchmark::building_4(net, name_to_token) }
	}),
	tasks({
		/*{"pyramid", benchmark::pyramid(net,name_to_token)}*/
	}),
	initial_marking(benchmark::to_marking(net, resource_pool, buildings)), // requires all tasks to be created
	rand(std::random_device()()),
	marking(std::make_shared<pn_belief_marking>(initial_marking)),
	start_time(start_time)
{
	std::string num;
	std::cout << "Choose building (0 = random in each iteration, 1, 2):";
	std::cin >> num;

	auto iter = buildings.find(num);
	if (iter == buildings.end())
		for (const auto& entry : buildings)
			tasks.emplace(entry.first, entry.second->get_goal());
	else
		tasks.emplace(iter->first, iter->second->get_goal());

	// decomposition must be first goal and must be the goal of every second task
	// otherwise place_classification_handler will stop checking the resource pool places
	net->set_goal(decomposition_goal.first);

	reset(start_time);
}

task_manager::task_manager(const simulation::sim_task& task,
	std::chrono::high_resolution_clock::time_point start_time)
	:
	net(task.env->net),
	resource_pool(to_object_instances(task.env->object_traces)),
	name_to_token(),
	decomposition_goal(task.init_goal),
	buildings({}),
	tasks({
		{task.name, task.task_goal}
	}),
	initial_marking(task.env->get_marking()), // requires all tasks to be created
	rand(std::random_device()()),
	marking(std::make_shared<pn_belief_marking>(initial_marking)),
	start_time(start_time)
{
	// decomposition must be first goal and must be the goal of every second task
	// otherwise place_classification_handler will stop checking the resource pool places
	net->set_goal(decomposition_goal.first);

	reset(start_time);
}

state::pn_belief_marking::ConstPtr task_manager::get_marking()
{
	std::lock_guard<std::mutex> lock(m);
	return marking;
}

void task_manager::update_marking(const state::pn_belief_marking::ConstPtr& marking)
{
	std::unique_lock<std::mutex> lock(m);
	this->marking = marking;
	lock.unlock();

	int completed = 0;
	auto goal_instances = marking->net.lock()->get_goal_instances();
	for (const auto& instance : goal_instances)
		if (marking->get_probability(instance) > 0.5f)
			completed++;

	file << duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count() << ",progress," << completed << "," << goal_instances.size() << std::endl;
	std::cout << "completed " << completed << " of " << goal_instances.size() << std::endl;


	if (marking->get_summed_probability(net->get_goal()) <= 0.5)
		return;

	{
		std::lock_guard<std::mutex> lock_net(net->mutex);

		pn_instance goal;
		if (net->get_goal() != decomposition_goal.first)
		{
			std::cout << "starting task decomposition" << std::endl;
			file << duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count() << ",task,decomposition" << std::endl;
			net->set_goal(decomposition_goal.first);
		}
		else
		{
			const auto& next = next_task();
			net->set_goal(next.first);
		}
	}

	erase_goal_tokens();	
}

void task_manager::next()
{
	file << duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count() << ",reset" << std::endl;
	
	{
		std::lock_guard<std::mutex> lock(m);
		marking = std::make_shared<pn_belief_marking>(initial_marking);
	}

	{
		std::lock_guard<std::mutex> lock_net(net->mutex);
		net->set_goal(next_task().first);
	}	

	(*emitter)(this->marking, enact_priority::operation::UPDATE);
}

void task_manager::decompose()
{
	file << duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count() << ",reset" << std::endl;
	{
		std::lock_guard<std::mutex> lock_net(net->mutex);
		net->set_goal(decomposition_goal.first);
	}

	erase_goal_tokens();
}

void task_manager::reset(std::chrono::high_resolution_clock::time_point start_time)
{
	if (file.is_open())
		file.close();

	this->start_time = start_time;

	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
	std::stringstream stream;
	stream << time.date().year()
		<< "-" << time.date().month().as_number()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes();

	std::filesystem::create_directory(stream.str());

	log_places(stream.str());

	file.open(stream.str() + "/task.csv");
	file << "time (ms),task name" << std::endl;

	{
		std::lock_guard<std::mutex> lock_net(net->mutex);
		net->set_goal(decomposition_goal.first);
	}

	(*emitter)(this->marking, enact_priority::operation::UPDATE);
}

pn_instance task_manager::next_task()
{
	std::uniform_int_distribution<> distrib(0, tasks.size() - 1);
	auto next = std::next(std::begin(tasks), distrib(rand));
	std::cout << "starting task " << next->first << std::endl;
	file << duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count() << ",task," << next->first << std::endl;
	return next->second;
}

void task_manager::erase_goal_tokens()
{
	pn_belief_marking::marking_dist_t new_dist;
	for (const auto& m : marking->distribution)
	{
		auto distribution = m.first->distribution;

		distribution.erase(decomposition_goal);
		for (const auto& entry : tasks)
		{
			distribution.erase(entry.second);
		}

		new_dist.emplace(std::make_shared<pn_binary_marking>(net, std::move(distribution)), m.second);
	}

	std::unique_lock<std::mutex> lock(m);
	this->marking = std::make_shared<pn_belief_marking>(net, std::move(new_dist));
	lock.unlock();

	(*emitter)(this->marking, enact_priority::operation::UPDATE);
}

std::vector<pn_object_instance> task_manager::to_object_instances(const simulation::environment::ObjectTraces& objects)
{
	std::vector<pn_object_instance> result;
	result.reserve(objects.size());

	for (const auto& entry : objects)
		if(auto token = std::dynamic_pointer_cast<pn_object_token>(entry.first.second))
			if(auto place = std::dynamic_pointer_cast<pn_boxed_place>(entry.first.first))
				result.emplace_back(place, token);

	return result;
}

void task_manager::log_places(const std::string& path) const
{
#ifdef DEBUG_PN_ID

	std::map<pn_place::Ptr, std::pair<std::string, pn_token::Ptr>> place_to_task;

	place_to_task.emplace(decomposition_goal.first, std::make_pair( "decomposition", decomposition_goal.second ));

	for (const auto& entry : decomposition_goal.first->get_incoming_transitions().begin()->lock()->get_side_conditions())
	{
		place_to_task.emplace(entry.first, std::make_pair("decomposition", entry.second));
	}

	for (const auto& goal : tasks)
	{
		place_to_task.emplace(goal.second.first, std::make_pair(goal.first, goal.second.second));

		for (const auto& entry : goal.second.first->get_incoming_transitions().begin()->lock()->get_side_conditions())
		{
			place_to_task.emplace(entry.first, std::make_pair(goal.first, entry.second));
		}
	}

	std::ofstream log(path+"/places.csv");
	log << "placeID,tokenID,task name,x,y,z" << std::endl;

	for (const auto& p : net->get_places())
	{
		log << p->id;
		auto iter1 = place_to_task.find(p);

		if (p->id == 0)
			log << ",,robot";
		else if (p->id == 1 || p->id == 2)
			log << ",,hand";
		else if (iter1 == place_to_task.end())
			log << ",,";
		else
			log << "," << iter1->second.second->id << "," << iter1->second.first;

		auto boxed_p = std::dynamic_pointer_cast<pn_boxed_place>(p);

		if (boxed_p)
		{
			const auto& pos = boxed_p->box.translation;
			log << "," << pos.x() << "," << pos.y() << "," << pos.z();
		}

		log << std::endl;
	}

	log.close();

	log.open(path+"/tokens.csv");
	log << "tokenID,name,object type,r,g,b,dimension x,dimension y, dimension z" << std::endl;

	for (const auto& t : net->get_tokens())
	{
		log << t->id;

		auto obj_t = std::dynamic_pointer_cast<pn_object_token>(t);

		if (obj_t)
		{
			const auto& col = obj_t->object->get_mean_color();
			const auto& dim = obj_t->object->get_bounding_box().diagonal;
			log << "," << obj_t->object->get_name() << "," << obj_t->object->get_type() 
				<< "," << std::to_string(col.r) << "," << std::to_string(col.g) << "," << std::to_string(col.b)
				<< "," << dim.x() << "," << dim.y() << "," << dim.z();
		}
		else if (std::dynamic_pointer_cast<pn_empty_token>(t))
			log << ",,empty_token";

		log << std::endl;
	}

	log.close();

	log.open(path + "/transitions.csv");
	log << "id,agent id,action type,object type,placeID,inputs: placeID tokenID ...,,outputs: placeID tokenID ..." << std::endl;

	for (const auto& t : net->get_transitions())
	{
		std::string agent;
		std::stringstream ss;

		auto agents = net->get_agent_places();

		for (const auto& inst : t->inputs)
		{
			ss << inst.first->id << " " << inst.second->id << ",";
			if (agents.count(inst.first))
				agent = std::to_string(inst.first->id);
		}

		//intentional empty cell between inputs and outputs
		for (const auto& inst : t->outputs)
		{
			ss << "," << inst.first->id << " " << inst.second->id;
			if (agents.count(inst.first))
				agent = std::to_string(inst.first->id);
		}


		log << t->id << "," << agent << "," << t->to_string() << "," << ss.str() << std::endl;
	}
#endif
}

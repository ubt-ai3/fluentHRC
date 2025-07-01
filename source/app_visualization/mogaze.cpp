#include "mogaze.hpp"

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

using namespace state_observation;
using namespace simulation;
using namespace simulation::mogaze;
using namespace hand_pose_estimation;
using namespace prediction;

mogaze_predictor::mogaze_predictor(int person, bool single_predictor)
	:
	threaded_actor("renderer", 0.01f),
	workspace_params(true),
	pc_prepro(std::make_shared<object_parameters>(), true),
	execution(*pc_prepro.object_params, person)
{
	auto& net = *execution.net;
	auto all_locations = net.get_places();
	const auto& agent_places = net.get_agent_places();
	erase_if(all_locations, [&agent_places](const pn_place::Ptr& p) {return agent_places.contains(p); });
	auto goal_completion_token = std::make_shared<pn_token>();

	auto create_goal = [&](const Eigen::AlignedBox3f& region)
	{
		auto place = net.create_place();
		std::vector<pn_instance> goal_instances;

		for (const auto& p : all_locations)
		{
			auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(p);
			if (!boxed_place || !region.contains(boxed_place->box.translation))
				continue;

			auto outgoing = p->get_outgoing_transitions();
			for (const auto& t: outgoing)
			{
				auto input_arcs = t.lock()->get_pure_input_arcs();
				for (const auto& arc : input_arcs)
					goal_instances.push_back(std::make_pair(p, arc.second));
			}
		}

		auto output_instances = goal_instances;
		output_instances.push_back(std::make_pair(place, goal_completion_token));
		net.create_transition(std::move(goal_instances), std::move(output_instances));
		return place;
	};

	if (single_predictor)
		goals = std::vector(17, create_goal(Eigen::AlignedBox3f(Eigen::Vector3f(-10, -10, -10), Eigen::Vector3f(10, 10, 10))));
	else
	{		
		// set table
		{
			auto place = create_goal(predicate::table);
			for (int i = 0; i < 4; i++)
				goals.emplace_back(place);
		}
		
		// clear table
		{
			auto place = create_goal(predicate::laiva_shelf.merged(predicate::vesken_shelf));

			for (int i = 4; i < 5; i++)
				goals.emplace_back(place);
		}

		// put on small shelf
		{
			auto place = create_goal(predicate::vesken_shelf);

			for (int i = 5; i < 7; i++)
				goals.emplace_back(place);
		}

		// put on big shelf
		{
			auto place = create_goal(predicate::laiva_shelf);

			for (int i = 7; i < 17; i++)
				goals.emplace_back(place);
		}
	}

	net.set_goal(goals.front());
	agent = std::make_shared<observed_agent>(world, execution.net, workspace_params, std::shared_ptr<enact_core::entity_id>(), execution.human);


	start_thread();
	

	std::thread t([&]() {
		std::cout << "Press enter to execute one action." << std::endl;
		auto prev_marking = execution.get_marking();
		do
		{
			std::string s;
			std::getline(std::cin, s, '\n');

			std::lock_guard<std::mutex> lock(m);
			int task = execution.get_task_id();
			auto candidates = execution.get_action_candidates();
			auto all_feasible = execution.all_feasible_candidates(execution.get_marking());
			auto a = execution.peek_next_action();
			prev_action = execution.next();

			if (!prev_action)
				break;

			if (!all_feasible.contains(prev_action)) {
				std::cerr << "unexpected: " << (a.pick ? "pick" : "place") << " " << a.object->get_name() << " at/from (" <<
					std::setprecision(2) << a.pose.translation.x() << ", " << a.pose.translation.y() << ", " << a.pose.translation.z() << ") at " << std::setprecision(4) << a.timestamp << "s during task " << task << std::endl;
			}
			else {
				std::vector<transition_context> contexts;
				for (const auto& t : candidates)
					if (t != prev_action)
						contexts.emplace_back(transition_context(workspace_params, prev_action, prev_marking, execution.human));

				get_predictor(task, prev_marking)->add_transition(transition_context(workspace_params, prev_action, prev_marking, execution.human), contexts);
			}

			prev_marking = execution.get_marking();
		} while (!pcl_viewer || !pcl_viewer->wasStopped());
	});

	if (t.joinable())
		t.join();
}

mogaze_predictor::~mogaze_predictor()
{
	stop_thread();
}

bool mogaze_predictor::reverses(const pn_transition& t1, const pn_transition& t2)
{
	for (const auto& input : t1.inputs)
	{
		if (t1.is_side_condition(input))
			continue;

		if (!t2.has_output_arc(input))
			return false;
	}

	for (const auto& input : t2.inputs)
	{
		if (t2.is_side_condition(input))
			continue;

		if (!t1.has_output_arc(input))
			return false;
	}

	for (const auto& output : t1.outputs)
	{
		if (t1.is_side_condition(output))
			continue;

		if (!t2.has_input_arc(output))
			return false;
	}

	for (const auto& output : t2.outputs)
	{
		if (t2.is_side_condition(output))
			continue;

		if (!t1.has_input_arc(output))
			return false;
	}

	return true;
}

void mogaze_predictor::update()
{
	using queue_entry_t = std::pair<std::vector<pn_transition::Ptr>, double>;

	auto add_box = [this](const std::string& name, const obb& obox, float intensity) {
		if (!pcl_viewer->contains(name))
		{

			pcl_viewer->addCube(obox.translation, obox.rotation,
				obox.diagonal.x(), obox.diagonal.y(), obox.diagonal.z(), name);
			pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
				pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, name);
			pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, intensity, 0, 0, name);

			box_names.push_back(name);
		}
	};

	auto to_box = [](const Eigen::AlignedBox3f& box) {
		return aabb(box.diagonal(), box.center());
	};

	if (!pcl_viewer) {
		pcl_viewer = std::make_unique< pcl::visualization::PCLVisualizer>("Mogaze scene");
		pcl_viewer->addCoordinateSystem();
		pcl_viewer->setCameraPosition(4, 2, 4, -1, 0, 2);

		add_box("table area", to_box(predicate::table), 1.f);
		add_box("big shelf area", to_box(predicate::laiva_shelf), 1.f);
		add_box("small shelf area", to_box(predicate::vesken_shelf), 1.f);

		for (const auto& place : execution.net->get_places())
		{
			auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place);
			if (boxed_place)
				add_box(std::to_string(boxed_place->id), boxed_place->box, 1);
		}
	}

	try {
		{
			std::lock_guard<std::mutex> lock(m);
			execution.render(*pcl_viewer);

			if (!prev_action)
			{
				pcl_viewer->spinOnce();
				return;
			}
		}

		for (const std::string& name : box_names)
			if (pcl_viewer->contains(name))
				pcl_viewer->removeShape(name);

		box_names.clear();

		log_prediction(*agent, prediction);

		double total_weight = 1.;
		constexpr double weight_threshold_to_abort = 0.75;
		double added_weight = 0.;
		static constexpr double infeasible_multiplier = 0.001;
		double max_weight = 0.;

		std::map<pn_transition::Ptr, double> future_transitions;
		auto add = [&](const std::vector<pn_transition::Ptr>& transitions, double w)
		{
			for (const auto& t : transitions) {
				auto iter = future_transitions.find(t);
				if (iter == future_transitions.end())
					future_transitions.emplace(t, w);
				else
					iter->second += w;
			}
			added_weight += w;
			max_weight = std::max(max_weight, w);
		};

		auto comp = [](const prediction_context& lhs, const prediction_context& rhs) -> bool {return lhs.weight < rhs.weight; };
		std::priority_queue<prediction_context,
			std::vector<prediction_context>,
			decltype(comp)>
			queue(comp);

		int task;

		{
			std::lock_guard<std::mutex> lock(m);
			queue.emplace(execution.get_marking());
			task = execution.get_task_id();
		}

		while (!queue.empty() && added_weight < weight_threshold_to_abort * total_weight)
		{
			auto pred_ctx = queue.top(); queue.pop();

			if (pred_ctx.actions.size() >= lookahead)
			{
				if (reverses(*pred_ctx.actions[lookahead - 1], *pred_ctx.actions[lookahead - 2]))
				{
					total_weight -= (1 - infeasible_multiplier) * pred_ctx.weight;
					add(pred_ctx.actions, infeasible_multiplier * pred_ctx.weight);
				}
				else
				{
					add(pred_ctx.actions, pred_ctx.weight);
				}

				continue;
			}

			//auto belief_marking = std::make_shared<pn_belief_marking>(pred_ctx.marking);

			std::vector<prediction::transition_context> candidates;
			{
				std::lock_guard<std::mutex> lock(m);

				for (const auto& transition : execution.get_action_candidates(pred_ctx.marking))
					candidates.emplace_back(workspace_params, transition, pred_ctx.marking, execution.human);
			}

			if (candidates.empty())
			{
				add(pred_ctx.actions, pred_ctx.weight);
				continue;
			}


			auto prediction = get_predictor(task, pred_ctx.marking)->predict(candidates, pred_ctx.actions, pn_belief_marking(pred_ctx.marking));


			for (const auto& pred : prediction) {
				std::vector<pn_transition::Ptr> new_actions(pred_ctx.actions);
				new_actions.push_back(pred.first);
				queue.emplace(pred_ctx.marking->fire(pred.first), std::move(new_actions), pred_ctx.weight * pred.second);
			}

		}

		for (const auto& entry : future_transitions) {
			if (entry.second < observed_agent::min_probability && entry.second < max_weight)
				continue;

			pn_boxed_place::Ptr boxed_p;
			for (const auto& place : entry.first->outputs) {
				if (entry.first->is_side_condition(place))
					continue;

				boxed_p = std::dynamic_pointer_cast<pn_boxed_place>(place.first);
				if (boxed_p)break;
			}

			if (!boxed_p)
				for (const auto& place : entry.first->get_inputs()) {
					boxed_p = std::dynamic_pointer_cast<pn_boxed_place>(place);
					if (boxed_p)break;
				}

			std::string id = std::to_string(std::hash<pn_transition::Ptr>{}(entry.first));

			const auto& obox = boxed_p->box;

			add_box(id, obox, entry.second / added_weight);
		}

		pcl_viewer->spinOnce();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
}

prediction::observed_agent::Ptr mogaze_predictor::get_predictor(int task_id, const state_observation::pn_binary_marking::ConstPtr& marking)
{
	if (task_id < 4) 
	{
		// set table for n persons
		pred_count_in_region::Ptr unfullfilled_pred;
		for (const auto& pred : execution.instruction_predicates.at(task_id))
		{
			if (pred->operator()(marking))
				continue;

			unfullfilled_pred = std::dynamic_pointer_cast<pred_count_in_region>(pred);

			if (unfullfilled_pred)
				break;
		}

		if (unfullfilled_pred && unfullfilled_pred->current_count(marking) > unfullfilled_pred->count)
		{
			auto goal = goals.at(4); // clear table predictor

			if (goal != execution.net->get_goal()) {
				std::lock_guard<std::mutex> lock(execution.net->mutex);
				execution.net->set_goal(goals.at(task_id));
			}
		}
	}

	auto goal = goals.at(task_id);

	if (goal != execution.net->get_goal()) {
		std::lock_guard<std::mutex> lock(execution.net->mutex);
		execution.net->set_goal(goals.at(task_id));
	}

	return agent;
}

void mogaze_predictor::init_logging()
{
	if (file.is_open())
		file.close();

	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
	std::stringstream stream;
	stream << std::to_string(time.date().year())
		<< "-" << time.date().month().as_number()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes();

	std::filesystem::create_directory(stream.str());

	file.open(stream.str() + "/prediction.csv");
	file << "timestamp,action type,object,place id,count candidates,transitionID probability correct action,transitionID probabilities other actions" << std::endl;

}

void mogaze_predictor::log_prediction(const prediction::observed_agent& agent, const prediction_entry& predict)
{
	const auto& actions = agent.get_executed_actions();
	const auto& ctx = actions.at(predict.prev_action_index + 1);
	const auto action = ctx.transition;
	auto& probs = predict.probabilities;
	auto iter_prob_true = probs.find(action);

	// log timestamp in same unit as input data
	int time = duration_cast<std::chrono::milliseconds>(ctx.timestamp).count() / 10;

	
	file << time << "," << action->to_string() << "," << predict.count_candidates << "," << iter_prob_true->first->id << " " << (iter_prob_true == probs.end() ? 0 : iter_prob_true->second);
	for (const auto& entry : probs)
		if (entry.first != action)
			file << "," << entry.first->id << " " << entry.second;
	file << std::endl;


	for (int i = predict.prev_action_index + 2; i < actions.size(); i++)
	{
		const auto& ctx = actions.at(i);
		const auto action = ctx.transition;

		file << time << "," << action->to_string() << std::endl;
	}

	file.flush();
}
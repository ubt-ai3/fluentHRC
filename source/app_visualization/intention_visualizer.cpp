#include "intention_visualizer.hpp"

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include "viewer.hpp"

#include "enact_core/access.hpp"
#include "enact_core/lock.hpp"
#include "enact_core/world.hpp"

#include "intention_prediction/agent_manager.hpp"
#include "hand_pose_estimation/hand_tracker_enact.hpp"

using namespace state_observation;
using namespace hand_pose_estimation;
using namespace std::chrono;

intention_visualizer::intention_visualizer(enact_core::world_context& world,
	const computed_workspace_parameters& workspace_params,
	prediction::agent_manager& agents,
	viewer& view,
	pn_belief_marking::ConstPtr initial_marking)
	:
	world(world),
	threaded_actor("intention visualizer", 0.25),
	workspace_params(workspace_params),
	agents(agents),
	view(view),
	marking(std::move(initial_marking)),
	start(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now().time_since_epoch()))
{
	reset();

	start_thread();
}

intention_visualizer::~intention_visualizer()
{
	stop_thread();
}

cv::Vec3b intention_visualizer::get_color(const std::shared_ptr<enact_core::entity_id>& id) const
{
	int int_id = std::hash<enact_core::entity_id*>{}(&*id);
	return cv::Vec3b(63 * (int_id / 16 % 4), 63 * (int_id / 4 % 4), 63 * (int_id % 4));
}

void intention_visualizer::update(const state_observation::pn_belief_marking::ConstPtr& marking)
{
	this->marking = marking;
}

void intention_visualizer::reset()
{
	if (file.is_open())
		file.close();

	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
	std::stringstream stream;
	stream << time.date().year()
		<< "-" << time.date().month().as_number()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes();

	std::filesystem::create_directory(stream.str());

	file.open(stream.str() + "/prediction.csv");
	file << "hand,timestamp,action type,object,place id,count candidates,transitionID probability correct action,transitionID probabilities other actions" << std::endl;
	
}

void intention_visualizer::update()
{
	using namespace prediction;
	using namespace state_observation;
	using queue_entry_t = std::pair<std::vector<pn_transition::Ptr>, double>;

	try
	{
		auto local_marking = marking; // avoid race conditions
		std::vector<pn_transition::Ptr> meta_transitions = local_marking->net.lock()->get_meta_transitions();

		std::vector<std::string> old_box_names;
		swap(old_box_names, box_names);

		//for (const std::string& name : box_names)
			//view.remove_bounding_box(name);

		//box_names.clear();

		if (!local_marking)
		{
			for (const std::string& name : old_box_names)
				view.remove_bounding_box(name);
			return;
		}
		std::map<pn_boxed_place::Ptr, double> destination_probabilities;

		size_t total_action_count = 0;

		for (const auto& agent : agents.get_agents())
		{
			auto pred_iter = predictions.find(agent);
			if (pred_iter != predictions.end() && pred_iter->second.prev_action_index < agent->get_executed_actions().size() - 1)
				log_prediction(*agent, pred_iter->second);

			const size_t action_count = agent->get_executed_actions().size();
			total_action_count += action_count;

			double total_weight = 1.;
			constexpr double weight_threshold_to_abort = 0.5;
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


			queue.emplace(local_marking);

			while (!queue.empty() && added_weight < weight_threshold_to_abort * total_weight)
			{
				auto pred_ctx = queue.top(); queue.pop();

				if (pred_ctx.actions.size() >= lookahead)
				{
					if (pred_ctx.actions[lookahead - 1]->reverses(*pred_ctx.actions[lookahead - 2]))
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

				auto candidates = agent->get_executable_actions(*pred_ctx.marking);

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


				auto prediction = agent->predict(candidates, pred_ctx.actions, *pred_ctx.marking);


				for (const auto& pred : prediction)
				{
					std::vector<pn_transition::Ptr> new_actions(pred_ctx.actions);
					new_actions.push_back(pred.first);
					queue.emplace(pred_ctx.marking->fire(pred.first), std::move(new_actions), pred_ctx.weight * pred.second);
				}

			}

			// memorize result and show boxes
			if (pred_iter == predictions.end())
				pred_iter = predictions.emplace(agent, prediction_entry{}).first;

			pred_iter->second.prev_action_index = action_count - 1;
			pred_iter->second.probabilities.clear();
			pred_iter->second.count_candidates = agent->get_executable_actions(*local_marking).size();

			for (const auto& entry : future_transitions.front())
			{
				if (entry.second < observed_agent::min_probability && entry.second < max_weight)
					continue;

				pred_iter->second.probabilities.emplace(entry.first, entry.second / added_weight);

				pn_boxed_place::Ptr boxed_p;
				for (const auto& place : entry.first->outputs)
				{
					if (entry.first->is_side_condition(place))
						continue;

					boxed_p = std::dynamic_pointer_cast<pn_boxed_place>(place.first);
					if (boxed_p)break;
				}

				if (!boxed_p)
					for (const auto& place : entry.first->get_inputs())
					{
						boxed_p = std::dynamic_pointer_cast<pn_boxed_place>(place);
						if (boxed_p)break;
					}

				if (!boxed_p)
					continue;

				auto iter = destination_probabilities.find(boxed_p);
				if (iter == destination_probabilities.end())
					iter = destination_probabilities.emplace(boxed_p, 0).first;

				iter->second += entry.second / added_weight * action_count;


			}
		}



		for (const auto& entry : destination_probabilities)
		{
			std::string id = "bPlace" + std::to_string(std::hash<pn_boxed_place::Ptr>{}(entry.first));
			cv::Scalar rgb(0.5, 0.5, 0.5);
			rgb *= entry.second / total_action_count;

			auto iter = std::find(old_box_names.begin(), old_box_names.end(), id);
			if(iter != old_box_names.end())
				old_box_names.erase(iter);
			box_names.push_back(id);

			view.add_bounding_box(entry.first->box, id, rgb[2] + 0.5, rgb[1] + 0.5, rgb[0] + 0.5);
		}

		for (const auto& old_box : old_box_names)
			view.remove_bounding_box(old_box);
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}
}

void intention_visualizer::log_prediction(const prediction::observed_agent& agent, const prediction_entry& predict)
{
	const auto& actions = agent.get_executed_actions();
	const auto& ctx = actions.at(predict.prev_action_index + 1);
	const auto action = ctx.transition;
	auto& probs = predict.probabilities;
	auto iter_prob_true = probs.find(action);
	bool right_hand = false;

	{
		enact_core::lock l(world, enact_core::lock_request(agent.tracked_hand, hand_trajectory::aspect_id, enact_core::lock_request::read));
		enact_core::const_access<enact_core::lockable_data_typed<hand_trajectory>> access_object(l.at(agent.tracked_hand, hand_trajectory::aspect_id));
		auto& obj = access_object->payload;

		right_hand = obj.right_hand > 0.5;
	}

	file << (right_hand ? "right," : "left,") << duration_cast<milliseconds>(ctx.timestamp - start).count() << "," << action->to_string() << "," << predict.count_candidates << "," << iter_prob_true->first->id << " " << (iter_prob_true == probs.end() ? 0 : iter_prob_true->second);
	for (const auto& entry : probs)
		if (entry.first != action)
			file << "," << entry.first->id << " " << entry.second;
	file << std::endl;


	for (int i = predict.prev_action_index + 2; i < actions.size(); i++)
	{
		const auto& ctx = actions.at(i);
		const auto action = ctx.transition;

		file << (right_hand ? "right," : "left,") << duration_cast<milliseconds>(ctx.timestamp - start).count() << "," << action->to_string() << std::endl;
	}

	file.flush();
}

#include "task_progress_visualizer.hpp"

#include <utility>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include "intention_prediction/agent_manager.hpp"

// #define RENDER_MISMATCH


using namespace state_observation;
using namespace std::chrono;

task_progress_visualizer::task_progress_visualizer(enact_core::world_context& world,
	place_classification_handler& tracing,
	pn_belief_marking::ConstPtr initial_marking,
	prediction::agent_manager::Ptr agent_manage,
	std::chrono::high_resolution_clock::time_point start_time)
	:
	threaded_actor("task progress visualizer", 0.01f),
	world(world),
	tracing(tracing),
	//renderer(tracing.get_net(), tracing.get_token_traces(), "Observed progress"),
	timestamp(std::chrono::duration<float>(0)),
	marking(std::move(initial_marking)),
	agent_manage(std::move(agent_manage)),
	start_time(start_time)
{
	differ = std::make_shared<pn_feasible_transition_extractor>(tracing.get_net(), marking->to_marking());

	start_thread();

	reset(start_time);
}


task_progress_visualizer::~task_progress_visualizer()
{
	stop_thread();
}

void task_progress_visualizer::update(std::chrono::duration<float> timestamp)
{
	this->timestamp = timestamp;
}

void task_progress_visualizer::update(const strong_id& id, enact_priority::operation op)
{
	std::lock_guard<std::mutex> lock(update_mutex);

	pending_instance_updates.emplace_back(id, op);
}

void task_progress_visualizer::update(const pn_transition::Ptr& executed)
{
	std::lock_guard<std::mutex> lock(update_mutex);
	executed_transitions.push_back(executed);
}

void task_progress_visualizer::update_goal(const state_observation::pn_belief_marking::ConstPtr& marking)
{
	std::lock_guard<std::mutex> lock(update_mutex);
	marking_update = marking;
}

void task_progress_visualizer::reset(std::chrono::high_resolution_clock::time_point start_time)
{
	if (file.is_open())
		file.close();

	if (file_markings.is_open())
		file_markings.close();

	this->start_time = start_time;

	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
	std::stringstream stream;
	stream << time.date().year()
		<< "-" << time.date().month().as_number()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes();

	std::filesystem::create_directory(stream.str());

	file.open(stream.str() + "/emissions.csv");
	file << "time (ms),type,placeID [tokenID probability],..." << std::endl;

	file_markings.open(stream.str() + "/markings.csv");
	file_markings << "time (ms),hypothesis number,hypothesis hash,probability,placeID tokenID,..." << std::endl;

}



void task_progress_visualizer::update()
{
	try
	{
		std::vector<std::pair<strong_id, enact_priority::operation>> updates;
		std::vector<pn_transition::Ptr> executed;
		pn_belief_marking::ConstPtr next_marking;
		bool goal_update = false;
		{
			std::lock_guard<std::mutex> lock(update_mutex);
			std::swap(updates, pending_instance_updates);
			std::swap(executed, executed_transitions);
			goal_update = marking_update != nullptr;
			if (goal_update)
			{
				marking = marking_update;
				marking_update = nullptr;
			}
		}

		

		//for (const auto& entry : updates)
		//	tracing.update_sync(entry.first, entry.second);

		if (marking->get_summed_probability(marking->net.lock()->get_goal()) > 0.5f)
			return;

		std::map<pn_transition::Ptr, double> execution_result;

		for (const auto& transition : executed)
		{
			if (marking->is_enabled(transition) > 0)
			{
				marking = marking->fire(transition);
				execution_result.emplace(transition, 1);
			}
			else
			{
				std::cout << "Could not fire robot action given the current marking." << std::endl;
			}
		}


		try
		{		

			if (!execution_result.empty() || goal_update)
			{
				for (const auto& transition : marking->net.lock()->get_meta_transitions())
					if (marking->is_enabled(transition))
					{
						marking = marking->fire(transition);
						break;
					}

				auto local_marking = marking;

				(*emitter)(std::make_pair(local_marking, execution_result), enact_priority::operation::UPDATE);
				log(*local_marking);
				goal_update = false;

				//renderer.update(*local_marking);
				differ->update(local_marking->to_marking(), false);
			}			
		}
		catch (...) {}

		evaluate_net(timestamp);

	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

void task_progress_visualizer::evaluate_net(std::chrono::duration<float> timestamp, bool use_tracking_data)
{
	auto start = std::chrono::high_resolution_clock::now();
	auto emissions = tracing.generate_emissions(timestamp);
	log(*emissions);

	//std::cout << "evaluate " << dbn->transition_nodes.size() << " transitions" << std::endl;

	differ->update(emissions);

	auto feasible_extractor = std::dynamic_pointer_cast<pn_feasible_transition_extractor>(differ);

	if (agent_manage && feasible_extractor)
	{
		agent_manage->update(emissions);
		feasible_extractor->set_blocked_transitions(agent_manage->get_blocked_transitions(use_tracking_data));
	}

	std::set<pn_transition::Ptr> feasible_transitions = differ->extract();

	sampling_optimizer_belief optimizer(feasible_transitions, marking, emissions);

	if (!initial_recognition_done)
	{
		if (optimizer.emission_consistency(marking->distribution.begin()->first) > 0.)
		{
			initial_recognition_done = true;
			//renderer.update(*marking);
			for (const auto& transition : marking->net.lock()->get_meta_transitions())
				if (marking->is_enabled(transition))
				{
					marking = marking->fire(transition);
					(*emitter)(std::make_pair(marking, optimizer.get_fired_transitions()), enact_priority::operation::UPDATE);
					break;
				}
			std::cout << "starting task" << std::endl;
			(*emitter)(std::make_pair(marking, std::map<pn_transition::Ptr, double>()), enact_priority::operation::CREATE);
			
			auto local_marking = marking;
			log(*local_marking);
		}
		else
			return;
	}

	try
	{

		auto start = std::chrono::high_resolution_clock::now();
		auto new_marking = optimizer.update(2000);
		//std::cout << "evaluation " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;

		consecutive_mismatches = 0;
		auto now = std::chrono::high_resolution_clock::now();
		//if(feasible_transitions.size())
		//	std::cout << "evaluation took " << std::chrono::duration_cast<std::chrono::milliseconds>(now - start) << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(now - last_successful_evaluation) << " since previous successful)" << std::endl;
		last_successful_evaluation = now;

		if (new_marking->distribution != marking->distribution)
		{
			std::vector<pn_transition::Ptr> executed;
			{
				std::lock_guard<std::mutex> lock(update_mutex);
				std::swap(executed, executed_transitions);
			}

			std::map<pn_transition::Ptr, double> execution_result = optimizer.get_fired_transitions();

			for (const auto& transition : executed)
			{
				if (new_marking->is_enabled(transition) > 0)
				{
					new_marking = new_marking->fire(transition);
					execution_result.emplace(transition, 1);
				}
				else
				{
					std::cout << "Could not fire robot action given the current marking." << std::endl;
				}
			}

			for (const auto& transition : new_marking->net.lock()->get_meta_transitions())
				if (new_marking->is_enabled(transition))
				{
					new_marking = new_marking->fire(transition);
					break;
				}

			if (agent_manage)
				agent_manage->update(optimizer.get_fired_transitions(), marking, new_marking, timestamp);

			(*emitter)(std::make_pair(new_marking, std::move(execution_result)), enact_priority::operation::UPDATE);
			log(*new_marking);
			marking = new_marking;

			differ->update(marking->to_marking());

			//renderer.update(*marking);
		}

		//std::cout << "evaluation took " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;
	}
	catch (const std::exception&)
	{
		long long no_update_for = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - last_successful_evaluation).count();
		if(no_update_for > 2000 && (consecutive_mismatches % 10) == 0)
			std::cout << "no successful evaluation for " << no_update_for << "ms" << std::endl;

		// no valid hypothesis found
		if (use_tracking_data && consecutive_mismatches >= 3)
		{
			evaluate_net(timestamp, false);
			return;
		}
				
		consecutive_mismatches++;
		if (no_update_for <= 2000)
			return;

		// compute mismatches


		std::map<pn_place::Ptr, int> mismatches;
		std::set<pn_place::Ptr> marked_places;

		for (const auto& entry : emissions->token_distribution)
			marked_places.emplace(entry.first.first);

		auto inc = [&mismatches](const pn_place::Ptr& p)
		{
			auto iter = mismatches.find(p);
			if (iter == mismatches.end())
				iter = mismatches.emplace(p, 1).first;
			iter->second++;
		};

		for (const auto& marking : optimizer.get_tested_markings())
		{
			std::set<pn_place::Ptr> remaining_marked_places(marked_places);

			for (const auto& instance : marking.first->distribution)
			{
				if (std::dynamic_pointer_cast<pn_empty_token>(instance.second) && emissions->is_empty(instance.first))
					continue;

				if (emissions->is_empty(instance))
					inc(instance.first);

				if (emissions->is_unobserved(instance))
					continue;

				remaining_marked_places.erase(instance.first);

				auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);

				if (emissions->get_probability(instance) == 0)
					inc(instance.first);
			}

			for (const auto& p : remaining_marked_places)
				inc(p);
		}

		pn_net::print_benchmark_state(*emissions, *marking);
		//print(*emissions, mismatches);

	}

}

void task_progress_visualizer::print(const pn_emission& emission, const std::map<pn_place::Ptr, int>& mismatches) const
{
#ifdef DEBUG_PN_ID

	auto empty_token = pn_empty_token::find_or_create(marking->net.lock());
	auto print_entry = [&](const std::pair<pn_instance, double>& entry)
	{
		std::cout << "(" << entry.first.first->id << ", ";
		auto obj_token = std::dynamic_pointer_cast<pn_object_token>(entry.first.second);

		if (obj_token)
			std::cout << obj_token->object->get_name()[0] << obj_token->object->get_type()[0];
		else if (entry.first.second == empty_token)
			std::cout << "--";
		else
			std::cout << "MM";

		std::cout << ") " << entry.second << ", ";
	};

	std::cout << "mismatches (" << mismatches.size() << "): ";
	for (const auto& entry : mismatches)
		std::cout << entry.first->id << ", ";
	std::cout << std::endl;

	const auto& places = marking->net.lock()->get_places();
	std::cout << "places (" << places.size() << "): ";
	for (const auto& p : places)
	{
		auto boxed_p = std::dynamic_pointer_cast<pn_boxed_place>(p);
		if (boxed_p)
		{
			const auto& pos = boxed_p->box.translation;
			std::cout << p->id << " " << pos.x() << " " << pos.y() << " " << pos.z() << ", ";
		}
	}
	std::cout << std::endl;

	const auto marking_dist = marking->to_marking()->distribution;
	std::cout << "marking (" << marking_dist.size() << "): " << std::setprecision(2);
	for (const auto& entry : marking_dist)
		print_entry(entry);
	std::cout << std::endl;


	std::cout << "empty (" << emission.empty_places.size() << "): ";
	for (const auto& p : emission.empty_places)
		std::cout << p->id << ", ";
	std::cout << std::endl;


	std::cout << "unobserved (" << emission.unobserved_places.size() << "): ";
	for (const auto& p : emission.unobserved_places)
		std::cout << p->id << ", ";
	std::cout << std::endl;


	std::cout << "observed tokens (" << emission.token_distribution.size() << "): ";
	for (const auto& entry : emission.token_distribution)
		print_entry(entry);
	std::cout << std::endl;

	if (mismatches.empty())
		return;



#endif
}


void task_progress_visualizer::log(const pn_emission& emission) const
{
#ifdef DEBUG_PN_ID
	auto elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count();
	auto empty_token = pn_empty_token::find_or_create(marking->net.lock());
	auto print_entry = [&](const std::pair<pn_instance, double>& entry)
	{
		if (entry.first.second == empty_token)
			return;

		file << entry.first.first->id << " "
			 << entry.first.second->id << " "
			 << entry.second << ",";
	};

	const auto& places = marking->net.lock()->get_places();

	file << elapsed
		<< ",empty,";
	for (const auto& p : emission.empty_places)
		file << p->id << ",";
	file << std::endl;


	file << elapsed
		<< ",unobserved,";
	for (const auto& p : emission.unobserved_places)
		file << p->id << ",";
	file << std::endl;


	file << elapsed
		<< ",observed," << std::setprecision(2);
	for (const auto& entry : emission.token_distribution)
		print_entry(entry);
	file << std::endl;

#endif
}

void task_progress_visualizer::log(const pn_belief_marking& marking) const
{
#ifdef DEBUG_PN_ID
	auto elapsed = duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count();
	int id = 0;

	for (const auto& entry : marking.distribution)
	{
		file_markings << elapsed
			<< "," << id++
			<< "," << entry.first->hash()
			<< "," << entry.second;

		for (const auto& instance : entry.first->distribution)
			file_markings << "," << instance.first->id << " " << instance.second->id;

		file_markings << std::endl;
	}
#endif
}
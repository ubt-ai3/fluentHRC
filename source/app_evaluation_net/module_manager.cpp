//#define DEBUG_PN_OBSERVATION

#include "module_manager.hpp"

#include <filesystem>
#include <ranges>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#include <enact_core/access.hpp>
#include <enact_core/data.hpp>
#include <enact_core/lock.hpp>
#include <enact_core/world.hpp>

#include <state_observation/object_prototype_loader.hpp>

#include <simulation/riedelbauch17.hpp>
#include <simulation/baraglia17.hpp>
#include <simulation/rendering.hpp>
#include <simulation/build_test.hpp>

#include "../app_visualization/task_progress_visualizer.hpp"
#include "simulation/hoellerich22.hpp"
#include "state_observation/pointcloud_util.hpp"

namespace state_observation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: progress_evaluation
//
//
/////////////////////////////////////////////////////////////

const double progress_evaluation::removal_threshold = 0.4;

progress_evaluation::progress_evaluation(const std::string& path)
	:
	threaded_actor("progress evaluation", 0.01f),
	simulation_id(0),
	timestamp{},
	output(path, std::fstream::out)
{

	output << "task,simulation id,timestamp,error type,operation,delay,certainty,place id,token id" << std::endl;

	start_thread();
}


progress_evaluation::~progress_evaluation()
{
	stop_thread();
	finish();
}

bool progress_evaluation::is_initial_recognition_done() const
{
	return initial_recognition_done;
}

void progress_evaluation::start(unsigned int simulation_id,
	const std::string& task_name,
	const std::shared_ptr<enact_core::world_context>& world, 
	const std::shared_ptr < place_classification_handler>& tracer,
	const state_observation::pn_binary_marking::ConstPtr& initial_marking)
{
	std::lock_guard<std::mutex> lock(run_mutex);


	this->simulation_id = simulation_id;
	this->task_name = task_name;
	this->world = world;
	this->tracing = tracer;
	this->marking = std::make_shared<pn_belief_marking>(initial_marking);
	this->timestamp = std::chrono::duration<float>{};

	differ = std::make_shared<pn_feasible_transition_extractor>(tracing->get_net(), marking->to_marking());

	const auto& dist = initial_marking->distribution;
	observed_instances.insert(dist.begin(), dist.end());

#ifdef LOG_EMISSIONS
	if (file_emissions.is_open())
		file_emissions.close();

	if (file_markings.is_open())
		file_markings.close();

	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
	std::stringstream stream;
	stream << time.date().year()
		<< "-" << time.date().month().as_number()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes();

	std::filesystem::create_directory(stream.str());

	file_emissions.open(stream.str() + "/emissions.csv");
	file_emissions << "time (ms),type,placeID [tokenID probability],..." << std::endl;

	file_markings.open(stream.str() + "/markings.csv");
	file_markings << "time (ms),hypothesis number,hypothesis hash,probability,placeID tokenID,..." << std::endl;
#endif
}

void progress_evaluation::finish()
{
	std::lock_guard<std::mutex> lock(run_mutex);

	marking = nullptr;
	tracing = nullptr;

	for (auto iter = deleted_instances.begin(); iter != deleted_instances.end();)
	{
		auto created_iter = created_instances.find(iter->first);
		if(created_iter != created_instances.end() && std::chrono::abs(created_iter->second - iter->second) < std::chrono::seconds(3))
		{
			created_instances.erase(created_iter);
			iter = deleted_instances.erase(iter);
			continue;
		}

		if (!std::dynamic_pointer_cast<pn_boxed_place>(iter->first.first)) {
			
			++iter;
			continue;
		}

		const auto& entry = *iter;
		output << task_name << "," << simulation_id << "," << entry.second.count() << ",missed,pick,,";
#ifdef DEBUG_PN_ID
		output << "," << entry.first.first->id << "," << entry.first.second->id;
#endif
		output << endl;
		++iter;
	}

	for (const auto& entry : created_instances)
	{
		if (!std::dynamic_pointer_cast<pn_boxed_place>(entry.first.first))
			continue;

		output << task_name << "," << simulation_id << "," << entry.second.count() << ",missed,place,,";
#ifdef DEBUG_PN_ID
		output << "," << entry.first.first->id << "," << entry.first.second->id;
#endif
		output << endl;
	}

	observed_instances.clear();
	deleted_instances.clear();
	created_instances.clear();

	//must be reset before clearing pending updates
	simulation_id = 0;

	std::lock_guard<std::mutex> u_lock(update_mutex);
	pending_instance_updates.clear();
	pending_action_updates.clear();


}

void progress_evaluation::update(std::chrono::duration<float> timestamp)
{
	this->timestamp = timestamp;
}

void progress_evaluation::update(const strong_id& id, enact_priority::operation op)
{
	if (simulation_id == 0)
		return;

	std::lock_guard<std::mutex> lock(update_mutex);
	pending_instance_updates.emplace_back(id, op);
}

void progress_evaluation::
update(std::chrono::duration<float> timestamp, const pn_transition::Ptr& action_in_progress)
{
	if (simulation_id == 0)
		return;

	std::lock_guard<std::mutex> lock(update_mutex);
	pending_action_updates.try_emplace(action_in_progress, timestamp);
}

void progress_evaluation::update()
{
	std::lock_guard<std::mutex> lock(run_mutex);

	if (!marking || !tracing || !tracing->get_net())
		return;

	std::vector<std::pair<strong_id, enact_priority::operation>> instance_updates;
	std::map<pn_transition::Ptr, std::chrono::duration<float>> action_updates;

	{
		std::lock_guard<std::mutex> updateLock(update_mutex);
		std::swap(instance_updates, pending_instance_updates);
		std::swap(action_updates, pending_action_updates);
	}

	if(world->live_entity_count() == 0 && !instance_updates.empty())
	{
		//Skip old data
		return;
	}
	for (const auto& entry : instance_updates)
	{
		{
			enact_core::lock l(*world, enact_core::lock_request(entry.first, object_instance::aspect_id, enact_core::lock_request::write));
			enact_core::const_access<object_instance_data> access_object(l.at(entry.first, object_instance::aspect_id));
			auto& obj = access_object->payload;

			if (!obj.observation_history.empty() && obj.observation_history.back()->timestamp > timestamp.load())
				timestamp = obj.observation_history.back()->timestamp;
		}

	}

	evaluate_net(timestamp);

	pn_token::Ptr empty_token = nullptr;
	for(const auto& token: this->tracing->get_net()->get_tokens())
	{
		empty_token = std::dynamic_pointer_cast<pn_empty_token>(token);
		if (empty_token)
			break;
	}

	for (const auto& action : action_updates)
	{
		for (const pn_instance& input : action.first->get_pure_input_arcs())
		{
			if (input.second == empty_token)
				continue;

			created_instances.erase(input);
			if (observed_instances.contains(input))
			{
				deleted_instances.insert_or_assign(input, action.second);
			}
		}

		for (const pn_instance& output : action.first->get_pure_output_arcs())
		{
			if (output.second == empty_token)
				continue;

			deleted_instances.erase(output);
			if (!observed_instances.contains(output))
			{
				created_instances.insert_or_assign(output, action.second);
			}
		}
	}

	print_evaluation(timestamp);
}

void progress_evaluation::evaluate_net(std::chrono::duration<float> timestamp)
{
	auto emissions = tracing->generate_emissions(timestamp);
#ifdef LOG_EMISSIONS
	log(*emissions);
#endif

	differ->update(emissions);
	auto dbn = differ->extract();
	sampling_optimizer_belief optimizer(dbn, marking, emissions);

	if (!initial_recognition_done)
	{
		if (optimizer.emission_consistency(marking->distribution.begin()->first) > 0.) {
			initial_recognition_done = true;

			for (const auto& transition : marking->net.lock()->get_meta_transitions())
				if (marking->is_enabled(transition))
				{
					marking = marking->fire(transition);
					break;
				}

			auto local_marking = marking;
#ifdef LOG_EMISSIONS
			log(*local_marking);
#endif
		}
		else
			return;
	}

	try {

		auto new_marking = optimizer.update(2000);

		auto now = std::chrono::high_resolution_clock::now();
		last_successful_evaluation = now;

		if (new_marking->distribution != marking->distribution)
		{
			marking = new_marking;
			differ->update(marking->to_marking());

#ifdef LOG_EMISSIONS
			log(*new_marking);
#endif
		}
	}
	catch (const std::exception&)
	{
		// no valid hypothesis found
	}
}



void progress_evaluation::print_evaluation(std::chrono::duration<float> timestamp)
{
	const pn_net::Ptr net = marking->net.lock();

	std::lock_guard<std::mutex> lock(net->mutex);
	for (const auto& place : net->get_places())
	{
		const auto b_place = std::dynamic_pointer_cast<pn_boxed_place>(place);
		if (!b_place)
			continue;

		//only take boxed_places into account
		for (const auto& [token, certainty] : marking->get_distribution(place))
		{
			if (!std::dynamic_pointer_cast<pn_object_token>(token))
				continue;

			pn_instance instance = std::make_pair(place, token);

			// too uncertain or instance already detected 
			if (certainty <= removal_threshold || observed_instances.contains(instance))
				continue;

			const auto iter = created_instances.find(instance); //had already been placed?
			if (iter == created_instances.end())
			{
				output << task_name << "," << simulation_id << "," << timestamp.count() << ",false,place,," << certainty;
#ifdef DEBUG_PN_ID
				output << "," << instance.first->id << "," << instance.second->id;
#endif
				output << endl;
			}
			else
			{
				output << task_name << "," << simulation_id << "," << timestamp.count() << ",correct,place," << ((timestamp > iter->second) ? (timestamp - iter->second) : std::chrono::duration<float>{}).count() << "," << certainty;
#ifdef DEBUG_PN_ID
				output << "," << instance.first->id << "," << instance.second->id;
#endif
				output << endl;
				created_instances.erase(iter);
			}

			observed_instances.emplace(instance);
		}
	}

	for (auto instance = observed_instances.begin(); instance != observed_instances.end();)
	{
		if (!std::dynamic_pointer_cast<pn_object_token>(instance->second)) {
			instance = observed_instances.erase(instance);
			continue;
		}

		const double certainty = marking->get_probability(*instance);
		if (certainty < removal_threshold)
		{
			auto iter = deleted_instances.find(*instance);
			if (iter == deleted_instances.end())
			{
				output << task_name << "," << simulation_id << "," << timestamp.count() << ",false,pick,," << (1 - certainty);
#ifdef DEBUG_PN_ID
				output << "," << instance->first->id << "," << instance->second->id;
#endif
				output << endl;
			}
			else
			{
				output << task_name << "," << simulation_id << "," << timestamp.count() << ",correct,pick," << ((timestamp > iter->second) ? (timestamp - iter->second) : std::chrono::duration<float>{}).count() << "," << (1 - certainty);
#ifdef DEBUG_PN_ID
				output << "," << instance->first->id << "," << instance->second->id;
#endif
				output << endl;
				deleted_instances.erase(iter);
			}

			instance = observed_instances.erase(instance);
		}
		else
		{
			++instance;
		}
	}
}

#ifdef LOG_EMISSIONS
void progress_evaluation::log(const pn_emission& emission) const
{
#ifdef DEBUG_PN_ID
	auto elapsed = duration_cast<std::chrono::milliseconds>(timestamp.load()).count();
	auto empty_token = pn_empty_token::find_or_create(marking->net.lock());
	auto print_entry = [&](const std::pair<pn_instance, double>& entry)
	{
		if (entry.first.second == empty_token)
			return;

		file_emissions << entry.first.first->id << " "
			<< entry.first.second->id << " "
			<< entry.second << ",";
	};

	const auto& places = marking->net.lock()->get_places();

	file_emissions << elapsed
		<< ",empty,";
	for (const auto& p : emission.empty_places)
		file_emissions << p->id << ",";
	file_emissions << std::endl;


	file_emissions << elapsed
		<< ",unobserved,";
	for (const auto& p : emission.unobserved_places)
		file_emissions << p->id << ",";
	file_emissions << std::endl;


	file_emissions << elapsed
		<< ",observed," << std::setprecision(2);
	for (const auto& entry : emission.token_distribution)
		print_entry(entry);
	file_emissions << std::endl;

#endif
}

void progress_evaluation::log(const pn_belief_marking& marking) const
{
#ifdef DEBUG_PN_ID
	auto elapsed = duration_cast<std::chrono::milliseconds>(timestamp.load()).count();
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
#endif

/////////////////////////////////////////////////////////////
//
//
//  Class: aging_evaluation
//
//
/////////////////////////////////////////////////////////////

bool aging_evaluation::object_properties::operator==(const object_properties& other) const
{
	return std::abs(center.x() - other.center.x()) < 0.03 && std::abs(center.y() - other.center.y()) < 0.03;
}

aging_evaluation::aging_evaluation(
	float likelihood_decay_per_second,
	const std::string& path)
	:
	simulation_id(0),
	timestamp{},
	likelihood_decay_per_second(likelihood_decay_per_second),
	output(path, std::fstream::out)
{
	output << "task,simulation id,timestamp,error type,operation,delay,place id,token id" << std::endl;
}

aging_evaluation::~aging_evaluation()
{
	stop_thread();
}

void aging_evaluation::start(unsigned int simulation_id, 
	const std::string& task_name, 
	const std::shared_ptr<enact_core::world_context>& world, 
	const std::shared_ptr<::simulation::environment>& env)
{
	std::lock_guard<std::mutex> lock(run_mutex);

	timestamp = std::chrono::duration<float>{};
	this->simulation_id = simulation_id;
	this->task_name = task_name;
	this->world = world;
	this->env = env;

	for (const auto& object : env->object_traces | std::views::values)
		properties.emplace_back(object_properties{
			object->center,
			object->prototype,
			{},
			nullptr,
			1
		});

	for (const auto& entry : env->token_traces)
		traces.emplace(entry.second, entry.first);
}

void aging_evaluation::finish()
{
	std::lock_guard<std::mutex> run_lock(run_mutex);
	// safely access get_place
	std::lock_guard<std::mutex> lock(env->net->mutex);

	for (auto iter = deleted_instances.begin(); iter != deleted_instances.end();)
	{
		bool erased = false;
		for (auto created_iter = created_instances.begin(); created_iter != created_instances.end();) {
			if(created_iter->center == iter->center && std::chrono::abs(created_iter->last_update - iter->last_update) < std::chrono::seconds(3))
			{
				created_instances.erase(created_iter);
				iter = deleted_instances.erase(iter);
				erased = true;
				break;
			}
		}

		if (erased)
			continue;

		const auto& prop = *iter;

		output << task_name << "," << simulation_id << "," << prop.last_update.count() << ",missed,pick,";
#ifdef DEBUG_PN_ID
		auto p = get_place(prop.center);
		if (p)
			output << "," << p->id << "," << env->token_traces.at(prop.prototype)->id;
#endif
		output << endl;
		++iter;
	}

	for (const object_properties& prop : created_instances)
	{
		output << task_name << "," << simulation_id << "," << prop.last_update.count() << ",missed,place,";
#ifdef DEBUG_PN_ID
		auto p = get_place(prop.center);
		if (p)
			output << "," << p->id << "," << env->token_traces.at(prop.prototype)->id;
#endif
		output << endl;
	}

	properties.clear();
	deleted_instances.clear();
	created_instances.clear();

	simulation_id = 0;
}

void aging_evaluation::update(const strong_id& id)
{
	auto sim_id = this->simulation_id;
	if (sim_id == 0)
		return;

	schedule([id, sim_id, this]() {

		std::lock_guard<std::mutex> run_lock(run_mutex);
		//check that we are still in the same simulation run
		if (sim_id != this->simulation_id)
			return;

		auto correspondence = std::ranges::find_if(properties, [id](const object_properties& p) {return p.id == id; });
		std::chrono::duration<float> current_timestamp;

		Eigen::Vector3f translation;
		object_prototype::ConstPtr prototype;


		{
			enact_core::lock l(*world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::read));
			const enact_core::const_access<object_instance_data> access_object(l.at(id, object_instance::aspect_id));
			const object_instance& obj = access_object->payload;

			current_timestamp = obj.observation_history.back()->timestamp;

			if (const auto segment = obj.get_classified_segment(); segment)
			{
				const auto& result = segment->classification_results;

				if (!obj.is_background() && result[0].local_certainty_score > 0.5f)
				{
					translation = segment->bounding_box.translation;
					prototype = result[0].prototype;
				}
			}
		}

		if (prototype && correspondence == properties.end())
			correspondence = std::ranges::find(properties, object_properties{ translation, prototype });

		// safely access get_place
		std::lock_guard<std::mutex> lock(env->net->mutex);

		if (correspondence != properties.end())
		{
			correspondence->id = id;
			correspondence->detection_likelihood = std::min(1.f, correspondence->detection_likelihood + (current_timestamp - correspondence->last_update).count() * 1e-6f * likelihood_decay_per_second);
			correspondence->last_update = current_timestamp;

			if (prototype)
				correspondence->center = translation;
		}
		else if (prototype)
		{
			properties.emplace_back(object_properties{
					translation,
					prototype,
					current_timestamp,
					id
				});

			auto iter = std::ranges::find(created_instances, object_properties{ translation, prototype });

			if (iter == created_instances.end())
			{
				output << task_name << "," << simulation_id << "," << current_timestamp.count() << ",false,place,";
#ifdef DEBUG_PN_ID
				auto p = get_place(translation);
				if (p)
					output << "," << p->id << "," << env->token_traces.at(prototype)->id;
#endif
				output << endl;
			}
			else
			{
				output << task_name << "," << simulation_id << "," << current_timestamp.count() << ",correct,place," << (current_timestamp - iter->last_update).count();
#ifdef DEBUG_PN_ID
				auto p = get_place(iter->center);
				if (p)
					output << "," << p->id << "," << env->token_traces.at(iter->prototype)->id;
#endif
				output << endl;
				created_instances.erase(iter);
			}
		}

		if (current_timestamp > timestamp.load())
		{
			for (auto prop = properties.begin(); prop != properties.end();)
			{
				if (prop->last_update < timestamp.load())
				{
					prop->detection_likelihood -= (timestamp.load() - prop->last_update).count() * likelihood_decay_per_second;
					prop->last_update = timestamp;

					if (prop->detection_likelihood < 0.4)
					{
						auto iter = std::ranges::find(deleted_instances, *prop);

						if (iter == deleted_instances.end())
						{
							output << task_name << "," << simulation_id << "," << current_timestamp.count() << ",false,pick,";
#ifdef DEBUG_PN_ID
							auto p = get_place(prop->center);
							if (p)
								output << "," << p->id << "," << env->token_traces.at(prop->prototype)->id;
#endif
							output << endl;
						}
						else
						{
							output << task_name << "," << simulation_id << "," << current_timestamp.count() << ",correct,pick," << (current_timestamp - iter->last_update).count();
#ifdef DEBUG_PN_ID
							auto p = get_place(iter->center);
							if (p)
								output << "," << p->id << "," << env->token_traces.at(iter->prototype)->id;
#endif
							output << endl;
							deleted_instances.erase(iter);
						}
						prop = properties.erase(prop);
						continue;
					}
				}
				++prop;
			}
			timestamp = current_timestamp;
		}
	});
}

void aging_evaluation::update(std::chrono::duration<float> timestamp,
	const pn_transition::Ptr& action_in_progress,
	object_prototype::ConstPtr obj)
{
	auto id = this->simulation_id;

	schedule([this, id, timestamp, action = action_in_progress, obj] {
		std::lock_guard<std::mutex> lock(run_mutex);

		// check were are still in the same simulation run
		if (id != this->simulation_id)
			return;

		auto pick = std::dynamic_pointer_cast<pick_action>(action);
		auto place = std::dynamic_pointer_cast<place_action>(action);
		auto stack = std::dynamic_pointer_cast<stack_action>(action);

		if (pick)
		{
			object_properties ref{ pick->from->box.translation, obj ? obj : traces.at(pick->inputs.begin()->second), timestamp };
			auto creation_iter = std::ranges::find(created_instances, ref);

			if (creation_iter != created_instances.end())
				created_instances.erase(creation_iter);

			auto prop_iter = std::ranges::find(properties, ref);
			if (prop_iter != properties.end())
			{ // effect of pick action not yet observed
				auto deletion_iter = std::ranges::find(deleted_instances, ref);


				if (deletion_iter == deleted_instances.end())
				{
					deleted_instances.push_back(ref);
				}
				else
				{
					deletion_iter->last_update = timestamp;
				}
			}
		}
		else if (place)
		{
			object_properties ref{ place->to->box.translation, obj ? obj : traces.at(place->outputs.begin()->second), timestamp };
			auto prop_iter = std::ranges::find(properties, ref);

			if (prop_iter == properties.end())
			{ // effect of place action not yet observed
				auto creation_iter = std::ranges::find(created_instances, ref);

				if (creation_iter == created_instances.end())
				{
					created_instances.push_back(ref);
				}
				else {
					creation_iter->last_update = timestamp;
				}
			}
		}
		else if (stack)
		{
			const object_properties ref{ stack->center, obj ? obj : traces.at(stack->to.second), timestamp };
			auto prop_iter = std::ranges::find(properties, ref);

			if (prop_iter == properties.end())
			{ // effect of place action not yet observed
				auto creation_iter = std::ranges::find(created_instances, ref);

				if (creation_iter == created_instances.end())
				{
					created_instances.push_back(ref);
				}
				else {
					creation_iter->last_update = timestamp;
				}
			}
		}
	});
}

pn_boxed_place::Ptr aging_evaluation::get_place(const Eigen::Vector3f& center) const
{
	for (const auto& place : env->net->get_places())
	{
		auto boxed_p = std::dynamic_pointer_cast<pn_boxed_place>(place);
		if (!boxed_p)
			continue;

		auto& box = boxed_p->box;

		if (((center - box.translation).cwiseAbs() - 0.5f * box.diagonal).maxCoeff() <= 0)
			return boxed_p;
	}
	return nullptr;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: module_manager
//
//
/////////////////////////////////////////////////////////////

const float module_manager::fps = 25;

module_manager::module_manager(int argc, char* argv[], float likelihood_decay_per_second)
	:
	object_params(new object_parameters),
	renderer(argc, argv, nullptr, 640, 720, false)
{

	pc_prepro = std::make_unique<pointcloud_preprocessing>(object_params, true);
	cloud_signal = std::make_shared<cloud_signal_t>();


	Eigen::Translation2f center(1920 * 0.5f, 1080 * 0.5f);

	pc_grabber =
		[this](const pcl::PointCloud<PointT>::ConstPtr& input) {
		if (cloud_signal)
			cloud_signal->operator()(pc_prepro->remove_table(input));
	};

	std::chrono::zoned_time cur_time = { std::chrono::current_zone(), std::chrono::system_clock::now() };
	
	std::stringstream stream;
	stream << "speed_" << std::setprecision(2) << simulation::simulated_arm::speed << "_"
		<< std::format("{:%Y.%m.%d.%H.%M}", cur_time)
		<< ".csv";

	net_eval = std::make_unique<progress_evaluation>("net_" + stream.str());
	aging_eval = std::make_unique<aging_evaluation>(likelihood_decay_per_second, "aging_decay_" + std::to_string(likelihood_decay_per_second) + "_" + stream.str());
}

void module_manager::run(SimFct fct, int count, int repetitions)
{
	world = std::make_shared<enact_core::world_context>();

	object_prototype_loader loader;
	auto task = fct(*object_params, loader);
	auto initial_marking = std::make_shared<pn_belief_marking>(task->env->get_marking());

	std::vector<object_prototype::ConstPtr> prototypes;
	for (const auto& prototype : task->env->token_traces | std::views::keys)
	{
		prototypes.push_back(prototype);
	}

	auto occlusion_detect = occlusion_detector::construct_from(*pc_prepro, *object_params);

	obj_classify = std::make_shared<place_classification_handler>(*world, *pc_prepro, object_params, occlusion_detect, prototypes, task->env->net, task->env->token_traces, false);


	tracer = std::make_shared<pn_world_tracer>(*world, *pc_prepro, occlusion_detect,/* *object_params,*/ task->env->net, task->env->token_traces);


	renderer.env = task->env;
	std::weak_ptr<cloud_signal_t> weak_cloud_signal = cloud_signal;

	std::function pc_grabber =
		[&weak_cloud_signal, this](const pcl::PointCloud<PointT>::ConstPtr& input) {

		auto pre_process = [this, &input]()
			-> pcl::PointCloud<pcl::PointXYZ>::Ptr
		{
			const auto& trafo = pc_prepro->get_cloud_transformation();
			const auto matrix = trafo.matrix();
			pcl::detail::Transformer tf(matrix);
			auto processed = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
			processed->header = input->header;
			processed->is_dense = input->is_dense;
			processed->height = 1;
			// trade-off between avoiding copies and an over-sized vector 
			processed->points.reserve(1000);
			processed->sensor_orientation_ = input->sensor_orientation_;
			processed->sensor_origin_ = input->sensor_origin_;

			const auto& points_in = input->points;
			auto& points_out = processed->points;
			pcl::PointXYZ p;

			for (const auto& p_in : points_in)
			{
				if (!std::isfinite(p_in.x) ||
					!std::isfinite(p_in.y) ||
					!std::isfinite(p_in.z))
					continue;

				p = pcl::PointXYZ(p_in.x, p_in.y, p_in.z);
				tf.se3(p.data, p.data);

				if (p.z >= -0.05f && p.y > -0.7)
					points_out.emplace_back(p);
			}
			processed->width = points_out.size();

			return processed;
		};

		//last_cloud = input;
		auto cloud_signal = weak_cloud_signal.lock();
		if (cloud_signal)
			cloud_signal->operator()(pc_prepro->remove_table(input));
	};

	auto cloud_signal_con = cloud_signal->connect([&](const pcl::PointCloud<PointT>::ConstPtr& cloud) {
		obj_classify->update(cloud);
		// header stamp is in microseconds but update expects seconds
		net_eval->update(std::chrono::microseconds(cloud->header.stamp));
		tracer->update(cloud);

		});



	auto sig2 = obj_classify->get_signal(enact_priority::operation::UPDATE);
	sig2->connect([&](const entity_id& id) {
		aging_eval->update(id);
		net_eval->update(id, enact_priority::operation::UPDATE);
	});

	auto sig3 = obj_classify->get_signal(enact_priority::operation::DELETED);
	sig3->connect([&](const entity_id& id) {
		net_eval->update(id, enact_priority::operation::DELETED);
	});


	auto sig4 = obj_classify->get_signal(enact_priority::operation::CREATE);
	sig4->connect([&](const entity_id& id) {
		net_eval->update(id, enact_priority::operation::CREATE);
		aging_eval->update(id);
	});


	std::srand(std::time(nullptr));

	std::shared_ptr<pcl::Grabber> grabber;	
	std::chrono::duration<float> timestamp{};

	simulation::task_execution execution(task, fps);

	std::cout << task->name << " run " << count << std::endl;
	unsigned int simulation_id = std::rand();
	net_eval->start(simulation_id, task->name, world, obj_classify, task->env->get_marking());
	if (repetitions == 1)
		aging_eval->start(simulation_id, task->name, world, task->env);

	// wait for 1s to detect initial scene
	pc_grabber(pointcloud_preprocessing::to_pc_rgba(renderer.render(timestamp)));
	std::this_thread::sleep_for(std::chrono::milliseconds(500));
	timestamp += std::chrono::duration<float>(0.5f);
	pc_grabber(pointcloud_preprocessing::to_pc_rgba(renderer.render(timestamp)));
	std::this_thread::sleep_for(std::chrono::milliseconds(500));
	timestamp += std::chrono::duration<float>(0.5f);
	if (!net_eval->is_initial_recognition_done()) {
		std::cerr << "Initial workspace state not detected, aborting run." << std::endl;
		net_eval->finish();
		aging_eval->finish();
		return;
	}

	if (repetitions == 1) {
		{
			std::lock_guard<std::mutex> lock(task->env->net->mutex);
			task->env->net->set_goal(task->task_goal.first);
		}

		rendering_loop(execution, task, timestamp);

		net_eval->finish();
		aging_eval->finish();
	}
	else
		composition_decomposition_loop(execution, task, timestamp, repetitions);

	cloud_signal_con.disconnect();
}

void module_manager::rendering_loop(simulation::task_execution& execution, simulation::sim_task::Ptr& task, std::chrono::duration<float>& timestamp, bool use_aging)
{
	std::atomic_bool termination_flag = false;
	auto duration = std::chrono::milliseconds(static_cast<int>(1000 / fps));

	do
	{
		auto frame_start = std::chrono::high_resolution_clock::now();
		execution.step();

		auto pc_rgb = renderer.render(timestamp);


		for (const auto& agent : task->agents)
			if (agent->executing_action)
			{
				net_eval->update(timestamp, agent->executing_action);
				if(use_aging)
					aging_eval->update(timestamp, agent->executing_action, agent->get_grabbed_object());
			}


		pc_grabber(pointcloud_preprocessing::to_pc_rgba(pc_rgb));

		termination_flag = true;
		for (const auto& agent : task->agents)
			if (!agent->is_idle(timestamp))
			{
				termination_flag = false;
				break;
			}

		std::this_thread::sleep_until(frame_start + duration);
		timestamp += std::chrono::duration<float>(1 / fps);

	} while (!termination_flag && timestamp < std::chrono::seconds(120));

	std::chrono::duration<float> stop_time = timestamp + std::chrono::seconds(4);
	while (timestamp < stop_time)
	{
		auto frame_start = std::chrono::high_resolution_clock::now();
		auto pc_rgb = renderer.render(timestamp);

		pc_grabber(pointcloud_preprocessing::to_pc_rgba(pc_rgb));

		std::this_thread::sleep_until(frame_start + duration);
		timestamp += std::chrono::duration < float>(1 / fps);
	}
}

void module_manager::composition_decomposition_loop(simulation::task_execution& execution, simulation::sim_task::Ptr& task, std::chrono::duration<float>& timestamp, int repetitions)
{
	unsigned int simulation_id = std::rand();

	for (int i = 0; i < repetitions; i++) {
		{
			std::lock_guard<std::mutex> lock(task->env->net->mutex);
			task->env->net->set_goal(task->task_goal.first);
		}
		net_eval->start(simulation_id, task->name, world, obj_classify, task->env->get_marking());

		std::cout << "composition " << i << std::endl;
		rendering_loop(execution, task, timestamp, false);

		net_eval->finish();

		{
			std::lock_guard<std::mutex> lock(task->env->net->mutex);
			task->env->net->set_goal(task->init_goal.first);
		}
		simulation_id = std::rand();
		net_eval->start(simulation_id, task->name + "_Decompose", world, obj_classify, task->env->get_marking());

		std::cout << "decomposition " << i << std::endl;
		rendering_loop(execution, task, timestamp, false);

		net_eval->finish();

		simulation_id = std::rand();
	}
}



} // namespace state_observation

// from: https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c?page=1&tab=scoredesc#tab-top
class input_parser {
public:
	input_parser(int& argc, char** argv) {
		for (int i = 1; i < argc; ++i)
			this->tokens.push_back(std::string(argv[i]));
	}
	/// @author iain
	int get_value(const std::string& option, int default_value) const {
		std::vector<std::string>::const_iterator itr;
		itr = std::find(this->tokens.begin(), this->tokens.end(), option);
		if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
			try {
				const int val = std::atoi(itr->c_str());
				return val <= 0 ? default_value : val;
			}
			catch (...) {}
		}
		return default_value;
	}
	/// @author iain
	bool exists(const std::string& option) const {
		return std::find(this->tokens.begin(), this->tokens.end(), option)
			!= this->tokens.end();
	}
private:
	std::vector <std::string> tokens;
};

using namespace state_observation;
using namespace simulation;
using namespace simulation::mogaze;



int main(int argc, char* argv[])
{
	input_parser parse(argc, argv);

	if(parse.exists( "-h") || parse.exists("--help"))
	{
		std::cout << "Run benchmark tasks and record the performance of aging based and a petri net based algorithm to track the state of the workbench.\n";
		std::cout << "Options (all require integers > 0):\n";
		std::cout << "--task\t\t\t" << "Pick the task to run (1 to 4). If no option is provided, all tasks are run.\n";
		std::cout << "--restarts\t\t" << "How often the simulation environment and algorithms are reset and run again (default: 50).\n";
		std::cout << "--decompositions\t" << "For each restart, compose and decompose the structure that many times (only task 4).\n";
		std::cout << "\t\t\t" << "Value must be >= 2 to take effect. Disables the evaluation of the aging algrithm.\n";
		std::cout << "--forget_duration\t" << "If object is not seen for that many seconds, aging assumes it is gone (default: 6).\n";
		std::cout << "--inverse_speed\t\t" << "Arm motion in seconds per meter (default: 5, i.e. 0.2 m/s).\n";
		std::cout << std::endl;
		std::exit(0);
	}

	int task = parse.get_value("--task", 0);
	int restarts = parse.get_value("--restarts", 50);
	int decompositions = parse.get_value("--decompositions", 1);
	int forget_duration = parse.get_value("--forget_duration", 6);
	int inverse_speed = parse.get_value("--inverse_speed", 5);

	simulated_arm::speed = 1. / inverse_speed;

	char* dummy_argv[1] = { argv[0] };
	state_observation::module_manager manager(1, dummy_argv, 0.6 / forget_duration);

	auto tasks = { &simulation::baraglia17::task_a_1,
								&simulation::riedelbauch17::pack_and_stack,
								&simulation::riedelbauch17::pack_and_stack_and_swap,
								&simulation::hoellerich22::structure_1 };

	if (task >= 1 && task <= tasks.size())
		for (int i = 0; i < restarts; i++)
			manager.run(*(tasks.begin() + (task - 1)), i, decompositions);

	else
		for (const auto& t : tasks)
			for (int i = 0; i < restarts; i++)
				manager.run(t, i, decompositions);

}

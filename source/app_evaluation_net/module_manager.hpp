#pragma once

//#define DEBUG_PN_OBSERVATION
//#define LOG_EMISSIONS

#include <boost/signals2/signal.hpp>

#include <pcl/io/grabber.h>

#include <enact_core/id.hpp>
#include <enact_priority/signaling_actor.hpp>

#include <state_observation/calibration.hpp>
#include <state_observation/classification_handler.hpp>
#include <state_observation/workspace_objects.hpp>
#include <state_observation/pointcloud_util.hpp>
#include <state_observation/object_detection.hpp>
#include <state_observation/object_tracking.hpp>

#include "state_observation/pn_world_traceability.hpp"
#include <state_observation/pn_reasoning.hpp>

#include <simulation/task.hpp>
#include <simulation/scene.hpp>
#include <simulation/rendering.hpp>
#include <simulation/mogaze.hpp>

namespace state_observation
{

/**
* Evaluates action detection based on petri net with fuzzy marking.
* Writes output to file.
*/
class progress_evaluation : public enact_core::threaded_actor
{
public:
	typedef std::shared_ptr<enact_core::entity_id> strong_id;
	typedef std::weak_ptr<enact_core::entity_id> weak_id;
	typedef enact_core::lockable_data_typed<state_observation::object_instance> object_instance_data;

	static const double removal_threshold;

	progress_evaluation(const std::string& path);
	~progress_evaluation() override;

	bool is_initial_recognition_done() const;

	void start(unsigned int simulation_id,
		const std::string& task_name,
		const std::shared_ptr<enact_core::world_context>& world,
		const std::shared_ptr < place_classification_handler>& tracing,
		const state_observation::pn_binary_marking::ConstPtr& initial_marking);

	void finish();
	

	void update(std::chrono::duration<float> timestamp);
	void update(const strong_id& id, enact_priority::operation op);
	void update(std::chrono::duration<float> timestamp, const pn_transition::Ptr& action_in_progress);

protected:
	virtual void update() override;

private:
	void evaluate_net(std::chrono::duration<float> timestamp);
	
	void print_evaluation(std::chrono::duration<float> timestamp);
#ifdef LOG_EMISSIONS
	void log(const pn_emission& emission) const;
	void log(const pn_belief_marking& marking) const;
#endif

	unsigned int simulation_id;
	std::string task_name;
	std::shared_ptr<enact_core::world_context> world;
	std::shared_ptr < place_classification_handler> tracing;

	std::atomic<std::chrono::duration<float>> timestamp;
	state_observation::pn_belief_marking::ConstPtr marking;
	state_observation::pn_transition_extractor::Ptr differ;

	bool initial_recognition_done = false;
	std::chrono::high_resolution_clock::time_point last_successful_evaluation = std::chrono::high_resolution_clock::now();


	std::fstream output;
#ifdef LOG_EMISSIONS
	mutable std::ofstream file_emissions;
	mutable std::ofstream file_markings;
#endif


	std::mutex update_mutex;
	std::mutex run_mutex;
	std::vector<std::pair<strong_id, enact_priority::operation>> pending_instance_updates;
	std::map<pn_transition::Ptr, std::chrono::duration<float>> pending_action_updates;

	std::set<pn_instance> observed_instances;
	std::map<pn_instance, std::chrono::duration<float>> deleted_instances;
	std::map<pn_instance, std::chrono::duration<float>> created_instances;
};

/**
* Evaluates action detection based on aging (following [Baraglia17]).
* Writes output to file.
*/
class aging_evaluation : public enact_priority::priority_actor
{
public:
	typedef std::shared_ptr<enact_core::entity_id> strong_id;
	typedef std::weak_ptr<enact_core::entity_id> weak_id;
	typedef enact_core::lockable_data_typed<state_observation::object_instance> object_instance_data;

private:
	struct object_properties
	{
		Eigen::Vector3f center;
		object_prototype::ConstPtr prototype;
		std::chrono::duration<float> last_update = std::chrono::duration<float>{};
		strong_id id = nullptr;
		float detection_likelihood = 0.6f;


		bool operator==(const object_properties&) const;
	};


public:

	aging_evaluation(float likelihood_decay_per_second,
		const std::string& path);

	~aging_evaluation() override;

	void start(unsigned int simulation_id,
		const std::string& task_name,
		const std::shared_ptr<enact_core::world_context>& world,
		const std::shared_ptr<::simulation::environment>& env);

	void finish();

	void update(const strong_id& id);
	void update(std::chrono::duration<float> timestamp, const pn_transition::Ptr& action_in_progress, object_prototype::ConstPtr obj = nullptr);


private:
	std::mutex run_mutex;
	unsigned int simulation_id;
	std::string task_name;
	std::shared_ptr<enact_core::world_context> world;
	simulation::environment::Ptr env;
	std::map<pn_token::Ptr, object_prototype::ConstPtr> traces;

	std::atomic<std::chrono::duration<float>> timestamp;
	const float likelihood_decay_per_second;
	std::fstream output;

	std::vector<object_properties> properties;

	std::vector<object_properties> deleted_instances;
	std::vector<object_properties> created_instances;

	pn_boxed_place::Ptr get_place(const Eigen::Vector3f& center) const;
};


	
/**
 *************************************************************************
 *
 * @class modules_manager
 *
 * Creates, starts and interwires the actors
 *
 ************************************************************************/

typedef simulation::sim_task::Ptr(* SimFct)(const object_parameters& object_params, object_prototype_loader loader);

class module_manager
{
public:
	#include "signal_types.hpp"

	typedef std::shared_ptr<enact_core::entity_id> entity_id;

	module_manager(int argc, char* argv[], float likelihood_decay_per_second = 0.1f);
	~module_manager() = default;

	void run(SimFct fct, int count, int repetitions = 1);

private:

	std::shared_ptr<enact_core::world_context> world;
	std::shared_ptr<const object_parameters> object_params;
	std::unique_ptr<pointcloud_preprocessing> pc_prepro;
	std::shared_ptr<place_classification_handler> obj_classify;


	::simulation::pc_renderer renderer;

	std::string path;
	std::shared_ptr<pn_world_tracer> tracer;
	std::unique_ptr<progress_evaluation> net_eval;
	std::unique_ptr<aging_evaluation> aging_eval;

	std::function<void(const pcl::PointCloud<PointT>::ConstPtr&)> pc_grabber;
	std::shared_ptr<cloud_signal_t> cloud_signal;


	const static float fps;
	const static float likelihood_decay_per_second;

	void rendering_loop(simulation::task_execution& execution, simulation::sim_task::Ptr& task, std::chrono::duration<float>& timestamp, bool use_aging = true);
	/*
	 * performs consecutive composition and decomposition tasks without resetting the world
	 * only the petri net evaluation is run
	 */
	void composition_decomposition_loop(simulation::task_execution& execution, simulation::sim_task::Ptr& task, std::chrono::duration<float>& timestamp, int repetitions);
};

} // namespace state_observation


#pragma once

#include <Eigen/Core>
#include <opencv2/highgui.hpp>

#include "enact_priority/priority_actor.hpp"
#include "state_observation/pn_model.hpp"
#include "state_observation/pn_reasoning.hpp"
#include "enact_core/id.hpp"
#include "state_observation/workspace_objects.hpp"
#include "state_observation/pn_world_traceability.hpp"

namespace state_observation
{

/**
 * @class task_description
 * @brief Visualizes and manages task descriptions using Petri nets
 * 
 * A priority actor that visualizes and manages task descriptions using
 * Petri nets. Integrates with the world tracing system to track objects
 * and their relationships.
 * 
 * Features:
 * - Task visualization
 * - Petri net rendering
 * - Object tracking
 * - Resource management
 * - State evaluation
 * - Image overlay support
 * - Priority-based updates
 * - Signal-based communication
 */
class task_description : public enact_priority::priority_actor,
	public enact_priority::signaling_actor<cv::Mat>
{
public:
	typedef std::shared_ptr<enact_core::entity_id> strong_id;
	typedef std::weak_ptr<enact_core::entity_id> weak_id;
	typedef enact_core::lockable_data_typed<object_instance> object_instance_data;

	
	task_description(enact_core::world_context& world, 
		pn_world_tracer& tracing,
		const std::vector<object_prototype::ConstPtr>& prototypes,
		const Eigen::Matrix<float, 3, 4>& projection);

	~task_description();

	void update(const cv::Mat& img);

	
	void evaluate_net(std::chrono::duration<float> timestamp);

	void update(const strong_id& id, enact_priority::operation op);

	static const std::map<std::string, int> resources;
	static const int place_radius = 5;
	static const int token_radius = 4;
	static const std::string window_name;

private:
	void draw_net();
	
	enact_core::world_context& world;
	Eigen::Matrix<float, 3, 4> projection;
	std::map<pn_token::Ptr, object_prototype::ConstPtr> token_to_prototype;
	std::vector<pn_place::Ptr> hands;
	std::map<pn_token::Ptr, pn_place::Ptr> resource_pools;

//	std::map<weak_id, pn_instance> instances;
	std::map<pn_place::Ptr, cv::Point2i> circle_centers;
	cv::Mat drawing_net_overlay;
	
	pn_net::Ptr net;
	pn_belief_marking::ConstPtr marking;
	pn_feasible_transition_extractor::Ptr differ;
	sampling_optimizer_belief::Ptr reasoning;

	bool initial_recognition_done = false;

	pn_world_tracer& tracing;
	
};


}

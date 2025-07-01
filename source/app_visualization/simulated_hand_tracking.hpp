#pragma once
#include <memory>

#include "enact_core/id.hpp"

#include "hand_pose_estimation/hand_tracker_enact.hpp"

#include "simulation/scene.hpp"
#include "simulation/task.hpp"

namespace prediction {
	class agent_manager;
}

namespace simulation {
	class environment;
}

/**
 * @class simulated_hand_tracker
 * @brief Tracks simulated hand movements in a virtual environment
 * 
 * A signaling actor that tracks and processes hand movements in a simulated
 * environment. Generates hand pose emissions for agent tracking and prediction.
 * 
 * Features:
 * - Simulated hand tracking
 * - Hand trajectory data management
 * - Pose emission generation
 * - Agent state synchronization
 * - Kinematic parameter handling
 * - Multi-agent support
 */
class simulated_hand_tracker : public enact_priority::signaling_actor < std::shared_ptr<enact_core::entity_id>>
{
public:
	typedef std::shared_ptr<enact_core::entity_id> entity_id;
	typedef enact_core::lockable_data_typed<::hand_pose_estimation::hand_trajectory> hand_trajectory_data;
	
	simulated_hand_tracker(enact_core::world_context& world, 
		const std::vector<simulation::agent::Ptr>& agents,
		prediction::agent_manager& agent_manage,
		const ::hand_pose_estimation::hand_kinematic_parameters& hand_kinematic_params);

	void generate_emissions(std::chrono::duration<float> timestamp);

private:
	enact_core::world_context& world;
	const simulation::environment::Ptr env;
	const ::hand_pose_estimation::hand_kinematic_parameters& hand_kinematic_params;

	std::map<simulation::simulated_arm::Ptr, entity_id> hand_to_ids;

	::hand_pose_estimation::hand_pose_18DoF get_pose(const simulation::simulated_arm& arm, std::chrono::duration<float> timestamp) const;
};


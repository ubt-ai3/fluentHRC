#include "simulated_hand_tracking.hpp"

#include "enact_core/access.hpp"
#include "enact_core/data.hpp"
#include "enact_core/lock.hpp"
#include "enact_core/world.hpp"

#include "intention_prediction/agent_manager.hpp"

#include "state_observation/object_tracking.hpp" // prevent LNK2005 error: ~signaling_actor<std::ahred_ptr<enitity_id>> already defined 

using namespace hand_pose_estimation;
using namespace enact_core;

simulated_hand_tracker::simulated_hand_tracker(enact_core::world_context& world,
	const std::vector<simulation::agent::Ptr>& agents,
	prediction::agent_manager& agent_manage,
	const hand_kinematic_parameters& hand_kinematic_params)
	:
	world(world),
	hand_kinematic_params(hand_kinematic_params)
{
	for (const simulation::agent::Ptr& agent : agents)
	{
		if (auto human_agent = std::dynamic_pointer_cast<simulation::human_agent>(agent); human_agent)
		{
			entity_id id(std::make_shared<enact_core::entity_id>());
			world.register_data(id, hand_trajectory::aspect_id,
				std::make_unique<lockable_data_typed<hand_trajectory>>(hand_trajectory(1.f, 0.f, std::chrono::duration<float>(0), get_pose(*human_agent->arm, std::chrono::duration<float>(0)))));
			hand_to_ids.emplace(human_agent->arm, id);
			agent_manage.add(id, agent->place);
		}
	}
}


void simulated_hand_tracker::generate_emissions(std::chrono::duration<float> timestamp)
{
	for (const auto& entry : hand_to_ids)
	{
		const auto& arm = entry.first;
		const auto& id = entry.second;

		{
			lock l(world, enact_core::lock_request(id, hand_trajectory::aspect_id, enact_core::lock_request::write));
			enact_core::access<hand_trajectory_data> access_object(l.at(id, hand_trajectory::aspect_id));
			auto& obj = access_object->payload;

			obj.poses.emplace_back(timestamp, get_pose(*arm, timestamp));
		}

		(*emitter)(id, enact_priority::operation::UPDATE);
	}
}

hand_pose_18DoF simulated_hand_tracker::get_pose(const simulation::simulated_arm& arm, std::chrono::duration<float> timestamp) const
{
	Eigen::Vector3f tcp = arm.get_tcp(timestamp);

	return { hand_kinematic_params,
		Eigen::Translation3f(tcp) * Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0.f, 1.f, 0.f), (tcp - arm.get_shoulder_pose(timestamp)).normalized()) * Eigen::Translation3f(0.f,-0.05f, 0.f),
		std::vector<std::pair<float, float>>(5, std::make_pair(M_PI_4, M_PI_4)),
		0.f,
		0.f };
}

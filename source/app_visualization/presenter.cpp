#include <iomanip>

#include "presenter.hpp"

#include "object.pb.h"
#include <state_observation/pn_model_extension.hpp>

using namespace state_observation;

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

inline generated::Voxels convert(const franka_proxy::Visualize::VoxelRobot& v)
{
	generated::Voxels out;
	out.set_voxel_side_length(v.voxel_length);
	*out.mutable_robot_origin() = server::convert(v.robot_origin);

	auto& voxel_coords = *out.mutable_voxel_indices();
	voxel_coords.Reserve(v.voxels.size());
	for (const auto& voxel_coord : v.voxels)
		voxel_coords.Add(server::convert<generated::index_3d>(voxel_coord));

	return out;
}

presenter::presenter(franka_visualizations visualizations, 
	const std::shared_ptr<server::server_module>& server,
	occlusion_detector::Ptr occlusion_detect)
	:
	franka_visualizer_(visualizations),
	server(server),
	occlusion_detect(std::move(occlusion_detect))
{
	if (!server) return;

	server->obj_service.sync_signal.connect([this]()
	{
		std::scoped_lock lock(live_boxes_mutex, live_mesh_prototypes_mutex);

		std::vector<std::shared_ptr<generated::Object_Instance>> out;
		out.reserve(live_mesh_prototypes.size() + live_boxes.size());

		for (const auto& mesh_proto : live_mesh_prototypes)
			out.emplace_back(std::make_shared<generated::Object_Instance>(
				convert<generated::Object_Instance>(std::make_pair(mesh_proto.first, mesh_proto.second))
				));

		for (const auto& box : live_boxes)
			out.emplace_back(std::make_shared<generated::Object_Instance>(
				convert<generated::Object_Instance>(std::make_pair(box.first, box.second))
				));

		return out;
	});

	franka_visualizer_.tcps_signal.connect([this](const TcpUpdate &tcpsUpdate)
		{
			server::TcpsData tcps;

			std::visit(overloaded{
			[&tcps](const std::vector<Eigen::Vector3f>& data)
			{
				tcps.emplace<generated::Tcps>(server::convert<generated::Tcps>(data));
			},
			[&tcps](const Visual_Change& data)
			{
				tcps.emplace<generated::Visual_Change>(static_cast<generated::Visual_Change>(data));
			} }, tcpsUpdate);

			this->server->voxel_service.tcps_slot(tcps);
		});

	franka_visualizer_.voxel_signal.connect([this](const VoxelUpdate& v)
		{
			server::VoxelData voxels;

			std::visit(overloaded{
			[&voxels](const franka_proxy::Visualize::VoxelRobot& data)
			{
				voxels.emplace<generated::Voxels>(convert(data));
			},
			[&voxels](const Visual_Change& data)
			{
				voxels.emplace<generated::Visual_Change>(static_cast<generated::Visual_Change>(data));
			} }, v);

			this->server->voxel_service.voxel_slot(voxels);
		});
	franka_visualizer_.joints_progress_signal.connect([this](const JointProgressUpdate& joints)
		{
			server::SyncJointsData sync_joints;

			std::visit(overloaded{
			[&sync_joints](const joints_progress& data)
			{
				sync_joints.emplace<server::Sync_Joints_Array>(server::convert<generated::Sync_Joints_Array>(data));
			},
			[&sync_joints](const Visual_Change& data)
			{
				sync_joints.emplace<generated::Visual_Change>(static_cast<generated::Visual_Change>(data));
			} }, joints);

			this->server->voxel_service.joints_progress_slot(sync_joints);
		});
}

void presenter::update_cloud(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud)
{
	this->cloud = cloud;
}

void presenter::update_net(const pn_net::Ptr& net)
{
	pn_transition::Ptr goal_transition;

	{
		std::lock_guard lock(net->mutex);
		goal_transition = net->get_goal()->get_incoming_transitions().begin()->lock();
	}

	if (!server)
		return;

	std::scoped_lock lock(live_boxes_mutex, live_mesh_prototypes_mutex);
	while (!displayed_target_objects.empty())
	{
		const auto& [instance, id] = *displayed_target_objects.begin();
		hide_target_instance(instance, id);
	}

	displayed_target_objects.clear();
	all_target_objects.clear();
	planned_actions.clear();

	for (const auto& instance : goal_transition->get_side_conditions())
	{
		auto place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);
		auto token = std::dynamic_pointer_cast<pn_object_token>(instance.second);

		if (!place || !token)
			continue;


		auto obj_instance = std::make_pair(place, token);
		const auto& id = show_target_instance(obj_instance);
		all_target_objects.emplace(obj_instance, id);

	}
}

std::string presenter::show_target_instance(const pn_object_instance& obj_instance)
{
	const auto& box = obj_instance.first->box;
	const auto& obj = obj_instance.second->object;

	std::stringstream ss;
	ss << obj->get_name() << "@" << std::setprecision(2) << box.translation.x() << ", " << box.translation.y() << ", " << box.translation.z();
	auto id = ss.str();

	auto c =obj->get_mean_color();

	if (live_mesh_prototypes.contains(id))
		return id;

	const auto& temp = live_mesh_prototypes.insert_or_assign(id,
		transformed_prototype{
			obj->get_name(),
			(Eigen::Translation3f(box.translation) * Eigen::Affine3f(box.rotation)).matrix()
		});

	server->obj_service.add_object(
		std::make_shared<generated::Object_Instance>(
			convert<generated::Object_Instance>(std::make_pair(id, temp.first->second)
				)));

	const auto& assigned = live_boxes.emplace(id + "_box", obj->get_name() == "blue block" ? colored_obb{ 0, 0, 1, box } : colored_obb{0, 0, 0, box});

	server->obj_service.add_object(
		std::make_shared<generated::Object_Instance>(
			convert<generated::Object_Instance>(std::make_pair(id + "_box", assigned.first->second))
			));

	displayed_target_objects.emplace(obj_instance, id);

	return id;
}

void presenter::hide_target_instance(const pn_object_instance& obj_instance, const std::string& id)
{
	server->obj_service.add_delete_object(id);
	live_mesh_prototypes.erase(id);
	server->obj_service.add_delete_object(id + "_box");
	live_boxes.erase(id + "_box");
	displayed_target_objects.erase(obj_instance);
}

void presenter::update_marking(const pn_belief_marking::ConstPtr& marking)
{
	std::scoped_lock lock(live_boxes_mutex, live_mesh_prototypes_mutex);

	if (!server)
		return;

	occlusion_detect->set_reference_cloud(cloud);
	for (const auto& entry : all_target_objects)
	{

		// only display boxes not occluded by robot
		//bool is_occluded = occlusion_detect && occlusion_detect->has_reference_cloud() && occlusion_detect->perform_detection(entry.first.first->box) == occlusion_detector::COVERED;

		constexpr bool is_occluded = false;

		auto displayed = displayed_target_objects.find(entry.first);
		if ((marking->get_probability(entry.first) > 0.5 || is_occluded)
			&& displayed != displayed_target_objects.end() ||
			planned_actions.contains(entry.first))
		{
			hide_target_instance(entry.first, entry.second);
		}
		else if (marking->get_probability(entry.first) < 0.5 && !is_occluded && displayed == displayed_target_objects.end())
		{
			show_target_instance(entry.first);
		}
	}
}

void presenter::update_goal(const pn_belief_marking::ConstPtr& marking)
{
	update_net(marking->net.lock());
	update_marking(marking);
}

void presenter::update_robot_action(const std::tuple<pn_transition::Ptr, state::pn_transition::Ptr, franka_proxy::robot_config_7dof> payload, enact_priority::operation op)
{
	const auto& [transition_0, transition_1, joints] = payload;
	franka_visualizer_.update_robot_action(payload, op);
	// execution of actions started
	if (transition_0)
		handle_robot_transition(transition_0, op);

	if (transition_1)
		handle_robot_transition(transition_1, op);
}

void presenter::handle_robot_transition(const state::pn_transition::Ptr& transition, enact_priority::operation op)
{
	if (op == enact_priority::operation::CREATE)
	{
		{
			// place action - add red bounding box
			auto obj = get_placed_object(transition);
			const auto& [box, token] = obj;

			if (box)
			{
				std::unique_lock lock(live_boxes_mutex);

				auto iter = displayed_target_objects.find(obj);
				if (iter != displayed_target_objects.end())
					hide_target_instance(obj, iter->second);

				planned_actions.emplace(obj);

				std::string id = "place " + std::to_string(box->id);
				const auto& assigned = live_boxes.emplace(id, colored_obb{ 1, 0, 0, box->box });

				if (place_target && place_target != box)
				{
					std::string old_id = "place " + std::to_string(place_target->id);
					if (server)
						server->obj_service.add_delete_object(old_id);
					live_boxes.erase(old_id);

					for (auto iter = planned_actions.begin(); iter != planned_actions.end(); ++iter)
						if (iter->first == place_target)
						{
							planned_actions.erase(iter);
							break;
						}
				}

				place_target = box;

				if (server)
					server->obj_service.add_object(
						std::make_shared<generated::Object_Instance>(
							convert<generated::Object_Instance>(std::make_pair(id, assigned.first->second))
						));
			}
		}
		{
			auto obj = get_picked_object(transition);
			const auto& [box, token] = obj;

			if (box)
			{
				planned_actions.emplace(obj);

				// pick action - add wireframe
				std::unique_lock lock(live_boxes_mutex);

				std::string id = "pick " + std::to_string(box->id);

				if (pick_target && pick_target != box)
				{
					std::string old_id = "pick " + std::to_string(pick_target->id);
					if (server)
						server->obj_service.add_delete_object(old_id);
					live_boxes.erase(old_id);

					for (auto iter = planned_actions.begin(); iter != planned_actions.end(); ++iter)
						if (iter->first == pick_target)
						{
							planned_actions.erase(iter);
							break;
						}
				}

				pick_target = box;

				const auto& temp = live_boxes.emplace(id, colored_obb{ 1, 0, 0, box->box });
				if (server)
					server->obj_service.add_object(
						std::make_shared<generated::Object_Instance>(
							convert<generated::Object_Instance>(std::make_pair(id, temp.first->second)
							)));
			}
		}
	}

	// execution of action finished
	if (op == enact_priority::operation::DELETED || op == enact_priority::operation::MISSING)
	{
		{
			auto obj = get_placed_object(transition);
			const auto& [box, token] = obj;

			if (box && place_target == box)
			{
				std::string id = "place " + std::to_string(box->id);

				// place action - remove red bounding box
				std::unique_lock lock(live_mesh_prototypes_mutex);
				planned_actions.erase(obj);
				place_target = nullptr;

				auto it = live_boxes.find(id);
				if (it != live_boxes.end())
				{
					if (server) server->obj_service.add_delete_object(id);
					live_boxes.erase(it);
				}
			}
		}
		{
			auto obj = get_picked_object(transition);
			const auto& [box, token] = obj;

			if (box && pick_target == box)
			{
				std::string id = "pick " + std::to_string(box->id);

				// pick action - remove wireframe
				std::unique_lock lock(live_boxes_mutex);
				planned_actions.erase(obj);
				pick_target = nullptr;

				auto it = live_boxes.find(id);
				if (it != live_boxes.end())
				{

					live_boxes.erase(it);
					if (server) server->obj_service.add_delete_object(id);
				}
			}
		}
	}
}
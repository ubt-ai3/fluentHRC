#pragma once

#include <WinSock2.h> // solve incompatibility issue of Windows.h and WinSock2.h (which are both included by libraries)

//#include <franka_planning/franka_actor.hpp>

#include <enact_priority/signaling_actor.hpp>

#include <state_observation/pn_model.hpp>
#include <state_observation/pn_model_extension.hpp>
#include <state_observation/workspace_objects_forward.hpp>

#include <grpc_server/server_module.h>
#include <grpc_server/util.h>

#include "franka_visualization.h"

namespace state_observation
{
/**
 * @struct colored_obb
 * @brief Oriented bounding box with color information
 * 
 * Represents a 3D bounding box with RGB color values
 * for visualization purposes.
 */
struct colored_obb
{
	double r = 1.;
	double g = 1.;
	double b = 1.;
	obb box;
};

/**
 * @struct transformed_prototype
 * @brief Prototype object with transformation
 * 
 * Represents a prototype object with its name and
 * 3D transformation matrix.
 */
struct transformed_prototype
{
	std::string prototype_name;
	Eigen::Matrix4f transform;
};

template<typename out, typename in>
out convert(const in&);

template<>
inline generated::Object_Instance convert(
	const std::pair<std::string, transformed_prototype>& in)
{
	generated::Object_Instance out;
	out.set_id(in.first);
	const auto m_obj = out.mutable_obj();
	m_obj->set_prototype_name(in.second.prototype_name);
	*m_obj->mutable_transform() =
		server::convert<4, 4>(in.second.transform);

	return out;
}

template<>
inline generated::Object_Instance convert(
	const std::pair<std::string, colored_obb>& in)
{
	generated::Object_Instance out;

	const auto& [id, color_box] = in;
	const auto& [r, g, b, box] = color_box;
	out.set_id(in.first);
	auto m_obb = out.mutable_box();
	auto color = m_obb->mutable_box_color();
	color->set_r(r * 255.);
	color->set_g(g * 255.);
	color->set_b(b * 255.);
	color->set_a(255);
	auto obbox = m_obb->mutable_obbox();
	*obbox->mutable_rotation() =
		server::convert<generated::quaternion>(box.rotation);
	auto aa = obbox->mutable_axis_aligned();
	*aa->mutable_diagonal() =
		server::convert<generated::size_3d>(box.diagonal);

	*aa->mutable_translation() =
		server::convert<generated::vertex_3d>(box.translation);

	return out;
}
}

namespace state = state_observation;

/*
class intent_visualizer
{
public:

	void update_robot_action(const state::pn_transition::Ptr transition, enact_priority::operation op);

private:

	void handle_place_transition();
};
*/

/**
 * @class presenter
 * @brief Manages and coordinates visualization of system state
 * 
 * Central class that decides what information to display to the user
 * and triggers the appropriate rendering. Handles synchronization
 * with external visualization systems like HoloLens.
 * 
 * Features:
 * - Point cloud visualization
 * - Petri net state display
 * - Robot action visualization
 * - Object instance tracking
 * - Occlusion handling
 * - HoloLens synchronization
 * - Thread-safe operation
 * - Color management
 */
class presenter
{
public:

	/**
	* If occlusion_detect is passed, target instances covered by the hand are not displayed.
	*/
	presenter(franka_visualizations visualizations, const std::shared_ptr<server::server_module>& server = nullptr,
		state::occlusion_detector::Ptr occlusion_detect = nullptr);

	void update_cloud(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud);
	void update_net(const state::pn_net::Ptr& net);
	void update_marking(const state::pn_belief_marking::ConstPtr& marking);
	void update_goal(const state::pn_belief_marking::ConstPtr& marking);
	void update_robot_action(const std::tuple<state::pn_transition::Ptr, state::pn_transition::Ptr, franka_proxy::robot_config_7dof> payload, enact_priority::operation op);

	franka_visualizer franka_visualizer_;

private:

	std::shared_ptr<server::server_module> server;

	std::map<state::pn_object_instance, std::string> displayed_target_objects;
	std::map<state::pn_object_instance, std::string> all_target_objects;
	std::set<state::pn_object_instance> planned_actions;
	std::mutex displayed_target_objects_mutex;

	pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud;
	state::occlusion_detector::Ptr occlusion_detect;

	void handle_robot_transition(const state::pn_transition::Ptr& transition, enact_priority::operation op);

	/*
	* keep track of displayed boxes and their colors
	* to sync with HoloLens
	*/
	std::map<std::string, state::colored_obb> live_boxes;
	std::mutex live_boxes_mutex;

	/*
	* keep track of displayed objects with mesh
	* to sync with HoloLens
	*/
	std::map<std::string, state::transformed_prototype> live_mesh_prototypes;
	std::mutex live_mesh_prototypes_mutex;

	state::pn_place::Ptr pick_target = nullptr;
	state::pn_place::Ptr place_target = nullptr;

	// utility methods for net

	/**
	* Renders a bounding box of the object's mean color
	* Adds it to live_objects and displayed_target_objects
	*/
	std::string show_target_instance(const state::pn_object_instance& obj_instance);

	void hide_target_instance(const state::pn_object_instance& obj_instance, const std::string& id);
};
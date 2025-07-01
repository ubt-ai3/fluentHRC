#pragma once

#include "framework.hpp"

#include <filesystem>

#include <Eigen/Core>
#include <pcl/common/common.h>
#include <pcl/PolygonMesh.h>
#include <pcl/simulation/scene.h>
#include <pcl/visualization/cloud_viewer.h>

#include <state_observation/pn_model_extension.hpp>
#include <state_observation/workspace_objects.hpp>

namespace simulation
{

/**
 * @class scene_object
 * @brief Abstract base class for objects that can be rendered in a simulation scene
 *
 * This class defines the interface for objects that can be rendered in both
 * PCL simulation scenes and PCL visualizers. It provides virtual methods for
 * rendering objects at specific timestamps.
 *
 * Features:
 * - Abstract base class for renderable objects
 * - Support for both PCL simulation and visualization
 * - Timestamp-based rendering
 */
class SIMULATION_API scene_object
{
public:
	typedef std::shared_ptr<scene_object> Ptr;

	scene_object() = default;
	virtual ~scene_object() = default;

	/**
	* timestamp - in seconds
	*/
	virtual void render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp) = 0;
	virtual void render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport = 0) = 0;
};

/**
 * @class simulated_table
 * @brief Simple model of a table top for simulation
 *
 * Represents a basic table model that can be rendered in the simulation.
 * The table has fixed dimensions and color properties.
 *
 * Features:
 * - Fixed width and breadth
 * - Standard RGB color
 * - Mesh-based rendering
 * - Support for both simulation and visualization
 */
class SIMULATION_API simulated_table : public scene_object
{
public:
	const static std::string cube_path;
	const static float width;
	const static float breadth;
	const static pcl::RGB color;

	simulated_table();

	/**
* timestamp - in seconds
*/
	virtual void render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp) override;
	virtual void render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport = 0) override;


private:
	pcl::PolygonMesh::ConstPtr table_mesh;


};

/**
 * @class movable_object
 * @brief Represents an object that can be moved in the simulation
 *
 * This class represents objects that can be manipulated in the simulation
 * environment. Each object is associated with a prototype and maintains
 * its position and orientation in 3D space.
 *
 * Features:
 * - Position and orientation tracking
 * - Prototype-based instantiation
 * - Petri net instance association
 * - Mesh-based visualization
 */
class SIMULATION_API movable_object : public scene_object
{
public:
	typedef std::shared_ptr<movable_object> Ptr;

	const state_observation::object_prototype::ConstPtr prototype;
	Eigen::Vector3f center;
	state_observation::pn_instance instance;


	movable_object(
		const state_observation::object_prototype::ConstPtr prototype,
		Eigen::Vector3f center,
		state_observation::pn_instance instance
	);

	movable_object(
		const state_observation::object_prototype::ConstPtr prototype,
		Eigen::Vector3f center,
		Eigen::Quaternionf rotation,
		state_observation::pn_instance instance
	);

	/**
* timestamp - in seconds
*/
	virtual void render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp) override;
	virtual void render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport = 0) override;

private:
	pcl::PolygonMesh::Ptr mesh;
	const std::string id;
};


/**
 * @class simulated_arm
 * @brief Simple model of a robotic arm for simulation
 *
 * Represents a basic arm model composed of a sphere (hand) and cylinder (arm).
 * The arm has a fixed shoulder position and can be moved to different positions
 * with speed control.
 *
 * Features:
 * - Fixed shoulder position
 * - Configurable movement speed
 * - TCP (Tool Center Point) tracking
 * - Mesh-based visualization
 * - Movement state tracking
 */
class SIMULATION_API simulated_arm : public scene_object
{
public:
	typedef std::shared_ptr<simulated_arm> Ptr;

	const static std::string sphere_path;
	const static std::string cylinder_path;
	const static float hand_radius;
	const static float arm_radius;
	static double speed;
	const static pcl::RGB color;

	simulated_arm(const Eigen::Vector3f& shoulder, const Eigen::Vector3f& hand);

	void move(const Eigen::Vector3f& destination, std::chrono::duration<float> timestamp);
	[[nodiscard]] bool is_moving(std::chrono::duration<float> timestamp) const;

	[[nodiscard]] Eigen::Vector3f get_tcp(std::chrono::duration<float> timestamp) const;

	[[nodiscard]] Eigen::Vector3f get_shoulder_pose(std::chrono::duration<float> timestamp) const;

	/**
	* timestamp - in seconds
	*/
	virtual void render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp) override;
	virtual void render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport = 0) override;


private:
	pcl::PolygonMesh::ConstPtr sphere_mesh;
	pcl::PolygonMesh::ConstPtr cylinder_mesh;
	Eigen::Vector3f shoulder_pose;
	Eigen::Vector3f start_pose;
	Eigen::Vector3f end_pose;
	std::chrono::duration<float> start;
	std::chrono::duration<float> duration;

	void render(std::function<void(const pcl::PolygonMesh::ConstPtr& mesh, const Eigen::Affine3f& matrix)> add, std::chrono::duration<float> timestamp);
};
/*
class SIMULATION_API simulated_franka : public scene_object
{
public:

	typedef std::shared_ptr<simulated_franka> Ptr;

	simulated_franka();

	inline static const std::vector<std::filesystem::path> links =
	{
		std::filesystem::path{"assets/models/franka/link0"},
		std::filesystem::path{"assets/models/franka/link1"},
		std::filesystem::path{"assets/models/franka/link2"},
		std::filesystem::path{"assets/models/franka/link3"},
		std::filesystem::path{"assets/models/franka/link4"},
		std::filesystem::path{"assets/models/franka/link5"},
		std::filesystem::path{"assets/models/franka/link6"},
		std::filesystem::path{"assets/models/franka/link7"}
	};

private:

	std::vector<pcl::PolygonMesh::ConstPtr> link_meshes;


};
*/

/**
 * @class environment
 * @brief Container class for simulation objects and task specification
 *
 * Manages the simulation environment, including all renderable objects and
 * the task specification represented by a Petri net. Handles object placement,
 * location management, and state updates.
 *
 * Features:
 * - Petri net-based task specification
 * - Object and token tracking
 * - Location management
 * - State distribution handling
 * - Object instance management
 * - Scene object collection
 */
class SIMULATION_API environment
{
public:
	typedef std::shared_ptr<environment> Ptr;
	typedef std::map<state_observation::pn_place::Ptr, state_observation::pn_token::Ptr> Distribution;
	typedef std::map<state_observation::pn_instance, movable_object::Ptr> ObjectTraces;
	typedef std::map<state_observation::object_prototype::ConstPtr, state_observation::pn_object_token::Ptr> TokenTraces;

	const state_observation::pn_net::Ptr net;
	Distribution distribution;
	ObjectTraces object_traces;
	TokenTraces token_traces;

	std::vector<scene_object::Ptr> additional_scene_objects;

	environment(const state_observation::object_parameters& object_params);
	environment(const state_observation::pn_net::Ptr& net,
	            Distribution distribution,
	            TokenTraces tokenTraces);

	state_observation::pn_boxed_place::Ptr add_location(const state_observation::aabb& box);
	state_observation::pn_boxed_place::Ptr try_add_location(const state_observation::aabb& box);
	[[nodiscard]] state_observation::pn_boxed_place::Ptr get_from_location(const state_observation::aabb& box) const;

	state_observation::pn_object_instance add_object(
		const state_observation::object_prototype::ConstPtr& prototype,
		const state_observation::obb& location,
		bool stack = false);

	state_observation::pn_binary_marking::Ptr update(const state_observation::pn_transition::Ptr& transition);

	[[nodiscard]] state_observation::pn_binary_marking::Ptr get_marking() const;
};

}
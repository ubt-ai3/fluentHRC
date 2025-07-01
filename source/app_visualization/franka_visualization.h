#pragma once

#include <boost/signals2.hpp>

#include <franka_voxel/motion_generator_joint_max_accel.h>
#include <state_observation/pn_model.hpp>
#include <franka_planning/franka_actor.hpp>

#include "franka_actor.h"
#include "franka_visualization_gpu.h"

/**
 * @enum Visual_Change
 * @brief Enumeration for tracking the state of visual elements
 * 
 * Represents the possible states of a visual element:
 * - ENABLED: Element is currently visible and active
 * - DISABLED: Element is temporarily hidden
 * - REVOKED: Element has been permanently removed
 */
enum Visual_Change
{
	ENABLED = 0,
	DISABLED = 1,
	REVOKED = 2
};

template<typename T>
using VisualChangeUpdate = std::variant<Visual_Change, T>;

typedef VisualChangeUpdate<std::vector<Eigen::Vector3f>> TcpUpdate;
typedef VisualChangeUpdate<franka_proxy::Visualize::VoxelRobot> VoxelUpdate;
typedef VisualChangeUpdate<joints_progress> JointProgressUpdate;

/**
 * @struct franka_visualizations
 * @brief Configuration structure for Franka robot visualization options
 * 
 * Controls which visualization features are enabled:
 * - shadow_robot: Show the robot's shadow/ghost position
 * - voxels: Shows the sweeping volume of the robot along a trajectory as a voxel-based visualization
 * - tcps: Show Tool Center Point visualization along a trajectory
 */
struct franka_visualizations
{
	bool shadow_robot, voxels, tcps;
};

/**
 * @class franka_visualizer
 * @brief High-level controller for Franka robot trajectory visualization
 * 
 * The franka_visualizer class manages multiple visualization methods for Franka robot trajectories:
 * - TCP (Tool Center Point) path visualization
 * - Voxel-based sweeping volume representation
 * - Robot ghost visualization
 * 
 * Features:
 * - Real-time trajectory updates
 * - Multiple visualization modes
 * - Signal-based update system
 * - Support for consecutive motion visualization
 * 
 * The class provides a unified interface for updating and managing different
 * visualization aspects of the Franka robot's motion.
 */
class franka_visualizer
{
public:

	explicit franka_visualizer(franka_visualizations visualizations);

	//callback which checks execution state of transitions and/or discretizes resulting motions for active visualizations
	void update_robot_action(const std::tuple<state_observation::pn_transition::Ptr, state_observation::pn_transition::Ptr, franka_proxy::robot_config_7dof> payload, enact_priority::operation op);


	franka_visualizations visualizations() const;
	void set_visual_generators(franka_visualizations visualizations);

	//signals for the resulting trajectory respresentations
	boost::signals2::signal<void(const TcpUpdate&)> tcps_signal;
	boost::signals2::signal<void(const VoxelUpdate&)> voxel_signal;
	boost::signals2::signal<void(const JointProgressUpdate&)> joints_progress_signal;

private:

	//void compute_shadow();
	//void compute_voxels();
	//void compute_tcps();

	/**
	 * \brief determine if the same object is moved by two consecutive transitions
	 * so that we can determine one consecutive path
	 * \param t0
	 * \param t1
	 */
	static bool are_transitions_consecutive(const state_observation::pn_transition::Ptr& t0, const state_observation::pn_transition::Ptr& t1);

	std::pair<state_observation::pn_transition::Ptr, state_observation::pn_transition::Ptr> visual_transitions;

	franka_proxy::Visualize::GVLInstanceFranka m_voxel_instance;
	franka_visualizations visualizations_;

	std::shared_ptr<state_observation::visual_controller_wrapper> controller_;
	state_observation::franka_agent vis_agent_;
};

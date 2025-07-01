#pragma once

#ifndef STATE_OBSERVATION__MODULE_MANAGER__HPP
#define STATE_OBSERVATION__MODULE_MANAGER__HPP

#define _SILENCE_CXX20_CISO646_REMOVED_WARNING

//#define DEBUG
#define USE_HOLOLENS
#define USE_ROBOT

#include <WinSock2.h> // solve incompatibility issue of Windows.h and WinSock2.h (which are both included by libraries)

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

#ifdef USE_HOLOLENS
#include <grpc_server/server_module.h>
#include <grpc_server/util.h>
#endif

#include <state_observation/building_estimation.hpp>
#include "intention_visualizer.hpp"
#include "task_progress_visualizer.hpp"
#ifdef USE_HOLOLENS
#include "presenter.hpp"
#endif
#include "viewer.hpp"
#include "hand_pose_estimation/hand_tracker_enact.hpp"
#include "intention_prediction/agent_manager.hpp"

#ifdef USE_ROBOT
#include "franka_actor.h"
#include <franka_high_level/franka_actor.h>
#include <franka_planning/franka_actor.hpp>
#include <simulation/franka_actor_sim.h>
#include <franka_planning/robot_agent.hpp>
#endif

namespace state = state_observation;

#ifdef USE_HOLOLENS
/**
 * @class server_callback
 * @brief Handles gRPC server callbacks for HoloLens communication
 * 
 * Manages server-side callbacks for gRPC communication with HoloLens.
 * Tracks connection state and provides signals for connection events.
 * 
 * Features:
 * - Connection state tracking
 * - Pre/post request handling
 * - First connection detection
 * - Signal-based event notification
 */
class server_callback final : public grpc::Server::GlobalCallbacks
{
public:
	~server_callback() override = default;
	void PreSynchronousRequest(grpc::ServerContext* context) override;
	void PostSynchronousRequest(grpc::ServerContext* context) override;
	
	boost::signals2::signal<void()> on_first_connect;
	bool get_first() const;

private:

	std::atomic<bool> first = false;
};

#endif
	
/**
 * @enum camera_type
 * @brief Types of supported cameras
 * 
 * Defines the different types of cameras that can be used:
 * - SIMULATION: Virtual camera for simulation
 * - KINECT_V2: Microsoft Kinect v2 sensor
 * - REALSENSE: Intel RealSense camera
 */
enum class camera_type
{
	SIMULATION,
	KINECT_V2,
	REALSENSE
};
	
/**
 * @class module_manager
 * @brief Manages and coordinates system modules and actors
 * 
 * Central class that creates, initializes, and interconnects all system
 * modules and actors. Handles configuration and lifecycle management
 * of the entire system.
 * 
 * Features:
 * - Module initialization
 * - Actor management
 * - Camera configuration
 * - Point cloud processing
 * - Object detection and tracking
 * - Building estimation
 * - HoloLens integration
 * - Robot control
 * - Hand tracking
 * - Visualization management
 */
class module_manager
{
public:
	#include "signal_types.hpp"

	typedef std::shared_ptr<enact_core::entity_id> entity_id;

	module_manager(int argc, char* argv[], camera_type camera);
	~module_manager();

	//TODO test
	const int cloud_skip_count = 1;
	int current_skipped_clouds = 0;
private:
	
	enact_core::world_context world;
	std::shared_ptr<const state::object_parameters> object_params;
	std::shared_ptr<const state::kinect2_parameters> kinect_params;
	std::unique_ptr<state::pointcloud_preprocessing> pc_prepro;
	//std::unique_ptr<segment_detector> obj_detect;
	//std::unique_ptr<object_tracker> obj_track;
	std::unique_ptr<state::place_classification_handler> obj_classify;
	std::unique_ptr<state::building_estimation> build_estimation;
#ifdef USE_HOLOLENS
	std::shared_ptr<server::server_module> server;
#endif
	prediction::agent_manager::Ptr agent_manage;

	std::unique_ptr<task_progress_visualizer> progress_viewer;
	std::unique_ptr<intention_visualizer> intention_view;
#ifdef USE_ROBOT
	std::unique_ptr<robot::agent> robot;
#endif
	std::unique_ptr<::hand_pose_estimation::hand_tracker_enact> hand_track;
#ifdef USE_HOLOLENS
	std::unique_ptr<presenter> present;
#endif
	std::unique_ptr<::viewer> view;

	pcl::PointCloud<PointT>::ConstPtr last_cloud = nullptr;
};

#endif // !STATE_OBSERVATION__MODULE_MANAGER__HPP

#pragma once

#ifndef STATE_OBSERVATION__VIEWER__HPP
#define STATE_OBSERVATION__VIEWER__HPP



#include <memory>

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core/core.hpp>

#include <boost/timer/timer.hpp>

#include <enact_core/id.hpp>
#include <enact_core/data.hpp>
#include <enact_core/world.hpp>
#include <enact_priority/priority_actor.hpp>

#include <enact_priority/signaling_actor.hpp>


#include <hand_pose_estimation/hand_tracker_enact.hpp>

#include <state_observation/pointcloud_util.hpp>
#include <state_observation/workspace_objects_forward.hpp>


namespace state = state_observation;

namespace state_observation {
	class building;
}

/**
 * @class viewer
 * @brief A visualization system for displaying hands, objects, and point clouds in real-time.
 * 
 * The viewer class provides a comprehensive visualization system that combines:
 * - Point cloud visualization using PCL
 * - OpenCV-based image display
 * - Real-time object tracking visualization
 * - Building and hand pose visualization
 * 
 * Key features:
 * - Thread-safe asynchronous updates
 * - Support for multiple viewports
 * - Real-time overlay capabilities
 * - Event-based update system
 * 
 * The viewer maintains synchronization between different visualization components
 * and provides a unified interface for updating various visual elements.
 */
class viewer
{
public:
	

	typedef pcl::PointXYZRGBA PointT;
	typedef std::shared_ptr<enact_core::entity_id> strong_id;
	typedef std::weak_ptr<enact_core::entity_id> weak_id;
	typedef enact_core::lockable_data_typed<state::object_instance> object_instance_data;
	typedef enact_core::lockable_data_typed<state::building> building_instance_data;
	typedef enact_core::lockable_data_typed<hand_pose_estimation::hand_trajectory> hand_instance_data;

	viewer(enact_core::world_context& world,
		std::string opencv_viewer_title = std::string("OpenCV Viewer"),
		std::string pcl_viewer_title = std::string("Point Cloud Viewer"));

	virtual ~viewer() noexcept;

	void run_sync();
	
	void run_async();

	/**
	* Add visual element to the event loop for update
	* Thread-safe, async.
	*/
	//@{
	void update_cloud(const pcl::PointCloud<PointT>::ConstPtr& cloud);
	void update_object(const strong_id& id, enact_priority::operation op);
	void update_building(const strong_id& id, enact_priority::operation op);
	void update_hand(const strong_id& id, enact_priority::operation op);
	void update_image(const cv::Mat& img);
	void update_overlay(const cv::Mat& img);
	void update_segment(const std::shared_ptr<state::pc_segment>& seg);
	void update_element(std::function<void(pcl::visualization::PCLVisualizer&)>&& f);
	//@}

	//debug
	void add_bounding_box(const state::obb& obox, const std::string& name,float r=1,float g=1,float b =1);
	void remove_bounding_box(const std::string& name);

	std::shared_ptr<boost::signals2::signal<void(void)>> get_close_signal();
	
	static std::string to_string(const strong_id& id);

	[[nodiscard]] int viewport_full_window_id() const;
	[[nodiscard]] int viewport_overlay_id() const;

private:
	std::set<std::string> live_objects;

	//map<pointer to building, set of blocks in pcl>
	std::map<weak_id, std::set<std::string>, std::owner_less<weak_id>> live_buildings;
	std::set<std::string> live_hands;

	std::shared_ptr<boost::signals2::signal<void(void)>> close_signal;
	std::unique_ptr<std::thread> internal_thread;




	
	void draw_hand_on_image(const weak_id& id);
	/*
	 * @ref{mutex} must be locked prior to calling this functions
	 */
	void draw_obb(const state::obb& box, const std::string& id, double r, double g, double b, int viewport = 0);
	void remove_obb(const std::string& id);
	void draw_object(const strong_id& id);
	void draw_building(const strong_id& id);
	void draw_hand(const strong_id& id);
	
	void show_image();

	/*
	 * @attend precond: seg != nullptr && prototype.hasmesh()
	 */
	static Eigen::Affine3f calc_proto_transform(const state::pc_segment& seg);
	
	/**
	 * Calls spin once
	 */
	void refresh();
	void redraw();

	enact_core::world_context& world;
	std::shared_ptr<pcl::visualization::PCLVisualizer> pcl_viewer;
	pcl::PointCloud<PointT>::ConstPtr cloud;

	int viewport_full_window = 0;
	int viewport_overlay = 1;

	cv::Mat img;
	cv::Mat overlay;
	std::atomic<uint64_t> cloud_stamp;
	boost::timer::cpu_timer img_timer;

	std::string cv_window_title;
	std::string pc_window_title;
	std::mutex viewer_mutex;

	bool redraw_scheduled;

		/*
		* Add function to the event loop queue to execute.
		* Execution happens in separate thread.
		* Thread-safe
		*/
	void schedule(std::function<void()>&& f, unsigned int priority = 0);

	std::list<std::function<void()>> queue;
	std::mutex queue_mutex;
	std::condition_variable queue_condition_variable;

	std::set<std::string> box_names;
	std::mutex box_mutex;
};


#endif // !STATE_OBSERVATION__VIEWER__HPP

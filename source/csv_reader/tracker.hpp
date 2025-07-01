#pragma once

#ifndef CSV_READER_TRACKER
#define CSV_READER_TRACKER

#include <map>
#include <vector>

#include <Eigen/Core>

#include "csvworker.h"


///////////////////////////////////////////////////
//
//
//              Enum: hand_type
//
//
///////////////////////////////////////////////////

enum class hand_type : uint8_t {
	COLOR_RIGHT,
	COLOR_LEFT,
	TAN_LEFT,
	TAN_RIGHT
};



///////////////////////////////////////////////////
//
//
//              Class: tracker
//
//
///////////////////////////////////////////////////

class tracker {
public:
	tracker(const std::string& path, Eigen::Matrix<float, 2, 4> projection_matrix);
	tracker(const std::string& path, Eigen::Matrix4f projection_matrix);

	Eigen::Vector2f get_position_2d(hand_type hand, float time);
	Eigen::Vector3f get_position_3d(hand_type hand, float time);



private:
	float frame_rate_;
	float first_frame_;
	float last_frame_;

	//optitrack.time + time_offset = kinect.time
	float time_offset_sec_; 

	std::map<hand_type, int> hand_to_rigid_body_id_;

	Eigen::Matrix<float, 2, 4> opti_track_to_rgb_;
	Eigen::Matrix4f opti_track_to_kinect_;

	static const std::map<std::string, float> time_offsets_;
	/*
	* maps the team id to the mapping from the hand type to its index in the vector movement_
	*/
//	static const std::map<std::string, std::map<hand_type, size_t>> team_to_tracker_mapping_;

	std::vector<std::string> markers_;

	// rigid body id[timestamp[markers]]
	std::vector<std::vector<std::vector<Eigen::Vector3f>>> movement_;


	tracker(const std::filesystem::path& path, Eigen::Matrix<float, 2, 4> projection_matrix, Eigen::Matrix4f kinect_mat);

	static float parseFloat(const std::string* value);

	/*
	* estimated center of the palm given the rigid body
	*/
	[[nodiscard]] Eigen::Vector3f get_offset_opti_track_position(hand_type hand, float time) const;
};

#endif // !CSV_READER_TRACKER
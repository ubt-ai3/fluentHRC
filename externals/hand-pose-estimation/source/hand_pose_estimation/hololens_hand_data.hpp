#pragma once

#include <chrono>

#include "framework.h"

#include <Eigen/Dense>

namespace hand_pose_estimation::hololens
{

	enum class HANDPOSEESTIMATION_API hand_index
	{
		LEFT,
		RIGHT
	};

	enum class HANDPOSEESTIMATION_API tracking_status
	{
		NOT_TRACKED,
		INERTIAL_ONLY,
		TRACKED
	};

	enum class HANDPOSEESTIMATION_API hand_key_point
	{
		PALM,
		WRIST,
		THUMB_METACARPAL,
		THUMB_PROXIMAL,
		THUMB_DISTAL,
		THUMB_TIP,
		INDEX_METACARPAL,
		INDEX_PROXIMAL,
		INDEX_INTERMEDIATE,
		INDEX_DISTAL,
		INDEX_TIP,
		MIDDLE_METACARPAL,
		MIDDLE_PROXIMAL,
		MIDDLE_INTERMEDIATE,
		MIDDLE_DISTAL,
		MIDDLE_TIP,
		RING_METACARPAL,
		RING_PROXIMAL,
		RING_INTERMEDIATE,
		RING_DISTAL,
		RING_TIP,
		LITTLE_METACARPAL,
		LITTLE_PROXIMAL,
		LITTLE_INTERMEDIATE,
		LITTLE_DISTAL,
		LITTLE_TIP,
		SIZE
	};

	inline static const std::map<hand_key_point, std::string> hr_hand_key_points =
	{
		{hand_key_point::PALM, "palm"},
		{hand_key_point::WRIST, "wrist"},
		{hand_key_point::THUMB_METACARPAL, "thumb metacarpal"},
		{hand_key_point::THUMB_PROXIMAL, "thumb proximal"},
		{hand_key_point::THUMB_DISTAL, "thumb distal"},
		{hand_key_point::THUMB_TIP, "thumb tip"},
		{hand_key_point::INDEX_METACARPAL, "index metacarpal"},
		{hand_key_point::INDEX_PROXIMAL, "index_proximal"},
		{hand_key_point::INDEX_INTERMEDIATE, "index intermediate"},
		{hand_key_point::INDEX_DISTAL, "index distal"},
		{hand_key_point::INDEX_TIP, "index tip"},
		{hand_key_point::MIDDLE_METACARPAL, "middle metacarpal"},
		{hand_key_point::MIDDLE_PROXIMAL, "middle proximal"},
		{hand_key_point::MIDDLE_INTERMEDIATE, "middle intermediate"},
		{hand_key_point::MIDDLE_DISTAL, "middle distal"},
		{hand_key_point::MIDDLE_TIP, "middle tip"},
		{hand_key_point::RING_METACARPAL, "ring metacarpal"},
		{hand_key_point::RING_PROXIMAL, "ring proximal"},
		{hand_key_point::RING_INTERMEDIATE, "ring intermediate"},
		{hand_key_point::RING_DISTAL, "ring distal"},
		{hand_key_point::RING_TIP, "ring tip"},
		{hand_key_point::LITTLE_METACARPAL, "little metacarpal"},
		{hand_key_point::LITTLE_PROXIMAL, "little proximal"},
		{hand_key_point::LITTLE_INTERMEDIATE, "little intermediate"},
		{hand_key_point::LITTLE_DISTAL, "little distal"},
		{hand_key_point::LITTLE_TIP, "little tip"}
	};
	
	struct HANDPOSEESTIMATION_API hand_key_data
	{		
		Eigen::Vector3f position;
		Eigen::Quaternionf rotation;
		float radius = 0.f;
	};
	
	struct HANDPOSEESTIMATION_API hand_data
	{
		typedef std::shared_ptr<hand_data> Ptr;
		typedef std::shared_ptr<const hand_data> ConstPtr;
		
		bool valid;
		hand_index hand;
		tracking_status tracking_stat;

		Eigen::Vector3f grip_position;
		Eigen::Quaternionf grip_rotation;

		Eigen::Vector3f aim_position;
		Eigen::Quaternionf aim_rotation;

		hand_key_data key_data[(size_t)hand_key_point::SIZE];

		bool is_grasped;
		std::chrono::utc_clock::time_point utc_timestamp;
	};
}
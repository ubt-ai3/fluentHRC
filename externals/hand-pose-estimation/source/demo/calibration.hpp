#pragma once

#ifndef STATE_OBSERVATION__CALIBRATION__HPP__HPP
#define STATE_OBSERVATION__CALIBRATION__HPP

#include <thread>

#include "parameter_set.hpp"

#include "eigen_serialization.hpp"

#include <Eigen/Core>
#include <opencv2/core/types.hpp>


/**
 *************************************************************************
 *
 * @class kinect2_parameters
 *
 * Static properties of the Microsoft Kinect v2 depth camera.
 *
 ************************************************************************/

class kinect2_parameters : public parameter_set {
public:
	kinect2_parameters();

	~kinect2_parameters();

	// Camera intrinsics for depth sensor
	float focal_length_x;
	float focal_length_y;
	Eigen::Vector2f principal_point;
	float radial_distortion_second_order;
	float radial_distortion_fourth_order;
	float radial_distortion_sixth_order;


	Eigen::Matrix<float,3,4> rgb_projection;
	Eigen::Matrix<float, 3, 4> depth_projection;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(focal_length_x);
		ar& BOOST_SERIALIZATION_NVP(focal_length_y);
		ar& BOOST_SERIALIZATION_NVP(principal_point);
		ar& BOOST_SERIALIZATION_NVP(radial_distortion_second_order);
		ar& BOOST_SERIALIZATION_NVP(radial_distortion_second_order);
		ar& BOOST_SERIALIZATION_NVP(radial_distortion_fourth_order);
		ar& BOOST_SERIALIZATION_NVP(radial_distortion_sixth_order);
		ar& BOOST_SERIALIZATION_NVP(rgb_projection);
		ar& BOOST_SERIALIZATION_NVP(depth_projection);
	}

private:
	std::shared_ptr<std::thread> init_thread;
};

#endif // !STATE_OBSERVATION__CALIBRATION__HPP

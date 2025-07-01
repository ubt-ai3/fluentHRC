#pragma once
#ifndef SAMPLE_REGISTRATION_TESTER__PROJECTION_MATRIX
#define SAMPLE_REGISTRATION_TESTER__PROJECTION_MATRIX

#include <Eigen/Core>

Eigen::Matrix<float, 2, 4> compute_projection_opti_track_to_rgb();
Eigen::Matrix<float, 4, 4> compute_transformation_opti_track_to_camera_space();
#endif // SAMPLE_REGISTRATION_TESTER__PROJECTION_MATRIX

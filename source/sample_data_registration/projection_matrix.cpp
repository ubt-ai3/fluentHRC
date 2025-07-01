#include "projection_matrix.hpp"

#include <Windows.h>
#include <Kinect.h>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry>


Eigen::Matrix<float, 2, 4> compute_projection_opti_track_to_rgb() {
	Eigen::MatrixXd A(18,4);

	A << -0.071305, 0.014916, 0.880532, 1.0,
		-0.004013, 0.012207, 0.877308, 1.0,
		0.000004, 0.015608, 1.001212, 1.0,

		0.547196, -0.029478, 0.110121, 1.0,
		0.549692, -0.033713, -0.010473, 1.0,
		0.465424, -0.032748, 0.033514, 1.0,

		-0.018503, -0.007714, 0.126539, 1.0,
		-0.085473, -0.006732, 0.128023, 1.0,
		-0.106615, -0.007659, 0.082798, 1.0,

		0.106219, 0.287371, 0.084524, 1.0,
		0.154349, 0.300004, 0.13002, 1.0,
		0.238173, 0.253898, 0.051106, 1.0,
	
		0.41936,	0.066429,	0.582817, 1.0,
		0.481844,	0.162646,	0.54533, 1.0,
		0.520747,	0.105416,	0.610509, 1.0, 

		0.546208,	0.087784,	0.379175, 1.0,
		0.58314,	0.090992,	0.434977, 1.0,
		0.560205,	0.072662,	0.475365, 1.0
		;



	Eigen::Matrix<double, 18, 2> B;
	B << 1300,	986,
		1297,	917,
		1416,	924,

		664,	381,
		565,	372,
		591,	439,

		588,	867,
		578,	933,
		533,	951,

		556,	678,
		626,	632,
		557,	540,

		1072,	469,
		1027,	367,
		1096,	365,
		
		874,	359,
		928,	331,
		961,	362
		;



	Eigen::MatrixXd M(2 * A.rows(), 2 * A.cols());
	M << A, Eigen::MatrixXd::Zero(A.rows(), A.cols()),
		Eigen::MatrixXd::Zero(A.rows(), A.cols()), A;

	Eigen::VectorXd b(2 * B.rows());
	b << B.col(0), B.col(1);

	std::cout << "Here is the matrix M:\n" << M << std::endl;
	std::cout << "Here is the right hand side b:\n" << b << std::endl;

	Eigen::VectorXd sol = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
	std::cout << "The least-squares solution is:\n"
		<< sol << std::endl;

	Eigen::Matrix<float, 2, 4> projection;
	projection << static_cast<float>(sol(0)), static_cast<float>(sol(1)), static_cast<float>(sol(2)), static_cast<float>(sol(3)),
		static_cast<float>(sol(4)), static_cast<float>(sol(5)), static_cast<float>(sol(6)), static_cast<float>(sol(7));

	return projection;
}

Eigen::Vector3f to_camera_space(UINT16 depth_x, UINT16 depth_y, UINT16 depth_z)
{
	IKinectSensor* sensor;
	ICoordinateMapper* mapper;

	GetDefaultKinectSensor(&sensor);
	sensor->Open();
	sensor->get_CoordinateMapper(&mapper);

	DepthSpacePoint depthSpacePoint = { static_cast<float>(depth_x), static_cast<float>(depth_y) };
	CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
	mapper->MapDepthPointToCameraSpace(depthSpacePoint, depth_z, &cameraSpacePoint);

	return Eigen::Vector3f(cameraSpacePoint.X, cameraSpacePoint.Y, cameraSpacePoint.Z);
}

/*
* Returns the normalized vector that lies in the plane of (s,v) and is orthogonal to s
*/
Eigen::Vector3f orthogonal(const Eigen::Vector3f& s, const Eigen::Vector3f& v)
{
	return (v - (v.dot(s) / s.squaredNorm()) * s).normalized();
}

Eigen::Matrix<float, 4, 4> compute_transformation_opti_track_to_camera_space()
{
	/*
	Eigen::Vector3f origin_kinect(to_camera_space(354, 172, 1282));
	origin_kinect = to_camera_space(354, 340, 1123);
	Eigen::Vector3f p1_kinect(to_camera_space(354, 172, 1282));
	Eigen::Vector3f p2_kinect(to_camera_space(166, 360, 1105));
	origin_kinect = to_camera_space(354, 340, 1123);
	*/

	Eigen::Vector3f origin_kinect(0.257966013f, -0.404168636f, 1.12300003f);
	Eigen::Vector3f p1_kinect(0.299640776f, 0.127107910f, 1.28200006f);
	Eigen::Vector3f p2_kinect(-0.270246437f, -0.417868099f, 1.10500002f);

	Eigen::Vector3f camera_space_scaling(1.319f, 0.8797f, 1.f);
	origin_kinect = origin_kinect.cwiseProduct(camera_space_scaling);
	p1_kinect = p1_kinect.cwiseProduct(camera_space_scaling);
	p2_kinect = p2_kinect.cwiseProduct(camera_space_scaling);

	Eigen::Vector3f origin_opti(-0.018503f, -0.007714f, 0.126539f);
	Eigen::Vector3f p1_opti(0.465424f, -0.032748f, 0.033514f);
	Eigen::Vector3f p2_opti(-0.004013f, 0.012207f, 0.877308f);

	Eigen::Vector3f x_axis_kinect = (p1_kinect - origin_kinect).normalized();
	Eigen::Vector3f x_axis_opti = (p1_opti - origin_opti).normalized();

	Eigen::Vector3f y_axis_kinect = orthogonal(x_axis_kinect, p2_kinect - origin_kinect);
	Eigen::Vector3f y_axis_opti = orthogonal(x_axis_opti, p2_opti - origin_opti);


	Eigen::Vector3f opti_x_kinect = Eigen::Quaternionf::FromTwoVectors(x_axis_opti, x_axis_kinect) * x_axis_opti;
	Eigen::Vector3f opti_y_kinect = Eigen::Quaternionf::FromTwoVectors(x_axis_opti, x_axis_kinect) * y_axis_opti;
	float angle = std::atan2(x_axis_kinect.dot(opti_y_kinect.cross(y_axis_kinect)), opti_y_kinect.dot(y_axis_kinect));

	Eigen::Quaternionf rotation =
		Eigen::AngleAxisf(angle, x_axis_kinect) *
		Eigen::Quaternionf::FromTwoVectors(x_axis_opti, x_axis_kinect);

	opti_x_kinect = rotation * x_axis_opti;
	opti_y_kinect = rotation * y_axis_opti;

	Eigen::Transform<float,3,Eigen::Affine> transformation =
//		Eigen::Scaling(camera_space_scaling.cwiseInverse()) *
		Eigen::Translation3f(origin_kinect) *
		rotation *
		Eigen::Translation3f(-origin_opti);

//	Eigen::Matrix4f scaling = Eigen::Matrix4f::Identity();
//	scaling.diagonal() = camera_space_scaling.cwiseInverse();

	return transformation.matrix();
}

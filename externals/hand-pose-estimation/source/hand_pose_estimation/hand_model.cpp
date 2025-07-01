#include "hand_model.hpp"

#include <cmath>
#include <fstream>
#include <queue>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <opencv2/imgproc.hpp>

#include <pcl/common/centroid.h>
#include <pcl/common/projection_matrix.h>
#include <pcl/features/normal_3d.h>


class cv::Mat;
template class cv::Rect_<int>;
template class cv::Point_<int>;

namespace hand_pose_estimation
{

/**
*************************************************************************
*
* @class XYZRGBA2XYZ
*
* Helper class to construct an octree
*
************************************************************************/


class XYZRGBA2XYZ : public pcl::PointRepresentation < visual_input::PointT>
{
public:
	XYZRGBA2XYZ()
	{
		this->nr_dimensions_ = 3;
		this->trivial_ = true;
	}

	void copyToFloatArray(const visual_input::PointT& p, float* out) const
	{
		out[0] = p.x;
		out[1] = p.y;
		out[2] = p.z;
	}

};


class XYZL2XY : public pcl::PointRepresentation < pcl::PointXYZL>
{
public:
	XYZL2XY()
	{
		this->nr_dimensions_ = 2;
		this->trivial_ = true;
	}

	void copyToFloatArray(const  pcl::PointXYZL& p, float* out) const
	{
		out[0] = p.x;
		out[1] = p.y;
	}

};



/////////////////////////////////////////////////////////////
//
//
//  Class: visual_input
//
//
/////////////////////////////////////////////////////////////


const std::chrono::high_resolution_clock::time_point  visual_input::system_time_start = std::chrono::high_resolution_clock().now();

visual_input::visual_input(pcl::PointCloud<PointT>::ConstPtr cloud, bool flip,
	Eigen::Matrix<float, 3, 4> projection,
	Eigen::Vector3f sensor_origin)
	:
	cloud(std::move(cloud)),
	img(this->cloud->height, this->cloud->width, CV_8UC3),
	img_projection(projection.isZero() ? compute_cloud_projection(this->cloud) : projection),
	cloud_projection(img_projection),
	img_to_cloud_pixels(Eigen::Affine2f::Identity()),
	sensor_origin(sensor_origin),
	timestamp_seconds(this->cloud->header.stamp ? std::chrono::microseconds(this->cloud->header.stamp) : relative_timestamp())
{
	if (this->cloud->empty())
		throw std::runtime_error("Point cloud is empty");

	if (!this->cloud->isOrganized())
		throw std::runtime_error("Point cloud is not organized");

	// We exploit that the copy constructor makes a shallow copy
	cv::Mat img2(img);

	for (int h = 0; h < img.rows; h++) {
		for (int w = 0; w < img.cols; w++) {
			pcl::PointXYZRGBA point = this->cloud->at(w, h);

			Eigen::Vector3i rgb = point.getRGBVector3i();
			cv::Vec3b& pixel = img2.at<cv::Vec3b>(h, flip ? img.cols - w - 1 : w);

			pixel[0] = rgb[2];
			pixel[1] = rgb[1];
			pixel[2] = rgb[0];
		}
	}


}

visual_input::visual_input(cv::Mat img)
	:
	cloud(nullptr),
	img(std::move(img)),
	img_projection(Eigen::Matrix<float, 3, 4>::Identity()),
	cloud_projection(Eigen::Matrix<float, 3, 4>::Identity()),
	img_to_cloud_pixels(Eigen::Affine2f::Identity()),
	timestamp_seconds(relative_timestamp())
{
}

visual_input::visual_input(pcl::PointCloud<PointT>::ConstPtr cloud,
	cv::Mat img,
	Eigen::Matrix<float, 3, 4> img_projection,
	Eigen::Matrix<float, 3, 4> cloud_projection,
	Eigen::Vector3f sensor_origin)
	:
	cloud(std::move(cloud)),
	img(std::move(img)),
	img_projection(std::move(img_projection)),
	cloud_projection(cloud_projection.isZero() ? compute_cloud_projection(this->cloud) : cloud_projection),
	img_to_cloud_pixels(compute_cloud_to_image_coordinates(this->cloud, this->img, this->img_projection)),
	sensor_origin(sensor_origin),
	timestamp_seconds(this->cloud->header.stamp ? std::chrono::microseconds(this->cloud->header.stamp) : relative_timestamp()),
	extra_image(true)
{
}

std::chrono::duration<float> visual_input::relative_timestamp()
{
	return std::chrono::high_resolution_clock().now() - system_time_start;
}

inline bool visual_input::has_cloud() const
{
	return !!cloud;
}

inline bool visual_input::is_valid_point(int x, int y) const
{
	if (!has_cloud() || x < 0 || y < 0 ||
		x >= cloud->width || y >= cloud->height)
		return false;

	const PointT& p = cloud->at(x, y);

	return !std::isnan(p.x) && !std::isnan(p.y) && !std::isnan(p.z) && p.z != 0;
}

inline bool visual_input::is_valid_point(const cv::Point2i& pixel) const
{
	return is_valid_point(pixel.x, pixel.y);
}

inline const visual_input::PointT& visual_input::get_point(const cv::Point2i& pixel) const
{
	return cloud->at(pixel.x, pixel.y);
}

cv::Point2i visual_input::to_cloud_pixel_coordinates(const cv::Point2i& img_p) const
{
	if (cloud->width == img.rows && cloud->height == img.cols)
		return img_p;

	Eigen::Vector2f cloud_p = img_to_cloud_pixels * Eigen::Vector2f(img_p.x, img_p.y);


	return cv::Point2i(cap(cloud_p.x(), cloud->width), cap(cloud_p.y(), cloud->height));
}

cv::Rect2i visual_input::to_cloud_pixel_coordinates(const cv::Rect2i& box) const
{
	return cv::Rect2i(to_cloud_pixel_coordinates(box.tl()), to_cloud_pixel_coordinates(box.br()));
}


//cv::Point2i visual_input::to_cloud_pixel_coordinates(const Eigen::Vector3f& p) const
//{
//	// Eigen::Vector2f cloud_p = (cloud_projection * p.homogeneous()).hnormalized();
//	
//	return cv::Point2i(cap(cloud_p.x(), cloud->width), cap(cloud_p.y(), cloud->height));
//}

cv::Point2i visual_input::to_cloud_pixel_coordinates(const Eigen::Vector3f& p) const
{
	// Eigen::Vector2f cloud_p = (cloud_projection * p.homogeneous()).hnormalized();
	float denom = cloud_projection(2, 0) * p.x() +
		cloud_projection(2, 1) * p.y() +
		cloud_projection(2, 2) * p.z() +
		cloud_projection(2, 3);

	float x = (cloud_projection(0, 0) * p.x() +
		cloud_projection(0, 1) * p.y() +
		cloud_projection(0, 2) * p.z() +
		cloud_projection(0, 3)) / denom;

	float y = (cloud_projection(1, 0) * p.x() +
		cloud_projection(1, 1) * p.y() +
		cloud_projection(1, 2) * p.z() +
		cloud_projection(1, 3)) / denom;

	return cv::Point2i(cap(x,cloud->width), cap(y, cloud->height));
}
	
cv::Point2i visual_input::to_cloud_pixel_coordinates_uncapped(const Eigen::Vector3f& p) const
{
	// Eigen::Vector2f cloud_p = (cloud_projection * p.homogeneous()).hnormalized();
	float denom = cloud_projection(2, 0) * p.x() +
		cloud_projection(2, 1) * p.y() +
		cloud_projection(2, 2) * p.z() +
		cloud_projection(2, 3);

	float x = (cloud_projection(0, 0) * p.x() +
		cloud_projection(0, 1) * p.y() +
		cloud_projection(0, 2) * p.z() +
		cloud_projection(0, 3)) / denom;

	float y = (cloud_projection(1, 0) * p.x() +
		cloud_projection(1, 1) * p.y() +
		cloud_projection(1, 2) * p.z() +
		cloud_projection(1, 3)) / denom;
	
	return cv::Point2i(x+0.5f, y+0.5f);
}

cv::Point2i visual_input::to_img_coordinates(const PointT& p) const
{
	if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
		cv::Point2i(-1, -1);

	return to_img_coordinates(p.getArray3fMap());
}

cv::Point2i visual_input::to_img_coordinates(const Eigen::Vector3f& p) const
{
	float denom = img_projection(2, 0) * p.x() +
		img_projection(2, 1) * p.y() +
		img_projection(2, 2) * p.z() +
		img_projection(2, 3);

	float x = (img_projection(0, 0) * p.x() +
		img_projection(0, 1) * p.y() +
		img_projection(0, 2) * p.z() +
		img_projection(0, 3)) / denom;

	float y = (img_projection(1, 0) * p.x() +
		img_projection(1, 1) * p.y() +
		img_projection(1, 2) * p.z() +
		img_projection(1, 3)) / denom;
	
	//Eigen::Vector2f x = (img_projection * p.homogeneous()).hnormalized();
	return cv::Point2i(cap(x, img.cols), cap(y, img.rows));
}

inline cv::Point2i visual_input::to_img_coordinates_uncapped(const Eigen::Vector3f& p) const
{
	float denom = img_projection(2, 0) * p.x() +
		img_projection(2, 1) * p.y() +
		img_projection(2, 2) * p.z() +
		img_projection(2, 3);

	float x = (img_projection(0, 0) * p.x() +
		img_projection(0, 1) * p.y() +
		img_projection(0, 2) * p.z() +
		img_projection(0, 3)) / denom;

	float y = (img_projection(1, 0) * p.x() +
		img_projection(1, 1) * p.y() +
		img_projection(1, 2) * p.z() +
		img_projection(1, 3)) / denom;

	//Eigen::Vector2f x = (img_projection * p.homogeneous()).hnormalized();
	return cv::Point2i(x+0.5f, y+0.5f);
}

inline Eigen::Vector2f visual_input::to_img_coordinates_vec(const Eigen::Vector3f& p) const
{
	float denom = img_projection(2, 0) * p.x() +
		img_projection(2, 1) * p.y() +
		img_projection(2, 2) * p.z() +
		img_projection(2, 3);

	float x = (img_projection(0, 0) * p.x() +
		img_projection(0, 1) * p.y() +
		img_projection(0, 2) * p.z() +
		img_projection(0, 3)) / denom;

	float y = (img_projection(1, 0) * p.x() +
		img_projection(1, 1) * p.y() +
		img_projection(1, 2) * p.z() +
		img_projection(1, 3)) / denom;

	//Eigen::Vector2f x = (img_projection * p.homogeneous()).hnormalized();
	return Eigen::Vector2f(x,y);
}

Eigen::Affine2f visual_input::compute_cloud_to_image_coordinates(pcl::PointCloud<PointT>::ConstPtr cloud, cv::Mat img, Eigen::Matrix<float, 3, 4> projection)
{
	Eigen::MatrixXf A = Eigen::Matrix<float, 8, 4>::Zero();
	Eigen::VectorXf b = Eigen::Matrix<float,8,1>::Zero();

	for (Eigen::Index i = 0; i < 4; i++)
	{
		cv::Point2i p((i % 2 ? 0.8 : 0.2) * cloud->width, (i / 2 ? 0.8 : 0.2) * cloud->height);
		cv::Point2i inc(i % 2 ? -1 : 1, i / 2 ? -1 : 1);

		int count = 0;
		while (count++ < 200 && (std::isnan(cloud->at(p.x, p.y).z) || cloud->at(p.x, p.y).z == 0))
			p += inc;

		if (count >= 200)
			continue;
		
		Eigen::Vector2f b_seg = (projection * cloud->at(p.x, p.y).getVector3fMap().homogeneous()).hnormalized();

		A(2 * i, 0) = b_seg.x();
		A(2 * i, 2) = 1;
		A(2 * i + 1, 1) = b_seg.y();
		A(2 * i + 1, 3) = 1;
		b(2 * i) = p.x;
		b(2 * i + 1) = p.y;


	}

	Eigen::Vector4f x = A.householderQr().solve(b);

	return Eigen::Translation2f(x.segment(2, 2)) * Eigen::Scaling(x.segment(0, 2));
}

Eigen::Matrix<float, 3, 4> visual_input::compute_cloud_projection(const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
	Eigen::Matrix<float, 3, 4, Eigen::RowMajor> projection;
	std::vector<int> indices;

	for (int i = 0; i < 4; i++)
	{
		cv::Point2i p((i % 2 ? 0.8 : 0.2) * cloud->width, (i / 2 ? 0.8 : 0.2) * cloud->height);
		cv::Point2i inc(i % 2 ? -1 : 1, i / 2 ? -1 : 1);

		while (std::isnan(cloud->at(p.x, p.y).z) || cloud->at(p.x, p.y).z == 0)
			p += inc;

		indices.push_back(p.y * cloud->width + p.x);
	}

	for (int i = 0; i < 4; i++)
	{
		cv::Point2i p((i % 2 ? 0.6 : 0.4) * cloud->width, (i / 2 ? 0.6 : 0.4) * cloud->height);
		cv::Point2i inc(i % 2 ? -1 : 1, i / 2 ? -1 : 1);

		while (std::isnan(cloud->at(p.x, p.y).z) || cloud->at(p.x, p.y).z == 0)
			p += inc;

		indices.push_back(p.y * cloud->width + p.x);
	}

	pcl::estimateProjectionMatrix<PointT>(cloud, projection, indices);

	return projection;
}

int inline visual_input::cap(float x, int up) const
{
	return std::max(0, std::min(up - 1, static_cast<int>(std::round(x))));
};




/////////////////////////////////////////////////////////////
//
//
//  Class: finger_kinematic_parameters
//
//
/////////////////////////////////////////////////////////////

const std::vector<float> finger_kinematic_parameters::alpha = { M_PI_2, 0.f,  0.f, 0.f, 0.f };

finger_kinematic_parameters::finger_kinematic_parameters(std::vector<float>&& a,
	std::vector<float>&& theta_min,
	std::vector<float>&& theta_max,
	Eigen::Vector3f&& base_offset,
	const Eigen::Quaternionf& base_rotation)
	:
	a(std::move(a)),
	theta_min(std::move(theta_min)),
	theta_max(std::move(theta_max)),
	base_offset(std::move(base_offset)),
	base_rotation(base_rotation)
{
}

Eigen::Affine3f finger_kinematic_parameters::transformation_to_base(const std::vector<float>& theta,
	const Eigen::Vector4f& bone_scaling) const
{
	if (theta.size() > a.size())
		throw std::exception("Too many parameters for theta");

	if (bone_scaling.size() < theta.size())
		throw std::exception("Too few parameters for bone_scaling");

	Eigen::Affine3f result = Eigen::Affine3f(Eigen::Translation3f(bone_scaling(0) * base_offset)) *
		Eigen::Affine3f(base_rotation);

	for (int i = 0; i < theta.size(); i++)
	{
		result = result *
			Eigen::AngleAxisf(std::max(theta_min[i], std::min(theta_max[i], theta[i])), Eigen::Vector3f::UnitZ()) *
			Eigen::Translation3f(bone_scaling(i) * a[i], 0.f, 0.f) *
			Eigen::AngleAxisf(alpha[i], Eigen::Vector3f::UnitX());
	}

	return result;
}

/*
 * @see transformation_to_base for a better readable but less efficient version of the computation
 */
Eigen::Vector3f finger_kinematic_parameters::relative_keypoint(const std::vector<float>& theta,
	const Eigen::Vector4f& bone_scaling) const
{
	const auto& translation = bone_scaling(0) * base_offset;

	switch (theta.size())
	{
	case 0:
		return translation;
	case 1:
		return base_rotation * Eigen::Vector3f(a[0] * std::cosf(theta[0]), a[0] * std::sinf(theta[0]), 0.f) + translation;
	case 2:
		return base_rotation * Eigen::Vector3f(
			a[1] * std::cosf(theta[0]) * std::cosf(theta[1]) + a[0] * std::cosf(theta[0]),
			a[1] * std::cosf(theta[1]) * std::sinf(theta[0]) + a[0] * std::sinf(theta[0]),
			a[1] * std::sinf(theta[1])
		) + translation;
	case 3:
		return base_rotation * Eigen::Vector3f(
			a[1] * std::cosf(theta[0]) * std::cosf(theta[1]) + (std::cosf(theta[0]) * std::cosf(theta[1]) * std::cosf(theta[2]) - std::cosf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * a[2] + a[0] * std::cosf(theta[0]),
			a[1] * std::cosf(theta[1]) * std::sinf(theta[0]) + (std::cosf(theta[1]) * std::cosf(theta[2]) * std::sinf(theta[0]) - std::sinf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * a[2] + a[0] * std::sinf(theta[0]),
			(std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[2])) * a[2] + a[1] * std::sinf(theta[1])
		) + translation;
	case 4:
		return base_rotation * Eigen::Vector3f(
			a[1] * std::cosf(theta[0]) * std::cosf(theta[1]) + (std::cosf(theta[0]) * std::cosf(theta[1]) * std::cosf(theta[2]) - std::cosf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * a[2] + ((std::cosf(theta[0]) * std::cosf(theta[1]) * std::cosf(theta[2]) - std::cosf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) - (std::cosf(theta[0]) * std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[0]) * std::cosf(theta[1]) * std::sinf(theta[2])) * std::sinf(theta[3])) * a[3] + a[0] * std::cosf(theta[0]),
			a[1] * std::cosf(theta[1]) * std::sinf(theta[0]) + (std::cosf(theta[1]) * std::cosf(theta[2]) * std::sinf(theta[0]) - std::sinf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * a[2] + ((std::cosf(theta[1]) * std::cosf(theta[2]) * std::sinf(theta[0]) - std::sinf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) - (std::cosf(theta[2]) * std::sinf(theta[0]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[0]) * std::sinf(theta[2])) * std::sinf(theta[3])) * a[3] + a[0] * std::sinf(theta[0]),
			(std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[2])) * a[2] + ((std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) + (std::cosf(theta[1]) * std::cosf(theta[2]) - std::sinf(theta[1]) * std::sinf(theta[2])) * std::sinf(theta[3])) * a[3] + a[1] * std::sinf(theta[1])
		) + translation;
	default:
		return base_rotation * Eigen::Vector3f(
			a[1] * std::cosf(theta[0]) * std::cosf(theta[1]) + (std::cosf(theta[0]) * std::cosf(theta[1]) * std::cosf(theta[2]) - std::cosf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * a[2] + ((std::cosf(theta[0]) * std::cosf(theta[1]) * std::cosf(theta[2]) - std::cosf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) - (std::cosf(theta[0]) * std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[0]) * std::cosf(theta[1]) * std::sinf(theta[2])) * std::sinf(theta[3])) * a[3] + (((std::cosf(theta[0]) * std::cosf(theta[1]) * std::cosf(theta[2]) - std::cosf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) - (std::cosf(theta[0]) * std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[0]) * std::cosf(theta[1]) * std::sinf(theta[2])) * std::sinf(theta[3])) * std::cosf(theta[4]) - ((std::cosf(theta[0]) * std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[0]) * std::cosf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) + (std::cosf(theta[0]) * std::cosf(theta[1]) * std::cosf(theta[2]) - std::cosf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * std::sinf(theta[3])) * std::sinf(theta[4])) * a[4] + a[0] * std::cosf(theta[0]),
			a[1] * std::cosf(theta[1]) * std::sinf(theta[0]) + (std::cosf(theta[1]) * std::cosf(theta[2]) * std::sinf(theta[0]) - std::sinf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * a[2] + ((std::cosf(theta[1]) * std::cosf(theta[2]) * std::sinf(theta[0]) - std::sinf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) - (std::cosf(theta[2]) * std::sinf(theta[0]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[0]) * std::sinf(theta[2])) * std::sinf(theta[3])) * a[3] + (((std::cosf(theta[1]) * std::cosf(theta[2]) * std::sinf(theta[0]) - std::sinf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) - (std::cosf(theta[2]) * std::sinf(theta[0]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[0]) * std::sinf(theta[2])) * std::sinf(theta[3])) * std::cosf(theta[4]) - ((std::cosf(theta[2]) * std::sinf(theta[0]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[0]) * std::sinf(theta[2])) * std::cosf(theta[3]) + (std::cosf(theta[1]) * std::cosf(theta[2]) * std::sinf(theta[0]) - std::sinf(theta[0]) * std::sinf(theta[1]) * std::sinf(theta[2])) * std::sinf(theta[3])) * std::sinf(theta[4])) * a[4] + a[0] * std::sinf(theta[0]),
			(std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[2])) * a[2] + ((std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) + (std::cosf(theta[1]) * std::cosf(theta[2]) - std::sinf(theta[1]) * std::sinf(theta[2])) * std::sinf(theta[3])) * a[3] + (((std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) + (std::cosf(theta[1]) * std::cosf(theta[2]) - std::sinf(theta[1]) * std::sinf(theta[2])) * std::sinf(theta[3])) * std::cosf(theta[4]) + ((std::cosf(theta[1]) * std::cosf(theta[2]) - std::sinf(theta[1]) * std::sinf(theta[2])) * std::cosf(theta[3]) - (std::cosf(theta[2]) * std::sinf(theta[1]) + std::cosf(theta[1]) * std::sinf(theta[2])) * std::sinf(theta[3])) * std::sinf(theta[4])) * a[4] + a[1] * std::sinf(theta[1])
		) + translation;
	}
}


/////////////////////////////////////////////////////////////
//
//
//  Class: hand_kinematic_parameters
//
//
/////////////////////////////////////////////////////////////

hand_kinematic_parameters::hand_kinematic_parameters()
{
	filename_ = std::string("hand_kinematic_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else
	{
		const float middle_finger_length = 0.18825f;
		const float palm_breadth = 0.08335f;

		std::vector<std::vector<double>> bone_lengths = {
			{0.508825, 0.361438,0.31327,0.294646},
			{1.06024,0.443465,0.297464,0.260544},
			{1.01398,0.556376,0.376575,0.312149},
			{0.998409,0.501625,0.310689,0.240564},
			{0.959939,0.320599,0.227937,0.236825}
		};



		float rel_middle_finger_length = 0;
		for (double length : bone_lengths[2])
			rel_middle_finger_length += length;

		auto scale = [&](double l) {return (float)(l / rel_middle_finger_length * middle_finger_length); };
		auto rad = [](int deg) {return static_cast<float>(deg * M_PI / 180); };

		std::vector<float> finger_base_angles = {
			rad(59.38), rad(19), rad(5.4), -rad(5.8), -rad(21.1)
		};

		auto base = [&](int i) {
			return Eigen::Vector3f(
				scale(bone_lengths[i][0]) * std::sin(finger_base_angles[i]),
				scale(bone_lengths[i][0]) * std::cos(finger_base_angles[i]),
				0.f);
		};
		float palm_length = scale(bone_lengths[2][0]);
		auto a_values = [&](int i) {
			return std::vector<float>({ 0.f, scale(bone_lengths[i][1]) , scale(bone_lengths[i][2]), scale(bone_lengths[i][3]) });
		};

		fingers = {
			// thumb
			finger_kinematic_parameters(
				{0.0f, 0.04f, 0.0325f,0.02f}, // a
				{rad(-50), rad(-40), rad(-80) , rad(-80)}, // theta_min
				{rad(20), rad(20), rad(20) , rad(5)}, // theta_max
				Eigen::AngleAxisf(-finger_base_angles[0], Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(-rad(30), Eigen::Vector3f::UnitX()) * Eigen::Vector3f(0.f, 0.035f, 0.f), // base_offset
				Eigen::AngleAxisf(rad(45), Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(rad(90), Eigen::Vector3f::UnitX())
			),

			// index
			finger_kinematic_parameters(
				a_values(1), // a
				{rad(-30), rad(-90) , rad(-110), rad(-90)}, // theta_min
				{rad(30), rad(20) , rad(0), rad(0)}, // theta_max
				base(1) // base_offset
			),

			// middle
			finger_kinematic_parameters(
				a_values(2),
				{ rad(-15), rad(-90) , rad(-110), rad(-90)}, // theta_min
				{ rad(15), rad(20) , rad(0), rad(0)}, // theta_max
				base(2) // base_offset
			),

			// ring
			finger_kinematic_parameters(
				a_values(3),
				{ rad(-15), rad(-90) , rad(-120), rad(-90)}, // theta_min
				{ rad(15), rad(20) , rad(0), rad(0)}, // theta_max
				base(3) // base_offset
			),

			// little
			finger_kinematic_parameters(
				a_values(4),
				{ rad(-10), rad(-90) , rad(-135), rad(-90)}, // theta_min
				{ rad(30), rad(20) , rad(0), rad(5)}, // theta_max
				base(4) // base_offset
			)
		};

		thickness = 0.0329f;
		min_bone_scaling = 0.8616f;
		max_bone_scaling = 1.11766f;
	}
}


hand_kinematic_parameters::~hand_kinematic_parameters()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const hand_kinematic_parameters& hand_kinematic_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(hand_kinematic_params);
}

finger_kinematic_parameters& hand_kinematic_parameters::get_finger(finger_type type)
{
	return fingers.at((int)type);
}

const finger_kinematic_parameters& hand_kinematic_parameters::get_finger(finger_type type) const
{
	return fingers.at((int)type);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: hand_pose_18DoF
//
//
/////////////////////////////////////////////////////////////

const std::vector<float> hand_pose_18DoF::centroid_weights = { 0.23664061,
	0.045347154, 0.035157289, 0.031676931, 0.015353241,
	0.078354179, 0.038607894, 0.029076354, 0.013576274,
	0.08182719, 0.048613664, 0.035887418, 0.016265277,
	0.078163205, 0.042327582, 0.028724368, 0.012535168,
	0.066725524, 0.028582792, 0.024217546, 0.012340338 };
	
inline Eigen::Vector3f hand_pose_18DoF::to_euler_angles(const Eigen::Matrix3f& R)
{
	Eigen::Vector3f angles = R.eulerAngles(0, 1, 2);
	for (int i = 0; i < 3; i++)
	{
		float& val = angles(i);
		if (val > M_PI)
			val = val - 2 * M_PI;
		if (val < -M_PI)
			val = val + 2 * M_PI;
	}

	return angles;
}

inline Eigen::Quaternionf hand_pose_18DoF::to_quaternion(const Eigen::Vector3f& euler_angles)
{
	return Eigen::AngleAxisf(euler_angles(0), Eigen::Vector3f::UnitX()) *
		Eigen::AngleAxisf(euler_angles(1), Eigen::Vector3f::UnitY()) *
		Eigen::AngleAxisf(euler_angles(2), Eigen::Vector3f::UnitZ());
}

inline hand_pose_18DoF hand_pose_18DoF::combine(Eigen::Affine3f wrist_pose,
	hand_pose_18DoF finger_poses)
{
	return hand_pose_18DoF(finger_poses.hand_kinematic_params,
		std::move(wrist_pose),
		std::move(finger_poses.finger_bending),
		finger_poses.thumb_adduction,
		finger_poses.finger_spreading,
		finger_poses.right_hand,
		std::move(finger_poses.bone_scaling));
}

inline float hand_pose_18DoF::rotation_distance(const Eigen::Affine3f& pose_1, const Eigen::Affine3f& pose_2)
{
	Eigen::Matrix3f R = pose_1.rotation() * pose_2.rotation().transpose();
	float t = R.trace();
	t = std::max(-1.f, std::min(1.f, (t - 1.f) / 2.f));
	return std::acos(t);
}

inline float hand_pose_18DoF::rotation_distance(const Eigen::Quaternionf& pose_1, const Eigen::Quaternionf& pose_2)
{
	return 2.f * std::acos(std::abs(pose_1.dot(pose_2)));
}

inline Eigen::Vector3f hand_pose_18DoF::get_centroid(const Eigen::Matrix3Xf& key_points)
{
	Eigen::Vector3f result = Eigen::Vector3f::Zero();

	for(int i = 0; i < 21; i++)
	{
		result.x() += centroid_weights[i] * key_points(0, i);
		result.y() += centroid_weights[i] * key_points(1, i);
		result.z() += centroid_weights[i] * key_points(2, i);
	}

	return result;
}



hand_pose_18DoF::hand_pose_18DoF(const hand_kinematic_parameters& hand_kinematic_params)
	:
	hand_pose_18DoF(hand_kinematic_params,
		Eigen::Affine3f::Identity(),
		std::vector<std::pair<float, float>>(5, std::make_pair(0.f, 0.f)),
		0.f, 0.f, false)
{
}

hand_pose_18DoF::hand_pose_18DoF(const hand_kinematic_parameters& hand_kinematic_params,
	Eigen::Affine3f wrist_pose,
	std::vector<std::pair<float, float>> finger_bending,
	float thumb_adduction,
	float finger_spreading,
	bool right_hand,
	Eigen::Matrix<float, 4, 5> bone_scaling)
	:
	hand_kinematic_params(hand_kinematic_params),
	wrist_pose(std::move(wrist_pose)),
	finger_bending(std::move(finger_bending)),
	thumb_adduction(thumb_adduction),
	finger_spreading(finger_spreading),
	right_hand(!!right_hand), // ensure 0 or 1 bool
	bone_scaling(std::move(bone_scaling))
{
	if (this->finger_bending.size() != 5)
		throw std::exception("Parameter finger_bending has invalid size");

	Vector15f lower = get_lower_bounds();
	Vector15f upper = get_upper_bounds();

	auto cap = [&](float& x, int i)
	{
		x = std::max(lower(i), std::min(upper(i), x));
	};

	cap(this->thumb_adduction, 3);

	for (int i = 0; i < 5; i++)
	{

		cap(this->finger_bending[i].first, 2 * i + 4);
		cap(this->finger_bending[i].second, 2 * i + 5);
	}

	cap(this->finger_spreading, 14);

	for (int col = 0; col < bone_scaling.cols(); col++)
		for (int row = 0; row < bone_scaling.rows(); row++)
			this->bone_scaling(row, col) = std::max(hand_kinematic_params.min_bone_scaling,
				std::min(hand_kinematic_params.max_bone_scaling, this->bone_scaling(row, col)));
}

hand_pose_18DoF::hand_pose_18DoF(const hand_kinematic_parameters& hand_kinematic_params,
	Vector15f params,
	Eigen::Quaternionf rotation,
	bool right_hand,
	Eigen::Matrix<float, 4, 5> bone_scaling)
	:
	hand_kinematic_params(hand_kinematic_params),
	wrist_pose(Eigen::Affine3f::Identity()),
	finger_bending(5, std::make_pair(0.f, 0.f)),
	thumb_adduction(0.f),
	finger_spreading(0.f),
	right_hand(!!right_hand), // ensure 0 or 1 bool
	bone_scaling(std::move(bone_scaling))
{
	Vector15f lower = get_lower_bounds();
	Vector15f upper = get_upper_bounds();

	for (int i = 0; i < params.size(); i++)
		params[i] = std::max(lower[i], std::min(upper[i], params[i]));

	wrist_pose = Eigen::Translation3f(params.head(3)) *
		rotation;

	thumb_adduction = params[3];

	for (int i = 0; i < 5; i++)
		finger_bending[i] = std::make_pair(params[2 * i + 4], params[2 * i + 5]);

	finger_spreading = params[14];

	for (int col = 0; col < bone_scaling.cols(); col++)
		for (int row = 0; row < bone_scaling.rows(); row++)
			this->bone_scaling(row, col) = std::max(hand_kinematic_params.min_bone_scaling,
				std::min(hand_kinematic_params.max_bone_scaling, this->bone_scaling(row, col)));
}

//hand_pose_18DoF::hand_pose_18DoF(hand_pose_18DoF&& o) noexcept
//	:
//	hand_kinematic_params(o.hand_kinematic_params),
//	wrist_pose(std::move(o.wrist_pose)),
//	finger_bending(std::move(o.finger_bending)),
//	thumb_adduction(o.thumb_adduction),
//	finger_spreading(o.finger_spreading),
//	right_hand(o.right_hand),
//	bone_scaling(std::move(o.bone_scaling))
//{
//}

hand_pose_18DoF& hand_pose_18DoF::operator=(const hand_pose_18DoF& o)
{
	wrist_pose = o.wrist_pose;
	finger_bending = o.finger_bending;
	thumb_adduction = o.thumb_adduction;
	finger_spreading = o.finger_spreading;
	right_hand = o.right_hand;
	bone_scaling = o.bone_scaling;

	if (finger_bending.size() != 5)
		throw std::runtime_error("finger_bending.size() != 5");

	return *this;
}

float hand_pose_18DoF::get_max_spreading() const
{
	return std::min(hand_kinematic_params.get_finger(finger_type::INDEX).theta_max[0], -hand_kinematic_params.get_finger(finger_type::LITTLE).theta_min[0]);
}

float hand_pose_18DoF::get_parameter(unsigned int index) const
{
	if (index > 14)
		throw std::invalid_argument("hand_pose_18DoF::get_parameter out of range");

	if (index < 3)
		return wrist_pose.translation()(index);
	if (index == 3)
		return thumb_adduction;
	if (index == 14)
		return finger_spreading;

	auto& finger = finger_bending[(index - 4) / 2];
	return index % 2 ? finger.second : finger.first;
}

hand_pose_18DoF::Vector15f hand_pose_18DoF::get_parameters() const
{
	Vector15f result;

	result.head(3) = wrist_pose.translation();
	for (int i = 3; i < 15; i++)
		result(i) = get_parameter(i);

	return result;
}

hand_pose_18DoF::Vector15f hand_pose_18DoF::get_lower_bounds() const
{
	hand_pose_18DoF::Vector15f result;

	result.head(3) = -std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones();

	result(3) = hand_kinematic_params.get_finger(finger_type::THUMB).theta_min[0];

	int i = 4;
	for (const auto& finger : hand_kinematic_params.fingers)
	{
		result(i++) = finger.theta_min[1];
		result(i++) = finger.theta_min[2] + finger.theta_min[3];
	}

	result(14) = -M_PI_4 / 10;

	return result;
}

hand_pose_18DoF::Vector15f hand_pose_18DoF::get_upper_bounds() const
{
	hand_pose_18DoF::Vector15f result;

	result.head(3) = std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones();

	result(3) = hand_kinematic_params.get_finger(finger_type::THUMB).theta_max[0];

	int i = 4;
	for (const auto& finger : hand_kinematic_params.fingers)
	{
		result(i++) = finger.theta_max[1];
		result(i++) = finger.theta_max[2] + finger.theta_max[3];
	}

	result(14) = get_max_spreading();

	return result;
}

std::pair<float, float> hand_pose_18DoF::get_interphalangeal_angles(finger_type finger) const
{
	float dip = 0.4f * finger_bending[(int)finger].second;
	float pip = 0.6f * finger_bending[(int)finger].second;

	const auto& finger_kin = hand_kinematic_params.get_finger(finger);

	if (pip < finger_kin.theta_min[2])
	{
		dip += pip - finger_kin.theta_min[2];
		pip = finger_kin.theta_min[2];
	}
	else if (pip > finger_kin.theta_max[2])
	{
		dip += pip - finger_kin.theta_max[2];
		pip = finger_kin.theta_max[2];
	}


	return std::make_pair(pip, dip);
}

std::vector<float> hand_pose_18DoF::get_theta_angles(hand_pose_estimation::finger_type finger,
	hand_pose_estimation::finger_joint_type joint) const
{
	using j_t = finger_joint_type;

	if (joint == j_t::MCP)
		return std::vector<float>();

	std::vector<float> result;
	result.reserve((int)joint);

	result.push_back(thumb_adduction);
	if (finger != finger_type::THUMB)
	{
		result.back() = -finger_spreading * (1 - 2 / 3.f * ((int)finger - 1));
	}

	result.push_back(finger_bending[(int)finger].first);

	if (joint == j_t::PIP)
		return result;

	std::pair pip_dip = get_interphalangeal_angles(finger);
	result.push_back(pip_dip.first);

	if (joint == j_t::DIP)
		return result;

	result.push_back(pip_dip.second);

	return result;
}

Eigen::Matrix<float, 3, 4> hand_pose_18DoF::get_key_points(finger_type finger) const
{
	using f_t = finger_type;
	using j_t = finger_joint_type;

	Eigen::Matrix<float, 3, 4> result;


	for (int joint = 0; joint < 4; joint++)
	{
		result.col(joint) = hand_kinematic_params.get_finger(finger).relative_keypoint(
			get_theta_angles(finger, (j_t)joint),
			bone_scaling.col((int)finger)
		);

	}

	Eigen::Affine3f transform = wrist_pose;

	if (right_hand)
		transform = transform * Eigen::Scaling(-1.f, 1.f, 1.f);

	return transform * result;
}

Eigen::Matrix3Xf hand_pose_18DoF::get_key_points() const
{
	using f_t = finger_type;
	using j_t = finger_joint_type;

	Eigen::Matrix3Xf result(3, 21);
	result.col(0) = Eigen::Vector3f::Zero();



	for (int f = 0; f < 5; f++)
	{
		const auto& finger = hand_kinematic_params.get_finger((f_t)f);
		for (int joint = 0; joint < 4; joint++)
		{
			result.col(4 * f + joint + 1) = finger.relative_keypoint(
				get_theta_angles((f_t)f, (j_t)joint),
				bone_scaling.col(f)
			);
		}
	}

	Eigen::Affine3f transform = wrist_pose;

	if (right_hand)
		transform = transform * Eigen::Scaling(-1.f, 1.f, 1.f);

	return transform * result;
}

Eigen::Vector3f hand_pose_18DoF::get_tip(finger_type finger) const
{
	if (right_hand)
		return wrist_pose * Eigen::Scaling(-1.f, 1.f, 1.f) *
		hand_kinematic_params.get_finger(finger)
		.relative_keypoint(
			get_theta_angles(finger),
			bone_scaling.col((int)finger)
		);
	else
		return wrist_pose *
		hand_kinematic_params.get_finger(finger)
		.relative_keypoint(
			get_theta_angles(finger),
			bone_scaling.col((int)finger)
		);
}

Eigen::Matrix<float, 3, 5> hand_pose_18DoF::get_tips() const
{

	Eigen::Matrix<float, 3, 5> result;

	for (int f = 0; f < 5; f++)
	{
		result.col(f) = get_tip((finger_type)f);
	}

	return result;
}

float hand_pose_18DoF::rotational_distance(const hand_pose_18DoF& other) const
{
	return rotation_distance(wrist_pose, other.wrist_pose);
}

Eigen::Vector3f hand_pose_18DoF::get_centroid() const
{
	return get_centroid(get_key_points());
}




/////////////////////////////////////////////////////////////
//
//
//  Class: hand_dynamic_parameters
//
//
/////////////////////////////////////////////////////////////

hand_dynamic_parameters::hand_dynamic_parameters()
{
	filename_ = std::string("hand_dynamic_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else
	{
		speed_extension_mcp = M_PI;
		speed_extension_pip = M_PI;
		speed_extension_dip = M_PI;

		speed_wrist = 2.f;

		speed_rotation = M_PI;

		speed_adduction_thumb = M_PI;
		speed_adduction = M_PI_2;
	}
}


hand_dynamic_parameters::~hand_dynamic_parameters()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const hand_dynamic_parameters& hand_dynamic_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(hand_dynamic_params);
}

hand_pose_18DoF::Vector15f hand_dynamic_parameters::get_constraints_18DoF() const
{
	hand_pose_18DoF::Vector15f result;

	result(0) = speed_wrist;
	result(1) = speed_wrist;
	result(2) = speed_wrist;
	result(3) = speed_adduction_thumb;

	int r = 4;
	for (int i = 0; i < 5; i++)
	{
		result(r++) = speed_extension_mcp;
		result(r++) = speed_extension_pip + 2 / 3.f * speed_extension_dip;
	}

	result(14) = speed_adduction;

	return result;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: net_evaluation
//
//
/////////////////////////////////////////////////////////////

net_evaluation::net_evaluation(cv::Rect2i input_box,
	heatmaps maps,
	bool right_hand,
	std::vector<cv::Point2i> key_points_2d,
	Eigen::VectorXf certainties,
	Eigen::Matrix3Xf left_hand_pose)
	:
	input_box(std::move(input_box)),
	maps(std::move(maps)),
	key_points_2d(key_points_2d),
	certainties(std::move(certainties)),
	certainty(certainties.mean()),
	right_hand(right_hand),
	left_hand_pose(std::move(left_hand_pose))
{}

net_evaluation::net_evaluation(cv::Rect2i input_box,
	heatmaps maps,
	bool right_hand,
	Eigen::Matrix3Xf left_hand_pose)
	:
	input_box(std::move(input_box)),
	maps(std::move(maps)),
	key_points_2d(key_points_2d),
	certainty(std::numeric_limits<float>::quiet_NaN()),
	right_hand(right_hand),
	left_hand_pose(std::move(left_hand_pose))
{
}




/////////////////////////////////////////////////////////////
//
//
//  Class: hand_pose_particle_instance
//
//
/////////////////////////////////////////////////////////////


hand_pose_particle_instance::hand_pose_particle_instance(hand_pose_18DoF pose,
	net_evaluation::ConstPtr net_eval,
	std::chrono::duration<float> time_seconds,
	Eigen::Matrix3Xf key_points,
	std::vector<float> surface_distances)
	:
	pose(std::move(pose)),
	rotation(this->pose.wrist_pose.rotation()),
	net_eval(std::move(net_eval)),
	key_points(key_points.size() ? std::move(key_points) : this->pose.get_key_points()),
	surface_distances(std::move(surface_distances)),
	time_seconds(time_seconds),
	error(std::numeric_limits<double>::infinity()),
	hand_certainty(0.f),
	hand_orientation(0.f)
{
}

Eigen::AlignedBox3f hand_pose_particle_instance::get_box() const
{
	const float radius= 0.5f * pose.hand_kinematic_params.thickness;

	Eigen::Vector3f min = std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones();
	Eigen::Vector3f max = -std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones();

	for (int i = 0; i < key_points.cols(); i++)
	{
		for(int d = 0; d < 3; d++)
		{
			min(d) = std::min(min(d), key_points.col(i)(d) - radius);
			max(d) = std::max(max(d), key_points.col(i)(d) + radius);
		}
	}

	return Eigen::AlignedBox3f(min, max);
}
	
cv::Rect2i hand_pose_particle_instance::get_box(const visual_input& input) const
{
	Eigen::Vector4f wrist = key_points.col(0).homogeneous();
	Eigen::Vector2f wrist_proj = (input.img_projection * wrist).hnormalized();
	wrist(0) += 0.5f * pose.hand_kinematic_params.thickness;
	const float radius = (wrist_proj - (input.img_projection * wrist).hnormalized()).norm();

	int min_x = input.img.cols - 1;
	int min_y = input.img.rows - 1;
	int max_x = 0;
	int max_y = 0;

	for (int i = 0; i < key_points.cols(); i++)
	{
		cv::Point2i p_abs = input.to_img_coordinates(key_points.col(i));

		min_x = std::min(min_x, (int)std::floor(p_abs.x - radius));
		min_y = std::min(min_y, (int)std::floor(p_abs.y - radius));
		max_x = std::max(max_x, (int)std::ceil(p_abs.x + radius));
		max_y = std::max(max_y, (int)std::ceil(p_abs.y + radius));
	}

	min_x = std::max(0, min_x);
	min_y = std::max(0, min_y);
	max_x = std::min(input.img.cols - 1, max_x);
	max_y = std::min(input.img.rows - 1, max_y);

	return cv::Rect2i(cv::Point2i(min_x, min_y), cv::Point2i(max_x, max_y));
}


/////////////////////////////////////////////////////////////
//
//
//  Class: hand_pose_particle
//
//
/////////////////////////////////////////////////////////////

hand_pose_particle::hand_pose_particle(const hand_pose_particle_instance::Ptr& current,
	const hand_pose_particle_instance::Ptr& prev_frame)
	:
	current(current),
	best(current),
	prev_frame(prev_frame)
{
}


void hand_pose_particle::add(const hand_pose_particle_instance::Ptr& next)
{
	current = next;

	if (best->error > next->error)
		best = next;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: img_segment_3d
//
//
/////////////////////////////////////////////////////////////
///
class z_buffer_comparator
{
public:
	using pair = std::pair<float, int>;

	bool operator()(const pair& lhs, const pair& rhs) const
	{
		return lhs.first > rhs.first;
	}
};

img_segment_3d::img_segment_3d(const visual_input& input, const img_segment& seg, const Eigen::Hyperplane<float, 3>& background)
	:
	extra_image(input.extra_image),
	sensor_origin(input.sensor_origin)
{
	auto img_box = input.to_cloud_pixel_coordinates(seg.bounding_box);
	Eigen::Vector3f sum = Eigen::Vector3f::Zero();


	// compute roi cloud
	if (input.extra_image)
	{
		int z_buffer_width = img_box.width;
		int z_buffer_height = img_box.height;

		if(z_buffer_width * z_buffer_height < 50)
			throw std::exception("No points in segment");

		std::vector<std::priority_queue<z_buffer_comparator::pair, std::vector<z_buffer_comparator::pair>, z_buffer_comparator>> z_buffer;
		z_buffer.resize(z_buffer_width * z_buffer_height);

		img_box.x = std::max(0, static_cast<int>(img_box.x - 0.05f * input.cloud->width));
		img_box.y = std::max(0, static_cast<int>(img_box.y - 0.05f * input.cloud->height));
		img_box.width += 0.1f * input.cloud->width;
		img_box.height += 0.1f * input.cloud->height;

		if (img_box.width + img_box.x > input.cloud->width)
			img_box.width = input.cloud->width - img_box.x;
		if (img_box.height + img_box.y > input.cloud->height)
			img_box.height = input.cloud->height - img_box.y;


		auto index_img = pcl::make_shared< pcl::PointCloud<pcl::PointXYZL>>();
		auto index_cloud = pcl::make_shared< pcl::PointCloud<pcl::PointXYZL>>();

		
		// skin cloud
		for (int y = img_box.y; y < img_box.y + img_box.height; y++)
			for (int x = img_box.x; x < img_box.x + img_box.width; x++)
			{
				if (!input.is_valid_point(x, y))
					continue;

				const visual_input::PointT& p = input.cloud->at(x, y);

				cv::Point2i p_img = input.to_img_coordinates(p);
				if (!seg.bounding_box.contains(p_img) || seg.mask.at<uchar>(p_img - seg.bounding_box.tl()) == 0)
					continue;

			//	if (background.signedDistance(p.getVector3fMap()) < 0.f)
			//		continue;

				float buffer_x = std::floorf((p_img.x - seg.bounding_box.x) * ((float)z_buffer_width) / seg.bounding_box.width);
				float buffer_y = std::floorf((p_img.y - seg.bounding_box.y) * ((float)z_buffer_height) / seg.bounding_box.height);

				z_buffer[buffer_y * z_buffer_width + buffer_x].emplace(input.cloud->at(x, y).z, y * input.cloud->width + x);
			}




		auto tmp_cloud = pcl::make_shared<pcl::PointCloud<visual_input::PointT>>();
		for (int buffer_y = 0; buffer_y < z_buffer_height; buffer_y++)
			for (int buffer_x = 0; buffer_x < z_buffer_width; buffer_x++)

			{
				auto& queue = z_buffer[buffer_y * z_buffer_width + buffer_x];
				if (queue.empty())
					continue;

				auto top = queue.top();
				while (!queue.empty() && std::abs(queue.top().first - top.first) < 0.02f)
				{

					pcl::PointXYZL index_p;

					auto closest = queue.top(); queue.pop();
					auto& p = input.cloud->at(closest.second);
					index_p.label = closest.second;
					tmp_cloud->push_back(p);

					// image pixel coordinates
					cv::Point2i pixel = input.to_img_coordinates(p);
					index_p.x = pixel.x;
					index_p.y = pixel.y;

					index_img->push_back(index_p);

					// cloud pixel coordinates
					int x = closest.second % input.cloud->width;
					int y = closest.second / input.cloud->width;

					index_p.x = x;
					index_p.y = y;

					index_cloud->push_back(index_p);



					sum += tmp_cloud->back().getVector3fMap();
				}
			}


		if (tmp_cloud->empty())
			throw std::exception("No points in segment");

		centroid_and_box_without_outliers(*tmp_cloud, centroid = sum / tmp_cloud->size());

		auto cloud_filtered = pcl::make_shared<pcl::PointCloud<visual_input::PointT>>();
		auto index_cloud_filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZL>>();
		auto index_img_filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZL>>();


		cloud_filtered->reserve(tmp_cloud->size());
		index_cloud_filtered->reserve(tmp_cloud->size());
		index_img_filtered->reserve(tmp_cloud->size());

		// Remove z outliers
		auto p0 = tmp_cloud->begin();
		auto p1 = index_cloud->begin();
		auto p2 = index_img->begin();
		for (; p0 != tmp_cloud->end() && p1 != index_cloud->end() && p2 != index_img->end(); ++p0, ++p1, ++p2)
			if (std::abs(centroid.z() - p0->z) < 0.15f)
			{
				cloud_filtered->push_back(*p0);
				index_cloud_filtered->push_back(*p1);
				index_img_filtered->push_back(*p2);
			}

		if (cloud_filtered->size() < tmp_cloud->size() / 2)
		{
			cloud_filtered->clear();
			index_cloud_filtered->clear();
			index_img_filtered->clear();

			auto p0 = tmp_cloud->begin();
			auto p1 = index_cloud->begin();
			auto p2 = index_img->begin();

			sum = Eigen::Vector3f::Zero();

			for (; p0 != tmp_cloud->end() && p1 != index_cloud->end() && p2 != index_img->end(); ++p0, ++p1, ++p2)
				if (std::abs(centroid.z() - p0->z) < 0.15f || p0->z < centroid.z())
				{
					cloud_filtered->push_back(*p0);
					index_cloud_filtered->push_back(*p1);
					index_img_filtered->push_back(*p2);

					sum += p0->getVector3fMap();
				}

			centroid_and_box_without_outliers(*tmp_cloud, centroid = sum / tmp_cloud->size());
		}

		cloud = cloud_filtered;
		cloud_box = bounding_box_2d(input, *tmp_cloud);

		if (cloud->empty())
			throw std::exception("No points in segment");
		
		compute_nearest_neighbor_lookup(index_cloud_filtered, cloud_box, nn_lookup_cloud, nn_lookup_box_cloud);

		img_quadtree.setPointRepresentation(pcl::make_shared<XYZL2XY>());
		img_quadtree.setEpsilon(0.25f);
		img_quadtree.setSortedResults(false);
		img_quadtree.setInputCloud(index_img_filtered);
	}
	else // !input.extra_image
	{
		auto tmp_cloud = pcl::make_shared<pcl::PointCloud<visual_input::PointT>>();
		auto index_cloud = pcl::make_shared< pcl::PointCloud<pcl::PointXYZL>>();

		// skin cloud
		Eigen::Vector3f sum = Eigen::Vector3f::Zero();
		for (int y = img_box.y; y < img_box.y + img_box.height; y++)
			for (int x = img_box.x; x < img_box.x + img_box.width; x++)
			{
				if (!input.is_valid_point(x, y))
					continue;

				const visual_input::PointT& p = input.cloud->at(x, y);

				cv::Point2i p_img(x, y);
				if (!seg.bounding_box.contains(p_img) || seg.mask.at<uchar>(p_img - seg.bounding_box.tl()) == 0)
					continue;

				pcl::PointXYZL index_p;
				index_p.label = y * input.cloud->width + x;
				index_p.x = x;
				index_p.y = y;

				index_cloud->push_back(index_p);

				tmp_cloud->push_back(p);
				sum += p.getVector3fMap();
			}

		if (tmp_cloud->empty())
			throw std::exception("No points in segment");

		centroid_and_box_without_outliers(*tmp_cloud, centroid = sum / tmp_cloud->size());

		auto cloud_filtered = pcl::make_shared<pcl::PointCloud<visual_input::PointT>>();
		auto index_cloud_filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZL>>();


		cloud_filtered->reserve(tmp_cloud->size());
		index_cloud_filtered->reserve(tmp_cloud->size());

		// Remove z outliers
		auto p0 = tmp_cloud->begin();
		auto p1 = index_cloud->begin();
		for (; p0 != tmp_cloud->end() && p1 != index_cloud->end(); ++p0, ++p1)
			if (std::abs(centroid.z() - p0->z) < 0.15f)
			{
				cloud_filtered->push_back(*p0);
				index_cloud_filtered->push_back(*p1);
			}

		if (cloud_filtered->size() < tmp_cloud->size() / 2)
		{
			cloud_filtered->clear();
			index_cloud_filtered->clear();

			auto p0 = tmp_cloud->begin();
			auto p1 = index_cloud->begin();

			sum = Eigen::Vector3f::Zero();

			for (; p0 != tmp_cloud->end() && p1 != index_cloud->end(); ++p0, ++p1)
				if (std::abs(centroid.z() - p0->z) < 0.15f || p0->z < centroid.z())
				{
					cloud_filtered->push_back(*p0);
					index_cloud_filtered->push_back(*p1);

					sum += p0->getVector3fMap();
				}

			centroid_and_box_without_outliers(*tmp_cloud, centroid = sum / tmp_cloud->size());
		}

		cloud = cloud_filtered;
		cloud_box = bounding_box_2d(input, *tmp_cloud);

		centroid_and_box_without_outliers(*tmp_cloud, centroid = sum / tmp_cloud->size());

		compute_nearest_neighbor_lookup(index_cloud_filtered, cloud_box, nn_lookup_cloud, nn_lookup_box_cloud);
	}

	init_normal_estimation();
}

img_segment_3d::img_segment_3d(const visual_input& input, const img_segment& seg, pcl::PointCloud<visual_input::PointT>::ConstPtr cloud)
	:
	extra_image(input.extra_image),
	cloud_box(bounding_box_2d(input, *cloud)),
	cloud(std::move(cloud)),
	sensor_origin(input.sensor_origin)
{
	init_normal_estimation();

	Eigen::Vector3f sum = Eigen::Vector3f::Zero();
	for (const auto& p : *this->cloud)
		sum += p.getVector3fMap();

	centroid_and_box_without_outliers(*this->cloud, centroid = sum / this->cloud->size());
}

void img_segment_3d::get_surface_point(const visual_input& input,
	const Eigen::Vector3f& reference,
	visual_input::PointT& point,
	pcl::Normal* normal,
	int* index) const
{
	get_surface_point_cloud(input.to_cloud_pixel_coordinates_uncapped(reference), point, normal, index);
}



void img_segment_3d::get_surface_point_img(const cv::Point2i& reference,
	visual_input::PointT& point, pcl::Normal* normal, int* index) const
{
	if (!extra_image)
		return get_surface_point_cloud(reference, point, normal, index);

	std::vector<int> closest_points(1);
	closest_points[0] = 0;
	std::vector<float> distances(1);

	pcl::PointXYZL p;
	p.x = reference.x;
	p.y = reference.y;
	img_quadtree.nearestKSearch(p, 1, closest_points, distances);

	int idx = closest_points[0];

	point = cloud->at(idx);
	if (normal)
		get_normal(idx, *normal);


	if (index)
		*index = idx;
}


void img_segment_3d::get_surface_point_cloud(const cv::Point2i& reference,
	visual_input::PointT& point,
	pcl::Normal* normal,
	int* index) const
{
	int x = std::max(0, reference.x - nn_lookup_box_cloud.x);
	int y = std::max(0, reference.y - nn_lookup_box_cloud.y);

	if (x >= nn_lookup_box_cloud.width)
		x = nn_lookup_box_cloud.width - 1;

	if (y >= nn_lookup_box_cloud.height)
		y = nn_lookup_box_cloud.height - 1;

	int idx = get_nearest_neighbor_cloud(x, y);

	point = cloud->at(idx);
	if (normal)
		get_normal(idx, *normal);

	if (index)
		*index = idx;
}

void img_segment_3d::compute_nearest_neighbor_lookup(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr& index_cloud,
	const cv::Rect2i& box,
	cv::Mat1i& lookup,
	cv::Rect2i& nn_box)
{
	nn_box = box;
	nn_box.x = std::max(0, static_cast<int>(box.x - 0.05f * box.width));
	nn_box.y = std::max(0, static_cast<int>(box.y - 0.05f * box.height));
	nn_box.width *= 1.1f;
	nn_box.height *= 1.1f;

	lookup = cv::Mat(nn_box.height, nn_box.width, CV_32SC1, cv::Scalar(-1));

	quadtree.setPointRepresentation(pcl::make_shared<XYZL2XY>());
	quadtree.setEpsilon(0.25f);
	quadtree.setSortedResults(false);
	quadtree.setInputCloud(index_cloud);

	//for (int y = 0; y < nn_box.height; y++)
	//	for (int x = 0; x < nn_box.width; x++)

	//	{
	//		lookup.at<int>(y,x) = get_nearest_neighbor_cloud(x,y);
	//	}
}

int img_segment_3d::get_nearest_neighbor_cloud(int x, int y) const
{
	int val = nn_lookup_cloud.at<int>(y, x);
	if (val != -1)
		return val;

	pcl::PointXYZL p;
	p.x = x + nn_lookup_box_cloud.x;
	p.y = y + nn_lookup_box_cloud.y;

	std::vector<int> closest_points(1);
	closest_points[0] = 0;
	std::vector<float> distances(1);

	quadtree.nearestKSearch(p, 1, closest_points, distances);

	val = static_cast<int>(closest_points[0]);
	nn_lookup_cloud.at<int>(y, x) = val;

	return val;
}

void img_segment_3d::init_normal_estimation()
{
	// octree
	auto octree = pcl::make_shared< pcl::search::KdTree<visual_input::PointT>>();
	octree->setPointRepresentation(pcl::make_shared<XYZRGBA2XYZ>());
	octree->setEpsilon(0.005f);
	octree->setSortedResults(false);
	octree->setInputCloud(cloud);

	pcl::Normal n;
	n.normal_x = n.normal_y = n.normal_z = std::numeric_limits<float>::quiet_NaN();

	// normals
	normals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
	normals->points.resize(cloud->size(), n);
	normal_estimator.setInputCloud(cloud);
	normal_estimator.setSearchMethod(octree);
	normal_estimator.setRadiusSearch(0.01);
	normal_estimator.setKSearch(1);
	//normal_estimator.compute(*normals); // lazy computed in get_normal
}

void img_segment_3d::get_normal(int idx, pcl::Normal& n) const
{
	if (std::isfinite(normals->at(idx).normal_x))
	{
		n = normals->at(idx);
		return;
	}

	const auto& cloud = normal_estimator.getInputCloud();
	std::vector<int> indices(1);
	std::vector<float> distances(1);
	const auto& p = cloud->at(idx);

	if (normal_estimator.getSearchMethod()->radiusSearch(p, normal_estimator.getRadiusSearch(), indices, distances) == 0 ||
		!normal_estimator.computePointNormal(*normal_estimator.getInputCloud(), indices, n.normal_x, n.normal_y, n.normal_z, n.curvature))
	{
		n.normal_x = n.normal_y = n.normal_z = n.curvature = 0.f;
	}
	else {
		const auto& vp = sensor_origin;
		float cos_theta = ((vp.x() - p.x) * n.normal_x + (vp.y() - p.y) * n.normal_y + (vp.z() - p.z) * n.normal_z);

		// Flip the plane normal
		if (cos_theta < 0)
		{
			n.normal_x *= -1;
			n.normal_y *= -1;
			n.normal_z *= -1;
		}
	}

	normals->at(idx) = n;
}

void img_segment_3d::centroid_and_box_without_outliers(const pcl::PointCloud<visual_input::PointT>& cloud,
	Eigen::Vector3f& initial_centroid)
{
	// filter outliers (10% outermost)
	using pair = std::pair<const visual_input::PointT*, float>;
	std::vector<pair> pairs;
	std::priority_queue< pair, std::vector<pair>, auto(*)(const pair&, const pair&)->bool >  queue([](const pair& lhs, const pair& rhs) -> bool {
		return lhs.second > rhs.second;
		});

	for (const visual_input::PointT& p : cloud)
	{
		pairs.emplace_back(std::make_pair(&p, (initial_centroid - p.getVector3fMap()).norm()));
	}


	std::sort(pairs.begin(), pairs.end(), [](const pair& lhs, const pair& rhs) -> bool {
		return lhs.second < rhs.second;
		});

	Eigen::Vector3f sum = Eigen::Vector3f::Zero();
	Eigen::Vector3f min = std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones();
	Eigen::Vector3f max = -std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones();
	for (int i = 0; i < 0.9 * cloud.size(); i++)
	{
		const Eigen::Vector3f& vec = pairs[i].first->getVector3fMap();
		for (int i = 0; i < 3; i++)
		{
			min(i) = std::min(min(i), vec(i));
			max(i) = std::max(max(i), vec(i));
		}

		sum += vec;
	}


	bounding_box = Eigen::AlignedBox3f(min, max);
	centroid = sum / (0.9 * cloud.size());
}

cv::Rect2i img_segment_3d::bounding_box_2d(const visual_input& input, const pcl::PointCloud<visual_input::PointT>& cloud)
{
	int cloud_min_x = input.cloud->width;
	int cloud_min_y = input.cloud->height;
	int cloud_max_x = 0;
	int cloud_max_y = 0;

	for (const auto& p : cloud)
	{
		// bounding box cloud pixel coordinates
		cv::Point2i pixel = input.to_cloud_pixel_coordinates(p.getVector3fMap());
		const int x = pixel.x;
		const int y = pixel.y;

		cloud_min_x = std::min(cloud_min_x, x);
		cloud_min_y = std::min(cloud_min_y, y);
		cloud_max_x = std::max(cloud_max_x, x);
		cloud_max_y = std::max(cloud_max_y, y);
	}

	return cv::Rect2i(cloud_min_x, cloud_min_y, cloud_max_x - cloud_min_x + 1, cloud_max_y - cloud_min_y + 1);
}






/////////////////////////////////////////////////////////////
//
//
//  Class: hand_instance
//
//
/////////////////////////////////////////////////////////////


void img_segment::compute_properties_3d(const visual_input& input, const Eigen::Hyperplane<float, 3>& background)
{
	if (!prop_3d)
		prop_3d = std::make_shared < img_segment_3d >(input, *this, background);
}

void img_segment::compute_properties_3d(const visual_input& input,
	const pcl::PointCloud<visual_input::PointT>::ConstPtr& cloud)
{
	if (!prop_3d)
		prop_3d = std::make_shared < img_segment_3d >(input, *this, cloud);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: hand_instance
//
//
/////////////////////////////////////////////////////////////

hand_instance::hand_instance(const img_segment::Ptr& observation,
	float certainty_score)
	:
	certainty_score(certainty_score)
{
	observation_history.push_back(observation);
}

size_t hand_instance::get_id() const
{
	return (size_t)this;
}

img_segment::Ptr hand_instance::get_segment(std::chrono::duration<float> time_seconds) const
{
	std::lock_guard<std::mutex> lock(update_mutex);
	for (auto iter = observation_history.rbegin(); iter != observation_history.rend(); ++iter)
		if ((*iter)->timestamp == time_seconds)
		{
			return *iter;
		}
	return nullptr;
}

void hand_instance::add_or_update(const hand_pose_particle_instance& pose, bool ignore_when_old)
{
	std::lock_guard<std::mutex> lock(update_mutex);
	if (!poses.size() || poses.back().time_seconds < pose.time_seconds)
		poses.push_back(pose);
	else if (poses.back().time_seconds == pose.time_seconds)
	{
		auto& last = poses.back();

		if (last.error > pose.error || !last.net_eval && pose.net_eval)
			last = pose;
		//else
		//	throw std::exception("Pose with same timestamp already in list but update would increases error");
	}
	else if (!ignore_when_old)
		throw std::exception("Tried to add an old segment to the list");
}


/////////////////////////////////////////////////////////////
//
//
//  Class: gesture_prototype
//
//
/////////////////////////////////////////////////////////////

gesture_prototype::gesture_prototype(const std::string& name,
	int fingers_count,
	const std::vector<std::vector<cv::Point>>& templates)
	:
	name(name),
	fingers_count(fingers_count),
	templates(templates)
{
}

gesture_prototype::gesture_prototype(std::string&& name,
	int fingers_count,
	std::vector<std::vector<cv::Point>>&& templates)
	:
	name(name),
	fingers_count(fingers_count),
	templates(templates)
{
}

const std::string& gesture_prototype::get_name() const
{
	return name;
}

int gesture_prototype::get_fingers_count() const
{
	return fingers_count;
}

const std::vector<std::vector<cv::Point>>& gesture_prototype::get_templates() const
{
	return templates;
}

const std::vector<std::vector<double>>& gesture_prototype::get_moments() const
{
	if (moments.size() != templates.size())
	{
		moments.resize(templates.size());
		int i = 0;
		for (const std::vector<cv::Point>& contour : get_templates())
		{
			cv::HuMoments(cv::moments(contour), moments[i++]);
		}
	}

	return moments;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: classification_result
//
//
/////////////////////////////////////////////////////////////

classification_result::classification_result(const gesture_prototype::ConstPtr& prototype,
	float certainty_score)
	:
	prototype(prototype),
	certainty_score(certainty_score)
{
}











} /* hand_pose_estimation */

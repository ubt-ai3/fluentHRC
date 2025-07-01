#include "gradient_decent.hpp"

#include <fstream>
#include <queue>
#include <random>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <opencv2/imgproc.hpp>

#include "hand_pose_estimation.h"
#include <boost/mpl/count_fwd.hpp>

namespace hand_pose_estimation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: gradient_decent_parameters
//
//
/////////////////////////////////////////////////////////////

gradient_decent_parameters::gradient_decent_parameters()
{
	filename_ = std::string("gradient_decent_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else
	{
		min_improvement = 0.0001;
		max_steps = 30;
		learning_rate = 0.1;
		best_net_eval_certainty_decay = 0.005f;
		rotational_tolerance_flateness = M_PI_4;
		rotational_tolerance_movement = 0.1f;
		translational_tolerance_movement = 0.01f;
		position_similarity_threshold = 0.1f;
	}
}

gradient_decent_parameters::~gradient_decent_parameters()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const gradient_decent_parameters& gradient_decent_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(gradient_decent_params);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: quality_criterion
//
//
/////////////////////////////////////////////////////////////

quality_criterion::quality_criterion(double weight)
	:
	max_weight(weight)
{
}

const float quality_criterion::EPSILON = 0.0001f;

/////////////////////////////////////////////////////////////
//
//
//  Class: quality_2d_key_points
//
//
/////////////////////////////////////////////////////////////


quality_criterion* quality_2d_key_points::clone() const
{
	return new quality_2d_key_points(max_weight);
}

std::pair<double, double> quality_2d_key_points::evaluate(const visual_input& input,
                                                          const hand_pose_particle_instance& particle,
                                                          const img_segment& seg) const
{
	const Eigen::Matrix3Xf& key_points = particle.key_points;
	if (!particle.net_eval || particle.net_eval->certainty == 0.f)
		return std::make_pair(std::numeric_limits<float>::quiet_NaN(), 0);
	const auto& net_eval = *particle.net_eval;

	const net_evaluation::heatmaps& maps = net_eval.maps;
	const auto& box = net_eval.input_box;

	double sum_prob = 0.;

	float distance_scaling = 18.f / box.width;

	std::vector<float> errors;
	std::vector<float> certainties;
	Eigen::Vector2f origin(box.x, box.y);
	Eigen::Vector2f scaling(maps.at(0).cols / (float)box.width,
		maps.at(0).rows / (float)box.height);

	for (int i = 0; i < key_points.cols(); i++)
	{
		if (net_eval.key_points_2d[i].x == -1 && net_eval.key_points_2d[i].y == -1)
			continue;

		Eigen::Vector2f pixel = input.to_img_coordinates_vec(key_points.col(i));

		Eigen::Vector2f p = (pixel - origin).cwiseProduct(scaling);
		float prob = -0.5f;

		// extract interpolated value and handle out-of-image values
		if (p.x() > 0 && p.y() > 0 && p.x() < maps.at(i).cols - 1 && p.y() < maps.at(i).rows - 1)
		{
			float tl = maps.at(i).at<float>(std::floor(p.y()), std::floor(p.x()));
			float bl = maps.at(i).at<float>(std::ceil(p.y()), std::floor(p.x()));
			float tr = maps.at(i).at<float>(std::floor(p.y()), std::ceil(p.x()));
			float br = maps.at(i).at<float>(std::ceil(p.y()), std::ceil(p.x()));

			float integral;
			float dx = std::modf(p.x(), &integral);
			float t = tl * (1.f - dx) + tr * dx;
			float b = bl * (1.f - dx) + br * dx;

			float dy = std::modf(p.y(), &integral);
			prob = hand_pose_estimation::correct_heatmap_certainty(t * (1.f - dy) + b * dy);
		}

		if (prob > 0.1f)
		{
			errors.push_back(std::max(0.f, prob > 0.4f ? 0.1f / 0.6f * (1 - prob) : 2.1f - 5.f * prob));
			sum_prob += errors.back();
		}
		else
		{
			errors.push_back(distance_scaling * (pixel - Eigen::Vector2f(net_eval.key_points_2d[i].x, net_eval.key_points_2d[i].y)).norm());
			sum_prob += errors.back();
		}
		certainties.push_back(net_eval.certainties[i]);
	}

	if (errors.empty())
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	// give wrist a higher weight
	double avg_error = sum_prob / errors.size();
	sum_prob = 4 * errors.front();
	int count = 4;
	double certainty = 4 * certainties.front();
	for (int i = 1; i < errors.size(); i++)
	{
		if (errors[i] < 1.5 * avg_error)
		{
			sum_prob += errors[i];
			count += 1;
			certainty += certainties[i];
		}
	}

	return std::make_pair(sum_prob / count, certainty * max_weight / count);
}

std::pair<double, double> quality_2d_key_points::optimal_stepsize(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance& current,
	const hand_pose_particle_instance& next) const
{
	if (!current.net_eval)
		return std::make_pair(std::numeric_limits<float>::quiet_NaN(), 0);
	const auto& net_eval = current.net_eval;

	const net_evaluation::heatmaps& maps = net_eval->maps;
	const auto& box = net_eval->input_box;

	double sum_errors = 0.;

	float distance_scaling = 18.f / box.width; // input box width = 1 hand length = 18 cm

	std::vector<float> errors;
	std::vector<float> steps;
	std::vector<float> certainties;
	Eigen::Vector2f origin(box.x, box.y);
	Eigen::Vector2f scaling(maps.at(0).cols / (float)box.width,
		maps.at(0).rows / (float)box.height);

	for (int i = 0; i < current.key_points.cols(); i++)
	{
		if ((current.key_points.col(i) - next.key_points.col(i)).norm() < EPSILON)
			continue;

		if (net_eval->key_points_2d[i].x == -1 && net_eval->key_points_2d[i].y == -1)
			continue;

		Eigen::Vector2f current_pixel = input.to_img_coordinates_vec(current.key_points.col(i));
		Eigen::Vector2f next_pixel = input.to_img_coordinates_vec(next.key_points.col(i));
		Eigen::Vector2f best_pixel(net_eval->key_points_2d[i].x, net_eval->key_points_2d[i].y);

		auto next_vec = next_pixel - current_pixel;
		auto best_vec = best_pixel - current_pixel;

		if (next_vec.norm() < EPSILON)
			continue;
		else if (best_vec.norm() < EPSILON)
		{
			errors.push_back(0.f);
			steps.push_back(0.f);
			certainties.push_back(net_eval->certainties[i]);
			continue;
		}

		float step = best_vec.dot(next_vec) / next_vec.squaredNorm();
		steps.push_back(std::max(-1.f, std::min(1.f, step)));

		Eigen::Vector2f p = (current_pixel - origin).cwiseProduct(scaling);
		float prob = -0.5f;

		// extract interpolated value and handle out-of-image values
		if (p.x() > 0 && p.y() > 0 && p.x() < maps.at(i).cols - 1 && p.y() < maps.at(i).rows - 1)
		{
			float tl = maps.at(i).at<float>(std::floor(p.y()), std::floor(p.x()));
			float bl = maps.at(i).at<float>(std::ceil(p.y()), std::floor(p.x()));
			float tr = maps.at(i).at<float>(std::floor(p.y()), std::ceil(p.x()));
			float br = maps.at(i).at<float>(std::ceil(p.y()), std::ceil(p.x()));

			float integral;
			float dx = std::modf(p.x(), &integral);
			float t = tl * (1.f - dx) + tr * dx;
			float b = bl * (1.f - dx) + br * dx;

			float dy = std::modf(p.y(), &integral);
			prob = hand_pose_estimation::correct_heatmap_certainty(t * (1.f - dy) + b * dy);
		}

		if (prob > 0.1f)
		{
			errors.push_back(std::max(0.f, prob > 0.4f ? 0.1f / 0.6f * (1 - prob) : 2.1f - 5.f * prob));
			sum_errors += errors.back();
		}
		else
		{
			errors.push_back(distance_scaling * (current_pixel - Eigen::Vector2f(net_eval->key_points_2d[i].x, net_eval->key_points_2d[i].y)).norm());
			sum_errors += errors.back();
		}
		certainties.push_back(net_eval->certainties[i]);
	}

	if (errors.empty())
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	// give wrist a higher weight
	double avg_error = sum_errors / errors.size();
	double sum_step = 4 * steps.front();
	int count = 4;
	double certainty = 4 * certainties.front();
	for (int i = 0; i < errors.size(); i++)
	{
		if (errors[i] < 1.5 * avg_error)
		{
			sum_step += steps[i];
			count += 1;
			certainty += certainties[i];
		}
	}

	return std::make_pair(sum_step / count, certainty * max_weight / 21);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: quality_3d_key_points
//
//
/////////////////////////////////////////////////////////////

quality_criterion* quality_3d_key_points::clone() const
{
	return new quality_3d_key_points(max_weight);
}

std::pair<double, double> quality_3d_key_points::evaluate(const visual_input& input,
	const hand_pose_particle_instance& particle,
	const img_segment& seg) const
{
	if (!particle.net_eval)
		return std::make_pair(std::numeric_limits<float>::quiet_NaN(), 0);
	const auto& net_eval = particle.net_eval;

	Eigen::Matrix3Xf net_pose = Eigen::UniformScaling<float>(particle.pose.bone_scaling(0, 2) * particle.pose.hand_kinematic_params.get_finger(finger_type::MIDDLE).base_offset.norm()) *
		net_eval->left_hand_pose;

	Eigen::Affine3f transform = hand_pose_estimation::fit_pose(net_pose, particle.key_points, true);

	return std::make_pair(100. * (transform * particle.key_points - net_pose).colwise().norm().mean(), max_weight);
}

std::pair<double, double> quality_3d_key_points::optimal_stepsize(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance& current,
	const hand_pose_particle_instance& next) const
{
	if (!current.net_eval)
		return std::make_pair(std::numeric_limits<float>::quiet_NaN(), 0);
	const auto& net_eval = current.net_eval;

	bool finger_moved = false;
	if ((current.pose.get_parameters() - next.pose.get_parameters()).tail(12).cwiseAbs().sum() < EPSILON)
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	Eigen::Matrix3Xf net_pose = Eigen::UniformScaling<float>(current.pose.bone_scaling(0, 2) * current.pose.hand_kinematic_params.get_finger(finger_type::MIDDLE).base_offset.norm()) *
		net_eval->left_hand_pose;

	Eigen::Affine3f transform = hand_pose_estimation::fit_pose(net_pose, current.key_points, true);

	Eigen::Matrix3Xf best_pose = transform * net_pose;

	float sum_step = 0.f;
	int count = 0;

	for (int i = 0; i < current.key_points.cols(); i++)
	{
		if ((current.key_points.col(i) - next.key_points.col(i)).norm() < EPSILON)
			continue;

		auto next_vec = next.key_points.col(i) - current.key_points.col(i);
		auto best_vec = best_pose.col(i) - current.key_points.col(i);

		if (next_vec.norm() < EPSILON)
			continue;
		else if (best_vec.norm() < EPSILON)
		{
			count++;
			continue;
		}

		float step = best_vec.dot(next_vec) / next_vec.squaredNorm();
		sum_step += std::max(-1.f, std::min(1.f, step));
		count++;
	}

	return std::make_pair(sum_step / count, max_weight);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: quality_key_points_below_surface
//
//
/////////////////////////////////////////////////////////////

quality_criterion* quality_key_points_below_surface::clone() const
{
	return new quality_key_points_below_surface(max_weight);
}

std::pair<double, double> quality_key_points_below_surface::evaluate(const visual_input& input,
	const hand_pose_particle_instance& particle,
	const img_segment& seg) const
{
	if (!seg.prop_3d)
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	const Eigen::Matrix3Xf& key_points = particle.key_points;
	const auto& box = seg.bounding_box;

	const float radius = 0.125f * particle.pose.hand_kinematic_params.thickness;

	double sum_prob = 0;
	for (int i = 0; i < key_points.cols(); i++)
	{
		const auto& key_point = key_points.col(i);

		float surface_distance = std::isfinite(particle.surface_distances[i]) ? particle.surface_distances[i] : distance_to_surface(input, seg, key_point, radius);
		if (surface_distance < -radius)
			continue;

		pcl::Normal normal;
		visual_input::PointT p;

		seg.prop_3d->get_surface_point(input, key_point, p, &normal);

		Eigen::Vector3f neg_normal = -normal.getNormalVector3fMap().normalized();
		Eigen::Vector3f surface = p.getVector3fMap();

		cv::Point2i pixel(input.to_cloud_pixel_coordinates(key_point));
		if (input.is_valid_point(pixel) && input.get_point(pixel).getVector3fMap().norm() + 2 * radius < key_point.norm() &&
			(input.get_point(pixel).getVector3fMap() - surface).norm() > EPSILON)
			continue; // occluded by non skin color pixel

		Eigen::Vector3f best_p = surface + radius * neg_normal;
		if (best_p.norm() < key_point.norm())
			best_p *= key_point.norm() / best_p.norm();
		else if (key_point.norm() < surface.norm())
		{
			visual_input::PointT inner_p;
			seg.prop_3d->get_surface_point(input, best_p, inner_p, &normal);

			Eigen::Vector3f dest_inner = inner_p.getVector3fMap() - radius * normal.getNormalVector3fMap().normalized();

			if ((dest_inner - key_point).norm() < (best_p - key_point).norm())
				best_p = dest_inner;
			else
			{
				float dist = std::min(best_p.norm(), std::max(dest_inner.norm(), key_point.norm()));
				best_p *= dist / best_p.norm();
			}
		}

		if (!creates_force(key_point, best_p, neg_normal, radius, surface_distance))
			continue;

		sum_prob += (key_point - best_p).squaredNorm();

	}

	return std::make_pair(10000. * sum_prob / key_points.cols(), max_weight);
}


std::pair<double, double> quality_key_points_below_surface::optimal_stepsize(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance& current,
	const hand_pose_particle_instance& next) const
{
	if (!input.has_cloud())
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	const auto& box = seg.bounding_box;

	const float radius = 0.125f * current.pose.hand_kinematic_params.thickness;

	double sum_step = 0.;
	double count = 0.;
	for (int i = 0; i < current.key_points.cols(); i++)
	{
		const auto& current_key_point = current.key_points.col(i);
		Eigen::Vector3f next_vec = next.key_points.col(i) - current_key_point;

		if (next_vec.norm() < EPSILON)
			continue;

		float surface_distance = std::isfinite(current.surface_distances[i]) ? current.surface_distances[i] : distance_to_surface(input, seg, current_key_point, radius);

		pcl::Normal normal;
		visual_input::PointT p;

		seg.prop_3d->get_surface_point(input, current_key_point, p, &normal);

		Eigen::Vector3f neg_normal = -normal.getNormalVector3fMap();
		Eigen::Vector3f surface = p.getVector3fMap();

		cv::Point2i pixel(input.to_cloud_pixel_coordinates(current_key_point));
		if (input.is_valid_point(pixel) && input.get_point(pixel).getVector3fMap().norm() + 2 * radius < current_key_point.norm() &&
			(input.get_point(pixel).getVector3fMap() - surface).norm() > EPSILON)
			continue; // occluded by non skin color pixel

		Eigen::Vector3f best_p = surface + radius * neg_normal;
		if (best_p.norm() < current_key_point.norm())
			best_p *= current_key_point.norm() / best_p.norm();
		else if (current_key_point.norm() < surface.norm())
		{
			visual_input::PointT inner_p;
			seg.prop_3d->get_surface_point(input, best_p, inner_p, &normal);

			Eigen::Vector3f dest_inner = inner_p.getVector3fMap() - radius * normal.getNormalVector3fMap();

			if ((dest_inner - current_key_point).norm() < (best_p - current_key_point).norm())
				best_p = dest_inner;
			else
			{
				float dist = std::min(best_p.norm(), std::max(dest_inner.norm(), current_key_point.norm()));
				best_p *= dist / best_p.norm();
			}
		}

		Eigen::Vector3f best_vec = best_p - current_key_point;
		if (!creates_force(current_key_point, best_p, neg_normal, radius, surface_distance))
		{
			// CASE: zero error - make sure that movement doesn't increase it
			float next_surface_distance = std::isfinite(next.surface_distances[i]) ? next.surface_distances[i] : distance_to_surface(input, seg, next.key_points.col(i), radius);
			if (next_surface_distance < -0.5f * radius ||
				distance_to_surface(input, seg, 2 * current_key_point - next.key_points.col(i), radius) < -0.5f * radius)
			{
				count += radius * radius;
				continue;
			}

			auto intersection = [&](const Eigen::Vector3f& ref)
			{
				pcl::Normal normal;
				visual_input::PointT p;

				seg.prop_3d->get_surface_point(input, ref, p, &normal);

				Eigen::Vector3f n = normal.getNormalVector3fMap();
				Eigen::Vector3f p0 = p.getVector3fMap() - radius * n;
				Eigen::Vector3f l0 = current_key_point;
				Eigen::Vector3f l = next_vec;

				if (std::abs(l.dot(n)) < EPSILON)
					return std::numeric_limits < float > ::quiet_NaN();

				return ((p0 - l0).dot(n)) / l.dot(n);
			};

			// create three planes and intersect the line [current_key_point, next.key_points.col(i)] with it
			for (auto& p : std::vector<Eigen::Vector3f>({ current_key_point, next.key_points.col(i), current_key_point - next_vec }))
			{
				float w = std::abs(intersection(p));
				if (w < 0.5f)
				{
					count += radius * radius;
					break;
				}

			}

			continue;
		}

		if (best_vec.z() < 0 && next_vec.z() >= 0)
		{
			best_vec.z() = 0;
			next_vec.z() = 0;

			if (next_vec.norm() < EPSILON)
				continue;
		}
		float step = best_vec.dot(next_vec) / next_vec.squaredNorm();
		step = std::max(-1.f, std::min(1.f, step));

		// CASE: movement orthogonal to force direction
		float cos_angle = std::abs(best_vec.normalized().dot(next_vec.normalized()));
		if (cos_angle < 0.707f)
		{
			step *= cos_angle / 0.707f;
		}

		if (surface_distance < 0)
		{
			float w = std::powf(std::max(0.f, radius - surface_distance), 2.f);
			sum_step += step * w;
			count += w;
		}
		else
		{
			float w = best_vec.squaredNorm();
			sum_step += step * w;
			count += w;
		}
	}

	if (!count)
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	return std::make_pair(sum_step / count, max_weight);
}

pcl::PointCloud<pcl::PointXYZLNormal>::Ptr quality_key_points_below_surface::get_forces(const visual_input& input,
	const hand_pose_particle_instance& current, const img_segment& seg) const
{
	auto result = pcl::make_shared<pcl::PointCloud<pcl::PointXYZLNormal>>();

	if (!input.has_cloud())
		return result;

	const auto& box = seg.bounding_box;

	const float radius = finger_radius(current.pose);

	for (int i = 0; i < current.key_points.cols(); i++)
	{
		const auto& key_point = current.key_points.col(i);

		float surface_distance = std::isfinite(current.surface_distances[i]) ? current.surface_distances[i] : distance_to_surface(input, seg, key_point, radius);
		if (surface_distance < -radius)
			continue;

		pcl::Normal normal;
		visual_input::PointT p;

		seg.prop_3d->get_surface_point(input, key_point, p, &normal);

		Eigen::Vector3f neg_normal = -normal.getNormalVector3fMap().normalized();
		Eigen::Vector3f surface = p.getVector3fMap();

		cv::Point2i pixel(input.to_cloud_pixel_coordinates(key_point));
		if (input.is_valid_point(pixel) && input.get_point(pixel).getVector3fMap().norm() + 2 * radius < key_point.norm() &&
			(input.get_point(pixel).getVector3fMap() - surface).norm() > EPSILON)
			continue; // occluded by non skin color pixel

		Eigen::Vector3f best_p = surface + radius * neg_normal;
		if (best_p.norm() < key_point.norm())
			best_p *= key_point.norm() / best_p.norm();
		else if (key_point.norm() < surface.norm())
		{
			visual_input::PointT inner_p;
			seg.prop_3d->get_surface_point(input, best_p, inner_p, &normal);

			Eigen::Vector3f dest_inner = inner_p.getVector3fMap() - radius * normal.getNormalVector3fMap().normalized();

			if ((dest_inner - key_point).norm() < (best_p - key_point).norm())
				best_p = dest_inner;
			else
			{
				float dist = std::min(best_p.norm(), std::max(dest_inner.norm(), key_point.norm()));
				best_p *= dist / best_p.norm();
			}
		}

		if (!creates_force(key_point, best_p, neg_normal, radius, surface_distance))
		{
			continue;
		}

		Eigen::Vector3f best_vec = best_p - key_point;
		if (surface_distance < 0)
			best_vec *= 1.f - surface_distance / radius;

		pcl::PointXYZLNormal result_p;
		result_p.x = key_point.x();
		result_p.y = key_point.y();
		result_p.z = key_point.z();
		result_p.label = i;
		result_p.normal_x = best_vec.x();
		result_p.normal_y = best_vec.y();
		result_p.normal_z = best_vec.z();

		result->push_back(result_p);
	}

	return result;
}

float quality_key_points_below_surface::distance_to_surface(const visual_input& input,
	const img_segment& seg, const Eigen::Vector3f& key_point, float radius)
{

	pcl::Normal normal;
	visual_input::PointT p;
	float dists[3];

	seg.prop_3d->get_surface_point(input, key_point, p, &normal);
	Eigen::Vector3f n = normal.getNormalVector3fMap();

	float dist = n.dot(key_point) - n.dot(p.getVector3fMap());
	if (std::isfinite(dist))
		dists[0] = dist;
	else
		dists[0] = (key_point - p.getVector3fMap()).norm();

	Eigen::Vector3f key_point_n = key_point.normalized();
	Eigen::Vector3f orthogonal = (n - n.dot(key_point_n) * key_point_n).normalized();


	seg.prop_3d->get_surface_point(input, 0.5f * radius * orthogonal + key_point, p, &normal);
	n = normal.getNormalVector3fMap();

	dist = n.dot(key_point) - n.dot(p.getVector3fMap());
	if (std::isfinite(dist))
		dists[1] = dist;
	else
		dists[1] = (key_point - p.getVector3fMap()).norm();

	//seg.prop_3d->get_surface_point(input, radius * orthogonal + key_point, p, &normal);
	//n = normal.getNormalVector3fMap().normalized();

	//dist = n.dot(key_point) - n.dot(p.getVector3fMap());
	//if (std::isfinite(dist))
	//	dists[2] = dist;
	//else
	//	dists[2] = (key_point - p.getVector3fMap()).norm();

	//std::sort(std::begin(dists), std::end(dists), [](float lhs, float rhs)
	//	{
	//		return std::abs(lhs) < std::abs(rhs);
	//	});

	//return dists[0];

	return std::abs(dists[0]) < std::abs(dists[1]) ? dists[0] : dists[1];
}

bool quality_key_points_below_surface::creates_force(const Eigen::Vector3f& key_point,
	const Eigen::Vector3f& best,
	const Eigen::Vector3f& neg_normal,
	float radius,
	float surface_distance) const
{
	Eigen::Vector3f best_vec = best - key_point;
	const float cos_pi_8 = 0.92388f;

	if (best_vec.norm() < EPSILON)
		return false;

	//if (Eigen::Vector3f::UnitZ().dot(neg_normal) > cos_pi_8 /* 22.5° */ && // normal towards camera
	//	std::abs(best.norm() - key_point.norm()) < EPSILON && // key point on same distance or behind best
	//	cv::norm(input.to_cloud_pixel_coordinates(best) - input.to_cloud_pixel_coordinates(key_point)) <= 2.001f) // key point not in gap 
	//	return false;

	if (std::isfinite(neg_normal.x()) && neg_normal.dot(best_vec.normalized()) < 0) // best_vec pointing in the direction of normal
		return false;

	if (key_point.normalized().dot(best_vec.normalized()) < -cos_pi_8) // best_vec pointing towards camera
		return false;

	if (surface_distance < -radius)
		return false;

	return true;
}

float quality_key_points_below_surface::finger_radius(const hand_pose_18DoF& pose)
{
	return 0.125f * pose.hand_kinematic_params.thickness;
}

/////////////////////////////////////////////////////////////
//
//
//  Class: quality_key_points_close_to_surface
//
//
/////////////////////////////////////////////////////////////

quality_criterion* quality_key_points_close_to_surface::clone() const
{
	return new quality_key_points_close_to_surface(max_weight);
}

std::pair<double, double> quality_key_points_close_to_surface::evaluate(const visual_input& input,
	const hand_pose_particle_instance& particle,
	const img_segment& seg) const
{
	if (!input.has_cloud())
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	auto cmp = [](const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
	{	return lhs.norm() < rhs.norm(); };

	std::priority_queue<Eigen::Vector3f, std::vector<Eigen::Vector3f>, decltype(cmp)> key_points(cmp);

	for (int i = 0; i < particle.key_points.cols(); i++)
		key_points.emplace(particle.key_points.col(i));


	const auto& box = seg.bounding_box;

	const float radius = 0.125f * particle.pose.hand_kinematic_params.thickness;

	double sum_prob = 0;
	int count = 0;
	while (count < 5 && !key_points.empty())
	{
		Eigen::Vector3f key_point = key_points.top();
		key_points.pop();

		pcl::Normal normal;
		visual_input::PointT p;

		seg.prop_3d->get_surface_point(input, key_point, p, &normal);

		Eigen::Vector3f neg_normal = -normal.getNormalVector3fMap().normalized();
		Eigen::Vector3f dest = p.getVector3fMap() + radius * neg_normal;

		cv::Point2i pixel(input.to_cloud_pixel_coordinates(key_point));
		cv::Point2i pixel_img(input.to_img_coordinates(key_point) - seg.bounding_box.tl());
		if (input.is_valid_point(pixel) &&
			input.get_point(pixel).getVector3fMap().norm() + radius < key_point.norm() &&
			(pixel_img.x < 0 || pixel_img.y < 0 || pixel_img.x >= seg.mask.cols || pixel_img.y >= seg.mask.rows || seg.mask.at<char>(pixel_img) == 0))
			continue; // occluded by non skin color pixel


		if (std::abs(Eigen::Vector3f::UnitZ().dot(neg_normal.normalized())) < 0.707f)
			seg.prop_3d->get_surface_point(input, dest, p, &normal);

		dest = p.getVector3fMap() - radius * normal.getNormalVector3fMap().normalized();
		const auto& vp = seg.prop_3d->sensor_origin;
		float z = std::max(0.f, (key_point - vp).norm() - (dest - vp).norm());

		sum_prob += z;
		count++;
	}

	if (!count)
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	return std::make_pair(100. * sum_prob / count, max_weight);
}

std::pair<double, double> quality_key_points_close_to_surface::optimal_stepsize(const visual_input& input, const img_segment& seg,
	const hand_pose_particle_instance& current, const hand_pose_particle_instance& next) const
{
	if (!input.has_cloud())
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	auto cmp = [&](int lhs, int rhs)
	{	return current.key_points.col(lhs).norm() < current.key_points.col(rhs).norm(); };

	std::priority_queue<int, std::vector<int>, decltype(cmp)> key_points(cmp);

	for (int i = 0; i < current.key_points.cols(); i++)
		key_points.emplace(i);


	const auto& box = seg.bounding_box;

	const float radius = 0.5f * current.pose.hand_kinematic_params.thickness;

	float sum_step = 0;
	int count = 0;
	while (count < 5 && !key_points.empty())
	{
		int i = key_points.top();
		key_points.pop();

		const auto& key_point = current.key_points.col(i);
		Eigen::Vector3f next_vec = next.key_points.col(i) - key_point;

		if (next_vec.norm() < EPSILON)
			continue;

		pcl::Normal normal;
		visual_input::PointT p;

		seg.prop_3d->get_surface_point(input, key_point, p, &normal);

		Eigen::Vector3f neg_normal = -normal.getNormalVector3fMap().normalized();
		Eigen::Vector3f dest = p.getVector3fMap() + radius * neg_normal;

		cv::Point2i pixel(input.to_cloud_pixel_coordinates(key_point));
		cv::Point2i pixel_img(input.to_img_coordinates(key_point) - seg.bounding_box.tl());
		if (input.is_valid_point(pixel) &&
			input.get_point(pixel).getVector3fMap().norm() + radius < key_point.norm() &&
			(pixel_img.x < 0 || pixel_img.y < 0 || pixel_img.x >= seg.mask.cols || pixel_img.y >= seg.mask.rows || seg.mask.at<char>(pixel_img) == 0))
			continue; // occluded by non skin color pixel


		if (std::abs(Eigen::Vector3f::UnitZ().dot(neg_normal.normalized())) < 0.707f)
			seg.prop_3d->get_surface_point(input, dest, p, &normal);

		dest = p.getVector3fMap() - radius * normal.getNormalVector3fMap().normalized();
		const Eigen::Vector3f vp = seg.prop_3d->cloud->sensor_origin_.head<3>();
		float delta_z = std::max(0.f, (key_point - vp).norm() - (dest - vp).norm());

		if (delta_z < EPSILON)
		{
			float max_z = (key_point - vp).norm() + 0.5f * std::abs(next_vec.dot((key_point - vp).normalized()));
			if ((dest - vp).norm() < max_z)
				if (std::abs(next_vec.normalized().dot(neg_normal)) < 0.707f)
					count++;

			continue;
		}

		float step = (-delta_z) * next_vec.z() / next_vec.squaredNorm();
		step = std::max(-1.f, std::min(1.f, step));

		// CASE: movement orthogonal to force direction
		float cos_angle = std::abs(Eigen::Vector3f::UnitZ().dot(next_vec.normalized()));
		if (cos_angle < 0.707f)
		{
			step *= cos_angle / 0.707f;
		}

		sum_step += std::max(-1.f, std::min(1.f, step));
		count++;

	}

	if (!count)
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	return std::make_pair(sum_step / count, max_weight);
}

pcl::PointCloud<pcl::PointXYZLNormal>::Ptr quality_key_points_close_to_surface::get_forces(const visual_input& input,
	const hand_pose_particle_instance& particle,
	const img_segment& seg) const
{
	auto result = pcl::make_shared<pcl::PointCloud<pcl::PointXYZLNormal>>();

	auto cmp = [](const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
	{	return lhs.norm() < rhs.norm(); };

	std::priority_queue<Eigen::Vector3f, std::vector<Eigen::Vector3f>, decltype(cmp)> key_points(cmp);

	for (int i = 0; i < particle.key_points.cols(); i++)
		key_points.emplace(particle.key_points.col(i));


	const auto& box = seg.bounding_box;

	const float radius = 0.5f * particle.pose.hand_kinematic_params.thickness;

	for (int i = 0; i < particle.key_points.cols(); i++)
	{
		Eigen::Vector3f key_point = key_points.top();
		key_points.pop();

		pcl::Normal normal;
		visual_input::PointT p;

		seg.prop_3d->get_surface_point(input, key_point, p, &normal);

		Eigen::Vector3f neg_normal = -normal.getNormalVector3fMap().normalized();
		Eigen::Vector3f dest = p.getVector3fMap() + radius * neg_normal;

		cv::Point2i pixel(input.to_cloud_pixel_coordinates(key_point));
		cv::Point2i pixel_img(input.to_img_coordinates(key_point) - seg.bounding_box.tl());
		if (input.is_valid_point(pixel) &&
			input.get_point(pixel).getVector3fMap().norm() + radius < key_point.norm() &&
			(pixel_img.x < 0 || pixel_img.y < 0 || pixel_img.x >= seg.mask.cols || pixel_img.y >= seg.mask.rows || seg.mask.at<char>(pixel_img) == 0))
			continue; // occluded by non skin color pixel


		if (std::abs(Eigen::Vector3f::UnitZ().dot(neg_normal.normalized())) < 0.707f)
			seg.prop_3d->get_surface_point(input, dest, p, &normal);

		dest = p.getVector3fMap() - radius * normal.getNormalVector3fMap().normalized();

		float delta_z = std::max(0.f, key_point.norm() - dest.norm());

		Eigen::Vector3f best_vec = -key_point / key_point.norm() * delta_z;

		pcl::PointXYZLNormal result_p;
		result_p.x = key_point.x();
		result_p.y = key_point.y();
		result_p.z = key_point.z();
		result_p.label = i;
		result_p.normal_x = best_vec.x();
		result_p.normal_y = best_vec.y();
		result_p.normal_z = best_vec.z();

		result->push_back(result_p);
	}

	return result;
}



/////////////////////////////////////////////////////////////
//
//
//  Class: quality_boundary_surface
//
//
/////////////////////////////////////////////////////////////

quality_criterion* quality_boundary_surface::clone() const
{
	return new quality_boundary_surface(max_weight, plane);
}

std::pair<double, double> quality_boundary_surface::evaluate(const visual_input& input,
	const hand_pose_particle_instance& particle,
	const img_segment& seg) const
{
	double sum_prob = 0;
	for (int i = 0; i < particle.key_points.cols(); i++)
	{
		sum_prob += std::powf(std::max(0.f, -plane.signedDistance(particle.key_points.col(i))), 2.f);
	}

	return std::make_pair(10000. * sum_prob / particle.key_points.cols(), max_weight);
}

std::pair<double, double> quality_boundary_surface::optimal_stepsize(const visual_input& input, const img_segment& seg,
	const hand_pose_particle_instance& current, const hand_pose_particle_instance& next) const
{
	double sum_step = 0;
	int count = 0;
	for (int i = 0; i < current.key_points.cols(); i++)
	{
		Eigen::Vector3f next_vec = next.key_points.col(i) - current.key_points.col(i);

		if (next_vec.norm() < EPSILON)
			continue;

		float dist = plane.signedDistance(current.key_points.col(i));
		if (dist > -EPSILON)
			continue;

		Eigen::Vector3f best_vec = (-dist / plane.normal().norm()) * plane.normal();

		if (best_vec.norm() < EPSILON)
		{
			count++;
			continue;
		}

		float step = best_vec.dot(next_vec) / next_vec.squaredNorm();
		step = std::max(-1.f, std::min(1.f, step));

		// CASE: movement orthogonal to force direction
		float cos_angle = std::abs(best_vec.normalized().dot(next_vec.normalized()));
		if (cos_angle < 0.707f)
		{
			step *= cos_angle / 0.707f;
		}

		sum_step += step;
		count++;
	}

	if (!count)
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	return std::make_pair(sum_step / count, max_weight);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: quality_acceleration
//
//
/////////////////////////////////////////////////////////////

quality_criterion* quality_acceleration::clone() const
{
	return new quality_acceleration(max_weight, reference);
}

std::pair<double, double> quality_acceleration::evaluate(const visual_input& input,
	const hand_pose_particle_instance& particle,
	const img_segment& seg) const
{
	if (!reference)
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	hand_pose_18DoF::Vector15f diff = particle.pose.get_parameters() - reference->pose.get_parameters();

	weight(diff);
	float a_dist = particle.rotation.angularDistance(reference->rotation);
	float sq_norm = diff.norm() + 5 * a_dist;

	return std::make_pair(sq_norm, max_weight);
}

std::pair<double, double> quality_acceleration::optimal_stepsize(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance& current,
	const hand_pose_particle_instance& next) const
{
	if (!reference)
		return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

	Eigen::AngleAxisf next_rotation;
	next_rotation = next.rotation * current.rotation.inverse();

	if (Eigen::AngleAxisf::Identity().isApprox(next_rotation))
	{
		hand_pose_18DoF::Vector15f best_vec = reference->pose.get_parameters() - current.pose.get_parameters();
		hand_pose_18DoF::Vector15f next_vec = next.pose.get_parameters() - current.pose.get_parameters();

		weight(best_vec);
		weight(next_vec);
		int count = 0;
		float sum_step = 0.f;

		for (int i = 0; i < 15; i++)
		{
			if (std::abs(next_vec(i)) < EPSILON)
				continue;

			sum_step += std::max(-1.f, std::min(1.f, best_vec(i) / next_vec(i)));
			count++;
		}

		if (!count)
			return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

		return std::make_pair(sum_step / count, max_weight);
	}
	else
	{
		Eigen::Quaternionf best_rotation = reference->rotation * current.rotation.inverse();
		float min_dist = std::numeric_limits<float>::infinity();
		float best_step;

	
		for (int i = 0; i < 9; i++)
		{
			float step = i / 4.f - 1.f;
			Eigen::AngleAxisf step_rotation = next_rotation;
			step_rotation.angle() *= step;
			
			float dist = best_rotation.angularDistance(Eigen::Quaternionf(step_rotation));

			if (dist < min_dist) {
				best_step = step;
				min_dist = dist;
			}
		}

		return std::make_pair(best_step, max_weight);
	}


}

void quality_acceleration::weight(hand_pose_18DoF::Vector15f& parameters)
{
	parameters.head(3) *= 50.f;
	//parameters.segment(3, 12) *= 0.5f;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: quality_fill_mask
//
//
/////////////////////////////////////////////////////////////

quality_criterion* quality_fill_mask::clone() const
{
	return new quality_fill_mask(max_weight);
}

std::pair<double, double> quality_fill_mask::evaluate(const visual_input& input, const hand_pose_particle_instance& particle, const img_segment& seg) const
{
	const Eigen::Matrix3Xf& key_points = particle.key_points;
	const auto& box = seg.bounding_box;
	const auto& mask = seg.mask;

	Eigen::Matrix<float, 3, 4> transform = Eigen::Affine2f(Eigen::Translation2f(-box.x, -box.y)) *
		input.img_projection;

	Eigen::Vector4f wrist = key_points.col(0).homogeneous();
	Eigen::Vector2f wrist_proj = (input.img_projection * wrist).hnormalized();
	wrist(0) += 0.4f * particle.pose.hand_kinematic_params.thickness;
	const float radius = (wrist_proj - (input.img_projection * wrist).hnormalized()).norm();
	const cv::Scalar color(0);

	cv::Mat canvas;
	mask.copyTo(canvas);
	std::vector<cv::Point2i> pixels;
	for (int i = 0; i < key_points.cols(); i++)
	{

		Eigen::Vector2f p = (transform * key_points.col(i).homogeneous()).hnormalized();
		cv::Point2f pixel(p.x(), p.y());

		cv::circle(canvas, pixel, radius, color, cv::FILLED);

		if (pixels.size())
		{
			if (pixels.size() % 4 == 1)
			{
				cv::line(canvas, pixels.front(), pixel, color, 2 * radius);
			}
			else
				cv::line(canvas, pixels.back(), pixel, color, 2 * radius);
		}

		pixels.push_back(pixel);
		//cv::putText(canvas, std::to_string(i), pixel,cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(0,0,255));
	}

	std::vector<cv::Point2i> palm;
	palm.push_back(pixels[2]);
	palm.push_back(pixels[5]);
	palm.push_back(pixels[17]);
	palm.push_back(pixels[1] - pixels[5] + pixels[17]);

	cv::fillConvexPoly(canvas, palm, color);

	//cv::imshow("maks", canvas);
	//cv::waitKey(1);

	return std::make_pair(10. * cv::countNonZero(canvas) / (double)cv::countNonZero(mask), max_weight);

}

std::pair<double, double> quality_fill_mask::optimal_stepsize(const visual_input& input, const img_segment& seg, const hand_pose_particle_instance& current, const hand_pose_particle_instance& next) const
{
	return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: quality_centroid
//
//
/////////////////////////////////////////////////////////////

quality_criterion* quality_centroid::clone() const
{
	return new quality_centroid(max_weight);
}
	
std::pair<double, double> quality_centroid::evaluate(const visual_input& input,
	const hand_pose_particle_instance& particle, const img_segment& seg) const
{
	float dist;
	if (seg.prop_3d)
		dist = 100.f * (hand_pose_18DoF::get_centroid(particle.key_points) - seg.prop_3d->centroid).norm();
	else
	{
		const Eigen::Matrix3Xf& key_points = particle.key_points;

		cv::Point2i sum;

		float distance_scaling = 18.f / std::max(seg.bounding_box.width, seg.bounding_box.height);

		for (int i = 0; i < key_points.cols(); i++)
		{
			sum += hand_pose_18DoF::centroid_weights[i] * input.to_img_coordinates(key_points.col(i));
		}
		dist = distance_scaling * cv::norm(sum - seg.palm_center_2d);
	}

	return std::make_pair(dist, max_weight);
}

std::pair<double, double> quality_centroid::optimal_stepsize(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance& current,
	const hand_pose_particle_instance& next) const
{
	if (seg.prop_3d)
	{
		Eigen::Vector3f centroid_current = hand_pose_18DoF::get_centroid(current.key_points);
		Eigen::Vector3f centroid_next = hand_pose_18DoF::get_centroid(next.key_points);

		Eigen::Vector3f next_vec = centroid_next - centroid_current;
		if (next_vec.norm() < EPSILON)
			return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

		Eigen::Vector3f best_vec = seg.prop_3d->centroid - centroid_current;
		if (best_vec.norm() < EPSILON)
			return std::make_pair(0.f, max_weight);

		float step = best_vec.dot(next_vec) / next_vec.squaredNorm();
		step = std::max(-1.f, std::min(1.f, step));
		float cos_angle = std::abs(best_vec.normalized().dot(next_vec.normalized()));
		float weight = max_weight;
		if (cos_angle < 0.707f)
		{
			weight *= cos_angle / 0.707f;
		}

		return std::make_pair(step, weight);
	}
	else
	{
		Eigen::Vector3f centroid_current = hand_pose_18DoF::get_centroid(current.key_points);
		Eigen::Vector3f centroid_next = hand_pose_18DoF::get_centroid(next.key_points);

		Eigen::Vector2f next_vec = (input.img_projection * centroid_next.homogeneous()).hnormalized() - (input.img_projection * centroid_current.homogeneous()).hnormalized();
		if (next_vec.norm() < EPSILON)
			return std::make_pair(std::numeric_limits<double>::quiet_NaN(), max_weight);

		Eigen::Vector2f best_vec = Eigen::Vector2f(seg.palm_center_2d.x, seg.palm_center_2d.y) - (input.img_projection * centroid_current.homogeneous()).hnormalized();
		if (best_vec.norm() < EPSILON)
			return std::make_pair(0.f, max_weight);

		float step = best_vec.dot(next_vec) / next_vec.squaredNorm();
		step = std::max(-1.f, std::min(1.f, step));
		float cos_angle = std::abs(best_vec.normalized().dot(next_vec.normalized()));
		float weight = max_weight;
		if (cos_angle < 0.707f)
		{
			weight *= cos_angle / 0.707f;
		}

		return std::make_pair(step, weight);
	}
}


/////////////////////////////////////////////////////////////
//
//
//  Class: gradient_decent_optimization_common
//
//
/////////////////////////////////////////////////////////////

double gradient_decent_optimization_common::bell_curve(double x)
{
	return std::exp(-x * x / (2 * 3. * 3.));
};

float gradient_decent_optimization_common::bell_curve(float x, float stdev)
{
	return std::expf(-x * x / (2 * stdev * stdev));
}

gradient_decent_optimization_common::gradient_decent_optimization_common(std::vector<quality_criterion::Ptr> objectives,
	Eigen::Vector3f back_palm_orientation,
	gradient_decent_parameters::ConstPtr params,
	hand_dynamic_parameters::ConstPtr dynamic_params)
	:
	params(params ? params : std::make_shared<gradient_decent_parameters>()),
	dynamic_params(dynamic_params ? dynamic_params : std::make_shared<hand_dynamic_parameters>()),
	timestamp(0.f),
	prev_timestamp(0.f),
	objectives(std::move(objectives)),
	back_palm_orientation(back_palm_orientation)
{
}

gradient_decent_optimization_common::gradient_decent_optimization_common(const gradient_decent_optimization_common& other)
	:
	params(other.params),
	dynamic_params(other.dynamic_params),
	back_palm_orientation(other.back_palm_orientation),
	best(other.best),
	prev_particle(other.prev_particle),
	timestamp(other.timestamp),
	prev_timestamp(other.prev_timestamp),
	extrapolated_pose(other.extrapolated_pose)
{
	objectives.reserve(other.objectives.size());
	for(const auto& obj : other.objectives)
	{
		objectives.emplace_back(quality_criterion::Ptr(obj->clone()));
	}
}


void gradient_decent_optimization_common::evaluate_objectives(const visual_input& input,
                                                              const img_segment& seg,
                                                              hand_pose_particle_instance& particle) const
{
	float radius = quality_key_points_below_surface::finger_radius(particle.pose);
	if (!std::isfinite(particle.surface_distances[0]))
		for (int i = 0; i < particle.key_points.cols(); i++)
			particle.surface_distances[i] = quality_key_points_below_surface::distance_to_surface(input, seg, particle.key_points.col(i), radius);


	double sum_prob = 0;
	double sum_weight = 0;
	for (const auto& objective : objectives)
	{
		auto prob_weight = objective->evaluate(input, particle, seg);
		double prob = prob_weight.first;
		double weight = prob_weight.second;
		if (!std::isnan(prob) && !std::isinf(prob))
		{
			sum_prob += prob * weight;
			sum_weight += weight;
		}
	}



	if (sum_weight > 0)
		particle.error = sum_prob / sum_weight;
	else
		particle.error = std::numeric_limits<double>::infinity();
}


hand_pose_18DoF::Ptr gradient_decent_optimization_common::constrain(const hand_pose_18DoF::Ptr& model)
{
	if (!prev_particle)
		return model;

	hand_pose_18DoF::Vector15f velocity_bounds = dynamic_params->get_constraints_18DoF();
	hand_pose_18DoF::Vector15f pose = model->get_parameters();
	float diff = std::chrono::duration<float>(timestamp - prev_timestamp).count();

	if (diff < quality_criterion::EPSILON)
		diff = 1 / 60.f;

	hand_pose_18DoF::Vector15f speed = pose - prev_particle->pose.get_parameters();
	speed /= diff;

	bool modified = false;
	for (int i = 0; i < pose.rows(); i++)
	{
		if (speed(i) > velocity_bounds(i))
		{
			pose(i) -= (speed(i) - velocity_bounds(i)) * diff;
			modified = true;
		}
		else if (speed(i) < -velocity_bounds(i))
		{
			pose(i) += (std::abs(speed(i)) - velocity_bounds(i)) * diff;
			modified = true;
		}
	}

	Eigen::Quaternionf rotation = Eigen::Quaternionf(model->wrist_pose.rotation());
	float angle = prev_particle->rotation.angularDistance(rotation);
	float angle_velocity = angle / diff;


	if (angle_velocity > dynamic_params->speed_rotation)
	{
		modified = true;
		rotation = prev_particle->rotation.slerp(dynamic_params->speed_rotation / angle_velocity, rotation);
	}

	if (modified)
		return std::make_shared<hand_pose_18DoF>(model->hand_kinematic_params,
			std::move(pose),
			std::move(rotation),
			model->right_hand,
			model->bone_scaling
			);
	else
		return model;
}

hand_pose_18DoF::Ptr gradient_decent_optimization_common::disturb(const hand_pose_18DoF& model)
{
	std::mt19937 generator;
	//float diff = std::chrono::duration<float>(timestamp - prev_timestamp).count();
	auto pose = model.get_parameters();

	hand_pose_18DoF::Vector15f velocity_bounds = dynamic_params->get_constraints_18DoF();
	hand_pose_18DoF::Vector15f lower = model.get_lower_bounds();
	hand_pose_18DoF::Vector15f upper = model.get_upper_bounds();

	for (int i = 0; i < pose.rows(); i++)
	{
		float stdev = prev_particle ? std::max(0.05f * velocity_bounds(i), 0.2f) : 0.1f * velocity_bounds(i);

		std::normal_distribution<float> dist(pose(i), stdev);
		float sample;
		int j = 0;
		do {
			sample = dist(generator);
		} while (j++ < 10 && (lower(i) > sample || upper(i) < sample));

		pose(i) = sample;
	}

	return std::make_shared<hand_pose_18DoF>(model.hand_kinematic_params,
		std::move(pose),
		Eigen::Quaternionf(model.wrist_pose.rotation()),
		model.right_hand,
		model.bone_scaling
		);
}

void gradient_decent_optimization_common::best_starting_point(const visual_input& input,
	const img_segment& seg,
	const std::vector<hand_pose_particle_instance::Ptr>& seeds,
	int max_steps)
{
	hand_pose_particle_instance::Ptr current;
	quality_2d_key_points quality_net(1.);

	//quality_fill_mask draw_keypoints(1);

	int best_index;
	int i = 0;
	for (const auto& seed : seeds)
	{
		evaluate_objectives(input, seg, *seed);
		float error = seed->error;
		hand_pose_particle_instance::Ptr seed_best = seed;

		//std::cout << " " << error;

		//if (seed_best->pose.right_hand)
		//	draw_keypoints.evaluate(input, *seed_best, seg);

		//if (error > 150)
		//	continue;

		for (int j = 0; j < max_steps; j++)
		{
			seed_best = gradient_step(input, seg, seed_best, j % 2);

			//if (seed_best->pose.right_hand)
			//	draw_keypoints.evaluate(input, *seed_best, seg);

			if (seed_best->error + params->min_improvement > error)
				break;

			error = seed_best->error;
		}


		if (!best || seed_best->error < best->error)
		{
			best = seed_best;
			best_index = i;
		}
		i++;
	}
	//std::cout << std::endl;
	//std::cout << "seed " << best_index << " " << seeds.size() << std::endl;

	//if (best->pose.right_hand)
	//	draw_keypoints.evaluate(input, *best, seg);

	if (!best)
		throw std::exception("No starting point found for gradient decent");
}

void gradient_decent_optimization_common::best_starting_point_parallel(const visual_input& input,
	const img_segment& seg,
	const std::vector<hand_pose_particle_instance::Ptr>& seeds)
{
	hand_pose_particle_instance::Ptr current;
	quality_2d_key_points quality_net(1.);


	int i = 0;
	int n = seeds.size();
	std::vector<hand_pose_particle_instance::Ptr> seeds_stepped(n);

#pragma omp parallel for
	for (i = 0; i < n; i++)
	{
		hand_pose_particle_instance::Ptr seed_best = seeds[i];

		evaluate_objectives(input, seg, *seed_best);
		float error = seed_best->error;

		for (int j = 0; j < 3; j++)
		{
			seed_best = gradient_step(input, seg, seed_best, j % 2);

			//if (seed_best->pose.right_hand)
			//	draw_keypoints.evaluate(input, *seed_best, seg);

			if (seed_best->error + params->min_improvement > error)
				break;

			error = seed_best->error;
		}

		seeds_stepped[i] = seed_best;
	}

	for (const auto& seed : seeds_stepped)
	{
		if (!best || seed->error < best->error)
			best = seed;
	}

	if (!best)
		throw std::exception("No starting point found for gradient decent");
}

std::pair<hand_pose_particle_instance::Ptr, float> gradient_decent_optimization_common::increment_parameter(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance& particle,
	int index,
	float inc) const
{
	auto pose_inc = index >= 3 && index < 6 ? increment_rotation_parameter(input, seg, particle, index - 3, inc) : increment_pose_parameter(input, seg, particle, index >= 3 ? index - 3 : index, inc);
	const auto& new_pose = pose_inc.first;

	// compute key points
	Eigen::Matrix3Xf key_points;
	if (index < 6)
	{
		Eigen::Affine3f transform = new_pose->wrist_pose * particle.pose.wrist_pose.inverse();
		key_points = transform * particle.key_points;
	}
	else if (index < 17)
	{
		int finger = (index - 7) / 2;
		key_points = particle.key_points;
		key_points.block<3, 4>(0, 1 + 4 * finger) = new_pose->get_key_points((finger_type)finger);
	}
	else
		key_points = new_pose->get_key_points();

	// compute distances of key points to surface
	std::vector<float> surface_distances;
	float radius = quality_key_points_below_surface::finger_radius(*new_pose);
	if (index >= 6 && index < 17)
	{
		int finger = (index - 7) / 2;
		surface_distances = particle.surface_distances;
		for (int i = 1 + 4 * finger; i < 5 + 4 * finger; i++)
		{
			surface_distances[i] = quality_key_points_below_surface::distance_to_surface(input, seg, key_points.col(i), radius);
		}
	}
	else
	{
		surface_distances.reserve(key_points.cols());
		for (int i = 0; i < key_points.cols(); i++)
			surface_distances.push_back(quality_key_points_below_surface::distance_to_surface(input, seg, key_points.col(i), radius));
	}

	auto next = std::make_shared<hand_pose_particle_instance>(*new_pose, particle.net_eval, particle.time_seconds, std::move(key_points), std::move(surface_distances));
	next->updates = particle.updates;
	
	 return std::make_pair(next, pose_inc.second);
}

std::pair<hand_pose_18DoF::Ptr, float> gradient_decent_optimization_common::increment_rotation_parameter(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance& particle,
	int index,
	float inc) const
{
	Eigen::Quaternionf rotation(Eigen::AngleAxisf(inc, Eigen::Vector3f(index == 2, index == 1, index == 0)));

	if (prev_particle)
	{
		float velocity_bound = dynamic_params->speed_wrist;
		float diff = std::chrono::duration<float>(timestamp - prev_timestamp).count();

		if (diff < quality_criterion::EPSILON)
			diff = 1 / 60.f;

		float speed = prev_particle->rotation.angularDistance(particle.rotation * rotation);
		speed /= diff;

		if (speed > velocity_bound)
			rotation = Eigen::Quaternionf::Identity().slerp(velocity_bound / speed, rotation);
	}

	return std::make_pair(std::make_shared<hand_pose_18DoF>(particle.pose.hand_kinematic_params,
		particle.pose.wrist_pose * Eigen::Affine3f(rotation),
		particle.pose.finger_bending,
		particle.pose.thumb_adduction,
		particle.pose.finger_spreading,
		particle.pose.right_hand,
		particle.pose.bone_scaling
		),
		inc);
}


std::pair<hand_pose_18DoF::Ptr, float> gradient_decent_optimization_common::increment_pose_parameter(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance& particle,
	int index,
	float inc) const
{
	hand_pose_18DoF::Vector15f pose = particle.pose.get_parameters();
	float new_val = std::max(particle.pose.get_lower_bounds()[index],
		std::min(particle.pose.get_upper_bounds()[index], pose[index] + inc)
	);

	inc = new_val - pose[index];
	pose[index] = new_val;

	if (prev_particle)
	{
		float velocity_bound = dynamic_params->get_constraints_18DoF()(index);
		float diff = std::chrono::duration<float>(timestamp - prev_timestamp).count();

		if (diff < quality_criterion::EPSILON)
			diff = 1 / 60.f;

		float speed = pose[index] - prev_particle->pose.get_parameter(index);
		speed /= diff;

		const int i = index;
		if (speed > velocity_bound)
		{
			inc -= (speed - velocity_bound) * diff;
			pose(i) -= (speed - velocity_bound) * diff;

		}
		else if (speed < -velocity_bound)
		{
			inc += (std::abs(speed) - velocity_bound) * diff;
			pose(i) += (std::abs(speed) - velocity_bound) * diff;

		}

	}

	return std::make_pair(std::make_shared<hand_pose_18DoF>(particle.pose.hand_kinematic_params,
		pose,
		particle.rotation,
		particle.pose.right_hand,
		particle.pose.bone_scaling
		),
		inc);



}

hand_pose_particle_instance::Ptr gradient_decent_optimization_common::gradient_step(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance::Ptr& particle,
	bool dir,
	int count_params) const
{
	auto local_best = particle;
	if (!std::isfinite(local_best->error))
		evaluate_objectives(input, seg, *local_best);

	for (int d = 0; d < count_params; d++)
	{
		float inc = d <= 3 ? 0.01f : 0.2f;
		auto next_inc = increment_parameter(input, seg, *local_best, d, dir ? inc : -inc);
		auto next = next_inc.first;

		double sum_step = 0;
		double sum_weight = 0;
		for (const auto& objective : objectives)
		{
			auto step_weight = objective->optimal_stepsize(input, seg, *local_best, *next);
			double step_size = step_weight.first;
			double weight = step_weight.second;
			if (!std::isnan(step_size) && !std::isinf(step_size))
			{
				sum_step += step_size * weight;
				sum_weight += weight;
			}
		}

		if (sum_weight == 0)
			continue;

		if (std::abs(sum_step / sum_weight * next_inc.second) < quality_criterion::EPSILON)
			continue;

		next = increment_parameter(input, seg, *local_best, d, sum_step / sum_weight * next_inc.second).first;

		evaluate_objectives(input, seg, *next);

		if (next->error < local_best->error) {
			local_best = next;
			//std::cout << "step " << particle->updates << " " << d << std::endl;
		}
	}

	local_best->updates++;

	return local_best;
}

bool gradient_decent_optimization_common::can_scale_bones(const hand_pose_particle_instance& particle) const
{
	if (!prev_particle)
		return false;

	if (prev_particle->pose.bone_scaling(0, 2) != 1.f)
		return false; // bones already scaled

	const auto& prev_pose = prev_particle->pose;
	const auto& pose = particle.pose;

	if ((prev_pose.wrist_pose.translation() - pose.wrist_pose.translation()).norm() >= params->translational_tolerance_movement)
		return false;


	Eigen::Vector3f z_axis = pose.wrist_pose.rotation().col(2);
	// std::abs to allow front and back facing palm
	if (std::acosf(std::abs(Eigen::Vector3f::UnitZ().dot(z_axis))) >= params->rotational_tolerance_flateness)
		return false;

	Eigen::Matrix3f prev_rotation = prev_pose.wrist_pose.rotation();
	Eigen::Matrix3f current_rotation = pose.wrist_pose.rotation();

	for (int i = 0; i < 3; i++)
	{
		if (std::acosf(prev_rotation.col(i).dot(current_rotation.col(i))) >= params->rotational_tolerance_movement)
			return false;
	}

	if (std::abs(prev_pose.thumb_adduction - pose.thumb_adduction) >= params->rotational_tolerance_movement)
		return false;

	if (std::abs(prev_pose.finger_spreading - pose.finger_spreading) >= params->rotational_tolerance_movement)
		return false;

	for (int i = 1; i < pose.finger_bending.size(); i++)
	{
		if (std::abs(prev_pose.finger_bending[i].first - pose.finger_bending[i].first) >= params->rotational_tolerance_movement)
			return false;

		if (std::abs(prev_pose.finger_bending[i].second - pose.finger_bending[i].second) >= 2 * params->rotational_tolerance_movement)
			return false;

		if (std::abs(pose.finger_bending[i].first) >= params->rotational_tolerance_flateness)
			return false;

		if (std::abs(pose.finger_bending[i].second) >= 2 * params->rotational_tolerance_flateness)
			return false;
	}

	return true;
}

void gradient_decent_optimization_common::scale_bones(const visual_input& input, const img_segment& seg, bool smaller)
{
	const auto& pose = best->pose;
	float scaling;
	if (smaller)
		scaling = std::max(pose.hand_kinematic_params.min_bone_scaling, 0.4f * (pose.hand_kinematic_params.min_bone_scaling - 1.f) + pose.bone_scaling(0, 2));
	else
		scaling = std::min(pose.hand_kinematic_params.max_bone_scaling, 0.4f * (pose.hand_kinematic_params.max_bone_scaling - 1.f) + pose.bone_scaling(0, 2));

	hand_pose_particle_instance::Ptr next = best;

	auto constrain_and_update = [&]() {
		scaling = std::max(best->pose.hand_kinematic_params.min_bone_scaling,
			std::min(best->pose.hand_kinematic_params.max_bone_scaling, scaling));

		auto next_pose = next->pose;

		for (int col = 0; col < next_pose.bone_scaling.cols(); col++)
			for (int row = 0; row < next_pose.bone_scaling.rows(); row++)
				next_pose.bone_scaling(row, col) = scaling;

		next = std::make_shared<hand_pose_particle_instance>(std::move(next_pose), best->net_eval, timestamp);
		next->updates = best->updates;
	};

	constrain_and_update();

	double sum_step = 0;
	double sum_weight = 0;
	for (const auto& objective : objectives)
	{
		auto step_weight = objective->optimal_stepsize(input, seg, *best, *next);
		double step_size = step_weight.first;
		double weight = step_weight.second;
		if (!std::isnan(step_size) && !std::isinf(step_size))
		{
			sum_step += step_size * weight;
			sum_weight += weight;
		}
	}

	if (sum_weight > 0)
	{

		scaling = sum_step / sum_weight * (scaling - pose.bone_scaling(0, 2)) + pose.bone_scaling(0, 2);

		constrain_and_update();

		evaluate_objectives(input, seg, *next);

		if (next->error < best->error)
			best = next;
	}
}

float gradient_decent_optimization_common::get_hand_certainty(const visual_input& input, const img_segment& seg, const hand_pose_particle_instance& particle) const
{
	return 1.f / 3 * bell_curve(particle.error) +
		1.f / 3 * bell_curve(quality_fill_mask(1.f).evaluate(input, particle, seg).first) +
		1.f / 3 * seg.max_net_certainty;
}

float gradient_decent_optimization_common::get_hand_orientation_fit(const visual_input& input, const img_segment& seg, const hand_pose_particle_instance& particle) const {
	float palm_fit = 1.f, key_point_fit = 1.f;
	if(!back_palm_orientation.isZero())
		palm_fit = particle.pose.wrist_pose.rotation().col(2).dot(back_palm_orientation) + 0.5f;
	if (particle.net_eval)
		key_point_fit = bell_curve(quality_2d_key_points(1).evaluate(input, particle, seg).first);

	return 0.5f * palm_fit + 0.5f * key_point_fit;
}

hand_pose_particle_instance::Ptr gradient_decent_optimization_common::extrapolate_pose_and_store(const hand_instance& hand,
	bool right_hand)
{
	extrapolated_pose = nullptr;
	std::vector<hand_pose_particle_instance::Ptr> prev_particles;
	
	{
		std::lock_guard<std::mutex> lock(hand.update_mutex);

		auto iter = hand.observation_history.rbegin();
		for (int i = 0; i < 2 && iter != hand.observation_history.rend(); ++iter)
		{
			if ((*iter)->particles[right_hand])
				prev_particles.push_back((*iter)->particles[right_hand]);
		}
	}

	if (prev_particles.size() == 1)
	{
		extrapolated_pose = std::make_shared<hand_pose_particle_instance>(*prev_particles.front());
		extrapolated_pose->time_seconds = timestamp;
	}
	else if (prev_particles.size() >= 2)
	{
		hand_pose_18DoF::Vector15f last = prev_particles.front()->pose.get_parameters();
		
		float elapsed = std::chrono::duration<float>(
			timestamp - prev_particles.front()->time_seconds).count();
		float prev_elapsed = std::chrono::duration<float>(
			prev_particles.front()->time_seconds - prev_particles[1]->time_seconds).count();

		if (elapsed < 0.002f)
			elapsed = 1 / 60.f;

		if (elapsed > 1.f)
			elapsed = 1.f;

		if (prev_elapsed < 0.002f)
			prev_elapsed = 1 / 60.f;

		hand_pose_18DoF::Vector15f parameters = last +
			elapsed / prev_elapsed * (last - prev_particles[1]->pose.get_parameters());

		Eigen::Quaternionf rotation = prev_particles[0]->rotation;

		if (prev_particles[0]->rotation.angularDistance(prev_particles[1]->rotation) > quality_criterion::EPSILON)
		{
			Eigen::Quaternionf prev_rotation = prev_particles[1]->rotation * prev_particles[0]->rotation.inverse();
			float dt = 1 + elapsed / prev_elapsed;

			Eigen::AngleAxisf angle_axis;
			angle_axis = prev_rotation;
			angle_axis.angle() *= dt;

			float speed = std::abs(angle_axis.angle()) / 
				std::chrono::duration<float>(timestamp - prev_particles[1]->time_seconds).count();
			
			if (speed > dynamic_params->speed_rotation / 2.f)
				angle_axis.angle() *= dynamic_params->speed_rotation / speed / 2.f;

			rotation = angle_axis * prev_particles[0]->rotation;

			//float angle = prev_particles[1]->rotation.angularDistance(rotation);
			//if (angle > 1.f)
			//	std::cout << "too fast rotation: " << angle << std::endl;
		}

		extrapolated_pose = std::make_shared<hand_pose_particle_instance>(
			hand_pose_18DoF(prev_particles.front()->pose.hand_kinematic_params, parameters, rotation, prev_particles.front()->pose.right_hand),
			nullptr,
			timestamp);
	}

	for (auto& objective : objectives)
	{
		auto ptr = std::dynamic_pointer_cast<quality_acceleration>(objective);
		if (ptr)
			ptr->reference = extrapolated_pose;
	}

	return extrapolated_pose;
}

void gradient_decent_optimization_common::decent_without_net_eval(const visual_input& input,
	const img_segment& seg,
	const hand_pose_particle_instance::Ptr& prev_particle,
	const Eigen::Vector3f& translation)
{
	std::vector<hand_pose_particle_instance::Ptr> seeds;
	seeds.push_back(std::make_shared<hand_pose_particle_instance>(
		prev_particle->pose,
		nullptr,
		timestamp,
		prev_particle->key_points,
		prev_particle->surface_distances));

	hand_pose_18DoF::Vector15f new_parameters = prev_particle->pose.get_parameters();
	new_parameters.head(3) += translation;

	const auto& wrist_pose = prev_particle->pose.wrist_pose;

	seeds.push_back(std::make_shared<hand_pose_particle_instance>(
		hand_pose_18DoF::combine(Eigen::Translation3f(translation) * wrist_pose, prev_particle->pose),
		nullptr,
		timestamp,
		Eigen::Affine3f(Eigen::Translation3f(translation)) * prev_particle->key_points));

	float radius = quality_key_points_below_surface::finger_radius(prev_particle->pose);
	for (int k = 0; k < prev_particle->key_points.cols(); k++)
		seeds.back()->surface_distances[k] = quality_key_points_below_surface::distance_to_surface(input, seg, prev_particle->key_points.col(k), radius);

	best = nullptr;
	best_starting_point(input, seg, seeds, 0);

}

void gradient_decent_optimization_common::bend_fingers(const visual_input& input, const img_segment& seg)
{
	for (int finger = 1; finger < 5; finger++) {
		std::vector<std::pair<float, float>> fingers = best->pose.finger_bending;
		fingers.at(finger) = std::make_pair(-M_PI_4, -M_PI_4);
		Eigen::Matrix3Xf key_points = best->key_points;

		hand_pose_18DoF new_pose(
			best->pose.hand_kinematic_params,
			best->pose.wrist_pose,
			fingers,
			best->pose.thumb_adduction,
			best->pose.finger_spreading,
			best->pose.right_hand,
			best->pose.bone_scaling);

		key_points.block<3, 4>(0, 1 + 4 * finger) = new_pose.get_key_points((finger_type)finger);

		auto new_particle = std::make_shared<hand_pose_particle_instance>(std::move(new_pose), best->net_eval, timestamp, key_points);
		new_particle->updates = best->updates;
		evaluate_objectives(input, seg, *new_particle);

		for (int d = finger * 2 + 7; d < finger * 2 + 9; d++)
		{
			float inc = d <= 3 ? 0.01f : 0.2f;
			auto next_inc = increment_parameter(input, seg, *new_particle, d, -inc);
			auto next = next_inc.first;

			double sum_step = 0;
			double sum_weight = 0;
			for (const auto& objective : objectives)
			{
				auto step_weight = objective->optimal_stepsize(input, seg, *new_particle, *next);
				double step_size = step_weight.first;
				double weight = step_weight.second;
				if (!std::isnan(step_size) && !std::isinf(step_size))
				{
					sum_step += step_size * weight;
					sum_weight += weight;
				}
			}

			if (sum_weight == 0)
				continue;

			if (std::abs(sum_step / sum_weight * next_inc.second) < quality_criterion::EPSILON)
				continue;

			next = increment_parameter(input, seg, *new_particle, d, sum_step / sum_weight * next_inc.second).first;

			evaluate_objectives(input, seg, *next);

			if (next->error < new_particle->error) {
				new_particle = next;
			}
		}

		if (new_particle->error < best->error)
			best = new_particle;
	}
}

std::vector<hand_pose_particle_instance::Ptr> gradient_decent_optimization_common::generate_seeds(const visual_input& input,
	const hand_pose_estimation& hand_pose_est,
	const hand_instance& hand,
	bool right_hand)
{
	img_segment::Ptr seg_ptr = nullptr;
	img_segment::Ptr prev_seg_ptr = nullptr;

	{
		std::lock_guard<std::mutex> lock(hand.update_mutex);
		for (auto iter = hand.observation_history.rbegin(); iter != hand.observation_history.rend(); ++iter)
			if ((*iter)->timestamp == timestamp)
			{
				seg_ptr = *iter;
			}
			else if (seg_ptr)
			{
				prev_seg_ptr = *iter;
				break;
			}
	}

	if (!seg_ptr)
		throw std::runtime_error("Associated segment not found");

	auto& seg = *seg_ptr;
	
	std::vector<hand_pose_particle_instance::Ptr> seeds;

	if (!extrapolated_pose || !prev_seg_ptr)
	{
		for (int j = 0; j < 2; j++)
		{
			if (!seg.net_evals[j])
				continue;

			auto current_est = hand_pose_est.initial_estimate(input, seg, *seg.net_evals[j], right_hand);
			if (current_est->wrist_pose.rotation().col(2).dot(back_palm_orientation) < 0.f)
				current_est->wrist_pose = current_est->wrist_pose * Eigen::Affine3f(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()));


			seeds.push_back(
				std::make_shared<hand_pose_particle_instance>(*current_est
					, seg.net_evals[j], timestamp));

			// hand parallel to camera image plane
			const auto& pose = current_est->wrist_pose;
			auto rotation = Eigen::Quaternionf::FromTwoVectors(pose.rotation().col(2), Eigen::Vector3f::UnitZ());
			if (rotation.angularDistance(Eigen::Quaternionf::Identity()) > M_PI_2)
				rotation = Eigen::Quaternionf::FromTwoVectors(pose.rotation().col(2), -Eigen::Vector3f::UnitZ());
			Eigen::Affine3f transform = Eigen::Translation3f(pose.translation() + Eigen::Vector3f(0.f, 0.f, seeds.back()->key_points.col(9).z() - seeds.back()->key_points.col(0).z())) *
				Eigen::Affine3f(rotation * pose.rotation());

			seeds.emplace_back(std::make_shared<hand_pose_particle_instance>(
				hand_pose_18DoF(hand_pose_18DoF::combine(transform, *current_est)),
				seg.net_evals[j],
				timestamp));
		}

		float radius = quality_key_points_below_surface::finger_radius(seeds[0]->pose);
		for (auto& particle : seeds)
			for (int i = 0; i < particle->key_points.cols(); i++)
				particle->surface_distances[i] = quality_key_points_below_surface::distance_to_surface(input, seg, particle->key_points.col(i), radius);

	}
	else
	{
		using particle_vec = std::vector<hand_pose_particle_instance::Ptr>;

		const img_segment& prev_seg = *prev_seg_ptr;
		const auto& prev_particle = prev_seg.particles[right_hand];

		auto generate_seeds = [&](net_evaluation::ConstPtr net_eval) // enforce copy to avoid race conditions
		{
			particle_vec seeds;

			auto current_est = hand_pose_est.initial_estimate(input, seg, *net_eval, right_hand);

			Eigen::Vector3f translation;
			if (seg.prop_3d && prev_seg.prop_3d)
				translation = seg.prop_3d->centroid - prev_seg.prop_3d->centroid;
			else
				translation = prev_particle->pose.wrist_pose.translation() - current_est->wrist_pose.translation();

			seeds.push_back(std::make_shared<hand_pose_particle_instance>(extrapolated_pose->pose,
				net_eval,
				timestamp,
				extrapolated_pose->key_points));

			seeds.emplace_back(std::make_shared<hand_pose_particle_instance>(*current_est, net_eval, timestamp));
			// current wrist pose + prev finger poses
			seeds.emplace_back(std::make_shared<hand_pose_particle_instance>(
				hand_pose_18DoF::combine(current_est->wrist_pose, prev_particle->pose)
				, net_eval, timestamp));

			/*seeds.emplace_back(std::make_shared<hand_pose_18DoF>(
				current_est->hand_kinematic_params,
				current_est->wrist_pose,
				std::vector<std::pair<float, float>>(5, std::make_pair(-M_PI_4, -M_PI_4)),
				extrapolated_pose->pose->thumb_adduction,
				extrapolated_pose->pose->finger_spreading,
				i,
				extrapolated_pose->pose->bone_scaling));*/

				// current wrist position + prev wrist orientation + prev finger poses
			seeds.emplace_back(
				std::make_shared<hand_pose_particle_instance>(
					hand_pose_18DoF::combine(
						Eigen::Translation3f(current_est->wrist_pose.translation() + translation) * current_est->wrist_pose,
						prev_particle->pose)
					, net_eval, timestamp));

			return seeds;
		};

		if (seg.net_evals[0] && seg.net_evals[1])
		{
			quality_2d_key_points quality_net(1.);
			std::vector<particle_vec> seed_variants;

			for (int j = 0; j < 2; j++)
			{
				seed_variants.emplace_back(generate_seeds(seg.net_evals[j]));

				const auto& p = *prev_particle;

				seed_variants[j].push_back(std::make_shared<hand_pose_particle_instance>(p.pose,
					seg.net_evals[j],
					timestamp,
					p.key_points));
			}

			for (int k = 0; k < std::max(seed_variants[0].size(), seed_variants[1].size()); k++)
			{
				double error_left = quality_net.evaluate(input, *seed_variants[0][k], seg).first;
				double error_right = quality_net.evaluate(input, *seed_variants[1][k], seg).first;

				if (std::isfinite(error_left) && error_left < error_right)
					seeds.push_back(seed_variants[0][k]);
				else
					seeds.push_back(seed_variants[1][k]);
			}
		}
		else
		{
			auto& net_eval = seg.net_evals[0] ? seg.net_evals[0] : seg.net_evals[1];
			seeds = generate_seeds(net_eval);

			const auto& p = *prev_particle;

			seeds.push_back(std::make_shared<hand_pose_particle_instance>(p.pose,
				net_eval,
				timestamp,
				p.key_points));
		}

		float radius = quality_key_points_below_surface::finger_radius(prev_particle->pose);
		for (auto& particle : seeds)
			for (int i = 0; i < particle->key_points.cols(); i++)
				particle->surface_distances[i] = quality_key_points_below_surface::distance_to_surface(input, seg, particle->key_points.col(i), radius);
	}

	return seeds;
}

/////////////////////////////////////////////////////////////
//
//
//  Class: gradient_decent_optimization_full
//
//
/////////////////////////////////////////////////////////////



gradient_decent_optimization_full::gradient_decent_optimization_full(hand_pose_estimation& hand_pose_est,
	std::vector<quality_criterion::Ptr> objectives,
	Eigen::Vector3f back_palm_orientation,
	gradient_decent_parameters::ConstPtr params,
	hand_dynamic_parameters::ConstPtr dynamic_params)
	:
	gradient_decent_optimization_common(std::move(objectives), std::move(back_palm_orientation), std::move(params), std::move(dynamic_params)),
	hand_pose_est(hand_pose_est)
{
}

gradient_decent_optimization_full::gradient_decent_optimization_full(hand_pose_estimation& hand_pose_est,
	gradient_decent_optimization_common common)
		:
	gradient_decent_optimization_full(hand_pose_est, std::move(common.objectives), std::move(common.back_palm_orientation), std::move(common.params), std::move(common.dynamic_params))
{
}

void gradient_decent_optimization_full::update(std::chrono::duration<float> timestamp)
{
	//std::cout << timestamp - this->timestamp << std::endl;
	prev_timestamp = this->timestamp;
	this->timestamp = timestamp;
}

hand_pose_particle_instance::Ptr gradient_decent_optimization_full::update(const visual_input& input,
	const img_segment& seg,
	const std::vector<hand_pose_particle_instance::Ptr>& seeds,
	const hand_pose_particle_instance::Ptr& prev_particle)
{
	if (seeds.empty())
		throw std::exception("No initial poses provided");

	best = nullptr;
	this->prev_particle = prev_particle;
	best_starting_point(input, seg, seeds);

	auto prev_step_best = best;
	bool disturbed_fingers = false;
	for (int step = 0; step < params->max_steps; step++)
	{
		auto next = gradient_step(input, seg, best, step % 2);
		if (next->error < best->error)
			best = next;

		if (can_scale_bones(*best))
			for (int i = 0; i < 3; i++)
				scale_bones(input, seg, step % 2);

		if (best->error + params->min_improvement > prev_step_best->error)
			if (disturbed_fingers)
				return best;
			else
			{
				disturbed_fingers = true;
				bend_fingers(input, seg);
			}

		prev_step_best = best;
	}

	return best;
}

void gradient_decent_optimization_full::update(const visual_input& input, hand_instance& hand)
{
	img_segment::Ptr seg_ptr = nullptr;
	img_segment::Ptr prev_seg_ptr = nullptr;

	{
		std::lock_guard<std::mutex> lock(hand.update_mutex);
		for (auto iter = hand.observation_history.rbegin(); iter != hand.observation_history.rend(); ++iter)
			if ((*iter)->timestamp == timestamp)
			{
				seg_ptr = *iter;
			}
			else if(seg_ptr)
			{
				prev_seg_ptr = *iter;
				break;
			}
	}
	
	if (!seg_ptr)
		return;

	auto& seg = *seg_ptr;

	if (seg.net_evals[0] || seg.net_evals[1])
		return; // old segment already evaluated

	if (seg.mask.empty())
		return; // skip out of sight segment
	

	if (input.has_cloud())
	{
		try {
			seg.compute_properties_3d(input);
		}
		catch (const std::exception&)
		{
			// segment cloud empty

			if (prev_seg_ptr)
			{
				const img_segment& prev_seg = *prev_seg_ptr;
				for (int i = 0; i < 2; i++)
				{
					seg.particles[i] = std::make_shared<hand_pose_particle_instance>(prev_seg.particles[i]->pose,
						prev_seg.particles[i]->net_eval,
						timestamp,
						prev_seg.particles[i]->key_points,
						prev_seg.particles[i]->surface_distances);
				}
			}

			hand.certainty_score = (1 - params->learning_rate) * hand.certainty_score;

			return;
		}
	}

	if (hand.certainty_score < hand_pose_est.params.hand_probability_threshold)
		return;

	std::vector<hand_pose_particle_instance::Ptr> prev_particles(2);
	bool nets_evaluated = false;

	if (!prev_seg_ptr)
	{
		for (int i = 0; i < 2; i++)
			hand_pose_est.estimate_keypoints(input.img, seg, i);
		nets_evaluated = true;
	}
	else {
		const img_segment& prev_seg = *prev_seg_ptr;
		seg.max_net_certainty = (1 - params->best_net_eval_certainty_decay) * prev_seg.max_net_certainty;
		
		if (cv::norm(seg.palm_center_2d - prev_seg.palm_center_2d) < 0.12 * std::max(seg.bounding_box.width, seg.bounding_box.height))
		{
			auto start = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < 2; i++) {
				hand_pose_est.estimate_keypoints(input.img, seg, i);
				seg.max_net_certainty = std::max(seg.max_net_certainty, seg.net_evals[i]->certainty);
			}

			// std::cout << "net evaluation  " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start) << std::endl;
			nets_evaluated = true;
		}
		prev_particles = prev_seg.particles;
	}

	std::vector<float> certainties(2, 0.f);
	std::vector<float> orientations(2, 0.f);

	using particle_vec = std::vector<hand_pose_particle_instance::Ptr>;

	for (int i = 0; i < 2; i++)
	{
		if (std::abs(i - hand.right_hand) > (1 - hand_pose_est.params.hand_probability_threshold))
			continue;

		auto extrapolated_pose = extrapolate_pose_and_store(hand, i);

		if (nets_evaluated)
		{
			seg.particles[i] = update(input, seg, generate_seeds(input, hand_pose_est, hand, i));

			certainties[i] = get_hand_certainty(input, seg, *seg.particles[i]);
			orientations[i] = get_hand_orientation_fit(input, seg, *seg.particles[i]);
		}
		else // !nets_evaluated
		{
			if (!prev_particles[i])
				continue;

			const img_segment& prev_seg = *prev_seg_ptr;

			Eigen::Vector3f translation;
			const auto& wrist_pose = prev_particles[i]->pose.wrist_pose;
			if (seg.prop_3d && prev_seg.prop_3d)
				translation = seg.prop_3d->centroid - prev_seg.prop_3d->centroid;
			else
			{
				float z = wrist_pose.translation().z();

				Eigen::MatrixXf A = input.img_projection.block<3, 2>(0, 0);
				Eigen::Vector3f b(seg.palm_center_2d.x, seg.palm_center_2d.y, 1.);
				b -= z * input.img_projection.col(2) + input.img_projection.col(3);

				Eigen::Vector2f x = A.householderQr().solve(b);
				translation = Eigen::Vector3f(x.x(), x.y(), z) - wrist_pose.translation();
			}

			decent_without_net_eval(input, seg, prev_particles[i], translation);

			for (int i = 0; i < 3; i++)
				best = gradient_step(input, seg, best, i % 2);
			seg.particles[i] = best;

			certainties[i] = prev_seg.hand_certainty;
			orientations[i] = get_hand_orientation_fit(input, seg, *seg.particles[i]);
		}

	}

	float left_hand_certainty = 0.75f * certainties[0] + 0.25f * orientations[0];
	float right_hand_certainty = 0.75f * certainties[1] + 0.25f * orientations[1];

	float certainty_inc = (right_hand_certainty / (left_hand_certainty + 0.00001) - 1) * std::max(left_hand_certainty, right_hand_certainty);


	hand.right_hand = (1 - params->learning_rate) * hand.right_hand + params->learning_rate *
		std::max(0.f, std::min(1.f, hand.right_hand + certainty_inc));

	double certainty = std::max(certainties[0], certainties[1]);
	seg.hand_certainty = certainty;
	hand.certainty_score = (1 - params->learning_rate) * hand.certainty_score + params->learning_rate * certainty;

	//for (const auto& objective : objectives)
	//{
	//	auto pair = objective->evaluate(input, *seg.particles[hand.right_hand > 0.5f], seg);
	//	std::cout << pair.first << " " << pair.second << std::endl;
	//}
	//std::cout << std::endl;

	//if (objectives.front()->evaluate(input, *seg.particles[1], seg).first > 6)
	//	std::cout << "jerk" << std::endl;

	std::unique_lock<std::mutex> lock(hand.update_mutex);
	if (hand.poses.size() && certainties[0] == certainties[1])
	{
		bool right_hand = hand.poses.back().pose.right_hand;
		
		seg.model_box_2d = seg.particles[right_hand]->get_box(input);
		seg.model_box_3d = seg.particles[right_hand]->get_box();

		lock.unlock();
		hand.add_or_update(*seg.particles[right_hand]);
	}
	else if (certainties[0] == 0 && certainties[1] == 0)
		return;
	else
	{
		
		seg.model_box_2d = seg.particles[hand.right_hand > 0.5f]->get_box(input);
		seg.model_box_3d = seg.particles[hand.right_hand > 0.5f]->get_box();

		lock.unlock();
		hand.add_or_update(hand.right_hand < 0.5f ? *seg.particles[0] : *seg.particles[1]);
	}
}



/////////////////////////////////////////////////////////////
//
//
//  Class: gradient_decent_optimization_incremental
//
//
/////////////////////////////////////////////////////////////

gradient_decent_optimization_incremental::gradient_decent_optimization_incremental(
	std::vector<quality_criterion::Ptr> objectives, Eigen::Vector3f back_palm_orientation,
	gradient_decent_parameters::ConstPtr params, hand_dynamic_parameters::ConstPtr dynamic_params,
	visual_input::ConstPtr input,
	hand_instance::Ptr hand,
	bool right_hand)
	:
	gradient_decent_optimization_common(std::move(objectives), std::move(back_palm_orientation), params, dynamic_params),
	input(input),
	hand(hand),
	seg(hand->observation_history.back()),
	right_hand(right_hand),
	step(0),
	hand_certainty(hand->certainty_score),
	right_hand_certainty(hand->right_hand),
	position_sim(1.f),
	finished(false),
	net_evaluated(false)
{
	//if (!seg->prop_3d)
	//	throw std::runtime_error("!seg->prop_3d");
	if (hand->observation_history.back()->timestamp != input->timestamp_seconds)
		throw std::exception("Timestamps of input and last segment do not match");

	timestamp = input->timestamp_seconds;

	std::unique_lock<std::mutex> lock_hand(hand->update_mutex);
	if (hand->observation_history.size() >= 2) {
		const img_segment& prev_seg = **(++hand->observation_history.rbegin());

		lock_hand.unlock();
		prev_particle = prev_seg.particles[right_hand];
		auto seg = *this->seg;

		if (prev_particle) {
			const auto& wrist_pose = prev_particle->pose.wrist_pose;
			if (seg.prop_3d && prev_seg.prop_3d)
			{
				translation = seg.prop_3d->centroid - prev_seg.prop_3d->centroid;
				position_sim = bell_curve(translation.norm(), 0.1f * seg.prop_3d->bounding_box.diagonal().norm());
			}
			else
			{
				float z = wrist_pose.translation().z();

				Eigen::MatrixXf A = input->img_projection.block<3, 2>(0, 0);
				Eigen::Vector3f b(seg.palm_center_2d.x, seg.palm_center_2d.y, 1.);
				b -= z * input->img_projection.col(2) + input->img_projection.col(3);

				Eigen::Vector2f x = A.householderQr().solve(b);
				translation = Eigen::Vector3f(x.x(), x.y(), z) - wrist_pose.translation();

				position_sim = bell_curve(cv::norm(seg.palm_center_2d - prev_seg.palm_center_2d), 
					0.15 * std::max(seg.bounding_box.width, seg.bounding_box.height));
			}

			prev_timestamp = prev_particle->time_seconds;
			extrapolate_pose_and_store(*hand, right_hand);

			hand_pose_18DoF::Vector15f new_parameters = prev_particle->pose.get_parameters();
			new_parameters.head(3) += translation;

			seg.particles[right_hand] = std::make_shared<hand_pose_particle_instance>(
				hand_pose_18DoF::combine(Eigen::Translation3f(translation) * wrist_pose, prev_particle->pose),
				nullptr,
				timestamp,
				Eigen::Affine3f(Eigen::Translation3f(translation)) * prev_particle->key_points);

			seg.particles[right_hand]->hand_certainty = prev_seg.hand_certainty;
			seg.particles[right_hand]->hand_orientation = prev_particle->hand_orientation;
			seg.max_net_certainty = seg.max_net_certainty;

			if (hand->right_hand > 0.5f && right_hand || hand->right_hand <= 0.5f && !right_hand)
			{
				seg.model_box_2d = seg.particles[right_hand]->get_box(*input);
				seg.model_box_3d = seg.particles[right_hand]->get_box();

				hand->add_or_update(*seg.particles[right_hand]);
			}
		}
		else // !prev_particle
		{
			step = 1;
		}
	} // !observation_history.size()
	else
		step = 1;

}


gradient_decent_optimization_incremental::gradient_decent_optimization_incremental(
	gradient_decent_optimization_common common, 
	visual_input::ConstPtr input, 
	hand_instance::Ptr hand, 
	bool right_hand)
	:
	gradient_decent_optimization_incremental(std::move(common.objectives), std::move(common.back_palm_orientation), std::move(common.params), std::move(common.dynamic_params), input, hand, right_hand)
{
}


float gradient_decent_optimization_incremental::get_priority() const
{
	if (step == 0) // neural net evaluations not yet avialable, start evaluating poses with lowest priority
		return 100.f * hand->certainty_score * position_sim;

	/*
	 * priorities (ordered in descending importance)
	 * new hand
	 * no previous pose
	 * likely a hand
	 * at similar position
	 * only few updates done for this frame
	 */
	
	float prio = (4.f - 2.f * hand->certainty_score - position_sim);
	if (!prev_particle)
		prio = 0.5f;
	else if (!prev_particle->net_eval)
		prio = 1.f;

	return prio * step; // lower priority in each step

}

int gradient_decent_optimization_incremental::update()
{
	//auto start = std::chrono::high_resolution_clock::now();
	
	hand_pose_particle_instance::Ptr particle = best;

	int current_step = step.fetch_add(1);

	if (!current_step)
	{
		decent_without_net_eval(*input, *seg, prev_particle, translation);
		particle = best;
	}
	else if (!seg->net_evals[right_hand])
	{
		particle = best;
	}
	else if (current_step == 1)
	{
		if (best && seg->net_evals[right_hand] && !best->net_eval)
			best = nullptr;

		if (prev_particle)
			particle = seeds[0];
		else
		{
			for (auto& seed : seeds)
			{
				evaluate_objectives(*input, *seg, *seed);
				if (!best || best->error > seed->error)
					best = seed;
			}

			particle = best;
		}
	}
	else if (prev_particle && current_step == 2)
		particle = seeds[4];
	else if (current_step == 3 && can_scale_bones(*best))
	{
		for (int i = 0; i < 3; i++)
			scale_bones(*input, *seg, current_step % 2);
		particle = best;
	}
	else if (current_step == 4 || current_step <= 4 && finished)
	{
		bend_fingers(*input, *seg);
		particle = best;
	}

	if (!current_step && net_evaluated)
		return current_step;
	
	for (int step = 0; step < 3; step++)
	{
		auto next = gradient_step(*input, *seg, particle, step % 2, current_step < 3 ? 6 : 18);

		if (particle == best && next->error + params->min_improvement > best->error)
		{
			finished = true;
			break;
		}

		if (next->error < particle->error)
			particle = next;
		else
			break;
	}

	//std::cout << "gd  " << current_step << " " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;
	
	if (!current_step && net_evaluated)
		return current_step;
	
	if (!best || particle->error < best->error)
		best = particle;

	best->hand_certainty = get_hand_certainty(*input, *seg, *best);
	best->hand_orientation = get_hand_orientation_fit(*input, *seg, *best);

	{
		std::unique_lock<std::mutex> lock(hand->update_mutex);

		seg->particles[right_hand] = best;

		if (hand->poses.size() && hand->poses.back().time_seconds != timestamp)
			return current_step;

		std::vector< hand_pose_particle_instance::Ptr> particles = seg->particles;

		if (right_hand && !particles[0])
			particles[0] = particles[1];
		else if (!right_hand && !particles[1])
			particles[1] = particles[0];


		float left_hand_certainty = 0.75f * particles[0]->hand_certainty + 0.25f * particles[0]->hand_orientation;
		float right_hand_certainty = 0.75f * particles[1]->hand_certainty + 0.25f * particles[1]->hand_orientation;

		float certainty_inc = (right_hand_certainty / (left_hand_certainty + 0.00001) - 1) * std::max(left_hand_certainty, right_hand_certainty);


		double certainty = std::max(particles[0]->hand_certainty, particles[1]->hand_certainty);
		seg->hand_certainty = certainty;

		if (current_step)
		{
			hand->right_hand = (1 - params->learning_rate) * this->right_hand_certainty + params->learning_rate *
				std::max(0.f, std::min(1.f, right_hand_certainty + certainty_inc));
			hand->certainty_score = (1 - params->learning_rate) * this->hand_certainty + params->learning_rate * certainty;
		}
		
		//for (const auto& objective : objectives)
		//{
		//	auto pair = objective->evaluate(input, *seg.particles[hand.right_hand > 0.5f], seg);
		//	std::cout << pair.first << " " << pair.second << std::endl;
		//}
		//std::cout << std::endl;

		//if (objectives.front()->evaluate(input, *seg.particles[1], seg).first > 6)
		//	std::cout << "jerk" << std::endl;

		if (particles[0]->hand_certainty == 0 && particles[1]->hand_certainty == 0)
			return current_step;
		else if (hand->poses.empty() || hand->right_hand > 0.5f && right_hand || hand->right_hand <= 0.5f && !right_hand)
		{
			seg->model_box_2d = best->get_box(*input);
			seg->model_box_3d = best->get_box();
			
			lock.unlock();
			hand->add_or_update(*best, true);
		}

	}

	return current_step;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: gradient_decent_scheduler
//
//
/////////////////////////////////////////////////////////////


gradient_decent_scheduler::gradient_decent_scheduler(gradient_decent_optimization_common prototype,
	int max_threads)
	:
	hand_pose_est(hand_kin_params, false),
	gd_proto(std::move(prototype)),
	terminate_flag(false),
	queue([](const prio_gd& lhs, const prio_gd& rhs) {return lhs.first > rhs.first; }),
	timestamp(0)
{
	if (!max_threads)
		return;

	if (max_threads == 1)
		decent_threads.emplace_back([this]()
			{
				hand_pose_est.init_neural_network();
				gradient_decent_optimization_full optimizer(hand_pose_est, gd_proto);

				while (true)
				{
					try {
						prio_gd pair;
						{
							std::unique_lock<std::mutex> lock(queue_mutex);

							if (terminate_flag)
								return;

							queue_condition_variable.wait(lock, [&] {return !queue.empty() || terminate_flag; });

							if (terminate_flag)
								return;

							pair = queue.top();
							queue.pop();
						}

						auto& gd = *pair.second;
						optimizer.update(gd.timestamp);
						optimizer.update(*gd.input, *gd.hand);
					}
					catch (const std::exception& e)
					{
						std::cout << e.what() << std::endl;
					}

				}
			});

	int count_nn_threads = max_threads >= 3 ? 2 : 1;
	int count_decent_threads = max_threads - count_nn_threads;

	for (int i = 0; i < count_nn_threads; i++)
	{
		nn_threads.emplace_back(hand_kin_params);
	}

	for (int i = 0; i < count_decent_threads; i++)
	{
		decent_threads.emplace_back([this]()
			{
				while (true)
				{
					try {
						prio_gd pair;
						{
							std::unique_lock<std::mutex> lock(queue_mutex);

							if (terminate_flag)
								return;

							queue_condition_variable.wait(lock, [&] {return !queue.empty() || terminate_flag; });

							if (terminate_flag)
								return;

							pair = queue.top();
							queue.pop();
						}

						auto& gd = *pair.second;
						try {
							int step = gd.update();
							if (step >= 1 && gd.timestamp == timestamp && !gd.finished)
							{
								std::unique_lock<std::mutex> lock(queue_mutex);
								queue.emplace(gd.get_priority(), pair.second);
							}
						}catch(const std::exception&){}
					}
					catch (const std::exception& e)
					{
						std::cout << e.what() << std::endl;
					}

				}

			});
	}
}

gradient_decent_scheduler::~gradient_decent_scheduler()
{
	terminate_flag = true;
	clear_queue();
	queue_condition_variable.notify_all();

	for (auto& thread : decent_threads)
		if (thread.joinable())
			thread.join();
}

void gradient_decent_scheduler::update(const visual_input::ConstPtr& input, const std::vector<hand_instance::Ptr>& hands)
{
	timestamp = input->timestamp_seconds;
	clear_queue();
	for (auto& est : nn_threads)
		est.clear_queue();

	using gdp = std::pair< gradient_decent_optimization_incremental::Ptr, gradient_decent_optimization_incremental::Ptr>;
	std::vector<gdp> gds;

	for (auto& hand : hands)
	{
		try
		{
			const auto& seg = *hand->observation_history.back();

			if (seg.timestamp != input->timestamp_seconds)
				continue;

			if (hand->certainty_score < hand_pose_est.params.hand_probability_threshold)
				continue;

			if (seg.mask.empty() || seg.hand_certainty == 1.f)
				continue;

			gdp pair;

			if (1 - hand->right_hand > hand_pose_est.params.hand_probability_threshold)
				pair.first = std::make_shared<gradient_decent_optimization_incremental>(gd_proto, input, hand, false);
			if (hand->right_hand > hand_pose_est.params.hand_probability_threshold)
				pair.second = std::make_shared<gradient_decent_optimization_incremental>(gd_proto, input, hand, true);

			if (pair.first || pair.second)
				gds.emplace_back(std::move(pair));
		}
		catch (...)
		{
			// skip invalid hands or hands which wer updated meanwhile
		}
	}

	float img_diag = std::powf(input->img.cols * input->img.cols + input->img.rows * input->img.rows, 0.5f);
	auto prio_net_eval = [&input, img_diag](const gdp& pair)
	{
		float l = 0.f, r = 0.f;
		float bb_center_dist;
		if (pair.first)
		{
			bb_center_dist = cv::norm(pair.first->seg->palm_center_2d - cv::Point2i(input->img.cols / 2, input->img.rows / 2) / img_diag);
			l = pair.first->hand_certainty * pair.first->position_sim + pair.first->step - 0.000001f * bb_center_dist;
		}
			

		if (pair.second)
		{
			bb_center_dist = cv::norm(pair.second->seg->palm_center_2d - cv::Point2i(input->img.cols / 2, input->img.rows / 2) / img_diag);
			r = pair.second->hand_certainty * pair.second->position_sim + pair.second->step - 0.000001f * bb_center_dist;
		}
		

		return std::max(r, l);
	};

	std::sort(gds.begin(), gds.end(), [&prio_net_eval](const gdp& lhs, const gdp& rhs)
		{
			return prio_net_eval(lhs) > prio_net_eval(rhs);
		});


	if (!nn_threads.empty())
	{
		auto nn_iter = nn_threads.begin();
		for (auto& pair : gds)
		{
			if (pair.first && (pair.first->step || pair.first->position_sim > gd_proto.params->position_similarity_threshold) || 
				pair.second && (pair.second->step || pair.second->position_sim > gd_proto.params->position_similarity_threshold))
			{
				auto callback = [this, pair, nn_iter](const net_evaluation::ConstPtr& net_eval)
				{
					auto hand = pair.first ? pair.first->hand : pair.second->hand;
					{
						std::lock_guard<std::mutex> lock_hand(hand->update_mutex);
						if (!hand->observation_history.back()->net_evals[0] || !hand->observation_history.back()->net_evals[1])
							return;
					}
					
					for (auto& gd : { pair.first, pair.second })
					{
									
						if (gd && timestamp == gd->timestamp)
						{
							{
								std::lock_guard<std::mutex> lock(queue_mutex);
								if (gd->net_evaluated)
									continue;

								gd->net_evaluated = true;
							}

							gd->seg->max_net_certainty = std::max(gd->seg->max_net_certainty, net_eval->certainty);
							
							{
								gd->seeds = gd->generate_seeds(*gd->input, *nn_iter, *hand, gd->right_hand);
								gd->step = 1;
							}
							
							std::lock_guard<std::mutex> lock(queue_mutex);
							queue.emplace(gd->get_priority(), gd);
							
						}
					}
					queue_condition_variable.notify_all();
				};

				auto seg = pair.first ? pair.first->seg : pair.second->seg;
				nn_iter->estimate_keypoints_async(input->img, seg, false, callback);

				nn_iter++;
				if (nn_iter == nn_threads.end())
					nn_iter = nn_threads.begin();

				nn_iter->estimate_keypoints_async(input->img, seg, true, callback);

				nn_iter++;
				if (nn_iter == nn_threads.end())
					nn_iter = nn_threads.begin();
			}
		}
	}

	if(decent_threads.empty())
	{
		gradient_decent_optimization_full sync_optimizer(hand_pose_est, gd_proto);
		sync_optimizer.update(gd_proto.prev_timestamp);
		sync_optimizer.update(gd_proto.timestamp);
		for (auto& hand : hands)
			sync_optimizer.update(*input, *hand);
	}
	else {
		std::lock_guard<std::mutex> lock(queue_mutex);
		for (auto& pair : gds)
		{
			for (auto& gd : { pair.first, pair.second })
				if (gd && (!gd->step || nn_threads.empty()))
					queue.emplace(gd->get_priority(), gd);
		}
		queue_condition_variable.notify_all();
	}
}

const hand_kinematic_parameters& gradient_decent_scheduler::get_hand_kinematic_parameters() const
{
	return hand_kin_params;
}

const hand_pose_parameters& gradient_decent_scheduler::get_hand_pose_parameters() const
{
	return hand_pose_est.params;
}

float gradient_decent_scheduler::get_hand_certainty_threshold() const
{
	return hand_pose_est.params.hand_probability_threshold;
}

void gradient_decent_scheduler::set_background(const Eigen::Hyperplane<float, 3>& plane)
{
	for(auto& objective : gd_proto.objectives)
	{
		auto obj_surface = std::dynamic_pointer_cast<quality_boundary_surface>(objective);
		if (obj_surface)
			obj_surface->plane = plane;
	}
}

void gradient_decent_scheduler::set_back_palm_orientation(const Eigen::Vector3f& normal)
{
	gd_proto.back_palm_orientation = normal;
}

void gradient_decent_scheduler::clear_queue()
{
	std::lock_guard<std::mutex> lock(queue_mutex);
	while (!queue.empty())
		queue.pop();
}
}

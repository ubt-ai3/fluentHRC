// hand_pose_estimation.cpp : Hiermit werden die exportierten Funktionen für die DLL definiert.
//

#include "framework.h"

#include <fstream>



#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "hand_pose_estimation.h"

#include <pcl/common/impl/accumulators.hpp>

namespace hand_pose_estimation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: simon_net_parameters
//
//
/////////////////////////////////////////////////////////////

simon_net_parameters::simon_net_parameters()
{
	filename_ = std::string("simon_net_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else
	{
		proto_file = "assets/network_models/pose_deploy.prototxt";
		weights_file = "assets/network_models/pose_iter_102000.caffemodel";
		n_points = 21;
	}
}

simon_net_parameters::~simon_net_parameters()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const simon_net_parameters& simon_net_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(simon_net_params);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: mueller_net_parameters
//
//
/////////////////////////////////////////////////////////////

mueller_net_parameters::mueller_net_parameters()
{
	filename_ = std::string("mueller_net_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else
	{
		proto_file = "assets/network_models/merged_net.prototxt";
		weights_file = "assets/network_models/merged_snapshot_iter_300000.caffemodel";
		n_points = 21;
	}
}

mueller_net_parameters::~mueller_net_parameters()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const mueller_net_parameters& mueller_net_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(mueller_net_params);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: hand_pose_parameters
//
//
/////////////////////////////////////////////////////////////

hand_pose_parameters::hand_pose_parameters()
{
	filename_ = std::string("hand_pose_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else
	{
		keypoint_probability_threshold = 0.09f;
		hand_probability_threshold = 0.2f;
		hand_min_keypoints = 5;
		hand_min_dimension = 0.05f;
		palm_length = 0.075f;
		bone_lengths = {
			0.508825, 0.361438,0.31327,0.294646,
			1.06024,0.443465,0.297464,0.260544,
			1.01398,0.556376,0.376575,0.312149,
			0.998409,0.501625,0.310689,0.240564,
			0.959939,0.320599,0.227937,0.236825
		};
		roi_box_padding = 0.08f;
	}
}

hand_pose_parameters::~hand_pose_parameters()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const hand_pose_parameters& hand_pose_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(hand_pose_params);
}



/////////////////////////////////////////////////////////////
//
//
//  Class: hand_pose_estimation
//
//
/////////////////////////////////////////////////////////////

hand_pose_estimation::hand_pose_estimation(const hand_kinematic_parameters& hand_kinematic_params, bool load_nets)
	:
	hand_kinematic_params(hand_kinematic_params)
{
	if (load_nets)
		init_neural_network();
}

const skin_detector& hand_pose_estimation::get_skin_detection() const
{
	return skin_detect;
}

double hand_pose_estimation::estimate_keypoints(const cv::Mat& input,
	img_segment& hand_candidate,
	bool right_hand)
{
	//auto eval = evaluate_simon_net(input, hand_candidate, right_hand);
	//cv::Mat1b canvas;
	//eval->maps.at(0).convertTo(canvas, CV_8UC1, 255);
	//cv::imshow("wrist", canvas);
	//cv::waitKey(0);

	//return eval->certainty;
	//
	auto net_eval = evaluate_mueller_net(input, hand_candidate, right_hand);
	
	int output_height = mueller_net.output_h;
	int output_width = mueller_net.output_w;
	std::vector<cv::Point2i> points(mueller_net_params.n_points);
	Eigen::VectorXf certainties(points.size());

	int count = 0;
	double sum_prob = 0;

	auto eval_points = [&]() {
		const auto& box = net_eval->input_box;

		Eigen::Matrix3f transform(Eigen::Matrix3f::Identity());
		transform(0, 0) = box.width / (float)output_width;
		transform(1, 1) = box.height / (float)output_height;
		transform(0, 2) = box.x;
		transform(1, 2) = box.y;


		for (int n = 0; n < points.size(); n++)
		{
			// Probability map of corresponding body's part.
			//cv::Mat probMap(cv::Size(output_height, output_width), CV_32F, keypoints_heatmap.ptr(0, n));

			cv::Point maxLoc;
			double prob;

			cv::minMaxLoc(net_eval->maps.at(n), 0, &prob, 0, &maxLoc);
			prob = correct_heatmap_certainty(prob);

			if (prob < params.keypoint_probability_threshold)
			{
				points[n] = cv::Point2i(-1, -1);
				certainties(n) = 0.f;
				continue;
			}

			Eigen::Vector2f p_input = (transform * Eigen::Vector2f(maxLoc.x, maxLoc.y).homogeneous()).hnormalized();
			points[n] = cv::Point2i(p_input.x(), p_input.y());

			certainties(n) = prob;
			sum_prob += prob;
			count++;
		}

		net_eval->key_points_2d = std::move(points);
		net_eval->certainties = std::move(certainties);
		net_eval->certainty = net_eval->certainties.mean();
	};

	eval_points();

	//if (net_eval->certainty < params.hand_probability_threshold)
	//{
	//	auto simon_eval = evaluate_simon_net(input, hand_candidate, right_hand);
	//	simon_eval->left_hand_pose = std::move(net_eval->left_hand_pose);
	//	simon_eval->maps = fuse_heatmaps(simon_eval->maps, net_eval->maps);
	//	net_eval = simon_eval;
	//	hand_candidate.net_evals[!!right_hand] = net_eval;



	//	//		cv::Mat1b canvas;
	//	//net_eval->maps.at(0).convertTo(canvas, CV_8UC1, 255);
	//	//cv::imshow("wrist", canvas);
	//	//cv::waitKey(0);

	//	output_height = simon_net.output_h;
	//	output_width = simon_net.output_w;
	//	points = std::vector<cv::Point2i>(simon_net_params.n_points);
	//	certainties = Eigen::VectorXf(points.size());

	//	count = 0;
	//	sum_prob = 0;

	//	eval_points();
	//}

	hand_candidate.net_evals[!!right_hand] = net_eval;
	
	if (count >= params.hand_min_keypoints && sum_prob / count >= params.hand_probability_threshold)
	{
		return std::min(1., (sum_prob / count - params.hand_probability_threshold) / (0.9 - params.hand_probability_threshold));
	}
	else
	{
		return 0.;
	}
}

double hand_pose_estimation::estimate_keypoints(const cv::Mat& input,
	hand_instance& hand_candidate)
{
	double certainty_left = 0, certainty_right = 0;

	if (hand_candidate.right_hand < 0.9)
		certainty_left = estimate_keypoints(input, *hand_candidate.observation_history.back(), false);
	if (hand_candidate.right_hand > 0.1)
		certainty_right = estimate_keypoints(input, *hand_candidate.observation_history.back(), true);

	hand_candidate.right_hand = 0.9 * hand_candidate.right_hand + 0.1 *
		std::max(0., std::min(1., certainty_right / (certainty_left + 0.00000000001) - 0.5));

	double certainty = std::max(certainty_left, certainty_right);
	hand_candidate.observation_history.back()->hand_certainty = certainty;
	hand_candidate.certainty_score = 0.9 * hand_candidate.certainty_score + 0.1 * certainty;


	return certainty;
}

/*
* Implementation follows: https://nghiaho.com/?page_id=671
*/
Eigen::Affine3f hand_pose_estimation::fit_pose(const std::vector<Eigen::Vector3f>& observed_points,
	const std::vector<Eigen::Vector3f>& model_points,
	bool remove_outliers)
{
	if (observed_points.size() != model_points.size())
		throw std::invalid_argument("Different number of points provided");

	if (observed_points.size() < 3)
		throw std::invalid_argument("Insufficient number of points provided");

	Eigen::Vector3f observed_centroid = Eigen::Vector3f::Zero();
	Eigen::Vector3f model_centroid = Eigen::Vector3f::Zero();

	for (int i = 0; i < observed_points.size(); i++)
	{
		observed_centroid += observed_points[i];
		model_centroid += model_points[i];
	}

	observed_centroid /= observed_points.size();
	model_centroid /= model_points.size();

	Eigen::Matrix<float, 3, -1> observed_vectors = Eigen::Matrix<float, 3, -1>(3, observed_points.size());
	Eigen::Matrix<float, 3, -1> model_vectors = Eigen::Matrix<float, 3, -1>(3, model_points.size());

	for (int i = 0; i < observed_points.size(); i++)
	{
		observed_vectors.col(i) = observed_points.at(i) - observed_centroid;
		model_vectors.col(i) = model_points.at(i) - model_centroid;
	}

	Eigen::Matrix3f H = model_vectors * observed_vectors.transpose();
	Eigen::JacobiSVD svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();

	if (R.determinant() < 0)
	{
		Eigen::Matrix3f V = svd.matrixV();
		V.col(2) *= -1;
		R = V * svd.matrixU().transpose();
	}

	Eigen::Affine3f transform = Eigen::Translation3f(observed_centroid) *
		Eigen::Quaternionf(R) *
		Eigen::Translation3f(-1 * model_centroid);

	if (!remove_outliers || observed_points.size() == 3)
		return transform;

	// search for greatest increment of error and remove all following points
	std::vector<std::pair<int, float>> errors;

	for (int i = 0; i < observed_points.size(); i++)
	{
		errors.emplace_back(i, (observed_points[i] - transform * model_points[i]).norm());
	}

	std::sort(errors.begin(), errors.end(),
		[](const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
			return lhs.second < rhs.second;
		});

	float max_error_increment = 0.f;
	int end;
	for (int i = 2; i < observed_points.size() - 1; i++)
	{
		float increment = errors[i + 1].second - errors[i].second;
		if (max_error_increment < increment)
		{
			max_error_increment = increment;
			end = i + 1;
		}
	}

	std::vector<Eigen::Vector3f> new_observed_points;
	std::vector<Eigen::Vector3f> new_model_points;
	for (int i = 0; i < end; i++)
	{
		new_observed_points.push_back(observed_points[errors[i].first]);
		new_model_points.push_back(model_points[errors[i].first]);
	}

	return fit_pose(new_observed_points, new_model_points, false);
}


Eigen::Affine3f hand_pose_estimation::fit_pose(const Eigen::Matrix3Xf& observed_points,
	const Eigen::Matrix3Xf& model_points,
	bool remove_outliers)
{
	if (observed_points.cols() != model_points.cols())
		throw std::exception("Different number of points provided");

	if (observed_points.cols() < 3)
		throw std::exception("Insufficient number of points provided");


	Eigen::Vector3f observed_centroid = observed_points.rowwise().mean();
	Eigen::Vector3f model_centroid = model_points.rowwise().mean();

	Eigen::Matrix3Xf observed_vectors = Eigen::Matrix<float, 3, -1>(3, observed_points.cols());
	Eigen::Matrix3Xf model_vectors = Eigen::Matrix<float, 3, -1>(3, model_points.cols());

	for (int i = 0; i < observed_points.cols(); i++)
	{
		observed_vectors.col(i) = observed_points.col(i) - observed_centroid;
		model_vectors.col(i) = model_points.col(i) - model_centroid;
	}

	Eigen::Matrix3f H = model_vectors * observed_vectors.transpose();
	Eigen::JacobiSVD svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3f R = svd.matrixV() * svd.matrixU().transpose();

	if (R.determinant() < 0)
	{
		Eigen::Matrix3f V = svd.matrixV();
		V.col(2) *= -1;
		R = V * svd.matrixU().transpose();
	}

	Eigen::Affine3f transform = Eigen::Translation3f(observed_centroid) *
		Eigen::Quaternionf(R) *
		Eigen::Translation3f(-1 * model_centroid);

	if (!remove_outliers || observed_points.cols() == 3)
		return transform;

	// search for greatest increment of error and remove all following points
	std::vector<std::pair<int, float>> errors;

	for (int i = 0; i < observed_points.cols(); i++)
	{
		errors.emplace_back(i, (observed_points.col(i) - transform * model_points.col(i)).norm());
	}

	std::sort(errors.begin(), errors.end(),
		[](const std::pair<int, float>& lhs, const std::pair<int, float>& rhs) {
			return lhs.second < rhs.second;
		});

	float max_error_increment = 0.f;
	int end;
	for (int i = 2; i < observed_points.cols() - 1; i++)
	{
		float increment = errors[i + 1].second - errors[i].second;
		if (max_error_increment < increment)
		{
			max_error_increment = increment;
			end = i + 1;
		}
	}

	std::vector<Eigen::Vector3f> new_observed_points;
	std::vector<Eigen::Vector3f> new_model_points;
	for (int i = 0; i < end; i++)
	{
		new_observed_points.push_back(observed_points.col(errors[i].first));
		new_model_points.push_back(model_points.col(errors[i].first));
	}

	return fit_pose(new_observed_points, new_model_points, false);
}

	
float hand_pose_estimation::correct_heatmap_certainty(float val)
{
	// s(0) = 0, s(1) = 1, s(0.1) = 0.1, s(0.5) = 0.9
	return 0.52129f * std::tanhf(5.5876 * val - 1.5824f) + 0.47905f;
}

	
Eigen::Affine3f hand_pose_estimation::estimate_absolute_pose(const visual_input& input,
	const img_segment& seg,
	const net_evaluation& net_eval, bool right_hand) const
{
	std::vector<Eigen::Vector3f> observed_points;
	std::vector<Eigen::Vector3f> model_points;

	for (int joint : {0, 2, 5, 9, 17})
	{
		if (net_eval.key_points_2d.at(joint).x == -1)
			continue;

		if (seg.mask.at<uchar>(net_eval.key_points_2d.at(joint) - net_eval.input_box.tl()) == 0)
			continue;

		visual_input::PointT p;
		seg.prop_3d->get_surface_point_img(net_eval.key_points_2d.at(joint), p);

		observed_points.push_back(p.getVector3fMap());
		model_points.push_back(Eigen::UniformScaling(params.palm_length) * net_eval.left_hand_pose.col(joint));

		if (right_hand)
			model_points.back().x() *= -1;

	}

	return fit_pose(observed_points, model_points, true) * Eigen::UniformScaling(params.palm_length);
}



Eigen::Affine3f hand_pose_estimation::estimate_wrist_pose(const visual_input& input,
	const img_segment& seg,
	const net_evaluation& net_eval, const hand_pose_18DoF& finger_pose) const
{
	if (!seg.prop_3d)
		throw std::exception("Segement has no 3D information");

	Eigen::Matrix3Xf default_key_points = finger_pose.get_key_points();

	std::vector<Eigen::Vector3f> observed_points;
	std::vector<Eigen::Vector3f> model_points;


	for (int joint : {0, 2, 5, 9, 17})
	{
		if (net_eval.key_points_2d.at(joint).x == -1 || !seg.bounding_box.contains(net_eval.key_points_2d.at(joint)))
			continue;

		if (seg.mask.at<uchar>(net_eval.key_points_2d.at(joint) - seg.bounding_box.tl()) == 0)
			continue;

		visual_input::PointT p;
		seg.prop_3d->get_surface_point_img(net_eval.key_points_2d.at(joint), p);

		observed_points.push_back(p.getVector3fMap());
		model_points.push_back(default_key_points.col(joint));

	}

	return fit_pose(observed_points, model_points, true);
}

hand_pose_18DoF::Ptr hand_pose_estimation::estimate_relative_pose(const hand_kinematic_parameters& hand_kinematic_params, const Eigen::Matrix3Xf& keypoints, bool right_hand)
{
	using f_t = finger_type;
	using j_t = finger_joint_type;
	using b_t = finger_bone_type;

	auto orthogonal = [](const Eigen::Vector3f& decompose, const Eigen::Vector3f& normal)
	{
		Eigen::Vector3f decompose_n = decompose.normalized();
		Eigen::Vector3f normal_n = normal.normalized();

		return (decompose_n - decompose_n.dot(normal_n) * normal_n).normalized();
	};

	/* vec onto x_axis around z_axis, input vectors must be normalized*/
	auto oriented_angle = [](const Eigen::Vector3f& x_axis,
		const Eigen::Vector3f& vec,
		const Eigen::Vector3f& z_axis)
	{
		float angle = std::acosf(vec.dot(x_axis));

		//test wheter it's a clockwise or counterclockwise rotation
		if ((vec - Eigen::AngleAxisf(-angle, z_axis) * x_axis).norm() < (vec - Eigen::AngleAxisf(angle, z_axis) * x_axis).norm())
			angle *= -1;

		return angle;
	};

	Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();

	rotation.col(1) = (keypoints.col(9) - keypoints.col(0)).normalized();
	rotation.col(0) = orthogonal(keypoints.col(17) - keypoints.col(5), rotation.col(1));
	rotation.col(2) = rotation.col(0).cross(rotation.col(1));

	float scaling = hand_kinematic_params.get_finger(finger_type::MIDDLE).base_offset.norm() / (keypoints.col(9) - keypoints.col(0)).norm();

	// transformes seg.pose into the coordinate system of hand_pose_18DoF
	Eigen::Affine3f transform = Eigen::Translation3f(hand_kinematic_params.get_finger(finger_type::MIDDLE).base_offset) *
		Eigen::Quaternionf(rotation).inverse() *
		Eigen::UniformScaling(scaling) *
		Eigen::Translation3f(-1 * keypoints.col(0));

	Eigen::Matrix3Xf pose = transform * keypoints;

	//check if fingers are bended into negative z, otherwise rotate
	float sum_z = 0.f; int count = 0;
	for (int i = 1; i < pose.cols(); i = i % 4 == 0 ? i + 2 : i + 1)
	{
		sum_z += pose.col(i).z();
		count++;
	}

	if (sum_z / count > 0.01f)
	{
		pose.row(0) *= -1.f;
		pose.row(2) *= -1.f;
	}

	auto get_joint = [&](f_t finger, j_t joint)
	{
		return pose.col(1 + (int)finger * 4 + (int)joint);
	};
	auto get_bone = [&](f_t finger, b_t bone)
	{
		if (bone == b_t::METACARPAL)
			return get_joint(finger, j_t::MCP) - pose.col(0);
		else
			return get_joint(finger, (j_t)bone) - get_joint(finger, (j_t)((int)bone - 1));
	};

	std::vector<std::pair<float, float>> finger_bending;

	// handle thumb
	const auto& finger = hand_kinematic_params.fingers.at(0);

	Eigen::Vector3f metacarpal_vec = (pose.col(2) - finger.base_offset - pose.col(0)).normalized();

	Eigen::Vector3f x_axis = Eigen::Affine3f(finger.base_rotation) * Eigen::Vector3f::UnitX();
	Eigen::Vector3f y_axis = Eigen::Affine3f(finger.base_rotation) * Eigen::Vector3f::UnitY();
	Eigen::Vector3f z_axis = Eigen::Affine3f(finger.base_rotation) * Eigen::Vector3f::UnitZ();

	float thumb_adduction = oriented_angle(x_axis, orthogonal(metacarpal_vec, z_axis), z_axis);

	float base_extension = oriented_angle(x_axis, orthogonal(metacarpal_vec, y_axis), y_axis);

	float phalanx_extension;
	Eigen::Vector3f proximal_n = get_bone(f_t::THUMB, b_t::PROXIMAL).normalized();
	Eigen::Vector3f distal_n = get_bone(f_t::THUMB, b_t::DISTAL).normalized();
	Eigen::Vector3f middle_n = get_bone(f_t::THUMB, b_t::MIDDLE).normalized();

	if ((proximal_n - distal_n).isMuchSmallerThan(0.001f))
		phalanx_extension = 0.f;
	else if ((proximal_n + distal_n).isMuchSmallerThan(0.001f))
		phalanx_extension = -M_PI;
	else
		phalanx_extension = oriented_angle(proximal_n,
			orthogonal(distal_n, (-1 * proximal_n).cross(middle_n)), // distal phalanx projected into proximal-middle-phalanx-plane
			(-1 * proximal_n).cross(middle_n)); // proximal x middle

	finger_bending.push_back(std::make_pair(base_extension, phalanx_extension));

	// other fingers
	for (const f_t i : { f_t::INDEX, f_t::MIDDLE, f_t::RING, f_t::LITTLE })
	{
		const auto& finger = hand_kinematic_params.get_finger(i);

		float base_extension = M_PI_2 - std::acosf(get_bone(i, b_t::PROXIMAL).normalized().dot(Eigen::Vector3f::UnitZ()));

		float phalanx_extension;
		Eigen::Vector3f proximal_n = get_bone(i, b_t::PROXIMAL).normalized();
		Eigen::Vector3f distal_n = get_bone(i, b_t::DISTAL).normalized();
		Eigen::Vector3f middle_n = get_bone(i, b_t::MIDDLE).normalized();

		if ((proximal_n - distal_n).isMuchSmallerThan(0.001f))
			phalanx_extension = 0.f;
		else if ((proximal_n + distal_n).isMuchSmallerThan(0.001f))
			phalanx_extension = -M_PI;
		else
			phalanx_extension = oriented_angle(proximal_n,
				orthogonal(distal_n, (-1 * proximal_n).cross(middle_n)), // distal phalanx projected into proximal-middle-phalanx-plane
				(-1 * proximal_n).cross(middle_n)); // proximal x middle

		finger_bending.push_back(std::make_pair(base_extension, phalanx_extension));
	}

	//angle betwen little and index finger in x-y-plane
	float finger_spreading = oriented_angle(orthogonal(get_bone(f_t::INDEX, b_t::PROXIMAL), Eigen::Vector3f::UnitZ()),
		orthogonal(get_bone(f_t::LITTLE, b_t::PROXIMAL), Eigen::Vector3f::UnitZ()),
		Eigen::Vector3f::UnitZ()) / 2;

	return std::make_shared<hand_pose_18DoF>(hand_kinematic_params,
		Eigen::Affine3f(Eigen::Affine3f::Identity()),
		std::move(finger_bending),
		thumb_adduction,
		finger_spreading,
		right_hand);
}


std::vector<img_segment::Ptr> hand_pose_estimation::detect_hands(const visual_input& input,
	const std::vector<hand_instance::Ptr>& hands)
{
	std::vector<img_segment::Ptr> segments(skin_detect.detect_segments(input, hands));

	//for (img_segment::Ptr& seg : segments)
	//{
	//	evaluate_mueller_net(input_img, *seg);
	//}

	return segments;
}

void hand_pose_estimation::draw_silhouette(const pcl::PointCloud<PointT>::ConstPtr& cloud,
	cv::Mat& output_image,
	const Eigen::Matrix<float, 3, 4>& projection,
	cv::Scalar color) const
{
	float min_x, min_y, max_x, max_y;
	min_x = min_y = -0.5f;
	max_x = output_image.cols - 0.5f;
	max_y = output_image.rows - 0.5f;

	if (output_image.channels() == 4)
	{
		cv::Vec4b color_4 = color;
		for (const PointT& p : *cloud)
		{
			Eigen::Vector2f p_img = (projection * Eigen::Vector3f(p.data).homogeneous()).hnormalized();
			if (p_img.x() > min_x && p_img.x() < max_x && p_img.y() > min_y && p_img.y() < max_y)
				output_image.at<cv::Vec4b>(static_cast<int>(std::round(p_img.y())), static_cast<int>(std::round(p_img.x()))) = color_4;
		}
	}
	else
	{
		cv::Vec3b color_3(color[0], color[1], color[2]);
		for (const PointT& p : *cloud)
		{
			Eigen::Vector2f p_img = (projection * Eigen::Vector3f(p.data).homogeneous()).hnormalized();
			if (p_img.x() > min_x && p_img.x() < max_x && p_img.y() > min_y && p_img.y() < max_y)
				output_image.at<cv::Vec3b>(static_cast<int>(std::round(p_img.y())), static_cast<int>(std::round(p_img.x()))) = color_3;
		}
	}

}

hand_pose_18DoF::Ptr hand_pose_estimation::initial_estimate(const visual_input& input,
	const img_segment& seg,
	const net_evaluation& net_eval, bool right_hand) const
{
	auto model = estimate_relative_pose(hand_kinematic_params, net_eval.left_hand_pose, right_hand);

	try {
		model->wrist_pose = estimate_wrist_pose(input, seg, net_eval, *model);
	}
	catch (const std::exception&)
	{

		Eigen::Vector3f centroid = seg.prop_3d ? seg.prop_3d->centroid : Eigen::Vector3f(0.f, 0.f, 1.f); // TODO

		cv::Point wrist = net_eval.key_points_2d.at(0);
		cv::Point mcp = net_eval.key_points_2d.at(9);

		// origin is top left
		cv::Point y_vec = wrist - mcp;

		float angle = std::atan2f(y_vec.y, y_vec.x) - M_PI_2;

		Eigen::Vector2f cloud_vec_x = input.img_projection.block<2, 2>(0, 0).inverse().col(0).normalized();
		angle += std::atan2f(cloud_vec_x.y(), cloud_vec_x.x());

		model->wrist_pose = Eigen::Translation3f(centroid) *
			Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()) *
			Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()) *
			Eigen::Translation3f(-model->hand_kinematic_params.get_finger(finger_type::MIDDLE).base_offset);
	}

	return model;
}

net_evaluation::heatmaps hand_pose_estimation::fuse_heatmaps(const net_evaluation::heatmaps& map_1, const net_evaluation::heatmaps& map_2)
{
	if (map_1.size() != map_2.size())
		throw std::exception("Vectors differ in length");

	const net_evaluation::heatmaps& smaller_vec = map_1.front().rows < map_2.front().rows ? map_1 : map_2;
	const net_evaluation::heatmaps& larger_vec = map_1.front().rows >= map_2.front().rows ? map_1 : map_2;

	float scale_x = larger_vec.front().cols / (float)smaller_vec.front().cols;
	float scale_y = larger_vec.front().rows / (float)smaller_vec.front().rows;

	net_evaluation::heatmaps result;

	for (int i = 0; i < larger_vec.size(); i++)
	{
		const auto& smaller = smaller_vec[i];
		const auto& larger = larger_vec[i];

		result.push_back(cv::Mat());
		larger.copyTo(result.back());
		auto& res = result.back();

		for (int x = 0; x < smaller.rows; x++)
			for (int y = 0; y < smaller.cols; y++)
			{
				int dest_x = (int)std::roundf(x * scale_y);
				int dest_y = (int)std::roundf(y * scale_x);

				res.at<float>(dest_x, dest_y) = std::max(smaller.at<float>(x, y), res.at<float>(dest_x, dest_y));
			}
	}

	return result;
}

cv::Rect2i hand_pose_estimation::input_to_net(const cv::Mat& img,
	cv::Rect2i roi_box,
	const net_context& net,
	bool swap_rb,
	bool flip,
	cv::InputArray mask) const
{
	float min_scaling = std::max(net.input_w / (float)img.cols, net.input_h / (float)img.rows);
	
	float scale_x = net.input_w / (roi_box.width * (1.f + params.roi_box_padding));
	float scale_y = net.input_h / (roi_box.height * (1.f + params.roi_box_padding));
	float scaling = std::max(min_scaling, std::min(2.f, std::min(scale_x, scale_y)));

	float width = std::min((float)img.cols, net.input_w / scaling);
	float height = std::min((float)img.rows, net.input_h / scaling);
	float x = roi_box.x - (width - roi_box.width) / 2;
	float y = roi_box.y - (height - roi_box.height) / 2;

	if (x + width > img.cols)
		x = img.cols - width;

	if (y + height > img.rows)
		y = img.rows - height;

	cv::Rect2i crop_box(
		std::max(0.f, x),
		std::max(0.f, y),
		std::round(net.input_w / scaling),
		std::round(net.input_h / scaling));

	cv::Mat canvas;
	if (mask.isMat())
	{
		cv::Mat extended_mask(crop_box.height, crop_box.width, mask.type(), cv::Scalar::all(0));
		mask.copyTo(extended_mask(roi_box - crop_box.tl()));

		canvas = cv::Mat(crop_box.height, crop_box.width, img.type(), cv::Scalar::all(0));
		cv::bitwise_and(img(crop_box), img(crop_box), canvas, extended_mask);
	}
	else
	{
		canvas = img(crop_box);
	}

	std::vector<cv::Mat> channels;
	cv::split(canvas, channels);

	if (channels.size() < 3)
		throw std::exception("Image does not have enough channels.");

	std::vector<cv::Mat> float_channels(3);
	for (int i = 0; i < 3; i++)
	{
		// change from BGR to RGB
		channels[swap_rb ? 2 - i : i].convertTo(float_channels[i], CV_32FC1, 1.0 / 255);
		if (flip)
			cv::flip(float_channels[i], float_channels[i], 1);

		//cv::rotate(float_channels[i], float_channels[i], cv::ROTATE_180);
	}

	caffe::Blob<float>* input_layer = net.net->input_blobs()[0];

	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i)
	{
		cv::Mat channel(net.input_h, net.input_w, CV_32FC1, input_data);
		cv::resize(float_channels[i], channel, cv::Size(net.input_w, net.input_h));

		if (reinterpret_cast<float*>(channel.data)
			!= input_data)
			throw std::exception("Input not transferred but copied to different location.");

		input_data += net.input_h * net.input_w;
	}

	return crop_box;
}

void hand_pose_estimation::hand_pose_estimation::init_neural_network()
{
	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	caffe::Caffe::SetDevice(0);

	//simon_net.net = std::make_unique<caffe::Net<float>>(simon_net_params.proto_file, caffe::TEST);
	//simon_net.net->CopyTrainedLayersFrom(simon_net_params.weights_file);


	mueller_net.net = std::make_unique<caffe::Net<float>>(mueller_net_params.proto_file, caffe::TEST);
	mueller_net.net->CopyTrainedLayersFrom(mueller_net_params.weights_file);

	struct cudaDeviceProp props;

	cudaGetDeviceProperties(&props, 0);

	//{
	//	net_context& net = simon_net;

	//	net.input_w = net.net->input_blobs()[0]->width();
	//	net.input_h = net.net->input_blobs()[0]->height();

	//	net.output_w = net.net->output_blobs()[0]->width();
	//	net.output_h = net.net->output_blobs()[0]->height();
	//}

	{
		net_context& net = mueller_net;

		net.input_w = net.net->input_blobs()[0]->width();
		net.input_h = net.net->input_blobs()[0]->height();

		net.output_w = net.net->output_blobs()[0]->width();
		net.output_h = net.net->output_blobs()[0]->height();
	}

	// std::cout << "using GPU: " << props.name << std::endl;
}

net_evaluation::Ptr hand_pose_estimation::evaluate_simon_net(const cv::Mat& img, const img_segment& seg, bool flip) const
{
	auto input_box = input_to_net(img, seg.bounding_box, simon_net, false, flip);

	simon_net.net->Forward();

	net_evaluation::heatmaps maps;
	maps.reserve(simon_net_params.n_points);
	float* raw_data = const_cast<float*>(simon_net.net->output_blobs()[0]->cpu_data());
	const unsigned int size = simon_net.output_h * simon_net.output_w;


	for (int i = 0; i < simon_net_params.n_points; i++)
	{
		maps.emplace_back(cv::Mat(simon_net.output_h, simon_net.output_w, CV_32FC1));
		std::memcpy(maps.back().data, raw_data + i * size, size * sizeof(float));

		if (flip)
			cv::flip(maps.back(), maps.back(), 1);

		//cv::rotate(maps.back(), maps.back(), cv::ROTATE_180);
	}

	return std::make_shared<net_evaluation>(std::move(input_box), std::move(maps), flip);
}

net_evaluation::Ptr hand_pose_estimation::evaluate_mueller_net(const cv::Mat& img, const img_segment& seg, bool flip) const
{
	auto input_box = input_to_net(img, seg.bounding_box, mueller_net, true, flip);

	mueller_net.net->Forward();

	net_evaluation::heatmaps maps;
	maps.reserve(mueller_net_params.n_points);
	float* raw_data = const_cast<float*>(mueller_net.net->output_blobs()[0]->cpu_data());
	const unsigned int size = mueller_net.output_h * mueller_net.output_w;

	for (int i = 0; i < mueller_net_params.n_points; i++)
	{
		maps.emplace_back(cv::Mat(mueller_net.output_h, mueller_net.output_w, CV_32FC1));
		std::memcpy(maps.back().data, raw_data + i * size, size * sizeof(float));

		if (flip)
			cv::flip(maps.back(), maps.back(), 1);

		//cv::rotate(maps.back(), maps.back(), cv::ROTATE_180);
	}

	return std::make_shared<net_evaluation>(std::move(input_box), std::move(maps), flip,
		Eigen::Map<Eigen::Matrix3Xf>((float*)mueller_net.net->output_blobs()[1]->cpu_data(), 3, mueller_net_params.n_points));
}

//void hand_pose_estimation::setProjectMatrix(const Eigen::Matrix<float,2,4>& mat)
//{
//	projection_3D_to_image = mat;
//	
//}

/////////////////////////////////////////////////////////////
//
//
//  Class: hand_pose_estimation_async
//
//
/////////////////////////////////////////////////////////////

hand_pose_estimation_async::hand_pose_estimation_async(const hand_kinematic_parameters& hand_kinematic_params)
	:
	hand_pose_estimation(hand_kinematic_params, false),
	terminate_flag(false)
{
	internal_thread = std::thread([this]()
		{
			init_neural_network();

			while (true)
				try {

				std::function<void()> f;
				{
					std::unique_lock<std::mutex> lock(queue_mutex);

					if (terminate_flag)
						return;

					queue_condition_variable.wait(lock, [&] {return !queue.empty() || terminate_flag; });

					if (terminate_flag)
						return;

					f = queue.front();
					queue.pop_front();
				}

				f();

			}
			catch (const std::exception& e)
			{
				std::cout << e.what() << std::endl;
			}
		});
}

hand_pose_estimation_async::~hand_pose_estimation_async()
{
	terminate_flag = true;
	queue_condition_variable.notify_all();

	if (internal_thread.joinable())
		internal_thread.join();
}
	
void hand_pose_estimation_async::estimate_keypoints_async(const cv::Mat& input, const img_segment::Ptr& hand_candidate, bool right_hand, std::function<void(const net_evaluation::ConstPtr&)> callback)
{
	std::unique_lock<std::mutex> lock(queue_mutex);
	queue.emplace_back([this, &input, hand_candidate, right_hand, callback]()
		{
			estimate_keypoints(input, *hand_candidate, right_hand);
			callback(hand_candidate->net_evals[right_hand]);
		});
	queue_condition_variable.notify_all();
}
	
void hand_pose_estimation_async::clear_queue()
{
	std::unique_lock<std::mutex> lock(queue_mutex);
	queue.clear();
}
	
} /* hand_pose_estimation */
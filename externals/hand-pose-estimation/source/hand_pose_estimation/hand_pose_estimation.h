#pragma once

#include "framework.h"

#include <string>

#pragma warning( push )
#pragma warning( disable : 4996 )
#include <caffe/caffe.hpp>
#pragma warning( pop )

#include <Eigen/core>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "parameter_set.hpp"
#include "hand_model.hpp"
#include "ra_skin_color_detector.hpp"

class HANDPOSEESTIMATION_API cv::Mat;
template class HANDPOSEESTIMATION_API cv::Rect_<int>;
template class HANDPOSEESTIMATION_API cv::Point_<int>;

namespace hand_pose_estimation
{

/**
 * @class simon_net_parameters
 * @brief Parameters for Simon's hand pose estimation network
 *
 * Configuration parameters for the hand pose estimation network described in
 * arxiv.org/abs/1704.07809, including network structure and weights file paths.
 *
 * Features:
 * - Network structure configuration
 * - Weights file management
 * - Keypoint count specification
 * - Parameter serialization
 */
class simon_net_parameters : public parameter_set {
public:

	simon_net_parameters();

	~simon_net_parameters();

	/*
	* Path to the file describing the DNN structure
	*/
	std::string proto_file;

	/*
	* Path to the file describing the DNN weights structure
	*/
	std::string weights_file;

	/*
	* Number of keypoints defining a hand
	*/
	int n_points;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(proto_file);
		ar& BOOST_SERIALIZATION_NVP(weights_file);
		ar& BOOST_SERIALIZATION_NVP(n_points);
	}

};


/**
 * @class mueller_net_parameters
 * @brief Parameters for Mueller's hand pose estimation network
 *
 * Configuration parameters for the hand pose estimation network described in
 * https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/ with DOI
 * 10.1109/CVPR.2018.00013.
 *
 * Features:
 * - Network structure configuration
 * - Weights file management
 * - Keypoint count specification
 * - Parameter serialization
 */
class mueller_net_parameters : public parameter_set {
public:

	mueller_net_parameters();

	~mueller_net_parameters();

	/*
	* Path to the file describing the DNN structure
	*/
	std::string proto_file;

	/*
	* Path to the file describing the DNN weights structure
	*/
	std::string weights_file;

	/*
	* Number of keypoints defining a hand
	*/
	int n_points;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(proto_file);
		ar& BOOST_SERIALIZATION_NVP(weights_file);
		ar& BOOST_SERIALIZATION_NVP(n_points);
	}

};


/**
 * @class hand_pose_parameters
 * @brief Parameters for hand segment detection
 *
 * Configuration parameters for detecting and processing hand segments in images,
 * including probability thresholds and geometric constraints.
 *
 * Features:
 * - Keypoint probability thresholds
 * - Hand detection criteria
 * - Bone length specifications
 * - ROI box padding
 * - Parameter serialization
 */
class hand_pose_parameters : public parameter_set {
public:

	hand_pose_parameters();

	~hand_pose_parameters();


	/*
	* Keypoints with a probability below this threshold are discarded
	*/
	float keypoint_probability_threshold;

	/*
	* If the average probability of the non-discarded keypoints of a img_segment is below hand_probability_threshold, the img_segment is not a hand
	*/
	float hand_probability_threshold;

	/*
	* If less than hand_min_keypoints are found in a img_segment, it is not a hand
	*/
	int hand_min_keypoints;

	/**
	* Minimum edge length of a bounding box containing a hand
	*/
	float hand_min_dimension;

	/*
	* Length of the bone in the center of the palm (3rd metacarpal)
	*/
	float palm_length;

	/*
	* Lengths relative to palm length. See assets/HandSkeleton.pdf for indexing
	*/
	std::vector<float> bone_lengths;

	/**
	 *
	 * Percent increase of the bounding box dimensions before feeding
	 * the image into the neural network.
	 */
	float roi_box_padding;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(keypoint_probability_threshold);
		ar& BOOST_SERIALIZATION_NVP(hand_probability_threshold);
		ar& BOOST_SERIALIZATION_NVP(hand_min_keypoints);
		ar& BOOST_SERIALIZATION_NVP(hand_min_dimension);
		ar& BOOST_SERIALIZATION_NVP(palm_length);
		ar& BOOST_SERIALIZATION_NVP(bone_lengths);
		ar& BOOST_SERIALIZATION_NVP(roi_box_padding);
	}
};


/**
 * @struct net_context
 * @brief Neural network context information
 *
 * Stores context information for neural networks, including network instance,
 * input/output dimensions, and related parameters.
 *
 * Features:
 * - Network instance management
 * - Dimension tracking
 * - Input/output size specification
 */
struct net_context
{
	std::unique_ptr<caffe::Net<float>> net;
	int input_w;
	int input_h;
	int output_w;
	int output_h;
};

/**
 * @class hand_pose_estimation
 * @brief 3D hand skeleton model computation
 *
 * Core class for computing 3D hand skeleton models from images and point clouds.
 * Integrates multiple neural networks for pose estimation and provides various
 * pose fitting and estimation methods.
 *
 * Features:
 * - 3D hand skeleton computation
 * - Multiple network integration
 * - Pose estimation and fitting
 * - Keypoint detection
 * - Hand detection
 * - Silhouette visualization
 */
class  hand_pose_estimation
{

public:
	typedef  pcl::PointXYZRGBA PointT;
	typedef std::shared_ptr<hand_pose_estimation> Ptr;
	typedef std::shared_ptr<const hand_pose_estimation> ConstPtr;


	HANDPOSEESTIMATION_API hand_pose_estimation(const hand_kinematic_parameters& hand_kinematic_params, bool load_nets = true);
	HANDPOSEESTIMATION_API ~hand_pose_estimation() = default;

	HANDPOSEESTIMATION_API const skin_detector& get_skin_detection() const;

	/*
	* Computes and stores the heatmap (and relative pose) of @param{hand_candidate}
	* Returns the certainty that @param{hand_candidate} is a hand based on the average key point probability
	*/
	HANDPOSEESTIMATION_API double estimate_keypoints(const cv::Mat& input,
		img_segment& hand_candidate,
		bool right_hand = false);

	/*
	* Calls @ref{estimate_keypoints} for the latest segment and updates @param{hand_instance}
	*/
	HANDPOSEESTIMATION_API double estimate_keypoints(const cv::Mat& input,
		hand_instance& hand_candidate);

	/*
	* Computes a transformation (rotation and translation only) that brings model_points onto observed_points
	* minimizing mean squared error. If more than 3 points are provided, outliers are dropped.
	*/
	HANDPOSEESTIMATION_API static Eigen::Affine3f fit_pose(const std::vector<Eigen::Vector3f>& observed_points,
		const std::vector<Eigen::Vector3f>& model_points,
		bool remove_outliers = false);

	HANDPOSEESTIMATION_API static Eigen::Affine3f fit_pose(const Eigen::Matrix3Xf& observed_points,
		const Eigen::Matrix3Xf& model_points,
		bool remove_outliers = false);

	HANDPOSEESTIMATION_API static float correct_heatmap_certainty(float val);

	/*
	* Uses depth points to estimate the absolute pose from the relative one
	* Maps from seg.pose to points in input.cloud
	* 
	* Throws if not enough valid points are found.
	*/
	HANDPOSEESTIMATION_API Eigen::Affine3f estimate_absolute_pose(const visual_input& input,
	                                                              const img_segment& seg,
	                                                              const net_evaluation& net_eval, bool right_hand = false) const;

	/*
* Uses depth points to estimate the absolute pose of the wrist for hand_model_18DoF
* Maps the origin of hand_model_18DoF to the wrist point in input.cloud
* 
* Throws if not enough valid points are found.
*/
	HANDPOSEESTIMATION_API Eigen::Affine3f estimate_wrist_pose(const visual_input& input,
	                                                           const img_segment& seg,
	                                                           const net_evaluation& net_eval, const hand_pose_18DoF& finger_pose) const;

	static HANDPOSEESTIMATION_API hand_pose_18DoF::Ptr estimate_relative_pose(const hand_kinematic_parameters& hand_params,
		const Eigen::Matrix3Xf& keypoints, 
		bool right_hand);

	/**
	* Updates the tracked hands from the given @ref{cloud}.
	* Optionally, subclouds of @ref{cloud} can be passed in {hand_location_hints} to define regions of interest.
	* If @ref{hand_location_hints} is empty, the whole image is the region of interest.
	* The search will be limited to the regions of interest
	*/
	HANDPOSEESTIMATION_API std::vector<img_segment::Ptr> detect_hands(const visual_input& input,
		const std::vector< hand_instance::Ptr>& hands
	);

	/**
	* Draws the silhouette that results from projecting @ref{cloud} back into pixel space.
	*/
	HANDPOSEESTIMATION_API void draw_silhouette(const pcl::PointCloud<PointT>::ConstPtr& cloud,
		cv::Mat& canvas,
		const Eigen::Matrix<float, 3, 4>& projection,
		cv::Scalar color = cv::Scalar(255,255,255,0)) const;


	HANDPOSEESTIMATION_API hand_pose_18DoF::Ptr initial_estimate(const visual_input&,
	                                                             const img_segment& seg,
	                                                             const net_evaluation& net_eval, bool right_hand = false) const;


	HANDPOSEESTIMATION_API static net_evaluation::heatmaps fuse_heatmaps(const net_evaluation::heatmaps& map_1,
		const net_evaluation::heatmaps& map_2);

	/**
	* @param{mat} transformes a point from the point cloud to 2D image coordinates
	*/
	//	HANDPOSEESTIMATION_API void setProjectMatrix(const Eigen::Matrix<float, 2, 4>& mat);

	const hand_pose_parameters params;
	const simon_net_parameters simon_net_params;
	const mueller_net_parameters mueller_net_params;

	const hand_kinematic_parameters& hand_kinematic_params;

	void init_neural_network();
	
private:
	net_context simon_net;
	net_context mueller_net;
	skin_detector skin_detect;

	cv::Rect2i input_to_net(const cv::Mat& img, cv::Rect2i box, const net_context& net, bool swap_rb, bool flip = false, cv::InputArray mask = cv::noArray()) const;

	net_evaluation::Ptr evaluate_simon_net(const cv::Mat& img, const img_segment& seg, bool flip = false) const;
	net_evaluation::Ptr evaluate_mueller_net(const cv::Mat& img, const img_segment& seg, bool flip = false) const;
};

/**
 * @class hand_pose_estimation_async
 * @brief Asynchronous hand pose estimation
 *
 * Extends hand_pose_estimation with asynchronous processing capabilities,
 * allowing for non-blocking pose estimation and callback-based results.
 *
 * Features:
 * - Asynchronous keypoint estimation
 * - Callback-based results
 * - Queue management
 * - Thread-safe operation
 */
class  hand_pose_estimation_async : public hand_pose_estimation
{
public:
	HANDPOSEESTIMATION_API hand_pose_estimation_async(const hand_kinematic_parameters& hand_kinematic_params);
	HANDPOSEESTIMATION_API ~hand_pose_estimation_async();

	/*
* Computes and stores the heatmap (and relative pose) of @param{hand_candidate}
* Returns the certainty that @param{hand_candidate} is a hand based on the average key point probability
*/
	HANDPOSEESTIMATION_API void estimate_keypoints_async(const cv::Mat& input,
		const img_segment::Ptr& hand_candidate,
		bool right_hand,
		std::function<void(const net_evaluation::ConstPtr&)> callback = [](const net_evaluation::ConstPtr&){});

	HANDPOSEESTIMATION_API void clear_queue();

private:
	std::atomic_bool terminate_flag;
	std::thread internal_thread;

	std::list<std::function<void()>> queue;
	std::mutex queue_mutex;
	std::condition_variable queue_condition_variable;
};
	
} /* hand_pose_estimation */

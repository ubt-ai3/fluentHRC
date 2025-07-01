#pragma once

#include "framework.h"

#include <list>
#include <memory>
#include <vector>
#include <chrono>

#include <Eigen/Core>

#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include "opencv-serialization/opencv_serialization.hpp"
#include "eigen_serialization.hpp"

#include <opencv2/core.hpp>

#include <pcl/common/common.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "parameter_set.hpp"
#include <pcl/search/kdtree.h>

class HANDPOSEESTIMATION_API cv::Mat;
template class HANDPOSEESTIMATION_API cv::Rect_<int>;
template class HANDPOSEESTIMATION_API cv::Point_<int>;
//template class HANDPOSEESTIMATION_API cv::Vec<int, 4>;

namespace hand_pose_estimation
{

/*
class HANDPOSEESTIMATION_API cv::Mat;
template class HANDPOSEESTIMATION_API cv::Rect_<int>;
template class HANDPOSEESTIMATION_API cv::Point_<int>;
class HANDPOSEESTIMATION_API std::vector<float>;
template class HANDPOSEESTIMATION_API cv::Vec<int, 4>;
*/



/**
 * @class visual_input
 * @brief Common representation of image and point cloud based input
 *
 * Provides a unified interface for handling both image and point cloud data,
 * with support for coordinate transformations and validation. Always includes
 * image data, with optional point cloud data.
 *
 * Features:
 * - Image and point cloud data management
 * - Coordinate transformation
 * - Point validation
 * - Projection matrix handling
 * - Timestamp tracking
 */
class HANDPOSEESTIMATION_API visual_input
{
public:
	using Ptr = std::shared_ptr<visual_input>;
	using ConstPtr = std::shared_ptr<const visual_input>;
	using PointT = pcl::PointXYZRGBA;

	static const std::chrono::high_resolution_clock::time_point system_time_start;


	const pcl::PointCloud<PointT>::ConstPtr cloud;
	const cv::Mat img;
	const Eigen::Matrix<float, 3, 4> img_projection;
	const Eigen::Matrix<float, 3, 4> cloud_projection;
	const Eigen::Affine2f img_to_cloud_pixels;

	const Eigen::Vector3f sensor_origin = Eigen::Vector3f::Zero();

	const std::chrono::duration<float> timestamp_seconds;
	const bool extra_image = false;

	/**
	 * @param{projection} is a projetion matrix that maps from 3D to pixel coordinates.
	 * If not provided, it will be estimated from the @param{cloud}.
	 */
	visual_input(pcl::PointCloud<PointT>::ConstPtr cloud, 
		bool flip = false,
		Eigen::Matrix<float, 3, 4> projection = Eigen::Matrix<float,3,4>::Zero(),
		Eigen::Vector3f sensor_origin = Eigen::Vector3f::Zero());
	
	visual_input(cv::Mat img);

	/**
* Provides the ability to supply an image with a higher resolution than @param{cloud}.
*  @ref{img_projection} must be the projective matrix from
*    point cloud coordinates to image pixel coordinates.
*/
	visual_input(pcl::PointCloud<PointT>::ConstPtr cloud,
		cv::Mat img,
		Eigen::Matrix<float, 3, 4> img_projection,
		Eigen::Matrix<float, 3, 4> cloud_projection = Eigen::Matrix<float, 3, 4>::Zero(),
		Eigen::Vector3f sensor_origin = Eigen::Vector3f::Zero());

	static std::chrono::duration<float> relative_timestamp();
	static constexpr bool has_timestamp(const pcl::PointCloud<PointT>& cloud);

	inline bool has_cloud() const;

	inline bool is_valid_point(int x, int y) const;
	inline bool is_valid_point(const cv::Point2i& pixel) const;

	/**
* @unchecked
* */
	inline const PointT& get_point(const cv::Point2i& pixel) const;

	/**
	* Transformes from the pixel space of the image to the pixel space of the cloud.
	* The result is forced into [0,cloud->width-1]x[0,cloud->height-1].
	* This is a rough approximation, close objects might be misaligned
	*/
	inline cv::Point2i to_cloud_pixel_coordinates(const cv::Point2i&) const;
	inline cv::Rect2i to_cloud_pixel_coordinates(const cv::Rect2i&) const;

	inline cv::Point2i to_cloud_pixel_coordinates(const Eigen::Vector3f&) const;
	inline cv::Point2i to_cloud_pixel_coordinates_uncapped(const Eigen::Vector3f&) const;

	/*
	* @Attention Returns (-1,-1) if @param{p} is a nan-point
	*/
	inline cv::Point2i to_img_coordinates(const PointT& p) const;
	inline cv::Point2i to_img_coordinates(const Eigen::Vector3f& p) const;
	inline cv::Point2i to_img_coordinates_uncapped(const Eigen::Vector3f& p) const;
	inline Eigen::Vector2f to_img_coordinates_vec(const Eigen::Vector3f& p) const;
private:

	
	static Eigen::Affine2f compute_cloud_to_image_coordinates(pcl::PointCloud<PointT>::ConstPtr cloud,
		cv::Mat img,
		Eigen::Matrix<float, 3, 4> projection);

	static Eigen::Matrix<float,3,4> compute_cloud_projection(const pcl::PointCloud<PointT>::ConstPtr& cloud);

	/*
	* Returns the value closest to @param{x} from {0, ..., up - 1}
	*/
	int inline cap(float x, int up) const;
};



/**
 * @enum finger_type
 * @brief Enumeration of finger types
 */
enum class HANDPOSEESTIMATION_API finger_type
{
	THUMB = 0,
	INDEX = 1,
	MIDDLE = 2,
	RING = 3,
	LITTLE = 4,
	UNKNOWN = 1000
};

/*
* TIP refers to the finger tip and is not a real joint. MCP (where the finger 
* is connected to the palm) has 2 DoF, the other joints 1.
* For the thumb the notion is as follows (in line with @ref{finger_bone_type})
*	interphalangeal = DIP
*	metacarpophalangeal = PIP
*	trapeziometacarpal = MCP
*/
/**
 * @enum finger_joint_type
 * @brief Enumeration of finger joint types
 */
enum class HANDPOSEESTIMATION_API finger_joint_type
{
	MCP = 0,
	PIP = 1,
	DIP = 2,
	TIP = 3,
};

/*
* The metacarpal of the thumb can move freely whereas the metacarpals of 
* the other fingers form the palm. The thumb has no middle phalanx. 
* That all fingers have the same kinematic chain, we use a different notation here:
*   Proximal phalanx of thumb is referred to by MIDDLE
*   Metacarpal of thumb is referred to by PROXIMAL
*   wrist center to trapeziometacarpal is referred to by METACARPAL
*/
/**
 * @enum finger_bone_type
 * @brief Enumeration of finger bone types
 */
enum class HANDPOSEESTIMATION_API finger_bone_type
{
	METACARPAL = 0,
	PROXIMAL = 1,
	MIDDLE = 2,
	DISTAL = 3,
};


/**
 * @class finger_kinematic_parameters
 * @brief Kinematic parameters for a single finger
 *
 * Stores and manages the kinematic parameters for a finger using Modified
 * Denavit-Hartenberg parameters, including joint angles, offsets, and rotations.
 *
 * Features:
 * - Joint angle limits
 * - Base transformations
 * - Serialization support
 * - Kinematic chain calculations
 */
class HANDPOSEESTIMATION_API finger_kinematic_parameters
{
public:
	finger_kinematic_parameters() = default;
	finger_kinematic_parameters(
		std::vector<float>&& a,
		std::vector<float>&& theta_min,
		std::vector<float>&& theta_max,
		Eigen::Vector3f&& base_offset, 
		const Eigen::Quaternionf& base_rotation = Eigen::Quaternionf(Eigen::AngleAxisf(M_PI_2, Eigen::Vector3f::UnitZ())));

	~finger_kinematic_parameters() = default;

	static const std::vector<float> alpha;
	std::vector<float> a;
	std::vector<float> theta_min;
	std::vector<float> theta_max;

	Eigen::Vector3f base_offset;
	Eigen::Quaternionf base_rotation;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_NVP(a);
		ar& BOOST_SERIALIZATION_NVP(theta_min);
		ar& BOOST_SERIALIZATION_NVP(theta_max);
		ar& BOOST_SERIALIZATION_NVP(base_offset);
		ar& BOOST_SERIALIZATION_NVP(base_rotation);
	}

	Eigen::Affine3f transformation_to_base(const std::vector<float>& theta = { 0 },
		const Eigen::Vector4f& bone_scaling = Eigen::Vector4f::Ones()) const;

	Eigen::Vector3f relative_keypoint(const std::vector<float>& theta = { 0 },
		const Eigen::Vector4f& bone_scaling = Eigen::Vector4f::Ones()) const;
};

/**
 * @class hand_kinematic_parameters
 * @brief Complete hand kinematic model
 *
 * Describes the hand as five kinematic chains using Modified Denavit-Hartenberg
 * parameters. Manages finger parameters and hand thickness for complete hand
 * modeling.
 *
 * Features:
 * - Multi-finger kinematic chains
 * - Parameter serialization
 * - Finger access methods
 * - Hand thickness management
 */
class HANDPOSEESTIMATION_API hand_kinematic_parameters : public parameter_set
{
public:
	hand_kinematic_parameters();
	~hand_kinematic_parameters();

	std::vector<finger_kinematic_parameters> fingers;
	float thickness;
	float min_bone_scaling;
	float max_bone_scaling;
	
	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(fingers);
		ar& BOOST_SERIALIZATION_NVP(thickness);
		ar& BOOST_SERIALIZATION_NVP(min_bone_scaling);
		ar& BOOST_SERIALIZATION_NVP(max_bone_scaling);
	}

	finger_kinematic_parameters& get_finger(finger_type type);
	const finger_kinematic_parameters& get_finger(finger_type type) const;
};

/**
 * @class hand_pose_18DoF
 * @brief 18-degree-of-freedom hand pose representation
 *
 * Represents a complete hand pose with wrist position and orientation,
 * finger bending angles, thumb adduction, and finger spreading.
 *
 * Features:
 * - 18-DoF pose representation
 * - Euler angle conversions
 * - Key point calculations
 * - Pose combination
 * - Distance metrics
 */
class HANDPOSEESTIMATION_API hand_pose_18DoF
{
public:
	typedef std::shared_ptr<hand_pose_18DoF> Ptr;
	typedef std::shared_ptr<const hand_pose_18DoF> ConstPtr;
	typedef Eigen::Matrix<float, 15, 1> Vector15f;

	const static std::vector<float> centroid_weights;
	
	static Eigen::Vector3f to_euler_angles(const Eigen::Matrix3f& R);
	static Eigen::Quaternionf to_quaternion(const Eigen::Vector3f& euler_angles);

	static hand_pose_18DoF combine(Eigen::Affine3f wrist_pose, hand_pose_18DoF finger_poses);

	static float rotation_distance(const Eigen::Affine3f& pose_1, const Eigen::Affine3f& pose_2);
	static float rotation_distance(const Eigen::Quaternionf& pose_1, const Eigen::Quaternionf& pose_2);

	static Eigen::Vector3f get_centroid(const Eigen::Matrix3Xf& key_points);
	
	hand_pose_18DoF(const hand_kinematic_parameters& hand_kinematic_params);

	hand_pose_18DoF(const hand_kinematic_parameters& hand_kinematic_params,
		Eigen::Affine3f wrist_pose,
		std::vector<std::pair<float, float>> finger_bending,
		float thumb_adduction,
		float finger_spreading,
		bool right_hand = false,
		Eigen::Matrix<float, 4, 5> bone_scaling = Eigen::Matrix<float, 4, 5>::Ones());

	/*
	* See @ref{get_parameters}
	*/
	hand_pose_18DoF(const hand_kinematic_parameters& hand_kinematic_params,
		Vector15f parameters,
		Eigen::Quaternionf rotation,
		bool right_hand = false,
		Eigen::Matrix<float, 4, 5> bone_scaling = Eigen::Matrix<float, 4, 5>::Ones());

	hand_pose_18DoF(const hand_pose_18DoF&) = default;
	hand_pose_18DoF(hand_pose_18DoF&&) noexcept = default;

	~hand_pose_18DoF() = default;

	hand_pose_18DoF& operator=(const hand_pose_18DoF&);

	const hand_kinematic_parameters& hand_kinematic_params;
	Eigen::Affine3f wrist_pose;
	std::vector<std::pair<float, float>> finger_bending;
	float thumb_adduction;
	float finger_spreading;
	bool right_hand;
	Eigen::Matrix<float,4,5> bone_scaling;


	float get_max_spreading() const;
	/*
	*  Parameter 0 - 2: (x,y,z) of wrist
	*  Parameter 3    : thumb adduction
	*  Parameter 4 -13: for fingers thumb to little: extension of metacarpal, extension of phalances (combines two joints)
	*  Parameter 14   : finger spreading
	*/
	inline float get_parameter(unsigned int index) const;
	Vector15f get_parameters() const;

	/*
	* See @ref{get_parameters}
	*/
	Vector15f get_lower_bounds() const;
	Vector15f get_upper_bounds() const;

	/*
	* Angles of joints closest to tip which are represented by one parameter only
	*/
	std::pair<float, float> get_interphalangeal_angles(finger_type finger) const;

	/*
	* Input to finger_kinematic_parameters::transformation_to_base(...)
	*/
	std::vector<float> get_theta_angles(finger_type finger,
		finger_joint_type joint = finger_joint_type::TIP) const;

	Eigen::Matrix<float,3,4> get_key_points(finger_type finger) const;
	Eigen::Matrix3Xf get_key_points() const;

	Eigen::Vector3f get_tip(finger_type finger) const;
	Eigen::Matrix<float, 3, 5> get_tips() const;

	float rotational_distance(const hand_pose_18DoF& other) const;

	Eigen::Vector3f get_centroid() const;
};


/**
 * @class hand_dynamic_parameters
 * @brief Dynamic parameters for hand movement
 *
 * Manages dynamic parameters for hand movement, including velocity limits
 * and constraints for different degrees of freedom.
 *
 * Features:
 * - Velocity constraints
 * - Parameter serialization
 * - DoF-specific limits
 */
class HANDPOSEESTIMATION_API hand_dynamic_parameters : public parameter_set
{
public:
	using ConstPtr = std::shared_ptr<const hand_dynamic_parameters>;
	
	hand_dynamic_parameters();
	~hand_dynamic_parameters();

	float speed_extension_mcp;
	float speed_extension_pip;
	float speed_extension_dip;
	float speed_wrist;
	float speed_rotation;
	float speed_adduction_thumb;
	float speed_adduction;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(speed_extension_mcp);
		ar& BOOST_SERIALIZATION_NVP(speed_extension_pip);
		ar& BOOST_SERIALIZATION_NVP(speed_extension_dip);
		ar& BOOST_SERIALIZATION_NVP(speed_wrist);
		ar& BOOST_SERIALIZATION_NVP(speed_rotation);
		ar& BOOST_SERIALIZATION_NVP(speed_adduction_thumb);
		ar& BOOST_SERIALIZATION_NVP(speed_adduction);
	}

	hand_pose_18DoF::Vector15f get_constraints_18DoF() const;
};


/**
 * @class net_evaluation
 * @brief Neural network evaluation results
 *
 * Stores and manages the results of neural network evaluations for hand
 * pose estimation, including heatmaps and key point detections.
 *
 * Features:
 * - Heatmap management
 * - Key point tracking
 * - Certainty scoring
 * - Hand identification
 */
class HANDPOSEESTIMATION_API net_evaluation {
public:
	typedef std::shared_ptr<net_evaluation> Ptr;
	typedef std::shared_ptr<const net_evaluation> ConstPtr;
	typedef std::vector<cv::Mat1f> heatmaps;

	cv::Rect2i input_box;
	heatmaps maps;
	std::vector<cv::Point2i> key_points_2d;
	Eigen::VectorXf certainties;
	float certainty;
	bool right_hand;
	Eigen::Matrix3Xf left_hand_pose;

	net_evaluation() = default;
	net_evaluation(const net_evaluation&) = default;
	net_evaluation(net_evaluation&&) noexcept = default;

	net_evaluation& operator=(const net_evaluation&) = default;

	net_evaluation(cv::Rect2i input_box,
		heatmaps maps,
		bool right_hand,
		std::vector<cv::Point2i> key_points_2d,
		Eigen::VectorXf certainties,
		Eigen::Matrix3Xf left_hand_pose = Eigen::Matrix3Xf());

	net_evaluation(cv::Rect2i input_box,
		heatmaps maps,
		bool right_hand,
		Eigen::Matrix3Xf left_hand_pose = Eigen::Matrix3Xf());
};
	
/**
 * @class hand_pose_particle_instance
 * @brief Single instance of a hand pose particle
 *
 * Represents a single instance of a hand pose with associated evaluation
 * results and timing information.
 *
 * Features:
 * - Pose storage
 * - Evaluation results
 * - Timing information
 * - Bounding box calculations
 */
class HANDPOSEESTIMATION_API hand_pose_particle_instance {
public:

	typedef std::shared_ptr<hand_pose_particle_instance> Ptr;
	typedef std::shared_ptr<const hand_pose_particle_instance> ConstPtr;

	hand_pose_18DoF pose;
	Eigen::Quaternionf rotation;
	net_evaluation::ConstPtr net_eval;
	Eigen::Matrix3Xf key_points;
	std::vector<float> surface_distances;

	std::chrono::duration<float> time_seconds;

	/* 0 = perfect match*/
	double error;

	/* set before adding pose to img_segment */
	float hand_certainty;
	float hand_orientation;

	int updates = 0;

	hand_pose_particle_instance(const hand_pose_particle_instance&) = default;
	hand_pose_particle_instance(hand_pose_particle_instance&&) noexcept = default;

	hand_pose_particle_instance& operator=(const hand_pose_particle_instance&) = default;

	hand_pose_particle_instance(hand_pose_18DoF pose,
		net_evaluation::ConstPtr net_eval,
		std::chrono::duration<float> time_seconds,
		Eigen::Matrix3Xf key_points = Eigen::Matrix3Xf(),
		std::vector<float> surface_distances = std::vector<float>(21, std::numeric_limits<float>::quiet_NaN()));

	cv::Rect2i get_box(const visual_input& input) const;

	Eigen::AlignedBox3f get_box() const;
};

/**
 * @class hand_pose_particle
 * @brief Collection of hand pose instances
 *
 * Manages a collection of hand pose instances over time, providing
 * functionality for adding and tracking pose changes.
 *
 * Features:
 * - Instance management
 * - Temporal tracking
 * - Pose sequence handling
 */
class HANDPOSEESTIMATION_API hand_pose_particle {
public:
	typedef std::shared_ptr<hand_pose_particle> Ptr;
	typedef std::shared_ptr<const hand_pose_particle> ConstPtr;

	hand_pose_particle_instance::Ptr best;
	hand_pose_particle_instance::Ptr current;
	/* might be null */
	hand_pose_particle_instance::Ptr prev_frame;


	hand_pose_particle(const hand_pose_particle&) = default;
	hand_pose_particle(hand_pose_particle&&) noexcept = default;

	hand_pose_particle& operator=(const hand_pose_particle&) = default;

	hand_pose_particle(const hand_pose_particle_instance::Ptr& current,
		const hand_pose_particle_instance::Ptr& prev_frame = nullptr);

	void add(const hand_pose_particle_instance::Ptr& next);
};





class img_segment;

/**
 * @class img_segment_3d
 * @brief 3D image segment representation
 *
 * Provides 3D representation and processing capabilities for image segments,
 * including surface point calculations and normal estimation.
 *
 * Features:
 * - 3D point cloud processing
 * - Surface point calculation
 * - Normal estimation
 * - Nearest neighbor lookup
 */
class HANDPOSEESTIMATION_API img_segment_3d {
public:
	typedef std::shared_ptr<img_segment_3d> Ptr;
	typedef std::shared_ptr<const img_segment_3d> ConstPtr;

	img_segment_3d(const visual_input& input, const img_segment& seg, const Eigen::Hyperplane<float, 3>& background = Eigen::Hyperplane<float, 3>(Eigen::Vector3f::Zero(), 0.f));
	img_segment_3d(const visual_input& input, const img_segment& seg, pcl::PointCloud<visual_input::PointT>::ConstPtr cloud);

	const bool extra_image;
	
	Eigen::AlignedBox3f bounding_box;
	Eigen::Vector3f centroid;
	cv::Rect2i cloud_box;

	pcl::PointCloud<visual_input::PointT>::ConstPtr cloud;
	mutable pcl::PointCloud<pcl::Normal>::Ptr normals;
	Eigen::Vector3f sensor_origin;

	/*
	 * Get the point closest to @param{reference} from @ref{cloud} (which only contains points from the region of interest).
	 * Additional information such as the normal, squared distance and its index in closest to @param reference and the squared distance.
	 * Returns whether @param{reference} is in the region of interest.
	 */
	void get_surface_point(const visual_input& input,
		const Eigen::Vector3f& reference, 
		visual_input::PointT& point,
		pcl::Normal* normal = nullptr,
		int* index = nullptr) const;

	/*
 * Get the point closest to @param{reference} from @ref{cloud} (which only contains points from the region of interest).
 * Input are the pixel coordinates w.r.t. input.image. In case of projection ambiguities, points closer to the camera are preferred.
 * Additional information such as the normal, squared distance and its index in closest to @param reference and the squared distance.
	 * Returns whether @param{reference} is in the region of interest.
	*/
	void get_surface_point_img(const cv::Point2i& reference,
		visual_input::PointT& point,
		pcl::Normal* normal = nullptr,
		int* index = nullptr) const;

	/*
* Get the point closest to @param{reference} from @ref{cloud} (which only contains points from the region of interest).
* Input are the pixel coordinates w.r.t. input.cloud
* Additional information such as the normal, squared distance and its index in closest to @param reference and the squared distance.
	 * Returns whether @param{reference} is in the region of interest.
	*/
	void get_surface_point_cloud(const cv::Point2i& reference,
		visual_input::PointT& point,
		pcl::Normal* normal = nullptr,
		int* index = nullptr) const;

private:
	// nearest neighbor search in 2D, stores indices for @param{cloud}
	mutable cv::Mat1i nn_lookup_cloud;
	cv::Rect2i nn_lookup_box_cloud;

	// only computed in case of extra_image
	pcl::search::KdTree<pcl::PointXYZL> img_quadtree;
	pcl::search::KdTree<pcl::PointXYZL> quadtree;

	mutable pcl::NormalEstimation<visual_input::PointT, pcl::Normal> normal_estimator;

	void compute_nearest_neighbor_lookup(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr& index_cloud,
		const cv::Rect2i& box,
		cv::Mat1i& lookup,
		cv::Rect2i& lookup_box);

	inline int get_nearest_neighbor_cloud(int x, int y) const;

	/*
	 * Results are stored in this->centroid and this->bounding_box
	 */
	void centroid_and_box_without_outliers(const pcl::PointCloud<visual_input::PointT>& cloud, Eigen::Vector3f& initial_centroid);

	void init_normal_estimation();

	void get_normal(int idx, pcl::Normal& n) const;

	static cv::Rect2i bounding_box_2d(const visual_input& input, const pcl::PointCloud<visual_input::PointT>& cloud);

};



	
/**
 * @class img_segment
 * @brief 2D image segment representation
 *
 * Manages 2D image segments with support for 3D property computation
 * and background handling.
 *
 * Features:
 * - 2D segment management
 * - 3D property computation
 * - Background handling
 */
class HANDPOSEESTIMATION_API img_segment {
public:
	using Ptr = std::shared_ptr<img_segment>;
	using ConstPtr = std::shared_ptr<const img_segment>;


	cv::Mat mask;
	std::vector<cv::Point> contour;
	cv::Rect bounding_box;
	cv::Rect model_box_2d;
	Eigen::AlignedBox3f model_box_3d;

	std::chrono::duration<float> timestamp;

	float hand_certainty = 0.5f;
	float max_net_certainty = 0.f;
	cv::Point2i palm_center_2d;


	// [left, right]
	std::vector<net_evaluation::ConstPtr> net_evals = std::vector<net_evaluation::ConstPtr>(2);
	// [left, right]
	std::vector<hand_pose_particle_instance::Ptr> particles = std::vector<hand_pose_particle_instance::Ptr>(2);

	img_segment_3d::Ptr prop_3d;
	
	void compute_properties_3d(const visual_input& input, const Eigen::Hyperplane<float, 3>& background = Eigen::Hyperplane<float, 3>(Eigen::Vector3f::Zero(), 0.f));
	void compute_properties_3d(const visual_input& input, const pcl::PointCloud<visual_input::PointT>::ConstPtr& cloud);
};




/**
 * @class hand_instance
 * @brief Complete hand instance representation
 *
 * Represents a complete hand instance with pose, segments, and timing
 * information.
 *
 * Features:
 * - Pose management
 * - Segment tracking
 * - Timing information
 * - Instance identification
 */
class HANDPOSEESTIMATION_API hand_instance {
public:
	typedef std::shared_ptr<hand_instance> Ptr;
	typedef std::shared_ptr<const hand_instance> ConstPtr;

	std::atomic<float> certainty_score = 0.5f;
	std::atomic<float> right_hand = 0.5f;
	// lock update_mutex before iterating over observation_history
	std::list<img_segment::Ptr> observation_history;
	// lock update_mutex before iterating over poses, readonly access - use add_or_update for modification
	std::list<hand_pose_particle_instance> poses;

	mutable std::mutex update_mutex;

	hand_instance(const img_segment::Ptr& observation,
		float global_certainty_score = 0.5f);

	size_t get_id() const;
	img_segment::Ptr get_segment(std::chrono::duration<float> time_seconds) const;

	/*
	 * Adds pose to poses if the last pose is older than pose.
	 * If both have the same timestamp the last pose is updated.
	 * If pose is older than the last in the list an exception is raised (unless ignore_when_old is true)
	 * An exception is raised if the update does not improve the pose error.
	 */
	void add_or_update(const hand_pose_particle_instance& pose, bool ignore_when_old = false);
};


/**
 * @class gesture_prototype
 * @brief Gesture prototype representation
 *
 * Defines and manages gesture prototypes with finger count and
 * classification capabilities.
 *
 * Features:
 * - Finger count tracking
 * - Gesture classification
 * - Parameter serialization
 */
class HANDPOSEESTIMATION_API gesture_prototype {
public:
	typedef std::shared_ptr<gesture_prototype> Ptr;
	typedef std::shared_ptr<const gesture_prototype> ConstPtr;

	gesture_prototype() = default;

	gesture_prototype(const std::string& name,
		int fingers_count,
		const std::vector<std::vector<cv::Point>>& templates);

	gesture_prototype(std::string&& name,
		int fingers_count,
		std::vector<std::vector<cv::Point>>&& templates);

	const std::string& get_name() const;
	int get_fingers_count() const;
	const std::vector<std::vector<cv::Point>>& get_templates() const;

	/**
	* Returns the hu moments for each template.
	*/
	const std::vector<std::vector<double>>& get_moments() const;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_NVP(name);
		ar& BOOST_SERIALIZATION_NVP(fingers_count);
		ar& BOOST_SERIALIZATION_NVP(templates);
	}

protected:
	std::string name;
	int fingers_count;
	std::vector<std::vector<cv::Point>> templates;
	mutable std::vector<std::vector<double>> moments;
};

/**
 * @class classification_result
 * @brief Certainty that the given @ref{img_segment} represents a @ref{gesture_prototype}
 *
 * The certainty ranges from 0 (no such gesture) to 1 (perfect match)
 */
class HANDPOSEESTIMATION_API classification_result {
public:
	const gesture_prototype::ConstPtr prototype;
	const float certainty_score;


	classification_result(const gesture_prototype::ConstPtr& prototype,
		float certainty_score = 0.f);

	classification_result() = default;

	virtual ~classification_result() = default;
};



} /* hand_pose_estimation */

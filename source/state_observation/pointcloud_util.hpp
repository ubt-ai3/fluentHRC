#pragma once

#ifndef STATE_OBSERVATION__POINTCLOUD_PREPROCESSING__HPP
#define STATE_OBSERVATION__POINTCLOUD_PREPROCESSING__HPP

#include "framework.hpp"

#include <boost/serialization/vector.hpp>
#include <eigen_serialization/eigen_serialization.hpp>

#include <opencv2/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/PolygonMesh.h>
#include <pcl/common/impl/transforms.hpp>

#include "parameter_set.hpp"
#include "workspace_objects_forward.hpp"

namespace state_observation
{

/**
 * @class object_parameters
 * @brief Defines common properties and parameters for workspace objects
 * 
 * This class stores configuration parameters that define the physical
 * properties and constraints of objects in the workspace, such as
 * height limits and base dimensions.
 * 
 * Features:
 * - Height constraints
 * - Base dimension specifications
 * - Parameter serialization
 * - Configuration management
 */
class STATEOBSERVATION_API object_parameters : public parameter_set {
public:
	typedef pcl::PointXYZRGBA PointT;

	object_parameters();
	~object_parameters() override;

	float min_object_height;
	float max_height;

	Eigen::Vector3f construction_base_dimensions;


	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(min_object_height);
		ar& BOOST_SERIALIZATION_NVP(max_height);
		ar& BOOST_SERIALIZATION_NVP(construction_base_dimensions);
	}

};

/**
 * @class computed_workspace_parameters
 * @brief Caches and manages computed workspace properties and transformations
 * 
 * Stores precomputed workspace parameters including crop boxes, construction
 * areas, and transformations. Handles both real and simulation environments.
 * 
 * Features:
 * - Workspace boundaries
 * - Transformation matrices
 * - Construction area limits
 * - Simulation mode support
 * - Parameter serialization
 */
class STATEOBSERVATION_API computed_workspace_parameters : public parameter_set{
public:
	using PointT = pcl::PointXYZRGBA;

	computed_workspace_parameters(bool simulation);
	~computed_workspace_parameters() override;

	Eigen::Vector3f crop_box_min;
	Eigen::Vector3f crop_box_max;

	Eigen::Vector3f construction_area_min;
	Eigen::Vector3f construction_area_max;

	// transformations applied after bringing the table
	// into x-y plane

	Eigen::Affine3f transformation;
	Eigen::Affine3f simulation_transformation;

	bool simulation;

	float max_object_dimension;

	[[nodiscard]] Eigen::Affine3f get_cloud_transformation() const;
	[[nodiscard]] Eigen::Affine3f get_inv_cloud_transformation() const;

	[[nodiscard]] Eigen::Hyperplane<float, 3> get_plane() const;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) { 
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);

		ar& BOOST_SERIALIZATION_NVP(crop_box_min);
		ar& BOOST_SERIALIZATION_NVP(crop_box_max);
		ar& BOOST_SERIALIZATION_NVP(construction_area_min);
		ar& BOOST_SERIALIZATION_NVP(construction_area_max);
		ar& BOOST_SERIALIZATION_NVP(transformation);
		ar& BOOST_SERIALIZATION_NVP(simulation_transformation);
		ar& BOOST_SERIALIZATION_NVP(max_object_dimension);
	}
};






/**
 * @class pointcloud_preprocessing
 * @brief Utility class for point cloud processing and manipulation
 * 
 * Provides a comprehensive set of tools for processing point cloud data,
 * including table removal, segmentation, clustering, and geometric
 * transformations.
 * 
 * Features:
 * - Table plane detection and removal
 * - Supervoxel clustering
 * - Conditional Euclidean clustering
 * - Segment extraction
 * - Point cloud fusion
 * - Mesh transformation
 * - Depth ramp removal
 * - Normal estimation
 */
class STATEOBSERVATION_API pointcloud_preprocessing {
public:

	pointcloud_preprocessing(const std::shared_ptr<const object_parameters>& object_params, bool simulation);


	typedef pcl::PointXYZRGBA PointT;
	typedef std::map<uint32_t, std::shared_ptr<pcl::Supervoxel<pointcloud_preprocessing::PointT>>> cluster_map;
	typedef std::multimap<uint32_t, uint32_t> adjacency_map;

	inline static const unsigned int min_points_per_object = 12;


	/**
	* Estimates the normals of @ref{input}
	*/
	[[nodiscard]] pcl::PointCloud<pcl::PointNormal>::ConstPtr normals(const pcl::PointCloud<PointT>::ConstPtr& input) const;
	
	/**
	* Uses @ref{computed_workspace_params} to remove the table and transform the point cloud such that the table is in the xy-plane
	* If no values for @ref{computed_workspace_params} are provided, they are computed by applying RANSAC to determine the largest plane. 
	*/
	pcl::PointCloud<PointT>::ConstPtr remove_table(const pcl::PointCloud<PointT>::ConstPtr& input);
	
	/**
	* Returns the transformation applied to input when calling remove_table().
	* In case @ref{workspace_params} are not loaded, the identity matrix is returned.
	*/
	[[nodiscard]] Eigen::Affine3f get_cloud_transformation() const;

	/**
	* Returns the inverse of get_cloud_transformation().
	*/
	[[nodiscard]] Eigen::Affine3f get_inv_cloud_transformation() const;

	[[nodiscard]] pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr super_voxel_clustering(const pcl::PointCloud<PointT>::ConstPtr& input) const;

	[[nodiscard]] pcl::IndicesClustersPtr conditional_euclidean_clustering(const pcl::PointCloud<PointT>::ConstPtr& input) const;

	/**
	* Performs Conditional Euclidean Clustering and some post-processing to fill the @ref{pc_segment} data structure
	*/
	[[nodiscard]] std::vector<segmentPtr> extract_segments(const pcl::PointCloud<PointT>::ConstPtr& input) const;

	/**
	* Projects all points onto the plane specified in @ref{computed_workspace_params}
	*/
	[[nodiscard]] pcl::PointCloud<PointT>::ConstPtr project_on_table(const pcl::PointCloud<PointT>::ConstPtr& input) const;

	/**
	* Projects all points onto the plane specified in @ref{computed_workspace_params}
	*/
	[[nodiscard]] std::vector<cv::Point2f> project_on_table_cv(const pcl::PointCloud<PointT>::ConstPtr& input) const;

	/**
	* Computes the oriented bounding box for a set of points assuming that the represented object is standing on the xy-plane
	*/
	[[nodiscard]] obb oriented_bounding_box_for_standing_objects(const pcl::PointCloud<PointT>::ConstPtr& input) const;

	/**
	* Assigns color to all vertices of the mesh
	*/
	static pcl::PolygonMesh::Ptr color(const pcl::PolygonMesh::ConstPtr& input, const pcl::RGB& color);

	/**
	* Assumes that input is contained in NDC, i.e. contained in [-1,1]³. 
	* Transforms (scale, rotate, translate) input to fit into @ref{bounding_box}
	*/
	static pcl::PolygonMesh::Ptr transform(const pcl::PolygonMesh::ConstPtr& input, const obb& bounding_box);

	/**
	* Applies a transformation to a polygon mesh
	*/
	static pcl::PolygonMesh::Ptr transform(const pcl::PolygonMesh::ConstPtr& input, const Eigen::Affine3f& matrix);

	/**
	* Outputs a point cloud where each point has the mean z-value of the input clouds
	*/
	[[nodiscard]] pcl::PointCloud<PointT>::ConstPtr fuse(const pcl::PointCloud<PointT>::ConstPtr& cloud1, const pcl::PointCloud<PointT>::ConstPtr& cloud2) const;

	/**
	* Converts a point cloud
	*/
	static pcl::PointCloud<PointT>::ConstPtr to_pc_rgba(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& pc_rgb);

	/**
	* Runs Moving Least Squares
	*/
	[[nodiscard]] std::pair<pcl::PointCloud<pointcloud_preprocessing::PointT>::ConstPtr, pcl::PointCloud<pcl::PointNormal>::Ptr>
		smooth(const pcl::PointCloud<PointT>::ConstPtr& input) const;

	/**
	* Removes ramps at the boundaries of an object that occur when the depth sensor interpolates
	* between foreground (object) and background
	* If indices_to_indices is true, the returned indices refer to entries in object_indices, otherwise to points in cloud.
	*/
	[[nodiscard]] pcl::PointIndices::Ptr remove_depth_ramps(const pcl::PointCloud<PointT> cloud,
		const pcl::PointIndices& object_indices,
		bool indices_to_indices = false) const;

	/**
	* Parameter sets
	*/
	computed_workspace_parameters workspace_params;
	const std::shared_ptr<const object_parameters> object_params;
		
private:


	[[nodiscard]] pcl::PointCloud<PointT>::ConstPtr filter_nan(const pcl::PointCloud<PointT>::ConstPtr& input) const;

	[[nodiscard]] pcl::PointCloud<PointT>::ConstPtr crop(const pcl::PointCloud<PointT>::ConstPtr& input) const;

	static bool AA(const PointT& pIn)
	{
		return
			std::isfinite(pIn.x) &&
			std::isfinite(pIn.y) &&
			std::isfinite(pIn.z) &&
			pIn.z != 0.f;
	}
	static PointT tfA(pcl::detail::Transformer<float> tf, PointT pIn)
	{
		PointT p;
		tf.se3(pIn.data, p.data);
		return p;
	}
	
	void add_point_size_sample(size_t point_size);

	size_t samples = 0;
	size_t average_points = 1000; //startup with 1000
	const size_t MAX_SAMPLES = 10;
};


//samples (almost) evenly distributed points on a triangle
//@return the 3 barycentric points, describing the point on the triangle
//@param u the sampling parameter in interval [0,1)
std::array<float, 3> LowDiscrepancySampleTriangle(float u);


/**
 * @class occlusion_detector
 * @brief Detects and analyzes occlusions in point cloud data
 * 
 * Analyzes point cloud data to detect when objects are occluded,
 * present, or have disappeared from view. Supports both point cloud
 * and bounding box-based detection.
 * 
 * Features:
 * - Occlusion detection
 * - Reference cloud management
 * - View projection computation
 * - Intersection testing
 * - Actor-based occlusion checking
 */
class STATEOBSERVATION_API occlusion_detector
{
public:

	using PointT = pcl::PointXYZRGBA;
	using Ptr = std::shared_ptr<occlusion_detector>;
	enum result
	{
		COVERED,
		PRESENT,
		DISAPPEARED
	};
	/**
		@param object_params, specifies the allowed room for error in detection object_params.min_object_height
		@param the view_projection matrix of the camera used to calculate the occlusion from, 
		can be calculated using occlusion_detector::compute_view_projection 
	*/
	occlusion_detector(
		float min_object_height,
		const Eigen::Matrix4f& camera_projection);

	virtual ~occlusion_detector() = default;


	[[nodiscard]] bool has_reference_cloud() const;

	// check if reference cloud exists and is not empty
	[[nodiscard]] bool has_valid_reference_cloud() const;

	/*
		sets the base cloud to compare against
		all points should lie in front of the camera within field of view
	*/
	void set_reference_cloud(const pcl::PointCloud<PointT>::ConstPtr& cloud);

	/*
	looks if the object represented by @param cloud or @param obox
	is present in the reference_cloud set with @set_reference_cloud
	ignores colors
	*/
	[[nodiscard]] result perform_detection(const pcl::PointCloud<occlusion_detector::PointT>& cloud) const;

	[[nodiscard]] result perform_detection(const obb& obox) const;

	/*
		converts the front facing part(as seen by the camera) of the obb into a mesh by sampling the front surface
	*/
	static pcl::PointCloud<pcl::PointXYZ>::Ptr convert_to_point_cloud_surface(const obb& obox,const Eigen::Matrix4f& camera_projection);

	/*
		computes a OpenGl view projection matrix, field of view and clipping planes don't really matter for occlusion_detetction 
	*/
	static Eigen::Matrix4f compute_view_projection(const Eigen::Vector3f& cam_pos, const Eigen::Vector3f& look_at);

	//struct occlusion_result
	//{
	//	float disappeared_pct;
	//	float occluded_pct;
	//	float present_pct;
	//	float near_point_pct;
	//};

	//struct occlusion_visualization_data
	//{
	//	using cloud_ptr = std::shared_ptr<pcl::PointCloud<PointT>>;
	//	cloud_ptr occluded;
	//	cloud_ptr disappeared;
	//	cloud_ptr present;
	//};

	static occlusion_detector::Ptr construct_from(const pointcloud_preprocessing& prepro, const object_parameters& params);

	static bool intersect(Eigen::Vector3f p1, Eigen::Vector3f p2, const std::array<Eigen::Vector3f, 3>& triangle);
	static bool intersect(Eigen::Vector3f p1, Eigen::Vector3f p2, const std::array<Eigen::Vector3f, 4>& quad);

	float covered_threshold = 0.1f;
	float present_threshold = 0.5f;
	//float disappeared_threshold = 0.1;
	float height_tolerance;

	// The depth camera intrinsic pointcloud smoothing leads to points located between objects (e.g. robot and table)
	// If we find a high enough point in the proximity of an object, we assume that we have a fake object created by the smoothing
	// The min_hand_height determines which z distance we consider high enough and the max_shadow_extent how far away from the object we search
	float min_hand_height = 0.12f;
	float max_shadow_extent = 0.02f;

private:

	float max_squared_distance = 0.001f; 
	
	Eigen::Matrix4f camera_projection_;

	pcl::KdTreeFLANN<pcl::PointXY> kd_tree_;
	pcl::PointCloud<pcl::PointXY>::Ptr reference_cloud_2D_;
	pcl::PointCloud<PointT>::ConstPtr reference_cloud_;

	mutable std::mutex kd_tree_mutex_;

protected:
	[[nodiscard]] virtual bool occluded_by_actor(const pcl::PointXY& p) const;
};
   
} // namespace state_observation
	
#endif /* !STATE_OBSERVATION__POINTCLOUD_PREPROCESSING__HPP */
#pragma once

#ifndef STATE_OBSERVATION__WORKSPACE_OBJECTS__HPP
#define STATE_OBSERVATION__WORKSPACE_OBJECTS__HPP

#include "framework.hpp"

#include "workspace_objects_forward.hpp"

#include <vector>

#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/string.hpp>

#include "eigen_serialization/eigen_serialization.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/auto_io.h>
#include <pcl/PointIndices.h>
#include <pcl/PolygonMesh.h>

#include "pointcloud_util.hpp"
#include "parameter_set.hpp"

#include "enact_core/id.hpp"

#define DEBUG_OBJECT_INSTANCE_ID

namespace state_observation
{

/**
 * @class aabb
 * @brief Axis-aligned bounding box representation
 * 
 * Represents an axis-aligned bounding box (AABB) in 3D space, defined by its
 * diagonal and translation. Provides methods for computing top, bottom, and center
 * positions along the z-axis, and for constructing from corner points.
 * 
 * Features:
 * - Diagonal and translation storage
 * - Top, bottom, and center z computation
 * - Serialization support
 * - Construction from corners
 */
class STATEOBSERVATION_API aabb {
public:
	Eigen::Vector3f diagonal;
	Eigen::Vector3f translation;

	/*
	 * calculate top and bottom z since the diagonal and translation 
	 * are subject to arbitrary changes
	 */
	virtual float top_z() const;
	virtual float bottom_z() const;
	virtual float center_z() const;

	aabb(const Eigen::Vector3f& diagonal,
		const Eigen::Vector3f& translation = Eigen::Vector3f::Zero());

	aabb();

	static aabb from_corners(const Eigen::Vector3f& min_point, const Eigen::Vector3f& max_point);

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_NVP(translation);
		ar& BOOST_SERIALIZATION_NVP(diagonal);
	}
};

/**
 * @class obb
 * @brief Oriented bounding box representation
 * 
 * Extends aabb to support arbitrary orientation using a quaternion rotation.
 * Useful for representing objects that are not aligned with the coordinate axes.
 * 
 * Features:
 * - Quaternion-based rotation
 * - Corner computation
 * - Inherited aabb features
 * - Serialization support
 */
class STATEOBSERVATION_API obb : public aabb{
public:
	Eigen::Quaternionf rotation;

	obb();

	obb(const Eigen::Vector3f& diagonal,
		const Eigen::Vector3f& translation = Eigen::Vector3f::Zero(),
		const Eigen::Quaternionf& rotation = Eigen::Quaternionf::Identity());

	obb(const aabb& box);
	
	float top_z() const override;
	float bottom_z() const override;
	float center_z() const override;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(aabb);
		ar& BOOST_SERIALIZATION_NVP(rotation);
	}

	std::vector<Eigen::Vector3f> get_corners() const;
};


/**
 * @class mesh_wrapper
 * @brief Mesh loading and serialization utility
 * 
 * Provides transparent loading and serialization of 3D meshes based on file paths.
 * Supports lazy loading and caching of mesh data for efficient access.
 * 
 * Features:
 * - File path management
 * - Mesh loading and caching
 * - Serialization support
 */
class STATEOBSERVATION_API mesh_wrapper {
public:
	typedef std::shared_ptr<mesh_wrapper> Ptr;
	typedef std::shared_ptr<const mesh_wrapper> ConstPtr;

	mesh_wrapper() = default;
	mesh_wrapper(const std::string& path);

	void set_path(const std::string& path);
	std::string get_path() const;

	pcl::PolygonMesh::ConstPtr load_mesh() const;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_NVP(path);
	}

protected:
	std::string path;
	mutable pcl::PolygonMesh::Ptr mesh;
};


/**
 * @class pc_segment
 * @brief Represents a segmented subset of a point cloud
 * 
 * Encapsulates a segment of a point cloud, including its points, indices,
 * reference frame, color, bounding box, centroid, and timestamp. Used for
 * object and workspace analysis.
 * 
 * Features:
 * - Point and index storage
 * - Reference frame association
 * - Bounding box and centroid computation
 * - Mean color calculation
 * - Outline extraction
 */
class STATEOBSERVATION_API pc_segment {
public:
	typedef pcl::PointXYZRGBA PointT;
	typedef std::shared_ptr<pc_segment> Ptr;
	typedef std::shared_ptr<const pc_segment> ConstPtr;

	static const std::shared_ptr<enact_core::aspect_id> aspect_id;

	pcl::PointCloud<PointT>::ConstPtr points;
	pcl::PointIndices::ConstPtr indices;
	pcl::PointCloud<PointT>::ConstPtr reference_frame;
	pcl::RGB mean_color;
	obb bounding_box;
	PointT centroid;
	std::chrono::duration<float> timestamp;

	std::vector<classification_result> classification_results;

	pc_segment(const pcl::PointCloud<PointT>::ConstPtr& points,
		const pcl::PointIndices::ConstPtr& indices,
		const pcl::PointCloud<PointT>::ConstPtr& reference_frame,
		const obb& bounding_box,
		const PointT& centroid,
		std::chrono::duration<float> timestamp);

	pc_segment();


	/**
	* Computes the outline of input projected onto the xy-plane
	*/
	pcl::PointCloud<PointT>::ConstPtr get_outline() const;
	float get_outline_area() const;

	void compute_mean_color();

private:
	mutable pcl::PointCloud<PointT>::ConstPtr outline;
	mutable float outline_area;
};


/**
 * @class object_prototype
 * @brief Represents a class of indistinguishable real-world objects
 * 
 * Defines the prototype for a class of objects, including bounding box,
 * color, mesh, name, and type. Used for object classification and recognition.
 * 
 * Features:
 * - Bounding box and color storage
 * - Mesh association
 * - Name and type metadata
 * - Serialization support
 */
class STATEOBSERVATION_API object_prototype {
public:
	typedef std::shared_ptr<object_prototype> Ptr;
	typedef std::shared_ptr<const object_prototype> ConstPtr;

	object_prototype() = default;

	object_prototype(const aabb& bounding_box,
					 const pcl::RGB& mean_color,
					 const mesh_wrapper::Ptr& base_mesh,
					 const std::string& name = std::string(),
					 const std::string& type = std::string());

	virtual ~object_prototype() = default;

	aabb get_bounding_box() const;
	pcl::RGB get_mean_color() const;
	pcl::PolygonMesh::ConstPtr load_mesh() const;

	const mesh_wrapper::ConstPtr get_base_mesh() const;
	bool has_mesh() const;
	const std::string& get_name() const;
	const std::string& get_type() const;

	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_NVP(bounding_box);
		ar& BOOST_SERIALIZATION_NVP(mean_color.r);
		ar& BOOST_SERIALIZATION_NVP(mean_color.g);
		ar& BOOST_SERIALIZATION_NVP(mean_color.b);
		ar& BOOST_SERIALIZATION_NVP(mean_color.a);
		ar& BOOST_SERIALIZATION_NVP(base_mesh);
		ar& BOOST_SERIALIZATION_NVP(name);
		ar& BOOST_SERIALIZATION_NVP(type);
	}



protected:
	aabb bounding_box;
	pcl::RGB mean_color;
	mesh_wrapper::Ptr base_mesh;
	mutable pcl::PolygonMesh::ConstPtr object_prototype_mesh;
	std::string name;
	std::string type;
};

/**
 * @class classification_result
 * @brief Represents the result of object classification
 * 
 * Stores the classification results for an object, including the prototype,
 * confidence score, and instance information. Used for object recognition
 * and tracking.
 * 
 * Features:
 * - Prototype association
 * - Confidence scoring
 * - Instance tracking
 * - Serialization support
 */
class STATEOBSERVATION_API classification_result {
public:

	object_prototype::ConstPtr prototype;
	Eigen::Quaternionf prototype_rotation;
	float local_certainty_score;

	classification_result(const object_prototype::ConstPtr& prototype,
						  const Eigen::Quaternionf prototype_rotation = Eigen::Quaternionf::Identity(),
						  const float local_certainty_score = 0.f);

	classification_result() = default;

};

/**
 * @class object_instance
 * @brief Represents a specific instance of an object in the workspace
 * 
 * Tracks a specific instance of an object, including its classification,
 * segment information, and background status. Used for object tracking
 * and state management.
 * 
 * Features:
 * - Classification result storage
 * - Segment association
 * - Background detection
 * - Instance tracking
 */
class STATEOBSERVATION_API object_instance {
public:
	typedef std::shared_ptr<object_instance> Ptr;
	typedef std::shared_ptr<const object_instance> ConstPtr;
	typedef std::weak_ptr<enact_core::entity_id> id;

	/*
* If the history is shorter, the object is new or it was not seen nor occluded recently
* If the history has equal length, the object stabely detected, i.e. does not change appeareance over time
* If the history time is longer, the object changes appeareance. Older segments are removed after purge_duration
*/
	inline static const int observation_history_intended_length = 3;

#ifdef DEBUG_OBJECT_INSTANCE_ID
	static int id_counter;
	int ID;
#endif

	bool covered = false;
	const object_prototype::ConstPtr prototype;
	float global_certainty_score;
	//@see observation_history_intended_length
	std::list<pc_segment::Ptr> observation_history;

	static const std::shared_ptr<enact_core::aspect_id> aspect_id;

	object_instance(const pc_segment::Ptr& seg);
	object_instance() = default;
	
	pc_segment::Ptr get_classified_segment() const;

	bool is_background() const;
	
};

} // namespace state_observation

#endif // !STATE_OBSERVATION__WORKSPACE_OBJECTS__HPP

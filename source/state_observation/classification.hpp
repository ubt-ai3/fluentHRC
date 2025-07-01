#pragma once

#ifndef STATE_OBSERVATION__CLASSIFICATION__HPP
#define STATE_OBSERVATION__CLASSIFICATION__HPP

#include "framework.hpp"

#include <memory>

#include <boost/serialization/access.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "eigen_serialization/eigen_serialization.hpp"
#include <Eigen/Core>

#include <pcl/point_types.h>


#include "workspace_objects.hpp"
#include "pointcloud_util.hpp"

namespace state_observation
{

/**
 * @class classifier
 * @brief Abstract base class for object classification
 * 
 * Provides an interface for determining whether a point cloud segment
 * corresponds to a given object prototype. Supports certainty estimation,
 * prototype retrieval, and parameter management.
 * 
 * Features:
 * - Abstract classification interface
 * - Prototype association
 * - Certainty estimation
 * - Parameter management
 */
class STATEOBSERVATION_API classifier
{
public:
	typedef pcl::PointXYZRGBA PointT;
	typedef std::shared_ptr<classifier> Ptr;
	typedef std::shared_ptr<const classifier> ConstPtr;

	classifier() = default;
	classifier(const object_prototype::ConstPtr& prototype);
	virtual ~classifier() = default;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_NVP(prototype);
	}

	/**
	* Returns the certainty that @ref{seg} is of object prototype specified by
	* @ref{get_object_prototype}
	*/
	virtual classification_result classify(const pc_segment& seg) const = 0;

	/**
	*  Returns the certainty that @ref{seg} contains the specifed object prototype
	* Succeeds, if at least two dimensions of the bounding box are approximatly equal to object prototype
	*/
//	virtual float contains_prototype(const pc_segmentbounding_box seg) const = 0;

	/**
	* Returns the object prototype classified by this.
	*/
	virtual object_prototype::ConstPtr get_object_prototype() const;

	/**
	* Computes a rotation fitting @ef{prototype} to @ref{seg}
	*/
//	virtual Eigen::Quaternionf rotation_guess(const pc_segment& seg) const = 0;

	static void set_object_params(const std::shared_ptr<const object_parameters>& object_params);

protected:
	friend class boost::serialization::access;

	object_prototype::ConstPtr prototype;
	static std::weak_ptr<const object_parameters> object_params;

	static float bell_curve(float x, float stdev);
	static double bell_curve(double x, double stdev);
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(classifier)






 /**
  * @class bounding_box_classifier
  * @brief Classifier based on bounding box similarity
  * 
  * Implements classification logic using geometric properties of bounding boxes.
  * Supports rotation estimation, similarity matrix computation, and feasible
  * transformation generation.
  * 
  * Features:
  * - Bounding box-based classification
  * - Rotation and similarity estimation
  * - Feasible transformation generation
  */
class STATEOBSERVATION_API bounding_box_classifier : virtual public classifier
{
public:
	bounding_box_classifier(const object_prototype::ConstPtr& prototype);
	bounding_box_classifier() = default;
	virtual ~bounding_box_classifier() = default;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(classifier);
	}

	/**
	* Returns the certainty that @ref{seg} is of object prototype specified by
	* @ref{get_object_prototype}
	*/
	virtual classification_result classify(const pc_segment& seg) const override;

	/**
	*  Returns the certainty that @ref{seg} contains the specifed object prototype
	* Succeeds, if at least two dimensions of the bounding box are approxematly equal to object prototype
	*/
//	virtual float contains_prototype(const pc_segment& seg) const override;

	/**
	* Greedily computes the best rotation to fit this.bounding_box to seg.bounding_box (untransformed)
	*/
	virtual Eigen::Quaternionf rotation_guess(const pc_segment& seg) const;

	/**
	* Greedily computes the best rotation to fit this.bounding_box to seg.bounding_box assuming there
	* is an object below seg.
	*/
	virtual Eigen::Quaternionf stacked_rotation_guess(const pc_segment& seg) const;

	/**
	* An entry a_ij \in [0,1] of the returned matrix states whether the BB of seg 
	* in dimension i has approximately the same length as the BB of this in dimension j
	*/
	virtual Eigen::Matrix3f similarity_matrix(const pc_segment& seg) const;

		/**
	* An entry a_ij \in [0,1] of the returned matrix states whether the BB of seg
	* in dimension i has approximately the same length as the BB of this in dimension j
	*/
	virtual Eigen::Matrix3f stacked_similarity_matrix(const pc_segment& seg) const;
	
	/**
	* generates all feasible transformations for @ref{seg}
	* based on the BBs of both objects. A transformation is feasible if
	* - the length of the bounding boxes of @ref{prototype} and @ref{seg} approximately match in every dimension
	* - the vector contains no other transformation that leads to a symmetrical result
	*
	* The basic implementation assumes symmetrical objects.
	*/
	virtual std::vector<Eigen::Quaternionf> get_feasible_transformations(const pc_segment& seg) const;
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(bounding_box_classifier)


/**
 * @class shape_classifier
 * @brief Classifier using polygon mesh shape comparison
 * 
 * Extends bounding_box_classifier to compare point cloud segments against
 * polygon mesh prototypes. Supports shape matching and geometric distance
 * calculations.
 * 
 * Features:
 * - Mesh-based shape comparison
 * - Shape matching
 * - Geometric distance computation
 */
class STATEOBSERVATION_API shape_classifier : public bounding_box_classifier
{
public:
	shape_classifier(const object_prototype::ConstPtr& prototype);
	shape_classifier() = default;
	virtual ~shape_classifier() = default;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(bounding_box_classifier);
	}

	/**
	* Returns the certainty that @ref{seg} is of object prototype specified by
	* @ref{get_object_prototype}
	*/
	virtual classification_result classify(const pc_segment& seg) const override;

	/**
	*  Returns the certainty that @ref{seg} contains the specifed object prototype
	* Succeeds, if at least two dimensions of the bounding box are approxematly equal to object prototype
	*/
//	virtual float contains_prototype(const pc_segment& seg) const override;

	/**
	* Returns the certainty that @ref{seg} resamples prototype oriented according to @ref{prototype_rotation} 
	*/
	virtual float match(const pc_segment& seg, Eigen::Quaternionf prototype_rotation) const;

	/**
	* Computes the distance from p0 to the triangle p1,p2,p3
	*/
	float distance(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3) const;

	/**
	* Projects point q0 onto line p1,p2
	*/
	Eigen::Vector3f project(const Eigen::Vector3f& q0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) const;
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(shape_classifier)




/**
 * @class monochrome_object_classifier
 * @brief Classifier for monochrome objects
 * 
 * Specializes the classifier for objects with uniform color properties.
 * Implements classification logic based on color similarity.
 * 
 * Features:
 * - Monochrome object classification
 * - Color-based certainty estimation
 */
class STATEOBSERVATION_API monochrome_object_classifier : virtual public classifier
{
public:
	monochrome_object_classifier(const object_prototype::ConstPtr& prototype);
	monochrome_object_classifier() = default;
	virtual ~monochrome_object_classifier();

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(classifier);
	}

	virtual classification_result classify(const pc_segment& seg) const override;


};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(monochrome_object_classifier)


/**
 * @class background_classifier
 * @brief Classifier for background objects
 * 
 * Specializes the monochrome object classifier for background elements.
 * Implements specific classification logic for identifying background
 * objects in the scene.
 * 
 * Features:
 * - Background object classification
 * - Color-based certainty estimation
 * - Background-specific logic
 */
class STATEOBSERVATION_API background_classifier : public monochrome_object_classifier
{
public:

	background_classifier() = default;
	background_classifier(const object_prototype::ConstPtr& prototype);
	virtual ~background_classifier() = default;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(classifier);
	}

	/**
	* Returns the certainty that @ref{seg} is of object prototype specified by
	* @ref{get_object_prototype}
	*/
	virtual classification_result classify(const pc_segment& seg) const override;

	/**
	*  Returns the certainty that @ref{seg} contains the specifed object prototype
	* Succeeds, if at least two dimensions of the bounding box are approxematly equal to object prototype
	*/
//	virtual float contains_prototype(const pc_segment& seg) const override;


};



/**
 * @class cuboid_classifier
 * @brief Classifier for cuboid objects
 * 
 * Combines shape and monochrome classification for cuboid objects.
 * Implements specific classification logic for rectangular prisms.
 * 
 * Features:
 * - Cuboid shape recognition
 * - Color-based classification
 * - Geometric property matching
 */
class STATEOBSERVATION_API cuboid_classifier : public shape_classifier, public monochrome_object_classifier
{
public:

	cuboid_classifier(const object_prototype::ConstPtr& prototype);
	cuboid_classifier() = default;
	virtual ~cuboid_classifier() = default;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(bounding_box_classifier);
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(monochrome_object_classifier);
	}

	virtual classification_result classify(const pc_segment& seg) const override;
};


/**
 * @class cylinder_classifier
 * @brief Classifier for cylindrical objects
 * 
 * Combines shape and monochrome classification for cylindrical objects.
 * Implements specific classification logic for cylinders.
 * 
 * Features:
 * - Cylinder shape recognition
 * - Color-based classification
 * - Geometric property matching
 */
class STATEOBSERVATION_API cylinder_classifier : public shape_classifier, public monochrome_object_classifier
{
public:

	cylinder_classifier(const object_prototype::ConstPtr& prototype);
	cylinder_classifier() = default;
	virtual ~cylinder_classifier() = default;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(bounding_box_classifier);
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(monochrome_object_classifier);
	}

	virtual classification_result classify(const pc_segment& seg) const override;
};


/**
 * @class semicylinder_classifier
 * @brief Classifier for semi-cylindrical objects
 * 
 * Combines shape and monochrome classification for semi-cylindrical objects.
 * Implements specific classification logic for half-cylinders.
 * 
 * Features:
 * - Semi-cylinder shape recognition
 * - Color-based classification
 * - Geometric property matching
 * - Transformation feasibility checking
 */
class STATEOBSERVATION_API semicylinder_classifier : public shape_classifier, public monochrome_object_classifier
{
public:

	semicylinder_classifier(const object_prototype::ConstPtr& prototype);
	semicylinder_classifier() = default;
	virtual ~semicylinder_classifier() = default;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(bounding_box_classifier);
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(monochrome_object_classifier);
	}

	virtual classification_result classify(const pc_segment& seg) const override;

	virtual std::vector<Eigen::Quaternionf> get_feasible_transformations(const pc_segment& seg) const override;
};


/**
 * @class bridge_classifier
 * @brief Classifier for bridge-like objects
 * 
 * Extends cuboid classification for bridge-like structures.
 * Implements specific classification logic for bridge objects.
 * 
 * Features:
 * - Bridge shape recognition
 * - Color-based classification
 * - Geometric property matching
 * - Transformation feasibility checking
 */
class STATEOBSERVATION_API bridge_classifier : public cuboid_classifier
{
public:

	bridge_classifier(const object_prototype::ConstPtr& prototype);
	bridge_classifier() = default;
	virtual ~bridge_classifier() = default;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(bounding_box_classifier);
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(monochrome_object_classifier);
	}

	virtual classification_result classify(const pc_segment& seg) const override;

	virtual std::vector<Eigen::Quaternionf> get_feasible_transformations(const pc_segment& seg) const override;
};


/**
 * @class triangular_prism_classifier
 * @brief Classifier for triangular prism objects
 * 
 * Combines shape and monochrome classification for triangular prisms.
 * Implements specific classification logic for triangular prism objects.
 * 
 * Features:
 * - Triangular prism shape recognition
 * - Color-based classification
 * - Geometric property matching
 * - Transformation feasibility checking
 */
class STATEOBSERVATION_API triangular_prism_classifier : public shape_classifier, public monochrome_object_classifier
{
public:

	triangular_prism_classifier(const object_prototype::ConstPtr& prototype);
	triangular_prism_classifier() = default;
	virtual ~triangular_prism_classifier() = default;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(bounding_box_classifier);
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(monochrome_object_classifier);
	}

	virtual classification_result classify(const pc_segment& seg) const override;

	virtual std::vector<Eigen::Quaternionf> get_feasible_transformations(const pc_segment& seg) const override;

	float slant_height;
};

} // namespace state_observation

#endif // !STATE_OBSERVATION__CLASSIFICATION__HPP

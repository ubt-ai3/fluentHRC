#pragma once

#include "framework.h"

#include <memory>

#include <boost/serialization/access.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "opencv-serialization/opencv_serialization.hpp"


#include <pcl/point_types.h>


#include "hand_model.hpp"


namespace hand_pose_estimation
{


/**
 *************************************************************************
 *
 * @class classifier
 *
 * Tells whether a @ref{img_segment} is of an @ef{gesture_prototype}
 *
 ************************************************************************/
class HANDPOSEESTIMATION_API classifier
{
public:
	typedef pcl::PointXYZRGBA PointT;
	typedef boost::shared_ptr<classifier> Ptr;
	typedef boost::shared_ptr<const classifier> ConstPtr;

	classifier() = default;
	classifier(const gesture_prototype::ConstPtr& prototype);
	virtual ~classifier();



	/**
	* Returns the certainty that @ref{seg} is of object prototype specified by
	* @ref{get_object_prototype}
	*/
	virtual classification_result classify(const hand_instance& seg) const = 0;

	/**
	* Returns the object prototype classified by this.
	*/
	virtual gesture_prototype::ConstPtr get_object_prototype() const;


protected:
	friend class boost::serialization::access;

	gesture_prototype::ConstPtr prototype;

	static float bell_curve(float x, float stdev);
	static double bell_curve(double x, double stdev);

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_NVP(prototype);
	}
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(classifier)




/**
 *************************************************************************
 *
 * @class shape_classifier
 *
 * Tests the contour against the vector of template contours
 *
 ************************************************************************/
class HANDPOSEESTIMATION_API shape_classifier : virtual public classifier
{
public:
	shape_classifier(const gesture_prototype::ConstPtr& prototype);
	shape_classifier() = default;
	virtual ~shape_classifier();



	/**
	* Returns the certainty that @ref{seg} is of object prototype specified by
	* @ref{get_object_prototype}
	*/
	virtual classification_result classify(const hand_instance& seg) const override;


	/**
	* Returns the certainty that @ref{seg} resamples prototype oriented according to @ref{prototype_rotation} 
	*/
	static float match(const std::vector<double>& lhs, const std::vector<double>& rhs);

protected:
	friend class boost::serialization::access;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(classifier);
	}
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(shape_classifier)




/**
 *************************************************************************
 *
 * @class stretched_fingers_classifier
 *
 *
 ************************************************************************/

class HANDPOSEESTIMATION_API stretched_fingers_classifier : public shape_classifier
{
public:

	stretched_fingers_classifier(const gesture_prototype::ConstPtr& prototype);
	stretched_fingers_classifier() = default;
	virtual ~stretched_fingers_classifier();

protected:
	friend class boost::serialization::access;

	int fingers_count;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(shape_classifier);
	}

	virtual classification_result classify(const hand_instance& seg) const override;
};




} /* hand_pose_estimation */


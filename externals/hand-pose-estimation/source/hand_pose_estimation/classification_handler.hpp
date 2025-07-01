#pragma once

#include "framework.h"

#include <vector>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "parameter_set.hpp"
#include "classification.hpp"

namespace hand_pose_estimation
{


/**
 *************************************************************************
 *
 * @class classifier_set
 *
 * Stores and loads @ref{classifier}s from the disk
 *
 ************************************************************************/

class HANDPOSEESTIMATION_API classifier_set : public parameter_set
{
public:
	typedef std::vector<classifier::Ptr>::iterator iterator;
	typedef std::vector<classifier::Ptr>::const_iterator const_iterator;

	classifier_set();
	~classifier_set();

	iterator begin();
	iterator end();

	const_iterator begin() const;
	const_iterator end() const;

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(classifiers);
	}

protected:
	std::vector<classifier::Ptr> classifiers;

	template<typename Archive>
	void register_classifiers(Archive& ar) const
	{
		ar.register_type<stretched_fingers_classifier>();
	};
};

} // namespace hand_pose_estimation

#pragma once

#ifndef STATE_OBSERVATION__PARAMETER_SET__HPP
#define STATE_OBSERVATION__PARAMETER_SET__HPP

#include "framework.hpp"

#include <string>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/assume_abstract.hpp>

namespace state_observation
{

/**
* loads values from file when creating a class instance and writes them back when destructing the class
* to define a new parameter set, derive from this class and implement:
*   static const std::string filename_ //a unique name for the file storing the values
*   template <typename Archive>	void serialize(Archive& ar, const unsigned int version) // handles serialization and deserialization
*   a constructor to read the file using boost::archive::text_iarchive
*/
class STATEOBSERVATION_API parameter_set {
public:
	parameter_set();

	virtual ~parameter_set();

	template<typename Archive>
	void serialize(Archive& ar, const unsigned version)
	{}

protected:
	friend class boost::serialization::access;

	std::string filename_;
	std::string folder_;

};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(parameter_set)

} // namespace state_observation

#endif // !STATE_OBSERVATION__PARAMETER_SET__HPP

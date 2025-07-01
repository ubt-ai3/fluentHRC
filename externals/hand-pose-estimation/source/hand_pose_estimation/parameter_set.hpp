#pragma once

#include "framework.h"

#include <string>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/assume_abstract.hpp>

// template class HANDPOSEESTIMATION_API std::basic_string<char, std::char_traits<char>, std::allocator<char>>;


namespace hand_pose_estimation
{

/**
 * @class parameter_set
 * @brief Base class for parameter management with file persistence
 *
 * Abstract base class that provides automatic loading and saving of parameter
 * values to files. When a derived class is instantiated, it loads values from
 * a file, and when destroyed, it writes the values back to the file.
 *
 * To create a new parameter set:
 * 1. Derive from this class
 * 2. Implement static const std::string filename_ with a unique name
 * 3. Implement serialize() method for parameter serialization
 * 4. Create a constructor that reads the file using boost::archive::text_iarchive
 *
 * Features:
 * - Automatic file persistence
 * - Parameter serialization
 * - File path management
 * - Boost serialization integration
 */
class HANDPOSEESTIMATION_API parameter_set {
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

} // namespace hand_pose_estimation

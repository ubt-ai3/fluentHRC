#include "classification_handler.hpp"

#include <cstdlib>
#include <iostream>
#include <fstream>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

namespace hand_pose_estimation 
{

/////////////////////////////////////////////////////////////
//
//
//  Class: classifier_set
//
//
/////////////////////////////////////////////////////////////

classifier_set::classifier_set()
{
	filename_ = std::string("classifier_set.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		register_classifiers(ia);
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else {

		classifiers.push_back(
			classifier::Ptr(new stretched_fingers_classifier(
				gesture_prototype::Ptr(new gesture_prototype(
					"stretched index finger",
					1,  // number of stretched fingers
					std::vector<std::vector<cv::Point>>({
						std::vector<cv::Point>({
							cv::Point(0,0),
							cv::Point(100,0),
							cv::Point(100,50),
							cv::Point(20,50),
							cv::Point(20,100),
							cv::Point(0,100)
						})
					})
				))
			))
		);
	}
}

classifier_set::~classifier_set()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	register_classifiers(oa);
	const classifier_set& classifiers = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(classifiers);
}

classifier_set::iterator classifier_set::begin()
{
	return classifiers.begin();
}

classifier_set::iterator classifier_set::end()
{
	return classifiers.end();
}

classifier_set::const_iterator classifier_set::begin() const
{
	return classifiers.begin();
}

classifier_set::const_iterator classifier_set::end() const
{
	return classifiers.end();
}

} // namespace hand_pose_estimation
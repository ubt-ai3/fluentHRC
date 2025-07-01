#include "classification.hpp"

#include <list>
#include <math.h>

namespace hand_pose_estimation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: classifier
//
//
/////////////////////////////////////////////////////////////

classifier::classifier(const gesture_prototype::ConstPtr& prototype)
	:
	prototype(prototype)
{
}

classifier::~classifier()
{
}

gesture_prototype::ConstPtr classifier::get_object_prototype() const
{
	return prototype;
}

float classifier::bell_curve(float x, float stdev)
{
	return std::expf(-x * x / (2 * stdev * stdev));
}


double classifier::bell_curve(double x, double stdev)
{
	return std::expf(-x * x / (2 * stdev * stdev));
}




/////////////////////////////////////////////////////////////
//
//
//  Class: shape_classifier
//
//
/////////////////////////////////////////////////////////////

shape_classifier::shape_classifier(const gesture_prototype::ConstPtr& prototype)
	:
	classifier(prototype)
{
}

shape_classifier::~shape_classifier()
{
}

classification_result shape_classifier::classify(const hand_instance& hand) const
{
	float similarity = 0.f;

	return classification_result(prototype, similarity);
}


float shape_classifier::match(const std::vector<double>& ma, const std::vector<double>& mb)
{
	// execute return 1.f - cv::matchShapes(hand.observation_history.back()->hull, seg.hull, cv::CONTOURS_MATCH_I1, 0.) / 7.;
	// with pre-computed hu moments

	int i, sma, smb;
	double eps = 1.e-5;
	double result = 0;
	bool anyA = false, anyB = false;


	for (i = 0; i < 7; i++)
	{
		double ama = fabs(ma[i]);
		double amb = fabs(mb[i]);

		if (ama > 0)
			anyA = true;
		if (amb > 0)
			anyB = true;

		if (ma[i] > 0)
			sma = 1;
		else if (ma[i] < 0)
			sma = -1;
		else
			sma = 0;
		if (mb[i] > 0)
			smb = 1;
		else if (mb[i] < 0)
			smb = -1;
		else
			smb = 0;

		if (ama > eps && amb > eps)
		{
			ama = sma * log10(ama);
			amb = smb * log10(amb);
			result += fabs(-ama + amb);
		}
	}


	//	cv::matchShapes(hand.observation_history.back()->hull, seg.hull, cv::CONTOURS_MATCH_I1, 0.);

	if (anyA != anyB)
		result = DBL_MAX;

	return (float) bell_curve(result, 1.);
}







/////////////////////////////////////////////////////////////
//
//
//  Class: stretched_fingers_classifier
//
//
/////////////////////////////////////////////////////////////

stretched_fingers_classifier::stretched_fingers_classifier(const gesture_prototype::ConstPtr& prototype)
	:
	classifier(prototype), 
	shape_classifier(prototype)
{
}

stretched_fingers_classifier::~stretched_fingers_classifier()
{
}



classification_result stretched_fingers_classifier::classify(const hand_instance& hand) const
{
	return shape_classifier::classify(hand);
}




} /* hand_pose_estimation */


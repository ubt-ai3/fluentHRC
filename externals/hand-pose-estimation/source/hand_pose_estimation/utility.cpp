#include "utility.hpp"
#include <opencv2/core/core.hpp>

namespace numeric
{
	double signed_angle(const cv::Point& p1, const cv::Point& p2)
	{
		double angle = numeric::angle(p1, p2);
		return  std::signbit(cv::norm(p1.cross(p2))) ? angle : -angle;
	}
	double angle(const cv::Point2f& p1, const cv::Point2f& p2)
	{
		double norm = std::sqrt(p1.dot(p1)* p2.dot(p2));
		return std::acos(p1.dot(p2)/norm);
	}
}

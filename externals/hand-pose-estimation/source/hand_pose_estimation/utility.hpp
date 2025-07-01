#pragma once
#include <vector>
#include <opencv2/core/core.hpp>
//utility header for often used functions
template
<class T>
class ArgMax
{
public:
	T arg_max;
	double  max;

	ArgMax(const T& arg,double value = std::numeric_limits<double>::lowest()):
		arg_max(arg),
		max(value)
	{}
	void operator ()(const T& arg,double value)
	{
		if (value > max)
		{
			arg_max = arg;
			max = value;
		}
	}
};

template
<class T>
class ArgMin
{
public:
	T arg_min;
	double  min;

	ArgMin(const T& arg, double value = std::numeric_limits<double>::max()) :
		arg_min(arg),
		min(value)
	{}
	void operator ()(const T& arg, double value)
	{
		if (value < min)
		{
			arg_min = arg;
			min = value;
		}
	}
};

template
<class T>	void reorder(std::vector<T>& in, const std::vector<std::size_t>& order)
{
	_ASSERT(in.size() == order.size());
	std::vector<T> tmp(in.size());
	std::size_t i = -1;
	for (std::size_t pos : order)
	{
		i++;
		tmp[i] = std::move(in[pos]);
	}
	in = std::move(tmp);
};

//converts a point @param p into a homogenized point (x,y,1).transpose
template
<class numeric>
cv::Mat homogenize(cv::Point p)
{
	cv::Mat mat(p);
	mat.resize(3);
	mat.convertTo(mat, cv::DataType<numeric>::type);
	mat.at<numeric>(2, 0) = 1;
	return mat;
}

class NotImplemented : public std::logic_error
{
public:
	NotImplemented() : std::logic_error("Function not yet implemented") { };
};

//namespace cv
//{
//	class Point;
//	class Point2f;
//
//}

namespace numeric
{
	constexpr double PI = 3.1415926535897932384626;

	//linearly interpolates between a and b with paramter t
	inline double lerp(double a, double b, double t)
	{
		return a + (b-a) *t;
	}

	//linearly interpolates between a and b with paramter t
	inline cv::Point2f lerp(cv::Point2f a,cv::Point2f b, double t)
	{
		return a + (b - a) * t;
	}

	inline double to_deg(double rad)
	{
		return rad * 180 / PI;
	}
	inline double to_rad(double deg)
	{
		return deg * PI/180;
	}

	//computes the signed angle between the vectors p1 and p2
	double signed_angle(const cv::Point& p1, const cv::Point& p2);
	//computes the unsigned angle between p1 and p2
	double angle(const cv::Point2f& p1, const cv::Point2f& p2);
}

inline double operator"" _deg(long double value)
{
	return value * numeric::PI / 180;
}

inline double operator"" _deg(unsigned long long value)
{
	return ((double)value) * numeric::PI / 180;
}
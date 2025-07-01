#pragma once

#include "framework.h"

#include <pcl/point_types.h>
#include <Eigen/Core>

class rgb;
class cielab;
class hsv;


struct HANDPOSEESTIMATION_API pcl::RGB;

/**
 *************************************************************************
 *
 * @class rgb
 *
 * Constructor based conversion between color spaces.
 *
 ************************************************************************/

class HANDPOSEESTIMATION_API rgb {
public:
	double r;       // a fraction between 0 and 1
	double g;       // a fraction between 0 and 1
	double b;       // a fraction between 0 and 1

	rgb(double r, double g, double b);
	rgb(uint8_t r, uint8_t g, uint8_t b);
	rgb() = default;

	rgb(const rgb&) = default;

	rgb(const pcl::RGB&);

	rgb(const cielab&);

	rgb(const hsv&);

	rgb& operator=(const rgb&) = default;
};


/**
 *************************************************************************
 *
 * @class cielab
 *
 * Constructor based conversion between color spaces.
 *
 ************************************************************************/

class HANDPOSEESTIMATION_API cielab {
public:
	static const double XN;
	static const double YN;
	static const double ZN;
	static const Eigen::Matrix3d XYZ2SRGB;
	static const Eigen::Matrix3d SRGB2XYZ;
	static const double MAX_DELTA;

	cielab(double L, double a, double b);
	cielab() = default;

	cielab(const cielab&) = default;

	cielab(const rgb&);

	cielab(const pcl::RGB&);

	cielab& operator=(const cielab&) = default;

	/**
	* CIE76 based distance measure with lower weight for luminance
	* See https://en.wikipedia.org/wiki/Color_difference
	*/
	double delta(const cielab& reference) const;

	double L;       // ∈ [0, 100]
	double a;       // ∈ [-200, 200]
	double b;       // ∈ [-200, 200]
};


/**
 *************************************************************************
 *
 * @class hsv
 *
 * Constructor based conversion between color spaces.
 *
 ************************************************************************/

class HANDPOSEESTIMATION_API hsv {
public:
	double h;       // a fraction between 0 and 360
	double s;       // a fraction between 0 and 1
	double v;       // a fraction between 0 and 1

	hsv(double h, double s, double v);
	hsv(uint8_t h, uint8_t s, uint8_t v);
	hsv() = default;

	hsv(const hsv&) = default;

	hsv(const rgb&);

	hsv& operator=(const hsv&) = default;
};


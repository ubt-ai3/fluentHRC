#include "color_space_conversion.hpp"

/////////////////////////////////////////////////////////////
//
//
//  Class: cielab
//
//
/////////////////////////////////////////////////////////////

const double cielab::XN = 95.0489;
const double cielab::YN = 100;
const double cielab::ZN = 108.8840;
const Eigen::Matrix3d cielab::XYZ2SRGB = (Eigen::Matrix3d() <<
	3.2406, -1.5372, -0.4986,
	-0.9689, 1.8758, 0.0415,
	0.0557, -0.2040, 1.0570
	).finished();
const Eigen::Matrix3d cielab::SRGB2XYZ = (Eigen::Matrix3d() <<
	0.4124, 0.3576, 0.1805,
	0.2126, 0.7152, 0.0722,
	0.0193, 0.1192, 0.9505
	).finished();


const double cielab::MAX_DELTA = cielab(100, -10, 100).delta(cielab(30, 80, -125));

cielab::cielab(const rgb& in)
	:
	L(0), a(0), b(0)
{
	// conversion from sRGB to CIEXYZ
	// https://en.wikipedia.org/wiki/SRGB

	std::function<double(double)> inv_gamma = [](double u)
	{
		return u <= 0.04045 ? u / 12.92 : std::pow((u + 0.055) / 1.055, 2.4);
	};

	Eigen::Vector3d lin(
		inv_gamma(in.r),
		inv_gamma(in.g),
		inv_gamma(in.b)
	);

	Eigen::Vector3d XYZ = 100 * (cielab::SRGB2XYZ * lin);

	// CIEXYZ to CIELAB using illumination D65
	// https://en.wikipedia.org/wiki/CIELAB_color_space#RGB_and_CMYK_conversions

	std::function<double(double)> f = [](double t)
	{
		double delta = 6. / 29.;
		return t > delta * delta * delta ? std::pow(t, 1. / 3.) : t / (3 * delta * delta) + 4. / 29.;
	};

	L = 116. * f(XYZ(1) / cielab::YN) - 16.;
	a = 500. * (f(XYZ(0) / cielab::XN) - f(XYZ(1) / cielab::YN));
	b = 200. * (f(XYZ(1) / cielab::YN) - f(XYZ(2) / cielab::ZN));
}

cielab::cielab(const pcl::RGB& in)
	:
	cielab(rgb(in))
{
}

double cielab::delta(const cielab& reference) const
{
	return std::sqrt(std::pow(reference.L - L, 2) * 0.25 + std::pow(reference.a - a, 2) + std::pow(reference.b - b, 2));
}

cielab::cielab(double L, double a, double b)
	:
	L(L), a(b), b(b)
{
}


/////////////////////////////////////////////////////////////
//
//
//  Class: rgb
//
//
/////////////////////////////////////////////////////////////

rgb::rgb(const cielab& in)
	:
	r(0), g(0), b(0)
{

	// CIELAB to CIEXYZ using illumination D65
	// https://en.wikipedia.org/wiki/CIELAB_color_space#RGB_and_CMYK_conversions

	std::function<double(double)> inv_f = [](double t)
	{
		double delta = 6. / 29.;
		return t > delta ? std::pow(t, 3) : 3 * delta * delta * (t - 4. / 29.);
	};



	Eigen::Vector3d XYZ(
		cielab::XN * inv_f((in.L + 16) / 116 + in.a / 500),
		cielab::YN * inv_f((in.L + 16) / 116),
		cielab::ZN * inv_f((in.L + 16) / 116 - in.b / 200)
	);

	// conversion from CIEXYZ to sRGB
	// https://en.wikipedia.org/wiki/SRGB
	Eigen::Vector3d lin = cielab::XYZ2SRGB * (0.01 * XYZ);

	std::function<double(double)> gamma = [](double u) {
		return u <= 0.0031308 ? 12.92 * u : 1.055 * std::pow(u, 1. / 2.4) - 0.055;
	};

	r = gamma(lin(0));
	g = gamma(lin(1));
	b = gamma(lin(2));
}

/*
* conversion from https://www.rapidtables.com/convert/color/hsv-to-rgb.html
*/
rgb::rgb(const hsv& color)
	:
	r(0), g(0), b(0)
{
	double c = color.v * color.s;
	double x = c * (1. - std::abs(std::fmod(color.h / 60, 2.) - 1.));
	double m = color.v - c;

	if (color.h < 60)
	{
		r = c; g = x; b = 0;
	}
	else if (color.h < 120)
	{
		r = x; g = c; b = 0;
	}
	else if (color.h < 180)
	{
		r = 0; g = c; b = x;
	}
	else if (color.h < 240)
	{
		r = 0; g = x; b = c;
	}
	else if (color.h < 300)
	{
		r = x; g = 0; b = c;
	}
	else
	{
		r = c; g = 0; b = x;
	}

	r += m;
	g += m;
	b += m;

}

rgb::rgb(double r, double g, double b)
	:
	r(r), g(g), b(b)
{
}

rgb::rgb(uint8_t r, uint8_t g, uint8_t b)
	:
	r(r / 255.), g(g / 255.), b(b / 255.)
{
}

rgb::rgb(const pcl::RGB& other)
	:
	r(other.r / 255.), g(other.g / 255.), b(other.b / 255.)
{
}


/////////////////////////////////////////////////////////////
//
//
//  Class: hsv
//
//
/////////////////////////////////////////////////////////////

hsv::hsv(double h, double s, double v)
	:
	h(h), s(s), v(v)
{
}

hsv::hsv(uint8_t h, uint8_t s, uint8_t v)
	:
	h(h / 255.), s(s / 255.), v(v / 255.)
{
}


/*
* conversion from https://www.rapidtables.com/convert/color/rgb-to-hsv.html
*/
hsv::hsv(const rgb& color)
	:
	h(0), s(0), v(0)
{
	v = std::max(std::max(color.r, color.g), color.b);
	double delta = v - std::min(std::min(color.r, color.g), color.b);

	if (delta == 0) // if r = g = b
		h = 0;
	else if (v == color.r)
		h = 60 * std::fmod((color.g - color.b) / delta + 6., 6.);
	else if (v == color.g)
		h = 60 * ((color.b - color.r) / delta + 2);
	else // color.b = v
		h = 60 * ((color.r - color.g) / delta + 4);

	if (v != 0)
		s = delta / v;
}
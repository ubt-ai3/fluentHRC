#include "skin_detection.hpp"

#include <fstream>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <list>
#include <opencv2/imgproc.hpp>



namespace hand_pose_estimation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: skin_detection_parameters
//
//
/////////////////////////////////////////////////////////////

skin_detection_parameters::skin_detection_parameters()
{
	filename_ = std::string("skin_detection_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		auto & this_ = *this;
		ia >> BOOST_SERIALIZATION_NVP(this_);
	}
	else
	{
		hsv_lower_bound = std::vector<uint8_t>({ 0, 23, 80 });
		hsv_upper_bound = std::vector<uint8_t>({ 20, 255, 255 });
		threshold = 10;
		min_hand_dimension_pixel = 20;
		min_hand_area_fraction = 0.003f;
		smoothing_factor_pixel = 2.f;
	}
}

skin_detection_parameters::~skin_detection_parameters()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const skin_detection_parameters& skin_detection_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(skin_detection_params);
}



/////////////////////////////////////////////////////////////
//
//
//  Class: skin_detection
//
//
/////////////////////////////////////////////////////////////

std::vector<img_segment::Ptr> skin_detection::detect_segments(const visual_input& input,
	const std::vector< std::vector<cv::Point>>& prev_hands)
{
	cv::Mat input_hsv, skin_mask, segmentation;
	std::vector<std::vector<cv::Point>> contours;


	cv::cvtColor(input.img, input_hsv, cv::COLOR_BGR2HSV);
	cv::inRange(input_hsv, params.hsv_lower_bound, params.hsv_upper_bound, skin_mask);

	cv::Mat roi(input.img.rows, input.img.cols, CV_8UC1);
	roi = cv::Scalar(255);

	if (prev_hands.size() && prev_img.size[0] && prev_timestamp <= input.timestamp_seconds)
	{
		roi = cv::Scalar(0);

		cv::Mat diff, diff_gray(input.img.rows, input.img.cols, CV_32F);
		cv::absdiff(input.img, prev_img, diff);
		cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
		cv::threshold(diff_gray, roi, params.threshold, 255, cv::THRESH_BINARY);

		cv::fillPoly(roi, prev_hands, cv::Scalar(255));


	}

//	cv::imshow("ROI", roi);
		
	cv::Mat input_gray = cv::Mat::zeros(input.img.rows, input.img.cols, CV_8UC1);


//	cv::cvtColor(skin_mask, hand_mask_rgb, cv::COLOR_GRAY2RGBA);
	cv::bitwise_and(roi,skin_mask,roi);
	cv::cvtColor(input.img, input_gray, cv::COLOR_BGRA2GRAY);
	cv::bitwise_and(input_gray, roi, segmentation);
	cv::dilate(segmentation, segmentation, cv::Mat());
	cv::erode(segmentation, segmentation, cv::Mat());

//	cv::threshold(segmentation, segmentation, threshold, 255, cv::THRESH_BINARY);

	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(segmentation, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	cv::Mat drawing = segmentation.clone();
	cv::drawContours(drawing, contours, -1, cv::Scalar(255, 255, 255), 2, 8, hierarchy);
//	cv::imshow("Input ROI", drawing);

	std::vector<img_segment::Ptr> segments(filter_countours(contours, hierarchy, cv::Size(input.img.rows, input.img.cols)));
	for (const img_segment::Ptr& seg : segments)
		evaluate_hand_segment(input.img, seg, input.timestamp_seconds);

	prev_img = input.img;
	prev_timestamp = input.timestamp_seconds;

	return segments;
}

std::chrono::duration<float> skin_detection::get_prev_timestamp() const
{
	return prev_timestamp;
}

std::vector<img_segment::Ptr> skin_detection::filter_countours(const std::vector<std::vector<cv::Point2i>>& contours, 
														       const std::vector<cv::Vec4i>& hierarchy,
															   const cv::Size& image_size) const
{
	std::list<img_segment::Ptr> segments;

	for (int i = 0; i < contours.size(); ++i)
	{
		if (hierarchy[i][3] < 0) // check that there is no parent contour
		{
			img_segment::Ptr seg(new img_segment);
			cv::approxPolyDP(contours[i], seg->contour, params.smoothing_factor_pixel, true);
			seg->bounding_box = cv::Rect(cv::boundingRect(seg->contour));
			seg->model_box_2d = seg->bounding_box;
			
			if (seg->bounding_box.width > params.min_hand_dimension_pixel && seg->bounding_box.height > params.min_hand_dimension_pixel && seg->bounding_box.area() > params.min_hand_area_fraction * image_size.area()) // discard small boxes
				segments.push_back(seg);
		}
	}

	/*
	for (const auto& contour : contours)
	{
		cv::Rect box(cv::boundingRect(contour));
		if (box.width > min_hand_dimension_pixel && box.height > min_hand_dimension_pixel) // discard small boxes
		{
			// blure boxes by 50%
			box.x = std::max(0, box.x - box.width / 4);
			box.y = std::max(0, box.y - box.height / 4);
			box.width = std::min(width, box.width + box.width / 2);
			box.height = std::min(height, box.height + box.height / 2);


			// make a square box
			int dimension = std::min(input.size[0], std::min(input.size[1], std::max(box.width, box.height)));
			if (box.width + box.x + (dimension - box.width) / 2 >= width)
			{
				box.x = width - dimension;
			}
			else
			{
				box.x = std::max(0, box.x - (dimension - box.width) / 2);
			}

			if (box.height + box.y + (dimension - box.height) / 2 >= height)
			{
				box.y = height - dimension;
			}
			else
			{
				box.y = std::max(0, box.y - (dimension - box.height) / 2);
			}

			box.width = dimension;
			box.height = dimension;


			boxes.push_back(box);
		}
	}
*/

	segments.sort([](const img_segment::Ptr& lhs, const img_segment::Ptr& rhs) {
		return lhs->bounding_box.area() > rhs->bounding_box.area();
		});



	for (auto super_iter = segments.begin(); super_iter != segments.end(); ++super_iter)
	{
		const cv::Rect& super_box = (*super_iter)->bounding_box;
		auto contains_box = [&super_box](cv::Rect sub_box) {
			for (cv::Point2i offset : {cv::Point2i(0, 0),
				cv::Point2i(sub_box.width - 1, 0),
				cv::Point2i(sub_box.width - 1, sub_box.height - 1),
				cv::Point2i(0, sub_box.height - 1)})
			{
				if (!super_box.contains(cv::Point2i(sub_box.x, sub_box.y) + offset))
					return false;
			}
			return true;
		};

		auto current = super_iter;
		for (auto sub_iter = ++current; sub_iter != segments.end(); ) // explicit increment in body
		{
			if (contains_box((*sub_iter)->bounding_box))
			{
				sub_iter = segments.erase(sub_iter);
			}
			else
			{
				++sub_iter;
			}

		}
	}

	return std::vector<img_segment::Ptr>(segments.begin(), segments.end());
}

void skin_detection::evaluate_hand_segment(const cv::Mat& img, 
	const img_segment::Ptr& seg,
	std::chrono::duration<float> timestamp) const
{
	seg->timestamp = timestamp;
	//img(seg->bounding_box).copyTo(seg->img);
	//cv::convexHull(seg->contour, seg->hull,false, false);

	//cv::convexityDefects(seg->contour, seg->hull, seg->convexity_defects);
	cv::Moments moments(cv::moments(seg->contour));
	//cv::HuMoments(moments, seg->hu_moments);

	seg->palm_center_2d = cv::Point2i(moments.m10 / moments.m00, moments.m01 / moments.m00);
	
	float sum_defects = 0.f;
	int dimension = std::min(seg->bounding_box.width, seg->bounding_box.height);
	//for (const cv::Vec4i& defect : seg->convexity_defects)
	//{
	//	sum_defects += std::abs((defect(3) / 256.f - 0.4f * dimension)/dimension);
	//}
	//seg->hand_certainty = std::max(0.f, 1 - 0.25f * sum_defects);
	//seg->finger_tips_2d = get_fingers(*seg);
}

std::vector<cv::Point2i> skin_detection::get_fingers(const img_segment& seg) const
{


	cv::Point center = seg.palm_center_2d;


	double max_sqr_dist = 0.f;
	for (const cv::Point& p : seg.contour)
	{
		max_sqr_dist = std::max(max_sqr_dist, (p - center).ddot(p - center));
	}

	std::vector<cv::Point2i> inner_circle_tips(get_outlier_tips(seg.contour, center, 0.7 * std::sqrt(max_sqr_dist)));
	std::vector<cv::Point2i> outer_circle_tips(get_outlier_tips(seg.contour, center, 0.9 * std::sqrt(max_sqr_dist)));
	
	std::vector<cv::Point2i> finger_tips;
	auto comp = [](const cv::Point2i& lhs, const cv::Point2i& rhs)
	{
		if (lhs.x != rhs.x)
			return lhs.x < rhs.x;
		return lhs.y < rhs.y;
	};
	std::sort(inner_circle_tips.begin(), inner_circle_tips.end(), comp);
	std::sort(outer_circle_tips.begin(), outer_circle_tips.end(), comp);

	std::set_intersection(inner_circle_tips.cbegin(), inner_circle_tips.cend(), 
		outer_circle_tips.cbegin(), outer_circle_tips.cend(), 
		std::back_inserter(finger_tips), comp);

	return finger_tips;
}

std::vector<cv::Point2i> skin_detection::get_outlier_tips(const std::vector<cv::Point2i>& contour, const cv::Point2i circle_center, double circle_radius) const
{
	std::vector<cv::Point2i> outlier_tips;

	auto dist_sqr = [&](const cv::Point& other)
	{
		return (other - circle_center).ddot(other - circle_center);
	};

	// determine intersections of contour and a circle
// which radius = 2/3 of contour
	double thres = circle_radius * circle_radius;
	double prev_angle = std::numeric_limits<double>::quiet_NaN(); // angle of previous intersection in radians
	cv::Point2i finger_tip_candidat; // point furthes from center in current out-of-circle segment
	double local_max_sqr_dist = 0; // squred distance of finger_tip_candidate to center
	double d_start;

	auto get_intersection_angle = [&](const cv::Point2i& p1, const cv::Point2i& p2)
	{

		double l = std::sqrt((p2 - p1).ddot(p2 - p1));
		double x1 = (p1.x - circle_center.x) / l;
		double dx = (p2.x - p1.x) / l;
		double y1 = (p1.y - circle_center.y) / l;
		double dy = (p2.y - p1.y) / l;

		double root = std::sqrt(std::pow(x1 * dx + y1 * dy, 2) - (x1 * x1 + y1 * y1 - thres / l / l));
		double t1 = -(x1 * dx + y1 * dy) + root;
		double t2 = -(x1 * dx + y1 * dy) - root;

		if (0 <= t1 && t1 <= 1)
			return std::atan2(y1 + t1 * dy, x1 + t1 * dx);

		else if (0 <= t2 && t2 <= 1)
			return std::atan2(y1 + t2 * dy, x1 + t2 * dx);

		return std::numeric_limits<double>::quiet_NaN();
	};

	d_start = dist_sqr(contour.front());


	if (d_start > thres) // first point outside threshold circle
	{// search for begin of out-of-circle segment
		finger_tip_candidat = contour.back();
		for (auto start = contour.cend(), end = --start - 1;
			end != contour.cend();
			--start, --end)
		{
			double d_start = dist_sqr(*start), d_end = dist_sqr(*end);
			if (d_start > local_max_sqr_dist)
			{
				local_max_sqr_dist = d_start;
				finger_tip_candidat = *start;
			}

			if (d_end < thres)
			{
				prev_angle = get_intersection_angle(*start, *end);
				break;
			}

			if (end == contour.cbegin())
				return outlier_tips;
		}
	}
	else {

		finger_tip_candidat = contour.front();
	}

	// test all out-of-circle segments whether they are fingers
	for (auto start = contour.cbegin(), end = start + 1;
		end != contour.cend();
		++start, ++end)
	{
		double d_start = dist_sqr(*start), d_end = dist_sqr(*end);
		if (d_start > thres && d_start > local_max_sqr_dist)
		{
			local_max_sqr_dist = d_start;
			finger_tip_candidat = *start;
		}
		if ((d_start > thres) != (d_end > thres))
		{ // intersection of outline and circle
			double angle = get_intersection_angle(*start, *end);
			if (!isnan(prev_angle) && d_end < thres)
			{
				double angle_between;
				if (prev_angle > angle)
					angle_between = prev_angle - angle;
				else
					angle_between = CV_2PI + prev_angle - angle;

				// between 9° and 35°
				if (angle_between < 0.61 && angle_between > 0.157)
				{
					outlier_tips.push_back(finger_tip_candidat);

					if (outlier_tips.size() > 5)
						return std::vector<cv::Point2i>();
				}

				local_max_sqr_dist = 0;
			}
			prev_angle = angle;
		}

	}

	return outlier_tips;
}

} /* hand_pose_estimation */

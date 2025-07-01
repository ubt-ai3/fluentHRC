#pragma once

#include "framework.h"

#include <opencv2/core.hpp>

#include "hand_model.hpp"
#include "parameter_set.hpp"

class HANDPOSEESTIMATION_API cv::Mat;
template class HANDPOSEESTIMATION_API cv::Rect_<int>;

namespace hand_pose_estimation
{

/**
 * @class skin_detection_parameters
 * @brief Parameters for skin color detection
 *
 * Configuration parameters for detecting skin-colored regions in images,
 * including HSV color bounds, thresholds, and geometric constraints.
 *
 * Features:
 * - HSV color range specification
 * - Detection thresholds
 * - Minimum hand dimensions
 * - Area fraction constraints
 * - Smoothing parameters
 * - Parameter serialization
 */
class skin_detection_parameters : public parameter_set {
public:

	skin_detection_parameters();

	~skin_detection_parameters();

	std::vector<uint8_t> hsv_lower_bound;
	std::vector<uint8_t> hsv_upper_bound;
	int threshold;
	int min_hand_dimension_pixel;
	float min_hand_area_fraction;
	float smoothing_factor_pixel;


	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(hsv_lower_bound);
		ar& BOOST_SERIALIZATION_NVP(hsv_upper_bound);
		ar& BOOST_SERIALIZATION_NVP(threshold);
		ar& BOOST_SERIALIZATION_NVP(min_hand_dimension_pixel);
		ar& BOOST_SERIALIZATION_NVP(min_hand_area_fraction);
		ar& BOOST_SERIALIZATION_NVP(smoothing_factor_pixel);
	}

};


/**
 * @class skin_detection
 * @brief Skin color region detection
 *
 * Implements skin color detection and region extraction from images,
 * with support for hand segment detection and finger tip identification.
 *
 * Features:
 * - Skin color detection
 * - Hand segment extraction
 * - Contour filtering
 * - Finger tip detection
 * - Outlier handling
 * - Timestamp tracking
 */
class skin_detection
{
public:

	HANDPOSEESTIMATION_API std::vector<img_segment::Ptr> detect_segments(const visual_input& input,
		const std::vector< std::vector<cv::Point>>& prev_hands = std::vector< std::vector<cv::Point>>());

	std::chrono::duration<float> get_prev_timestamp() const;

	const skin_detection_parameters params;

private:


	cv::Mat prev_img;
	std::chrono::duration<float> prev_timestamp;

	std::vector<img_segment::Ptr> filter_countours(const std::vector<std::vector<cv::Point2i>>& contours,
										           const std::vector<cv::Vec4i>& hierarchy,
												   const cv::Size& image_size) const;

	void evaluate_hand_segment(const cv::Mat& img,
							   const img_segment::Ptr& seg,
							   std::chrono::duration<float> timestamp) const;

	std::vector<cv::Point2i> get_fingers(const img_segment& seg) const;
	std::vector<cv::Point2i> get_outlier_tips(const std::vector<cv::Point2i>& contour,
											  const cv::Point2i circle_center,
											  double circle_radius) const;
};

} /* hand_pose_estimation */
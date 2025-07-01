#pragma once

#include "framework.h"

#include<iostream>
#include <pcl/point_types.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/flann/miniflann.hpp"
#include <vector>

#include "skin_detection.hpp"
#include "ra_point_cloud_classifier.hpp"

namespace hand_pose_estimation
{

/**
 * @class skin_detection_bayes
 * @brief Bayesian-based skin color detection implementation
 *
 * Implements skin color detection using Bayesian classification and point cloud
 * analysis. Provides functionality for detecting skin regions, calculating
 * probability maps, and managing skin detection parameters.
 *
 * Features:
 * - Bayesian skin color classification
 * - Probability map generation
 * - Contour and convex hull calculation
 * - Region and bounding box extraction
 * - Skin region visualization
 * - Calibration with hand images
 */
class HANDPOSEESTIMATION_API skin_detection_bayes
{

private:
	std::string File_;
	bool neighbor = true;
	bool show = false;
	cv::Mat skin_map;



	std::vector<std::vector<cv::Point>> convex_hull, contours;
	std::vector<cv::Point> centers;
	std::vector<cv::Rect2i> boxes;
	std::vector<cv::Mat> regions;

	point_cloud_classifier classifier;

public:
	double true_skin_threshold = 0.70;
	double skin_threshold = 0.15;
	const skin_detection_parameters params;

	typedef uchar Pixel;
	skin_detection_bayes() = default;
	~skin_detection_bayes() = default;

	void skin_candidate(const cv::Mat3b & input_image, const std::vector<hand_instance::Ptr>& prev_hands = std::vector<hand_instance::Ptr>());

	bool  is_near_skin_pixels(int img_point1, int img_point2,
		const cv::Mat& probability_map);

	const std::vector<std::vector<cv::Point>>& get_contours() { return contours; }

	const std::vector<std::vector<cv::Point>>& get_convex_hull() { return convex_hull; }

	const std::vector<cv::Point>& get_centers() { return centers; }
	
	const std::vector<cv::Mat>& get_regions() { return regions; }

	const std::vector<cv::Rect2i>& get_boxes() { return boxes; }

	void show_skin_regions();

	void calibrate_with_hand_image(const cv::Mat& hand_image);
};

/**
 * @class skin_detector
 * @brief High-level skin region detection interface
 *
 * Provides a simplified interface for detecting skin-like regions in images.
 * Wraps the Bayesian skin detection implementation and manages the detection
 * process for hand tracking applications.
 *
 * Features:
 * - Skin region detection
 * - Image segment extraction
 * - Parameter management
 * - Region visualization
 * - Previous hand tracking integration
 */
class HANDPOSEESTIMATION_API skin_detector
{
public:

	skin_detector() = default;

	~skin_detector() = default;

	std::vector<img_segment::Ptr> detect_segments(const visual_input& input,
		const std::vector<hand_instance::Ptr>& prev_hands = std::vector<hand_instance::Ptr>());

	const skin_detection_parameters& get_params() const { return impl.params; }

	void show_skin_regions();

private:
	skin_detection_bayes impl;
};

}
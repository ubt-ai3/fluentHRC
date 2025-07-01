#pragma once
#include <vector>
#include <optional>

#include <opencv2/opencv.hpp>
#include "framework.h"
#include "hand_model.hpp"

#include "utility.hpp"
#include "ra_skin_color_detector.hpp"

namespace hand_pose_estimation
{	


	class HANDPOSEESTIMATION_API finger_classification
	{
	public:
		//Structure of arrays
		std::vector<cv::RotatedRect> bounding_boxes; //bounding boxes of recognized fingers
		std::vector<finger_type> finger_types; //recognized fingertype
		std::vector<cv::Point2f> fingertip_pos; //position of the finger tip
		std::vector<cv::Point2f> fingerroot_pos; //position of the start of the finger (opposite of finger tip)
		std::vector<cv::Point2f> fingerpointing_direction; //pointing direction

		std::optional<cv::RotatedRect> get_finger(finger_type finger) const;

		//updates bounding_boxes and finger_types, leveas the other ones untouched
		void insert_finger(finger_type finger, const cv::RotatedRect& bb);

	};

	class HANDPOSEESTIMATION_API pointing_gesture_result
	{
	public:

		cv::Point3f pointing_direction3D;
		cv::Point3f finger_tip3D;
		//DEBUG
		cv::Point3f finger_root3D;

		cv::Point2f pointing_direction2D;
		cv::Point2f finger_tip2D;
	};

	/*
	* @class finger_detector
		determines the bounding boxes of fingers in a binary image of a hand
	*/
	class HANDPOSEESTIMATION_API finger_detector
	{
	public:

		/*
		*  applies a thinning algorithm for the given binary image
		*  TODO finish
		*/
		void thin(const cv::Mat& image);

		/*
		* detect and segment fingers in a black-white image of EXACTLY ONE hand
		*/
		finger_classification detect(const cv::Mat& image,std::vector<cv::Mat>& debug) const;


		/*
		* @input binary image with black background
		* @ouput every "black hole" not connected to the background gets filled with white
		*/
		void fill_holes(cv::InputOutputArray image);

		static void draw_rotated_rect(cv::Mat& mat, const cv::RotatedRect& rect, cv::Scalar color = cv::Scalar(255, 0, 0));


		static ArgMin<cv::Point> find_nearest(const cv::Mat& mat, cv::Point point, bool filter(float pix_val));

		/*
			generates points on a cirlce with radius @radius and center point @center.
			Every point has a angle difference of @delta_theta to the previous point
		*/
		static std::vector<cv::Point2f> sample_circle(cv::Point2f center, double radius, double delta_theta);
		/*
		* rotates @param image around @param rot_center for @param angle_deg (in degrees)
		 * transformation matrix is the trafo from old image pixel to rotated new image pixel, this is not just a rotation, since it scales the image to prevent cutting corners*
		*/
		static cv::Mat rotate_image(const cv::Mat& image,cv::Point2i rot_center, double angle_deg, cv::Mat* out_transformation_matrix = nullptr);
		
		/*
		* searches in hand from value @param start_val_y upwards for the palm line
		* bool in tuple indicates if a palm line was found
		*/
		static std::tuple<bool,cv::Point, cv::Point> detect_palm_line(const cv::Mat& hand,int start_val_y, bool thumb_detected,bool thumb_right);

	};


	/*
		@class pointing_gesture_detector 
		detects the occurence and position of a pointing gesture in a colored 3D visual_input
	*/

	class HANDPOSEESTIMATION_API  pointing_gesture_detector
	{
	public:
		/*
		@throws if no 3d point cloud is avaiable
		@returns an object indicating if and where a pointing gesture was found
		the 3D-coordinates don't work yet only 2D is supported
		*/
		std::vector<pointing_gesture_result> detect(const visual_input& vis_input);

		Eigen::Hyperplane<float,3> table_plane;
		skin_detection_bayes skin_detect;
		finger_detector finger_detect;
		std::optional<cv::Rect> roi; //2d region of vis_input where to look for hands

		//matrizes contain debug information and visualization of the recognition process -> have a look at them :)
		cv::Mat visualization;
		std::vector<cv::Mat> evaluation_mat;
	};
}
#include "ra_skin_color_detector.hpp"

#include<iostream>
#include <pcl/point_types.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/flann/miniflann.hpp"
#include <vector>

#include "ra_point_cloud_classifier.hpp"


namespace hand_pose_estimation
{


void skin_detection_bayes::skin_candidate(const cv::Mat3b & input_image, const std::vector<hand_instance::Ptr>& prev_hands) {
	contours.clear();
	convex_hull.clear();
	centers.clear();
	regions.clear();
	boxes.clear();


	//convert BGR/RGB to YUV and drop Y chanel.
	const cv::Mat image_rgb = input_image;
	cv::Mat image_yuv;
	cv::cvtColor(image_rgb, image_yuv, cv::COLOR_BGR2YCrCb);


	cv::Mat probability_map_certain(image_rgb.rows, image_rgb.cols, CV_8UC1);
	cv::Mat probability_map_likely(image_rgb.rows, image_rgb.cols, CV_8UC1);
	cv::Mat exact_probability_image = cv::Mat(image_yuv.rows, image_yuv.cols, CV_64F);

	for (auto i = 0; i < image_yuv.rows; i++)
	{
		cv::Vec3b* in = (cv::Vec3b*)image_yuv.ptr(i);

		uchar* out1 = probability_map_likely.ptr(i);
		uchar* out2 = probability_map_certain.ptr(i);

		for (auto j = 0; j < image_yuv.cols; j++)
		{
			double prob = classifier.bayes_classifier((*in)(1), (*in)(2));
			exact_probability_image.at<double>(i, j) = prob;
			*out1 = 0;
			*out2 = 0;

			if (prob > true_skin_threshold) {
				*out1 = 255;
				*out2 = 255;
			}

			if (prob > skin_threshold)
				*out1 = 255;


			++in;
			++out1;
			++out2;
		}
	}
	
	//cv::imshow("Exact", exact_probability_image);
	//cv::waitKey(5);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::dilate(probability_map_certain, probability_map_certain, kernel);

	for (const auto& hand : prev_hands)
	{
		const auto& seg = *hand->observation_history.back();

		auto br = seg.bounding_box.br();
		auto tl = seg.bounding_box.tl();

		tl.x = std::max(0, std::min(probability_map_certain.cols, tl.x));
		br.x = std::max(0, std::min(probability_map_certain.cols, br.x));
		tl.y = std::max(0, std::min(probability_map_certain.rows, tl.y));
		br.y = std::max(0, std::min(probability_map_certain.rows, br.y));

		auto img_box = cv::Rect(tl, br);
		auto mask_box = cv::Rect(tl.x - seg.bounding_box.x, tl.y - seg.bounding_box.y, img_box.width, img_box.height);
		if (img_box.width && img_box.height)
			seg.mask(mask_box).copyTo(probability_map_certain(img_box), seg.mask(mask_box));
	}


	cv::bitwise_and(probability_map_certain, probability_map_likely, probability_map_certain);



	//cv::imshow("Raw skin segmentation", probability_map_certain);

	//morphological operations to remove noise
	cv::dilate(probability_map_certain, probability_map_certain, cv::Mat(),cv::Point(-1,-1),1);// modify according to image output

	if (show) {
		//cv::Mat regions;
		//cv::cvtColor(probability_map_certain, regions, cv::COLOR_GRAY2BGR);

		//for (const auto& hand : prev_hands)
		//{
		//	regions(hand->get_segment()->mask) = cv::Color
		//}

		cv::imshow("Skin regions", probability_map_certain);
		cv::waitKey(1);
	}


	// find connected components and other properties of the components
	cv::Mat stats, centroids;
	int total_labels;
	cv::Mat labeled_image(image_rgb.rows, image_rgb.cols, CV_32S);
	total_labels = cv::connectedComponentsWithStats(probability_map_certain, labeled_image, stats, centroids, 8, CV_32S);


	for (int label = 1; label < total_labels; label++) {

		if (stats.at<int>(label, cv::CC_STAT_WIDTH) > params.min_hand_dimension_pixel
			&& stats.at<int>(label, cv::CC_STAT_HEIGHT) > params.min_hand_dimension_pixel
			&& stats.at<int>(label, cv::CC_STAT_AREA) > params.min_hand_area_fraction * input_image.rows * input_image.cols
			) {
			//std::cout << "Area of " << stats.at<int>(i, cv::CC_STAT_AREA) << std::endl;


			//bounding box parameters
			int left = stats.at<int>(label, cv::CC_STAT_LEFT);
			int top = stats.at<int>(label, cv::CC_STAT_TOP);
			int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
			int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
			int col_max = left + width;
			int row_max = top + height;
			cv::Rect2i box(left, top, width, height);

			// compute contours
			std::vector<cv::Vec4i> hierarchy;
			std::vector<std::vector<cv::Point>> seg_contours;
			cv::findContours(probability_map_certain(box), seg_contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
			int max_countour_index, max_countour_size = 0;

			for (int i = 0; i < seg_contours.size(); ++i)
			{
				//biggest contour which has no parent 
				if (hierarchy[i][3] < 0 && max_countour_size < seg_contours.at(i).size())
				{
					max_countour_index = i;
					max_countour_size = seg_contours.at(i).size();
				}
			}

			contours.push_back(std::vector<cv::Point>());
			cv::approxPolyDP(seg_contours[max_countour_index], contours.back(), params.smoothing_factor_pixel, true);

			regions.push_back(cv::Mat(box.height, box.width, CV_8UC1, cv::Scalar::all(0)));
			cv::drawContours(regions.back(), contours, contours.size() - 1, cv::Scalar(255), cv::FILLED);

			centers.push_back(cv::Point(centroids.at<double>(label, 0),
				centroids.at<double>(label, 1)));
			//candidate_blobs.push_back(candidate_blob);
			boxes.push_back(box);
		}

		
	}
}


bool skin_detection_bayes::is_near_skin_pixels(int row, int col, const cv::Mat& probability_map) {

	for (int i = std::max(0, row - 2); i < std::min(probability_map.rows, row + 2); i++)
		for (int j = std::max(0, col - 2); j < std::min(probability_map.cols, col + 2); j++)
		{
			//if pixel in the neighbour is true skin and belong to the same c.c
			if (probability_map.at<double>(i, j) >= true_skin_threshold)
				return true;
		}

	return false;


}

void skin_detection_bayes::calibrate_with_hand_image(const cv::Mat& hand_image)
{
	cv::Mat3b input3b;
	if (hand_image.channels() == 4)
	{
		cv::cvtColor(hand_image,input3b, cv::COLOR_BGRA2BGR);
	}
	else
		input3b = hand_image;

	bool show_before = show;
	show = true;
	const char up = 'w';
	const char down = 's';
	std::cout << "Press " << up << " for more sensitivity and " << down << "for less\n";
	while (true)
	{
		this->skin_candidate(input3b);
		char c = cv::waitKey();
		if (c == up)
			true_skin_threshold -= 0.04;
		else if (c == down)
			true_skin_threshold += 0.04;
		else
		{
			std::cout << "Exiting calibrating";
			break;
		}

	}
	show = show_before;
}

void skin_detection_bayes::show_skin_regions()
{
	show = true;
}

std::vector<img_segment::Ptr> skin_detector::detect_segments(const visual_input& input,
	const std::vector<hand_instance::Ptr>& prev_hands)
{
	cv::Mat3b bgr_img;

	if (input.img.channels() == 4)
	{
		cv::cvtColor(input.img, bgr_img, cv::COLOR_BGRA2BGR);
	}
	else
		bgr_img = input.img;

	impl.skin_candidate(bgr_img, prev_hands);

	std::vector < img_segment::Ptr> result;
	int i = 0;
	for (const auto& box : impl.get_boxes())
	{
		auto seg = std::make_shared<img_segment>();
		seg->timestamp = input.timestamp_seconds;
		seg->bounding_box = box;
		seg->model_box_2d = seg->bounding_box;
		seg->contour = impl.get_contours().at(i);
		//seg->hull = impl.get_convex_hull).at(i);
		//seg->img = input(box);
		seg->mask = impl.get_regions().at(i);
		seg->palm_center_2d = impl.get_centers().at(i);

		result.push_back(seg);

		i++;
	}

	return result;
}

void skin_detector::show_skin_regions()
{
	impl.show_skin_regions();
}

}
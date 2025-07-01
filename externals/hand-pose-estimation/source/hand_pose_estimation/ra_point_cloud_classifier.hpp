#pragma once

#include "framework.h"

#include<iostream>
#include<fstream>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <cmath>
#include <numeric>
#include <map>
#include <algorithm>
#include <string>


namespace hand_pose_estimation
{

struct HANDPOSEESTIMATION_API vec2b_lt
{
	bool operator()(const cv::Vec2b& lhs, const cv::Vec2b& rhs)const
	{
		if (lhs[0] < rhs[0])
			return true;
		else if (lhs[0] == rhs[0] && lhs[1] < rhs[1])
			return true;

		return false;
	}
};

class HANDPOSEESTIMATION_API point_cloud_classifier {
public:
	//&typedef std::map<float, float, std::function<bool(const float, const float)>> point_map;
	typedef std::map<cv::Vec2b, std::pair<int, float>, vec2b_lt> point_map;


private:
	const std::string skin_image_file{ "assets/skin_detection_training/skin_file_paths.txt" };
	const std::string non_skin_image_file{ "assets/skin_detection_training/non_skin_file_paths.txt" };

	const std::string skin_image_directory{ "assets/skin_detection_training/skin" };
	const std::string non_skin_image_directory{ "assets/skin_detection_training/non_skin" };

	const std::string prob_file_skin{ "assets/hand_config/probability_skin.txt" };
	const std::string prob_file_nskin{ "assets/hand_config/probability_nskin.txt" };


	cv::Mat skin_colors_image;
	cv::Mat non_skin_colors_image;

	static const std::function<bool(const cv::Vec2b, const cv::Vec2b)> point_lt_operator;

	point_map conditional_probability_map;
	point_map skin_probability_map;
	point_map nskin_probability_map;

	cv::Mat1d probability_lookup;

	bool use_trained_model = true;
	int total_training_pixels{ 0 };
	int skin_training_pixels{ 0 };
	int nskin_training_pixels{ 0 };
	double class_prob_skin{ 0.0 };

	const int color_threshold{ 1 };

public:
	// Default constructor for model training
	point_cloud_classifier();


	~point_cloud_classifier() = default;

	std::vector<std::string> data_file_reader(const std::string& file);

	std::vector<std::string> data_directory_reader(const std::string& directory);

	std::vector<cv::Mat> read_and_convert(const std::vector<std::string>& image_paths);

	double bayes_classifier(const cv::Vec2b& pixel_ptr);
	double bayes_classifier(uchar u, uchar v);

	//Extracts color values from all skin and non skin images(training data) for training.
	// Returns a pair of vector of pair<Point2f,bool>

	std::pair<std::vector<std::pair<cv::Vec2b, bool>>,
		std::vector<std::pair<cv::Vec2b, bool>>> training_data(const std::vector<std::string>& skin_images,
			const std::vector<std::string>& non_skin_images);

	void train();

	unsigned int frequency_image(const std::vector<cv::Mat>& images, cv::Mat& output);

	void load_probabilities();
	void create_lookup_table();

	double posterior(double conditonal_prob, double class_prob_skin, double class_prob_color);


	void write_probabilities();

	double color_distance(const cv::Vec2b color1, const cv::Vec2b color2);

	void show_colors_image(const std::string& window_name, const cv::Mat& img);
};

}
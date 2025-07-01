#include "ra_point_cloud_classifier.hpp"

#include <filesystem>

namespace hand_pose_estimation
{

const std::function<bool(const cv::Vec2b, const cv::Vec2b)> point_cloud_classifier::point_lt_operator = [](const cv::Vec2b lhs, const cv::Vec2b rhs)
{
	if (lhs[0] < rhs[0])
		return true;
	else if (lhs[0] == rhs[0] && lhs[1] < rhs[1])
		return true;

	return false;
};

point_cloud_classifier::point_cloud_classifier()
	:
	probability_lookup(256, 256)
{

	std::ifstream input_prob_file(prob_file_skin);

	if (!input_prob_file) {

		std::cout << "Starting new training process...." << std::endl;
		//Read skin color images to a separate vector 
		//std::vector<cv::Mat> skin_images = read_and_convert(data_file_reader(skin_image_file));
		std::vector<cv::Mat> skin_images = read_and_convert(data_directory_reader(skin_image_directory));

		//Read non_skin color images to a separate vector 
		//std::vector<cv::Mat> non_skin_images = read_and_convert(data_file_reader(non_skin_image_file));
		std::vector<cv::Mat> non_skin_images = read_and_convert(data_directory_reader(non_skin_image_directory));
		// extract training data(pixels)
		unsigned int count_skin_pixels = frequency_image(skin_images, skin_colors_image);
		unsigned int count_non_skin_pixels = frequency_image(non_skin_images, non_skin_colors_image);

		skin_training_pixels = count_skin_pixels;
		nskin_training_pixels = count_non_skin_pixels;

		total_training_pixels = count_skin_pixels + count_non_skin_pixels;
		class_prob_skin = count_skin_pixels / (float) total_training_pixels;

		// compute liklihoods of individual pixel values
		train();
		write_probabilities();


		show_colors_image("skin color space", skin_colors_image);
		show_colors_image("non skin color space", non_skin_colors_image);
	}

	else {

		// std::cout << "Trained classifier available, call bayes_classifer() to use.." << std::endl;
		input_prob_file.close();
		load_probabilities();
	}

}



std::vector<std::string> point_cloud_classifier::data_file_reader(const std::string& file_) {

	std::vector<std::string> file_paths;
	const std::filesystem::path directory = std::filesystem::path(file_).parent_path();
	std::ifstream input_image_file(file_);

	if (!input_image_file) {
		std::cerr << "No such file. File path does not exist" << std::endl;

	}

	else {
		std::string image;
		// std::cout << "Loading file paths....." << std::endl;
		while (std::getline(input_image_file, image))
		{
			file_paths.push_back(std::filesystem::path(directory).append(image).generic_string());
		}
		input_image_file.close();
	}
	// std::cout << "File path successfully loaded....." << std::endl;
	return file_paths;
}

std::vector<std::string> point_cloud_classifier::data_directory_reader(const std::string& directory)
{
	std::vector<std::string> file_paths;
	std::filesystem::path dir(directory);
	for (auto& p : std::filesystem::directory_iterator(directory))
	{
		if(p.is_regular_file())
			file_paths.push_back(p.path().string());
	}
	return file_paths;
}



std::vector<cv::Mat> point_cloud_classifier::read_and_convert(const std::vector<std::string>& image_paths)
{

	std::vector<cv::Mat> images;
	int i = 1;
	for (std::string image_file : image_paths) {
		if (!image_file.empty()) {

			cv::Mat img = cv::imread(image_file, cv::IMREAD_COLOR);
			i++;

					// convert to yuv color space
			cv::cvtColor(img, img, cv::COLOR_BGR2YCrCb);
			cv::Mat image_uv;
			std::vector<cv::Mat>color_channels, channels_without_y;
			cv::split(img, color_channels);

			channels_without_y.push_back(color_channels[1]);
			channels_without_y.push_back(color_channels[2]);

			images.push_back(cv::Mat());
			cv::merge(channels_without_y, images.back());
		}

		else { std::cout << i << " images FAILED!!" << std::endl; }
	}

	return images;
}

double point_cloud_classifier::bayes_classifier(const cv::Vec2b& pixel_ptr)
{
	return probability_lookup.at<double>(pixel_ptr(0), pixel_ptr(1));
}

double point_cloud_classifier::bayes_classifier(uchar u, uchar v)
{
	return probability_lookup.at<double>(u,v);
}

void point_cloud_classifier::create_lookup_table() {
	for(int x = 0; x < 256; x++)
		for (int y = 0; y < 256; y++)
		{
			double color_freq = 0.0;
			int count_in_skin = 0;
			int count_in_nskin = 0;

			double posterior_prob = 0.0;

			//find pixel value in skin and non skin prob maps
			auto it = skin_probability_map.find(cv::Vec2b(x,y));
			if (it != skin_probability_map.end()) {

				count_in_skin = it->second.first;

			}

			auto it_ = nskin_probability_map.find(cv::Vec2b(x, y));
			if (it_ != nskin_probability_map.end()) {

				count_in_nskin += it_->second.first;

			}

			probability_lookup.at<double>(x,y) = count_in_skin / (count_in_skin + count_in_nskin + 0.00001);
		}
}

// loads probability files into std::map container
void point_cloud_classifier::load_probabilities() {
	std::string val1, val2, val3, val4;
	std::ifstream input_prob_file(prob_file_skin);
	skin_colors_image = cv::Mat(256, 256, CV_32SC1, cv::Scalar::all(0));


	if (input_prob_file) {

		//load probabilities for skin pixels

		while (input_prob_file >> val1 >> val2 >> val3)//>> val4
		{
			skin_colors_image.at<int>(std::stoi(val1), std::stoi(val2)) = std::stoi(val3);

		}
		input_prob_file.close();
		val1.clear();
		val2.clear();
		val3.clear();
		//val4.clear();

	}

	input_prob_file.open(prob_file_nskin);
	non_skin_colors_image = cv::Mat(256, 256, CV_32SC1, cv::Scalar::all(0));
	if (input_prob_file) {

		//load probabilities for non skin pixels

		while (input_prob_file >> val1 >> val2 >> val3)//>> val4
		{
			non_skin_colors_image.at<int>(std::stoi(val1), std::stoi(val2)) = std::stoi(val3);

		}
	}
	input_prob_file.close();

	skin_training_pixels = cv::sum(skin_colors_image)[0];
	nskin_training_pixels = cv::sum(non_skin_colors_image)[0];
	total_training_pixels = skin_training_pixels + nskin_training_pixels;
	class_prob_skin = skin_training_pixels / (double)total_training_pixels;

	train();
}

//counts number of pixel colors(u,v) within a distance of 1(color_threshold)
//calculates probabilites and creates corresponding probabiliy maps. 
void point_cloud_classifier::train()
{
//	cv::dilate(skin_colors_image, skin_colors_image,cv::Mat());
//	cv::dilate(non_skin_colors_image, non_skin_colors_image, cv::Mat());

	auto get_value = [&](const cv::Mat& img, int row, int col) {
		cv::Point2i center(col,row);
		std::vector< cv::Point2i> dirs;

		if (row > 0)
			dirs.push_back(cv::Point2i(0,-1));

		if (row < img.rows -1)
			dirs.push_back(cv::Point2i(0,1));

		if (col > 0)
			dirs.push_back(cv::Point2i(-1,0));

		if (col < img.cols - 1)
			dirs.push_back(cv::Point2i(1,0));


		double count = 0;
		for (const cv::Point2i& dir : dirs)
		{
			count += img.at<int>(dir + center);
		}

		return std::max(static_cast<double>(img.at<int>(center)), count / dirs.size());
	};

	for (int row = 0; row < 256; row++)
		for (int col= 0; col < 256; col++)
		{
			double skin_count = get_value(skin_colors_image, row, col);
			//f_a/(f_a+f_b)
			probability_lookup.at<double>(row,col) = skin_count/ (skin_count + get_value(non_skin_colors_image, row, col) * (skin_training_pixels/(double) nskin_training_pixels) + 0.00001);
			//probability_lookup.at<double>(row, col) = skin_colors_image.at<int>(row, col) / ((double)skin_colors_image.at<int>(row, col) + non_skin_colors_image.at<int>(row, col) + 0.00001);

		}

	//cv::Mat skin_norm;
	//cv::Mat nskin_norm;
	//cv::normalize(skin_colors_image, skin_norm,10,0,cv::NormTypes::NORM_INF,CV_64FC1);
	//cv::normalize(non_skin_colors_image, nskin_norm, 10, 0, cv::NormTypes::NORM_INF, CV_64FC1);

	//cv::imshow("Probability map", probability_lookup);
	//cv::imshow("Skin map", skin_norm);
	//cv::imshow("Non Skin map", nskin_norm);
	//cv::waitKey(0);
}

unsigned point_cloud_classifier::frequency_image(const std::vector<cv::Mat>& images, cv::Mat& output)
{
	unsigned int total = 0;
	output = cv::Mat(256, 256, CV_32SC1, cv::Scalar::all(0));
	for(const auto& img : images)
	{
		img.forEach<cv::Vec2b>([&output](const cv::Vec2b& pixel, const int*)
			{
				output.at<int>(pixel(0), pixel(1)) += 1;
			});
		total += img.rows * img.cols;
	}

	return total;
}



// calculates posterior probabilities P(s/c) = P(c/s)P(S)/P(c) for a given pixel color
double point_cloud_classifier::posterior(double conditonal_prob, double class_prob_skin, double class_prob_color) {
	//class_prob_color = total representation of color c in both skin and non skin dataset
	double posterior = (conditonal_prob * class_prob_skin) / (class_prob_color);

	return posterior;
}


void point_cloud_classifier::write_probabilities() {
	std::fstream skin_file(prob_file_skin, std::ios_base::app | std::ios_base::out);
	std::fstream non_skin_file(prob_file_nskin, std::ios_base::app | std::ios_base::out);
	


	for (int row = 0; row < 256; row++)
		for (int col = 0; col < 256; col++)
		{
			if(skin_colors_image.at<int>(row, col) > 0)
				skin_file << std::to_string(row) << " " << std::to_string(col) << " " << std::to_string(skin_colors_image.at<int>(row, col)) << std::endl;
			
			if(non_skin_colors_image.at<int>(row, col) >0)
			non_skin_file << std::to_string(row) << " " << std::to_string(col) << " " << std::to_string(non_skin_colors_image.at<int>(row, col)) << std::endl;
		}

}

double point_cloud_classifier::color_distance(const cv::Vec2b color1, const cv::Vec2b color2) {
	double dist = std::sqrt(std::pow(color1[0] - color2[0], 2) + std::pow(color1[1] - color2[1], 2));
	return dist;
}

void point_cloud_classifier::show_colors_image(const std::string& window_name, const cv::Mat& img)
{
	cv::Mat display_img(256, 256, CV_8UC1);
	double min_val, max_val;
	cv::minMaxIdx(img, &min_val, &max_val);

	img.convertTo(display_img, CV_8UC1, 255. / max_val);
	cv::imshow(window_name, display_img);
}

}
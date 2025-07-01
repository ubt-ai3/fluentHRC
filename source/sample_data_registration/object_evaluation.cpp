#include <boost/date_time/posix_time/posix_time.hpp>

#include "hand_pose_estimation/hand_model.hpp"
#include "hand_pose_estimation/classification.hpp"

#include "object_evaluation.hpp"

#include <opencv2/opencv.hpp>

#include "enact_core/access.hpp"
#include "enact_core/data.hpp"
#include <state_observation/classification_new.hpp>
#include <state_observation/object_prototype_loader.hpp>

using namespace state_observation;

/*
std::string object_evaluation::get_color_name(const pcl::RGB& color)
{
	const int threshold = 20;
	for (const auto& entry : color_name_map)
	{
		if (std::abs((int)color.r - entry.second.r) < threshold &&
			std::abs((int)color.g - entry.second.g) < threshold &&
			std::abs((int)color.b - entry.second.b) < threshold)
			return entry.first;
	}

	return "unknown";
}


#undef RGB
const std::map<std::string, pcl::RGB> object_evaluation::color_name_map = {
		{"red", pcl::RGB(230, 80, 55)},
		{"blue", pcl::RGB(5, 20, 110)},
		{"cyan", pcl::RGB(5, 115, 185)},
		{"wooden", pcl::RGB(200, 190, 180)},
		{"magenta", pcl::RGB(235, 45, 135)},
		{"purple", pcl::RGB(175, 105, 180)},
		{"yellow", pcl::RGB(250, 255, 61)},
		{"dark_green", pcl::RGB(51, 82, 61)}
};
*/

object_evaluation::object_evaluation(
	state_observation::pointcloud_preprocessing& pc_prepro,
	state_observation::segment_detector& obj_detect,
	state_observation::kinect2_parameters& kinect2_params)
	:
	classifier_(pc_prepro.object_params, classifier::generate_aspects(
		            object_prototype_loader().get_prototypes()
	            )),
	pc_prepro(pc_prepro),
	obj_detect(obj_detect),
	kinect2_params(kinect2_params)

{
	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();

	std::stringstream stream;

	stream << "classified_objects_" << time.date().year()
		<< "-" << time.date().month().as_number()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes();

	file_name = stream.str();

	output.open(file_name + ".csv", std::fstream::out);
}



void object_evaluation::update(const pcl::PointCloud<PointT>::ConstPtr& cloud,
								const cv::Mat& image)
{
	pcl::PointCloud<PointT>::ConstPtr transformed_cloud = pc_prepro.remove_table(cloud);
	//DebUG std::vector<pc_segment::Ptr> segments = obj_detect.update(transformed_cloud);
	std::vector<pc_segment::Ptr> segments;
	
	cv::Mat img = image.clone();

	unsigned int id = 0;
	for (state_observation::pc_segment::Ptr& seg : segments)
	{

		auto results = classifier_.classify_all(*seg);
		std::sort(results.begin(), results.end(), [](const classification_result& lhs, const classification_result& rhs) {
			return lhs.local_certainty_score > rhs.local_certainty_score;
			});


		//for (const auto& res : results)
		//	std::cout << res.local_certainty_score << ", ";

		//std::cout << std::endl;

		if (results.front().prototype->has_mesh() && results.front().local_certainty_score > 0.2f)
		{
			
			std::vector<cv::Point2i> contour;
			Eigen::Translation2f center(img.size[1] * 0.5f, img.size[0] * 0.5f);
			Eigen::Transform<float, 3, 2, 0> transform = center *
												   Eigen::Affine2f(Eigen::Rotation2D<float>(M_PI)) * 
												   center.inverse() * 
												   kinect2_params.rgb_projection * 
												   pc_prepro.get_inv_cloud_transformation();

			for (PointT p : *seg->get_outline())
			{
				Eigen::Vector2f p_2d = (transform * Eigen::Vector3f(p.getArray3fMap()).homogeneous()).hnormalized();
				contour.emplace(contour.end(), p_2d.x(), p_2d.y());
			}

			cv::Rect box = cv::boundingRect(contour);

//			Eigen::Vector2f center_2d = (transform * Eigen::Vector3f(seg->centroid.getArray3fMap()).homogeneous()).hnormalized();
			
			cv::Scalar color(seg->centroid.b, seg->centroid.g, seg->centroid.r, 255);
			cv::drawContours(img, std::vector<std::vector<cv::Point2i>>({ std::move(contour) }), -1, color, 2);
			cv::putText(img, cv::format("%d", id), cv::Point2i(box.tl().x, box.br().y), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 0, 255), 1);
			
//			cv::imshow("Classified objects", img);

			for (int i = 0; i < 3 && i < results.size(); i++)
			{
				output << id << "," << results[i].prototype->get_name() << "," << results[i].local_certainty_score << std::endl;
			}

			id++;
		}

	}



	cv::imwrite(file_name + ".png", img, std::vector<int>({ cv::IMWRITE_PNG_COMPRESSION }));
}

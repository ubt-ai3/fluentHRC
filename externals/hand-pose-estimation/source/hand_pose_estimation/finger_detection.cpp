#include "finger_detection.hpp"
#include "hand_tracker.hpp"
#include <algorithm>
#include "utility.hpp"
#include <opencv2/imgproc.hpp>
/*
	shows a cv::Mat as an image this is needed for use in the immediate window
*/
void show(const cv::Mat& mat)
{
	using namespace std::string_literals;
	static int  i = 0;
	auto name= "Test "s + std::to_string(++i);
	cv::namedWindow(name, cv::WINDOW_NORMAL);
	cv::imshow(name, mat);
	cv::waitKey();
}

/*linearly interpolates between two matrices return an error message if an error occured*/
const char* show_blend(const cv::Mat& mat1,const cv::Mat& mat2,float blend = 0.5)
{
	static std::string error_msg;
	error_msg.clear();
	try {
		using namespace std::string_literals;
		static int  i = 0;
		cv::Mat mat1d, mat2d;
		cv::Mat res;
		cv::cvtColor(mat1, mat1d, cv::COLOR_GRAY2BGR);
		cv::cvtColor(mat2, mat2d, cv::COLOR_GRAY2BGR);
		cv::addWeighted(mat1d, blend, mat2d, 1 - blend, 0, res);
		auto name = "Grey scale "s + std::to_string(++i);
		cv::namedWindow(name, cv::WINDOW_NORMAL);
		cv::imshow(name, res);
		cv::waitKey();
	}
	catch (const std::exception& e)
	{
		error_msg = e.what();
	}
	return error_msg.c_str();
}


/*
prints the content of the @mat as string
calling the function again invalidates the return value
needed for immidate window
*/
const char * print(const cv::Mat& mat)
{
	static std::string buffer;
	std::cout << mat;
	std::stringstream ss;
	ss << mat;
	buffer = ss.str();
	buffer += '\n';
	return buffer.data();
}

/*
Fixed bugs
Mat.type() mismatches
normalized coordinates fixes
Mat(Point(col,row)) takes column first


*/
namespace hand_pose_estimation
{
	void finger_detector::draw_rotated_rect(cv::Mat& mat, const cv::RotatedRect& rect, cv::Scalar color)
	{
		cv::Point2f temp[4];
		rect.points(temp);
		cv::line(mat, temp[0], temp[1], color);
		cv::line(mat, temp[1], temp[2], color);
		cv::line(mat, temp[2], temp[3], color);
		cv::line(mat, temp[3], temp[0], color);
	}

	ArgMin<cv::Point> finger_detector::find_nearest (const cv::Mat& mat, cv::Point point, bool filter(float pix_val))
	{
		CV_DbgAssert(mat.type() == CV_32F);
		ArgMin arg_min(cv::Point{}, std::numeric_limits<double>::max());

		for (int row = 0; row < mat.rows; ++row) {
			const float* p = mat.ptr<float>(row);
			for (int col = 0; col < mat.cols; ++col) {
				if (filter(p[col]))
				{
					//put col first
					arg_min({ col,row }, cv::norm(cv::Point(col, row) - point));
				}
			}
		}
		return arg_min;
	};

	void finger_detector::thin(const cv::Mat& image)
	{
		//	cv::Mat small_mask = image;
		//	{
		//		//apply thinning algorithm
		//		CV_Assert(small_mask.rows > 10 && small_mask.cols > 10, "Hand segment to small");
		//		cv::Mat copy;//TODO generate white border around image
		//		small_mask.copyTo(copy);
		//		bool progress = true;
		//		while (progress)
		//		{
		//			progress = false;
		//			uchar* r[4] = {
		//			small_mask.ptr<CV_8U>(0),
		//			small_mask.ptr<CV_8U>(1),
		//			small_mask.ptr<CV_8U>(2),
		//			small_mask.ptr<CV_8U>(3) };
		//			int i, j;
		//			for (i = 1; i < small_mask.rows - 2; ++i)
		//			{
		//				r[0] = r[1];
		//				r[1] = r[2];
		//				r[2] = r[3];
		//				r[3] = small_mask.ptr<uchar>(i);

		//				for (j = 0; j < small_mask.cols - 3; ++j)
		//				{
		//					std::array<uchar, 11> pixels;
		//					pixels[1] = r[2][1];//pixel 1
		//					pixels[2] = r[1][1];//pixel 2
		//					pixels[3] = r[1][2];
		//					pixels[4] = r[2][2];
		//					pixels[5] = r[3][2];
		//					pixels[6] = r[3][1];
		//					pixels[7] = r[3][0];
		//					pixels[8] = r[2][0];
		//					pixels[9] = r[1][0];
		//					pixels[10] = r[1][1];//pixel 2 again

		//					int B = 0;
		//					int A = 0;
		//					for (int i = 2; i < 10; i++)
		//					{
		//						B += pixels[i];
		//						A += !pixels[i] && pixels[i + 1];
		//					}
		//					bool erase =
		//						pixels[1] &&
		//						(2 <= B && B <= 6) &&
		//						(A == 1) &&
		//						((pixels[2] && pixels[4] && pixels[8] == 0) || r[0][1]/*p11*/) &&
		//						((pixels[2] && pixels[4] && pixels[6] == 0) || r[2][3]/*p15*/);
		//					if (erase)
		//					{
		//						copy.ptr<uchar>(i + 2)[i + 1] = 0;
		//						progress = true;
		//					}
		//				}
		//			}
		//			copy.copyTo(small_mask);
		//		}
		//	}
			//plan: get segmented hand
		//black white it
		//filter with defined filter
		//apply thinning algorithm
	}

	 finger_classification finger_detector::detect(const cv::Mat& image,std::vector<cv::Mat>& debug) const
	{
		static constexpr double epsilon = 0.00001;
		static constexpr double min_finger_size = 0.04;//relative area size to palm_mask
		static constexpr double finger_width_ratio = 0.3; //relative to palm area
		
		CV_Assert(image.depth() == CV_8U);
		//fill_holes(image);

		//1. compute distance transform
		cv::Mat distance_mat;
		cv::distanceTransform(image, distance_mat, cv::DIST_L1, 3);

		//cv::namedWindow("Rotated hand", cv::WINDOW_AUTOSIZE);
		cv::Mat visualize;
		cv::cvtColor(image, visualize, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		cv::Mat dist_gray;
		cv::normalize(distance_mat, dist_gray, 0, 1.0, cv::NORM_MINMAX);
		cv::cvtColor(dist_gray, dist_gray, cv::ColorConversionCodes::COLOR_GRAY2BGR);
		//show(dist_gray);
		//show(distance_mat);

		//2. find palm point
		double min_val;
		double max_val;
		cv::Point max_loc;
		cv::Point min_loc;
		cv::minMaxLoc(distance_mat, &min_val, &max_val, &min_loc, &max_loc);
		const cv::Point& palm_point = max_loc;
		
		visualize.at<cv::Vec3b>(palm_point) = cv::Vec3b(255, 255, 0);

		//3. compute contour
		double outer_circle_radius = find_nearest(distance_mat,palm_point, [](float val) 
			{return val == 0; }).min * 1.15;

		auto find_nearest_boundary_point = [&distance_mat](cv::Point p)
		{
			cv::Mat_<uchar> mat = distance_mat;
			bool exceeds_mat = p.y <0 || p.x <0 ||p.y >= mat.rows || p.x >= mat.cols;
			if (exceeds_mat || mat(p) <= epsilon)//pixel is background
			{
				return find_nearest(distance_mat, p, [](float val) {return val > epsilon; }).arg_min;
			}
			else //pixel is inside
			{
				return find_nearest(distance_mat, p, [](float val) {return val <= epsilon; }).arg_min;
			}
		};

		std::vector<cv::Point2i> contour;
		for (const auto p : sample_circle(palm_point, outer_circle_radius, 10))
		{
			auto point = find_nearest_boundary_point(p);
			visualize.at<cv::Vec3b>(point) = cv::Vec3b(255, 0, 0);
			//visualize.at<cv::Vec3b>(p) = cv::Vec3b(0, 0, 255);
			cv::line(visualize, point, p, { 255,0,0 }, 1);
			contour.push_back(point);
		}

		//4. find wrist_points
		contour.push_back(contour[0]);//close the contour
		ArgMax max(0, cv::norm(contour[0] - contour[1]));
		for (int i = 0; i < contour.size() - 1; i++)
		{
			max(i, cv::norm(contour[i] - contour[i+1]));
		}
		auto wrist_point1 = contour[max.arg_max];
		auto wrist_point2 = contour[max.arg_max+1];

		//visualize contour points
		visualize.at<cv::Vec3b>(wrist_point1) = cv::Vec3b(0, 255, 0);
		visualize.at<cv::Vec3b>(wrist_point2) = cv::Vec3b(0, 255, 0);
		cv::line(visualize, wrist_point1, wrist_point2, cv::Vec3b(0, 100, 100));
		cv::circle(visualize, palm_point,3, { 0,0,10 });
		//show(visualize);

		//5. rotate image to normal position
		cv::Point2f direction = (wrist_point1 + wrist_point2) / 2 - palm_point;
		direction /= cv::norm(direction);
		float angle_deg = std::acos(direction.dot(cv::Point2f(0/*x*/, 1/*y*/)));
		angle_deg *= 180. / CV_PI;
		if (direction.x > 0) //rotate clockwise
			angle_deg = -angle_deg;
		cv::Mat rot_mat;
		cv::Mat rotated = rotate_image(image, palm_point, angle_deg,&rot_mat);

		//rotate wrist points to make wrist_middle - palm_point vertical
		cv::Mat w1 = homogenize<double>(wrist_point1);
		cv::Mat w2 = homogenize<double>(wrist_point2);

		cv::Mat palm_pointd = rot_mat * homogenize<double>(palm_point);
		cv::Mat wrist_point1_rot = rot_mat * w1;
		cv::Mat wrist_point2_rot = rot_mat * w2;
		cv::Mat middle = (w1 + w2) / 2;
		cv::Mat middle_rot = rot_mat * middle;
		cv::Mat vec = middle_rot-palm_pointd;
		CV_DbgAssert(std::abs(vec.at<double>(0, 0)/vec.at<double>(1,0)) < 1e-1);//this vector should be the same direction as (0,1)
		//cut arm at wrist
		/*int ydiff = wrist_point1_rot.at<double>(1, 0) - wrist_point2_rot.at <double>(1, 0);
		int xdiff = wrist_point1_rot.at<double>(0, 0) - wrist_point2_rot.at <double>(0, 0);
		float abs = cv::norm(wrist_point1_rot - wrist_point2_rot) - cv::norm(w1 - w2);
		CV_DbgAssert(std::abs(ydiff) <= 2);*/


		//6. remove area below wrist
		std::vector<cv::Point> below_wrist_contour;
		cv::Point w1_rot;
		cv::Point w2_rot;
		{
			cv::Mat tmp1;
			cv::Mat tmp2;
			wrist_point1_rot.convertTo(tmp1, CV_32S);
			wrist_point2_rot.convertTo(tmp2, CV_32S);
			w1_rot = cv::Point(tmp1);
			w2_rot = cv::Point(tmp2);
		}
		if (w1_rot.x > w2_rot.x)
			std::swap(w1_rot, w2_rot);

		//original
		cv::circle(image, wrist_point1, 4, cv::Scalar(255));
		cv::circle(image, wrist_point2, 4, cv::Scalar(255));
		//transformed
		//show(image);
		//cv::circle(rotated, w1_rot, 4, cv::Scalar(255));
		//cv::circle(rotated, w2_rot, 4, cv::Scalar(255));
		//show(rotated);
		below_wrist_contour.emplace_back(0, w1_rot.y);
		below_wrist_contour.push_back(w1_rot);
		below_wrist_contour.push_back(w2_rot);
		below_wrist_contour.emplace_back(rotated.cols, w2_rot.y);
		below_wrist_contour.emplace_back(rotated.cols, rotated.rows);
		below_wrist_contour.emplace_back(0, rotated.rows);
		cv::Mat cont(below_wrist_contour, false);
		cv::fillPoly(rotated, cont, cv::Scalar(0));


		//create rotated palm_mask
		//rotate points
		cv::Mat palm_mask = cv::Mat(rotated.rows, rotated.cols, CV_8U);
		palm_mask = cv::Scalar(255);
		for (auto& p : contour)
		{
			cv::Mat rot = rot_mat* homogenize<double>(p);
			rot.convertTo(rot, CV_32S);
			p = cv::Point(rot);
		}
		cv::Mat cont_wrapper(contour, false);
		cv::fillPoly(palm_mask, cont_wrapper, cv::Scalar(0));
		cv::Mat finger_mask;
		cv::bitwise_and(rotated, palm_mask, finger_mask);

		//connected components
		//cv::Mat labels;
		//cv::Mat stats;
		//cv::Mat centroids;
		//int num_labels = cv::connectedComponentsWithStats(finger, labels, stats, centroids);

		//compute oriented bounding boxes 
		//find contour
		//find minrect
		/*
		  elements with the same index belong together
		*/

		//7. find connected components
		std::vector<std::vector<cv::Point>> finger_contours;
		std::vector<double> finger_area;
		std::vector<cv::RotatedRect> bounding_boxes;

		cv::findContours(finger_mask, finger_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);


		for (auto& finger : finger_contours)
		{
			finger_area.push_back(cv::contourArea(finger));
		}


		std::vector<std::size_t> indices(finger_contours.size());
		std::iota(indices.begin(), indices.end(), 0);
		std::sort(indices.begin(), indices.end(), [&finger_area](std::size_t lhs, std::size_t rhs) {return finger_area[lhs] > finger_area[rhs]; });
		reorder(finger_contours, indices);
		reorder(finger_area, indices);

		//TODO analyze cutoff
		//remove small bounding boxes created by noise
		double palm_area = cv::contourArea(contour);
		for (int i = 0; i < finger_contours.size();i++)
		{
			if (finger_area[i] < min_finger_size * palm_area)
			{
				if (i == 0)
				{
					//no fingers detected
					//TODO
					std::cout << "no fingers detected\n";
					debug.push_back(rotated);
					debug.push_back(visualize);
					//debug.push_back(finger_mask_col);
					//debug.push_back(palmline_vis);
					return {};
					//__debugbreak();
				}
				finger_contours.resize(i);
				finger_area.resize(i);
				indices.resize(i );
				break;
			}
		}
		if (finger_contours.size() > 5)
		{
			std::cout << "Too many fingers found: " << finger_contours.size() << "\n";
		}
		else if (finger_contours.empty())
		{
			std::cout << "no fingers detected\n";
		}
		cv::Mat finger_mask_col;
		cv::cvtColor(finger_mask, finger_mask_col, cv::COLOR_GRAY2BGR);
		cv::drawContours(finger_mask_col, finger_contours, -1, cv::Scalar(255, 0, 0));


		for (auto& finger : finger_contours)
			bounding_boxes.push_back(cv::minAreaRect(finger));


		//detect thumb
		cv::Point wrist_line{ 1,0 };//wrist_point2-wrist_point1;
		ArgMin min_angle(std::size_t(-1));
		bool thumb_detected = false;
		bool right_handed = false;
		std::size_t index = -1;
		for (auto& box : bounding_boxes)
		{
			index++;
			double angle = numeric::angle((box.center - cv::Point2f(palm_pointd)),(wrist_line));
			min_angle(index, std::min(angle,numeric::PI-angle));
			//std::cout << "Angle:  " << angle<<"\n";
			//max_angle(index, angle);
		}
		cv::RotatedRect thumb_bb;
		if (min_angle.arg_min != -1 && (min_angle.min < 50._deg || min_angle.min > numeric::PI - 50._deg))
		{
			thumb_detected = true;
			thumb_bb = bounding_boxes[min_angle.arg_min];
		}

		// sort fingers according x coordinate
		std::iota(indices.begin(), indices.end(), 0);
		std::sort(indices.begin(), indices.end(), [&bounding_boxes](std::size_t lhs, std::size_t rhs) {return bounding_boxes[lhs].center.x < bounding_boxes[rhs].center.x; });
		reorder(bounding_boxes,indices);
		reorder(finger_contours,indices);
		reorder(finger_area, indices);

		finger_classification classified_fingers;

		std::size_t thumb_index = -1;
		if (thumb_detected)
		{

			//remove thumb
			if (thumb_bb.center.x == bounding_boxes.front().center.x)
			{
				right_handed = true;//thumb left
				thumb_index = 0;
			}
			else if (thumb_bb.center.x == bounding_boxes.back().center.x)
			{
				right_handed = false;//thumb right
				thumb_index = bounding_boxes.size()-1;
			}
			else
			{
				auto res = std::find_if(bounding_boxes.begin(), bounding_boxes.end(), [&thumb_bb](auto val) {return val.center.x == thumb_bb.center.x; });
				right_handed = std::distance(bounding_boxes.begin(), res) * 2 > bounding_boxes.size() ? false:true; 
				std::cout <<"Thumb is in the middle, assume no thumb\n";
				thumb_detected = false;
			}

		}

		if (thumb_detected)
		{
			classified_fingers.insert_finger(finger_type::THUMB, bounding_boxes[thumb_index]);
			bounding_boxes.erase(bounding_boxes.begin() + thumb_index);
			finger_contours.erase(finger_contours.begin() + thumb_index);
			finger_area.erase(finger_area.begin() + thumb_index);
		}

		/**
		* classify fingers
		*/

		//no thumb at this point
		int y = std::min(w1_rot.y,w2_rot.y);
		const auto [success,start_line,end_line] = detect_palm_line(rotated,y, thumb_detected, !right_handed);
		if (!success)
		{
			std::cerr << "No palm line detected\n";
			debug.push_back(rotated);
			debug.push_back(visualize);
			debug.push_back(finger_mask_col);
			//debug.push_back(palmline_vis);
			return {};
		}

		const float finger_width = std::sqrt(palm_area) * finger_width_ratio;
		//find finger axis of bb
		cv::Mat palmline_vis;
		cv::cvtColor(rotated, palmline_vis, cv::COLOR_GRAY2BGR);
		cv::line(palmline_vis, start_line, end_line, cv::Scalar(0, 0, 255));
		
		std::vector<cv::RotatedRect> palm_fingers;

		for (auto& finger_bb : bounding_boxes)
		{
			cv::Point2f temp[4];
			finger_bb.points(temp);
			cv::Scalar color(255, 255, 0);
			cv::line(palmline_vis, temp[0], temp[1], color);
			cv::line(palmline_vis, temp[1], temp[2], color);
			cv::line(palmline_vis, temp[2], temp[3], color);
			cv::line(palmline_vis, temp[3], temp[0], color);

			//compute finger orientation, finger width and number of fingers in this box
			cv::Point2f finger_dir = finger_bb.center - cv::Point2f(palm_pointd);
			finger_dir /= cv::norm(finger_dir);
			cv::Point2f points[4];
			finger_bb.points(points);
			cv::Point2f up = points[1] - points[0];
			float up_len = cv::norm(up);
			up /= cv::norm(up);
			cv::Point2f right = points[2] - points[1];
			float right_len = cv::norm(right);
			right /= cv::norm(right);
			float up_alignment = std::abs(finger_dir.dot(up));
			float right_alignment = std::abs(finger_dir.dot(right));
			if(std::abs(up_alignment - right_alignment) < 0.03)//undecisive finger direction weird finger
			{
				std::cout << "Finger discarded\n";
				continue;
			}
			
			bool flipped = up_alignment < right_alignment;
			float finger_bb_width =  !flipped? right_len : up_len;//smaller value means more orthogonal
			
			int finger_count = std::roundf(finger_bb_width/ finger_width);
			float concrete_finger_width = finger_bb_width / finger_count;
			if (finger_count == 0)
			{
				std::cout << "finger skipped as it was to thin\n";
				continue;
			}




			for(int i = 0;i<finger_count;i++)
			{
				float center_t = (i + 0.5) / finger_count;
				cv::RotatedRect res;
				if (flipped)
				{
					cv::Point2f center_right = numeric::lerp(points[0], points[3], 0.5);
					cv::Point2f center_left = numeric::lerp(points[1], points[2], 0.5);
					cv::Point2f center = numeric::lerp(center_left, center_right, center_t);
					res = cv::RotatedRect(center, { finger_bb.size.width,concrete_finger_width }, finger_bb.angle);//height stays the same
				}
				else
				{	
					cv::Point2f center_left = numeric::lerp(points[0], points[1],0.5);
					cv::Point2f center_right = numeric::lerp(points[2], points[3], 0.5);
					cv::Point2f center = numeric::lerp(center_left, center_right, center_t);
					res = cv::RotatedRect(center, { concrete_finger_width,finger_bb.size.height }, finger_bb.angle);//height stays the same		
				}
				palm_fingers.push_back(res);
			}
		}

		std::sort(palm_fingers.begin(), palm_fingers.end(), [](const cv::RotatedRect& lhs, const cv::RotatedRect& rhs) {return lhs.center.x < rhs.center.x; });
		for (const auto& bb : palm_fingers)
		{
			int i = 0;
			//find part of palmline which falls into this line segment
			//MAYBE project on palm_line instead of use x coordinate
			for (; bb.center.x > numeric::lerp(start_line.x, end_line.x, i / 4.0); i++)
				;//semicolon
			if (i <= 0 || i > 4)
			{
				classified_fingers.insert_finger(finger_type::UNKNOWN, bb);
			}
			else
			{
				if (right_handed)
				{
					classified_fingers.insert_finger((finger_type)i, bb);
				}
				else
					classified_fingers.insert_finger((finger_type)(5 - i), bb);
			}
	
		}

		for (size_t i = 0; i < classified_fingers.bounding_boxes.size(); i++)
		{
			draw_rotated_rect(palmline_vis, classified_fingers.bounding_boxes[i]);

			cv::Point2i center = classified_fingers.bounding_boxes[i].center;
			const char* name = "not named";
			switch (classified_fingers.finger_types[i])
			{
			case finger_type::THUMB:
				name = "thumb";
				break;
			case finger_type::INDEX:
				name = "index finger";
				break;
				case finger_type::MIDDLE:
				name = "middle finger";
				break;
			case finger_type::RING:
				name = "ring finger";
				break;
			case finger_type::LITTLE:
				name = "little finger";
				break;
			case finger_type::UNKNOWN:
				name = "unknown";
				break;
			default:
				__debugbreak();
			}
			cv::putText(palmline_vis, name, center, cv::FONT_HERSHEY_PLAIN, 0.6, cv::Scalar(0, 255, 1));
			//cv::addText(palmline_vis,name,center , cv::fontQt("Times"));
		}
		//show(palmline_vis);
		//show(finger_mask_col);
		debug.push_back(rotated);
		debug.push_back(visualize);
		debug.push_back(finger_mask_col);
		debug.push_back(palmline_vis);

		//cv::imshow("Rotated Hand", rotated);
		//cv::waitKey();
		finger_classification back_rotated_fingers;
		for (auto& finger : classified_fingers.bounding_boxes)
		{
			//std::array<cv::Point2f,4> points;
			//finger.points(&points[0]);
			cv::Mat inv;
			cv::invertAffineTransform(rot_mat, inv);
				cv::Point2f new_center;
				cv::Mat vec = inv * homogenize<double>(finger.center);
				new_center.x = vec.at<double>(0, 0);
				new_center.y = vec.at<double>(1, 0);

			//for some reason cv::RotatedRect(p1,p2,p3) almost always throws -> probably a bug in opencv, so we use a different constructor
			back_rotated_fingers.bounding_boxes.push_back(cv::RotatedRect(new_center,finger.size,finger.angle+angle_deg));
		}
		//compute the fingertip position and the fingerpointing direction
		//assumption: fingertip is at the far away side of the bounding box relative to the palm_point
		for (const auto& finger : back_rotated_fingers.bounding_boxes)
		{
			std::array<cv::Point2f, 4> corners;
			finger.points(corners.data());
			cv::Point2f t (palm_point);
			std::sort(corners.begin(), corners.end(), [&t](const cv::Point2f& lhs,const cv::Point2f rhs) {return cv::norm(lhs - t) < cv::norm(rhs -t); });
			cv::Point2f finger_root = (corners[0] + corners[1]) / 2;//bottom points
			cv::Point2f fingertip = (corners[2] + corners[3]) / 2; //top points
			back_rotated_fingers.fingerpointing_direction.push_back(fingertip - finger_root);
			back_rotated_fingers.fingertip_pos.push_back(0.7 * fingertip + 0.3 * finger_root);//the fingertip is not at the border of the bounding box, but somewhere in the middle
			back_rotated_fingers.fingerroot_pos.push_back(finger_root);
		}

		back_rotated_fingers.finger_types = classified_fingers.finger_types;
		return back_rotated_fingers;
	}

	std::vector<pointing_gesture_result> pointing_gesture_detector::detect(const visual_input& vis_input)
	{
		if (!vis_input.has_cloud())
			throw std::runtime_error("No cloud present in visual_input");

		cv::Mat3b input3b;
		if (vis_input.img.channels() == 4)
		{
			cv::cvtColor(vis_input.img, input3b, cv::COLOR_BGRA2BGR);
		}
		else
			input3b = vis_input.img;
		
		vis_input.img.copyTo(visualization);


		cv::Rect roi_general;
		if (roi)
			roi_general = *roi;
		else
			roi = cv::Rect({ 0,0 }, input3b.size());

		skin_detect.skin_candidate(input3b(roi_general));
		std::vector<pointing_gesture_result> out;
		evaluation_mat.clear();
		int i = 0;

		if(roi)
			finger_detector::draw_rotated_rect(visualization, cv::RotatedRect((roi->br() + roi->tl()) / 2, roi->size(), 0));

		for (const auto& region : skin_detect.get_regions())
		{	

			finger_classification detected_finger = finger_detect.detect(region, this->evaluation_mat);
			
			//convert 2d image pointing classification to 3d gesture
			const auto& bb = skin_detect.get_boxes()[i];
			finger_detector::draw_rotated_rect(visualization, cv::RotatedRect((bb.br() + bb.tl()) / 2 + roi_general.tl(), bb.size() , 0));

			for (int j = 0; j < detected_finger.fingertip_pos.size();j++)//transform every finger back to the original coordinate system
			{
				const auto& roi = skin_detect.get_boxes()[i];

				//transform fingerposition to skin detection coordinates to global image coordinates
				detected_finger.fingertip_pos[j] += cv::Point2f(roi.tl()) + cv::Point2f(roi_general.tl());
				detected_finger.fingerroot_pos[j] += cv::Point2f(roi.tl()) + cv::Point2f(roi_general.tl());
				detected_finger.bounding_boxes[j].center += cv::Point2f(roi.tl()) + cv::Point2f(roi_general.tl());
				finger_detector::draw_rotated_rect(visualization, detected_finger.bounding_boxes[j], { 0,255,0 });
			}

			if (detected_finger.bounding_boxes.size() == 1)//only one finger detected -> assumes zeigegeste
			{
				finger_detector::draw_rotated_rect(visualization, detected_finger.bounding_boxes[0], { 0,0,255 });
				cv::line(visualization, detected_finger.fingertip_pos[0], detected_finger.fingerroot_pos[0], cv::Scalar(256, 0, 0));
				pointing_gesture_result gesture;


				//convert 2D points to 3D coordinates TODO doesn't work yet
				img_segment seg2D;
				seg2D.bounding_box = skin_detect.get_boxes()[i];
				seg2D.bounding_box.x += roi_general.x;
				seg2D.bounding_box.y += roi_general.y;
				seg2D.mask = region;
				int finger_width = std::min(detected_finger.bounding_boxes[0].size.height, detected_finger.bounding_boxes[0].size.width);
				//cv::erode(seg2D.mask, seg2D.mask, cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,cv::Size(finger_width/2, finger_width/2)));
				img_segment_3d seg = img_segment_3d(vis_input, seg2D, table_plane);

				visual_input::PointT fingertip3D;
				seg.get_surface_point_img(cv::Point2i(detected_finger.fingertip_pos[0]),fingertip3D);
				visual_input::PointT fingerroot3D;
				seg.get_surface_point_img(cv::Point2i(detected_finger.fingerroot_pos[0]), fingerroot3D);

				gesture.finger_tip3D = { fingertip3D.x,fingertip3D.y,fingertip3D.z };

				cv::Point3f root = { fingerroot3D.x,fingerroot3D.y,fingerroot3D.z };
				gesture.finger_root3D = root;

				gesture.pointing_direction3D = (gesture.finger_tip3D - root);
				gesture.pointing_direction3D /= std::sqrt(gesture.pointing_direction3D.dot(gesture.pointing_direction3D));

				gesture.pointing_direction2D = detected_finger.fingerpointing_direction[0];
				gesture.finger_tip2D = detected_finger.fingertip_pos[0];

				out.push_back(gesture);
			}
			i++;
		}
		return out;
	}


	void finger_detector::fill_holes(cv::InputOutputArray input)
	{
		int background_color = 42;
		cv::floodFill(input,cv::Point { 0,0 },cv::Scalar_<uchar>(background_color));
		std::vector<uchar> lut(256, 255);//everything white except pixels with value background_color
		//lut[0] = 255;
		//lut[255] = 255;
		lut[background_color] = 0;
		cv::LUT(input, lut, input);
	}

	std::vector<cv::Point2f> finger_detector::sample_circle(cv::Point2f center, double radius, double delta_theta)
	{
		std::vector<cv::Point2f> out;
		for (double theta = 0; theta < 360; theta += delta_theta)
		{
			cv::Point2f p;
			p.x = radius * std::cos(CV_PI * theta/180);
			p.y = radius * std::sin(CV_PI * theta / 180);
			out.push_back(center + p);
		}
		return out;
	}

	cv::Mat finger_detector::rotate_image(const cv::Mat& image, cv::Point2i rot_center, double angle_deg, cv::Mat* transformation_matrix)
	{

		cv::Size new_size;
		cv::Point2i new_origin;
		{	//compute the rotation parameters and the new size of the image holding enoug space
			//by transforming the corners of the image and finding the max min coordinates
			std::vector<cv::Point2d> corners;
			corners.push_back(cv::Point2d{ 0,0 });
			corners.push_back(cv::Point2d(0, image.rows));
			corners.push_back(cv::Point2d(image.cols, 0));
			corners.push_back(cv::Point2d(image.cols, image.rows));
			corners.push_back(cv::Point2d(rot_center));
			auto rot_mat = cv::getRotationMatrix2D(rot_center, angle_deg, 1);
			cv::Point2d min;
			cv::Point2d max;
			for (int i = 0; i < corners.size(); i++)
			{
				cv::Point3d hom;
				cv::Point2d euclidean;
				cv::Mat euclidean_wrapper(euclidean, false);
				cv::Mat hom_wrapper(hom, false);
				hom.x = corners[i].x, hom.y = corners[i].y, hom.z = 1;
				euclidean_wrapper = rot_mat * hom_wrapper;
				//euclidean.x = res.x, euclidean.y = res.y;

				if (i == 0)
				{
					min = euclidean;
					max = euclidean;
				}
				else
				{
					min.x = std::min(min.x, euclidean.x);
					min.y = std::min(min.y, euclidean.y);
					max.x = std::max(max.x, euclidean.x);
					max.y = std::max(max.y, euclidean.y);
				}
			}
			new_size = cv::Size(cv::Point2i{ max + cv::Point2d{1, 1} - min });//round up
			new_origin = -min;
		}

		cv::Mat out(new_size.height, new_size.width, image.type());
		out = cv::Scalar(0);
		auto rot_mat = cv::getRotationMatrix2D(rot_center, angle_deg, 1);
		//performed transformation
		//translation rot_center -> new_rotcenter
		//rotation around new
		rot_mat.at<double>(0, 2) += new_origin.x;
		rot_mat.at<double>(1, 2) += new_origin.y;
		cv::warpAffine(image, out, rot_mat, out.size());
		if (transformation_matrix)
		{
		//	cv::Mat p = homogenize<double>(trafo_origin);
		//	p.at<double>(2, 0) = 0;
		//	cv::Mat temp = rot_mat* p;
		//	cv::Mat trafo;
		//	rot_mat.copyTo(trafo);
		//	trafo.at<double>(0, 2) += temp.at<double>(0, 0);
		//	trafo.at<double>(1, 2) += temp.at<double>(1, 0);
		//	cv::Point in = rot_center;
		//	cv::Mat origin = homogenize<double>(trafo_origin);
		//	origin.at<double>(2, 0) = 0;
		//	//cv::Mat palm = rot_mat * homogenize<double>(in + trafo_origin);
		//	//cv::Mat palm_ = rot_mat * homogenize<double>(in)+rot_mat * origin;
		//	//cv::Mat palm2 = trafo * homogenize<double>(in);
		//	cv::Point center = rot_center + trafo_origin;
		//	cv::Mat zero = rot_mat* homogenize<double>(in + trafo_origin) - trafo * homogenize<double>(in);
			*transformation_matrix = rot_mat;
		}
		return out;
	}


	std::tuple<bool,cv::Point, cv::Point> finger_detector::detect_palm_line(const cv::Mat& rotated_hand,int start_val_y, bool thumb_detected, bool thumb_right)
	{
		cv::Point palm_line_begin;
		cv::Point palm_line_end;
		bool found_line = false;
		bool on_palm_line = false;
		bool finished = false;
		//search palm_line
		const cv::Mat& normedHand = rotated_hand;
		cv::Point current_segment_start;
		int x_start = 0;
		int x_end = normedHand.cols;
		int x_incr = 1;
		if (thumb_detected && !thumb_right)
		{
			x_start = normedHand.cols - 1;
			x_end = -1;
			x_incr = -1;
		}
		for (int y = start_val_y; y >= 0 && !finished; y--)
		{
			bool prev_black = true;
			std::vector <std::pair<int, int>> white_lines;
			int line_start{};
			const uchar* row = normedHand.ptr(y);

			for (int x = x_start; x != x_end; x+=x_incr)
			{


				uchar pixel = row[x];
				if (prev_black && pixel > 0) //switch to white
				{
					line_start = x;
					prev_black = false;
				}
				else if (!prev_black && pixel == 0)//switch to black
				{
					white_lines.push_back({ line_start,x });
					prev_black = true;
				}


			}
			if (!prev_black)//simulate white to black transitions at the end of each row
			{
				white_lines.push_back({ line_start, x_end-x_incr});
				break;
			}
			if (thumb_detected)
			{
				if (white_lines.size() >= 3)
				{
					int start = white_lines[0].first;
					int end = white_lines[1].second;
					if (start > end)
						std::swap(start, end);
					return { true,{start,y},{end,y} };
				}
			}
			else
			{
				if (white_lines.size() == 2)
				{
					int start = white_lines[0].first;
					int end = white_lines[1].second;
					if (start > end)
						std::swap(start, end);
					return { true,{start,y},{end,y} };
				}
			}

			//output palm line
		}

		//CV_DbgAssert(found_line && !on_palm_line && palm_line_begin.y == palm_line_end.y);
		if ((!found_line && !on_palm_line && palm_line_begin.y == palm_line_end.y))
		{
			return { false,{},{} };
		}
		//report the line below since we want the highest connected line paralleling the wrist line
		if (palm_line_begin.y < normedHand.rows - 1)
		{
			++palm_line_begin.y;
			++palm_line_end.y;
		}

		if (palm_line_begin.x <= palm_line_end.x)
			return std::tuple{true, palm_line_begin ,palm_line_end };
		else //inverse coodinates for thumb_right == true
			return {true, palm_line_end,palm_line_begin };
	}


	std::optional<cv::RotatedRect> finger_classification::get_finger(finger_type finger) const
	{
		int index = std::find(finger_types.begin(), finger_types.end(), finger)-finger_types.begin();
		if (index == finger_types.size())
			return {};
		else
			return bounding_boxes[index];
	}

	void finger_classification::insert_finger(finger_type finger, const cv::RotatedRect& bb)
	{
		bounding_boxes.push_back(bb);
		finger_types.push_back(finger);
	}


}
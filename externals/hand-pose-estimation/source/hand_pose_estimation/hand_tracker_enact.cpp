#include "hand_tracker_enact.hpp"

#include <filesystem>

#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include "enact_core/access.hpp"

#include "hololens_hand_data.hpp"

namespace hand_pose_estimation
{

using namespace std::chrono;

/////////////////////////////////////////////////////////////
//
//
//  Class: hand_trajectory
//
//
/////////////////////////////////////////////////////////////

hand_trajectory::hand_trajectory(const hand_instance& source, const Eigen::Affine3f& transformation)
{
	std::lock_guard<std::mutex> lock(source.update_mutex);

	if (source.poses.size() <= 1)
		throw std::exception("Hand has no initial pose");

	this->certainty_score = source.certainty_score;
	this->right_hand = source.right_hand;

	auto last = --source.poses.cend();
	for (auto iter = source.poses.cbegin(); iter != last; ++iter)
	{
		poses.emplace_back(iter->time_seconds, iter->pose);
		poses.back().second.wrist_pose = transformation * poses.back().second.wrist_pose;
	}
}

hand_trajectory::hand_trajectory(float certainty_score, 
	float right_hand, 
	std::chrono::duration<float> timestamp,
	const hand_pose_18DoF& pose)
		:
	certainty_score(certainty_score),
	right_hand(right_hand)
{
	poses.emplace_back(timestamp, pose);
}

bool hand_trajectory::update(const hand_instance& source, const Eigen::Affine3f& transformation)
{
	bool updated = false;

	std::lock_guard<std::mutex> lock(source.update_mutex);

	if (source.poses.size() <= 1)
		throw std::exception("Hand has no initial pose");

	this->certainty_score = source.certainty_score;
	this->right_hand = source.right_hand;

	auto first = source.poses.cend();
	for (auto iter = --source.poses.cend(); iter != source.poses.begin() && iter->time_seconds > poses.back().first; --iter)
	{
		first = iter;
	}

	for (auto iter = first; iter != source.poses.cend(); ++iter)
	{
		updated = true;
		poses.emplace_back(iter->time_seconds, iter->pose);
		poses.back().second.wrist_pose = transformation * poses.back().second.wrist_pose;
	}

	return updated;
}


const std::shared_ptr<enact_core::aspect_id> hand_trajectory::aspect_id
= std::shared_ptr<enact_core::aspect_id>(new enact_core::aspect_id("hand"));


/////////////////////////////////////////////////////////////
//
//
//  Class: hand_tracker_enact
//
//
/////////////////////////////////////////////////////////////

hand_tracker_enact::hand_tracker_enact(enact_core::world_context& world,
	Eigen::Affine3f world_transformation,
	std::chrono::duration<float> purge_duration,
	std::chrono::duration<float> frame_duration,
	int max_threads,
	bool spawn_hands,
	std::chrono::high_resolution_clock::time_point start_time,
	Eigen::Vector3f back_palm_orientation,
	Eigen::Hyperplane<float, 3> background)
	:
	hand_tracker(purge_duration, max_threads, spawn_hands, back_palm_orientation, background),
	world(world),
	transformation(std::move(world_transformation)),
	inv_transformation(transformation.inverse()),
	frame_duration(frame_duration),
	cloud_stamp_offset{},
	last_input_stamp{},
	last_hololens_stamp{},
	start_time(start_time)
{
	reset(start_time);
}

hand_tracker_enact::~hand_tracker_enact()
{

}

void hand_tracker_enact::reset(std::chrono::high_resolution_clock::time_point start_time)
{
	if (file.is_open())
		file.close();

	this->start_time = start_time;

	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
	std::stringstream stream;
	stream << time.date().year()
		<< "-" << time.date().month().as_number()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes();

	std::filesystem::create_directory(stream.str());

	file.open(stream.str() + "/hand_tracking.csv");
	file << "time (ms),source hololens2,hand,wrist x, wrist y, wrist z, other keypoints ..." << std::endl;
}

void hand_tracker_enact::add_hand_pose(const hololens::hand_data::ConstPtr& pose, enact_priority::operation op)
{
	try
	{
		auto input_stamp = last_input_stamp.load();
		if (input_stamp.count() == 0.f)
			return;

		bool right = pose->hand == hololens::hand_index::RIGHT;

		entity_id hand_id = nullptr;
		{
			auto& hand = right ? right_hand : left_hand;
			auto& other_hand = right ? left_hand : right_hand;

			auto hand_stamp = std::chrono::duration_cast<duration_t>(
				std::chrono::time_point_cast<duration_t>(pose->utc_timestamp).time_since_epoch()
				);

			std::lock_guard<std::mutex> lock(update_mutex);

			last_hololens_stamp = std::max(last_hololens_stamp.load(), hand_stamp);

			hand_id = hand.id;
			++hand.received_messages;
			hand.delay += (std::chrono::file_clock::now() - hand_stamp).time_since_epoch();
			hand.delay /= std::min((unsigned int)2, hand.received_messages);
		}


		if (!pose->valid)
			return;

		using H = hololens::hand_key_point;
		Eigen::Matrix3Xf keypoints(3, 21);
		int i = 0;

		for (int j = 1; j < (int)H::SIZE; ++j)
		{
			if ((H)j == H::INDEX_METACARPAL || (H)j == H::MIDDLE_METACARPAL || (H)j == H::RING_METACARPAL || (H)j == H::LITTLE_METACARPAL)
				continue;
			//for (int j : {H::WRIST, H::THUMB_METACARPAL, H::THUMB_PROXIMAL, H::THUMB_DISTAL, H::THUMB_TIP,
			//	H::INDEX_METACARPAL, H::INDEX_PROXIMAL, H::INDEX_INTERMEDIATE, H::INDEX_DISTAL, H::INDEX_TIP,
			//	H::MIDDLE_METACARPAL, H::MIDDLE_PROXIMAL, H::MIDDLE_INTERMEDIATE, H::MIDDLE_DISTAL, H::MIDDLE_TIP,
			//	H::RING_METACARPAL, H::RING_PROXIMAL, H::RING_INTERMEDIATE, H::RING_DISTAL, H::RING_TIP,
			//	H::LITTLE_METACARPAL, H::INDEX_PROXIMAL, H::INDEX_INTERMEDIATE, H::INDEX_DISTAL, H::INDEX_TIP}) {
			keypoints.col(i) = (pose->key_data)[j].position;
			i++;
		}

		net_evaluation eval;
		eval.left_hand_pose = keypoints;

		const auto& wrist = (pose->key_data)[(int)H::WRIST];

		auto pose_18DoF = hand_pose_estimation::estimate_relative_pose(this->gd_scheduler.get_hand_kinematic_parameters(), keypoints, pose->hand == hololens::hand_index::RIGHT);
		//pose_18DoF->wrist_pose = Eigen::Translation3f(wrist.position) * wrist.rotation;


		Eigen::Matrix3Xf default_key_points = pose_18DoF->get_key_points();

		std::vector<Eigen::Vector3f> observed_points;
		std::vector<Eigen::Vector3f> model_points;


		for (int joint : {0, 2, 5, 9, 17})
		{
			observed_points.emplace_back(keypoints.col(joint));
			model_points.emplace_back(default_key_points.col(joint));
		}

		pose_18DoF->wrist_pose = hand_pose_estimation::fit_pose(observed_points, model_points, true);

		std::chrono::duration<float> relative_cloud_stamp = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::file_clock::now().time_since_epoch() - cloud_stamp_offset.load());

		if (hand_id==nullptr)
		{
			std::stringstream id;
			id << "hand@" << std::setprecision(3) << relative_cloud_stamp.count() << "@" << pose_18DoF->wrist_pose.translation();

			entity_id obj_id(new enact_core::entity_id(id.str()));
			auto hand_instance_p = std::make_unique<hand_trajectory_data>(hand_trajectory(1.f, right * 1.f, relative_cloud_stamp, *pose_18DoF));

			world.register_data(obj_id, hand_trajectory::aspect_id, std::move(hand_instance_p));

			{
				std::lock_guard<std::mutex> lock(update_mutex);

				(right ? right_hand : left_hand).id = obj_id;
			}

			(*emitter)(std::make_pair(obj_id,nullptr), enact_priority::operation::CREATE);
		}
		else if (op == enact_priority::operation::UPDATE)
		{
			{
				enact_core::lock l(world, enact_core::lock_request(hand_id, hand_trajectory::aspect_id, enact_core::lock_request::write));
				enact_core::access<hand_trajectory_data> access_object(l.at(hand_id, hand_trajectory::aspect_id));
				auto& trajectory = access_object->payload;

				trajectory.poses.emplace_back(relative_cloud_stamp, *pose_18DoF);

				file << duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count()
					<< ",1"
					<< (pose->hand == hololens::hand_index::RIGHT ? ",right," : ",left,")
					<< trajectory.poses.back().second.get_key_points().transpose().format(csv_format) << std::endl;

			}

			(*emitter)(std::make_pair(hand_id, nullptr), enact_priority::operation::UPDATE);
		}
	}
	catch (...)
	{
		//prevent exceptions reaching grpc server
	}
}

void hand_tracker_enact::process_input()
{
	if (!input)
		return;

	visual_input::ConstPtr local_input;
	{
		std::lock_guard<std::mutex> lock(update_mutex);
		local_input = input;
		last_input_stamp.store(input->relative_timestamp());
		input = nullptr;
	}

	auto now = std::chrono::file_clock::now().time_since_epoch();
	cloud_stamp_offset = now - std::chrono::duration_cast<duration_t>(local_input->timestamp_seconds);

	if (now < last_hololens_stamp.load() + frame_duration)
	{
		int hand_updates = 0;

		auto handle_hand = [&](hand_info& hand, float right)
		{
			entity_id id;
			{
				std::lock_guard<std::mutex> lock(update_mutex);

				id = hand.id;
			}

			if (!id)
				return;

			try
			{
				img_segment::Ptr seg;
				{
					hand_pose_18DoF::ConstPtr pose;
					std::chrono::duration<float> hand_timestamp;
					{
						enact_core::lock l(world, enact_core::lock_request(id, hand_trajectory::aspect_id, enact_core::lock_request::write));
						enact_core::access<hand_trajectory_data> access_object(l.at(id, hand_trajectory::aspect_id));
						auto& trajectory = access_object->payload;

						hand_timestamp = trajectory.poses.back().first;
						if (hand_timestamp + frame_duration < local_input->timestamp_seconds)
							return;

						pose = std::make_shared<hand_pose_18DoF>(trajectory.poses.back().second);
					}

					seg = get_segment(*local_input, *pose, hand_timestamp);

					{
						std::unique_lock<std::mutex> lock(update_mutex);
						auto& tracked = hand.kinect_tracked_hand;

						if (tracked)
						{
							lock.unlock();

							tracked->add_or_update(*seg->particles[pose->right_hand], true);

							std::lock_guard<std::mutex> hand_lock(tracked->update_mutex);
							tracked->observation_history.push_back(seg);
							tracked->certainty_score = 1;
							tracked->right_hand = right;

							if (tracked->observation_history.size() > 4)
								tracked->observation_history.pop_front();
						}
						else
						{
							tracked = std::make_shared<hand_instance>(seg);
							tracked->certainty_score = 1;
							tracked->right_hand = right;
							hands.emplace(tracked);
							hand_to_ids.emplace(tracked, hand.id);
							bb_tracker.add_hand(tracked);
						}
					}
				}

				(*emitter)(std::make_pair(id, seg), enact_priority::operation::UPDATE);

				hand_updates++;
			}
			catch (...)
			{
				// don't update right now
			}
		};

		handle_hand(left_hand, 0.f);
		handle_hand(right_hand, 1.f);

		if(hand_updates >= 2)
			return;
	};



	// run skin detection and segment to hand association
	std::vector<hand_instance::Ptr> prev_candidates = bb_tracker.get_hands();
	std::vector<hand_instance::Ptr> prev_hands;
	float hand_probability_threshold = gd_scheduler.get_hand_pose_parameters().hand_probability_threshold;
	for (auto& hand : prev_candidates)
		if (hand->certainty_score > hand_probability_threshold)
			prev_hands.push_back(hand);

	auto segments = skin_detector.detect_segments(*local_input, prev_hands);
	bb_tracker.update(*local_input, segments);
	auto current_hands = bb_tracker.get_hands();

	gd_scheduler.update(local_input, current_hands);

	// detect new, deleted updated hands
	std::map<entity_id, img_segment::ConstPtr> updated_hands;
	std::map<entity_id, img_segment::ConstPtr> created_hands;
	{
		float threshold = get_hand_certainty_threshold();

		std::unique_lock<std::mutex> lock(update_mutex);
		for (const auto& hand : current_hands)
		{
			// do not lock hand mutex here, this is done when creating hand_trajectory
			if (!hand_to_ids.contains(hand) && (hand->observation_history.size() < 3 || hand->certainty_score <= threshold))
				continue;

			if (!hand_to_ids.contains(hand))
			{ // new hand
				std::stringstream id;
				id << "hand@" << std::setprecision(3) << hand->observation_history.front()->timestamp.count() << "s@" << hand->poses.front().key_points.col(0);

				entity_id obj_id(new enact_core::entity_id(id.str()));
				auto hand_instance_p = std::make_unique<hand_trajectory_data>(hand_trajectory(*hand, transformation));

				world.register_data(obj_id, hand_trajectory::aspect_id, std::move(hand_instance_p));
				hand_to_ids.emplace(hand, obj_id);
				

				new_hands.emplace_back(hand);
				created_hands.emplace(obj_id, hand->observation_history.back());
			}
			else {
				auto id_iter = hand_to_ids.find(hand);

				if (hand->certainty_score <= threshold)
				{ // no longer a hand
					if (id_iter != hand_to_ids.end())
					{
						(*emitter)(std::make_pair(id_iter->second, nullptr), enact_priority::operation::DELETED);
						hand_to_ids.erase(id_iter);
					}
				}
				else
				{ // update hand

						enact_core::lock l(world, enact_core::lock_request(id_iter->second, hand_trajectory::aspect_id, enact_core::lock_request::write));
						enact_core::access<hand_trajectory_data> access_object(l.at(id_iter->second, hand_trajectory::aspect_id));
						auto& trajectory = access_object->payload;

						if (trajectory.update(*hand, transformation))
						{
							updated_hands.emplace(id_iter->second, hand->observation_history.back());
							file << duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count()
								<< ",0"
								<< (hand->right_hand > 0.5 ? ",right," : ",left,") 								
								<< trajectory.poses.back().second.get_key_points().transpose().format(csv_format) << std::endl;

						}

				}

			}
		}

		hands = std::set<hand_instance::Ptr>(current_hands.begin(), current_hands.end());

		// purged hands
		for (auto id_iter = hand_to_ids.begin(); id_iter != hand_to_ids.end();)
		{
			auto iter = hands.find(id_iter->first);
			if (iter == hands.end())
			{

				(*emitter)(std::make_pair(id_iter->second, nullptr), enact_priority::operation::DELETED);
				id_iter = hand_to_ids.erase(id_iter);
			}
			else
				++id_iter;
		}
	}

	for(const auto& id : created_hands)
		(*emitter)(id, enact_priority::operation::CREATE);

	for (const auto& id : updated_hands)
		(*emitter)(id, enact_priority::operation::UPDATE);

	if (!new_hands.empty())
		new_hands_condition.notify_all();



	batch_completion_condition.notify_all();
}

img_segment::Ptr hand_tracker_enact::get_segment(const visual_input& input, const hand_pose_18DoF& workspace_pose, std::chrono::duration<float> timestamp) const
{
	hand_pose_18DoF pose(workspace_pose.hand_kinematic_params, 
		inv_transformation * workspace_pose.wrist_pose, 
		workspace_pose.finger_bending, 
		workspace_pose.thumb_adduction, 
		workspace_pose.finger_spreading, 
		workspace_pose.right_hand, 
		workspace_pose.bone_scaling);

	float thickness = 0.4f * get_hand_kinematic_parameters().thickness;

	std::vector<Eigen::Vector2f> points_2d;
	const auto& keypoints = pose.get_key_points();

	auto project_into_image = [&](const Eigen::Vector3f& p)
	{
		return (input.img_projection * inv_transformation * p.homogeneous()).hnormalized();
	};


	for (int i = 0; i < keypoints.cols(); i++)
		points_2d.emplace_back(project_into_image(keypoints.col(i)));

	Eigen::Vector4f wrist = keypoints.col(0).homogeneous();
	Eigen::Vector2f wrist_proj = ((input.img_projection * inv_transformation.matrix()) * wrist).hnormalized();
	wrist(0) += thickness;
	const float radius = (wrist_proj - ((input.img_projection * inv_transformation.matrix()) * wrist).hnormalized()).norm();

	const auto inf = std::numeric_limits<float>::infinity();
	cv::Point2f min(inf, inf), max(-inf, -inf);
	for (const auto& point : points_2d)
	{
		min.x = std::min(min.x, point.x() - radius);
		min.y = std::min(min.y, point.y() - radius);
		max.x = std::max(max.x, point.x() + radius);
		max.y = std::max(max.y, point.y() + radius);
	}
	max += cv::Point2f(1, 1);

	const float width = max.x - min.x;
	const float height = max.y - min.y;

	auto seg = std::make_shared<img_segment>();
	seg->hand_certainty = 1.f;
	seg->timestamp = timestamp;

	
	seg->bounding_box = cv::Rect2i(min, max);
	
	if (2 * width * height < static_cast<float>(input.cloud->width * input.cloud->height))
	{
		Eigen::Matrix<float, 3, 4> transform = Eigen::Affine2f(Eigen::Translation2f(-seg->bounding_box.x, -seg->bounding_box.y)).matrix() *
			input.img_projection * inv_transformation.matrix();

		std::vector<cv::Point2i> pixels;
		const cv::Scalar color(255, 255, 255);
		seg->mask = cv::Mat(seg->bounding_box.height, seg->bounding_box.width, CV_8UC1, cv::Scalar(0, 0, 0));
		for (auto p : points_2d)
		{
			cv::Point2f pixel(p.x() - min.x, p.y() - min.y);

			cv::circle(seg->mask, pixel, radius, color, cv::FILLED);

			if (!pixels.empty())
			{
				if (pixels.size() % 4 == 1)
				{
					cv::line(seg->mask, pixels.front(), pixel, color, 2 * radius);
				}
				else
					cv::line(seg->mask, pixels.back(), pixel, color, 2 * radius);
			}

			pixels.push_back(pixel);
			//cv::putText(canvas, std::to_string(i), pixel,cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(0,0,255));
		}

		std::vector<cv::Point2i> palm;
		palm.push_back(pixels[2]);
		palm.push_back(pixels[5]);
		palm.push_back(pixels[17]);
		palm.push_back(pixels[1] - pixels[5] + pixels[17]);

		cv::fillConvexPoly(seg->mask, palm, color);

		std::vector<std::vector<cv::Point2i>> contours;
		cv::findContours(seg->mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		if(!contours.empty())
			seg->contour = contours.front();
	}
	else
	{
		seg->bounding_box.width = 0;
		seg->bounding_box.height = 0;
	}

	seg->model_box_2d = seg->bounding_box;

	auto cloud = std::make_shared<pcl::PointCloud<visual_input::PointT>>();
	for (int i = 0; i < keypoints.cols(); i++)
	{
		const auto& k = keypoints.col(i);
		visual_input::PointT p;
		p.x = k.x();
		p.y = k.y();
		p.z = k.z();

		cloud->push_back(p);
	}

	// add 10% outliers that are removed when constructing img_segment_3d
	visual_input::PointT p;
	cloud->push_back(p);
	cloud->push_back(p);


	seg->prop_3d = std::make_shared<img_segment_3d>(input, *seg, cloud);
	seg->prop_3d->centroid = pose.get_centroid();

	auto centroid_2d = project_into_image(seg->prop_3d->centroid);
	seg->palm_center_2d = cv::Point2i(centroid_2d.x(), centroid_2d.y());

	hand_pose_particle_instance particle(pose, nullptr, timestamp, pose.get_key_points(), std::vector<float>(21, 0.f));
	particle.error = 0;
	particle.hand_certainty = 1;
	seg->particles[pose.right_hand] = std::make_shared<hand_pose_particle_instance>(particle);

	return seg;
}

}

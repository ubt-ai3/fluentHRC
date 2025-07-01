#define _CRT_SECURE_NO_WARNINGS

#include "hand_evaluation.hpp"

#include <set>

#include <pcl/kdtree/kdtree_flann.h>

#include "hand_pose_estimation/hand_tracker_enact.hpp"

hand_evaluator::hand_evaluator(tracker& opti_track, 
                               const hand_pose_estimation::classifier_set& gesture_classifiers,
                               std::string output_path,
                               float certainty_threshold,
                               float distance_threshold)
	: 
	opti_track(opti_track),
	gesture_classifiers(gesture_classifiers),
	output(output_path, std::fstream::out),
	mapping({
		{hand_type::COLOR_LEFT, 0},
		{hand_type::COLOR_RIGHT, 0},
		{hand_type::TAN_LEFT, 0},
		{hand_type::TAN_RIGHT, 0},
	}),
	certainty_threshold(certainty_threshold),
	distance_threshold(distance_threshold),
	latest_time(0),
	mapping_start({
		{hand_type::COLOR_LEFT, 0.f},
		{hand_type::COLOR_RIGHT, 0.f},
		{hand_type::TAN_LEFT, 0.f},
		{hand_type::TAN_RIGHT, 0.f},
	})
{
}

hand_evaluator::~hand_evaluator()
{
	close();
	output.close();
}

void hand_evaluator::update(const hand_pose_estimation::visual_input::ConstPtr& input,
	float time)
{
	// mapping holds the association of ground truth data object and tracked object

	std::set<size_t> found_hands;
	std::set<hand_type> updated_mappings;
	pcl::KdTreeFLANN<PointT> kd_tree;
	kd_tree.setInputCloud(input->cloud);
	
	kd_tree.setMinPts(1);
	std::vector<int> closest_points(1);
	std::vector<float> distances(1);



	//for (auto& hand : hands)
	//{ 
	//	if (hand->certainty_score < certainty_threshold)
	//		continue;

	//	bool found_mapping = false;
	//	for (auto& entry : mapping)
	//	{
	//		size_t hand_id = (size_t) & *hand;
	//		if (hand_id == entry.second) // tracked object still exists
	//		{
	//			updated_mappings.emplace(entry.first);

	//			// use measured point closest to extrapolated opti track position
	//			Eigen::Vector3f opti_track_position = opti_track.get_position_3d(entry.first, time);

	//			if (isnan(opti_track_position(0)))
	//			{
	//				if (mapping_start[entry.first] < latest_time)
	//				{
	//					output << "duration," << time << "," << (latest_time - mapping_start[entry.first]) << std::endl;
	//					mapping_start[entry.first] = time;
	//					entry.second = 0;
	//				}
	//				break;
	//			}

	//			PointT pos;
	//			pos.getArray3fMap() = opti_track_position;
	//			if(kd_tree.nearestKSearch(pos, 1, closest_points, distances))
	//				opti_track_position = cloud->at(closest_points[0]).getArray3fMap();



	//			float distance = (opti_track_position - 
	//				hand->observation_history.back()->palm_center_3d).norm();
	//			
	//			if (distance < distance_threshold) {
	//				output << "distance," << time << "," << distance << std::endl;
	//				found_hands.emplace(hand_id);
	//			}
	//			else // tracked object too far from ground truth, delete association
	//			{
	//				output << "duration," << time << "," << (latest_time - mapping_start[entry.first]) << std::endl;
	//				mapping_start[entry.first] = time;
	//				entry.second = 0;
	//			}


	//			break;
	//		}
	//	}

	//}

	//// delete associations for all no longer existing objects
	//for (auto& entry : mapping)
	//{
	//	if (updated_mappings.find(entry.first) == updated_mappings.end())
	//	{
	//		if (entry.second)
	//		{
	//			output << "duration," << time << "," << (latest_time - mapping_start[entry.first]) << std::endl;
	//			mapping_start[entry.first] = time;
	//		}
	//		entry.second = 0;
	//	}
	//}

	//// try to find new association
	//for (auto& entry : mapping)
	//{
	//	if (!entry.second) {
	//		Eigen::Vector3f opti_track_position = opti_track.get_position_3d(entry.first, time);
	//		if (isnan(opti_track_position(0)))
	//		{
	//			mapping_start[entry.first] = time;
	//			continue;
	//		}

	//		PointT pos;
	//		pos.getArray3fMap() = opti_track_position;
	//		if (kd_tree.nearestKSearch(pos, 1, closest_points, distances))
	//			opti_track_position = cloud->at(closest_points[0]).getArray3fMap();

	//		float min_distance = std::numeric_limits<float>::infinity();
	//		size_t closest_hand = 0;

	//		for (auto& hand : hands)
	//		{
	//			if (hand->certainty_score < certainty_threshold)
	//				continue;

	//			size_t hand_id = (size_t) & *hand;
	//			float distance = (opti_track_position -
	//				hand->observation_history.back()->palm_center_3d).norm();
	//			
	//			if (found_hands.find(hand_id) == found_hands.end() && 
	//				distance < distance_threshold &&
	//				distance < min_distance)
	//			{
	//				min_distance = distance;
	//				closest_hand = hand_id;
	//			}
	//		}

	//		if (closest_hand)
	//		{
	//			if(mapping_start[entry.first] < latest_time)
	//				output << "untracked duration," << time << "," << (latest_time - mapping_start[entry.first])  << std::endl;

	//			mapping_start[entry.first] = time;
	//			mapping[entry.first] = closest_hand;
	//			output << "distance," << time << "," << min_distance << std::endl;
	//			found_hands.emplace(closest_hand);
	//		}
	//	}
	//}

	//// determine other objects
	//for (auto& hand : hands)
	//{
	//	if (hand->certainty_score < certainty_threshold)
	//		continue;

	//	for (const auto& classifier : gesture_classifiers)
	//	{

	//		float certainty = classifier->classify(*hand).certainty_score;
	//		if (certainty > 0.1f)
	//		{
	//			output << classifier->get_object_prototype()->get_name() << "," << time << "," << certainty << std::endl ;
	//		}
	//	}

	//	size_t hand_id = (size_t) & *hand;
	//	if (found_hands.find(hand_id) == found_hands.end())
	//	{
	//		float min_distance = std::numeric_limits<float>::infinity();
	//		for (auto& entry : mapping)
	//		{
	//			Eigen::Vector3f opti_track_position = opti_track.get_position_3d(entry.first, time);
	//			if (isnan(opti_track_position(0)))
	//			{
	//				continue;
	//			}

	//			PointT pos;
	//			pos.getArray3fMap() = opti_track_position;
	//			if (kd_tree.nearestKSearch(pos, 1, closest_points, distances))
	//				opti_track_position = cloud->at(closest_points[0]).getArray3fMap();

	//			float distance = (opti_track_position -
	//				hand->observation_history.back()->palm_center_3d).norm();

	//			min_distance = std::min(distance, min_distance);

	//		}
	//		output << "zombi," << time  << ", " << min_distance << std::endl;
	//	}
	//}

	latest_time = time;
}

void hand_evaluator::close() noexcept
{
	for (const auto& entry : mapping)
	{
		if(entry.second)
		try {
			output << "duration," << latest_time << "," << (latest_time - mapping_start[entry.first]) << std::endl;
		}
		catch (...) {}
	}

}



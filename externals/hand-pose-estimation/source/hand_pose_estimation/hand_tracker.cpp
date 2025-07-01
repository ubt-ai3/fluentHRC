#include "hand_tracker.hpp"

namespace hand_pose_estimation
{

hand_tracker::hand_tracker(std::chrono::duration<float> purge_duration,
	int max_threads, 
	bool spawn_hands,
	Eigen::Vector3f back_palm_orientation, 
	Eigen::Hyperplane<float, 3> background)
	:
	skin_detector(),
	bb_tracker(0.2f, purge_duration,2.f, spawn_hands),
	gd_scheduler(gradient_decent_optimization_common(
		{
			std::make_shared < quality_acceleration>(0.1),
			std::make_shared<quality_2d_key_points>(0.6),
			//std::make_shared<quality_3d_key_points>(0.1),
			//std::make_shared < quality_key_points_below_surface>(0.4),
			std::make_shared < quality_key_points_close_to_surface>(0.2),
			std::make_shared < quality_boundary_surface>(0.1,background),
			std::make_shared<quality_centroid>(0.05)
		},
		back_palm_orientation), std::max(0, max_threads - 1)),
	terminate_flag(false)
{
	bb_tracker.set_background_plane(background);
	
	if (max_threads >= 1)
	{
		internal_thread = std::thread([&]()
			{
				while (true)
				{
					try {
						{
							std::unique_lock<std::mutex> lock(update_mutex);

							if (terminate_flag)
								return;

							new_input_condition.wait(lock, [&] {return input || terminate_flag; });

							if (terminate_flag)
								return;
						}

						process_input();
					}
					catch (const std::exception& e)
					{
						std::cout << e.what() << std::endl;
					}

				}
			});
	}
}

hand_tracker::~hand_tracker()
{
	terminate_flag = true;

	new_input_condition.notify_all();
	new_hands_condition.notify_all();
	batch_completion_condition.notify_all();
	
	if (internal_thread.joinable())
		internal_thread.join();
}

void hand_tracker::update(const visual_input::ConstPtr& input)
{
	{
		std::lock_guard<std::mutex> lock(update_mutex);
		//if (!input)
		//	std::cout << "frame skipped" << std::endl;
		this->input = input;
	}

	if (internal_thread.joinable())
		new_input_condition.notify_all();
	else
		process_input();
}

std::vector<hand_instance::Ptr> hand_tracker::get_hands() const
{

	
	std::lock_guard<std::mutex> lock(update_mutex);
	return std::vector<hand_instance::Ptr>(hands.begin(), hands.end());

}

void hand_tracker::wait_for_batch_completed()
{
	if (terminate_flag)
		return;
	
	std::unique_lock<std::mutex> lock(update_mutex);
	batch_completion_condition.wait(lock);
}

const hand_kinematic_parameters& hand_tracker::get_hand_kinematic_parameters() const
{
	return gd_scheduler.get_hand_kinematic_parameters();
}

float hand_tracker::get_hand_certainty_threshold() const
{
	return gd_scheduler.get_hand_certainty_threshold();
}

void hand_tracker::set_background(const Eigen::Hyperplane<float, 3>& plane)
{
	bb_tracker.set_background_plane(plane);
	gd_scheduler.set_background(plane);
}

void hand_tracker::set_back_palm_orientation(const Eigen::Vector3f& normal)
{
	gd_scheduler.set_back_palm_orientation(normal);
}

void hand_tracker::show_skin_regions()
{
	skin_detector.show_skin_regions();
}

void hand_tracker::process_input()
{
	if (!input)
		return;
	
	visual_input::ConstPtr local_input;
	{
		std::lock_guard<std::mutex> lock(update_mutex);
		local_input = input;
		input = nullptr;
	}

	std::vector<hand_instance::Ptr> prev_candidates = bb_tracker.get_hands();
	std::vector<hand_instance::Ptr> prev_hands;
	float hand_probability_threshold = gd_scheduler.get_hand_pose_parameters().hand_probability_threshold;
	for (auto& hand : prev_candidates)
		if (hand->certainty_score > hand_probability_threshold)
			prev_hands.push_back(hand);
	
	auto segments = skin_detector.detect_segments(*local_input, prev_hands);
	bb_tracker.update(*local_input, segments);
	auto current_hands = bb_tracker.get_hands();

	{
		std::lock_guard<std::mutex> lock(update_mutex);
		for (const auto& hand : current_hands)
			if (hands.find(hand) == hands.end())
				new_hands.emplace_back(hand);

		hands = std::set(current_hands.begin(), current_hands.end());

		for (auto iter = new_hands.begin(); iter != new_hands.end();)
			if (hands.find(*iter) == hands.end())
				iter = new_hands.erase(iter);
			else
				++iter;
	}

	if (!new_hands.empty())
		new_hands_condition.notify_all();

	gd_scheduler.update(local_input, current_hands);

	batch_completion_condition.notify_all();
}
}

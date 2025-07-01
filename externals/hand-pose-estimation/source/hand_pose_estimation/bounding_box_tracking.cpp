#include "bounding_box_tracking.hpp"

#include <set>

#include <Eigen/Core>

#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include "classification.hpp"


namespace hand_pose_estimation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: bounding_box_tracker
//
//
/////////////////////////////////////////////////////////////

bounding_box_tracker::bounding_box_tracker(float certainty_threshold,
	std::chrono::duration<float> purge_duration,
	float smoothing_factor,
	bool spawn_hands)
	:
	certainty_threshold(certainty_threshold),
	purge_duration(purge_duration),
	smoothing_factor(smoothing_factor),
	timestamp(std::chrono::duration<float>(0)),
	spawn_hands(spawn_hands),
	helper_terminate(false),
	helper_sleep(false)
{
	helper = std::thread([this]()
		{
			while (!helper_terminate)
			{
				img_segment::Ptr seg = nullptr;
				{
					std::lock_guard<std::mutex> lock(helper_mutex);
					if (!helper_input.empty())
					{
						seg = helper_input.back();
						helper_input.pop_back();
					}
				}

				if (!seg)
				{
					std::unique_lock<std::mutex> lock(helper_mutex);
					helper_sleep = true;
					helper_wake_up.notify_all();
					helper_wake_up.wait(lock);
					helper_sleep = false;
					continue;
				}


				try {
					seg->compute_properties_3d(*helper_visual_input, plane);

					std::lock_guard<std::mutex> lock(helper_mutex);
					helper_output.push_back(seg);
				}
				catch (const std::exception&)
				{
					//ignore segment without cloud
				}

			}
		});
}

bounding_box_tracker::~bounding_box_tracker()
{
	helper_terminate = true;
	helper_wake_up.notify_all();
	if (helper.joinable())
		helper.join();
}

void bounding_box_tracker::update(const visual_input& input,
	const std::vector<img_segment::Ptr>& objects)
{
	if (input.timestamp_seconds < timestamp)
	{
		reset();
		this->timestamp = input.timestamp_seconds;
	}
	else if (input.timestamp_seconds == timestamp)
		return;
	else
	{
		timestamp = input.timestamp_seconds;
	}

	if (objects.empty())
		return;

	std::vector<img_segment::Ptr> candidates;
	if (input.has_cloud()) {
		img_segment::Ptr seg;

		std::unique_lock<std::mutex> lock(helper_mutex);
		helper_input = objects;
		helper_output.clear();
		helper_output.reserve(objects.size());
		helper_visual_input = &input;
		seg = helper_input.back();
		helper_input.pop_back();

		lock.unlock();

		if (helper_input.size() > 1)
			helper_wake_up.notify_all();

		while (seg)
		{
			try {
				seg->compute_properties_3d(input, plane);

				if (seg->prop_3d) {
					lock.lock();
					helper_output.push_back(seg);
				}
			}
			catch (const std::exception&)
			{
				//ignore segment without cloud
				lock.lock();
			}

			if (helper_input.empty())
				seg = nullptr;
			else
			{
				seg = helper_input.back();
				helper_input.pop_back();
			}

			lock.unlock();
		}

		lock.lock();
		if (!helper_sleep)
			helper_wake_up.wait(lock);
		candidates = helper_output;
		lock.unlock();
	}
	else
		candidates = objects;
	
	typedef struct _match
	{
		float sim_position;
		float sim_shape;
		hand_instance::Ptr hand;
		img_segment::Ptr seg;
	} match;

	std::priority_queue< match, std::vector<match>, auto(*)(const match&, const match&)->bool >  queue([](const match& lhs, const match& rhs) -> bool {
		return 0.8f * lhs.sim_position + 0.2f * lhs.sim_shape < 0.8f * rhs.sim_position + 0.2f * rhs.sim_shape;
		});

	int i = 0;
	std::set<hand_instance::Ptr> definitive_hands;
	for (hand_instance::Ptr& hand : hands)
	{
		if (hand->certainty_score > 0.75)
			definitive_hands.emplace(hand);

		const img_segment& ref_seg = *hand->observation_history.back();
		for (const img_segment::Ptr& seg : candidates)
		{
			queue.emplace(match{
				compare_position(input, *hand, *seg),
				0.f,//compare_shape(*hand, *seg),
				hand,
				seg
				});
		}

		i++;
	}


	// match segments and hands
	// in case the new segment is 50% larger, consider it as a fusion of segements
	std::set<hand_instance::Ptr> matched_hands;
	std::map<img_segment::Ptr, hand_instance::Ptr> matched_segments;
	std::map<img_segment::Ptr, std::vector<hand_instance::Ptr>> fused_segments;

	float similarity = 1.f;
	while (queue.size() && (similarity > certainty_threshold || !definitive_hands.empty()))
	{
		const match m = queue.top(); queue.pop();
		similarity = 1.f * m.sim_position + 0.f * m.sim_shape;

		if (similarity < certainty_threshold)
			break;

		if (matched_hands.find(m.hand) != matched_hands.end())
			continue;

		bool matched_segment = matched_segments.find(m.seg) != matched_segments.end();
		bool definitive_hand = m.hand->certainty_score > 0.75f;


		if (definitive_hand && matched_segment ||
			!matched_segment && m.seg->model_box_2d.area() > 1.5 * m.hand->observation_history.back()->model_box_2d.area() &&
			m.hand->certainty_score > 0.05)
		{
			auto iter = fused_segments.find(m.seg);
			if (iter == fused_segments.end())
				iter = fused_segments.emplace(m.seg, std::vector<hand_instance::Ptr>()).first;

			iter->second.push_back(m.hand);
			matched_hands.emplace(m.hand);

			definitive_hands.erase(m.hand);
		}
		else if (!matched_segment)
		{
			m.seg->timestamp = timestamp;
			matched_hands.emplace(m.hand);
			matched_segments.emplace(m.seg, m.hand);
			definitive_hands.erase(m.hand);
		}

	}

	// a fused segment candidate might got a match, correct it
	for (auto iter = matched_segments.begin(); iter != matched_segments.end();)
	{
		auto fused_iter = fused_segments.find(iter->first);

		if (fused_iter == fused_segments.end())
			++iter;
		else
		{
			if (iter->second->certainty_score > 0.05)
				fused_iter->second.push_back(iter->second);

			iter = matched_segments.erase(iter);
		}
	}

	// try to subsegment fused segments
	for (auto& entry : fused_segments)
	{
		if (entry.second.size() == 1)
		{
			matched_segments.emplace(entry.first, entry.second.front());
			matched_hands.emplace(entry.second.front());
			continue;
		}

		//only maintain hand candidates with high certainty
		std::sort(entry.second.begin(), entry.second.end(), [](const hand_instance::Ptr& lhs, const hand_instance::Ptr rhs)
			{
				return lhs->certainty_score > rhs->certainty_score;
			});

		if (entry.second[1]->certainty_score < 0.4f)
		{
			matched_segments.emplace(entry.first, entry.second.front());
			matched_hands.emplace(entry.second.front());
			continue;
		}

		std::vector<img_segment::Ptr> sub_segs(1, entry.first);

		// try depth based segmentation first
		if (input.has_cloud())
		{
			entry.first->compute_properties_3d(input);

			/*if (entry.first->prop_3d)
				sub_segs = subsegment(input, entry.first, entry.second.size());*/
		}

		typedef struct _match
		{
			float distance;
			hand_instance::Ptr hand;
			img_segment::Ptr seg;
		} match;

		std::priority_queue< match, std::vector<match>, auto(*)(const match&, const match&)->bool >  queue([](const match& lhs, const match& rhs) -> bool {
			return lhs.distance < rhs.distance;
			});

		int i = 0;
		std::sort(entry.second.begin(), entry.second.end(), [](const hand_instance::Ptr& lhs, const hand_instance::Ptr& rhs) {
			return lhs->certainty_score > rhs->certainty_score;
			});
		entry.second.resize(std::min((size_t)3, entry.second.size()));
		for (const hand_instance::Ptr& hand : entry.second)
		{
			const img_segment& ref_seg = *hand->observation_history.back();
			for (int j = 0; j < std::min(sub_segs.size(), entry.second.size()); j++)
			{
				auto get_center = [](const cv::Rect2i& box) {
					return 0.5 * box.tl() + 0.5 * box.br();
				};

				float distance = cv::norm(get_center(extrapolate_pose(input, *hand)) - get_center(sub_segs[j]->model_box_2d));

				queue.emplace(match{
					distance,
					hand,
					sub_segs[j]
					});
			}

			if (++i > 3)
				break;
		}

		// match the hand closest to each subsegment
		std::set<hand_instance::Ptr> remaining_hands(entry.second.begin(), entry.second.end());
		while (queue.size())
		{
			const match m = queue.top(); queue.pop();

			if (matched_hands.find(m.hand) == matched_hands.end() &&
				matched_segments.find(m.seg) == matched_segments.end())
			{
				m.seg->timestamp = timestamp;
				matched_hands.emplace(m.hand);
				matched_segments.emplace(m.seg, m.hand);
				remaining_hands.erase(m.hand);
			}
		}

		// deal with the remaining hands - no sufficient subsegmentation was possible
		// move the bounding box inside the segment
		for (const hand_instance::Ptr& hand : remaining_hands)
		{
			const cv::Rect2i seg_box = entry.first->model_box_2d;
			cv::Rect2i hand_box = extrapolate_pose(input, *hand);

			int width = std::min(seg_box.width, std::max(hand_box.width, hand->observation_history.back()->bounding_box.width));
			int height = std::min(seg_box.height, std::max(hand_box.height, hand->observation_history.back()->bounding_box.height));

			int x = std::max(seg_box.x, hand_box.x);
			if (x + width > seg_box.x + seg_box.width)
				x = seg_box.x + seg_box.width - width;

			int y = std::max(seg_box.y, hand_box.y);
			if (y + height > seg_box.y + seg_box.height)
				y = seg_box.y + seg_box.height - height;


			auto cluster_seg = std::make_shared<img_segment>();
			cluster_seg->bounding_box = cv::Rect2i(x, y, width, height);
			cluster_seg->model_box_2d = cluster_seg->bounding_box;

			const cv::Rect2i& box = cluster_seg->model_box_2d;
			entry.first->mask(cv::Rect2i(box.x - seg_box.x, box.y - seg_box.y, box.width, box.height)).copyTo(cluster_seg->mask);

			// compute contours
			std::vector<cv::Vec4i> hierarchy;
			std::vector<std::vector<cv::Point>> seg_contours;
			cv::findContours(cluster_seg->mask, seg_contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
			int max_countour_index, max_countour_size = 0;

			if (seg_contours.empty())
				continue;

			for (int i = 0; i < seg_contours.size(); ++i)
			{
				if (hierarchy[i][3] < 0 && max_countour_size < seg_contours.at(i).size())
				{
					max_countour_index = i;
					max_countour_size = seg_contours.at(i).size();
				}
			}

			cv::approxPolyDP(seg_contours[max_countour_index], cluster_seg->contour, smoothing_factor, true);
			cluster_seg->timestamp = timestamp;

			cv::Moments moments(cv::moments(cluster_seg->mask, true));
			cluster_seg->palm_center_2d = cv::Point2i(moments.m10 / moments.m00 + x, moments.m01 / moments.m00 + y);

			if (input.has_cloud())
				try
			{
				cluster_seg->compute_properties_3d(input, plane);
			}
			catch (const std::exception&)
			{
				continue;
			}

			matched_segments.emplace(cluster_seg, hand);
			matched_hands.emplace(hand);
		}
	}

	//cv::Mat drawing = input.img.clone();
	//for (const auto& entry : matched_segments)
	//{
	//	cv::rectangle(drawing, entry.first->bounding_box, cv::Scalar(0,0,255), 2);
	//}
	//cv::imshow("Boxes", drawing);
	//cv::waitKey(1);

	//std::cout << matched_segments.size() << " matches" << std::endl;

	std::set<hand_instance::Ptr> updated_hands;

	const float max_hand_length = hand_kin_params.thickness +
		hand_kin_params.max_bone_scaling * hand_pose_18DoF(hand_kin_params).get_tip(finger_type::MIDDLE).norm();

	for (const img_segment::Ptr& seg : candidates)
	{
		if (matched_segments.find(seg) == matched_segments.end() &&
			fused_segments.find(seg) == fused_segments.end())
		{

			if (input.has_cloud())
			{
				try {
					seg->compute_properties_3d(input);
					const Eigen::Vector3f diagonal = seg->prop_3d->bounding_box.diagonal();

					if (diagonal.x() > max_hand_length || diagonal.y() > max_hand_length)
						continue;

				}
				catch (const std::exception&)
				{
				} // segment contains no points
			}

			// check whether there is a valid hand candidate in a fused segment before we create a new one
			std::vector<std::pair<float, hand_instance::Ptr>> fused_candidates;
			for (auto& entry : fused_segments)
			{
				if (entry.second.size() <= 1)
					continue;

				for (auto& hand : entry.second)
				{
					float sim = compare_position(input, *hand, *seg);

					if (sim > 0.5f * certainty_threshold && updated_hands.find(hand) == updated_hands.end())
						fused_candidates.emplace_back(sim, hand);
				}

			}

			if (!fused_candidates.empty())
			{
				std::sort(fused_candidates.begin(), fused_candidates.end(), [](const std::pair<float, hand_instance::Ptr>& lhs, const std::pair<float, hand_instance::Ptr>& rhs)
					{
						return lhs.first > rhs.first;
					});

				{
					std::lock_guard<std::mutex> lock(fused_candidates.front().second->update_mutex);
					if (fused_candidates.front().second->observation_history.back()->timestamp >= timestamp)
						continue;

					fused_candidates.front().second->observation_history.push_back(seg);
					updated_hands.emplace(fused_candidates.front().second);

				}

				continue;
			}

			if(spawn_hands)
				hands.push_back(hand_instance::Ptr(new hand_instance(seg)));

		}
	}

	for (const auto& entry : matched_segments)
	{
		if (updated_hands.find(entry.second) != updated_hands.end())
			continue;
		
		std::lock_guard<std::mutex> lock(entry.second->update_mutex);

		if (entry.second->observation_history.back()->timestamp >= timestamp)
			continue;

		entry.second->observation_history.push_back(entry.first);
		updated_hands.emplace(entry.second);

		//if (!entry.second->observation_history.back()->prop_3d)
		//	throw std::runtime_error("!seg->prop_3d");
	}

	for (auto iter = hands.begin(); iter != hands.end(); ) // increment in body
	{
		std::lock_guard<std::mutex> lock((*iter)->update_mutex);
		if (spawn_hands && (*iter)->observation_history.front()->timestamp + purge_duration < timestamp)
			(*iter)->observation_history.pop_front();

		if (spawn_hands &&
			(*iter)->observation_history.size() >= 2 &&
			!(*iter)->observation_history.front()->particles[0] &&
			!(*iter)->observation_history.front()->particles[1])
			(*iter)->observation_history.pop_front();

		if ((*iter)->observation_history.size() > 4)
			(*iter)->observation_history.pop_front();

		if (!(*iter)->observation_history.size())
			iter = hands.erase(iter);
		else if ((*iter)->observation_history.size() == 1 && (*iter)->observation_history.back()->timestamp != timestamp)
			iter = hands.erase(iter);
		else
			++iter;
	}


}

void bounding_box_tracker::add_hand(const std::shared_ptr<hand_instance>& hand)
{
	hands.emplace_back(hand);
}

void bounding_box_tracker::reset()
{
	hands.clear();
	timestamp = std::chrono::duration<float>(0);
}

std::vector<std::shared_ptr<hand_instance>> bounding_box_tracker::get_hands() const
{
	return std::vector<std::shared_ptr<hand_instance>>(hands.begin(), hands.end());
}

std::chrono::duration<float> bounding_box_tracker::get_latest_timestamp() const
{
	return timestamp;
}

void bounding_box_tracker::set_background_plane(Eigen::Hyperplane<float, 3> plane)
{
	this->plane = std::move(plane);
}

std::vector<img_segment::Ptr> bounding_box_tracker::subsegment(const visual_input& input,
	const img_segment::Ptr seg,
	int max_clusters)
{
	auto roi_cloud = seg->prop_3d->cloud;

	pcl::IndicesClustersPtr clusters(new pcl::IndicesClusters);
	pcl::search::KdTree<visual_input::PointT>::Ptr search_tree(new pcl::search::KdTree<visual_input::PointT>);

	pcl::ConditionalEuclideanClustering<visual_input::PointT> cec(false);
	cec.setInputCloud(roi_cloud);
	cec.setConditionFunction([&](const visual_input::PointT& p1, const visual_input::PointT& p2, float squared_distance)
		{
			return std::abs(p1.z - p2.z) < 0.01;
		});
	cec.setClusterTolerance(0.02);
	cec.setMinClusterSize(roi_cloud->size() / max_clusters / 3);
	cec.setMaxClusterSize(2 * roi_cloud->size() / max_clusters);
	cec.segment(*clusters);

	if (clusters->size() <= 1)
		return std::vector<img_segment::Ptr>(1, seg);

	std::sort(clusters->begin(), clusters->end(), [](const pcl::PointIndices& lhs, const pcl::PointIndices& rhs) {
		return lhs.indices.size() > rhs.indices.size();
		});

	//cv::imshow("mask", seg->mask);

	std::vector<img_segment::Ptr> result;
	for (const auto& cluster : *clusters)
	{
		pcl::ExtractIndices<visual_input::PointT> extractor;
		extractor.setInputCloud(roi_cloud);
		extractor.setIndices(pcl::make_shared<pcl::PointIndices>(cluster));

		auto cluster_cloud = pcl::make_shared<pcl::PointCloud<visual_input::PointT>>();
		extractor.filter(*cluster_cloud);

		std::vector<cv::Point2i> points;
		points.reserve(cluster.indices.size());
		for (const auto& p : *cluster_cloud)
		{
			points.push_back(input.to_img_coordinates(p.getVector3fMap()));
		}

		auto cluster_seg = std::make_shared<img_segment>();
		auto cluster_box = cv::boundingRect(points);
		cv::convexHull(points, cluster_seg->contour);
		cluster_seg->timestamp = seg->timestamp;

		cv::Point2i br, tl;

		tl.x = std::max(cluster_box.tl().x, seg->model_box_2d.tl().x);
		tl.y = std::max(cluster_box.tl().y, seg->model_box_2d.tl().y);
		br.x = std::min(cluster_box.br().x, seg->model_box_2d.br().x);
		br.y = std::min(cluster_box.br().y, seg->model_box_2d.br().y);

		cv::Rect2i relative_box(tl - seg->model_box_2d.tl(), br - seg->model_box_2d.tl());
		cluster_seg->bounding_box = cv::Rect2i(tl, br);
		cluster_seg->model_box_2d = cluster_seg->bounding_box;

		cluster_seg->mask = cv::Mat(relative_box.height, relative_box.width, CV_8UC1, cv::Scalar::all(0));
		cv::drawContours(cluster_seg->mask, std::vector<std::vector<cv::Point2i>>({ cluster_seg->contour }), 0, cv::Scalar(255), cv::FILLED, cv::LINE_8, cv::noArray(), 0, -1 * tl);
		cv::bitwise_and(cluster_seg->mask, seg->mask(relative_box), cluster_seg->mask);

		cv::Moments moments(cv::moments(points));
		cluster_seg->palm_center_2d = cv::Point2i(moments.m10 / moments.m00, moments.m01 / moments.m00);

		cluster_seg->compute_properties_3d(input, cluster_cloud);

		//cv::imshow("mask" + std::to_string(result.size()), cluster_seg->mask);
		//cv::waitKey(1);

		result.push_back(cluster_seg);
	}

	return result;
}

cv::Rect2i bounding_box_tracker::extrapolate_pose(const visual_input& input, const hand_instance& hand) const
{
	std::vector<Eigen::Vector2f> extrapolated_pos;


	if (hand.observation_history.size() == 1 || hand.poses.empty())
	{
		return hand.observation_history.back()->model_box_2d;
	}
	else
	{
		bool right_hand = hand.poses.back().pose.right_hand;

		auto iter = hand.observation_history.rbegin();
		const auto& particle = (*iter)->particles[right_hand];
		++iter;
		const auto& prev_particle = (*iter)->particles[right_hand];

		if (!prev_particle || !particle)
			return hand.observation_history.back()->model_box_2d;

		hand_pose_18DoF::Vector15f last = particle->pose.get_parameters();
		float elapsed = std::chrono::duration<float>(timestamp - particle->time_seconds).count();
		float prev_elapsed = std::chrono::duration<float>(particle->time_seconds - prev_particle->time_seconds).count();

		if (elapsed < 0.002f)
			elapsed = 1 / 60.f;

		if (elapsed > 1.f)
			elapsed = 1.f;

		if (prev_elapsed < 0.002f)
			prev_elapsed = 1 / 60.f;

		hand_pose_18DoF::Vector15f parameters = last +
			elapsed / prev_elapsed * (last - prev_particle->pose.get_parameters());

		Eigen::AngleAxisf rotation_dir;
		rotation_dir = particle->rotation * prev_particle->rotation.inverse();
		rotation_dir.angle() *= 1 + elapsed / prev_elapsed;
		Eigen::Quaternionf rotation = prev_particle->rotation * rotation_dir;

		return hand_pose_particle_instance(
			hand_pose_18DoF(particle->pose.hand_kinematic_params, parameters, rotation),
			nullptr,
			timestamp).get_box(input);

	}
}

float bounding_box_tracker::compare_position(const visual_input& input, const hand_instance& hand, const img_segment& seg) const
{
	float sim = 0.f;

	if (seg.prop_3d)
	{
		auto weight_intersection = [&](const Eigen::AlignedBox3f& reference)
		{
			Eigen::AlignedBox3f intersect = seg.prop_3d->bounding_box.intersection(reference);
			if (intersect.isEmpty())
				return 1.f;

			return 1.f - 0.76f * intersect.volume() / reference.volume();
		};

		float dist = (seg.prop_3d->centroid - hand.observation_history.back()->prop_3d->centroid).norm();

		sim = bell_curve(dist * weight_intersection(hand.observation_history.back()->prop_3d->bounding_box), hand.observation_history.back()->prop_3d->bounding_box.diagonal().norm() / 3.f);
		if (hand.observation_history.size() >= 2)
		{
			auto& prev_seg = *(++hand.observation_history.rbegin());

			/*if (!prev_seg->model_box_3d.isEmpty() && prev_seg->model_box_3d.intersects(seg.prop_3d->bounding_box))
				return 0.f;*/
			
			Eigen::Vector3f translation = hand.observation_history.back()->prop_3d->centroid - prev_seg->prop_3d->centroid;
			dist = (hand.observation_history.back()->prop_3d->centroid + translation - seg.prop_3d->centroid).norm();

			float weight = weight_intersection(Eigen::AlignedBox3f(hand.observation_history.back()->prop_3d->bounding_box).translate(translation));

			sim = std::max(sim, bell_curve(dist * weight,
				hand.observation_history.back()->prop_3d->bounding_box.diagonal().norm()));
		}

		return sim;
	}
	else
	{
		cv::Rect2i ref_box = extrapolate_pose(input, hand);
		const cv::Rect2i& seg_box = seg.model_box_2d;

		for (const cv::Rect2i ref_box : {extrapolate_pose(input, hand), hand.observation_history.back()->model_box_2d})
		{
			float diff_tl = cv::norm(ref_box.tl() - seg_box.tl());
			if (seg_box.contains(ref_box.tl()))
				diff_tl *= 0.5f;

			float diff_br = cv::norm(ref_box.br() - seg_box.br());
			if (seg_box.contains(ref_box.br()))
				diff_br *= 0.5f;

			sim = std::max(sim,
				bell_curve(diff_tl, std::max(ref_box.width, ref_box.height))
				* bell_curve(diff_br, std::max(ref_box.width, ref_box.height)));
		}

		return std::powf(sim, 0.5);
	}


}




/**
* Returns the similarity of the last snapshot of hand and seg.
*/

inline float bounding_box_tracker::compare_shape(const hand_instance& hand, const img_segment& seg) const
{
	const img_segment& hand_seg = *hand.observation_history.back();

	int height = std::max(hand_seg.model_box_2d.height, seg.model_box_2d.height);
	int width = std::max(hand_seg.model_box_2d.width, seg.model_box_2d.width);
	cv::Mat canvas1(height, width, CV_8UC1);

	canvas1 = cv::Scalar(0);
	cv::Point offset(-hand_seg.model_box_2d.x, -hand_seg.model_box_2d.y);
	cv::fillPoly(canvas1, std::vector<std::vector<cv::Point>>({ hand_seg.contour }), cv::Scalar(255), 0, 0, offset);
	//	cv::imshow("Canvas1", canvas1);

	cv::Mat canvas2(height, width, CV_8UC1);
	canvas2 = cv::Scalar(0);
	offset = cv::Point(-seg.model_box_2d.x, -seg.model_box_2d.y);
	cv::fillPoly(canvas2, std::vector<std::vector<cv::Point>>({ seg.contour }), cv::Scalar(255), 0, 0, offset);
	//	cv::imshow("Canvas2", canvas2);

	std::vector<std::vector<cv::Point>> intersection;
	cv::bitwise_and(canvas1, canvas2, canvas1);
	//	cv::imshow("Intersection", canvas1);
	cv::findContours(canvas1, intersection, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	if (!intersection.size() || !intersection[0].size())
		return 0;
	return cv::contourArea(intersection[0]) / std::min(cv::contourArea(hand_seg.contour), cv::contourArea(seg.contour));

	//	return shape_classifier::match(hand.observation_history.back()->hu_moments, seg.hu_moments);
}

float bounding_box_tracker::bell_curve(float x, float stdev) const
{
	return std::expf(-x * x / (2 * stdev * stdev));
}

} /* hand_pose_estimation */
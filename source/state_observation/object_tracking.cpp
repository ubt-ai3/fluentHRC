#include "object_tracking.hpp"

#include <set>


#include <Eigen/Core>

#include  "workspace_objects.hpp"

#include "enact_core/data.hpp"
#include "enact_core/access.hpp"

//#include <app_visualization/module_manager.hpp>
//#include <app_visualization/viewer.cpp>

#include <iostream>



namespace state_observation
{

class XYZRGBA2XY : public pcl::PointRepresentation<object_tracker::PointT>
{
public:
	XYZRGBA2XY()
	{
		this->nr_dimensions_ = 2;
		this->trivial_ = true;
	}

	void copyToFloatArray(const object_tracker::PointT& p, float* out) const
	{
		out[0] = p.x;
		out[1] = p.y;
	}

};


/////////////////////////////////////////////////////////////
//
//
//  Class: object_tracker
//
//
/////////////////////////////////////////////////////////////

object_tracker::object_tracker(enact_core::world_context& world,
							   const pointcloud_preprocessing& pc_prepro,
							   const object_parameters& object_params,
							   float certainty_threshold,
							   std::chrono::duration<float> purge_duration)
	:
	world(world),
	object_params(object_params),
	certainty_threshold(certainty_threshold),
	purge_duration(purge_duration),
	timestamp(std::chrono::duration<float>(0.f)),
	occlusion_detect(occlusion_detector::construct_from(pc_prepro, object_params))
{
}

object_tracker::object_tracker(enact_core::world_context& world,
	occlusion_detector::Ptr occlusion_detect,
	const object_parameters& object_params,
	float certainty_threshold,
	std::chrono::duration<float> purge_duration)
	:
	world(world),
	object_params(object_params),
	certainty_threshold(certainty_threshold),
	purge_duration(purge_duration),
	timestamp(std::chrono::duration<float>(0.f)),
	occlusion_detect(std::move(occlusion_detect))
{}

object_tracker::~object_tracker()
{
	stop_thread();
}

void object_tracker::update(const std::vector<pc_segment::Ptr>& segs)
{
	if (segs.empty())
		return;

	std::chrono::duration<float> timestamp = segs[0]->timestamp;

	if (timestamp == std::chrono::duration<float>(0))
		timestamp = this->timestamp.load() + std::chrono::microseconds(1);

	this->timestamp = timestamp;

	
	schedule([this, segments(segs)]()
	{
		std::chrono::duration<float> timestamp = this->timestamp;
		if (segments[0]->timestamp < timestamp)
			return;

		auto start = std::chrono::high_resolution_clock::now();


		typedef struct _match
		{
			float sim_position;
			float sim_shape;
			float sim_color;
			entity_id obj;
			pc_segment::Ptr seg;
		} match;

		std::priority_queue< match, std::vector<match>, auto(*)(const match&, const match&)->bool >  queue([](const match& lhs, const match& rhs) -> bool {
			return 0.6f * lhs.sim_position + 0.2f * lhs.sim_shape + 0.2f * lhs.sim_color < 0.6f * rhs.sim_position + 0.2f * rhs.sim_shape + 0.2f * rhs.sim_color;
			});

		int i = 0;
		//match every life object against every pc_segment
		for (entity_id& object_id : live_objects)
		{
			enact_core::lock l(world, enact_core::lock_request(object_id, object_instance::aspect_id, enact_core::lock_request::read));
			enact_core::const_access<object_instance_data> access_object(l.at(object_id, object_instance::aspect_id));

			const auto& ref_seg = access_object->payload;
			for (const auto& segment : segments)
			{
				const pc_segment& seg = *segment;
				queue.emplace(match{
					compare_position(access_object->payload, seg),
					compare_size(access_object->payload, seg),
					compare_color(access_object->payload, seg),
					object_id,
					segment
				});
			}

			i++;
		}

		std::set<entity_id> matched_objects;
		std::set<pc_segment::Ptr> matched_segments;

		float similarity = 1.f;
		int identical_objects = 0;

		//update observation_history for live_objects which got matched to a segment
		while (!queue.empty() && similarity > certainty_threshold)
		{
			const match m = queue.top(); queue.pop();
			similarity = 0.6f * m.sim_position + 0.2f * m.sim_shape + 0.2f * m.sim_color;

			if (similarity > certainty_threshold &&
				!matched_objects.contains(m.obj) &&
				!matched_segments.contains(m.seg))
			{
				try
				{
					{
						enact_core::lock l(world, enact_core::lock_request(m.obj, object_instance::aspect_id, enact_core::lock_request::write));
						enact_core::access<object_instance_data> access_object(l.at(m.obj, object_instance::aspect_id));
						object_instance& obj = access_object->payload;

						m.seg->timestamp = timestamp;
	
						if (similarity > 0.95f && obj.observation_history.size() >= object_instance::observation_history_intended_length)
						{
							//TODO do we want to store that a object was matched pretty confidently?
							obj.observation_history.back()->timestamp = timestamp;
							identical_objects++;
							obj.covered = false; // non identical objects are uncovered after classification
						}
						else {
							obj.observation_history.push_back(m.seg);
							//TODO what are we doing here?
							if(obj.get_classified_segment() == obj.observation_history.front() && 
								obj.observation_history.size() > object_instance::observation_history_intended_length)
								obj.observation_history.erase(++obj.observation_history.begin());
							else if (obj.observation_history.size() > object_instance::observation_history_intended_length && 
								obj.observation_history.front()->timestamp + purge_duration < timestamp)
								obj.observation_history.pop_front();
						}
						
	
						
						matched_objects.emplace(m.obj);
						matched_segments.emplace(m.seg);
					}

					(*emitter)(m.obj, enact_priority::operation::UPDATE);

				}
				catch (const std::exception&)
				{ /* skip when object no longer exists */
				}
			}
		}

		//std::cout << "live: " << live_objects.size() << "; matched: " << matched_objects.size() << "; identical: " << identical_objects << std::endl;

		pcl::PointCloud<PointT>::ConstPtr cloud;
		{
			std::lock_guard<std::mutex> lock(clouds_mutex);
			while (!clouds.empty() && std::chrono::microseconds(clouds.front()->header.stamp) < timestamp - std::chrono::duration<float>(0.001))
				clouds.pop();

			if (!clouds.empty())
				cloud = clouds.front();
		}
		//std::cout << "live: " << live_objects.size() << "; matched: " << matched_objects.size() << "; identical: " << identical_objects <<"\n";
		//TODO remove hardcoded value
		if (cloud)
			occlusion_detect->set_reference_cloud(cloud);
		
		//deal with live_objects which didnt got matched
		for (auto iter = live_objects.begin(); iter != live_objects.end();/*increment in body*/)
		{
			if (!matched_objects.contains(*iter))
			{
				entity_id id = *iter;
				bool deletion = false;
				bool occluded = false;
				{
					enact_core::lock l(world, enact_core::lock_request(*iter, object_instance::aspect_id, enact_core::lock_request::write));
					enact_core::access<object_instance_data> access_object(l.at(*iter, object_instance::aspect_id));
					object_instance& obj = access_object->payload;

					bool checked_occlusion = false;
					if (cloud && obj.get_classified_segment())
					{
						// check for occlusion
						checked_occlusion = true;
						auto test_cloud = obj.observation_history.back();
						auto result = occlusion_detect->perform_detection(*test_cloud->points);

						//module_manager::debug_view->add_bounding_box(test_cloud->bounding_box, "occlusion test");
						//module_manager::debug_view->remove_bounding_box("occlusion test");
						if (result == occlusion_detector::COVERED)
						{
							++iter;
							obj.covered = true;
							occluded = true; 
						}
						else if (result == occlusion_detector::DISAPPEARED)
						{
							//std::cout << "disappeared";
							iter = live_objects.erase(iter);
							deletion = true;
						}

					}
					
					if (!occluded && !deletion) {
						// keep classified segments and purge outdated
						if (obj.get_classified_segment() == obj.observation_history.front() &&
							obj.observation_history.size() > object_instance::observation_history_intended_length)
						{
							obj.observation_history.erase(++obj.observation_history.begin());
						}
						else if(obj.observation_history.size() > object_instance::observation_history_intended_length) {

							obj.observation_history.pop_front();
						}

						if (obj.observation_history.empty())
						{
							iter = live_objects.erase(iter);
							if (checked_occlusion)
							{
								deletion = true;
							// std::cout << "permanently removed" << std::endl;
							}
							else
								deletion = true;
						}
						else
							++iter;
					}
				}

				if (deletion) // emission outside lock
					(*emitter)(id, enact_priority::operation::DELETED);
				else 
					(*emitter)(id, enact_priority::operation::MISSING);
			}
			else
				++iter;
		}


		//deal with segments which didn't get matched
		for (const pc_segment::Ptr& seg : segments)
		{
			if (!matched_segments.contains(seg))
			{
				std::stringstream id;
				id << "object@" << std::setprecision(3) << timestamp.count() << "s@" << seg->centroid;

				entity_id obj_id(new enact_core::entity_id(id.str()));
				auto object_instance_p = std::make_unique<object_instance_data>(seg);

				world.register_data(obj_id, object_instance::aspect_id, std::move(object_instance_p));
				live_objects.push_back(obj_id);

				(*emitter)(obj_id, enact_priority::operation::CREATE);
			}
		}

		world.purge();
//		batch_completed();

		//std::cout << "tracking took " << ((std::chrono::high_resolution_clock::now() - start) / 1000000).count() << " ms" << std::endl;
	});
}

void object_tracker::update(const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
	std::lock_guard<std::mutex> lock(clouds_mutex);
	clouds.emplace(cloud);
}

void object_tracker::reset()
{
	for (const entity_id& id : live_objects)
		(*emitter)(id, enact_priority::operation::DELETED);

	live_objects.clear();
	clouds = std::queue<pcl::PointCloud<PointT>::ConstPtr>();
	timestamp = std::chrono::duration<float>(0);
}

std::chrono::duration<float> object_tracker::get_latest_timestamp() const
{
	return timestamp;
}

float object_tracker::compare_position(const object_instance& hand, const pc_segment& seg) const
{
	Eigen::Vector3f hand_center = hand.observation_history.back()->bounding_box.translation;
	hand_center.z() += 0.5f * hand.observation_history.back()->bounding_box.diagonal.z();
	Eigen::Vector3f seg_center = seg.bounding_box.translation;
	seg_center.z() += 0.5f * seg.bounding_box.diagonal.z();

	return bell_curve((hand_center-seg_center).norm(), object_params.min_object_height);
}




/**
* Compares size based on bounding boxes.
*/

inline float object_tracker::compare_size(const object_instance& hand, const pc_segment& seg) const
{

	const Eigen::Vector3f& hand_box = hand.observation_history.back()->bounding_box.diagonal;
	const Eigen::Vector3f& seg_box = seg.bounding_box.diagonal;
	float stdev = object_params.min_object_height;

	float similarity_unrotated = 
		bell_curve(hand_box.x() - seg_box.x(), stdev) *
		bell_curve(hand_box.y() - seg_box.y(), stdev);

	float similarity_rotated =
		bell_curve(hand_box.y() - seg_box.x(), stdev) *
		bell_curve(hand_box.x() - seg_box.y(), stdev);

	float similarity = 
		std::max(similarity_unrotated, similarity_rotated) *
		bell_curve(hand_box.z() - seg_box.z(), stdev);

	return std::powf(similarity, 1.f / 3.f);
}

/**
* Compares size based on bounding boxes.
*/

inline float object_tracker::compare_color(const object_instance& hand, const pc_segment& seg) const
{
	const pcl::RGB& hand_col = hand.observation_history.back()->mean_color;
	const pcl::RGB& seg_col = seg.mean_color;
	const float stdev = 100.f / 255.f;
	
	float similarity = std::powf(hand_col.r / 255.f - seg_col.r / 255.f, 2.f)+
		std::powf(hand_col.g / 255.f - seg_col.g / 255.f, 2.f)+
		std::powf(hand_col.b / 255.f - seg_col.b / 255.f, 2.f);


	return bell_curve(similarity, stdev);
}

float object_tracker::bell_curve(float x, float stdev)
{
	return std::expf(-x * x / (2 * stdev * stdev));
}

} //namespace state_observation
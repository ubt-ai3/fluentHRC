#include "pn_world_traceability.hpp"

#include "enact_core/access.hpp"
#include "enact_core/data.hpp"
#include "enact_core/id.hpp"



namespace state_observation
{

const float pn_world_tracer::classification_certainty_threshold = 0.3f;

pn_world_tracer::pn_world_tracer(enact_core::world_context& world,
	const pointcloud_preprocessing& pc_prepro,
	occlusion_detector::Ptr occlusion_detect,
	const pn_net::Ptr& net,
	const std::vector<object_prototype::ConstPtr>& prototypes,
	bool allow_dynamic_places)
	:
	world(world),
	pc_prepro(pc_prepro),
	net(net),
	occlusion_detect(std::move(occlusion_detect)),
	timestamp(0),
	allow_dynamic_places(allow_dynamic_places)
{
	std::lock_guard<std::mutex> lock(this->net->mutex);
	
	for (auto& classifier : prototypes)
	{
		token_traces.emplace(classifier, std::make_shared<pn_object_token>(classifier));
	}

	for (const auto& p : net->get_places())
	{
		if (const auto box_p = std::dynamic_pointer_cast<pn_boxed_place>(p))
			empty_places.emplace(box_p);
	}
}

pn_world_tracer::pn_world_tracer(enact_core::world_context& world,
	const pointcloud_preprocessing& pc_prepro,
	occlusion_detector::Ptr occlusion_detect,
	const pn_net::Ptr& net,
	std::map<object_prototype::ConstPtr, pn_object_token::Ptr> token_traces,
	bool allow_dynamic_places)
	:
	world(world),
	pc_prepro(pc_prepro),
	net(net),
	token_traces(std::move(token_traces)),
	occlusion_detect(std::move(occlusion_detect)),
	timestamp(0),
	allow_dynamic_places(allow_dynamic_places)
{
	std::lock_guard<std::mutex> lock(this->net->mutex);
	
	for (const auto& p : net->get_places())
	{
		if (const auto box_p = std::dynamic_pointer_cast<pn_boxed_place>(p))
			empty_places.emplace(box_p);
	}
}

pn_world_tracer::~pn_world_tracer() = default;

const std::map<object_prototype::ConstPtr, pn_object_token::Ptr>& pn_world_tracer::get_token_traces() const
{
	return token_traces;
}

pn_net::Ptr pn_world_tracer::get_net() const
{
	return net;
}



pn_emission::ConstPtr pn_world_tracer::generate_emissions(std::chrono::duration<float> timestamp)
{
	// TODO occluded static places without an object are considered empty
		//forward to current cloud

	if (clouds.empty() && !occlusion_detect->has_valid_reference_cloud())
		return std::make_shared<pn_emission>(std::set<pn_place::Ptr>(),
			std::set<pn_place::Ptr>(),
			std::map<pn_instance, double>(),
			std::map<pn_place::Ptr, double>());


	{
		pcl::PointCloud<PointT>::ConstPtr cloud;
		std::lock_guard<std::mutex> lock(clouds_mutex);
		while (clouds.size() > 1 && std::chrono::microseconds(clouds.front()->header.stamp) < timestamp - std::chrono::duration<float>(0.001))
			clouds.pop();

		if (!clouds.empty())
			cloud = clouds.front();
		else
			std::cout << "No cloud with timestamp " << timestamp << " given\n";

		if(cloud)
			occlusion_detect->set_reference_cloud(cloud);
	}

	std::set<pn_place::Ptr> empty_places(this->empty_places.begin(), this->empty_places.end());
	std::set<pn_place::Ptr> unobserved_places;
	std::map<pn_instance, double> token_distribution;
	std::map<pn_place::Ptr, double> max_probabilities;

	empty_places.insert(this->cleared_places.begin(), this->cleared_places.end());

	for (const pn_place::Ptr& place : net->get_places())
	{
		if (!empty_places.contains(place))
			unobserved_places.emplace(place);

		const auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place);

		if (boxed_place && occlusion_detect->perform_detection(boxed_place->box) == state_observation::occlusion_detector::result::COVERED)
		{
			empty_places.erase(place);
			unobserved_places.emplace(place);
		}
	}

	for (const std::shared_ptr<enact_core::entity_id>& id : world.live_entities(object_instance::aspect_id))
	{
		enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::read));
		const enact_core::const_access<object_instance_data> access_object(l.at(id, object_instance::aspect_id));
		const object_instance& obj = access_object->payload;

		//corresponding place for entity_id id
		pn_boxed_place::Ptr place;

		{
			const auto iter = static_places.find(id);

			if (iter != static_places.end())
				place = iter->second;
		}

		{
			const auto iter = dynamic_places.find(id);

			if (iter != dynamic_places.end())
				place = iter->second;
		}

		//if (!place)
		//	std::cout << "not found " << id->debug_name << std::endl;

		if (!place)
			continue;

		const auto occlusion_type = occlusion_detect->perform_detection(*obj.observation_history.back()->points);
		if (occlusion_type == occlusion_detector::COVERED)
		{
			unobserved_places.emplace(place);
			continue;
		}

		if (obj.get_classified_segment())
		{
			// compute total weight
			float weight = 0.f;
			for (const auto& entry : obj.get_classified_segment()->classification_results)
			{
				if (entry.local_certainty_score > classification_certainty_threshold)
					weight += entry.local_certainty_score;
			}

			if (weight == 0.f)
				continue;
			
			unobserved_places.erase(place);
			const auto peak = max_probabilities.emplace(place, 0).first;

			for (const auto& entry : obj.get_classified_segment()->classification_results)
			{
				auto tr = token_traces.find(entry.prototype);
				if (tr == token_traces.end())
					continue;
				
				peak->second = std::max(peak->second, static_cast<double>(entry.local_certainty_score / weight));

				if (entry.local_certainty_score > classification_certainty_threshold) {
					token_distribution.emplace(std::make_pair(place, tr->second), entry.local_certainty_score / weight);
				}
				else
					token_distribution.emplace(std::make_pair(place, tr->second), 0.f);
			}

		}

	}

	cleared_places.clear();

	// unobserved places that overlap with an occupied one must be empty.
	for(auto iter = unobserved_places.begin(); iter != unobserved_places.end();)
	{
		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(*iter);

		bool found_overlap = false;
		if (boxed_place)
		{
			for (const auto& overlap : boxed_place->overlapping_places)
				if (!unobserved_places.contains(overlap.lock()) &&
					!empty_places.contains(overlap.lock()))
				{
					empty_places.emplace(*iter);
					iter = unobserved_places.erase(iter);
					found_overlap = true;
					break;
				}
		}

		if(!found_overlap)
			++iter;
	}

	return std::make_shared<pn_emission>(std::move(empty_places),
		std::move(unobserved_places),
		std::move(token_distribution),
		std::move(max_probabilities));
}

void pn_world_tracer::update_sync(const entity_id& id, enact_priority::operation op)
{


	if (op == enact_priority::operation::CREATE || op == enact_priority::operation::UPDATE)
	{
		obb box;

		//std::cout << "update " << id->debug_name << std::endl;

		{
			enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::write));
			enact_core::const_access<object_instance_data> access_object(l.at(id, object_instance::aspect_id));
			auto& obj = access_object->payload;

			// ignore briefly existing false positives due to arm movement
			if (obj.observation_history.size() <= 1)
			{
				//std::cout << "no history " << id->debug_name << std::endl;
				return;
			}

			if (obj.is_background())
			{
				//std::cout << "background " << id->debug_name << std::endl;
				return;
			}

			// ignore spourious segments
			const auto classification = obj.get_classified_segment();
			if (!classification || classification->classification_results[0].local_certainty_score < classification_certainty_threshold)
			{
				// std::cout << "unsure " << id->debug_name << std::endl;
				return;
			}

			box = access_object->payload.observation_history.back()->bounding_box;
		}

		//check empty static places
		bool found = false;
		for (auto iter = empty_places.begin(); iter != empty_places.end();/*increment in body*/)
		{
			auto& p = *iter;

			// check center of box contained in p->box (extended to x-y-plane)
			Eigen::Vector3f distance = (box.translation - p->box.translation).cwiseAbs();
			if (distance.x() < 0.5f * p->box.diagonal.x() && distance.y() < 0.5f * p->box.diagonal.y())
			{
				// top of box is above top of place box
				if (box.top_z() > p->box.top_z() + 0.02)
				{
					covered_places.emplace(p);
					iter = empty_places.erase(iter);
					continue;
				}

				// top of boxes differ more than 2cm
				if (std::abs(box.top_z() - p->box.top_z()) > 0.02)
				{
					++iter;
					continue;
				}

				//if (n == 'r' && (*iter)->box.translation.z() > 0.05)
				//	std::cout << "fail" << std::endl;

				static_places.emplace(id, *iter);
				empty_places.erase(iter);
				cleared_places.erase(p);
				//(*emitter)(p, enact_priority::operation::UPDATE);
				found = true;

				break;
			}
			++iter;
		}

		if (found)
			return;

		// check filled static places and replace
		for (auto iter = static_places.begin(); iter != static_places.end(); ++iter)
		{
			auto& p = iter->second;

			if (id == iter->first)
				return;

			// top of box is above top of place box
			if (box.top_z() > p->box.top_z() + 0.01)
			{
				continue;
			}

			// top of box is below center of flying place box
			if (p->box.bottom_z() - 0.01 > 0 &&
				box.top_z() < p->box.translation.z())
			{
				continue;
			}
			
			// check overlap of box and p->box (extended to x-y-plane)
			Eigen::Vector3f distance = (box.translation - p->box.translation).cwiseAbs();
			if (distance.x() + 0.01 < p->box.diagonal.x() && distance.y() + 0.01 < p->box.diagonal.y())
			{
				auto place = iter->second;
				static_places.erase(iter); 
				static_places.emplace(id, place);	// would invalidate iter if put before erase			
				cleared_places.erase(p);
				//(*emitter)(p, enact_priority::operation::UPDATE);
				return;
			}
		}

		// check filled dynamic places and replace
		for (auto iter = dynamic_places.begin(); iter != dynamic_places.end(); ++iter)
		{
			const auto& p = iter->second;
						
			if (id == iter->first)
				return;

			// check center of box contained in p->box (extended to x-y-plane)
			Eigen::Vector3f distance = (box.translation - p->box.translation).cwiseAbs();
			if (distance.x() < 0.5f * p->box.diagonal.x() && distance.y() < 0.5f * p->box.diagonal.y())
			{
				auto place = iter->second;
				dynamic_places.erase(iter);
				dynamic_places.emplace(id, place);
				cleared_places.erase(p);
				//(*emitter)(p, enact_priority::operation::UPDATE);
				return;
			}
		}

		if (!allow_dynamic_places)
			return;

		// add a new dynamic place
		constexpr float inf = std::numeric_limits<float>::infinity();
		Eigen::Vector3f min(inf,inf,inf);
		Eigen::Vector3f max(-inf,-inf,-inf);

		for (const Eigen::Vector3f& corner : box.get_corners())
		{
			min = min.cwiseMin(corner);
			max = max.cwiseMax(corner);
		}

		auto p = std::make_shared<pn_boxed_place>(obb::from_corners(min, max));
		net->add_place(p);
		dynamic_places.emplace(id, p);

		(*emitter)(p, enact_priority::operation::CREATE);
	}
	else if (op == enact_priority::operation::DELETED) 
	{
	// std::cout << "delete " << id->debug_name << std::endl;

		{ // check static places
			const auto iter = static_places.find(id);
			if (iter != static_places.end())
			{
				auto p = iter->second; 
				empty_places.emplace(p);
				cleared_places.emplace(p);

				// method modifies static_places, we cannot pass iter->second here
				check_covered_places(p);  
				static_places.erase(iter);
				
				//(*emitter)(p, enact_priority::operation::UPDATE);
				return;
			}
		}

		{ // check dynamic places
			const auto iter = dynamic_places.find(id);
			if (iter != dynamic_places.end())
			{
				auto p = iter->second;
				cleared_places.emplace(iter->second);
				dynamic_places.erase(iter);
				(*emitter)(p, enact_priority::operation::DELETED);
				return;
			}
		}
	}
}

void pn_world_tracer::update(const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
	std::lock_guard<std::mutex> lock(clouds_mutex);
	if (!cloud->empty())
		clouds.emplace(cloud);
}

void pn_world_tracer::check_covered_places(const pn_boxed_place::Ptr& empty_place)
{
	pn_boxed_place::Ptr uncovered_place;
	for (const pn_boxed_place::Ptr& covered_place : covered_places)
	{
		// check center of covered_place box contained in empty_place box (regarding x and y coordinate)
		Eigen::Vector3f distance = (covered_place->box.translation - empty_place->box.translation).cwiseAbs();
		if (distance.x() > 0.5f * empty_place->box.diagonal.x() || distance.y() > 0.5f * empty_place->box.diagonal.y())
			continue;

		const float bottom_empty_box = empty_place->box.bottom_z();
		const float top_covered_box = covered_place->box.top_z();
		
		// boxes may overlap
		if (bottom_empty_box - top_covered_box < 0.01)
		{
			uncovered_place = covered_place;
			break;
		}
	}

	if (uncovered_place)
	{
		empty_places.emplace(uncovered_place);
		covered_places.erase(uncovered_place);		
		check_covered_places(uncovered_place);
	}
}

}

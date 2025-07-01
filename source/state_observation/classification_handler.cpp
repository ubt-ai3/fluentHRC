#include "classification_handler.hpp"

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <chrono>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <ranges>


#include <enact_core/access.hpp>
#include <enact_core/data.hpp>
#include <enact_core/lock.hpp>
#include <enact_core/world.hpp>

#include "pn_model_extension.hpp"

namespace state_observation
{


/////////////////////////////////////////////////////////////
//
//
//  Class: classifier_set
//
//
/////////////////////////////////////////////////////////////

//classifier_set::classifier_set(const std::shared_ptr<const object_parameters>& object_params)
//{
//	classifier::set_object_params(object_params);
//
//	filename_ = std::string("classifier_set.xml");
//
//	std::ifstream file(folder_ + filename_);
//	if (false /*file.good()*/) {
//		//boost::archive::xml_iarchive ia{ file };
//		//register_classifiers(ia);
//		//ia >> BOOST_SERIALIZATION_NVP(*this);
//	}
//	else {
//		aabb block(Eigen::Vector3f(0.028f, 0.028f, 0.058f));
//		aabb cube(Eigen::Vector3f(0.028f, 0.028f, 0.028f));
//		aabb flat_block(Eigen::Vector3f(0.058f, 0.028f, 0.014f));
//		aabb plank(Eigen::Vector3f(0.098f, 0.028f, 0.014f));
//		aabb semicylinder(Eigen::Vector3f(0.035f, 0.028f, 0.016f));
//		pcl::RGB red(230, 80, 55);
//		pcl::RGB blue(5, 20, 110);
//		pcl::RGB cyan(5, 115, 185);
//		pcl::RGB wooden(200, 190, 180);
//		pcl::RGB magenta(235, 45, 135);
//		pcl::RGB purple(175, 105, 180);
//		pcl::RGB yellow(250, 255, 61);
//		pcl::RGB dark_green(51, 82, 61);
//		const std::string mesh_loc("assets/object_meshes/");
//		mesh_wrapper::Ptr cuboid_mesh			= std::make_shared < mesh_wrapper>(mesh_loc + "cube.obj");
//		mesh_wrapper::Ptr cylinder_mesh			= std::make_shared < mesh_wrapper>(mesh_loc + "cylinder.obj");
//		mesh_wrapper::Ptr semicylinder_mesh		= std::make_shared < mesh_wrapper>(mesh_loc + "semicylinder.obj");
//		mesh_wrapper::Ptr triangular_prism_mesh = std::make_shared < mesh_wrapper>(mesh_loc + "triangular_prism.obj");
//		mesh_wrapper::Ptr bridge_mesh			= std::make_shared < mesh_wrapper>(mesh_loc + "bridge.obj");
//		classifiers.push_back(
//			classifier::Ptr(new background_classifier(
//				object_prototype::Ptr(new object_prototype(
//					aabb(),
//					dark_green,
//					nullptr,
//					"dark green background"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cylinder_classifier(
//				object_prototype::Ptr(new object_prototype(
//					cube,
//					wooden,
//					cylinder_mesh,
//					"wooden small cylinder"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new semicylinder_classifier(
//				object_prototype::Ptr(new object_prototype(
//					semicylinder,
//					wooden,
//					semicylinder_mesh,
//					"wooden semicylinder"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cylinder_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					wooden,
//					cylinder_mesh,
//					"wooden cylinder"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					cube,
//					wooden,
//					cuboid_mesh,
//					"wooden cube"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					wooden,
//					cuboid_mesh,
//					"wooden block"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new triangular_prism_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					wooden,
//					triangular_prism_mesh,
//					"wooden triangular prism"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new bridge_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					wooden,
//					bridge_mesh,
//					"wooden bridge"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new bridge_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					purple,
//					bridge_mesh,
//					"purple bridge"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new bridge_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					magenta,
//					bridge_mesh,
//					"magenta bridge"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					flat_block,
//					wooden,
//					cuboid_mesh,
//					"wooden flat block"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					plank,
//					wooden,
//					cuboid_mesh,
//					"wooden plank"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					plank,
//					cyan,
//					cuboid_mesh,
//					"cyan plank"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					plank,
//					red,
//					cuboid_mesh,
//					"red plank"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					plank,
//					yellow,
//					cuboid_mesh,
//					"yellow plank"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					red,
//					cuboid_mesh,
//					"red block"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					blue,
//					cuboid_mesh,
//					"blue block"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					yellow,
//					cuboid_mesh,
//					"yellow block"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cylinder_classifier(
//				object_prototype::Ptr(new object_prototype(
//					cube,
//					red,
//					cylinder_mesh,
//					"red small cylinder"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cylinder_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					magenta,
//					cylinder_mesh,
//					"magenta cylinder"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					flat_block, 
//					red,
//					cuboid_mesh,
//					"red flat block"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					flat_block,
//					yellow,
//					cuboid_mesh,
//					"yellow flat block"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					flat_block,
//					purple,
//					cuboid_mesh,
//					"purple flat block"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new semicylinder_classifier(
//				object_prototype::Ptr(new object_prototype(
//					semicylinder,
//					purple,
//					semicylinder_mesh,
//					"purple semicylinder"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new semicylinder_classifier(
//				object_prototype::Ptr(new object_prototype(
//					semicylinder,
//					magenta,
//					semicylinder_mesh,
//					"magenta semicylinder"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new triangular_prism_classifier(
//				object_prototype::Ptr(new object_prototype(
//					block,
//					cyan,
//					triangular_prism_mesh,
//					"cyan triangular prism"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					cube,
//					red,
//					cuboid_mesh,
//					"red cube"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					cube,
//					purple,
//					cuboid_mesh,
//					"purple cube"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					cube,
//					cyan,
//					cuboid_mesh,
//					"cyan cube"))
//			))
//		);
//
//		classifiers.push_back(
//			classifier::Ptr(new cuboid_classifier(
//				object_prototype::Ptr(new object_prototype(
//					cube,
//					yellow,
//					cuboid_mesh,
//					"yellow cube"))
//			))
//		);
//	}
//}
//
//classifier_set::~classifier_set()
//{
//	std::ofstream file(folder_ + filename_);
//	boost::archive::xml_oarchive oa{ file };
//	register_classifiers(oa);
//	const classifier_set& classifiers = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
//	oa << BOOST_SERIALIZATION_NVP(classifiers);
//}
//
//classifier_set::iterator classifier_set::begin()
//{
//	return classifiers.begin();
//}
//
//classifier_set::iterator classifier_set::end()
//{
//	return classifiers.end();
//}
//
//classifier_set::const_iterator classifier_set::begin() const
//{
//	return classifiers.begin();
//}
//
//classifier_set::const_iterator classifier_set::end() const
//{
//	return classifiers.end();
//}

/////////////////////////////////////////////////////////////
//
//
//  Class: classification_handler
//
//
/////////////////////////////////////////////////////////////

classification_handler::classification_handler(enact_core::world_context& world,
	const std::shared_ptr<const object_parameters>& object_params,
	const std::vector<object_prototype::ConstPtr>& prototypes)
	:
	classify(object_params, classifier::classifier::generate_aspects(prototypes)),
	world(world),
	object_params(object_params),
	latest_timestamp(0)
{}

classification_handler::~classification_handler()
{
	stop_thread();
}

void classification_handler::update(const entity_id& id)
{
	float priority = 0.f;

	{
		enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::read));
		const enact_core::const_access<object_instance_data> access_object(l.at(id, object_instance::aspect_id));
		const object_instance& obj = access_object->payload;

		if (obj.observation_history.size() < 2 || !obj.observation_history.back()->classification_results.empty())
			return;

		const pc_segment& latest_obs = *obj.observation_history.back();
		latest_timestamp = std::max(latest_timestamp, latest_obs.timestamp);

		priority += latest_obs.timestamp.count();
		priority += latest_obs.bounding_box.translation.x() * 100 + 100;
		//if (!latest_obs.classification_results.empty())
		//	return; 
		for (const pc_segment::Ptr& seg : obj.observation_history)
			if (!seg->classification_results.empty())
				priority += 10000;
	}
	//std::cout << "scheduled classification\n";
	schedule([this, w_id = std::weak_ptr<enact_core::entity_id>(id)]()
	{

		pc_segment::Ptr seg;
		entity_id id = w_id.lock();
		if (!id)
		{
			//std::cout << "aborted classification invalid id\n";
			return;
		}


		{
			enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::read));
			const enact_core::const_access<object_instance_data> access_object(l.at(id, object_instance::aspect_id));

			const object_instance& obj = access_object->payload;
			if (obj.observation_history.empty() || obj.observation_history.back()->timestamp != latest_timestamp)
				return;
			seg = obj.observation_history.back();

			if (batch_timestamp != seg->timestamp)
			{
				//if (batch_count)
				//{
				//	std::cout << "classification took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - batch_start).count()
				//		<< " ms for " << batch_count << " objects" << std::endl;
				//}

				batch_count = 0;
				batch_timestamp = seg->timestamp;
				batch_start = std::chrono::high_resolution_clock::now();
			}
			else
				batch_count++;
		}
		//pn_boxed_place

		std::vector<classification_result> results = classify_segment(*seg);

		//std::cout << "classified segment\n";
		{
			enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::write));
			const enact_core::access<object_instance_data> access_object(l.at(id, object_instance::aspect_id));
			//HACK
			//object_instance& obj = *const_cast<object_instance*>(&access_object->payload);
			object_instance& obj = access_object->payload;
			if (!obj.observation_history.empty())
			{
				//QUESTION why only swap and not pushback
				obj.observation_history.back()->classification_results.swap(results);
				obj.covered = false;
			}
			//else
			//	std::cout << "Error";
			//	auto& x = obj.observation_history.back();
				//decltype(x);
		}

		(*emitter)(id, enact_priority::operation::UPDATE);
	}, priority);

	//	std::cout << "classification queue: " << queue.size() << std::endl;
}



std::vector<classification_result> classification_handler::classify_segment(const pc_segment& segment) const
{

	//new version
	//auto start_time = std::chrono::high_resolution_clock::now();
	//std::cout << "new classifer:";

	//std::cout << "used : " << results2.size() << "classifiers\n";
	/*std::sort(results2.begin(), results2.end(), [](const classification_result& lhs, const classification_result& rhs) {
		return lhs.local_certainty_score > rhs.local_certainty_score;
		});*/
		//	auto duration_new = std::chrono::high_resolution_clock::now()-start_time;
		//
		//	old version
		//	start_time = std::chrono::high_resolution_clock::now();
		//	const float max_similarity = 0.f;
		//	std::vector<classification_result> results;
		//	int classified_count = 0;
		//	int clas_counter = 0;
		//	for (const classifier::Ptr& clas : old_classifier_)
		//	{
		//			std::cout << ".";
		//			results.push_back(clas->classify(segment));
		//			/*if (++clas_counter == 0)
		//				break;*/
		//
		//			 stop further classification for background objects
		//			if (max_similarity > 0.5f)
		//			{
		//				if (!clas->get_object_prototype()->has_mesh())
		//				{
		//				std::cout << "should be background\n";
		//				break;
		//				}
		//
		//			}
		//
		//		classified_count++;
		//
		//	}
		//	std::cout << "\nused : " << classified_count << "classifiers\n";
		//	std::sort(results.begin(), results.end(), [](const classification_result& lhs, const classification_result& rhs) {
		//		return lhs.local_certainty_score > rhs.local_certainty_score;
		//		});
		//	auto duration_old = std::chrono::high_resolution_clock::now()-start_time;
		//	return results;
		///*	std::cout << "Duration new: " << std::chrono::duration<float, std::milli>(duration_new).count()
		//			  << "\nDuration old: "<< std::chrono::duration<float, std::milli>(duration_old).count() << "\n";*/
		//	comparison
		//	if (results.size() != results2.size())
		//	{
		//		std::cout << "classifier count doesnt match\n";
		//		return results2;
		//	}
		//		
		//	for (auto& result: results)
		//	{
		//		float certainty = result.local_certainty_score;
		//		auto it = std::find_if(results2.begin(), results2.end(),
		//			[result](const classification_result& res2)
		//			{
		//				return result.prototype->get_name() == res2.prototype->get_name();
		//			});
		//		if (std::fabs(it->local_certainty_score - result.local_certainty_score) > 0.0001)
		//			std::cout << "difference between classifications detected\n";
		//		else
		//			std::cout << "classification successful";
		//	}
		//	static int counter = 0;
		//	std::cout << "Finished " << ++counter << " classification (comparisons)\n";
		//
		//	if (duration_new.count() > duration_old.count())
	auto results2 = classify.classify_all(segment);
	std::ranges::sort(results2, [](const classification_result& lhs, const classification_result& rhs)
	{
		return lhs.local_certainty_score > rhs.local_certainty_score;
	});
	return results2;
}



/////////////////////////////////////////////////////////////
//
//
//  Class: place_classification_handler
//
//
/////////////////////////////////////////////////////////////

const float place_classification_handler::classification_certainty_threshold = 0.2f;

place_classification_handler::place_classification_handler(enact_core::world_context& world,
	const pointcloud_preprocessing& pc_prepro,
	const std::shared_ptr<const object_parameters>& object_params,
	const std::vector<object_prototype::ConstPtr>& prototypes,
	const pn_net::Ptr& net,
	const std::map<object_prototype::ConstPtr, pn_object_token::Ptr>& token_traces,
	bool rotate)
	:
	place_classification_handler(world, pc_prepro, object_params, occlusion_detector::construct_from(pc_prepro, *object_params), prototypes, net, token_traces, rotate)
{}

place_classification_handler::place_classification_handler(enact_core::world_context& world,
	const pointcloud_preprocessing& pc_prepro,
	const std::shared_ptr<const object_parameters>& object_params,
	occlusion_detector::Ptr occlusion_detect,
	const std::vector<object_prototype::ConstPtr>& prototypes,
	const pn_net::Ptr& net,
	std::map<object_prototype::ConstPtr, pn_object_token::Ptr> token_traces,
	bool rotate)
	:
	classify(object_params, classifier::classifier::generate_aspects(prototypes)),
	world(world),
	pc_prepro(pc_prepro),
	object_params(object_params),
	occlusion_detect(std::move(occlusion_detect)),
	net(net),
	token_traces(std::move(token_traces)),
	cloud_stamp(0)
{
	const auto classifiers = classify.get_prototype_classifier();

	for (const auto& place : net->get_places())
	{
		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place);

		if (!boxed_place)
			continue;

		const auto& box = boxed_place->box;

		std::vector<classifier_rotation> prototypes_and_rotations;

		for (const auto& prototype : prototypes)
		{
			const auto equal = [&](int i, int j)
			{
				return std::abs(box.diagonal(i) - prototype->get_bounding_box().diagonal(j)) < object_params->min_object_height;
			};

			if (!rotate)
			{
				if(equal(0, 0) && equal(1, 1) && equal(2, 2))
				for (const auto& classifier : classifiers)
					if (classifier.prototype == prototype)
						prototypes_and_rotations.emplace_back(classifier, Eigen::Quaternionf::Identity());
			}
			else if (equal(0, 0) && equal(1, 1) && equal(2, 2) ||
				equal(0, 0) && equal(1, 2) && equal(2, 1) ||
				equal(0, 1) && equal(1, 0) && equal(2, 2) ||
				equal(0, 1) && equal(1, 2) && equal(2, 0) ||
				equal(0, 2) && equal(1, 0) && equal(2, 1) ||
				equal(0, 2) && equal(1, 1) && equal(2, 0))
			{
				for (const auto& classifier : classifiers)
					if (classifier.prototype == prototype)
						prototypes_and_rotations.emplace_back(classifier, classifier::bounding_box::rotation_guess(box, prototype->get_bounding_box()));
			}
		}

		place_infos.emplace_back(boxed_place, std::move(prototypes_and_rotations));
	}
}

place_classification_handler::~place_classification_handler()
{
	stop_thread();
}

void place_classification_handler::update(const pcl::PointCloud<PointT>::ConstPtr& cloud)
{
	if (cloud->empty())
		return;

	cloud_stamp = std::chrono::microseconds(cloud->header.stamp);

	schedule([this, cloud]()
	{
		if (cloud_stamp != std::chrono::microseconds(cloud->header.stamp))
			return;

		std::chrono::duration<float> timestamp = cloud_stamp;
		occlusion_detect->set_reference_cloud(cloud);

		if (net->get_goal() != current_goal)
		{
			prev_goal = current_goal;
			current_goal = net->get_goal();
		}

		std::set<pn_place::Ptr> relevant_instances;

		if(net->get_goal() != nullptr)
		{
			for (const auto& entry : net->get_goal_instances())
				relevant_instances.emplace(entry.first);

			if (prev_goal)
			{
				for (const auto& entry : prev_goal->get_incoming_transitions().begin()->lock()->get_side_conditions())
					relevant_instances.emplace(entry.first);
			}
		}

		for (auto& info : place_infos)
		{
			std::lock_guard<std::mutex> lock(*info.m);

			if (!relevant_instances.empty() && !relevant_instances.contains(info.place))
			{
				if(info.live_object)
					(*emitter)(info.live_object, enact_priority::operation::DELETED);

				info.covered = false;
				info.live_object = nullptr;

				continue;
			}

			// due to the properties of a depth camera, points from lower and higher objects might fall into a bounding box where no object is. 
			// Therefore we must check whether there is a higher, partial occluding object which generates these points
			auto result = occlusion_detect->perform_detection(info.box);
			info.covered = result == occlusion_detector::COVERED;

			if (info.covered)
			{
				if (!info.live_object)
					continue;

				enact_core::lock l(world, enact_core::lock_request(info.live_object, object_instance::aspect_id, enact_core::lock_request::write));
				enact_core::access<object_instance_data> access_object(l.at(info.live_object, object_instance::aspect_id));
				object_instance& obj = access_object->payload;

				obj.covered = true;

				continue;
			}

			pc_segment::Ptr seg = std::make_shared<pc_segment>();
			seg->timestamp = timestamp;

			auto object_cloud = classify.clip(cloud, info.box);


			if (object_cloud->size() < pointcloud_preprocessing::min_points_per_object)
			{
				if (!info.live_object)
					continue;

				bool deletion = false;
				auto live_obj = info.live_object;
				{
					enact_core::lock l(world, enact_core::lock_request(info.live_object, object_instance::aspect_id, enact_core::lock_request::write));
					enact_core::access<object_instance_data> access_object(l.at(info.live_object, object_instance::aspect_id));
					object_instance& obj = access_object->payload;

					obj.covered = false;

					if (cloud && obj.get_classified_segment())
					{
						if (result == occlusion_detector::DISAPPEARED ||
							classify.clip(cloud, obj.observation_history.front()->bounding_box)->size() < pointcloud_preprocessing::min_points_per_object / 2)
						{

							// std::cout << "removed box " << info.place->id << std::endl;
							info.live_object = nullptr;

							deletion = true;
						}

					}

					if (!deletion)
					{
						// keep classified segments and purge outdated
						if (obj.get_classified_segment() == obj.observation_history.front() &&
							obj.observation_history.size() > 3)
						{
							obj.observation_history.erase(++obj.observation_history.begin());
						}
						else if (obj.observation_history.size() > 3)
						{

							obj.observation_history.pop_front();
						}

						if (obj.observation_history.empty())
						{
							info.live_object = nullptr;
							deletion = true;

						}
					}
				}

				if (deletion) // emission outside lock
					(*emitter)(live_obj, enact_priority::operation::DELETED);

				continue;
			}

			// all cases of occlusion or no object ruled out

			seg->reference_frame = cloud;
			seg->bounding_box = pc_prepro.oriented_bounding_box_for_standing_objects(object_cloud);
			seg->bounding_box.translation.z() += 0.5f * info.box.bottom_z();
			seg->bounding_box.diagonal.z() -= info.box.bottom_z();

			pcl::CentroidPoint<PointT> centroid_computation;
			for (const auto& iter : *object_cloud)
				centroid_computation.add(iter);

			centroid_computation.get(seg->centroid);
			seg->points = object_cloud;
			seg->compute_mean_color();

			if (info.live_object)
			{
				enact_core::lock l(world, enact_core::lock_request(info.live_object, object_instance::aspect_id, enact_core::lock_request::write));
				enact_core::access<object_instance_data> access_object(l.at(info.live_object, object_instance::aspect_id));
				object_instance& obj = access_object->payload;

				obj.covered = false;

				if (obj.observation_history.back()->classification_results.front().local_certainty_score > classification_certainty_threshold && 
					compare_color(obj, *seg) > 0.99f && obj.observation_history.size() >= object_instance::observation_history_intended_length)
					obj.observation_history.back()->timestamp = seg->timestamp;
				else
				{
					obj.observation_history.push_back(seg);

					for (const auto& [aspect, rotation] : info.fitting_boxes)
					{
						seg->classification_results.emplace_back(classifier::classify(*seg, aspect, object_params->min_object_height, rotation));
						//
						//const auto& res = seg->classification_results.back();
						//int p_id = info.place->id;
						//int t_id = token_traces.at(clas_rot.first.prototype)->id;
						//float prob = res.local_certainty_score;

						//if(32 <= p_id && p_id <= 34 && t_id == 6 && prob < 0.3f)
						//	std::cout << p_id << " " << t_id << " " << prob << std::endl;

						//classifier::classify(*seg, clas_rot.first, object_params->min_object_height, clas_rot.second);
					}
					
					std::ranges::sort(seg->classification_results, [](const classification_result& lhs, const classification_result& rhs)
					{
						return lhs.local_certainty_score > rhs.local_certainty_score;
					});

					if (obj.get_classified_segment() == obj.observation_history.front() && obj.observation_history.size() > object_instance::observation_history_intended_length)
						obj.observation_history.erase(++obj.observation_history.begin());
					else if (obj.observation_history.size() > object_instance::observation_history_intended_length && obj.observation_history.front()->timestamp + purge_duration < timestamp)
						obj.observation_history.pop_front();
				}

				(*emitter)(info.live_object, enact_priority::operation::UPDATE);
			}
			else
			{
				for (const auto& [aspect, rotation] : info.fitting_boxes)
					seg->classification_results.emplace_back(classifier::classify(*seg, aspect, object_params->min_object_height, rotation));

				std::ranges::sort(seg->classification_results, std::greater(), &classification_result::local_certainty_score);

				if (seg->classification_results.empty() || seg->classification_results.front().local_certainty_score < 0.35f)
					continue;


				std::stringstream id;
				id << "object@" << std::setprecision(3) << seg->timestamp.count() << "s@" << seg->centroid;

				entity_id obj_id(new enact_core::entity_id(id.str()));
				auto object_instance_p = std::make_unique<object_instance_data>(seg);



				world.register_data(obj_id, object_instance::aspect_id, std::move(object_instance_p));
				info.live_object = obj_id;
				(*emitter)(obj_id, enact_priority::operation::CREATE);
			}
		}

		world.purge();

	});

	//	std::cout << "classification queue: " << queue.size() << std::endl;
}

pn_emission::ConstPtr state_observation::place_classification_handler::generate_emissions(std::chrono::duration<float> timestamp) const
{
	std::set<pn_place::Ptr> empty_places;
	std::set<pn_place::Ptr> unobserved_places;
	std::map<pn_instance, double> token_distribution;
	std::map<pn_place::Ptr, double> max_probabilities;

	{
		std::lock_guard<std::mutex> lock(net->mutex);
		unobserved_places.insert(net->get_places().begin(), net->get_places().end());
	}

	for (const auto& info : std::ranges::reverse_view(place_infos))
	{
		std::lock_guard<std::mutex> lock(*info.m);

		if (info.covered)
			continue;

		auto id = info.live_object;
		if (!id)
		{
			unobserved_places.erase(info.place);
			empty_places.emplace(info.place);
			continue;
		}

		enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::read));
		const enact_core::const_access<object_instance_data> access_object(l.at(id, object_instance::aspect_id));
		const object_instance& obj = access_object->payload;

		if (obj.observation_history.size() > 1 && obj.get_classified_segment())
		{
			// compute total weight
			float weight = 0.f;
			for (const auto& entry : obj.get_classified_segment()->classification_results)
			{
				if (entry.local_certainty_score > classification_certainty_threshold)
					weight += entry.local_certainty_score;
			}

			if (weight == 0.f)
			{
				unobserved_places.erase(info.place);
				empty_places.emplace(info.place);
				continue;
			}

			unobserved_places.erase(info.place);
			const auto peak = max_probabilities.emplace(info.place, 0).first;

			for (const auto& entry : obj.get_classified_segment()->classification_results)
			{
				auto tr = token_traces.find(entry.prototype);
				if (tr == token_traces.end())
					continue;

				double prob = static_cast<double>(entry.local_certainty_score / weight);
				peak->second = std::max(peak->second, prob);

				if (entry.local_certainty_score <= classification_certainty_threshold)
					prob = 0.0;
				
				token_distribution.emplace(std::make_pair(info.place, tr->second), prob);
			}
		}
	}

	return std::make_shared<pn_emission>(std::move(empty_places),
		std::move(unobserved_places),
		std::move(token_distribution),
		std::move(max_probabilities));
}

inline float place_classification_handler::compare_color(const object_instance& obj, const pc_segment& seg)
{
	const pcl::RGB& hand_col = obj.observation_history.back()->mean_color;
	const pcl::RGB& seg_col = seg.mean_color;
	constexpr float stdev = 100.f / 255.f;

	const float similarity = std::powf(hand_col.r / 255.f - seg_col.r / 255.f, 2.f) +
		std::powf(hand_col.g / 255.f - seg_col.g / 255.f, 2.f) +
		std::powf(hand_col.b / 255.f - seg_col.b / 255.f, 2.f);


	return std::expf(-similarity * similarity / (2 * stdev * stdev));
}


place_classification_handler::place_info::place_info(const pn_boxed_place::Ptr& place,
	std::vector<classifier_rotation> fitting_boxes)
	:
	place(place),
	box(place->box),
	live_object(nullptr),
	covered(false),
	fitting_boxes(std::move(fitting_boxes)),
	m(std::make_unique<std::mutex>())
{}

} //namespace state_observation
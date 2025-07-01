#pragma once

#ifndef STATE_OBSERVATION__CLASSIFICATION_HANDLER__HPP
#define STATE_OBSERVATION__CLASSIFICATION_HANDLER__HPP

#include "framework.hpp"

#include <vector>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "parameter_set.hpp"
#include "classification_new.hpp"

#include "enact_core/actor.hpp"
#include "enact_core/id.hpp"
#include "enact_core/world.hpp"
#include "enact_priority/priority_actor.hpp"
#include "enact_priority/signaling_actor.hpp"

#include "pn_model_extension.hpp"

namespace state_observation
{

//class STATEOBSERVATION_API classifier_set : public parameter_set
//{
//public:
//	typedef std::vector<classifier::Ptr>::iterator iterator;
//	typedef std::vector<classifier::Ptr>::const_iterator const_iterator;
//
//	classifier_set(const std::shared_ptr<const object_parameters>& object_params);
//
//	~classifier_set();
//
//	const_iterator begin() const;
//	const_iterator end() const;
//
//	iterator begin();
//	iterator end();
//
//	template<typename Archive>
//	void serialize(Archive& ar, const unsigned version)
//	{
//		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
//		ar& BOOST_SERIALIZATION_NVP(classifiers);
//	}
//
//protected:
//	std::vector<classifier::Ptr> classifiers;
//
//	template<typename Archive>
//	void register_classifiers(Archive& ar) const
//	{
//		ar.register_type<background_classifier>();
//		ar.register_type<cuboid_classifier>();
//		ar.register_type<cylinder_classifier>();
//		ar.register_type<semicylinder_classifier>();
//		ar.register_type<bridge_classifier>();
//		ar.register_type<triangular_prism_classifier>();
//	};
//};

class pn_net;

/**
 * @class classification_handler
 * @brief Handles object classification and updates in the workspace
 * 
 * Manages the classification of objects in the workspace by associating
 * segments with prototypes and updating their state. Integrates with the
 * world context and supports batch processing and live object management.
 * 
 * Features:
 * - Object classification and update
 * - Prototype association
 * - Batch processing support
 * - Live object management
 */
class STATEOBSERVATION_API classification_handler :
	public enact_priority::priority_actor,
	public enact_priority::signaling_actor<std::shared_ptr<enact_core::entity_id>>
{
public:
	typedef std::shared_ptr<enact_core::entity_id> entity_id;
	typedef enact_core::lockable_data_typed<object_instance> object_instance_data;

	classification_handler(
		enact_core::world_context& world,
		const std::shared_ptr<const object_parameters>& object_params,
		const std::vector<object_prototype::ConstPtr>& prototypes
		);

	virtual ~classification_handler() override;

	void update(const entity_id& id);
	std::vector<classification_result> classify_segment(const pc_segment& segment) const;

	classifier classify;

private:
	enact_core::world_context& world;
	const std::shared_ptr<const object_parameters> object_params;

	//classifier_set	  old_classifier_;

	std::priority_queue<std::shared_ptr<enact_core::entity_id>> live_objects;
	std::chrono::duration<float> latest_timestamp;

	std::chrono::steady_clock::time_point batch_start;
	unsigned int batch_count = 0;
	std::chrono::duration<float> batch_timestamp = std::chrono::duration<float>(0);
};


/**
 * @class place_classification_handler
 * @brief Handles place-based classification and emission generation
 * 
 * Extends classification_handler to support place-based classification,
 * emission generation, and token trace management. Integrates with Petri net
 * models and supports occlusion detection and rotation handling.
 * 
 * Features:
 * - Place-based classification
 * - Emission generation
 * - Token trace management
 * - Occlusion detection integration
 * - Rotation and goal management
 */
class STATEOBSERVATION_API place_classification_handler :
	public enact_priority::priority_actor,
	public enact_priority::signaling_actor<std::shared_ptr<enact_core::entity_id>>
{
public:
	using PointT = pcl::PointXYZRGBA;
	using entity_id = std::shared_ptr<enact_core::entity_id> ;
	using object_instance_data = enact_core::lockable_data_typed<object_instance>;
	using classifier_rotation = std::pair<classifier::classifier_aspect, Eigen::Quaternionf>;

	constexpr inline static std::chrono::duration<float> purge_duration = std::chrono::duration<float>(1.f);
	const static float classification_certainty_threshold;

	place_classification_handler(
		enact_core::world_context& world,
		const pointcloud_preprocessing& pc_prepro,
		const std::shared_ptr<const object_parameters>& object_params,
		const std::vector<object_prototype::ConstPtr>& prototypes,
		const pn_net::Ptr& net,
		const std::map<object_prototype::ConstPtr, pn_object_token::Ptr>& token_traces,
		bool rotate = true
	);

	place_classification_handler(enact_core::world_context& world,
		const pointcloud_preprocessing& pc_prepro,
		const std::shared_ptr<const object_parameters>& object_params,
		occlusion_detector::Ptr occlusion_detect,
		const std::vector<object_prototype::ConstPtr>& prototypes,
		const pn_net::Ptr& net,
		std::map<object_prototype::ConstPtr, pn_object_token::Ptr> token_traces,
		bool rotate = true
	);

	virtual ~place_classification_handler() override;

	void update(const pcl::PointCloud<PointT>::ConstPtr& cloud);

	pn_emission::ConstPtr generate_emissions(std::chrono::duration<float> timestamp) const;

	const std::map<object_prototype::ConstPtr, pn_object_token::Ptr>& get_token_traces() const
	{
		return token_traces;
	}

	pn_net::Ptr get_net() const
	{
		return net;
	}

	classifier classify;

private:
	struct place_info
	{
		place_info(const pn_boxed_place::Ptr& place,
			std::vector<classifier_rotation> fitting_boxes);

		place_info(place_info&&) = default;

		place_info& operator=(const place_info&) = default;
		place_info& operator=(place_info&&) = default;

		pn_boxed_place::Ptr place;
		obb box;
		std::shared_ptr<enact_core::entity_id> live_object;
		bool covered;
		std::vector<classifier_rotation> fitting_boxes;

		std::unique_ptr<std::mutex> m;
	};


	enact_core::world_context& world;
	const pointcloud_preprocessing& pc_prepro;
	const std::shared_ptr<const object_parameters> object_params;
	occlusion_detector::Ptr occlusion_detect;
	pn_net::Ptr net;
	std::map<object_prototype::ConstPtr, pn_object_token::Ptr> token_traces;

	//classifier_set	  old_classifier_;

	
	std::vector<obb> boxes;
	std::vector<std::vector<classifier_rotation>> clasifiers_fitting_boxes;

	// size identical to boxes.size()
	std::vector<place_info> place_infos;
	std::chrono::microseconds cloud_stamp;

	pn_place::Ptr prev_goal = nullptr;
	pn_place::Ptr current_goal = nullptr;

	static float compare_color(const object_instance& hand, const pc_segment& seg);
};

} // namespace state_observation

	
#endif // !STATE_OBSERVATION__CLASSIFICATION_HANDLER__HPP

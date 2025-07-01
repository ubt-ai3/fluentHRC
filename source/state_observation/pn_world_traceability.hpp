#pragma once

#ifndef STATE_OBSERVATION__PN_WORLD_TRACEABILITY_HPP
#define STATE_OBSERVATION__PN_WORLD_TRACEABILITY_HPP

#include "framework.hpp"

#include "enact_core/id.hpp"
#include "enact_core/world.hpp"

#include "enact_priority/signaling_actor.hpp"

#include "pn_model_extension.hpp"
#include "pn_reasoning.hpp"

#include "object_detection.hpp"
#include "classification_new.hpp"
#include "pointcloud_util.hpp"

namespace state_observation
{
	class classifier_set;

	/**
	 * @class pn_world_tracer
	 * @brief Maintains links between detected objects and Petri net tokens
	 * 
	 * Tracks the relationship between physical objects in the world and their
	 * corresponding tokens in a Petri net model. Handles object detection,
	 * classification, and state tracking.
	 * 
	 * Features:
	 * - Object-token mapping
	 * - Dynamic place management
	 * - Static place tracking
	 * - Occlusion detection
	 * - Point cloud processing
	 * - Emission generation
	 * - State synchronization
	 * - Classification certainty thresholds
	 */
	class STATEOBSERVATION_API pn_world_tracer : public enact_priority::signaling_actor<pn_boxed_place::Ptr>
	{
	public:
		typedef pcl::PointXYZRGBA PointT;
		typedef std::shared_ptr<enact_core::entity_id> entity_id;
		typedef enact_core::lockable_data_typed<object_instance> object_instance_data;


		/**
		 * \brief A classification result with local_certainty_score below
		 * * classification_certainty_threshold is considered to be a non-detection
		 */
		static const float classification_certainty_threshold;

		pn_world_tracer(enact_core::world_context& world,
			const pointcloud_preprocessing& pc_prepro,
			occlusion_detector::Ptr occlusion_detect,
			const pn_net::Ptr& net,
			const std::vector<object_prototype::ConstPtr>& object_prototypes,
			bool allow_dynamic_places = true);

		pn_world_tracer(enact_core::world_context& world,
			const pointcloud_preprocessing& pc_prepro,
			occlusion_detector::Ptr occlusion_detect,
			const pn_net::Ptr& net,
			std::map<object_prototype::ConstPtr, pn_object_token::Ptr> token_traces,
			bool allow_dynamic_places = true);

		~pn_world_tracer() override;

		[[nodiscard]] const std::map<object_prototype::ConstPtr, pn_object_token::Ptr>& get_token_traces() const;

		[[nodiscard]] pn_net::Ptr get_net() const;
		
		pn_emission::ConstPtr generate_emissions(std::chrono::duration<float> timestamp);

		void update_sync(const entity_id& id, enact_priority::operation op);

		void update(const pcl::PointCloud<PointT>::ConstPtr& cloud);
		
	private:
		enact_core::world_context& world;
		const pointcloud_preprocessing& pc_prepro;
		pn_net::Ptr net;
		std::map<object_prototype::ConstPtr, pn_object_token::Ptr> token_traces;

		std::map<entity_id, pn_boxed_place::Ptr, std::owner_less<>> dynamic_places;
		std::map<entity_id, pn_boxed_place::Ptr, std::owner_less<>> static_places;
		std::set<pn_boxed_place::Ptr> covered_places;
		std::set<pn_boxed_place::Ptr> empty_places;
		std::set<pn_boxed_place::Ptr> cleared_places;

		std::queue<pcl::PointCloud<PointT>::ConstPtr> clouds;
		std::mutex clouds_mutex;

		occlusion_detector::Ptr occlusion_detect;

		std::chrono::duration<float> timestamp;

		const bool allow_dynamic_places;

		void check_covered_places(const pn_boxed_place::Ptr& empty_place);
	};

}

#endif /* !STATE_OBSERVATION__PN_WORLD_TRACEABILITY_HPP */
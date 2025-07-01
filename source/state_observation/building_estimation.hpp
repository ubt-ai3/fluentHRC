#pragma once

#include "enact_priority/priority_actor.hpp"
#include "state_observation/pn_model.hpp"
#include "state_observation/pn_reasoning.hpp"
#include "state_observation/pn_model_extension.hpp"
#include "state_observation/object_prototype_loader.hpp"
#include "enact_core/id.hpp"
#include "state_observation/workspace_objects.hpp"
#include "state_observation/pn_world_traceability.hpp"
#include "state_observation/classification_handler.hpp"
//#include "simulation/task.hpp"

#include <map>
#include <thread>
#include <memory>
#include <list>
#include <string>
#include <utility>
#include <numbers>

typedef std::shared_ptr<enact_core::entity_id> entity_id;

namespace state_observation
{
	/**
	 * @class Position
	 * @brief Represents a position in a building structure
	 * 
	 * Defines a position in terms of layer and x-offset within a building structure.
	 * Used for precise placement of building elements.
	 * 
	 * Features:
	 * - Layer tracking
	 * - X-offset management
	 * - Position validation
	 */
	class STATEOBSERVATION_API Position
	{
	public:

		Position(int layer, int x_offset);

		[[nodiscard]] int Layer() const;
		[[nodiscard]] int xOffset() const;

	private:

		int layer;
		int x_offset;
	};

	/**
	 * @class building_element
	 * @brief Base class for building elements
	 * 
	 * Represents a fundamental building element with geometric properties
	 * and prototype associations. Manages element loading and token tracing.
	 * 
	 * Features:
	 * - Element loading and management
	 * - Bounding box computation
	 * - Prototype association
	 * - Token tracing
	 */
	class STATEOBSERVATION_API building_element
	{
	public: 
		typedef std::shared_ptr<building_element> Ptr;

		building_element(const building_element&) = default;
		building_element& operator=(const building_element&) = default;

		virtual ~building_element() noexcept = default;

		static void load_building_elements(const state_observation::object_prototype_loader& loader);
		static void load_building_elements(const std::vector<pn_object_token::Ptr>& prototypes);
		static void load_building_elements(const std::vector<pn_object_instance>& prototypes);
		static void load_building_element(const object_prototype& object, const pn_object_token::Ptr& token);
		//static void load_building_element(const object_prototype& object);
		static std::map<object_prototype::ConstPtr, pn_object_token::Ptr> get_token_traces();


		// Pose is specified in the local reference coordinate system of the building
		obb obbox;
		Eigen::Vector3f precomputed_diag;

	protected:

		building_element() = default;

		static std::map<std::string, aabb> element_bBox;
		static std::map<std::string, pn_object_token::Ptr> element_prototypes;
	};

	/**
	 * @class single_building_element
	 * @brief Represents a single building element
	 * 
	 * Extends building_element to represent individual building components
	 * with specific names and orientations.
	 * Pose is specified in the local reference coordinate system of the building
	 * 
	 * Features:
	 * - Element name tracking
	 * - Token association
	 * - Orientation management
	 */

	class STATEOBSERVATION_API single_building_element : public building_element
	{
	public:
		typedef std::shared_ptr<single_building_element> Ptr;
		/*enum class element_orientation
		{
			MIRRORED_HORIZONTAL,
			HORIZONTAL,
			MIRRORED_VERTICAL,
			VERTICAL
		};*/

		
		single_building_element(const std::string& element_name,
			const Eigen::Quaternionf& rotation = Eigen::Quaternionf(1.f, 0.f, 0.f, 0.f));

		const std::string element_name;
		const pn_object_token::Ptr token;
	};

	/**
	 * @class composed_building_element
	 * @brief Represents a composite building element
	 * 
	 * Combines multiple building elements into a single composite structure.
	 * Supports different orientations and element types.
	 * 
	 * Features:
	 * - Multiple element composition
	 * - Orientation control
	 * - Element type management
	 * - Dependency resolution
	 */

	class STATEOBSERVATION_API composed_building_element : public building_element
	{
	public:
		enum class element_type
		{
			BRIDGE,
			CYLINDER,
			SEMI_CYLINDER
		};
		enum class composed_orientation
		{
			VERTICAL = 0,
			HORIZONTAL = 1
		};

		/*
		 * Constructs a composed_building_element
		 * @throws if only less than 2 elements are supplied
		 * which are placeable in each others bb
		 */
		composed_building_element(
			const std::vector<std::string>& elements,
			bool mirror = false,
			composed_orientation orientation = composed_orientation::VERTICAL);

		static void load_type_diags();

		[[nodiscard]] std::vector<single_building_element::Ptr> getElements() const;

		//internal dependencies //TODO:: replace with internal dependency resolve strategy
		[[nodiscard]] std::vector<building_element::Ptr> getTopExposed() const;
		[[nodiscard]] std::vector<building_element::Ptr> getBottomExposed() const;

	private:

		std::vector<single_building_element::Ptr> all_elements;

		std::vector<single_building_element::Ptr> bridges;
		single_building_element::Ptr lower_cylinder;
		single_building_element::Ptr upper_cylinder;

		element_type get_element_type(const std::string& element);

		static std::map<element_type, Eigen::Vector3f> type_diagonal;
		const static std::vector<Eigen::Quaternionf> orientation_quaternion;
		const static std::map<std::string, element_type, std::greater<>> element_string_type;
	};

	/**
	 * @class building
	 * @brief Represents a complete building structure
	 * 
	 * Manages a multi-layer structure of building elements with spatial
	 * relationships and network representation.
	 * 
	 * Features:
	 * - Multi-layer structure
	 * - Collision detection
	 * - Network generation
	 * - Dependency management
	 * - Visualization support
	 */
	class STATEOBSERVATION_API building
	{
	public:

		typedef std::shared_ptr<building> Ptr;
		static const std::shared_ptr<enact_core::aspect_id> aspect_id;

		/*
		template<typename building_element_type>
		static std::pair<building_element::Ptr, Position> make_positioned_element(
			building_element_type&& element, Position pos)
		{
			building_element::Ptr element_ptr = std::make_shared<building_element_type>(std::move(element));
			return std::make_pair(element_ptr, pos);
		}
		*/
		inline static const float epsilon = 0.00001f;
		inline static const float spacing = 0.1f; // percentage of the block size
		inline static const float z_spacing = 0.0001f;

		building(const std::list<std::pair<building_element::Ptr, Position>>& elements, 
			Eigen::Vector3f translation, 
			Eigen::Quaternionf rotation,
			const pn_net::Ptr& network = nullptr);
		
		[[nodiscard]] std::list<single_building_element::Ptr> visualize() const;

		[[nodiscard]] const pn_net::Ptr& get_network() const;
		[[nodiscard]] pn_instance get_goal() const;

		//pn_binary_marking::Ptr get_marking() const;
		[[nodiscard]] const std::map<pn_place::Ptr, pn_token::Ptr> get_distribution() const;

		static void load_building_elements(const std::vector<object_prototype::ConstPtr>& prototypes);
		static void set_building_elements(std::map<std::string, pn_object_token::Ptr> building_elements);

		[[nodiscard]] const std::vector<std::map<int, building_element::Ptr>>& get_building_structure() const;

	private:

		[[nodiscard]] static std::map<int, std::set<int>> calculate_occupation_list(
			const std::list<std::pair<building_element::Ptr, Position>>& elements);

		void calculate_boundaries();
		void check_collisions();
		void adjust_z_translation();
		void calculate_width();
		void create_network(const pn_net::Ptr& net = nullptr);

		struct Extend
		{
			float start;
			float end;
		};

		//returns the start and end of the transformed element in x dimension
		static Extend get_x_extend(const building_element::Ptr& element);

		struct DependencyGraph
		{
			//element -> elements which depend on it
			std::map<building_element::Ptr, std::set<building_element::Ptr>> bottom_up;
			std::map<building_element::Ptr, std::set<building_element::Ptr>> top_down;
		};

		[[nodiscard]] DependencyGraph dependency_resolve_building() const;

		typedef std::map<float, building_element::Ptr>::const_iterator dep_it;

		//returns a range of elements the current element depends on
		static std::pair<dep_it, dep_it> dependency_resolve_element(
			const building_element::Ptr& element,
			const std::map<float, building_element::Ptr>& dependables);

		//generates places for the network
		//side effects change the structure of the building network
		[[nodiscard]] std::map<single_building_element::Ptr, pn_boxed_place::Ptr> generate_resources() const;

		Eigen::Vector3f translation = Eigen::Vector3f(0.f, 0.f, 0.f);
		Eigen::Quaternionf rotation = Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 5.f), std::sin(std::numbers::pi_v<float> / 5.f) * 1.f, 0.f, 0); //Eigen::Quaternionf(1.f, 0.f, 0.f, 0.f);

		pn_net::Ptr net;
		pn_instance goal;

		// <offset, element>[layer]
		std::vector<std::map<int, building_element::Ptr>> building_structure;

		struct Bound
		{
			float lower;
			float upper;
		};

		std::vector<Bound> layer_bound;

		/*
		std::vector<float> layer_lower_bound;
		std::vector<float> layer_upper_bound;
		*/
		float width = 0.f;
		float height = 0.f;

		static std::map<std::string, pn_object_token::Ptr> building_elements;
	};

	/**
	 * @class builder
	 * @brief Builder pattern implementation for buildings
	 * 
	 * Provides a fluent interface for constructing building structures
	 * by adding elements and specifying their positions.
	 * 
	 * Features:
	 * - Fluent building interface
	 * - Element addition
	 * - Position management
	 * - Building creation
	 */
	class STATEOBSERVATION_API builder
	{
	public:

		builder() = default;
		builder(const builder& builder) = delete;

		builder& add_single_element(const std::string& type, Position pos);

		template<typename building_element_type>
		builder& add_element(building_element_type&& element, Position pos)
		{
			const auto& element_ptr = std::make_shared<building_element_type>(element);
			elements.emplace_back(element_ptr, pos);

			return *this;
		}
		//template<> STATEOBSERVATION_API builder& add_element<single_building_element>(const std::string& type, Position pos);
		//template<> STATEOBSERVATION_API builder& add_element<composed_building_element>(const std::string& type, Position pos);

		building::Ptr create_building(Eigen::Vector3f translation, Eigen::Quaternionf rotation, const pn_net::Ptr& net = nullptr);

	private:

		typedef std::pair<building_element::Ptr, Position> PtrPos;
		std::list<PtrPos> elements;

		//std::list<std::pair<building_element::Ptr, Position>> elements;
	};

	enum class STATEOBSERVATION_API update_state
	{
		RELOCATED,
		SHAPE_CHANGED,
		NEWLY_DETECTED,
		NO_CHANGE
	};

	/**
	 * @class building_net
	 * @brief Petri net representation of a building
	 * 
	 * Extends the Petri net model to represent building structures
	 * and their relationships.
	 * 
	 * Features:
	 * - Building state representation
	 * - Network management
	 * - State transitions
	 */
	class STATEOBSERVATION_API building_net : public pn_net
	{
		friend building;
	};

	/**
	 * @class building_estimation
	 * @brief Estimates and tracks building states
	 * 
	 * Implements building state estimation and tracking using
	 * priority-based actor model and Petri nets.
	 * 
	 * Features:
	 * - State estimation
	 * - Object tracking
	 * - Workspace monitoring
	 * - Priority-based updates
	 */
	class STATEOBSERVATION_API building_estimation : public enact_priority::priority_actor
	{
	public:
		typedef std::shared_ptr<enact_core::entity_id> strong_id;
		typedef std::weak_ptr<enact_core::entity_id> weak_id;

		typedef enact_core::lockable_data_typed<object_instance> object_instance_data;
		//typedef enact_core::lockable_data_typed<hand_pose_estimation::hand_instance> hand_instance_data;

		//Test function for classify_box_in_segment
		//static void test_function(const entity_id& id);

		building_estimation(enact_core::world_context& world,
			const std::vector<object_prototype::ConstPtr>& prototypes);

		~building_estimation() noexcept override;

		void update(const strong_id& id, enact_priority::operation op);
	private:

		const std::vector<object_prototype::ConstPtr>& prototypes;
		enact_core::world_context& world;

		bool is_workspace_object(const pc_segment::Ptr& object) const;

		bool is_picked_up(const object_instance& object) const;
		bool is_placed_down() const;

		float distance(const pc_segment::PointT& centroid_0, const pc_segment::PointT& centroid_1) const;

		float displacement_tolerance = 0.03f;
		std::map<weak_id, pc_segment::PointT, std::owner_less<weak_id>> segments;
		std::map<weak_id, pc_segment, std::owner_less<weak_id>> classified_shape;
		//std::map<std::string, pc_segment::Ptr> id_mapping;

		update_state update(const pc_segment::Ptr& seg, const strong_id& s_id);
	};

	/*struct observed_object
	{
		observed_object(
			const obb& obb,
			const object_prototype::ConstPtr& object);

		const pn_boxed_place::Ptr boxed_place;
		const object_prototype::ConstPtr object;
		const pn_agent_place::Ptr& agent();
		const pn_agent_place::Ptr& agent() const;

	private:

		pn_agent_place::Ptr agent_place;
	};*/

	/*class agent
	{
	public:
		/*agent(const std::map<object_prototype::ConstPtr, pn_object_token::Ptr>& proto_token
			); //TODO:: Add workspace as parameter
		agent(const std::vector<pn_agent_place::Ptr>& agent_place);*/
		//agent(const pn_net::Ptr& net);
		//void place(const object_prototype::ConstPtr& object, const obb& box)
		//const std::vector<pn_transition>& get_place_transitions() const;

		/*const std::vector<pn_transition::Ptr>& get_constructive_transitions() const;
		const std::vector<pn_agent_place::Ptr>& get_agent_places() const;

	private:

		std::vector<pn_agent_place::Ptr> agent_places;
		std::vector<pn_transition::Ptr> constructive_transitions;
		std::vector<pn_transition::Ptr>  destructive_transitions;
		//const std::map<object_prototype::ConstPtr, pn_object_token::Ptr>& prototype_token;
	};*/










	/*
	struct building_simulation_test
	{
		simulation::sim_task::Ptr task;
		building::Ptr building;
	};

	std::shared_ptr<building_simulation_test> create_building_simulation_task(
		const object_parameters& object_params,
		object_prototype_loader loader);*/
}

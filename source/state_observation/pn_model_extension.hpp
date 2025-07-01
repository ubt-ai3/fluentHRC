#pragma once

#include "pn_model.hpp"
#include "workspace_objects.hpp"

namespace state_observation
{
/*
 * A token representing the absence of a ressource in a network
 * Must not share a place with other tokens at the same time
 * Unique in a network
 */
/**
 * @class pn_empty_token
 * @brief Represents the absence of a resource in a Petri net
 * 
 * A special token type that indicates the absence of a resource.
 * Must not share a place with other tokens simultaneously.
 * Each network can have only one empty token.
 */
class STATEOBSERVATION_API pn_empty_token : public pn_token
{
public:
	typedef std::shared_ptr<pn_empty_token> Ptr;

	pn_empty_token() = default;

	pn_empty_token& operator=(const pn_empty_token&) = default;
	pn_empty_token& operator=(pn_empty_token&&) = default;

#ifdef DEBUG_PN_ID

	pn_empty_token(pn_empty_token&&);
	pn_empty_token(const pn_empty_token&);

#else
	pn_empty_token(pn_empty_token&&) = default;
	pn_empty_token(const pn_empty_token&) = default;
#endif

	~pn_empty_token() override = default;

	static pn_empty_token::Ptr find_or_create(const pn_net::Ptr& network);
};

/**
 * @class pn_object_token
 * @brief Represents an object in a Petri net
 * 
 * A token type that carries information about a specific object.
 * Links Petri net tokens to object prototypes in the workspace.
 */
class STATEOBSERVATION_API pn_object_token : public pn_token
{
public:
	typedef std::shared_ptr<pn_object_token> Ptr;
	typedef std::shared_ptr<const pn_object_token> ConstPtr;

	const object_prototype::ConstPtr object;

	pn_object_token(const object_prototype::ConstPtr& object);

	~pn_object_token() override = default;
};

/**
 * @class pn_boxed_place
 * @brief Represents a spatial area in a Petri net
 * 
 * A place that corresponds to a physical location in 3D space.
 * Tracks overlapping places and maintains spatial relationships.
 * 
 * Features:
 * - 3D bounding box representation
 * - Top height tracking
 * - Overlapping place management
 * - Spatial relationship tracking
 */
class STATEOBSERVATION_API pn_boxed_place : public pn_place
{
public:
	typedef std::shared_ptr<pn_boxed_place> Ptr;
	typedef std::weak_ptr<pn_boxed_place> WPtr;

	const state_observation::obb box;
	const float top_z;

	struct comparer
	{
		bool operator()(const WPtr& lhs, const WPtr& rhs) const
		{
			return &*lhs.lock() < &*rhs.lock();
		}
	};

	std::set<WPtr, comparer> overlapping_places;

	explicit pn_boxed_place(const obb& box);
	~pn_boxed_place() override = default;
};


using pn_object_instance = std::pair<pn_boxed_place::Ptr, pn_object_token::Ptr>;

/**
 * @class pick_action
 * @brief Represents a pick operation in a Petri net
 * 
 * A transition that models picking up an object from a location.
 * Tracks source and destination places and the object being moved.
 * 
 * Features:
 * - Source location tracking
 * - Destination tracking
 * - Object token association
 * - String representation
 */
class STATEOBSERVATION_API pick_action : public pn_transition
{
public:
	typedef std::shared_ptr<pick_action> Ptr;

	pick_action(const pn_object_token::Ptr& token,
		const pn_boxed_place::Ptr& from,
		const pn_place::Ptr& to);

	pick_action(
		const pn_net::Ptr& net,
		const pn_object_token::Ptr& token,
		const pn_boxed_place::Ptr& from,
		const pn_place::Ptr& to);

	//const Eigen::Vector3f center;
	const pn_boxed_place::Ptr from;
	const pn_place::Ptr to;
	const pn_object_token::Ptr token;

	std::string to_string() const override;
};

/**
 * @class place_action
 * @brief Represents a place operation in a Petri net
 * 
 * A transition that models placing an object at a location.
 * Tracks source and destination places and the object being moved.
 * 
 * Features:
 * - Source tracking
 * - Destination location tracking
 * - Object token association
 * - Optional empty token usage
 * - String representation
 */
class STATEOBSERVATION_API place_action : public pn_transition
{
public:
	typedef std::shared_ptr<place_action> Ptr;

	place_action(const pn_object_token::Ptr& token,
		const pn_place::Ptr& from,
		const pn_boxed_place::Ptr& to);

	place_action(
		const pn_net::Ptr& net,
		const pn_object_token::Ptr& token,
		const pn_place::Ptr& from,
		const pn_boxed_place::Ptr& to,
		bool use_empty_token = true);


	const pn_place::Ptr from;
	const pn_boxed_place::Ptr to;
	const pn_object_token::Ptr token;

	std::string to_string() const override;
};

/**
 * @class stack_action
 * @brief Represents a stacking operation in a Petri net
 * 
 * A transition that models stacking objects on top of each other.
 * Supports both 1:1 and n:1 stacking operations.
 * 
 * Features:
 * - Multiple creation methods
 * - Side condition handling
 * - Bottom object tracking
 * - Top object tracking
 * - Center point calculation
 * - String representation
 */
class STATEOBSERVATION_API stack_action : public pn_transition
{
public:
	/*
	 *	"use_empty_token = false" is used to not break any possible dependencies
	 *  if empty_token's are used they have be initialized before first evaluating a net
	 */

	typedef std::shared_ptr<stack_action> Ptr;

	/**
	* Creates a new transition that represents a stack operation and adds it to the net.
	* The object upon which the stacking is performed is considered a side condition.
	*/
	static stack_action::Ptr create(
		const pn_net::Ptr& net,
		const std::map< object_prototype::ConstPtr, pn_object_token::Ptr>& token_traces,
		const pn_place::Ptr& from,
		const pn_boxed_place::Ptr& bottom_location,
		const object_prototype::ConstPtr& bottom_object,
		const object_prototype::ConstPtr& top_object,
		bool use_empty_token = false);

	/**
	* Creates a new transition that represents a stack operation and adds it to the net.
	* The object upon which the stacking is performed is considered a side condition.
	*/
	static stack_action::Ptr create(
		const pn_net::Ptr& net,
		const pn_place::Ptr& from,
		const pn_boxed_place::Ptr& bottom_location,
		const pn_object_token::Ptr& bottom_object,
		const pn_object_token::Ptr& top_object,
		bool use_empty_token = false);

	/**
	* Same as above method but with association between location and object
	* Above kept to not break any depending code
	*/
	static stack_action::Ptr create(
		const pn_net::Ptr& net,
		const pn_place::Ptr& from,
		const pn_object_instance& bottom_located_objects,
		const pn_object_token::Ptr& top_object,
		bool use_empty_token = false);

	/**
	* Creates a new transition that represents a n:1 stack operation and adds it to the net.
	* The objects upon which the stacking is performed are considered a side condition.
	*/
	static stack_action::Ptr create(
		const pn_net::Ptr& net,
		const pn_place::Ptr& from,
		const std::vector<pn_object_instance>& bottom_located_objects,
		const pn_object_instance& top_located_object,
		bool use_empty_token = false);

	const pn_instance from;
	const pn_object_instance to;
	const object_prototype::ConstPtr top_object;

	const std::vector<object_prototype::ConstPtr> bottom_objects;

	const Eigen::Vector3f center;

	std::string to_string() const override;

private:

	stack_action(
		const pn_object_token::Ptr& token,
		const pn_place::Ptr& from,
		const pn_boxed_place::Ptr& to,
		const std::vector<pn_instance>& side_conditions,
		const std::vector<object_prototype::ConstPtr>& bottom_objects,
		const object_prototype::ConstPtr& top_object,
		const Eigen::Vector3f& center,
		const std::vector<pn_instance>& pre_conditions = {}
	);
};

/**
 * @class reverse_stack_action
 * @brief Represents an unstacking operation in a Petri net
 * 
 * A transition that models removing objects from a stack.
 * Supports 1:n unstacking operations.
 * 
 * Features:
 * - Multiple object handling
 * - Side condition tracking
 * - Bottom object tracking
 * - Top objects tracking
 * - String representation
 */
class STATEOBSERVATION_API reverse_stack_action : public pn_transition
{
public:
	typedef std::shared_ptr<reverse_stack_action> Ptr;

	/**
	* Creates a new transition that represents a 1:n inverse stack operation and adds it to the net.
	* The absence of the above objects is considered a side condition.
	*/
	static reverse_stack_action::Ptr create(
		const pn_net::Ptr& net,
		const pn_place::Ptr& to,
		const std::vector<pn_object_instance>& top_located_objects,
		const pn_object_instance& bottom_located_object
	);

	const pn_instance to;
	const pn_object_instance from;
	const object_prototype::ConstPtr bottom_object;

	const std::vector<object_prototype::ConstPtr> top_objects;

	std::string to_string() const override;

private:

	reverse_stack_action(
		const pn_object_token::Ptr& token,
		const pn_place::Ptr& to,
		const pn_boxed_place::Ptr& from,
		const std::vector<pn_instance>& side_conditions,
		const std::vector<object_prototype::ConstPtr>& top_objects,
		const object_prototype::ConstPtr& bottom_object,
		const std::vector<pn_instance>& post_conditions = {}
	);
};

/*
* Returns the object that is placed if it is a stack or place action
* Returns nullptr otherwise
*/
[[nodiscard]] STATEOBSERVATION_API pn_object_instance get_placed_object(const pn_transition::Ptr& transition);


/*
* Returns the object that is picked if it is a pick action
* Returns nullptr otherwise
*/
[[nodiscard]] STATEOBSERVATION_API pn_object_instance get_picked_object(const pn_transition::Ptr& transition);
}
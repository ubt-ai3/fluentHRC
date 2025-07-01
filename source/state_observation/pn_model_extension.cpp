#include "pn_model_extension.hpp"
namespace state_observation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: pn_boxed_place
//
//
/////////////////////////////////////////////////////////////

pn_boxed_place::pn_boxed_place(const obb& box)
	:
	box(box),
	top_z(box.top_z())
{}


/////////////////////////////////////////////////////////////
//
//
//  Class: pn_empty_token
//
//
/////////////////////////////////////////////////////////////

#ifdef DEBUG_PN_ID

pn_empty_token::pn_empty_token(pn_empty_token&& token)
{
	this->id = token.id;
}

pn_empty_token::pn_empty_token(const pn_empty_token& token)
{
	this->id = token.id;
}

#endif

pn_empty_token::Ptr pn_empty_token::find_or_create(const pn_net::Ptr& network)
{
	auto& tokens = network->get_tokens();
	for (const auto& token : tokens)
	{
		if (auto cast = std::dynamic_pointer_cast<pn_empty_token>(token); cast)
			return cast;
	}
	auto empty_token = std::make_shared<pn_empty_token>();
	network->add_token(empty_token);
	return empty_token;
}

pick_action::pick_action(const pn_object_token::Ptr& token,
	const pn_boxed_place::Ptr& from,
	const pn_place::Ptr& to)
	:
	pn_transition({ std::make_pair(from, token) }, { std::make_pair(to, token) }),
	//center(from->box.translation)
	from(from),
	to(to),
	token(token)
{}

/////////////////////////////////////////////////////////////
//
//
//  Class: pick_action
//
//
/////////////////////////////////////////////////////////////

pick_action::pick_action(
	const pn_net::Ptr& net,
	const pn_object_token::Ptr& token,
	const pn_boxed_place::Ptr& from,
	const pn_place::Ptr& to)
	:
	pn_transition(
		{ std::make_pair(from, token) },
		[&]()
		{
			std::vector<pn_instance> out;
			out.emplace_back(to, token);
			out.emplace_back(from, pn_empty_token::find_or_create(net));
			return out;
		}()
	),
	from(from),
	to(to),
	token(token)
{

}


std::string pick_action::to_string() const
{
	std::stringstream ss;
	ss << "pick," << token->object->get_name() << "," << from->id;
	return ss.str();
}


/////////////////////////////////////////////////////////////
//
//
//  Class: place_action
//
//
/////////////////////////////////////////////////////////////

place_action::place_action(const pn_object_token::Ptr& token, const pn_place::Ptr& from, const pn_boxed_place::Ptr& to)
	:
	pn_transition({ std::make_pair(from, token) }, { std::make_pair(to, token) }),
	//center(to->box.translation)
	from(from),
	to(to),
	token(token)
{}

place_action::place_action(
	const pn_net::Ptr& net,
	const pn_object_token::Ptr& token,
	const pn_place::Ptr& from,
	const pn_boxed_place::Ptr& to,
	bool use_empty_token)
	: 
	pn_transition(
		[&]()
		{
			std::vector<pn_instance> input;
			input.emplace_back(from, token);
			if (use_empty_token)
				input.emplace_back(to, pn_empty_token::find_or_create(net));
			return input;
		}(),
		{ std::make_pair(to, token) }
	),
	from(from),
	to(to),
	token(token)
{}

std::string place_action::to_string() const
{
	std::stringstream ss;
	ss << "place," << token->object->get_name() << "," << to->id;
	return ss.str();
}


/////////////////////////////////////////////////////////////
//
//
//  Class: stack_action
//
//
/////////////////////////////////////////////////////////////

stack_action::Ptr stack_action::create(
	const pn_net::Ptr& net,
	const std::map< object_prototype::ConstPtr, pn_object_token::Ptr>& token_traces,
	const pn_place::Ptr& from,
	const pn_boxed_place::Ptr& bottom_location,
	const object_prototype::ConstPtr& bottom_object,
	const object_prototype::ConstPtr& top_object,
	bool use_empty_token)
{
	return create(net, from, bottom_location, token_traces.at(bottom_object), token_traces.at(top_object));
}

stack_action::Ptr stack_action::create(
	const pn_net::Ptr& net,
	const pn_place::Ptr& from,
	const pn_boxed_place::Ptr& bottom_location,
	const pn_object_token::Ptr& bottom_object,
	const pn_object_token::Ptr& top_object,
	bool use_empty_token)
{
	Eigen::Vector3f center(
		bottom_location->box.translation.x(),
		bottom_location->box.translation.y(),
		bottom_location->box.translation.z() - 0.5 * bottom_location->box.diagonal.z() +
		bottom_object->object->get_bounding_box().diagonal.z() +
		0.5 * top_object->object->get_bounding_box().diagonal.z() + 0.001f //shift by 1mm as epsilon to avoid overlap with lower object
	);

	pn_boxed_place::Ptr top_place;

	for (const auto& p : net->get_places())
	{
		auto boxed_p = std::dynamic_pointer_cast<pn_boxed_place>(p);

		if (boxed_p && (boxed_p->box.translation - center).cwiseAbs().maxCoeff() < net->object_params->min_object_height)
		{
			top_place = boxed_p;
			break;
		}
	}

	if (!top_place)
	{
		top_place = std::make_shared<pn_boxed_place>(obb(
			top_object->object->get_bounding_box().diagonal,
			center
		));

		net->add_place(top_place);
	}

	std::vector<pn_instance> pre_conditions;
	if (use_empty_token)
		pre_conditions.emplace_back(
			top_place, pn_empty_token::find_or_create(net));

	auto transition = std::make_shared<stack_action>(stack_action(
		top_object,
		from,
		top_place,
		{ std::make_pair(bottom_location, bottom_object) },
		{ bottom_object->object },
		top_object->object,
		center,
		pre_conditions
	));

	net->add_transition(transition);

	return transition;
}

stack_action::Ptr stack_action::create(
	const pn_net::Ptr& net,
	const pn_place::Ptr& from,
	const pn_object_instance& bottom_located_object,
	const pn_object_token::Ptr& top_object,
	bool use_empty_token)
{
	return create(net, from,
		 bottom_located_object.first, 
		bottom_located_object.second, 
		top_object, 
		use_empty_token);
}

stack_action::Ptr stack_action::create(
	const pn_net::Ptr& net,
	const pn_place::Ptr& from,
	const std::vector<pn_object_instance>& bottom_located_objects,
	const pn_object_instance& top_located_object,
	bool use_empty_token)
{
	Eigen::Vector3f center(top_located_object.first->box.translation);

	auto& places = net->get_places();
	const auto& search_result = std::find(places.begin(), places.end(), top_located_object.first);

	if (search_result == places.end())
		net->add_place(top_located_object.first);

	std::vector<pn_instance> pre_conditions;
	if (use_empty_token)
		pre_conditions.emplace_back(
			top_located_object.first, pn_empty_token::find_or_create(net));

	auto transition = std::make_shared<stack_action>(stack_action(
		top_located_object.second,
		from,
		top_located_object.first,
		[&]()
	{
		std::vector<pn_instance> side_conditions;
		side_conditions.reserve(bottom_located_objects.size());
		for (const auto& located_object : bottom_located_objects)
			side_conditions.emplace_back(
				located_object.first,
					located_object.second);
		return side_conditions;
	}(),
		[&]()
	{
		std::set<object_prototype::ConstPtr> objects;
		for (const auto& located_object : bottom_located_objects)
			objects.insert(located_object.second->object);
		std::vector<object_prototype::ConstPtr> out =
			std::vector<object_prototype::ConstPtr>(objects.begin(), objects.end());

		return out;
	}(),
		top_located_object.second->object,
		center,
		pre_conditions
		));
	net->add_transition(transition);

	return transition;
}

stack_action::stack_action(
	const pn_object_token::Ptr& token,
	const pn_place::Ptr& from,
	const pn_boxed_place::Ptr& to,
	const std::vector<pn_instance>& side_conditions,
	const std::vector<object_prototype::ConstPtr>& bottom_objects,
	const object_prototype::ConstPtr& top_object,
	const Eigen::Vector3f& center,
	const std::vector<pn_instance>& pre_conditions)
	: 
	pn_transition(
		[&]()
		{
			auto copy = side_conditions;
			copy.emplace_back(from, token);

			for (const auto& pre_condition : pre_conditions)
				copy.emplace_back(pre_condition);

			return copy;
		}(),
		[&]()
		{
			auto copy = side_conditions;
			copy.emplace_back(to, token);
			return copy;
		}()
	),
	from(std::make_pair(from, token)),
	to(std::make_pair(to, token)),
	bottom_objects(bottom_objects),
	top_object(top_object),
	center(center)
{}


std::string stack_action::to_string() const
{
	std::stringstream ss;
	ss << "stack," << top_object->get_name() << "," << to.first->id;
	return ss.str();
}

/////////////////////////////////////////////////////////////
//
//
//  Class: reverse_stack_action
//
//
/////////////////////////////////////////////////////////////

reverse_stack_action::Ptr reverse_stack_action::create(
	const pn_net::Ptr& net,
	const pn_place::Ptr& to,
	const std::vector<pn_object_instance>& top_located_objects,
	const pn_object_instance& bottom_located_object)
{
	const auto& empty_token = pn_empty_token::find_or_create(net);

	auto transition = std::make_shared<reverse_stack_action>(reverse_stack_action(
		bottom_located_object.second,
		to,
		bottom_located_object.first,
		[&]()
	{
		std::vector<pn_instance> side_conditions;
		side_conditions.reserve(top_located_objects.size());
		for (const auto& located_object : top_located_objects)
			side_conditions.emplace_back(
				located_object.first,
					empty_token);
		return side_conditions;
	}(),
	[&]()
	{
		std::set<object_prototype::ConstPtr> objects;
		for (const auto& located_object : top_located_objects)
			objects.insert(located_object.second->object);
		std::vector<object_prototype::ConstPtr> out =
			std::vector<object_prototype::ConstPtr>(objects.begin(), objects.end());

		return out;
	}(),
		bottom_located_object.second->object,
		std::vector<pn_instance>({ std::make_pair(bottom_located_object.first, empty_token) })
		));
	net->add_transition(transition);

	return transition;
}

reverse_stack_action::reverse_stack_action(
	const pn_object_token::Ptr& token,
	const pn_place::Ptr& to,
	const pn_boxed_place::Ptr& from,
	const std::vector<pn_instance>& side_conditions,
	const std::vector<object_prototype::ConstPtr>& top_objects,
	const object_prototype::ConstPtr& bottom_object,
	const std::vector<pn_instance>& post_conditions)
	: 
	pn_transition(
		[&]()
		{
			auto copy = side_conditions;
			copy.insert(copy.begin(), std::make_pair(from, token));
			return copy;
		}(),
		[&]()
		{
			auto copy = side_conditions;
			copy.insert(copy.begin(), std::make_pair(to, token));
			for (const auto& post_condition : post_conditions)
				copy.emplace_back(post_condition);
			return copy;
		}()
	),
	to(std::make_pair(to, token)),
	from(std::make_pair(from, token)),
	bottom_object(bottom_object),
	top_objects(top_objects)
{}

std::string reverse_stack_action::to_string() const
{
	std::stringstream ss;
	ss << "unstack," << bottom_object->get_name() << "," << from.first->id;
	return ss.str();
}


/////////////////////////////////////////////////////////////
//
//
//  Class: pn_object_token
//
//
/////////////////////////////////////////////////////////////

pn_object_token::pn_object_token(const object_prototype::ConstPtr& object)
	:
	object(object)
{}


pn_object_instance get_placed_object(const pn_transition::Ptr& transition)
{
	{
		if (const auto action = std::dynamic_pointer_cast<stack_action>(transition))
			return action->to;
	}

	{
		if (const auto action = std::dynamic_pointer_cast<place_action>(transition))
			return { action->to, action->token };
	}

	return {};
}

pn_object_instance get_picked_object(const pn_transition::Ptr& transition)
{
	{
		if (const auto action = std::dynamic_pointer_cast<reverse_stack_action>(transition))
			return action->from;
	}

	{
		if (const auto action = std::dynamic_pointer_cast<pick_action>(transition))
			return { action->from, action->token };
	}

	return {};
}

}
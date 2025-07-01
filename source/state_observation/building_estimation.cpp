#include "building_estimation.hpp"

#include <ranges>

#include "enact_core/world.hpp"
#include "enact_core/access.hpp"
#include "enact_core/data.hpp"
#include "enact_core/lock.hpp"
#include "enact_core/id.hpp"

namespace state_observation
{
	/*
	std::shared_ptr<building_simulation_test> create_building_simulation_task(
		const object_parameters& object_params,
		object_prototype_loader loader)
	{
		building_simulation_test test;
		const auto& prototypes = loader.get_prototypes();

		std::vector<pn_object_token::Ptr> tokens;
		for (const auto& proto : prototypes)
			tokens.emplace_back(std::make_shared<pn_object_token>(proto));

		building_element::load_building_elements(tokens);
		composed_building_element::load_type_diags();
		building::load_building_elements(prototypes);

		std::string wb = "wooden block";
		std::string wp = "wooden plank";
		std::string wcu = "wooden cube";
		std::string wcy = "wooden cylinder";
		std::string mcy = "magenta cylinder";
		std::string bb = "blue block";
		std::string pfb = "purple flat block";
		std::string yfb = "yellow flat block";

		const auto& rotation = Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 4.f), 0, std::sin(std::numbers::pi_v<float> / 4.f) * 1.f, 0);
		//const auto& rotation1 = Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 4.f), 0, 0, std::sin(std::numbers::pi_v<float> / 4.f) * 1.f);

		test.building = builder()
			.add_element(single_building_element(wb), Position(0, 0))
			.add_element(single_building_element(wb), Position(0, 2))
			.add_element(single_building_element(wp), Position(0, 1))
			.add_element(single_building_element(wp), Position(1, 1))
			.add_element(single_building_element(wcu), Position(2, 1))
			.add_element(single_building_element(wcu), Position(2, 2))
			.add_element(single_building_element(wcu), Position(2, 3))
			.add_element(single_building_element(wb), Position(3, 4))
			.add_element(single_building_element(wcy), Position(3, 0))
			.add_element(single_building_element(wcy), Position(3, 1))
			.add_element(single_building_element(wcy), Position(3, 2))
			.add_element(single_building_element(wcy), Position(3, 3))
			.add_element(single_building_element(mcy), Position(4, 2))
			.add_element(single_building_element(bb, rotation), Position(4, 1))
			.add_element(single_building_element(pfb), Position(5, 1))
			.add_element(single_building_element(yfb), Position(6, 1))
			.add_element(single_building_element(bb, rotation), Position(4, 0))
			.add_element(single_building_element(pfb), Position(5, 0))
			.add_element(single_building_element("red flat block"), Position(6, 0))
			.create_building(Eigen::Vector3f::Zero(), Eigen::Quaternionf::Identity());

		using namespace simulation;

		auto hand = test.building->get_network()->create_place(true);

		auto token_traces = building_element::get_token_traces();
		auto env = std::make_shared<environment>(
			test.building->get_network(),
			test.building->get_distribution(),
			token_traces);

		env->additional_scene_objects.push_back(std::make_shared<simulated_table>());
		

		float startX = 0.1f;
		float startY = 0.1f;

		int temp = 0;

		int i = 0;

		for (const auto& building_element : test.building->visualize())
		{
			obb tempObb(building_element->obbox.diagonal, Eigen::Vector3f(startX, startY, building_element->obbox.diagonal.z() * 0.5),
				building_element->obbox.rotation);
			//obb& tempObb = building_element->obbox;
			auto instance = env->add_object(building_element->token->object, tempObb);
			env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));

			//const auto& agent_places = env->net->get_agent_places();

			

			//env->net->add_transition(std::make_shared<pick_action>(instance.second, instance.first, hand));
			//Transition to dummy agent place

			startX += 0.08;
			if (startX > 0.5)
			{
				startX = 0.1f;
				startY += 0.05;
			}
			++i;
		};

		std::vector<place_action::Ptr> ground_actions;

		for (const auto& transition : env->net->get_transitions())
		{
			if (auto action = std::dynamic_pointer_cast<place_action>(transition))
			{
				ground_actions.push_back(action);
			}
		}

		auto human = std::make_shared<simulation::agent>(env,
			Eigen::Vector3f(0.15f, -simulated_table::width / 2 - 0.1f, 0),
			Eigen::Vector3f(0, 2. / 10 * simulated_table::width, 0),
			hand,
			[actions(env->net->get_transitions()), env, next_action(pn_transition::Ptr())](const pn_transition::Ptr& t, agent&) mutable
		{
			if (!actions.empty())
			{
				if (next_action == t)
				{
					actions.erase(std::find(actions.begin(), actions.end(), next_action));
					next_action = nullptr;
					return 1.;
				}
				if (next_action)
					return 0.;
				
				if (auto action = std::dynamic_pointer_cast<pick_action>(t))
				{
					auto token = t->outputs.begin()->second;
					auto marking = env->get_marking();

					for (const auto& action : actions)
					{
						if (auto placing = std::dynamic_pointer_cast<place_action>(action))
						{
							if (token == placing->token && marking->would_enable(placing, { std::pair(placing->from, placing->token) }))
							{
								next_action = placing;
								return 1.;
							}
						}
						else if (auto stacking = std::dynamic_pointer_cast<stack_action>(action))
						{
							const auto& instance = stacking->from;
							if (token == instance.second && marking->would_enable(stacking, { instance }))
							{
								next_action = stacking;
								return 1.;
							}
						}
						else
						{
							//throw std::exception("action not supported");
						}
					}
				}
			}

			return 0.;
		});

		test.task = std::make_shared<sim_task>(env, std::vector<simulation::agent::Ptr>({ human }));
		

		return std::make_shared<building_simulation_test>(std::move(test));
		*/



		//auto build = builder()
		/*	.add_element(single_building_element(bb), Position(0, 0))
			.add_element(single_building_element("yellow plank"), Position(0, 1))
			.add_element(single_building_element("yellow block"), Position(0, 2))
			.add_element(single_building_element("cyan plank"), Position(1, 1))
			.add_element(single_building_element("red small cylinder", rotation1), Position(2, 1))
			.add_element(single_building_element("red small cylinder", rotation1), Position(2, 2))
			.add_element(single_building_element("red small cylinder", rotation1), Position(2, 3))
			.add_element(single_building_element(yfb), Position(3, 0))
			.add_element(single_building_element("red plank"), Position(3, 1))*/
			/*.add_element(single_building_element(bb, rotation), Position(0, 0))
				.add_element(composed_building_element(
					{ "wooden bridge", "purple semicylinder",
					  "purple bridge", "wooden semicylinder" }), Position(1, 0))
				.add_element(single_building_element(bb, rotation), Position(2, 0))
				.create_building();*/


				/*for (const auto& constructive : agent_ge->get_constructive_transitions())
					{
						const auto& inputs = constructive->inputs;
						initial_marking->distribution.insert({ std::make_shared<pn_binary_marking>(net, inputs), 1.0 });
						// update(*initial_marking);
					}*/

					/*std::unique_ptr<enact_core::lockable_data_typed<building>> a = std::make_unique<enact_core::lockable_data_typed<building>>(building({
							{std::make_shared<single_building_element>("wooden block"),	Position(0,0)},
							{std::make_shared<single_building_element>("wooden block"), Position(0, 2)},
							{std::make_shared<single_building_element>("wooden block"), Position(3, 4)},
							{std::make_shared<single_building_element>("wooden plank"), Position(0, 1)},
							{std::make_shared<single_building_element>("wooden plank"), Position(1, 1)},
							{std::make_shared<single_building_element>("wooden cube"), Position(2, 1)},
							{std::make_shared<single_building_element>("wooden cube"), Position(2, 2)},
							{std::make_shared<single_building_element>("wooden cube"), Position(2, 3)},
							{std::make_shared<single_building_element>("wooden cylinder"), Position(3, 0)},
							{std::make_shared<single_building_element>("wooden cylinder"), Position(3, 1)},
							{std::make_shared<single_building_element>("wooden cylinder"), Position(3, 2)},
							{std::make_shared<single_building_element>("wooden cylinder"), Position(3, 3)},
							{std::make_shared<single_building_element>("magenta cylinder"), Position(4, 2)},
							{std::make_shared<single_building_element>("blue block", Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 4.f), 0, std::sin(std::numbers::pi_v<float> / 4.f) * 1.f, 0)), Position(4, 1)},
							{std::make_shared<single_building_element>("purple flat block"), Position(5, 1)},
							{std::make_shared<single_building_element>("yellow flat block"), Position(6, 1)},
							{std::make_shared<composed_building_element>(composed_building_element(
								{single_building_element("wooden bridge"), single_building_element("purple semicylinder"),
								 single_building_element("purple bridge"), single_building_element("wooden semicylinder")})), Position(4, 0)}
							}));*/
	//}

	/*//Test function for classify_box_in_segment
	auto test_function = [&](const entity_id& id) {
		enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::read));
		enact_core::const_access<enact_core::lockable_data_typed<object_instance>> access_object(l.at(id, object_instance::aspect_id));
		const object_instance& obj = access_object->payload;

		//			const auto& seg_box = obj.observation_history.back()->bounding_box;
		//		auto model_box = seg_box;
		auto buildings = build_and_simulation->building->visualize();

		for (const auto& block : buildings)
			this->view->add_bounding_box(block->obbox, std::to_string((size_t)(&*block)));

		for (const auto& block : buildings)
		{
			classifier::classifier_aspect debug_classifier_aspect;
			debug_classifier_aspect.prototype = block->prototype;

			if (block->element_name.find("semicylinder") != std::string::npos)
				debug_classifier_aspect.type_ = classifier::type::semicylinder;
			else if (block->element_name.find("cylinder") != std::string::npos)
				debug_classifier_aspect.type_ = classifier::type::cylinder;
			else if (
				block->element_name.find("block") != std::string::npos ||
				block->element_name.find("plank") != std::string::npos ||
				block->element_name.find("cube") != std::string::npos)
				debug_classifier_aspect.type_ = classifier::type::cuboid;
			else if (block->element_name.find("bridge") != std::string::npos)
				debug_classifier_aspect.type_ = classifier::type::bridge;
			//else if (block->element_name.find("bridge") != std::string::npos)
			//debug_classifier_aspect.type_ = classifier::type::triangular_prism;
			else
				continue;
			//obb transformed_bb;

			/*for (const auto& block : buildings)
			{
				if (block->element_name == "magenta cylinder")
					transformed_bb = block->obbox;
			}*/

			//pn_boxed_place boxed_place(transformed_bb);
			//this->view->add_bounding_box(obj.observation_history.back()->bounding_box, "segment_real", 1, 0, 1);
		/*	auto res = obj_classify->classify.classify_box_in_segment(*obj.observation_history.back(), pn_boxed_place(block->obbox), debug_classifier_aspect);
			if (res.local_certainty_score > 0.)
			{
				view->update_object(id, enact_priority::operation::CREATE);
			}
			if (res.local_certainty_score >= 0.2)
			{
				//std::cout << "some proto";
			}
			if (res.local_certainty_score >= 0.5)
			{
				//	std::cout << "some proto";
			}
		}

		//compute model to real_transform
		//const auto& model_diag = model_box.diagonal;
		//const auto& seg_diag = seg_box.diagonal;

		//determines the permutation_matrix to rotate an aabb with shortest side along x and longest side along z axis
		//into the aabb with diagnoal box_diagonal; all aabb centers are located at the origin
		//diagonal = permutation_matrix * sorted_diagonal;
		/*auto compute_permutation_matrix3D = [](const Eigen::Vector3f& box_diagonal)
		{
			std::vector<std::reference_wrapper<const float>> diagonal(&box_diagonal.x(), &box_diagonal.x() + 3);
			std::sort(diagonal.begin(), diagonal.end());

			Eigen::Affine3f identity, permuted_identity;
			identity.setIdentity();
			const float* start = &box_diagonal.x();
			int i = 0;
			for (auto f : diagonal)
			{
				int index = &f.get() - start;
				permuted_identity.matrix().row(i) = identity.matrix().row(index);
				i++;
			}
			return permuted_identity;
		};*/

		/*Eigen::Affine3f model_to_real_transform =
			[](const obb& source_box, const obb& target_box)->Eigen::Affine3f
		{
			auto rotation_to_match_diagonals = [](const Eigen::Vector3f& source_diag, const Eigen::Vector3f& target_diag)
			{
				Eigen::Affine3f ret;
				if ((source_diag.x() < source_diag.y()) ==
					(target_diag.x() < target_diag.y()))
					ret.setIdentity();
				else
					//rotate +-90 deg around z;
					ret = Eigen::Affine3f(Eigen::AngleAxisf(M_PI / 2, Eigen::Vector3f::UnitZ()));

				//align top z;
				ret = Eigen::Translation3f(0, 0, 0.5 * (target_diag.z() - source_diag.z())) * ret;
				return ret;
			};

			return Eigen::Translation3f(target_box.translation) * Eigen::Affine3f(target_box.rotation) *
				rotation_to_match_diagonals(source_box.diagonal, target_box.diagonal) *
				Eigen::Affine3f(source_box.rotation.inverse()) * Eigen::Translation3f(-1 * source_box.translation);
		}(model_box, seg_box);

		auto transform_obb = [&model_to_real_transform](const obb& box)->obb
		{
			Eigen::Affine3f box_pose = Eigen::Translation3f(box.translation) * Eigen::Affine3f(box.rotation);
			Eigen::Affine3f transformed_box = model_to_real_transform * box_pose;
			Eigen::Vector3f translation(transformed_box.translation().x(), transformed_box.translation().y(), transformed_box.translation().z());
			Eigen::Quaternionf rotation(transformed_box.rotation());
			return obb(box.diagonal, translation, rotation);
		};*/

		/*Eigen::Affine3f model_transform = Eigen::Translation3f() * Eigen::Affine3f(model_box.rotation);
		//TEST
		Eigen::Affine3f computed_seg_transform = model_to_real_transform * model_transform;

		Eigen::Vector3f computed_seg_translation(computed_seg_transform.translation().x(),
			computed_seg_transform.translation().y(), computed_seg_transform.translation().z());
		Eigen::Quaternionf computed_seg_rotation(computed_seg_transform.rotation());
		bool is_equal = computed_seg_rotation.isApprox(seg_box.rotation);
		bool is_equal2 = computed_seg_translation.isApprox(seg_box.translation);
		Eigen::Vector3f trans(model_to_real_transform.translation().x(),
			model_to_real_transform.translation().y(), model_to_real_transform.translation().z());*/
	//};


	std::map<std::string, aabb> building_element::element_bBox;
	std::map<std::string, pn_object_token::Ptr> building_element::element_prototypes;

	void building_element::load_building_elements(const object_prototype_loader& loader)
	{
		for (const auto& object : loader.get_prototypes())
			load_building_element(*object, std::make_shared<pn_object_token>(object));
	}

	void building_element::load_building_elements(const std::vector<pn_object_token::Ptr>& prototypes)
	{
		for (auto& it : prototypes)
			load_building_element(*it->object, it);
	}

	void building_element::load_building_elements(const std::vector<pn_object_instance>& prototypes)
	{
		for (const auto& val : prototypes | std::views::values)
			load_building_element(*val->object, val);
	}

	void building_element::load_building_element(const object_prototype& object, const pn_object_token::Ptr& token)
	{
		const auto& name = object.get_name();

		element_bBox.emplace(name, object.get_bounding_box());

		const auto iter = element_prototypes.find(name);
		if (iter == element_prototypes.end())
			element_prototypes.emplace(name, token);
		else
			iter->second = token;
	}

	single_building_element::single_building_element(
		const std::string& element_name, 
		const Eigen::Quaternionf& rotation)
		: element_name(element_name), token(element_prototypes.at(element_name))
	{
		if (!element_bBox.contains(element_name))
			throw std::exception("Element does not exist");
		const auto& temp = element_bBox.at(element_name);

		precomputed_diag = (rotation.toRotationMatrix() * temp.diagonal).cwiseAbs();
		obbox = obb(temp.diagonal, 0.5 * precomputed_diag, rotation);
	}

	/*
	 *
	 */

	std::map<composed_building_element::element_type, Eigen::Vector3f> composed_building_element::type_diagonal;

	const std::vector<Eigen::Quaternionf> composed_building_element::orientation_quaternion =
	{
		Eigen::Quaternionf(1, 0, 0, 0),
		Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 4.f), 0, std::sin(std::numbers::pi_v<float> / 4.f) * 1.f, 0)
	};

	const std::map<std::string, composed_building_element::element_type, std::greater<>>
		composed_building_element::element_string_type =
	{
		{"bridge", element_type::BRIDGE},
		{"cylinder", element_type::CYLINDER},
		{"semicylinder", element_type::SEMI_CYLINDER}
	};

	composed_building_element::composed_building_element(
		const std::vector<std::string>& elements, bool mirror,
		composed_orientation orientation)
	{
		//restriction to min/max elements
		if (elements.size() > 4 || elements.size() == 1)
			throw std::exception("Invalid building compose");

		int bridges = 0, cylinders = 0, semicylinders = 0;
		Eigen::Vector3f bridge_diag;
		Eigen::Vector3f cylinder_diag;
		Eigen::Vector3f semicylinder_diag;

		//restriction to allowed amounts of allowed elements
		for (const auto& element : elements)
		{
			switch (get_element_type(element))
			{
			case element_type::BRIDGE:
				if (++bridges > 3)
					throw std::exception("Too many bridges");
				else if (bridges == 1)
					bridge_diag = type_diagonal.at(element_type::BRIDGE);
				break;
			case element_type::CYLINDER:
				if (++cylinders > 1 || semicylinders)
					throw std::exception("Invalid cylinders");
				cylinder_diag = type_diagonal.at(element_type::CYLINDER);
				break;
			case element_type::SEMI_CYLINDER:
				if (++semicylinders > 2 || cylinders)
					throw std::exception("Invalid semicylinders");
				else if (semicylinders == 1)
					semicylinder_diag = type_diagonal.at(element_type::SEMI_CYLINDER);
				break;
			}
		}
		if (!bridges || bridges == 1 && (semicylinders > 1 || cylinders))
			throw std::exception("Invalid: Bridges not matching");
		this->bridges.reserve(bridges);

		all_elements.reserve(bridges + semicylinders + cylinders);

		for (auto& element : elements)
		{
			switch (get_element_type(element))
			{
			case element_type::BRIDGE:
				if (this->bridges.empty())
				{
					const auto& rotation = 
						Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 4.f), 0, std::sin(std::numbers::pi_v<float> / 4.f) * 1.f, 0);
					auto& emplaced = 
						this->bridges.emplace_back(std::make_shared<single_building_element>(element, rotation));
					all_elements.emplace_back(emplaced);
				}
				else
				{
					const auto& rotation = 
						Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 2.f), std::sin(std::numbers::pi_v<float> / 2.f) * 1.f, 0, 0) * 
						this->bridges[0]->obbox.rotation;

					const auto& emplaced = 
						this->bridges.emplace_back(std::make_shared<single_building_element>(element, rotation));
					all_elements.emplace_back(emplaced);
					emplaced->obbox.translation.z() += this->bridges[0]->precomputed_diag.z();
				}
				break;
			case element_type::CYLINDER:
				lower_cylinder = std::make_shared<single_building_element>(element);
				all_elements.emplace_back(lower_cylinder);
				break;
			case element_type::SEMI_CYLINDER:
				if (!lower_cylinder)
				{
					const auto& rotation = 
						Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 2.f), std::sin(std::numbers::pi_v<float> / 2.f) * 1.f, 0, 0) * 
						Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 4.f), 0, 0, std::sin(std::numbers::pi_v<float> / 4.f) * 1.f);
					lower_cylinder = std::make_shared<single_building_element>(element, rotation);
					all_elements.emplace_back(lower_cylinder);
				}
				else
				{
					const auto& rotation = 
						Eigen::Quaternionf(std::cos(std::numbers::pi_v<float> / 4.f), 0, 0, std::sin(std::numbers::pi_v<float> / 4.f) * 1.f);
					upper_cylinder = std::make_shared<single_building_element>(element, rotation);
					all_elements.emplace_back(upper_cylinder);
					upper_cylinder->obbox.translation.z() += lower_cylinder->precomputed_diag.z();
				}
				break;
			}
		}

		if (lower_cylinder)
		{
			const auto& bridge0_diag = this->bridges[0]->precomputed_diag;

			const auto& lower_diag = lower_cylinder->precomputed_diag;
			auto& lower_trans = lower_cylinder->obbox.translation;

			const auto& bridge_offset_z = 
				(bridge0_diag.z() - lower_diag.z());

			lower_trans.z() += bridge_offset_z;
			lower_trans.x() += (bridge0_diag.x() - lower_diag.x()) / 2.f;

			if (upper_cylinder)
			{
				const auto& bridge1_diag = this->bridges[1]->precomputed_diag;

				const auto& upper_diag = upper_cylinder->precomputed_diag;
				auto& upper_trans = upper_cylinder->obbox.translation;
				
				upper_trans.z() += bridge_offset_z;
				upper_trans.x() += (bridge1_diag.x() - upper_diag.x()) / 2.f;
			}
		}

		Eigen::Vector3f diag = this->bridges[0]->precomputed_diag;

		if (bridges == 2)
			diag.z() += this->bridges[1]->precomputed_diag.z();
		else if (cylinders == 1)
			diag.z() += 0.5f * lower_cylinder->precomputed_diag.z();
		else if (semicylinders == 2)
			diag.z() += upper_cylinder->precomputed_diag.z();

		const auto& half_diag = 0.5f * diag;
		obbox = obb(diag, half_diag);
		precomputed_diag = diag;

		for (auto& element : all_elements)
			element->obbox.translation -= half_diag;
	}

	void composed_building_element::load_type_diags()
	{
		type_diagonal.emplace(element_type::BRIDGE, element_bBox.at("wooden bridge").diagonal);
		type_diagonal.emplace(element_type::CYLINDER, element_bBox.at("wooden cylinder").diagonal);
		type_diagonal.emplace(element_type::SEMI_CYLINDER, element_bBox.at("wooden semicylinder").diagonal);
	}

	composed_building_element::element_type composed_building_element::get_element_type(const std::string& element)
	{
		for (const auto& it : element_string_type)
		{
			if (element.rfind(it.first) != std::string::npos)
				return it.second;
		}

		throw std::exception("Element type not registered");
	}

	std::vector<single_building_element::Ptr> composed_building_element::getElements() const
	{
		std::vector<single_building_element::Ptr> out;
		out.reserve(all_elements.size());

		for (const auto& element : all_elements)
		{
			auto&& temp = std::make_shared<single_building_element>(*element);
			temp->obbox.rotation *= this->obbox.rotation;
			temp->obbox.translation += this->obbox.translation;
			out.emplace_back(temp);
		}
		return out;
	}

	std::vector<building_element::Ptr> composed_building_element::getTopExposed() const
	{
		return std::vector<building_element::Ptr>();
	}

	std::vector<building_element::Ptr> composed_building_element::getBottomExposed() const
	{
		return std::vector<building_element::Ptr>();
	}

	const std::shared_ptr<enact_core::aspect_id> building::aspect_id = 
		std::make_shared<enact_core::aspect_id>();

	building::building(const std::list<std::pair<building_element::Ptr, Position>>& elements, 
		Eigen::Vector3f translation,
		Eigen::Quaternionf rotation,
		const pn_net::Ptr& network)
		:
		translation(translation),
		rotation(rotation)
	{

		if (building_elements.empty() || elements.empty())
			throw std::exception("No building possible");
		{
			//Create map<layer, x_offsets>
			std::map<int, std::set<int>> known_layers = calculate_occupation_list(elements);

			building_structure.resize(known_layers.size());
			layer_bound.resize(known_layers.size());
		}

		for (auto& it : elements)
			building_structure[it.second.Layer()].emplace(it.second.xOffset(), it.first);

		calculate_boundaries();
		calculate_width();
		adjust_z_translation();

		for (auto& layer : building_structure)
		{
			float temp = 0.f;
			for (const auto& element : layer | std::views::values)
			{
				element->obbox.translation.x() += temp;
				element->obbox.translation.x() *= (1 + spacing);
				temp += element->precomputed_diag.x();
			}
		}

		check_collisions();

		create_network(network);

		//TODO:: symmetry for horizontal?
	}

	std::list<single_building_element::Ptr> building::visualize() const
	{
		std::list<single_building_element::Ptr> elements;
		for (const auto& vecIt : building_structure)
		{
			for (const auto& bElement : vecIt | std::views::values)
			{
				if (auto e =
					std::dynamic_pointer_cast<composed_building_element>(bElement))
				{
					for (const auto& element : e->getElements())
						elements.push_back(element);
				}
				else if(auto e =
					std::dynamic_pointer_cast<single_building_element>(bElement))
				{
					elements.push_back(e);
				}
			}
		}
		return elements;
	}

	const pn_net::Ptr& building::get_network() const
	{
		return net;
	}

	/*pn_binary_marking::Ptr building::get_marking() const
	{
		return std::make_shared<pn_binary_marking>(net, initial_distribution);
	}*/

	pn_instance building::get_goal() const
	{
		return goal;
	}

	const std::map<pn_place::Ptr, pn_token::Ptr> building::get_distribution() const
	{
		std::map<pn_place::Ptr, pn_token::Ptr> initial_distribution;
		const auto& empty_token = pn_empty_token::find_or_create(net);

		for (const auto& transition : net->get_transitions())
		{
			//empty tokens are always once present in transition inputs
			for (const auto& input : transition->inputs)
			{
				if (input.second == empty_token)
				{
					initial_distribution.emplace(input);
					break;
				}
			}
		}

		return initial_distribution;
	}

	std::map<object_prototype::ConstPtr, pn_object_token::Ptr> building_element::get_token_traces()
	{
		std::map<object_prototype::ConstPtr, pn_object_token::Ptr> result;

		for (const auto& entry : element_prototypes)
			result.emplace(entry.second->object, entry.second);
		
		return result;
	}

	std::map<int, std::set<int>> building::calculate_occupation_list(
		const std::list<std::pair<building_element::Ptr, Position>>& elements)
	{
		std::map<int, std::set<int>> known_layers;
		for (const auto& position : elements | std::views::values)
		{
			//Create entry for layer, offset
			auto layerEntry = known_layers.find(position.Layer());
			if (layerEntry == known_layers.end())
			{
				//This layer did not already exist
				known_layers.emplace(position.Layer(), std::set<int>{position.xOffset()});
			}
			else if (layerEntry->second.count(position.xOffset()) == 1)
			{
				//There was already an element at this offset (and layer)
				throw std::exception("Too many elements for a single offset");
			}
			else //Offset is unoccupied and layer already existed
				layerEntry->second.emplace(position.xOffset());
		}

		//TODO:: maybe just reduce layer numbers of affected layers?
		//Checks if all layers from bottom to top exist
		if (known_layers.rbegin()->first != (known_layers.size() - 1))
			throw std::exception("Airborne building parts");

		//Check for valid offsets
		for (auto layer = known_layers.begin(); layer != known_layers.end(); ++layer)
		{
			//not first layer
			if (layer != known_layers.begin())
			{
				size_t prev_offset = 0;
				for (auto& offset : layer->second)
				{
					//holes can be at max 1 offset number apart
					if (offset - prev_offset > 1)
						throw std::exception("Invalid offsets");
					else
						prev_offset = offset;
				}
			} //offset amount - 1 != offsetNumber for first layer
			else if (layer->second.size() - 1 != *layer->second.rbegin())
			{
				throw std::exception("First layer must be fully occupied");
			}
		}
		return known_layers;
	}

	void building::calculate_boundaries()
	{
		layer_bound[0].lower = 0.f;

		for (size_t i = 0; i < building_structure.size(); ++i)
		{
			float min = std::numeric_limits<float>::max();
			float max = 0.f;
			for (const auto& element : building_structure[i] | std::views::values)
			{
				//const auto& it_name = it->second.get_element_name();
				float it_z = element->precomputed_diag.z();
				
				min = std::min(min, it_z);
				max = std::max(max, it_z);
			}
			if (i < building_structure.size() - 1)
				layer_bound[i + 1].lower = layer_bound[i].lower + min;
			layer_bound[i].upper = layer_bound[i].lower + max;
		}
	}

	void building::check_collisions()
	{
		std::vector<float> layer_length(building_structure.size());
		layer_length[0] = width;

		for (size_t i = 0; i < building_structure.size() - 1; ++i)
		{
			int intersections = 0;
			for (size_t j = i + 1; j < building_structure.size(); ++j)
			{
				//calculate the amount of intersecting layers with the i-th layer
				if (layer_bound[i].upper > layer_bound[j].lower + epsilon)
					++intersections;
				else break;
			}

			if (intersections == 0)
			{
				//if there are no intersections then the length of the layer is
				//based solely on the individual elements of it
				for (const auto& element : building_structure[i + 1] | std::views::values)
					layer_length[i + 1] += element->precomputed_diag.x() * (1 + spacing);
				continue;
			}

			//if there are intersections we have to find the intersecting elements 
			//and add them to the intersecting layers 
			for (size_t j = i + 1; j < i + 1 + intersections; ++j)
			{
				auto offset_i = building_structure[i].begin();
				size_t prev_offset_j = 0;
				
				for (auto offset_j = building_structure[j].begin(); 
					offset_j != building_structure[j].end(); ++offset_j)
				{
					auto trans = [](const auto& iter)
					{
						return (iter->second->obbox.translation); // return by ref
					};

					auto diag = [](const auto& iter)
					{
						return (iter->second->precomputed_diag);
					};

					auto contained = [&]()
					{
						return 
							(trans(offset_i).z() - trans(offset_j).z()) +
							0.5f * (diag(offset_i).z() + diag(offset_j).z());
					};

					size_t relative_offset = offset_j->first - prev_offset_j;
					if (relative_offset > j - i)
						throw std::exception("Hole in construction");
					else if (relative_offset > 0)
					{
						//bool added_translations = false;
						float relative_translation_x = 0.f;
						while (offset_i != building_structure[i].end() &&
							trans(offset_j).x() + 0.5f * diag(offset_j).x() > trans(offset_i).x() - 0.5f * diag(offset_i).x())
						{
							if(contained() > epsilon)
								relative_translation_x += offset_i->second->precomputed_diag.x();
							++offset_i;
						}

						relative_translation_x *= (1 + spacing);

						layer_length[j] += relative_translation_x;
						for (auto it = offset_j; it != building_structure[j].end(); ++it)	
							it->second->obbox.translation.x() += relative_translation_x;
					}
					else //There should not be any intersection
					{
						if (contained() > epsilon)
							throw std::exception("Collision detected");
					}
					prev_offset_j = offset_j->first;
				}
				while (offset_i != building_structure[i].end())
				{
					//add blocks behind the others which are intersecting to the layer_length
					if ((offset_i->second->obbox.translation.z() + 
						0.5f * offset_i->second->precomputed_diag.z()) >= layer_bound[j].lower)
						layer_length[j] += offset_i->second->precomputed_diag.x() * (1 + spacing);
						
					++offset_i;
				}

			}
		}
		for (const auto& length : layer_length)
		{
			if (length - width > epsilon)
				throw std::exception("Unstable configuration");
			//TODO:: refactor for Gebilde 3, 4
			//Adding that layers at positions with no blocks are not stackable on top
		}
	}

	void building::adjust_z_translation()
	{
		for (size_t i = 1; i < building_structure.size(); ++i)
		{
			for (const auto& bElement : building_structure[i] | std::views::values)
			{
				bElement->obbox.translation.z() += layer_bound[i].lower + (i * z_spacing);
			}
		}
	}

	void building::calculate_width()
	{
		for (const auto& bElement : building_structure[0] | std::views::values)
		{
			width += bElement->precomputed_diag.x() * (1 + spacing);
		}
	}

	building::Extend building::get_x_extend(const building_element::Ptr& element)
	{
		const float mid = element->obbox.translation.x();
		const float half_size = 0.5f * element->precomputed_diag.x();

		return { mid - half_size, mid + half_size };
	}

	void building::create_network(const pn_net::Ptr& network)
	{
		//typedef std::map<float, building_element::Ptr>::const_iterator it;		
		

		net = network ? network : std::make_unique<pn_net>(object_parameters());
		goal = std::make_pair(net->create_place(), std::make_shared<pn_token>());
		const auto& empty_token = pn_empty_token::find_or_create(net);
		

		const auto& [bottom_up, top_down] = dependency_resolve_building();
		const auto& resources = generate_resources();

		for (const auto& layer : building_structure)
		{
			for (const auto& element : layer | std::views::values)
			{
				const auto& dependencies_it = top_down.find(element);
				
				const auto single_element = 
					std::dynamic_pointer_cast<single_building_element>(element);
				const auto& resource_top = resources.at(single_element);
				//const auto& from = resource_top.agent();

				if (dependencies_it != top_down.end())
				{
					std::vector<pn_object_instance> bottom_located_objects;

					const auto& dependencies = dependencies_it->second;
					bottom_located_objects.reserve(dependencies.size());

					for (const auto& dependency : dependencies)
					{
						const auto single_dependency = 
							std::dynamic_pointer_cast<single_building_element>(dependency);
						const auto& resource_bottom = resources.at(single_dependency);
						bottom_located_objects.emplace_back(
								resource_bottom, 
								single_dependency->token
						);
					}

					const auto& top_located_object =
						std::make_pair(resource_top, single_element->token);

					std::lock_guard<std::mutex> lock(net->mutex);

					for (const auto& agent : net->get_agent_places())
						stack_action::create(net, agent, bottom_located_objects, top_located_object, true);
				}
				else
				{
					std::lock_guard<std::mutex> lock(net->mutex);

					for (const auto& agent : net->get_agent_places()){
						const auto& transition = std::make_shared<place_action>(net, single_element->token, agent, resource_top, true);
						net->add_transition(transition);
					}
				}
			}
		}

		{
			std::vector<pn_instance> goal_side_conditions;
			for (const auto& layer : building_structure)
			{
				for (const auto& element : layer | std::views::values)
				{
					const auto& dependencies_it = bottom_up.find(element);

					//if (dependencies_it == bottom_up.end()) //no blocks above
					//{
						auto single_element = std::dynamic_pointer_cast<single_building_element>(element);
						const auto& resource = resources.at(single_element);
						goal_side_conditions.emplace_back(resource, single_element->token);
					//}
				}
			}
			std::vector<pn_instance> output;
			output.reserve(goal_side_conditions.size() + 1);
			for (const auto& condition : goal_side_conditions)
				output.emplace_back(condition);
			output.emplace_back(goal);

			net->add_transition(std::make_shared<pn_transition>(std::move(goal_side_conditions), std::move(output)));
			net->set_goal(goal.first);
		}

#ifndef _DEBUG

		for (const auto& layer : building_structure)
		{
			for (const auto& position : layer)
			{
				const auto& element = position.second;
				const auto& dependencies_it = bottom_up.find(element);

				const auto single_element = std::dynamic_pointer_cast<single_building_element>(element);
				const auto& resource_bottom = resources.at(single_element);

				if (dependencies_it != bottom_up.end())
				{
					const auto& bottom_located_object =
						std::make_pair(resource_bottom, single_element->token);

					std::vector<pn_object_instance> top_located_objects;

					const auto& dependencies = dependencies_it->second;
					top_located_objects.reserve(dependencies.size());

					for (const auto& dependency : dependencies)
					{
						const auto single_dependency = std::dynamic_pointer_cast<single_building_element>(dependency);
						const auto& resource_top = resources.at(single_dependency);
						top_located_objects.emplace_back(
							resource_top, single_dependency->token);
					}


					for(const auto& agent_place : net->get_agent_places())
						reverse_stack_action::create(net, agent_place, top_located_objects, bottom_located_object);
					
				}
				else //no blocks above
				{
					std::lock_guard<std::mutex> lock(net->mutex);

					for (const auto& agent_place : net->get_agent_places())
					{
						const auto& transition = std::make_shared<pick_action>(net, single_element->token, resource_bottom, agent_place);

						net->add_transition(transition);
					}
				}
			}
		}
#endif // !_DEBUG
	}

	building::DependencyGraph building::dependency_resolve_building() const
	{
		//there are no dependencies if there's only one layer
		if (building_structure.size() <= 1)
			return {};

		DependencyGraph out;
		std::map<float, building_element::Ptr> visible_elements;

		for (size_t i = 0; i < building_structure.size(); ++i)
		{
			const auto& layer = building_structure[i];

			if (i == 0)
			{
				///Adding lowest layer as visible_elements
				std::ranges::for_each(building_structure[0],
					[&visible_elements](
						const std::map<float, building_element::Ptr>::value_type& val)
					{visible_elements.emplace(
						val.second->obbox.translation.x() -
						val.second->precomputed_diag.x(), val.second); });
				continue;
			}

			std::map<float, building_element::Ptr> new_visible_elements;
			std::set<float> covered_elements;

			for (const auto& element : layer)
			{
				const auto& dep = dependency_resolve_element(element.second, visible_elements);
				if (dep.first != visible_elements.end())
				{
					new_visible_elements.emplace(
						element.second->obbox.translation.x() -
						//TODO:: do we need to subtract here?
						element.second->precomputed_diag.x(),
						element.second);

					//TODO: composed
					std::for_each(dep.first, dep.second,
						[&covered_elements, &out, &element]
					(const std::map<float, building_element::Ptr>::value_type& val)
						{
							covered_elements.insert(val.first);

							const auto it_bu = out.bottom_up.find(val.second);
							if (it_bu == out.bottom_up.end())
								out.bottom_up.emplace(
									val.second,
									std::set<building_element::Ptr>({ element.second }));
							else
								it_bu->second.insert(element.second);


							const auto it_td = out.top_down.find(element.second);
							if (it_td == out.top_down.end())
								out.top_down.emplace(
									element.second,
									std::set<building_element::Ptr>({ val.second }));
							else
								it_td->second.insert(val.second);
						});
				}
			}
			//remove covered elements
			for (float key : covered_elements)
				visible_elements.erase(key);

			//add new visible elements
			visible_elements.insert(
				new_visible_elements.begin(),
				new_visible_elements.end());
		}
		return out;
	}

	std::pair<building::dep_it, building::dep_it> building::dependency_resolve_element(
		const building_element::Ptr& element,
		const std::map<float, building_element::Ptr>& dependables)
	{
		std::pair<dep_it, dep_it> out = { dependables.end(), dependables.end() };

		const auto& el_x_extend = get_x_extend(element);

		for (auto it = dependables.begin(); it != dependables.end(); ++it)
		{
			const auto& depend_x_extend = get_x_extend(it->second);

			//we've surpassed the last possible dependency
			if (depend_x_extend.start >= el_x_extend.end)
			{
				//ensures that dependency is always <= element on x-axis
				out.second = it;

				//if we haven't seen a dependency until now then it's the previous one
				if (out.first == dependables.end())
					out.first = std::prev(it);
				break;
			}

			//return x2 >= y1 and y2 >= x1
			//TODO:: check logic once again (comment below)
			//checks whether the block below is fully encompassing our current element (above)
			if (out.first == dependables.end() &&
				(el_x_extend.end >= depend_x_extend.start) &&
				(depend_x_extend.end >= el_x_extend.start) &&
				//last condition may not be necessary
				(depend_x_extend.end != el_x_extend.start))

			{
				out.first = it;
			}
		}

		/*if (out.first != dependables.end())
			std::cout << std::distance(out.first, out.second)
			<< " dependencies found" << std::endl;*/

		return out;
	}

	std::map<single_building_element::Ptr, pn_boxed_place::Ptr> building::generate_resources() const
	{
		auto transform = [&](obb box)
		{
			Eigen::Affine3f pose = Eigen::Translation3f(translation) *
				rotation *
				Eigen::Translation3f(box.translation) *
				box.rotation;

			box.translation = pose.translation();
			box.rotation = pose.rotation();
			return box;
		};

		std::map<single_building_element::Ptr, pn_boxed_place::Ptr> resources;

		for (const auto& layer : building_structure)
		{
			for (const auto& element : layer)
			{
				std::vector<single_building_element::Ptr> single_elements;

				if (const auto single =
					std::dynamic_pointer_cast<single_building_element>(element.second); single)
					single_elements.push_back(single);
				else
				{
					const auto composed =
						std::dynamic_pointer_cast<composed_building_element>(element.second);
					if (!composed)
						throw std::exception("building element type not recognized");
					single_elements = composed->getElements();
				}

				for (const auto& single_element : single_elements)
				{
					auto place = net->get_place(transform(single_element->obbox));

					if (!place)
					{
						place = std::make_shared<pn_boxed_place>(transform(single_element->obbox));
						net->add_place(place);
					}

					const auto& emplaced = resources.emplace(
						single_element,
						place
					);
				}
			}
		}
		return resources;
	}


	std::map<std::string, pn_object_token::Ptr> building::building_elements;

	void building::load_building_elements(const std::vector<object_prototype::ConstPtr>& prototypes)
	{
		for (auto& it : prototypes)
			if(!building_elements.contains(it->get_name()))
				building_elements.emplace(it->get_name(), std::make_shared<pn_object_token>(it));
	}

	void building::set_building_elements(std::map<std::string, pn_object_token::Ptr> elements)
	{
		building_elements = std::move(elements);
	}

	const std::vector<std::map<int, building_element::Ptr>>& building::get_building_structure() const
	{
		return building_structure;
	}

	/*
	 * @class building_estimation
	 *
	 * Runtime class keeping track of developments inside the
	 * working space
	 * Keeps track of possible building arrangements
	 */

	building_estimation::building_estimation(enact_core::world_context& world,
		const std::vector<object_prototype::ConstPtr>& prototypes) : world(world), prototypes(prototypes)
	{

	}

	building_estimation::~building_estimation() noexcept
	{
		stop_thread();
	}

	bool building_estimation::is_workspace_object(const pc_segment::Ptr& object) const
	{
		return (object->centroid.x < -0.02);
	}

	/*float building_estimation::distance(const pc_segment::PointT& centroid_0, const pc_segment::PointT& centroid_1) const
	{
		auto min = [](const pc_segment::PointT& centroid)
		{
			return std::min(centroid.x, centroid.y, centroid.z);
		};

		return 0.0f;
	}*/

	update_state building_estimation::update(
		const pc_segment::Ptr& seg, const strong_id& s_id)
	{
		{
			auto it = classified_shape.find(s_id);
			if (it != classified_shape.end())
			{
				if (it->second.classification_results.front().prototype != 
					seg->classification_results.front().prototype)
				{
					it->second = *seg;
					segments.at(s_id) = seg->centroid;
					return update_state::SHAPE_CHANGED;
				}
			}
		}

		auto it = segments.find(s_id);
		if (it != segments.end())
		{
			float displacement = pow(it->second.x - seg->centroid.x, 2);
			displacement += pow(it->second.y - seg->centroid.y, 2);
			displacement += pow(it->second.z - seg->centroid.z, 2);
			displacement = sqrt(displacement);

			it->second = seg->centroid;
			
			return (displacement > displacement_tolerance) 
				? update_state::RELOCATED 
				: update_state::NO_CHANGE;
		}
		else
		{
			segments.emplace(s_id, seg->centroid);
			classified_shape.emplace(s_id, *seg);
			//id_mapping.emplace(str_id, seg);
			return update_state::NEWLY_DETECTED;
		}
	}

	void building_estimation::update(const strong_id& id, enact_priority::operation op)
	{
		if (op == enact_priority::operation::DELETED)
		{
			weak_id w_id(id);

			schedule([this, w_id]()
				{
					bool deleted = false;

					{
						auto it = segments.find(w_id);
						if (it != segments.end())
						{
							deleted = true;
							segments.erase(it);
						}
					}

					{
						auto it = classified_shape.find(w_id);
						if (it != classified_shape.end())
						{
							deleted = true;
							classified_shape.erase(it);
						}
					}

					if (deleted)
						std::cout << "Deleted" << std::endl;
				});
		}
		else if (op == enact_priority::operation::UPDATE)
		{
			schedule([this, id]() {

				if (!id)
					return;

				pc_segment::Ptr seg;
				bool background;
				{
					enact_core::lock l(
						world, enact_core::lock_request(
							id, 
							object_instance::aspect_id, 
							enact_core::lock_request::read)
					);
					enact_core::const_access<object_instance_data> 
						access_object(l.at(id, object_instance::aspect_id));
					const object_instance& obj = access_object->payload;
					seg = obj.get_classified_segment();
					background = obj.is_background();
				}

				if (seg && is_workspace_object(seg))
				{
					auto result = seg->classification_results.begin();

					if (background)
					{
						if (seg->classification_results.size() == 1)
							return;
						else ++result;
					}

					if (result->local_certainty_score < 0.2f)
						return;

					update_state state = update(seg, id);
					if (state == update_state::NO_CHANGE)
						return;

					std::string pre_message = "";

					switch (state)
					{
					case update_state::NEWLY_DETECTED:
						pre_message = "Detected: ";
						break;
					case update_state::RELOCATED:
						pre_message = "Relocation of: ";
						break;
					case update_state::SHAPE_CHANGED:
						pre_message = "Shape changed: ";
						break;
					}

					std::cout << pre_message 
						<< result->prototype->get_name() << " \t@" 
						<< seg->centroid.x << ":\t" << seg->centroid.y << ":\t" << seg->centroid.z
						<< " Certainty: " << result->local_certainty_score 
						<< std::endl;
				}
				});
		}
	}

	Position::Position(int layer, int x_offset)
		: layer(layer), x_offset(x_offset)
	{
		if (layer < 0 || x_offset < 0)
			throw std::exception("Invalid position");
	}

	int Position::Layer() const
	{
		return layer;
	}

	int Position::xOffset() const
	{
		return x_offset;
	}

	/*agent::agent(const std::map<object_prototype::ConstPtr, pn_object_token::Ptr>& proto_token)
		//: prototype_token(proto_token)
	{
	}
	
	agent::agent(const std::vector<pn_agent_place::Ptr>& agent_places)
		: agent_places(agent_places)
	{}*/

	/*agent::agent(const pn_net::Ptr& net)
		:
		agent_places(
			[&]()
			{
				std::vector<pn_agent_place::Ptr> agent_places;
				for (const auto& place : net->get_places())
				{
					auto dynamic = std::dynamic_pointer_cast<pn_agent_place>(place);
					if (dynamic)
						agent_places.emplace_back(dynamic);
				}
				return agent_places;
			}()
		)
	{	
		std::list<pn_transition::Ptr> reachable;

		for (const auto& transition : net->get_transitions())
		{
			if (std::dynamic_pointer_cast<place_action>(transition) ||
				std::dynamic_pointer_cast<stack_action>(transition))
			{
				if (transition->get_side_conditions().size() == 0)
					reachable.emplace_back(transition);
				constructive_transitions.emplace_back(transition);
			}
			else if
			   (std::dynamic_pointer_cast<pick_action>(transition) ||
			    std::dynamic_pointer_cast<reverse_stack_action>(transition))
				destructive_transitions.emplace_back(transition);
		}

		for (auto& reach : reachable)
		{
			const auto& temp = std::dynamic_pointer_cast<place_action>(reach);
			if (temp)
			{
				for (auto& input : temp->inputs)
				{
					//input.
				}
			}
		}
	}

	const std::vector<pn_transition::Ptr>& agent::get_constructive_transitions() const
	{
		return constructive_transitions;
	}

	const std::vector<pn_agent_place::Ptr>& agent::get_agent_places() const
	{
		return agent_places;
	}

	/*const std::vector<pn_transition>& agent::get_place_transitions() const
	{
		for (const auto& agent_place : agent_places)
		{
			agent_place->get_transitions
		}
	}*/

	/*observed_object::observed_object(
		const obb& obb, 
		const object_prototype::ConstPtr& object)
		: boxed_place(std::make_shared<pn_boxed_place>(obb)),
		object(object)
	{
		agent_place = std::make_shared<pn_agent_place>(boxed_place, object);
	}

	const pn_agent_place::Ptr& observed_object::agent()
	{
		return agent_place;
	}

	const pn_agent_place::Ptr& observed_object::agent() const
	{
		return agent_place;
	}
	*/
	builder& builder::add_single_element(const std::string& type, Position pos)
	{
		auto element_ptr = std::make_shared<single_building_element>(type);
		elements.emplace_back(element_ptr, pos);

		return *this;
	}

	building::Ptr builder::create_building(Eigen::Vector3f translation, Eigen::Quaternionf rotation, const pn_net::Ptr& net)
	{
		return std::make_shared<building>(elements, translation, rotation, net);
	}

	/*void agent::place(const object_prototype::ConstPtr& object, const obb& box)
	{
		for (const auto& action : )
	}*/
}

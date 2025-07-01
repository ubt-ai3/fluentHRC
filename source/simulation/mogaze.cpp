#include "mogaze.hpp"

#include <ranges>
#include <boost/algorithm/string.hpp>

using namespace state_observation;

namespace simulation
{
#undef RGB
namespace mogaze
{
/////////////////////////////////////////////////////////////
//
//
//  Class: object_prototype_loader
//
//
/////////////////////////////////////////////////////////////
std::vector<state_observation::object_prototype::Ptr> object_prototype_loader::generate_default_prototypes()
{
	std::vector<object_prototype::Ptr> result;

	aabb plate(Eigen::Vector3f(0.183f, 0.183f, 0.00716f));
	aabb bowl(Eigen::Vector3f(0.283f, 0.375f, 0.106f));
	aabb cup(Eigen::Vector3f(0.0801f, 0.0801f, 0.0708f));
	aabb jug(Eigen::Vector3f(0.118f, 0.153f, 0.289f));

	pcl::RGB red(229, 0, 0);
	pcl::RGB pink(229, 105, 180);
	pcl::RGB blue(0, 0, 229);
	pcl::RGB green(0, 229, 0);
	pcl::RGB olive(26, 128, 26);
	pcl::RGB wooden(255, 76, 26);

	pcl::RGB brown(128, 64, 0);
	pcl::RGB birch(255, 204, 102);
	pcl::RGB white(229, 229, 229);

	const std::string mesh_loc("assets/mogaze/meshfiles/");
	auto plate_mesh = std::make_shared<mesh_wrapper>(mesh_loc + "plate.obj");
	auto bowl_mesh = std::make_shared<mesh_wrapper>(mesh_loc + "bowl.obj");
	auto cup_mesh = std::make_shared<mesh_wrapper>(mesh_loc + "cup.obj");
	auto jug_mesh = std::make_shared<mesh_wrapper>(mesh_loc + "jug.obj");

	//cups

	result.push_back(
		std::make_shared<mogaze_object>(
		red,
		cup_mesh,
		"cup_red",
		"cup"));

	result.push_back(
		std::make_shared<mogaze_object>(
		pink,
		cup_mesh,
		"cup_pink",
		"cup"));

	result.push_back(
		std::make_shared<mogaze_object>(
		green,
		cup_mesh,
		"cup_green",
		"cup"));

	result.push_back(
		std::make_shared<mogaze_object>(
		blue,
		cup_mesh,
		"cup_blue",
		"cup"));

	// plates
	result.push_back(
		std::make_shared<mogaze_object>(
		red,
		plate_mesh,
		"plate_red",
		"plate"));

	result.push_back(
		std::make_shared<mogaze_object>(
		pink,
		plate_mesh,
		"plate_pink",
		"plate"));

	result.push_back(
		std::make_shared<mogaze_object>(
		green,
		plate_mesh,
		"plate_green",
		"plate"));

	result.push_back(
		std::make_shared<mogaze_object>(
		blue,
		plate_mesh,
		"plate_blue",
		"plate"));

	result.push_back(
		std::make_shared<mogaze_object>(
		olive,
		jug_mesh,
		"jug",
		"jug"));

	result.push_back(
		std::make_shared<mogaze_object>(
		white,
		bowl_mesh,
		"bowl",
		"bowl"));

	return result;
}

object_prototype_loader::object_prototype_loader()
{
	prototypes = generate_default_prototypes();
}

/////////////////////////////////////////////////////////////
//
//
//  Class: furnishing
//
//
/////////////////////////////////////////////////////////////

furnishing::furnishing(const std::string& mesh_path, const obb& pose, pcl::RGB color)
{
	const auto tmp_mesh = std::make_shared_for_overwrite<pcl::PolygonMesh>();
	if (pcl::io::load(mesh_path, *tmp_mesh))
		throw std::runtime_error("Could not load polygon file");

	mesh = pointcloud_preprocessing::color(
		pointcloud_preprocessing::transform(tmp_mesh, Eigen::Affine3f(Eigen::Translation3f(pose.translation) * pose.rotation)),
		color);

}

void furnishing::render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp)
{

	scene.add(std::make_shared<pcl::simulation::TriangleMeshModel>(
		std::make_shared<pcl::PolygonMesh>(*mesh)
	));

}

void furnishing::render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport)
{
	auto id = std::to_string(reinterpret_cast<size_t>(this));
	if (!viewer.contains(id))
	{
		viewer.addPolygonMesh(*mesh, id);
	}
}

/////////////////////////////////////////////////////////////
//
//
//  Class: action
//
//
/////////////////////////////////////////////////////////////

action::action(unsigned int timestamp,
	bool pick,
	object_prototype::ConstPtr object,
	obb pose)
	:
	// The timestamp format in the mogaze data set is specified as an integer with a resolution of 1/100 s,
	// we convert it to the base unit of 1 s here
	timestamp(timestamp * 1e-2),
	pick(pick),
	object(std::move(object)),
	pose(std::move(pose))
{
}

/////////////////////////////////////////////////////////////
//
//
//  Class: mogaze_object
//
//
/////////////////////////////////////////////////////////////

mogaze_object::mogaze_object(const pcl::RGB& mean_color, const state_observation::mesh_wrapper::Ptr& base_mesh, const std::string& name, const std::string& type)
	:
	object_prototype(compute_bounding_box(*base_mesh->load_mesh()), mean_color, base_mesh, name, type)
{
	object_prototype_mesh = pointcloud_preprocessing::color(
		pointcloud_preprocessing::transform(base_mesh->load_mesh(), Eigen::Affine3f(Eigen::Scaling(2.f * bounding_box.diagonal.cwiseInverse()) * Eigen::Translation3f(-bounding_box.translation)))
		, mean_color);
}

state_observation::aabb mogaze_object::compute_bounding_box(const pcl::PolygonMesh& mesh)
{
	int field_count = mesh.cloud.point_step / 4;

	if (field_count < 3)
		throw std::exception("Invalid polygon mesh format");

	Eigen::Vector3f min = std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones();
	Eigen::Vector3f max = -std::numeric_limits<float>::infinity() * Eigen::Vector3f::Ones();



	int size = mesh.cloud.width * mesh.cloud.height;
	int in_pstep = mesh.cloud.point_step;


	for (int p_idx = 0; p_idx < size; ++p_idx) {
		auto p = reinterpret_cast<const float*>(&mesh.cloud.data[in_pstep * p_idx]);
		for (int field_offset = 0; field_offset < 3; ++field_offset) {
			float val = p[field_offset];
			min(field_offset) = std::min(min(field_offset), val);
			max(field_offset) = std::max(max(field_offset), val);
		}
	}

	return { max - min, 0.5f * max + 0.5f * min };
}

/////////////////////////////////////////////////////////////
//
//
//  Class: predicate
//
//
/////////////////////////////////////////////////////////////
const Eigen::AlignedBox3f predicate::table = Eigen::AlignedBox3f(Eigen::Vector3f(0, 1.7, 0.6), Eigen::Vector3f(1.2, 2.7, 1.2));
const Eigen::AlignedBox3f predicate::laiva_shelf = Eigen::AlignedBox3f(Eigen::Vector3f(-1.3, -0.5, 0), Eigen::Vector3f(-1, 0.3, 2.5));
const Eigen::AlignedBox3f predicate::vesken_shelf = Eigen::AlignedBox3f(Eigen::Vector3f(0.4, -0.5, 0), Eigen::Vector3f(0.9, -0.1, 1.5));


std::set<state_observation::pn_token::Ptr> predicate::find_matching_tokens(const std::string& substring, const traces& token_traces)
{
	std::set<state_observation::pn_token::Ptr> relevant_tokens;
	for (const auto& [prototype, token] : token_traces)
		if (prototype->get_name().find(substring) != prototype->get_name().npos)
			relevant_tokens.emplace(token);

	return relevant_tokens;
}

predicate::Ptr predicate::all_types_in_region(const std::string& type, const Eigen::AlignedBox3f& region, const traces& token_traces)
{
	return std::make_shared<pred_all_in_region>(find_matching_tokens(type, token_traces), region);
}

predicate::Ptr predicate::all_colors_in_region(const std::string& color, const Eigen::AlignedBox3f& region, const traces& token_traces)
{
	return std::make_shared<pred_all_in_region>(find_matching_tokens(color, token_traces), region);
}

predicate::Ptr predicate::count_types_in_region(int n, const std::string& type, const Eigen::AlignedBox3f& region, const traces& token_traces)
{
	return std::make_shared<pred_count_in_region>(n, find_matching_tokens(type, token_traces), region);
}

predicate::Ptr predicate::bowl_or_jug_on_table(bool placed, const traces& token_traces)
{
	std::set<state_observation::pn_token::Ptr> relevant_tokens;
	for (const auto& entry : token_traces)
		if (!entry.first->get_name().compare("jug") || !entry.first->get_name().compare("bowl"))
			relevant_tokens.emplace(entry.second);

	return std::make_shared<pred_bowl_or_jug_on_table>(placed,std::move(relevant_tokens));
}

predicate::predicate(std::set<state_observation::pn_token::Ptr> relevant_tokens, Eigen::AlignedBox3f region)
	:
	relevant_tokens(std::move(relevant_tokens)),
	region(std::move(region))
{
}

bool predicate::in_region(const state_observation::pn_boxed_place::Ptr& place) const
{
	const auto& center = place->box.translation;
	return region.contains(center);
}

std::set<state_observation::pn_transition::Ptr> predicate::get_outgoing(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	std::set<state_observation::pn_transition::Ptr> result;

	for (const auto& place : marking->net.lock()->get_places()) {
		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place);

		if (!boxed_place)
			continue;

		if (!in_region(boxed_place))
			continue;

		for (const auto& w_trans : boxed_place->get_outgoing_transitions()) {
			auto transition = w_trans.lock();
			if (relevant_tokens.contains(transition->inputs.begin()->second))
				result.emplace(transition);
		}
	}

	return result;
}

std::set<state_observation::pn_transition::Ptr> predicate::get_incoming(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	std::set<state_observation::pn_transition::Ptr> result;

	for (const auto& place : marking->net.lock()->get_places()) {
		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place);

		if (!boxed_place)
			continue;

		if (!in_region(boxed_place))
			continue;

		for (const auto& w_trans : boxed_place->get_incoming_transitions()) {
			auto transition = w_trans.lock();
			if (relevant_tokens.contains(transition->outputs.begin()->second)) {
				result.emplace(transition);
			}
		}
	}

	return result;
}

/////////////////////////////////////////////////////////////
//
//
//  Class: pred_all_in_region
//
//
/////////////////////////////////////////////////////////////

pred_all_in_region::pred_all_in_region(std::set<state_observation::pn_token::Ptr> relevant_tokens, Eigen::AlignedBox3f region)
	:
	predicate(std::move(relevant_tokens), std::move(region))
{
}

bool pred_all_in_region::operator()(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	for (const auto& instance : marking->distribution) {
		if (!relevant_tokens.contains(instance.second))
			continue;

		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);
		if (!boxed_place || !in_region(boxed_place))
			return false;
	}

	return true;
}

std::set<state_observation::pn_transition::Ptr> pred_all_in_region::get_blocked(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	return get_outgoing(marking);
}

std::set<state_observation::pn_transition::Ptr> pred_all_in_region::get_feasible(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	int count = 0;
	for (const auto& instance : marking->distribution) {
		if (!relevant_tokens.contains(instance.second))
			continue;

		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);
		if (boxed_place && in_region(boxed_place))
			count++;
	}

	if (count == relevant_tokens.size())
		return {};

	return get_incoming(marking);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: pred_count_in_region
//
//
/////////////////////////////////////////////////////////////

pred_count_in_region::pred_count_in_region(int count, std::set<state_observation::pn_token::Ptr> relevant_tokens, Eigen::AlignedBox3f region)
	:
	predicate(std::move(relevant_tokens), std::move(region)),
	count(count)
{
}

int pred_count_in_region::current_count(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	int count = 0;
	for (const auto& instance : marking->distribution) {
		if (!relevant_tokens.contains(instance.second))
			continue;

		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);
		if (boxed_place && in_region(boxed_place))
			count++;
	}

	return count;
}


bool pred_count_in_region::operator()(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	return current_count(marking) == count;
}

std::set<state_observation::pn_transition::Ptr> pred_count_in_region::get_blocked(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	const int current = current_count(marking);

	if (current < count)
		return get_outgoing(marking);
	else if (current > count)
		return get_incoming(marking);
	else {
		std::set<state_observation::pn_transition::Ptr> result = get_outgoing(marking);
		const auto& other = get_incoming(marking);
		result.insert(other.begin(), other.end());
		return result;
	}
}

std::set<state_observation::pn_transition::Ptr> pred_count_in_region::get_feasible(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	const int current = current_count(marking);

	if (current < count)
		return get_incoming(marking);
	else if (current > count)
		return get_outgoing(marking);
	else
		return {};
}



/////////////////////////////////////////////////////////////
//
//
//  Class: pred_bowl_or_jug_on_table
//
//
/////////////////////////////////////////////////////////////

pred_bowl_or_jug_on_table::pred_bowl_or_jug_on_table(bool placed, std::set<state_observation::pn_token::Ptr> relevant_tokens)
	:
	predicate(std::move(relevant_tokens), table),
	placed(placed)
{
}

bool pred_bowl_or_jug_on_table::operator()(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	for (const auto& instance : marking->distribution) {
		if (!relevant_tokens.contains(instance.second))
			continue;

		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);
		if (boxed_place && in_region(boxed_place))
			return placed;
	}

	return !placed;
}

std::set<state_observation::pn_transition::Ptr> pred_bowl_or_jug_on_table::get_blocked(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	return placed ? get_outgoing(marking) : get_incoming(marking);
}

std::set<state_observation::pn_transition::Ptr> pred_bowl_or_jug_on_table::get_feasible(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	for (const auto& instance : marking->distribution)
		if (relevant_tokens.contains(instance.second)) {
			auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);

			if (boxed_place && in_region(boxed_place))
				return placed ? std::set<state_observation::pn_transition::Ptr>() : get_outgoing(marking);
		}

	return placed ? get_incoming(marking) : std::set<state_observation::pn_transition::Ptr>();
}

/////////////////////////////////////////////////////////////
//
//
//  Class: task_execution
//
//
/////////////////////////////////////////////////////////////

const std::map<std::string, Eigen::Vector3f> task_execution::object_dimensions =
{
	{"plate",Eigen::Vector3f(.183f, 0.183f, 0.0071f)},
	{"bowl",Eigen::Vector3f(0.283f, 0.375f, 0.106f)},
	{"cup",Eigen::Vector3f(0.0826f,0.0708f,0.0801f)},
	{"jug",Eigen::Vector3f(0.118f, 0.153f, 0.289f)}
};

std::vector<std::vector<predicate::Ptr>> task_execution::get_instruction_predicates()
{
	return std::vector<std::vector<predicate::Ptr>>({
		//0 Set the table for 1 person
		std::vector<predicate::Ptr>({
			predicate::count_types_in_region(1, "plate", predicate::table, token_traces),
			predicate::count_types_in_region(1, "cup", predicate::table, token_traces),
			predicate::bowl_or_jug_on_table(true, token_traces)
}),
//1 Set the table for 2 persons
		std::vector<predicate::Ptr>({
			predicate::count_types_in_region(2, "plate", predicate::table, token_traces),
			predicate::count_types_in_region(2, "cup", predicate::table, token_traces),
			predicate::bowl_or_jug_on_table(true, token_traces)
}),
//2 Set the table for 3 persons
		std::vector<predicate::Ptr>({
			predicate::count_types_in_region(3, "plate", predicate::table, token_traces),
			predicate::count_types_in_region(3, "cup", predicate::table, token_traces),
			predicate::bowl_or_jug_on_table(true, token_traces)
}),
//3 Set the table for 4 persons
		std::vector<predicate::Ptr>({
			predicate::count_types_in_region(4, "plate", predicate::table, token_traces),
			predicate::count_types_in_region(4, "cup", predicate::table, token_traces),
			predicate::bowl_or_jug_on_table(true, token_traces)
}),
//4 Clear table
		std::vector<predicate::Ptr>({
			predicate::count_types_in_region(0, "plate", predicate::table, token_traces),
			predicate::count_types_in_region(0, "cup", predicate::table, token_traces),
			predicate::bowl_or_jug_on_table(false, token_traces)
}),
//5 Put the jug and the bowl on small shelf
		std::vector<predicate::Ptr>({
			predicate::all_types_in_region("jug", predicate::vesken_shelf, token_traces),
			predicate::all_types_in_region("bowl", predicate::vesken_shelf, token_traces)
}),
//6 Put all cups on small shelf
		std::vector<predicate::Ptr>({
			predicate::all_types_in_region("cup", predicate::vesken_shelf, token_traces)
}),
//7 Put blue and pink objects on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_colors_in_region("blue", predicate::laiva_shelf, token_traces),
			predicate::all_colors_in_region("pink", predicate::laiva_shelf, token_traces)
}),
//8 Put blue and red objects on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_colors_in_region("blue", predicate::laiva_shelf, token_traces),
			predicate::all_colors_in_region("red", predicate::laiva_shelf, token_traces)
}),
//9 Put blue and green objects on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_colors_in_region("blue", predicate::laiva_shelf, token_traces),
			predicate::all_colors_in_region("green", predicate::laiva_shelf, token_traces)
}),
//10 Put pink and red objects on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_colors_in_region("red", predicate::laiva_shelf, token_traces),
			predicate::all_colors_in_region("pink", predicate::laiva_shelf, token_traces)
}),
//11 Put pink and green objects on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_colors_in_region("green", predicate::laiva_shelf, token_traces),
			predicate::all_colors_in_region("pink", predicate::laiva_shelf, token_traces)
}),
//12 Put red and green objects on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_colors_in_region("red", predicate::laiva_shelf, token_traces),
			predicate::all_colors_in_region("green", predicate::laiva_shelf, token_traces)
}),
//13 Put all cups on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_types_in_region("cup", predicate::laiva_shelf, token_traces)
}),
//14 Put bowl and jug on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_types_in_region("jug", predicate::laiva_shelf, token_traces),
			predicate::all_types_in_region("bowl", predicate::laiva_shelf, token_traces)
}),
//15 Put all cups on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_types_in_region("cup", predicate::laiva_shelf, token_traces)
}),
//16 Put all plates on big shelf
		std::vector<predicate::Ptr>({
			predicate::all_types_in_region("plate", predicate::laiva_shelf, token_traces)
})
		});

}

task_execution::task_execution(const object_parameters& object_params, int person)
	:
	net(std::make_shared<pn_net>(object_params)),
	human(net->create_place(true)),
	timestamp(0.f),
	task_id(-1)
{
	const auto path = std::string("assets/mogaze/") + std::to_string(person) + '/';

	for (const auto& obj : loader.get_prototypes()) {
		const auto iter = instances.find(obj->get_type());
		if (iter == instances.end())
			instances.emplace(obj->get_type(), std::vector<object_prototype::ConstPtr>({ obj }));
		else
			iter->second.push_back(obj);

		auto token = std::make_shared<pn_object_token>(obj);
		token_traces.emplace(obj, token);
	}
	instruction_predicates = get_instruction_predicates();

	auto iterate_lines = [&](const std::string& path, std::function<void(std::vector<std::string>)>&& callback) {
		std::ifstream stream(path);

		if (!stream.is_open())
			throw std::exception(path.c_str());

		std::string line;
		while (std::getline(stream, line)) {
			std::vector<std::string> entries;
			boost::split(entries, line, [](char c) { return c == ','; });

			callback(std::move(entries));
		}
	};


	iterate_lines("assets/mogaze/locations.csv", [&](std::vector<std::string> entries) {
		auto pose = get_box(entries, 1);

		auto dimension_iter = object_dimensions.find(entries[0]);
		if (dimension_iter != object_dimensions.end())
		{
			pose.diagonal = dimension_iter->second;
			auto place = std::make_shared<pn_boxed_place>(pose);
			auto iter = locations.find(entries[0]);
			if (iter == locations.end())
				locations.emplace(entries[0], std::vector<pn_boxed_place::Ptr>({ place }));
			else
				iter->second.push_back(place);

			net->add_place(place);
			for (const auto& instance : instances.at(entries[0])) {
				net->add_transition(std::make_shared<pick_action>(token_traces.at(instance), place, human));
				net->add_transition(std::make_shared<place_action>(token_traces.at(instance), human, place));
			}
		}
	});

	iterate_lines(path + "init_locations.csv", [&](std::vector<std::string> entries) {
		obb pose = get_box(entries, 1);

		std::vector<std::string> name;
		boost::split(name, entries[0], [](char c) { return c == '_'; });

		std::string type = name.front();

		auto dimension_iter = object_dimensions.find(type);
		if (dimension_iter == object_dimensions.end())
		{
			// furnishing
			furnishings.push_back(std::make_shared<furnishing>(std::string("assets/mogaze/meshfiles/") + entries[0] + ".obj", pose, pcl::RGB(255, 255, 255)));
		}
		else
		{
			auto prototype = loader.get(entries[0]);
			movable_objects.emplace(
				prototype,
				std::make_shared<movable_object>(
				prototype,
				pose.translation,
				pose.rotation,
				std::make_pair(get_location(*prototype, pose), token_traces.at(prototype))
			));


		}
	});


	iterate_lines(path + "actions.csv", [&](std::vector<std::string> entries) {
		actions.emplace(std::stoi(entries[0]), entries[1] == "pick", loader.get(entries[2]), get_box(entries, 3));
	});

	iterate_lines(path + "instructions.csv", [&](std::vector<std::string> entries) {
		instructions.emplace(std::chrono::duration<float>(std::stoi(entries[0]) * 0.01f), std::stoi(entries[1]), entries[2]);
	});

	task_id = instructions.front().id;
	std::cout << "current task " << task_id << ": " << instructions.front().text << std::endl;
	instructions.pop();
}

void task_execution::render(pcl::simulation::Scene& scene)
{
	for (const auto& f : furnishings)
		f->render(scene, timestamp);

	for (const auto& obj : movable_objects | std::views::values)
		obj->render(scene, timestamp);
}

void task_execution::render(pcl::visualization::PCLVisualizer& viewer)
{
	for (const auto& f : furnishings)
		f->render(viewer, timestamp);

	for (const auto& obj : movable_objects | std::views::values)
		obj->render(viewer, timestamp);
}

pn_binary_marking::Ptr task_execution::get_marking() const
{
	std::set<pn_instance> distribution;
	for (const auto& obj : movable_objects | std::views::values)
		distribution.emplace(obj->instance);

	return std::make_shared<pn_binary_marking>(net, std::move(distribution));

}

action task_execution::peek_next_action() const
{
	if (actions.empty())
		return { 0, false, nullptr, obb() };

	return actions.front();
}

pn_transition::Ptr task_execution::next()
{
	if (actions.empty())
		return nullptr;

	const auto action = actions.front();
	actions.pop();
	timestamp = action.timestamp;

	if (!instructions.empty() && !actions.empty() && actions.front().timestamp >= instructions.front().time && actions.front().pick) {
		task_id = instructions.front().id;
		std::cout << "current task " << task_id << ": " << instructions.front().text << std::endl;
		instructions.pop();
		
	}

	return execute(action);
}

pn_transition::Ptr task_execution::execute(const action& a)
{
	auto token = token_traces.at(a.object);
	const auto mov_obj = movable_objects.at(a.object);

	pn_transition::Ptr transition = nullptr;

	const auto marking = get_marking();

	pn_place::Ptr place = get_location(*a.object, a.pose);
	const auto& transitions = a.pick ? mov_obj->instance.first->get_outgoing_transitions() : place->get_incoming_transitions();

	for (const auto& trans : transitions)
		if (marking->is_enabled(trans.lock()))
		{
			transition = trans.lock();
			break;
		}

	if (!transition) {
		for (const auto& trans : transitions)
			if (trans.lock()->outputs.begin()->second == token)
			{
				transition = trans.lock();
				break;
			}
	}

	if (a.pick)
	{
		mov_obj->center = Eigen::Vector3f(1000, 1000, 1000); //move out of view
		mov_obj->instance = std::make_pair(human, token);
	}
	else
	{
		mov_obj->center = a.pose.translation;
		mov_obj->instance = std::make_pair(place, token);
	}

	if (transition)
		executed_transitions.push_back(transition);

	return transition;
}

std::vector<pn_transition::Ptr> task_execution::get_action_candidates() const
{
	return get_action_candidates(get_marking());
}

std::vector<pn_transition::Ptr> task_execution::get_action_candidates(const pn_binary_marking::ConstPtr& marking) const
{
	std::vector<pn_transition::Ptr> all_candidates;
	std::vector<pn_transition::Ptr> feasible_candidates;

	if (task_id < 0)
		return all_candidates;

	std::set<pn_transition::Ptr> blocked_by_instruction;
	std::set<pn_transition::Ptr> enforced_by_instruction;
	for (const auto& pred : instruction_predicates.at(task_id))
	{
		auto blocked = pred->get_blocked(marking);
		blocked_by_instruction.insert(blocked.begin(), blocked.end());
		auto enforced = pred->get_feasible(marking);
		enforced_by_instruction.insert(enforced.begin(), enforced.end());
	}

	//if (task_id == 6)
	//{
	//	auto bowl = movable_objects.at(predicate::name_to_token.at("bowl")->object);
	//	if (predicate::vesken_shelf.contains(bowl->center))
	//		enforced_by_instruction.emplace(bowl->instance.first->get_outgoing_transitions().front());
	//}

	pn_instance instance = std::make_pair(human, nullptr);
	for (const auto& inst : marking->distribution)
		if (inst.first == human) {
			instance = inst;
			break;
		}

	auto is_candidate = [&](const pn_transition::Ptr& t) {
		return marking->is_enabled(t) &&
			/*!is_blocked(marking, t) &&*/
			!blocked_by_instruction.contains(t);
	};

	if (instance.second) {
		// human carries object, search for place to put it down

		for (const auto& w_t_out : human->get_outgoing_transitions())
		{
			const auto& t_out = w_t_out.lock();
			if (!is_candidate(t_out))
				continue;

			all_candidates.emplace_back(t_out);

			if (enforced_by_instruction.contains(t_out))
				feasible_candidates.emplace_back(t_out);
		}
	}
	else {
		// pick object
		if (enforced_by_instruction.empty())
			return all_candidates;

		std::set<pn_token::Ptr> objects_to_pick;
		for (const auto& enforced : enforced_by_instruction) {
			if (enforced->outputs.begin()->first == human) {
				if (is_candidate(enforced))
					feasible_candidates.emplace_back(enforced);
			}
			else {
				objects_to_pick.emplace(enforced->inputs.begin()->second);
			}
		}

		if (!feasible_candidates.empty())
			return feasible_candidates;

		for (const auto& w_t_in : human->get_incoming_transitions())
		{
			const auto& t_in = w_t_in.lock();
			if (!is_candidate(t_in))
				continue;

			if (!objects_to_pick.contains(t_in->inputs.begin()->second))
				all_candidates.emplace_back(t_in);
			else
				feasible_candidates.emplace_back(t_in);
		}
	}

	return feasible_candidates.empty() ? all_candidates : feasible_candidates;
}

std::set<state_observation::pn_transition::Ptr> task_execution::all_feasible_candidates(const state_observation::pn_binary_marking::ConstPtr& marking) const
{
	std::set<pn_transition::Ptr> feasible_candidates;
	std::set<pn_transition::Ptr> all_candidates;

	if (task_id < 0)
		return feasible_candidates;

	std::set<pn_transition::Ptr> blocked_by_instruction;
	std::set<pn_transition::Ptr> enforced_by_instruction;
	std::set<pn_token::Ptr> relevant_tokens;
	for (const auto& pred : instruction_predicates.at(task_id))
	{
		auto blocked = pred->get_blocked(marking);
		blocked_by_instruction.insert(blocked.begin(), blocked.end());
		auto enforced = pred->get_feasible(marking);
		enforced_by_instruction.insert(enforced.begin(), enforced.end());

		relevant_tokens.insert(pred->relevant_tokens.begin(), pred->relevant_tokens.end());
	}

	//if (task_id == 6)
	//{
	//	auto bowl = movable_objects.at(predicate::name_to_token.at("bowl")->object);
	//	if (predicate::vesken_shelf.contains(bowl->center)) {
	//		enforced_by_instruction.emplace(bowl->instance.first->get_outgoing_transitions().front());
	//		relevant_tokens.insert(predicate::name_to_token.at("bowl"));
	//	}
	//}

	pn_instance instance = std::make_pair(human, nullptr);
	for (const auto& inst : marking->distribution)
		if (inst.first == human) {
			instance = inst;
			break;
		}

	auto is_candidate = [&](const pn_transition::Ptr& t) {
		return !blocked_by_instruction.contains(t) &&
			relevant_tokens.contains(t->inputs.begin()->second);
	};

	if (instance.second) {
		// human carries object, search for place to put it down

		for (const auto& w_t_out : human->get_outgoing_transitions())
		{
			const auto& t_out = w_t_out.lock();
			if (!is_candidate(t_out) || *t_out->inputs.begin() != instance)
				continue;

			all_candidates.emplace(t_out);

			if (enforced_by_instruction.contains(t_out))
				feasible_candidates.emplace(t_out);
		}
	}
	else {
		// pick object
		if (enforced_by_instruction.empty())
			return all_candidates;

		std::set<pn_token::Ptr> objects_to_pick;
		for (const auto& enforced : enforced_by_instruction) {
			if (enforced->outputs.begin()->first == human) {
				if (is_candidate(enforced))
					feasible_candidates.emplace(enforced);
			}
			else {
				objects_to_pick.emplace(enforced->inputs.begin()->second);
			}
		}

		if (!feasible_candidates.empty())
			return feasible_candidates;

		for (const auto& w_t_in : human->get_incoming_transitions())
		{
			const auto& t_in = w_t_in.lock();
			if (!is_candidate(t_in))
				continue;

			if (!objects_to_pick.contains(t_in->inputs.begin()->second))
				all_candidates.emplace(t_in);
			else
				feasible_candidates.emplace(t_in);
		}
	}

	return feasible_candidates.empty() ? all_candidates : feasible_candidates;
}

bool task_execution::is_blocked(const state_observation::pn_binary_marking::ConstPtr& marking, const state_observation::pn_transition::Ptr& transition)
{
	auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(transition->outputs.begin()->first);

	if (!boxed_place)
		return false;

	for (const auto& other : boxed_place->overlapping_places)
		if (marking->is_occupied(other.lock()))
			return true;

	return false;
}


state_observation::obb task_execution::get_box(const std::vector<std::string>& entries, int start)
{
	if (entries.size() - start < 3)
		throw std::exception("Line has insufficient entries");

	Eigen::Vector3f position;
	int i = 0;
	int index = start;
	position(i++) = std::stof(entries[index++]);
	position(i++) = std::stof(entries[index++]);
	position(i++) = std::stof(entries[index++]);

	i = 0;
	Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();
	if (entries.size() >= start + 7)
	{
		rotation.x() = std::stof(entries[index++]);
		rotation.y() = std::stof(entries[index++]);
		rotation.z() = std::stof(entries[index++]);
		rotation.w() = std::stof(entries[index++]);
	}

	return { Eigen::Vector3f::Ones(), position, rotation };
}

pn_boxed_place::Ptr task_execution::get_location(const object_prototype& object, const obb& box) const
{
	pn_boxed_place::Ptr min;
	float min_dist = std::numeric_limits<float>::infinity();

	for (const auto& place : locations.at(object.get_type()))
	{
		float dist = (box.translation - place->box.translation).norm();
		if (dist < min_dist)
		{
			min_dist = dist;
			min = place;
		}
	}

	return min;
}

}  /* namespace mogaze */
} /* namespace simulation */
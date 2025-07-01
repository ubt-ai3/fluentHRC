#include "scene.hpp"

#include <Eigen/Geometry>

#include <state_observation/pointcloud_util.hpp>
#include <state_observation/pn_model_extension.hpp>

using namespace state_observation;

#undef RGB

namespace simulation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: scene_object
//
//
/////////////////////////////////////////////////////////////
	

/////////////////////////////////////////////////////////////
//
//
//  Class: simulated_table
//
//
/////////////////////////////////////////////////////////////

const std::string simulated_table::cube_path = "assets/object_meshes/cube.obj";
const float simulated_table::width = 0.8f;
const float simulated_table::breadth = 0.5f;
const pcl::RGB simulated_table::color = pcl::RGB(185, 122, 87);

simulated_table::simulated_table()
{
	{
		const auto mesh = std::make_shared<pcl::PolygonMesh>();
		if (pcl::io::load(cube_path, *mesh))
			throw std::runtime_error("Could not load polygon file");

		table_mesh = pointcloud_preprocessing::color(
			pointcloud_preprocessing::transform(mesh, Eigen::Affine3f(Eigen::Translation3f(0,0,-0.02f)*Eigen::Scaling(width, breadth, 0.02f))),
			color);
	}
}



void simulated_table::render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp)
{

	scene.add(std::make_shared<pcl::simulation::TriangleMeshModel>(
		std::make_shared<pcl::PolygonMesh>(*table_mesh)
	));

}

void simulated_table::render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport)
{
	if (!viewer.contains("table"))
	{
		viewer.addPolygonMesh(*table_mesh, "table");
		viewer.updatePointCloudPose("table", Eigen::Affine3f(Eigen::Translation3f(0, 0, -0.01f)));
	}
}

/////////////////////////////////////////////////////////////
//
//
//  Class: movable_object
//
//
/////////////////////////////////////////////////////////////

movable_object::movable_object(const object_prototype::ConstPtr prototype,
	Eigen::Vector3f center,
	state_observation::pn_instance instance)
	:
	prototype(prototype),
	center(center),
	instance(instance),
	id(std::to_string(std::hash<movable_object*>{}(this)))
{
	//mesh = pointcloud_preprocessing::color(prototype->get_base_mesh()->load_mesh(), prototype->get_mean_color());
	// prototype mesh is 2m x 2m x 2m, so we have to scale it down to half the diagonal
	mesh = pointcloud_preprocessing::transform(prototype->load_mesh(), Eigen::Affine3f(0.5 * Eigen::Scaling(prototype->get_bounding_box().diagonal)));
}

movable_object::movable_object(
	const state_observation::object_prototype::ConstPtr prototype,
	Eigen::Vector3f center,
	Eigen::Quaternionf rotation,
	state_observation::pn_instance instance
)
	:
	movable_object(prototype,center,instance)
{
	mesh = pointcloud_preprocessing::transform(mesh, Eigen::Affine3f(rotation));
}

void movable_object::render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp)
{
	scene.add(std::make_shared<pcl::simulation::TriangleMeshModel>(
		pointcloud_preprocessing::transform(mesh, Eigen::Affine3f(Eigen::Translation3f(center)))
	));
}

void movable_object::render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport)
{
	const Eigen::Affine3f transform = Eigen::Affine3f(Eigen::Translation3f(center));

	if (viewer.updatePointCloudPose(id, transform))
		return;

	viewer.addPolygonMesh(*mesh, id);
	viewer.updatePointCloudPose(id, transform);
}

/////////////////////////////////////////////////////////////
//
//
//  Class: simulated_arm
//
//
/////////////////////////////////////////////////////////////

const std::string simulated_arm::sphere_path = "assets/object_meshes/sphere.obj";
const std::string simulated_arm::cylinder_path = "assets/object_meshes/cylinder.obj";
const float simulated_arm::hand_radius = 0.06f;
const float simulated_arm::arm_radius = 0.05f;
double simulated_arm::speed = 0.2;
const pcl::RGB simulated_arm::color = pcl::RGB(243, 203, 191);

simulated_arm::simulated_arm(const Eigen::Vector3f& shoulder, const Eigen::Vector3f& hand)
	:
	shoulder_pose(shoulder),
	start_pose(hand),
	end_pose(hand),
	start(0),
	duration(0)
{
	{
		auto mesh = std::make_shared<pcl::PolygonMesh>();
		if (pcl::io::load(cylinder_path, *mesh))
			throw std::runtime_error("Could not load polygon file");
		cylinder_mesh = mesh;
	}

	{
		auto mesh = std::make_shared<pcl::PolygonMesh>();
		if (pcl::io::load(sphere_path, *mesh))
			throw std::runtime_error("Could not load polygon file");
		sphere_mesh = mesh;
	}



	start_pose = end_pose = hand;
}

void simulated_arm::move(const Eigen::Vector3f& destination, std::chrono::duration<float> timestamp)
{
	start_pose = get_tcp(timestamp);
	start = timestamp;
	end_pose = destination;
	duration = std::chrono::duration<float>((end_pose - start_pose).norm() / speed);
}

bool simulated_arm::is_moving(std::chrono::duration<float> timestamp) const
{
	return duration > std::chrono::duration<float>(0) && start < timestamp && timestamp < start + duration;
}

Eigen::Vector3f simulated_arm::get_tcp(std::chrono::duration<float> timestamp) const
{
	if (timestamp <= start)
		return start_pose;
	
	if (timestamp < start + duration)
		return ((timestamp - start) / duration) * (end_pose - start_pose) + start_pose;
	
	return end_pose;
}


Eigen::Vector3f simulated_arm::get_shoulder_pose(std::chrono::duration<float> timestamp) const
{
	return shoulder_pose;
}

void simulated_arm::render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp)
{
	render([&scene](const pcl::PolygonMesh::ConstPtr& mesh, const Eigen::Affine3f& matrix)
		{
			scene.add(std::make_shared<pcl::simulation::TriangleMeshModel>(
				pointcloud_preprocessing::color(
					pointcloud_preprocessing::transform(mesh, matrix),
					color
				)
			));
		},
		timestamp);
}

void simulated_arm::render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport)
{
	render([&viewer, this](const pcl::PolygonMesh::ConstPtr& mesh, const Eigen::Affine3f& matrix)
		{
			std::string id = std::to_string(std::hash<simulated_arm*>{}(this)) + std::to_string(std::hash<const pcl::PolygonMesh*>{}(&*mesh));

			if (!viewer.updatePointCloudPose(id, matrix))
			{

				viewer.addPolygonMesh(*pointcloud_preprocessing::color(mesh, color), id);
				viewer.updatePointCloudPose(id, matrix);
			}
		},
		timestamp);
}

void simulated_arm::render(std::function<void(const pcl::PolygonMesh::ConstPtr& mesh, const Eigen::Affine3f& matrix)> add, std::chrono::duration<float> timestamp)
{
	if (timestamp >= start + duration)
	{
		start_pose = end_pose;
		start = timestamp;
		duration = std::chrono::duration<float>(0);

	}

	Eigen::Vector3f pose = get_tcp(timestamp);

	Eigen::Affine3f matrix = Eigen::Translation3f(pose) * Eigen::Scaling(hand_radius, hand_radius, hand_radius);

	add(sphere_mesh, matrix);

	matrix =
		Eigen::Translation3f(shoulder_pose) *
		Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0, 0, 1), pose - shoulder_pose) *
		Eigen::Scaling(0.5f * arm_radius, 0.5f * arm_radius, 0.5f * (pose - shoulder_pose).norm()) *
		Eigen::Translation3f(0, 0, 1);

	add(cylinder_mesh, matrix);
}

/////////////////////////////////////////////////////////////
//
//
//  Class: environment
//
//
/////////////////////////////////////////////////////////////

environment::environment(const object_parameters& object_params)
	:
	net(new pn_net(object_params))
{
}

environment::environment(const state_observation::pn_net::Ptr& net,
     Distribution distribution,
     TokenTraces tokenTraces)
	: net(net), distribution(std::move(distribution)), token_traces(std::move(tokenTraces))
{
}

state_observation::pn_boxed_place::Ptr environment::add_location(const aabb& box)
{
	auto place = std::make_shared<pn_boxed_place>(box);
	net->add_place(place);
	return place;
}

state_observation::pn_boxed_place::Ptr environment::try_add_location(const state_observation::aabb& box)
{
	auto place = get_from_location(box);
	if (!place)
	{
		place = std::make_shared<pn_boxed_place>(box);
		net->add_place(place);
	}
	return place;
}

state_observation::pn_boxed_place::Ptr environment::get_from_location(const state_observation::aabb& box) const
{
	return net->get_place(box);
}

state_observation::pn_object_instance environment::add_object(
	const object_prototype::ConstPtr& prototype, 
	const obb& location, 
	bool stack)
{
	auto place = add_location(location);
	pn_object_token::Ptr token;
	
	auto iter = token_traces.find(prototype);
	if(iter == token_traces.end())
	{
		token = std::make_shared<pn_object_token>(prototype);
		token_traces.emplace(prototype, token);
	}
	else
	{
		token = iter->second;
	}

	auto instance = std::make_pair(place, token);
	distribution.insert(instance);
	
	Eigen::Vector3f center = location.translation;
	if(!stack)
		center.z() = prototype->get_bounding_box().diagonal.z() / 2;
	//object_traces.emplace(instance, std::make_shared<movable_object>(prototype, center, instance));
	object_traces.emplace(instance, std::make_shared<movable_object>(prototype, center, location.rotation, instance));

	return instance;
}

state_observation::pn_binary_marking::Ptr environment::update(const pn_transition::Ptr& transition)
{
	//list ensure that distribution doesn't change if transition is not enabled
	std::list<Distribution::iterator> consumption;

	for (const pn_transition::pn_arc& arc : transition->get_side_conditions())
	{
		auto it = distribution.find(arc.first);
		if (it == distribution.end())
			throw std::runtime_error("Transition not enabled: " + transition->to_string());
	}

	/**
	 * Ignore side conditions since they won't change
	 * distribution and traces
	 */
	for (const pn_transition::pn_arc& arc : transition->get_pure_input_arcs())
	{
		auto it = distribution.find(arc.first);
		if (it == distribution.end())
			throw std::runtime_error("Transition not enabled");
		consumption.emplace_back(it);
	}
	//Consume inputs from distribution
	for (const auto& it : consumption)
		distribution.erase(it);
	consumption.clear();

	for (const pn_transition::pn_arc& arc : transition->get_pure_input_arcs())
	{
		const auto iter = object_traces.find(arc);
		if (iter == object_traces.end())
			continue;

		auto object = iter->second;
		if (object == nullptr)
			throw std::exception("Try to emplace empty object");

		/**
		 * Get place where object is getting place[d] in the output
		 */
		for (const pn_place::Ptr& out_p : transition->get_outputs({ arc.second }))
		{
			if (transition->is_side_condition(std::make_pair(out_p, arc.second)))
				continue;

			/**
			 * Add output place with object to traces and distribution
			 */
			auto pair = std::make_pair(out_p, arc.second);
			//Consume old instance 
			object_traces.erase(arc);
			object_traces.emplace(pair, object); 
			distribution.emplace(pair);

			if (const auto out_p_boxed = std::dynamic_pointer_cast<pn_boxed_place>(out_p); out_p_boxed)
			{
				object->center = out_p_boxed->box.translation;

				if (!std::dynamic_pointer_cast<stack_action>(transition)) // place on table if it is not a stack action
					object->center.z() = object->prototype->get_bounding_box().diagonal.z() / 2;
			}
			break;
		}
		
	}

	//Don't add side conditions as we ignored them during consumption
	//Add other outputs
	const auto pure_outputs = transition->get_pure_output_arcs();
	distribution.insert(pure_outputs.begin(), pure_outputs.end());

	return get_marking();
}

state_observation::pn_binary_marking::Ptr environment::get_marking() const
{
	return std::make_shared<pn_binary_marking>(net, std::set<pn_instance>(distribution.begin(), distribution.end()));
}

}
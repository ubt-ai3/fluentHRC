#include "object_prototype_loader.hpp"

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

namespace state_observation
{

std::vector<object_prototype::Ptr> object_prototype_loader::generate_default_prototypes()
{
	std::vector<object_prototype::Ptr> result;

	aabb block(Eigen::Vector3f(0.028f, 0.028f, 0.056f));
	aabb cube(Eigen::Vector3f(0.028f, 0.028f, 0.028f));
	aabb flat_block(Eigen::Vector3f(0.056f, 0.028f, 0.014f));
	aabb plank(Eigen::Vector3f(0.084f, 0.028f, 0.014f));
	aabb semicylinder(Eigen::Vector3f(0.028f, 0.035f, 0.016f));

	pcl::RGB red(230, 80, 55);
	pcl::RGB blue(5, 20, 110);
	pcl::RGB cyan(5, 115, 185);
	pcl::RGB wooden(200, 190, 180);
	pcl::RGB magenta(235, 45, 135);
	pcl::RGB purple(175, 105, 180);
	pcl::RGB yellow(250, 255, 61);
	pcl::RGB dark_green(51, 82, 61);

	const std::string mesh_loc("assets/object_meshes/");
	mesh_wrapper::Ptr cylinder_mesh = std::make_shared<mesh_wrapper>(mesh_loc + "cylinder.obj");
	mesh_wrapper::Ptr cuboid_mesh = std::make_shared<mesh_wrapper>(mesh_loc + "cube.obj");

	mesh_wrapper::Ptr semicylinder_mesh = std::make_shared<mesh_wrapper>(mesh_loc + "semicylinder.obj");
	mesh_wrapper::Ptr triangular_prism_mesh = std::make_shared<mesh_wrapper>(mesh_loc + "triangular_prism.obj");
	mesh_wrapper::Ptr bridge_mesh = std::make_shared<mesh_wrapper>(mesh_loc + "bridge.obj");

	result.push_back(

		std::make_shared<object_prototype>(
			aabb(),
			dark_green,
			nullptr,
			"dark green background",
			"background"));

	result.push_back(
		std::make_shared<object_prototype>(
			cube,
			wooden,
			cylinder_mesh,
			"wooden small cylinder",
			"cylinder"));

	result.push_back(
		std::make_shared<object_prototype>(
			semicylinder,
			wooden,
			semicylinder_mesh,
			"wooden semicylinder",
			"semicylinder"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			wooden,
			cylinder_mesh,
			"wooden cylinder",
			"cylinder"));

	result.push_back(
		std::make_shared<object_prototype>(
			cube,
			wooden,
			cuboid_mesh,
			"wooden cube",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			wooden,
			cuboid_mesh,
			"wooden block",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			wooden,
			triangular_prism_mesh,
			"wooden triangular prism",
			"triangular_prism"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			wooden,
			bridge_mesh,
			"wooden bridge",
			"bridge"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			purple,
			bridge_mesh,
			"purple bridge",
			"bridge"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			magenta,
			bridge_mesh,
			"magenta bridge",
			"bridge"));

	result.push_back(
		std::make_shared<object_prototype>(
			flat_block,
			wooden,
			cuboid_mesh,
			"wooden flat block",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			plank,
			wooden,
			cuboid_mesh,
			"wooden plank",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			plank,
			cyan,
			cuboid_mesh,
			"cyan plank",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			plank,
			red,
			cuboid_mesh,
			"red plank",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			plank,
			yellow,
			cuboid_mesh,
			"yellow plank",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			red,
			cuboid_mesh,
			"red block",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			blue,
			cuboid_mesh,
			"blue block",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			yellow,
			cuboid_mesh,
			"yellow block",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			cube,
			red,
			cylinder_mesh,
			"red small cylinder",
			"cylinder"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			magenta,
			cylinder_mesh,
			"magenta cylinder",
			"cylinder"));

	result.push_back(
		std::make_shared<object_prototype>(
			flat_block,
			red,
			cuboid_mesh,
			"red flat block",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			flat_block,
			yellow,
			cuboid_mesh,
			"yellow flat block",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			flat_block,
			purple,
			cuboid_mesh,
			"purple flat block",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			semicylinder,
			purple,
			semicylinder_mesh,
			"purple semicylinder",
			"semicylinder"));

	result.push_back(
		std::make_shared<object_prototype>(
			semicylinder,
			magenta,
			semicylinder_mesh,
			"magenta semicylinder",
			"semicylinder"));

	result.push_back(
		std::make_shared<object_prototype>(
			block,
			cyan,
			triangular_prism_mesh,
			"cyan triangular prism",
			"triangular_prism"));

	result.push_back(
		std::make_shared<object_prototype>(
			cube,
			red,
			cuboid_mesh,
			"red cube",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			cube,
			purple,
			cuboid_mesh,
			"purple cube",
			"cuboid"));


	result.push_back(
		std::make_shared<object_prototype>(
			cube,
			cyan,
			cuboid_mesh,
			"cyan cube",
			"cuboid"));

	result.push_back(
		std::make_shared<object_prototype>(
			cube,
			yellow,
			cuboid_mesh,
			"yellow cube",
			"cuboid"));

	return result;

}

object_prototype_loader::object_prototype_loader()
{
	filename_ = std::string("object_prototypes.xml");

	std::ifstream file(folder_ + filename_);

	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(prototypes);
	}
	else {
		prototypes = generate_default_prototypes();

		std::ofstream out_file(folder_ + filename_);
		boost::archive::xml_oarchive oa{ out_file };
		oa << BOOST_SERIALIZATION_NVP(prototypes);
	}
}

std::vector<object_prototype::ConstPtr> object_prototype_loader::get_prototypes() const
{
	return { prototypes.begin(), prototypes.end() };
}

object_prototype::ConstPtr object_prototype_loader::get(const std::string& name) const
{
	for (const object_prototype::Ptr& prototype : prototypes)
	{
		if (!prototype->get_name().compare(name))
			return prototype;
	}
	

	throw std::invalid_argument("No such prototype");
}

};
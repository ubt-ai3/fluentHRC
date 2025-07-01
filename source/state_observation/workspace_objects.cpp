#include "workspace_objects.hpp"

#include <pcl/common/transforms.h>

namespace state_observation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: aabb
//
//
/////////////////////////////////////////////////////////////

	float aabb::top_z() const
	{
		return aabb::center_z() + 0.5f * diagonal.z();
	}

	float aabb::bottom_z() const
	{
		return aabb::center_z() - 0.5f * diagonal.z();
	}

	float aabb::center_z() const
	{
		return translation.z();
	}

	aabb::aabb(const Eigen::Vector3f& diagonal,
	       const Eigen::Vector3f& translation)
	:
	diagonal(diagonal),
	translation(translation)
{
}

aabb::aabb()
	:
	diagonal(Eigen::Vector3f::Zero()),
	translation(Eigen::Vector3f::Zero())
{
}

aabb aabb::from_corners(const Eigen::Vector3f& min_point, 
	const Eigen::Vector3f& max_point)
{
	return aabb((max_point - min_point).cwiseAbs(),
		0.5 * max_point + 0.5 * min_point
		);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: obb
//
//
/////////////////////////////////////////////////////////////

obb::obb()
	:
	rotation(Eigen::Quaternionf::Identity())
{
}

obb::obb(const Eigen::Vector3f& diagonal,
	const Eigen::Vector3f& translation,
	const Eigen::Quaternionf& rotation)
	:
	aabb(diagonal, translation),
	rotation(rotation)
{
}

obb::obb(const aabb& box)
	:
	aabb(box),
	rotation(Eigen::Quaternionf::Identity())
{
}

float obb::top_z() const
{
	return std::abs(center_z()) + translation.z();
}

float obb::bottom_z() const
{
	return -std::abs(center_z()) + translation.z();
}

float obb::center_z() const
{
	/*Eigen::Vector3f vec = rotation * Eigen::Vector3f::UnitZ();
	if (vec.z() - 1 < -0.0001f)
	{
		return (rotation * Eigen::Vector3f((2 * std::signbit(vec.x()) - 1) * diagonal.x(),
			(2 * std::signbit(vec.y()) - 1) * diagonal.y(),
			(2 * std::signbit(vec.z()) - 1) * diagonal.z())).z();
	}
	else
		return aabb::center_z();*/

	return 0.5f * (rotation * diagonal).z();
}



std::vector<Eigen::Vector3f> obb::get_corners() const
{

	std::vector<Eigen::Vector3f> corners;
	corners.reserve(8);

	Eigen::Affine3f transform = Eigen::Translation3f(translation) *
		Eigen::Affine3f(rotation) *
		Eigen::Scaling(0.5f * diagonal);

	for (int x = -1; x <= 1; x += 2)
		for (int y = -1; y <= 1; y += 2)
			for (int z = -1; z <= 1; z += 2)
				corners.push_back(transform * Eigen::Vector3f(x, y, z));

	return corners;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: mesh_wrapper
//
//
/////////////////////////////////////////////////////////////

mesh_wrapper::mesh_wrapper(const std::string& path)
	:	
	path(path)
{
}

void mesh_wrapper::set_path(const std::string& path)
{
	this->path = path;
	mesh = nullptr;
}

std::string mesh_wrapper::get_path() const
{
	return path;
}

pcl::PolygonMesh::ConstPtr mesh_wrapper::load_mesh() const
{
	if (!mesh) {
		mesh = pcl::PolygonMesh::Ptr(new pcl::PolygonMesh);
		pcl::io::load(path, *mesh); // ignore warnings
	}
	return mesh;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: pc_segment
//
//
/////////////////////////////////////////////////////////////

const std::shared_ptr<enact_core::aspect_id> pc_segment::aspect_id 
	= std::shared_ptr<enact_core::aspect_id>(new enact_core::aspect_id("pc_segment"));

pc_segment::pc_segment(const pcl::PointCloud<PointT>::ConstPtr& points,
	const pcl::PointIndices::ConstPtr& indices,
	const pcl::PointCloud<PointT>::ConstPtr& reference_frame,
	const obb& bounding_box,
	const PointT& centroid,
	std::chrono::duration<float> timestamp)
	:
	points(points),
	indices(indices),
	//reference_frame(reference_frame),
	bounding_box(bounding_box),
	centroid(centroid),
	outline(nullptr),
	outline_area(std::numeric_limits<float>::quiet_NaN()),
	timestamp(timestamp)
{
	compute_mean_color();
}

pc_segment::pc_segment()
	:
	outline_area(std::numeric_limits<float>::quiet_NaN()),
	timestamp{}
{}

pcl::PointCloud<pc_segment::PointT>::ConstPtr pc_segment::get_outline() const
{
	if (outline)
		return outline;

	std::vector<cv::Point2f> transformed_points;
	for (const PointT& p : *points)
	{
		transformed_points.emplace_back(p.x, p.y);
	}

	std::vector<int> hull_indices;

	cv::convexHull(transformed_points, hull_indices, false);

	pcl::PointCloud<PointT>::Ptr output(new pcl::PointCloud<PointT>);
	output->header.frame_id = points->header.frame_id;
	output->header.seq = points->header.seq;
	output->header.stamp = points->header.stamp;
	output->reserve(hull_indices.size());

	for (int index : hull_indices)
	{
		output->push_back(points->at(index));
	}

	outline = output;

	return output;
}

float pc_segment::get_outline_area() const
{
	if (!isnan(outline_area))
		outline_area = pcl::calculatePolygonArea(*get_outline());

	return outline_area;
}

void pc_segment::compute_mean_color()
{
	unsigned int r = 0, g = 0, b = 0;
	for (const auto& point : *points)
	{
		r += point.r;
		g += point.g;
		b += point.b;
	}

	mean_color.r = r / points->size();
	mean_color.g = g / points->size();
	mean_color.b = b / points->size();
}


/////////////////////////////////////////////////////////////
//
//
//  Class: object_prototype
//
//
/////////////////////////////////////////////////////////////

object_prototype::object_prototype(const aabb& bounding_box,
								   const pcl::RGB& mean_color,
								   const mesh_wrapper::Ptr& base_mesh,
								   const std::string& name,
								   const std::string& type)
	:
	bounding_box(bounding_box),
	mean_color(mean_color),
	base_mesh(base_mesh),
	name(name),
	type(type)
{
}

aabb object_prototype::get_bounding_box() const
{
	return bounding_box;
}

pcl::RGB object_prototype::get_mean_color() const
{
	return mean_color;
}

pcl::PolygonMesh::ConstPtr object_prototype::load_mesh() const
{
	assert(base_mesh && "base_mesh has to be valid for this function to be called");

	if (!object_prototype_mesh) {
		object_prototype_mesh = pointcloud_preprocessing::color(base_mesh->load_mesh(), mean_color);
	}
	return object_prototype_mesh;
}

const mesh_wrapper::ConstPtr object_prototype::get_base_mesh() const
{
	return base_mesh;
}

bool object_prototype::has_mesh() const
{
	return base_mesh && base_mesh->load_mesh();
}

const std::string& object_prototype::get_name() const
{
	return name;
}

const std::string& object_prototype::get_type() const
{
	return type;
}

/////////////////////////////////////////////////////////////
//
//
//  Class: classification_result
//
//
/////////////////////////////////////////////////////////////

classification_result::classification_result(const object_prototype::ConstPtr& prototype,
											 const Eigen::Quaternionf prototype_rotation,
											 const float local_certainty_score)
	:
	prototype(prototype),
	prototype_rotation(prototype_rotation),
	local_certainty_score(local_certainty_score)
{
}


/////////////////////////////////////////////////////////////
//
//
//  Class: object_instance
//
//
/////////////////////////////////////////////////////////////

const std::shared_ptr<enact_core::aspect_id> object_instance::aspect_id
	= std::shared_ptr<enact_core::aspect_id>(new enact_core::aspect_id("object_instance"));

#ifdef DEBUG_OBJECT_INSTANCE_ID
int object_instance::id_counter = 0;
#endif	

object_instance::object_instance(const pc_segment::Ptr& segment)
	:
#ifdef DEBUG_OBJECT_INSTANCE_ID
	ID(id_counter++),
#endif
	prototype(nullptr),
	global_certainty_score(std::numeric_limits<float>::quiet_NaN()),
	observation_history({segment})
{
}

pc_segment::Ptr object_instance::get_classified_segment() const
{
	pc_segment::Ptr latest_seg;
	for (const pc_segment::Ptr& seg : observation_history)
		if (!seg->classification_results.empty())
			latest_seg = seg;

	return latest_seg;
}

bool object_instance::is_background() const
{
	pc_segment::Ptr classified_segment = get_classified_segment();
	return classified_segment && !classified_segment->classification_results.front().prototype->has_mesh();
}

} //namespace state_observation

#include "classification.hpp"

#include <list>
#include <numbers>
#include <math.h>


namespace state_observation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: classifier
//
//
/////////////////////////////////////////////////////////////

std::weak_ptr<const object_parameters> classifier::object_params = std::shared_ptr< const object_parameters>(nullptr);

classifier::classifier(const object_prototype::ConstPtr& prototype)
	:
	prototype(prototype)
{
}

object_prototype::ConstPtr classifier::get_object_prototype() const
{
	return prototype;
}

void classifier::set_object_params(const std::shared_ptr<const object_parameters>& object_params)
{
	if (object_params)
		classifier::object_params = object_params;
}

float classifier::bell_curve(float x, float stdev)
{
	float temp = x / stdev;
	temp *= temp;
	return std::expf(-temp / 2);
}


double classifier::bell_curve(double x, double stdev)
{
	double temp = x / stdev;
	temp *= temp;
	return std::exp(-temp / 2);
}




/////////////////////////////////////////////////////////////
//
//
//  Class: bounding_box_classifier
//
//
/////////////////////////////////////////////////////////////

bounding_box_classifier::bounding_box_classifier(const object_prototype::ConstPtr& prototype)
	:
	classifier(prototype)
{
}

classification_result bounding_box_classifier::classify(const pc_segment& seg) const
{
	Eigen::Matrix3f similarities(stacked_similarity_matrix(seg));
	Eigen::Quaternionf rotation = stacked_rotation_guess(seg);
	Eigen::Matrix3f temp = similarities.cwiseProduct(Eigen::Matrix3f(rotation));
	float det = temp.determinant();
	return classification_result(prototype, rotation, std::abs(det));
}

Eigen::Quaternionf bounding_box_classifier::rotation_guess(const pc_segment& seg) const
{
	Eigen::Matrix3f rotation(Eigen::Matrix3f::Zero());
	std::list<int> seg_dimensions({ 2,1,0 }); 
	
	const aabb& bounding_box = prototype->get_bounding_box();
	for (int i = 2; i >= 0; --i)
		// match height first since other dimensions might be shorter
		// due to discarded samples close to the table
	{
		float min_diff = std::numeric_limits<float>::infinity();
		std::list<int>::iterator min_j;

		for (auto j = seg_dimensions.begin(); j != seg_dimensions.end(); ++j) 
		{
			if (std::abs(seg.bounding_box.diagonal(i) - bounding_box.diagonal(*j)) < min_diff)
			{
				min_diff = std::abs(seg.bounding_box.diagonal(i) - bounding_box.diagonal(*j));
				min_j = j;
			}

		}

		rotation(i, *min_j) = 1.f;
		seg_dimensions.erase(min_j);
	}

	rotation.col(2) = rotation.col(0).cross(rotation.col(1));

	return Eigen::Quaternionf(rotation).normalized();
}

Eigen::Quaternionf bounding_box_classifier::stacked_rotation_guess(const pc_segment& seg) const
{
	Eigen::Matrix3f rotation(Eigen::Matrix3f::Zero());
	std::list<int> seg_dimensions({ 2,1,0 });

	const aabb& bounding_box = prototype->get_bounding_box();
	for (int i = 0; i<3; i++)
	{
		float min_diff = std::numeric_limits<float>::infinity();
		std::list<int>::iterator min_j;

		for (auto j = seg_dimensions.begin(); j != seg_dimensions.end(); ++j)
		{
			if (std::abs(seg.bounding_box.diagonal(i) - bounding_box.diagonal(*j)) < min_diff)
			{
				min_diff = std::abs(seg.bounding_box.diagonal(i) - bounding_box.diagonal(*j));
				min_j = j;
			}

		}

		rotation(i, *min_j) = 1.f;
		seg_dimensions.erase(min_j);
	}

	rotation.col(2) = rotation.col(0).cross(rotation.col(1));

	return Eigen::Quaternionf(rotation).normalized();
}


Eigen::Matrix3f bounding_box_classifier::stacked_similarity_matrix(const pc_segment& seg) const
{
	Eigen::Matrix3f similarities(Eigen::Matrix3f::Zero());
	float object_min_dimension = object_params.lock()->min_object_height;

	const aabb& bounding_box = prototype->get_bounding_box();
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			similarities(i, j) = bell_curve(seg.bounding_box.diagonal(i) - bounding_box.diagonal(j),
				object_min_dimension);
		}
	}

	for (int j = 0; j < 3; ++j) {
		if (seg.bounding_box.diagonal(2) > bounding_box.diagonal(j))
			similarities(2, j) = 1.f;
		else
			similarities(2, j) = bell_curve(seg.bounding_box.diagonal(2) - bounding_box.diagonal(j),
			object_min_dimension);
	}

	return similarities;
}

Eigen::Matrix3f bounding_box_classifier::similarity_matrix(const pc_segment& seg) const
{
	Eigen::Matrix3f similarities(Eigen::Matrix3f::Zero());
	float object_min_dimension = object_params.lock()->min_object_height;

	const aabb& bounding_box = prototype->get_bounding_box();
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			similarities(i, j) = bell_curve(seg.bounding_box.diagonal(i) - bounding_box.diagonal(j),
				object_min_dimension);
		}
	}

	return similarities;
}

std::vector<Eigen::Quaternionf> bounding_box_classifier::get_feasible_transformations(const pc_segment& seg) const
{
	std::vector<Eigen::Quaternionf> transformations;

	transformations.push_back(stacked_rotation_guess(seg));

	return transformations;
}



/////////////////////////////////////////////////////////////
//
//
//  Class: shape_classifier
//
//
/////////////////////////////////////////////////////////////

shape_classifier::shape_classifier(const object_prototype::ConstPtr& prototype)
	:
	classifier(prototype)
{
}

classification_result shape_classifier::classify(const pc_segment& seg) const
{
	float best_similarity = 0.f;
	Eigen::Quaternionf best_rotation;

	for (const Eigen::Quaternionf& rotation : get_feasible_transformations(seg))
	{
		float similarity = match(seg, rotation);

		if (best_similarity < similarity)
		{
			best_similarity = similarity;
			best_rotation = rotation;
		}
	}
	
	return classification_result(prototype, best_rotation, best_similarity);
}


float shape_classifier::match(const pc_segment& seg, Eigen::Quaternionf prototype_rotation) const
{
	/*
	 * 	Eigen::Matrix4f transform(Eigen::Matrix4f::Identity());
	transform.block<3, 3>(0, 0) = seg.bounding_box.rotation
								  * prototype_rotation
								  * (0.5f * prototype->get_bounding_box().diagonal.asDiagonal());
	transform.block<3, 1>(0, 3) = seg.bounding_box.translation;
//	transform(2, 3) = std::abs(transform(2, 2));
	transform(2, 3) = transform.row(2).cwiseAbs().sum();
	 */

	Eigen::Affine3f transform = Eigen::Translation3f(seg.bounding_box.translation) *
		Eigen::Affine3f(seg.bounding_box.rotation) * Eigen::Affine3f(prototype_rotation) *
		Eigen::Scaling(0.5f * prototype->get_bounding_box().diagonal);

	const pcl::PolygonMesh& mesh = *prototype->load_mesh();
	const object_parameters& object_params = *classifier::object_params.lock();
	float similarity = 0.f;


	std::function<Eigen::Vector3f(int)> get_point =
		[&mesh, &transform](int index) 
	{
		const float* point = reinterpret_cast<const float*>(&mesh.cloud.data.at(mesh.cloud.point_step * index));
		const Eigen::Vector3f vec(point);
		return transform * vec;
	};

	float mesh_max_z = -std::numeric_limits<float>::infinity();
	for (int i = 0; i < mesh.cloud.width * mesh.cloud.height; i++)
		mesh_max_z = std::max(get_point(i).z(), mesh_max_z);

	if (mesh_max_z + object_params.min_object_height < seg.bounding_box.diagonal.z())
		transform(2, 3) = mesh_max_z - prototype->get_bounding_box().diagonal.z() / 2;

	for (const PointT& p : *seg.points)
	{
		float dist = std::numeric_limits<float>::infinity();
		Eigen::Vector3f p0 = p.getVector3fMap();

		for (const pcl::Vertices& polygon : mesh.polygons)
		{
			Eigen::Vector3f p1 = get_point(polygon.vertices[0]);
			Eigen::Vector3f p2 = get_point(polygon.vertices[1]);
			Eigen::Vector3f p3 = get_point(polygon.vertices[2]);

			dist = std::min(dist, distance(p0, p1, p2, p3));
		}
		similarity += bell_curve(dist, 0.5f * object_params.min_object_height);
		
	}
	//compare vectors
	int count = 0;
	//for (auto it1 = novel::classifier::debug->correct_points.begin(), it2 = novel::classifier::debug->new_points.begin();
	//	it1 != novel::classifier::debug->correct_points.end();
	//	++it1, ++it2,++count)
	//{
	//	if (*it1 != *it2)
	//		std::cout << "error detected\n";
	//}
	/*for (auto it1 = novel::classifier::debug->correct_prototype_rotation.begin(), it2 = novel::classifier::debug->new_prototype_rotation.begin();
		it1 != novel::classifier::debug->correct_prototype_rotation.end();
		++it1, ++it2, ++count)
	{
		if (it1->angularDistance(*it2) > 0)
		{
			float dist = it1->angularDistance(*it2);
			std::cout << "error detectetd\n";
		}
	}*/
	return similarity / seg.points->size();
}

/**
* implementation following http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.104.4264&rep=rep1&type=pdf
* with N_p = normal; P_0' = q0, P_0'' = q1
*/
float shape_classifier::distance(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3) const
{
	float min_dist = std::numeric_limits<float>::infinity();

	Eigen::Vector3f normal((p2-p1).cross(p3-p1).normalized());

	float cosalpha = (p0 - p1).normalized().dot(normal);
	// q0 = p0 projected into the plane spanned by p1,p2,p3
	Eigen::Vector3f q0 = p0 - ((p1 - p0).norm() * cosalpha) * normal;

	Eigen::Vector3f v1 = normal.cross(p2 - p1);
	Eigen::Vector3f v2 = normal.cross(p3 - p2);
	Eigen::Vector3f v3 = normal.cross(p1 - p3);

	bool outside = false;
	if (v1.dot(q0) < v1.dot(p1))
	{
		outside = true;
		Eigen::Vector3f q1 = project(q0, p1, p2);

		min_dist = std::min(min_dist, std::sqrt(std::powf((q1 - q0).norm(), 2.f) + std::powf((q0 - p0).norm(), 2.f)));
	}
	if (v2.dot(q0) < v2.dot(p2))
	{
		outside = true;
		Eigen::Vector3f q1 = project(q0, p2, p3);

		min_dist = std::min(min_dist, std::sqrt(std::powf((q1 - q0).norm(), 2.f) + std::powf((q0 - p0).norm(), 2.f)));
	}
	if (v3.dot(q0) < v3.dot(p3))
	{
		outside = true;
		Eigen::Vector3f q1 = project(q0, p3, p1);

		min_dist = std::min(min_dist, std::sqrt(std::powf((q1 - q0).norm(), 2.f) + std::powf((q0 - p0).norm(), 2.f)));
	}

	if (!outside)
		return (p0 - q0).norm();
	else
		return min_dist;
}

Eigen::Vector3f shape_classifier::project(const Eigen::Vector3f& q0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) const
{
	float t = (q0- p1).dot(p2-p1)/(p2-p1).squaredNorm();
	t = std::max(0.f, std::min(1.f, t));
	return p1 + t * (p2 - p1);
}




/////////////////////////////////////////////////////////////
//
//
//  Class: monochrome_object_classifier
//
//
/////////////////////////////////////////////////////////////

monochrome_object_classifier::monochrome_object_classifier(const object_prototype::ConstPtr& prototype)
	:
	classifier(prototype)
{
}

monochrome_object_classifier::~monochrome_object_classifier()
{
}

classification_result monochrome_object_classifier::classify(const pc_segment& seg) const
{
	float similarity = 0.f;
	pcl::RGB ref(prototype->get_mean_color());
	const float stdev = 20;

	for (const PointT& p : *seg.points) {
//		double delta = cielab(rgb(p.r,p.g,p.b)).delta(ref_color);
//		similarity += bell_curve(delta, 0.2f * cielab::MAX_DELTA);
		
		float sim_r = bell_curve((float) std::abs(ref.r - p.r), stdev);
		float sim_g = bell_curve((float)std::abs(ref.g - p.g), stdev);
		float sim_b = bell_curve((float)std::abs(ref.b - p.b), stdev);
		similarity += std::powf(sim_r * sim_g * sim_b, 1.f / 3.f);


	}

	return classification_result(prototype, Eigen::Quaternionf::Identity(), similarity / seg.points->size());
}



/////////////////////////////////////////////////////////////
//
//
//  Class: background_classifier
//
//
/////////////////////////////////////////////////////////////

background_classifier::background_classifier(const object_prototype::ConstPtr& prototype)
	:
	classifier(prototype)
{
}

classification_result background_classifier::classify(const pc_segment& seg) const
{
	unsigned int max_x = 0, max_y = 0;
	unsigned int min_x, min_y;
	min_x = min_y = std::numeric_limits<unsigned int>::max();

	//for (int i : seg.indices->indices)
	//{
	//	min_x = std::min(min_x, i % seg.reference_frame->width);
	//	min_y = std::min(min_y, i / seg.reference_frame->width);
	//	max_x = std::max(max_x, i % seg.reference_frame->width);
	//	max_y = std::max(max_y, i / seg.reference_frame->width);
	//}

	//float max_points = (max_x - min_x + 1) * (max_y - min_y + 1);
	
	float similarity = monochrome_object_classifier::classify(seg).local_certainty_score;
		//*		bell_curve(seg.indices->indices.size(), 0.5f * max_points);

	return classification_result(prototype, Eigen::Quaternionf::Identity(), similarity);
}






/////////////////////////////////////////////////////////////
//
//
//  Class: cuboid_classifier
//
//
/////////////////////////////////////////////////////////////

cuboid_classifier::cuboid_classifier(const object_prototype::ConstPtr& prototype)
	:
	classifier(prototype)
{
}

classification_result cuboid_classifier::classify(const pc_segment& seg) const
{
	float min_height = object_params.lock()->min_object_height;

	if (seg.bounding_box.diagonal(0) < prototype->get_bounding_box().diagonal(2) - min_height
		&& seg.bounding_box.diagonal(1) < prototype->get_bounding_box().diagonal(2) - min_height
		&& seg.bounding_box.diagonal(2) < prototype->get_bounding_box().diagonal(2) - min_height)
		return classification_result(prototype);

	const object_parameters& object_params = *classifier::object_params.lock();
	const obb& box = seg.bounding_box;


	if (box.diagonal.z() > object_params.min_object_height + box.diagonal.x()
		&& box.diagonal.z() > object_params.min_object_height + box.diagonal.y())
	{ // object standing, use outline for shape comparison
		
		std::vector<cv::Point2f> outline;
		outline.reserve(seg.get_outline()->size());

		//transform outline such that bounding box is centered in origin and axis aligned
		for (const PointT& p : *seg.get_outline())
		{
			Eigen::Vector3f untransformed(p.x, p.y, p.z);
			Eigen::Vector3f transformed = box.rotation.inverse() * (untransformed - box.translation);
			outline.emplace_back(cv::Point2f(transformed.x(), transformed.y()));
		}

		//measure similarity by the distance points on the outline to their enclosing rectangle
		float similarity = 1.f;
		for (const cv::Point2f& p : outline)
		{
			float dist_x = std::abs(std::abs(p.x) - box.diagonal(0) / 2.f);
			float dist_y = std::abs(std::abs(p.y) - box.diagonal(1) / 2.f);
			similarity *= bell_curve(std::min(dist_x, dist_y), object_params.min_object_height);
		}

		classification_result box_sim = bounding_box_classifier::classify(seg);
		classification_result color_sim = monochrome_object_classifier::classify(seg);
		float shape_sim = std::powf(similarity, 1. / outline.size());


		//auto debug = novel::classifier::debug.get();

		similarity = std::powf(box_sim.local_certainty_score
			* color_sim.local_certainty_score * color_sim.local_certainty_score
			* shape_sim
			, 1.f / 4.f);

		return classification_result(prototype, box_sim.prototype_rotation, similarity);
	}
	else
	{
		
		classification_result box_sim = bounding_box_classifier::classify(seg);
		classification_result shape_sim = shape_classifier::classify(seg);
		classification_result color_sim = monochrome_object_classifier::classify(seg);


		float similarity = std::powf(shape_sim.local_certainty_score
			* box_sim.local_certainty_score
			* color_sim.local_certainty_score
			, 1.f / 3.f);

		return classification_result(prototype, shape_sim.prototype_rotation, similarity);
	}


}


/////////////////////////////////////////////////////////////
//
//
//  Class: cylinder_classifier
//
//
/////////////////////////////////////////////////////////////

cylinder_classifier::cylinder_classifier(const object_prototype::ConstPtr& prototype)
	:
	classifier(prototype)
{
}

classification_result cylinder_classifier::classify(const pc_segment& seg) const
{
	float min_height = object_params.lock()->min_object_height;

	if (seg.bounding_box.diagonal(0) < prototype->get_bounding_box().diagonal(2) - min_height
		&& seg.bounding_box.diagonal(1) < prototype->get_bounding_box().diagonal(2) - min_height
		&& seg.bounding_box.diagonal(2) < prototype->get_bounding_box().diagonal(2) - min_height)
		return classification_result(prototype);

	const object_parameters& object_params = *classifier::object_params.lock();
	const obb& box = seg.bounding_box;
	float width = box.diagonal(0);
	float breadth = box.diagonal(1);

	if (box.diagonal.z() > object_params.min_object_height + width
		&& box.diagonal.z() > object_params.min_object_height + breadth)
	{ // object standing, use outline for shape comparison

	// Point cloud to vector
		std::vector<cv::Point2f> outline;
		outline.reserve(seg.get_outline()->size());

		const pcl::PointCloud<PointT>& seg_outline = seg.get_outline()->size() ? *seg.get_outline() : *seg.points;
		for (const PointT& p : seg_outline)
		{
			outline.emplace_back(cv::Point2f(p.x, p.y));
		}


		//measure similarity by the distance points on the outline to their enclosing circle
		cv::Point2f center;
		float radius;
		cv::minEnclosingCircle(outline, center, radius);

		float similarity = 1.f;
		for (const cv::Point2f& p : outline)
		{
			cv::Point2f vec = p - center;
			float dist = std::abs(std::sqrt(vec.x * vec.x + vec.y * vec.y) - radius);
			similarity *= bell_curve(dist, object_params.min_object_height);
		}

		classification_result box_sim = bounding_box_classifier::classify(seg);
		classification_result color_sim = monochrome_object_classifier::classify(seg);
		float shape_sim = std::pow(similarity, 1. / outline.size());

		similarity = std::powf(box_sim.local_certainty_score
			* color_sim.local_certainty_score * color_sim.local_certainty_score
			* shape_sim
			, 1.f / 4.f);

		return classification_result(prototype, box_sim.prototype_rotation, similarity);
	}
	else
	{
		classification_result box_sim = bounding_box_classifier::classify(seg);		
		classification_result color_sim = monochrome_object_classifier::classify(seg);
		classification_result shape_sim = shape_classifier::classify(seg);

		float similarity = std::powf(shape_sim.local_certainty_score
			* box_sim.local_certainty_score
			* color_sim.local_certainty_score * color_sim.local_certainty_score
			, 1.f / 4.f);

		return classification_result(prototype, shape_sim.prototype_rotation, similarity);
	}
}


/////////////////////////////////////////////////////////////
//
//
//  Class: semicylinder_classifier
//
//
/////////////////////////////////////////////////////////////

semicylinder_classifier::semicylinder_classifier(const object_prototype::ConstPtr& prototype)
	:
	classifier(prototype)
{
}

classification_result semicylinder_classifier::classify(const pc_segment& seg) const
{
	float min_height = object_params.lock()->min_object_height;

	if (seg.bounding_box.diagonal(0) > prototype->get_bounding_box().diagonal(0) + min_height
		|| seg.bounding_box.diagonal(1) > prototype->get_bounding_box().diagonal(0) + min_height
		|| seg.bounding_box.diagonal(2) > prototype->get_bounding_box().diagonal(0) + min_height)
		return classification_result(prototype);

	classification_result shape_sim = shape_classifier::classify(seg);
	classification_result color_sim = monochrome_object_classifier::classify(seg);

	// assume that something small is a semicylinder
	float box_sim;
	const Eigen::Vector3f& seg_diag = seg.bounding_box.diagonal;
	const Eigen::Vector3f& proto_diag = prototype->get_bounding_box().diagonal;

	if (seg_diag(0) < proto_diag(0) && seg_diag(1) < proto_diag(1) && seg_diag(2) < proto_diag(2))
	{
		box_sim = 1.f;
	}
	else
	{
		box_sim = bounding_box_classifier::classify(seg).local_certainty_score;
	}

	float similarity = std::powf(shape_sim.local_certainty_score
		* box_sim
		* color_sim.local_certainty_score * color_sim.local_certainty_score
		, 1.f / 4.f);

	return classification_result(prototype, shape_sim.prototype_rotation, similarity);
}

std::vector<Eigen::Quaternionf> semicylinder_classifier::get_feasible_transformations(const pc_segment& seg) const
{
	std::vector<Eigen::Quaternionf> transformations;

	std::vector<Eigen::Quaternionf> bb_transformations(bounding_box_classifier::get_feasible_transformations(seg));
	const Eigen::Quaternionf counter_clockwise_rotation(Eigen::Quaternionf::FromTwoVectors(
		Eigen::Vector3f(1.f, 0.f, 0.f),
		Eigen::Vector3f(0.f, 1.f, 0.f)
	));

	for (const Eigen::Quaternionf& bb_trans : bb_transformations) 
	{
		transformations.push_back(bb_trans);
		if ((bb_trans * Eigen::Vector3f(0.f, 0.f, 1.f))(2) > 0.9f) //curve pointing upwards
		{
			transformations.push_back((bb_trans * counter_clockwise_rotation).normalized());
		}

	}
	
	return transformations;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: bridge_classifier
//
//
/////////////////////////////////////////////////////////////

bridge_classifier::bridge_classifier(const object_prototype::ConstPtr& prototype)
	:
	classifier(prototype)
{
}

classification_result bridge_classifier::classify(const pc_segment& seg) const
{
	return cuboid_classifier::classify(seg);
}

std::vector<Eigen::Quaternionf> bridge_classifier::get_feasible_transformations(const pc_segment& seg) const
{
	std::vector<Eigen::Quaternionf> transformations;

	std::vector<Eigen::Quaternionf> bb_transformations(bounding_box_classifier::get_feasible_transformations(seg));
	const Eigen::Quaternionf counter_clockwise_rotation(Eigen::Quaternionf::FromTwoVectors(
		Eigen::Vector3f(1.f, 0.f, 0.f),
		Eigen::Vector3f(0.f, 1.f, 0.f)
	));

	for (const Eigen::Quaternionf& bb_trans : bb_transformations)
	{
		transformations.push_back(bb_trans);
		for (int i = 1; i <= 3; ++i)
		{
			transformations.push_back((transformations.back() * counter_clockwise_rotation).normalized());
		}

	}

	return transformations;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: triangular_prism_classifier
//
//
/////////////////////////////////////////////////////////////

triangular_prism_classifier::triangular_prism_classifier(const object_prototype::ConstPtr& prototype)
	:
	classifier(prototype),
	slant_height(
		std::sqrt(std::powf(0.5f * prototype->get_bounding_box().diagonal(0), 2.f) 
			    + std::powf(prototype->get_bounding_box().diagonal(2), 2.f)))
{
}

classification_result triangular_prism_classifier::classify(const pc_segment& seg) const
{
	if (seg.bounding_box.diagonal(0) > prototype->get_bounding_box().diagonal(2)
		|| seg.bounding_box.diagonal(1) > prototype->get_bounding_box().diagonal(2)
		|| seg.bounding_box.diagonal(2) > std::numbers::sqrt2 * prototype->get_bounding_box().diagonal(1))
		return classification_result(prototype);

	// give differing colors only a small weight since the surface color varies a lot
	// due to specular reflection

	classification_result shape_sim = shape_classifier::classify(seg);
	classification_result color_sim = monochrome_object_classifier::classify(seg);


	float similarity = std::powf(shape_sim.local_certainty_score
		* color_sim.local_certainty_score * color_sim.local_certainty_score
		, 1.f / 3.f);

	return classification_result(prototype, shape_sim.prototype_rotation, similarity);
}

std::vector<Eigen::Quaternionf> triangular_prism_classifier::get_feasible_transformations(const pc_segment& seg) const
{
	std::vector<Eigen::Quaternionf> transformations;

	const Eigen::Quaternionf rot_90_z(Eigen::Quaternionf::FromTwoVectors(
		Eigen::Vector3f(1.f, 0.f, 0.f),
		Eigen::Vector3f(0.f, 1.f, 0.f)
	));

	const Eigen::Quaternionf rot_45_z(Eigen::Quaternionf::FromTwoVectors(
		Eigen::Vector3f(1.f, 0.f, 0.f),
		Eigen::Vector3f(1.f, 1.f, 0.f)
	));

	const Eigen::Quaternionf rot_270_y(Eigen::Quaternionf::FromTwoVectors(
		Eigen::Vector3f(0.f, 0.f, 1.f),
		Eigen::Vector3f(-1.f, 0.f, 1.f)
	));

	const Eigen::Quaternionf rot_90_y(Eigen::Quaternionf::FromTwoVectors(
		Eigen::Vector3f(1.f, 0.f, 0.f),
		Eigen::Vector3f(0.f, 0.f, -1.f)
	));

	const Eigen::Vector3f& proto_dimensions = prototype->get_bounding_box().diagonal;
	const Eigen::Vector3f& seg_dimensions = seg.bounding_box.diagonal;
	if (seg_dimensions.z() - proto_dimensions.z() > 0.5f * (proto_dimensions.z() + slant_height))
	{ // base pointing upwards


		transformations.push_back(rot_270_y);
		for (int i = 1; i <= 3; ++i)
		{
			transformations.push_back((rot_90_z * transformations.back()).normalized());
		}
	}
	else
	{ // base pointing downwards or sidewards
		Eigen::Vector3f centroid(seg.centroid.x, seg.centroid.y, seg.centroid.z);
		
		Eigen::Vector3f normalized_centroid = // transforming BB of seg to BB of prototype
			seg.bounding_box.rotation.inverse()
			* Eigen::Translation3f(-1.f * seg.bounding_box.translation)
			* centroid;

		const object_parameters& object_params = *classifier::object_params.lock();

		if (normalized_centroid.z() < 0.5f * (proto_dimensions.x() + object_params.min_object_height))
		{// base pointing downwards
			transformations.push_back(rot_90_y);
			transformations.push_back((rot_90_z * rot_90_y).normalized());
		}
		else
		{// base pointing sidewards
			double angle = std::atan2(normalized_centroid.y(), normalized_centroid.x());
			angle += 0.125 * std::numbers::pi;
			angle = angle < 0 ? angle + 2.0 / std::numbers::pi : angle;
			int rotation = angle / std::numbers::pi / 4.0;

			const Eigen::Quaternionf rot_90_x(Eigen::Quaternionf::FromTwoVectors(
				Eigen::Vector3f(0.f, 1.f, 0.f),
				Eigen::Vector3f(0.f, 0.f, 1.f)
			));


			if (rotation % 2)
			{ // bounding box aligns with the slants
			// 0 = base pointing (-1, -1)
				Eigen::Quaternionf rot = (rot_90_z.inverse() * rot_45_z.inverse() * rot_90_x).normalized();
				for (int i = 1; i <= rotation / 2; ++i)
				{
					rot = (rot_90_z * rot).normalized();
				}
				transformations.push_back(rot);

			}
			else
			{
				//bounding box aligns with the base
				Eigen::Quaternionf rot = rot_90_x;
				for (int i = 1; i <= rotation / 2; ++i)
				{
					rot = (rot_90_z * rot).normalized();
				}
				transformations.push_back(rot);
			}
		}
	}

	return transformations;
}


} //namespace state_observation


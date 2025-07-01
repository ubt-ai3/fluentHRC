#include "classification_new.hpp"

#include <pcl/filters/box_clipper3D.h>

#include <algorithm>

#include "pn_model_extension.hpp"
#include "color_space_conversion.hpp"

#ifdef _MSC_VER
#define UNREACHABLE() __assume(0)
#else
#define UNREACHABLE() __builtin_unreachable()
#endif

namespace state_observation
{

	inline float bell_curve(float x, float stdev)
	{
		return std::expf(-x * x / (2 * stdev * stdev));
	}

	inline double bell_curve(double x, double stdev)
	{
		return std::expf(-x * x / (2 * stdev * stdev));
	}

	inline Eigen::Vector3f inline_project(const Eigen::Vector3f& q0,
		const Eigen::Vector3f& p1, const Eigen::Vector3f& p2)
	{
		Eigen::Vector3f p2p1 = p2 - p1;
		float t = (q0 - p1).dot(p2p1) / p2p1.squaredNorm();
		t = std::max(0.f, std::min(1.f, t));
		return p1 + t * p2p1;
	}

	inline float inline_distance(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1,
		const Eigen::Vector3f& p2, const Eigen::Vector3f& p3)
	{
		float min_dist = std::numeric_limits<float>::infinity();

		Eigen::Vector3f normal((p2 - p1).cross(p3 - p1).normalized());

		Eigen::Vector3f p0p1 = (p0 - p1);
		float p0p1norm = p0p1.norm();
		p0p1 /= p0p1norm;

		float cosalpha = p0p1.dot(normal);
		// q0 = p0 projected into the plane spanned by p1,p2,p3
		Eigen::Vector3f q0 = p0 - (p0p1norm * cosalpha) * normal;

		Eigen::Vector3f v1 = normal.cross(p2 - p1);
		Eigen::Vector3f v2 = normal.cross(p3 - p2);
		Eigen::Vector3f v3 = normal.cross(p1 - p3);

		bool outside = false;
		float length = (q0 - p0).squaredNorm();

		if (v1.dot(q0) < v1.dot(p1))
		{
			outside = true;
			Eigen::Vector3f q1 = inline_project(q0, p1, p2);


			min_dist = std::min(min_dist, std::sqrt((q1 - q0).squaredNorm() + length));
		}
		if (v2.dot(q0) < v2.dot(p2))
		{
			outside = true;
			Eigen::Vector3f q1 = inline_project(q0, p2, p3);

			min_dist = std::min(min_dist, std::sqrt((q1 - q0).squaredNorm() + length));
		}
		if (v3.dot(q0) < v3.dot(p3))
		{
			outside = true;
			Eigen::Vector3f q1 = inline_project(q0, p3, p1);

			min_dist = std::min(min_dist, std::sqrt((q1 - q0).squaredNorm() + length));
		}

		if (!outside)
			return std::sqrt(length);
		else
			return min_dist;
	}

	const std::vector<classifier::classifier_aspect>& classifier::get_prototype_classifier()
	{
		return prototype_classifier_;
	}

	classification::monochrome_object classifier::monochrome::classify(const pc_segment& seg, pcl::RGB color)
	{
		float similarity = 0.f;
		const float stdev = 20;

		auto lab_ref = cielab(color);
		lab_ref.L /= 5;

		for (const auto& p : *seg.points) {
			auto lab = cielab(rgb(p.r, p.g, p.b));
			lab.L /= 5;

			double delta = lab.delta(lab_ref);
			similarity += bell_curve(delta, 0.12f * cielab::MAX_DELTA);

			//float sim_r = bell_curve((float)std::abs(color.r - p.r), stdev);
			//float sim_g = bell_curve((float)std::abs(color.g - p.g), stdev);
			//float sim_b = bell_curve((float)std::abs(color.b - p.b), stdev);
			//similarity += std::powf(sim_r * sim_g * sim_b, 1.f / 3.f);


		}

		return { similarity / seg.points->size() };
	}

	std::string classifier::to_string(type t)
	{
		try
		{
			return classifier::get_type_names().at(static_cast<int>(t));
		}
		catch (const std::exception&)
		{
			throw std::runtime_error("invalid enum value");
		}
	}

	classifier::type classifier::to_classifier_type(const std::string& s)
	{
		auto type_names = get_type_names();

		auto it = std::find(type_names.begin(), type_names.end(), s);
		if (it != type_names.end())
			return static_cast<type>(std::distance(type_names.begin(), it));
		else throw std::runtime_error("invalid Conversion to Enum classifier::type from String");
	}



	//std::unique_ptr<classifier::debug_data> classifier::debug = nullptr;

	std::vector<classifier::classifier_aspect> classifier::generate_aspects(const std::vector<object_prototype::ConstPtr>& prototypes)
	{
		std::vector<classifier::classifier_aspect> result;
		result.reserve(prototypes.size());

		for (const object_prototype::ConstPtr& proto : prototypes)
		{
			result.push_back({
				to_classifier_type(proto->get_type()),
				proto
				});
		}

		return result;
	}

	classifier::classifier(
		const std::shared_ptr<const object_parameters>& object_params,
		const std::vector<classifier::classifier_aspect>& aspects_to_classify)
		:
		object_params_{ object_params },
		prototype_classifier_(aspects_to_classify)
	{
		//Find all unique bounding-boxes
		//colors and meshes
		//background classifiers dont have meshes nor bounding boxes
		//but we can store references to dummy objects (nullptr and default bb)
		for (const auto& aspect : aspects_to_classify)
		{
			const auto& prototype = aspect.prototype;

			prototype->has_mesh() ?
				meshes_.push_back(prototype->get_base_mesh()->load_mesh()) :
				meshes_.push_back(nullptr);
			colors_.push_back(prototype->get_mean_color());
			bounding_boxes_.push_back(prototype->get_bounding_box());
		}
		//side effect: sorts container
		auto remove_duplicates = [](auto& container, auto less, auto equal)
		{
			std::sort(container.begin(), container.end(), less);

			auto new_end = std::unique(container.begin(), container.end(), equal);
			container.erase(new_end, container.end());
		};
		auto less_mesh = [](const pcl::PolygonMesh::ConstPtr& lhs,
			const pcl::PolygonMesh::ConstPtr& rhs)
		{
			return lhs < rhs;
		};
		auto equal_mesh = [](const pcl::PolygonMesh::ConstPtr& lhs,
			const pcl::PolygonMesh::ConstPtr& rhs)
		{
			return lhs == rhs;
		};
		remove_duplicates(meshes_, less_mesh, equal_mesh);

		auto less_color = [](pcl::RGB lhs, pcl::RGB rhs)
		{
			if (lhs.r != rhs.r)
				return lhs.r < rhs.r;
			if (lhs.g != rhs.g)
				return lhs.g < rhs.g;
			return lhs.b < rhs.b;
		};
		auto equal_color = [](pcl::RGB lhs, pcl::RGB rhs)
		{
			return lhs.r == rhs.r && lhs.g == rhs.g && lhs.b == rhs.b;
		};
		remove_duplicates(colors_, less_color, equal_color);

		auto less_box = [](const aabb& lhs, const aabb& rhs)
		{
			if (lhs.diagonal(0) != rhs.diagonal(0))
				return lhs.diagonal(0) < rhs.diagonal(0);
			if (lhs.diagonal(1) != rhs.diagonal(1))
				return lhs.diagonal(1) < rhs.diagonal(1);
			return lhs.diagonal(2) < rhs.diagonal(2);
		};
		auto equal_box = [](const aabb& lhs, const aabb& rhs)
		{
			return lhs.diagonal(0) == rhs.diagonal(0)
				&& lhs.diagonal(1) == rhs.diagonal(1)
				&& lhs.diagonal(2) == rhs.diagonal(2);
		};
		remove_duplicates(bounding_boxes_, less_box, equal_box);

		auto find_index = [](const auto& container, const auto& value, auto equal)
		{
			return static_cast<size_t> (std::distance(
				container.begin(),
				std::find_if(container.begin(), container.end(),
					[equal, value](auto other)
			{
				return equal(value, other);
			})));
		};

		//create prototype_indices from classifyer_aspects
		for (const auto& aspect : aspects_to_classify)
		{
			const auto& prototype = aspect.prototype;
			pcl::PolygonMesh::ConstPtr mesh_ptr = prototype->has_mesh() ?
				prototype->get_base_mesh()->load_mesh() : nullptr;
			indexed_prototypes_.push_back(
				prototype_index{
				find_index(bounding_boxes_, prototype->get_bounding_box(),equal_box),
				find_index(meshes_, mesh_ptr,equal_mesh),
				find_index(colors_, prototype->get_mean_color(),equal_color),
				prototype->get_name(),
				aspect.type_,
				aspect.prototype
				});
		}


	}

	std::vector<classification_result> classifier::classify_all(const pc_segment& segment) const noexcept
	{
		//auto start_time = std::chrono::high_resolution_clock::now();

		float min_object_dimension = object_params_->min_object_height;
		using transformations = std::vector<Eigen::Quaternion<float>>;
		//Plan:
		//Generate chrome classification for every color
		//Generate bounding_box_classification for every aabb
		//Generate feasible trafos for bbs
		//elementary classification

		//rotation guesses from aabbs
		std::vector<Eigen::Quaternionf> rotation_guesses;
		rotation_guesses.reserve(bounding_boxes_.size());
		for (const auto& box : bounding_boxes_)
			rotation_guesses.push_back(bounding_box::stacked_rotation_guess(segment, box));

		//bb_clas from aabbs + rotation_guesses
		std::vector<classification::bounding_box> bb_results;
		bb_results.reserve(bounding_boxes_.size());
		for (int i = 0; i < bounding_boxes_.size(); i++)
		{
			bb_results.push_back(bounding_box::classify(
				segment,
				bounding_boxes_[i],
				rotation_guesses[i],
				min_object_dimension));
		}

		//chrome_clas from colors
		std::vector<classification::monochrome_object> color_results;
		color_results.reserve(colors_.size());
		for (const auto& color : colors_)
			color_results.push_back(monochrome::classify(segment, color));

		//bb_trafo from rotation_guesses
		std::vector<transformation::bounding_box> bb_trafos;
		bb_trafos.reserve(bounding_boxes_.size());
		for (const auto& guess : rotation_guesses)
			bb_trafos.push_back(bounding_box::get_feasible_transformation(guess));

		std::vector<classification_result> return_val;
		return_val.reserve(indexed_prototypes_.size());

		bool is_background = false;

		//auto dur = std::chrono::high_resolution_clock::now() - start_time;
		//std::cout << std::chrono::duration<float,std::milli>(dur).count();

		for (const auto& indexed_prototype : indexed_prototypes_)
		{
			switch (indexed_prototype.type_)
			{
			case type::background:
			{
				return_val.push_back(background::classify(segment, color_results[indexed_prototype.color], indexed_prototype.prototype).result);
				if (return_val.back().local_certainty_score > 0.5)
				{
					is_background = true;
				}
				//return return_val;
				break;
			}
			case type::bridge:
			{
				//needs own feasible transformations and own shape_clas
				auto bridge_shape_clas = shape::classify(segment,
					bounding_boxes_[indexed_prototype.box],
					*meshes_[indexed_prototype.mesh],
					bridge::get_feasible_transformations(
						segment,
						bb_trafos[indexed_prototype.box],
						min_object_dimension).transformations,
					min_object_dimension);

				return_val.push_back(bridge::classify(
					segment,
					bb_results[indexed_prototype.box],
					color_results[indexed_prototype.color],
					bridge_shape_clas,
					indexed_prototype.prototype,
					min_object_dimension).result);

				break;
			}
			case type::cuboid:
			{
				auto shape_clas = shape::classify(segment,
					bounding_boxes_[indexed_prototype.box],
					*meshes_[indexed_prototype.mesh],
					bb_trafos[indexed_prototype.box].transformations,
					min_object_dimension);

				return_val.push_back(cuboid::classify(
					segment,
					bb_results[indexed_prototype.box],
					color_results[indexed_prototype.color],
					shape_clas,
					indexed_prototype.prototype,
					min_object_dimension).result);
				break;
			}
			case type::cylinder:
			{
				auto shape_clas = shape::classify(segment,
					bounding_boxes_[indexed_prototype.box],
					*meshes_[indexed_prototype.mesh],
					bb_trafos[indexed_prototype.box].transformations,
					min_object_dimension);

				return_val.push_back(cylinder::classify(
					segment,
					bb_results[indexed_prototype.box],
					color_results[indexed_prototype.color],
					shape_clas,
					indexed_prototype.prototype,
					min_object_dimension).result);
				break;
			}
			case type::semicylinder:
			{
				//needs own feasible transformations and shape_clas
				auto semicylinder_shape = shape::classify(
					segment,
					bounding_boxes_[indexed_prototype.box],
					*meshes_[indexed_prototype.mesh],
					semicylinder::get_feasible_transformations(segment, bb_trafos[indexed_prototype.box]).transformations,
					min_object_dimension);

				return_val.push_back(
					semicylinder::classify(
						segment,
						bb_results[indexed_prototype.box],
						color_results[indexed_prototype.color],
						semicylinder_shape,
						indexed_prototype.prototype,
						min_object_dimension
					).result);
				break;
			}

			case type::triangular_prism:
			{
				//needs own feasible transformations and shape_clas
				float slant_height = std::sqrt(std::powf(0.5f * bounding_boxes_[indexed_prototype.box].diagonal(0), 2.f)
					+ std::powf(bounding_boxes_[indexed_prototype.box].diagonal(2), 2.f));

				auto triangular_prism_shape = shape::classify(
					segment,
					bounding_boxes_[indexed_prototype.box],
					*meshes_[indexed_prototype.mesh],
					triangular_prism::get_feasible_transformations(
						segment,
						bounding_boxes_[indexed_prototype.box],
						slant_height,
						min_object_dimension).transformations,
					min_object_dimension);

				return_val.push_back(
					triangular_prism::classify(
						segment,
						bb_results[indexed_prototype.box],
						color_results[indexed_prototype.color],
						triangular_prism_shape,
						indexed_prototype.prototype,
						slant_height,
						min_object_dimension).result);
				break;
			}
			default: UNREACHABLE(); //will be available with C++23 -> https://en.cppreference.com/w/cpp/utility/unreachable
			}
			//std::cout << "*";
			if (is_background)
				break;
		}
		/*//transformations per semicylinder //TODO for later
		std::vector<transformation::semicylinder> semicylinder_trafos;
		semicylinder_trafos.reserve()*/

		return return_val;
	}

	classification_result classifier::classify(const pc_segment& segment, const classifier::classifier_aspect& classifier, float min_object_height)
	{
		return classify(segment,
			classifier,
			min_object_height,
			bounding_box::stacked_rotation_guess(segment, classifier.prototype->get_bounding_box()));
	}

	classification_result classifier::classify(const pc_segment& segment, const classifier::classifier_aspect& classifier, float min_object_dimension, const Eigen::Quaternionf& rotation_guess)
	{
		auto& prototype = *classifier.prototype;
		auto color_classification = monochrome::classify(segment, prototype.get_mean_color());
		auto bounding_box_classification = bounding_box::classify(
			segment, prototype.get_bounding_box(), rotation_guess, min_object_dimension);
		auto feasible_transformations = bounding_box::get_feasible_transformation(rotation_guess);

		switch (classifier.type_)
		{
		case type::background:
		{
			return background::classify(segment, color_classification, classifier.prototype).result;
		}
		case type::bridge:
		{
			auto bridge_shape_clas = shape::classify(segment,
				prototype.get_bounding_box(),
				*prototype.load_mesh(),
				bridge::get_feasible_transformations(
					segment,
					feasible_transformations,
					min_object_dimension).transformations,
				min_object_dimension);

			return bridge::classify(
				segment,
				bounding_box_classification,
				color_classification,
				bridge_shape_clas,
				classifier.prototype,
				min_object_dimension).result;
		}
		case type::cuboid:
		{
			auto shape_clas = shape::classify(segment,
				prototype.get_bounding_box(),
				*prototype.load_mesh(),
				feasible_transformations.transformations,
				min_object_dimension);

			return cuboid::classify(
				segment,
				bounding_box_classification,
				color_classification,
				shape_clas,
				classifier.prototype,
				min_object_dimension).result;
			break;
		}
		case type::cylinder:
		{
			auto shape_clas = shape::classify(segment,
				prototype.get_bounding_box(),
				*prototype.load_mesh(),
				feasible_transformations.transformations,
				min_object_dimension);

			return cylinder::classify(
				segment,
				bounding_box_classification,
				color_classification,
				shape_clas,
				classifier.prototype,
				min_object_dimension).result;
			break;
		}
		case type::semicylinder:
		{
			//needs own feasible transformations and shape_clas
			auto semicylinder_shape = shape::classify(
				segment,
				prototype.get_bounding_box(),
				*prototype.load_mesh(),
				semicylinder::get_feasible_transformations(segment, feasible_transformations).transformations,
				min_object_dimension);

			return semicylinder::classify(
				segment,
				bounding_box_classification,
				color_classification,
				semicylinder_shape,
				classifier.prototype,
				min_object_dimension
			).result;
			break;
		}
		case type::triangular_prism:
		{
			//needs own feasible transformations and shape_clas
			float slant_height = std::sqrt(std::powf(0.5f * prototype.get_bounding_box().diagonal(0), 2.f)
				+ std::powf(prototype.get_bounding_box().diagonal(2), 2.f));

			auto triangular_prism_shape = shape::classify(
				segment,
				prototype.get_bounding_box(),
				*prototype.load_mesh(),
				triangular_prism::get_feasible_transformations(
					segment,
					prototype.get_bounding_box(),
					slant_height,
					min_object_dimension).transformations,
				min_object_dimension);

			return
				triangular_prism::classify(
					segment,
					bounding_box_classification,
					color_classification,
					triangular_prism_shape,
					classifier.prototype,
					slant_height,
					min_object_dimension).result;
			break;
		}
		UNREACHABLE();
		}
		UNREACHABLE();
	}

	classification::background classifier::background::classify(
		const pc_segment& seg,
		const classification::monochrome_object& monochrome,
		const object_prototype::ConstPtr& prototype)
	{
		unsigned int max_x = 0, max_y = 0;
		unsigned int min_x, min_y;
		min_x = min_y = std::numeric_limits<unsigned int>::max();

		for (int i : seg.indices->indices)
		{
			min_x = std::min(min_x, i % seg.reference_frame->width);
			min_y = std::min(min_y, i / seg.reference_frame->width);
			max_x = std::max(max_x, i % seg.reference_frame->width);
			max_y = std::max(max_y, i / seg.reference_frame->width);
		}

		float max_points = (max_x - min_x + 1) * (max_y - min_y + 1);

		float similarity = monochrome.similarity *
			bell_curve(seg.indices->indices.size(), 0.5f * max_points);

		return { classification_result{ prototype, Eigen::Quaternionf::Identity(), similarity } };
	}

	classification::cuboid classifier::cuboid::classify(
		const pc_segment& seg,
		const classification::bounding_box& bounding_box_classification,
		const classification::monochrome_object& monochrome_classification,
		const classification::shape& shape_classification,
		const object_prototype::ConstPtr& prototype,
		float min_object_height)

	{
		float min_height = 0;
		//if (seg.bounding_box.diagonal(0) < prototype->get_bounding_box().diagonal(2) - min_object_height
		//	&& seg.bounding_box.diagonal(1) < prototype->get_bounding_box().diagonal(2) - min_object_height
		//	&& seg.bounding_box.diagonal(2) < prototype->get_bounding_box().diagonal(2) - min_object_height)
		//	return { classification_result(prototype) };

		const obb& box = seg.bounding_box;


		if (box.diagonal.z() > min_object_height + box.diagonal.x()
			&& box.diagonal.z() > min_object_height + box.diagonal.y())
		{ // object standing, use outline for shape comparison

			std::vector<cv::Point2f> outline;
			outline.reserve(seg.get_outline()->size());

			//transform outline such that bounding box is centered in origin and axis aligned
			for (const auto& p : *seg.get_outline())
			{
				Eigen::Vector3f untransformed(p.x, p.y, p.z);
				Eigen::Vector3f transformed = box.rotation.inverse() * (untransformed - box.translation);
				outline.push_back(cv::Point2f(transformed.x(), transformed.y()));
			}

			//measure similarity by the distance points on the outline to their enclosing rectangle
			float similarity = 1.f;
			for (const cv::Point2f& p : outline)
			{
				float dist_x = std::abs(std::abs(p.x) - box.diagonal(0) / 2.f);
				float dist_y = std::abs(std::abs(p.y) - box.diagonal(1) / 2.f);
				similarity *= bell_curve(std::min(dist_x, dist_y), min_object_height);
			}
			float shape_sim = std::pow(similarity, 1. / outline.size());

			similarity = std::powf(bounding_box_classification.similarity
				* monochrome_classification.similarity * monochrome_classification.similarity
				* shape_sim
				, 1.f / 4.f);

			return { classification_result(prototype, bounding_box_classification.rotation, similarity) };
		}
		else
		{
			float similarity = std::powf(shape_classification.similarity
				* bounding_box_classification.similarity
				* monochrome_classification.similarity
				, 1.f / 3.f);

			return { classification_result(prototype, bounding_box_classification.rotation, similarity) };
		}
	}

	classification::bounding_box classifier::bounding_box::classify(const pc_segment& seg,
		const aabb& bounding_box,
		const Eigen::Quaternionf& rotation_guess,
		float object_min_dimension)
	{
		Eigen::Matrix3f similarities(stacked_similarity_matrix(seg, bounding_box, object_min_dimension));
		Eigen::Matrix3f temp = similarities.cwiseProduct(Eigen::Matrix3f(rotation_guess));
		float det = temp.determinant();
		return { rotation_guess, std::abs(det) };
	}

	Eigen::Quaternionf classifier::bounding_box::stacked_rotation_guess(const pc_segment& segment, const aabb& bounding_box)
	{
		const auto& seg_diag = segment.bounding_box.diagonal;
		const auto& box_diag = bounding_box.diagonal;
		if (box_diag.x() == box_diag.y() && box_diag.y() == box_diag.z())
			return Eigen::Quaternionf::Identity();

		Eigen::Matrix3f rotation(Eigen::Matrix3f::Zero());
		std::list<int> seg_dimensions({ 2,1,0 });

		for (int i = 0; i < 3; i++)
		{
			float min_diff = std::numeric_limits<float>::infinity();
			std::list<int>::iterator min_j;

			for (auto j = seg_dimensions.begin(); j != seg_dimensions.end(); ++j)
			{
				if (std::abs(segment.bounding_box.diagonal(i) - bounding_box.diagonal(*j)) < min_diff)
				{
					min_diff = std::abs(segment.bounding_box.diagonal(i) - bounding_box.diagonal(*j));
					min_j = j;
				}

			}

			rotation(i, *min_j) = 1.f;
			seg_dimensions.erase(min_j);
		}

		rotation.col(2) = rotation.col(0).cross(rotation.col(1));

		return Eigen::Quaternionf(rotation).normalized();
	}


	Eigen::Matrix3f classifier::bounding_box::stacked_similarity_matrix(const pc_segment& segment,
		const aabb& bounding_box,
		float object_min_dimension)
	{
		Eigen::Matrix3f similarities(Eigen::Matrix3f::Zero());

		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 3; ++j) {
				similarities(i, j) = bell_curve(segment.bounding_box.diagonal(i) - bounding_box.diagonal(j),
					object_min_dimension);
			}
		}

		for (int j = 0; j < 3; ++j) {
			if (segment.bounding_box.diagonal(2) > bounding_box.diagonal(j))
				similarities(2, j) = 1.f;
			else
				similarities(2, j) = bell_curve(segment.bounding_box.diagonal(2) - bounding_box.diagonal(j),
					object_min_dimension);
		}

		return similarities;
	}


	Eigen::Matrix3f classifier::bounding_box::similarity_matrix(const pc_segment& seg,
		const aabb& bounding_box,
		float object_min_dimension)
	{
		Eigen::Matrix3f similarities(Eigen::Matrix3f::Zero());

		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 3; ++j) {
				similarities(i, j) = bell_curve(seg.bounding_box.diagonal(i) - bounding_box.diagonal(j),
					object_min_dimension);
			}
		}

		return similarities;
	}

	Eigen::Quaternionf classifier::bounding_box::rotation_guess(const pc_segment& segment, const aabb& bounding_box)
	{
		return rotation_guess(segment.bounding_box, bounding_box);
	}

	Eigen::Quaternionf classifier::bounding_box::rotation_guess(const obb& seg, const aabb& bounding_box)
	{
		Eigen::Matrix3f rotation(Eigen::Matrix3f::Zero());
		std::list<int> seg_dimensions({ 2,1,0 });
		for (int i = 2; i >= 0; --i)
			// match height first since other dimensions might be shorter
			// due to discarded samples close to the table
		{
			float min_diff = std::numeric_limits<float>::infinity();
			std::list<int>::iterator min_j;

			for (auto j = seg_dimensions.begin(); j != seg_dimensions.end(); ++j)
			{
				if (std::abs(seg.diagonal(i) - bounding_box.diagonal(*j)) < min_diff)
				{
					min_diff = std::abs(seg.diagonal(i) - bounding_box.diagonal(*j));
					min_j = j;
				}

			}

			rotation(i, *min_j) = 1.f;
			seg_dimensions.erase(min_j);
		}

		rotation.col(2) = rotation.col(0).cross(rotation.col(1));

		return Eigen::Quaternionf(rotation).normalized();
	}

	transformation::bounding_box classifier::bounding_box::get_feasible_transformation(const Eigen::Quaternionf& rotation_guess)
	{
		std::vector<Eigen::Quaternionf> transformations;
		transformations.push_back(rotation_guess);
		return { transformations };
	}

	classification::cylinder classifier::cylinder::classify(const pc_segment& seg,
		const classification::bounding_box& box_classification,
		const classification::monochrome_object& monochrome_classification,
		const classification::shape& shape_classification,
		const object_prototype::ConstPtr prototype,
		float min_object_height)
	{

		if (seg.bounding_box.diagonal(0) < prototype->get_bounding_box().diagonal(2) - min_object_height
			&& seg.bounding_box.diagonal(1) < prototype->get_bounding_box().diagonal(2) - min_object_height
			&& seg.bounding_box.diagonal(2) < prototype->get_bounding_box().diagonal(2) - min_object_height)
			return { classification_result(prototype) };

		const obb& box = seg.bounding_box;
		float width = box.diagonal(0);
		float breadth = box.diagonal(1);

		if (box.diagonal.z() > min_object_height + width
			&& box.diagonal.z() > min_object_height + breadth)
		{ // object standing, use outline for shape comparison

		// Point cloud to vector
			std::vector<cv::Point2f> outline;
			outline.reserve(seg.get_outline()->size());

			const auto& seg_outline = seg.get_outline()->size() ? *seg.get_outline() : *seg.points;
			for (const auto& p : seg_outline)
			{
				outline.push_back(cv::Point2f(p.x, p.y));
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
				similarity *= bell_curve(dist, min_object_height);
			}
			float shape_sim = std::pow(similarity, 1. / outline.size());

			similarity = std::powf(box_classification.similarity
				* monochrome_classification.similarity * monochrome_classification.similarity
				* shape_sim
				, 1.f / 4.f);

			return { classification_result(prototype, box_classification.rotation, similarity) };
		}
		else
		{

			float similarity = std::powf(shape_classification.similarity
				* box_classification.similarity
				* monochrome_classification.similarity * monochrome_classification.similarity
				, 1.f / 4.f);

			return { classification_result(prototype, shape_classification.rotation, similarity) };
		}
	}

	classification::semicylinder classifier::semicylinder::classify(const pc_segment& seg,
		const classification::bounding_box& box_classification,
		const classification::monochrome_object& monochrome_classification,
		const classification::shape& shape_classification,
		const object_prototype::ConstPtr prototype,
		float min_object_height)
	{
		if (seg.bounding_box.diagonal(0) > prototype->get_bounding_box().diagonal(0) + min_object_height
			|| seg.bounding_box.diagonal(1) > prototype->get_bounding_box().diagonal(0) + min_object_height
			|| seg.bounding_box.diagonal(2) > prototype->get_bounding_box().diagonal(0) + min_object_height)
			return { classification_result(prototype) };

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
			box_sim = box_classification.similarity;
		}

		float similarity = std::powf(shape_classification.similarity
			* box_sim
			* monochrome_classification.similarity * monochrome_classification.similarity
			, 1.f / 4.f);

		return { classification_result(prototype, shape_classification.rotation, similarity) };
	}

	classification::triangular_prism classifier::triangular_prism::classify(const pc_segment& seg,
		const classification::bounding_box& box_classification,
		const classification::monochrome_object& monochrome_classification,
		const classification::shape& shape_classification,
		const object_prototype::ConstPtr prototype,
		float slant_height,
		float min_object_height)
	{
		//TODO should be diagonal zero?
		if (seg.bounding_box.diagonal(0) > prototype->get_bounding_box().diagonal(2)
			|| seg.bounding_box.diagonal(1) > prototype->get_bounding_box().diagonal(2)
			|| seg.bounding_box.diagonal(2) > M_SQRT2 * prototype->get_bounding_box().diagonal(1))
			return { classification_result(prototype) };

		// give differing colors only a small weight since the surface color varies a lot
		// due to specular reflection

		float similarity = std::powf(shape_classification.similarity
			* monochrome_classification.similarity * monochrome_classification.similarity
			, 1.f / 3.f);

		return { classification_result(prototype, shape_classification.rotation, similarity) };
	}

	transformation::triangular_prism classifier::triangular_prism::get_feasible_transformations(
		const pc_segment& seg,
		const aabb& bounding_box,
		float slant_height,
		float min_object_height)
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

		const Eigen::Vector3f& proto_dimensions = bounding_box.diagonal;
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

			if (normalized_centroid.z() < 0.5f * (proto_dimensions.x() + min_object_height))
			{// base pointing downwards
				transformations.push_back(rot_90_y);
				transformations.push_back((rot_90_z * rot_90_y).normalized());
			}
			else
			{// base pointing sidewards
				float angle = std::atan2(normalized_centroid.y(), normalized_centroid.x());
				angle += 0.125f * M_PI;
				angle = angle < 0 ? angle + 2 * M_PI : angle;
				int rotation = angle / M_PI_4;

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

		return { transformations };
	}

	classification::bridge classifier::bridge::classify(const pc_segment& seg,
		const classification::bounding_box& box_classification,
		const classification::monochrome_object& monochrome_classification,
		const classification::shape& shape_classification,
		const object_prototype::ConstPtr prototype,
		float min_object_height)
	{
		return { cuboid::classify(
			seg,
			box_classification,
			monochrome_classification,
			shape_classification,
			prototype,
			min_object_height).result };
	}

	transformation::bridge classifier::bridge::get_feasible_transformations(const pc_segment& seg,
		const transformation::bounding_box& bb_transformations, float min_object_height)
	{
		std::vector<Eigen::Quaternionf> transformations;

		const Eigen::Quaternionf counter_clockwise_rotation(Eigen::Quaternionf::FromTwoVectors(
			Eigen::Vector3f(1.f, 0.f, 0.f),
			Eigen::Vector3f(0.f, 1.f, 0.f)
		));

		for (const Eigen::Quaternionf& bb_trans : bb_transformations.transformations)
		{
			transformations.push_back(bb_trans);
			for (int i = 1; i <= 3; ++i)
			{
				transformations.push_back((transformations.back() * counter_clockwise_rotation).normalized());
			}

		}

		return { transformations };
	}


	classification::shape classifier::shape::classify(const pc_segment& seg,
		const aabb& bounding_box,
		const pcl::PolygonMesh& prototype_mesh,
		const std::vector<Eigen::Quaternionf>& feasible_transformations,
		float min_object_height)
	{

		float best_similarity = 0.f;
		Eigen::Quaternionf best_rotation;

		for (const Eigen::Quaternionf& rotation : feasible_transformations)
		{
			float similarity = cached_match(seg, bounding_box, prototype_mesh, rotation, min_object_height);


			if (best_similarity < similarity)
			{
				best_similarity = similarity;
				best_rotation = rotation;
			}
		}

		return { best_rotation, best_similarity };
	}

	float classifier::shape::match(const pc_segment& seg,
		const aabb& bounding_box,
		const pcl::PolygonMesh& prototype_mesh,
		const Eigen::Quaternionf& prototype_rotation,
		float min_object_height)
	{

		Eigen::Affine3f transform = Eigen::Translation3f(seg.bounding_box.translation) *
			Eigen::Affine3f(seg.bounding_box.rotation) * Eigen::Affine3f(prototype_rotation) *
			Eigen::Scaling(0.5f * bounding_box.diagonal);

		const pcl::PolygonMesh& mesh = prototype_mesh;
		float similarity = 0.f;

		auto get_point =
			[&data = mesh.cloud.data, point_step = mesh.cloud.point_step, &transform](int index)->Eigen::Vector3f
		{
			return transform * Eigen::Vector3f(reinterpret_cast<const float*>(&data[point_step * index]));
		};

		const uint8_t* data = mesh.cloud.data.data();
		int point_step = mesh.cloud.point_step;
		auto& pc = *seg.points;
		for (const auto& p : pc)
		{
			float dist = std::numeric_limits<float>::infinity();
			Eigen::Vector3f p0 = p.getVector3fMap();

			for (const pcl::Vertices& polygon : prototype_mesh.polygons)
			{
				/*dist = std::min(dist, inline_distance(
					p0,
					get_point(polygon.vertices[0]),
					get_point(polygon.vertices[1]),
					get_point(polygon.vertices[2])));*/

				dist = std::min(dist, inline_distance(
					p0,
					transform * *reinterpret_cast<const Eigen::Vector3f*>(data + point_step * polygon.vertices[0]),
					transform * *reinterpret_cast<const Eigen::Vector3f*>(data + point_step * polygon.vertices[1]),
					transform * *reinterpret_cast<const Eigen::Vector3f*>(data + point_step * polygon.vertices[2])
				));
				if (dist < 0) [[unlikely]]
					break;
			}

			similarity += bell_curve(dist, 0.5f * min_object_height);

		}

		return similarity / seg.points->size();
	}

	std::vector<std::string> classifier::get_type_names()
	{
		return std::vector<std::string>({
			"background",
				"cylinder",
				"semicylinder",
				"cuboid",
				"bridge",
				"triangular_prism"
			});
	}

	float classifier::shape::cached_match(const pc_segment& seg,
		const aabb& bounding_box,
		const pcl::PolygonMesh& prototype_mesh,
		const Eigen::Quaternionf& prototype_rotation,
		float min_object_height)
	{
		static std::mutex cache_mutex;
		std::lock_guard<std::mutex> lock(cache_mutex);
		//code smell: static variables
		static int _cache_misses = 0;
		static int _cache_hits = 0;
		struct parameter_pack
		{
			//keys
			const pc_segment* segment;
			const aabb* bounding_box;
			const pcl::PolygonMesh* mesh;
			Eigen::Quaternionf rotation;
			float min_object_height;

			//value
			float result;
			bool operator==(const parameter_pack& other) const
			{
				return segment == other.segment //ptr
					&& bounding_box == other.bounding_box //ptr
					&& mesh == other.mesh //ptr
					&& rotation.angularDistance(other.rotation) < 0.001 //value approx
					&& min_object_height == other.min_object_height; //value identity
			}
		};
		static std::vector<parameter_pack> cached_results;

		parameter_pack pack{ &seg,&bounding_box,&prototype_mesh,prototype_rotation,min_object_height };

		auto it = std::find(cached_results.begin(), cached_results.end(), pack);
		if (it != cached_results.end())
		{
			_cache_hits++;
			return it->result;
		}
		else
		{
			if (cached_results.size() && cached_results[0].segment != &seg)
			{
				//new segment processed; results of old segments isnt needed anymore
				cached_results.clear();
				_cache_misses = 0;
				_cache_hits = 0;
			}

			_cache_misses++;
			float result = classifier::shape::match(seg, bounding_box, prototype_mesh, prototype_rotation, min_object_height);
			pack.result = result;
			cached_results.push_back(pack);
			return result;
		}
	}


	Eigen::Vector3f utility::project(const Eigen::Vector3f& q0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2)
	{
		return inline_project(q0, p1, p2);
	}


	float utility::distance(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1,
		const Eigen::Vector3f& p2, const Eigen::Vector3f& p3)
	{
		return inline_distance(p0, p1, p2, p3);
	}

	transformation::semicylinder classifier::semicylinder::get_feasible_transformations(const pc_segment& seg, const transformation::bounding_box& bounding_box_trafos)
	{
		std::vector<Eigen::Quaternionf> transformations;
		const Eigen::Quaternionf counter_clockwise_rotation(Eigen::Quaternionf::FromTwoVectors(
			Eigen::Vector3f(1.f, 0.f, 0.f),
			Eigen::Vector3f(0.f, 1.f, 0.f)
		));

		for (const Eigen::Quaternionf& bb_trans : bounding_box_trafos.transformations)
		{
			transformations.push_back(bb_trans);
			if ((bb_trans * Eigen::Vector3f(0.f, 0.f, 1.f))(2) > 0.9f) //curve pointing upwards
			{
				transformations.push_back((bb_trans * counter_clockwise_rotation).normalized());
			}

		}

		return { transformations };
	}

	pcl::PointCloud<pc_segment::PointT>::ConstPtr classifier::clip(const pcl::PointCloud<pc_segment::PointT>::ConstPtr& cloud, const obb& box) const
	{
		Eigen::Affine3f oobb_pose;
		oobb_pose.setIdentity();// (-1.f * boxed_place.box.translation);
		float tolerance = this->object_params_->min_object_height;
		Eigen::Vector3f tolerance3f(tolerance, tolerance, 0.f);
		oobb_pose = Eigen::Translation3f(0.f, 0.f, tolerance)*
			Eigen::Translation3f(box.translation) *
			Eigen::Affine3f(box.rotation) *
			Eigen::Scaling(0.5 * box.diagonal + tolerance3f);

		Eigen::Affine3f inverse = oobb_pose.inverse();
		oobb_pose = inverse;

		pcl::BoxClipper3D<pcl::PointXYZ> oobb_clipper(oobb_pose);

		float min_object_height = object_params_->min_object_height;

		auto clipped_points = pcl::make_shared<pcl::PointCloud<pc_segment::PointT>>();
		for (auto& point : cloud->points)
		{
			bool inside = oobb_clipper.clipPoint3D(pcl::PointXYZ(point.x, point.y, point.z));
			Eigen::Vector4f point_coordinates(oobb_pose.matrix() * point.getVector4fMap());
			Eigen::Vector4f t = point_coordinates.array().abs();

			if (inside)
				clipped_points->push_back(point);
		}

		return clipped_points;
	}

	classification_result classifier::classify_box_in_segment(const pc_segment& segment, const pn_boxed_place& boxed_place,
		const classifier::classifier_aspect& classifier) const
	{
		//classify objects in a segment consisting of multiple smaller objects
		//it is known at which position objects have to be within the segment

		//clip segment outside oobb
		//and classify it

		Eigen::Affine3f oobb_pose;
		oobb_pose.setIdentity();// (-1.f * boxed_place.box.translation);
		float tolerance = this->object_params_->min_object_height;
		Eigen::Vector3f tolerance3f(tolerance, tolerance, tolerance);
		oobb_pose = Eigen::Translation3f(boxed_place.box.translation) *
			Eigen::Affine3f(boxed_place.box.rotation) *
			Eigen::Scaling(0.5 * boxed_place.box.diagonal + tolerance3f);

		Eigen::Affine3f inverse = oobb_pose.inverse();
		oobb_pose = inverse;

		pcl::BoxClipper3D<pcl::PointXYZ> oobb_clipper(oobb_pose);

		float min_object_height = object_params_->min_object_height;

		auto clipped_points = pcl::make_shared<pcl::PointCloud<pc_segment::PointT>>();
		for (auto& point : *segment.points)
		{
			bool inside = oobb_clipper.clipPoint3D(pcl::PointXYZ(point.x, point.y, point.z));
			Eigen::Vector4f point_coordinates(oobb_pose.matrix() * point.getVector4fMap());
			Eigen::Vector4f t = point_coordinates.array().abs();

			if (inside)
				clipped_points->push_back(point);
		}
		pc_segment clipped = segment;
		pcl::PointCloud<pcl::PointXYZRGBA> c;
		clipped.points = clipped_points;
		//clipped subsegment doesnt have new centroid or bb yet, in case of later use
		return classify(clipped, classifier, 0.5 * min_object_height);
	}

}//namespace state_observation

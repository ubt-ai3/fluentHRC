#pragma once

#ifndef STATE_OBSERVATION__CLASSIFICATION__NEW__HPP
#define STATE_OBSERVATION__CLASSIFICATION__NEW__HPP

#include <memory>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "eigen_serialization/eigen_serialization.hpp"
#include <Eigen/Core>

#include <pcl/point_types.h>

#include "workspace_objects.hpp"
#include "pointcloud_util.hpp"

namespace state_observation {
	
	class pn_boxed_place;

		namespace {
			namespace classification
			{
				struct bounding_box
				{
					Eigen::Quaternionf rotation;
					float similarity;
				};
				struct monochrome_object
				{
					float  similarity;
				};
				struct shape
				{
					Eigen::Quaternionf rotation;
					float similarity;
				};
				struct background
				{
					classification_result result;
				};

				struct cuboid
				{
					classification_result result;
				};
				struct cylinder
				{
					classification_result result;
				};
				struct semicylinder
				{
					classification_result result;
				};
				struct triangular_prism
				{
					classification_result result;
				};
				struct bridge
				{
					classification_result result;
				};
			}

			using transformations_t = std::vector<Eigen::Quaternionf>;

			/*  strong typing to indicate which function needs which precomputed feasible_transformations*/
			namespace transformation
			{
				struct bounding_box
				{
					std::vector<Eigen::Quaternionf> transformations;
				};
				struct semicylinder
				{
					std::vector<Eigen::Quaternionf> transformations;
				};
				struct bridge
				{
					std::vector<Eigen::Quaternionf> transformations;
				};
				struct triangular_prism
				{
					std::vector<Eigen::Quaternionf> transformations;
				};
			}
		}

		/**
	 * @class classifier
	 * @brief Utility class for advanced object classification
	 *
	 * Provides a comprehensive interface and utility functions for classifying
	 * point cloud segments against multiple object prototypes and types. Supports
	 * bounding box, shape, color, and type-based classification, as well as
	 * transformation estimation and prototype management.
	 *
	 * Features:
	 * - Multi-type classification (background, cuboid, cylinder, etc.)
	 * - Prototype and aspect management
	 * - Bounding box and shape matching
	 * - Feasible transformation estimation
	 * - Color and mesh-based classification
	 * - Debugging and utility functions
	 */
		class STATEOBSERVATION_API classifier
		{
		public:

			enum class type
			{
				background,
				cylinder,
				semicylinder,
				cuboid,
				bridge,
				triangular_prism
			};

			// DLL import does not allow static data members
			static std::vector<std::string> get_type_names();
			static std::string to_string(type t);
			static type to_classifier_type(const std::string& s);



			struct classifier_aspect
			{
				classifier::type type_;
				object_prototype::ConstPtr prototype;				
			};

			static std::vector<classifier::classifier_aspect> generate_aspects(const std::vector<object_prototype::ConstPtr>& prototypes);


			classifier() = default;
			classifier(const std::shared_ptr<const object_parameters>& object_params,
				const std::vector<classifier::classifier_aspect>& aspects_to_classify);

			std::vector<classification_result> classify_all(const pc_segment& segment) const noexcept;

			static classification_result classify(
				const pc_segment& segment,
				const classifier::classifier_aspect& classifier,float min_object_height);

			static classification_result classify(
				const pc_segment& segment,
				const classifier::classifier_aspect& classifier, 
				float min_object_height,
				const Eigen::Quaternionf& rotation);

			pcl::PointCloud<pc_segment::PointT>::ConstPtr clip(const pcl::PointCloud<pc_segment::PointT>::ConstPtr& cloud, const obb& box) const;

			classification_result classify_box_in_segment(const pc_segment& segment,
				const pn_boxed_place& boxed_place,
				const classifier::classifier_aspect& prototype) const;


			const std::vector<classifier_aspect>& get_prototype_classifier();
	
			struct bounding_box
			{
				static classification::bounding_box classify(const pc_segment& segment,
					const aabb& bounding_box,
					const Eigen::Quaternionf& rotation_guess,
					float object_min_dimension);

				/**
				* An entry a_ij \in [0,1] of the returned matrix states whether the BB of seg
				* in dimension i has approximately the same length as the BB of this in dimension j
				*/
				static Eigen::Matrix3f similarity_matrix(const pc_segment& segment,
					const aabb& bounding_box,
					float object_min_dimension);
				static Eigen::Matrix3f stacked_similarity_matrix(const pc_segment& segment,
					const aabb& bounding_box,
					float object_min_dimension);
				static Eigen::Quaternionf rotation_guess(const obb& segment, const aabb& bounding_box);
				static Eigen::Quaternionf rotation_guess(const pc_segment& segment, const aabb& bounding_box);
				static Eigen::Quaternionf stacked_rotation_guess(const pc_segment& segment, const aabb& bounding_box);


				/**
				* generates all feasible transformations for @ref{seg}
				* based on the BBs of both objects. A transformation is feasible if
				* - the length of the bounding boxes of @ref{prototype} and @ref{seg} approximately match in every dimension
				* - the vector contains no other transformation that leads to a symmetrical result
				*
				* assumes symmetrical objects.
				*/
				static transformation::bounding_box get_feasible_transformation(const Eigen::Quaternionf& rotation_guess);
			};
		
		
		private:

			std::shared_ptr<const object_parameters> object_params_;

			struct prototype_index
			{
				size_t box;
				size_t mesh;
				size_t color;
				std::string name;
				type type_;
				object_prototype::ConstPtr prototype;
			};

			//const after constructing
			std::vector<aabb> bounding_boxes_;
			std::vector<pcl::PolygonMeshConstPtr> meshes_;
			std::vector<pcl::RGB> colors_;

			std::vector<prototype_index> indexed_prototypes_;

			std::vector<classifier_aspect> prototype_classifier_;

			struct monochrome
			{
				static classification::monochrome_object classify(const pc_segment& segment, pcl::RGB color);
			};



			struct shape
			{
				static classification::shape classify(const pc_segment& seg,
					const aabb& bounding_box,
					const pcl::PolygonMesh& prototype_mesh,
					const std::vector<Eigen::Quaternionf>& feasible_transformations,
					float min_object_height);

				static float match(const pc_segment& seg,
					const aabb& bounding_box,
					const pcl::PolygonMesh& prototype_mesh,
					const Eigen::Quaternionf& prototype_rotation,
					float min_object_height);

				static float cached_match(const pc_segment& seg,
					const aabb& bounding_box,
					const pcl::PolygonMesh& prototype_mesh,
					const Eigen::Quaternionf& prototype_rotation,
					float min_object_height);
				
			};

			//Higher Level Classifers
			struct background
			{
				static classification::background classify(
					const pc_segment& seg,
					const classification::monochrome_object& monochrome,
					const object_prototype::ConstPtr& prototype);
			};

			struct cuboid
			{
				static classification::cuboid classify(
					const pc_segment& segment,
					const classification::bounding_box& bb_classification,
					const classification::monochrome_object& monochrome_classification,
					const classification::shape& shape_classification,
					const object_prototype::ConstPtr& protoptype,
					float min_object_height);
			};

			struct cylinder
			{
				static classification::cylinder  classify(
					const pc_segment& segment,
					const classification::bounding_box& bb_classification,
					const classification::monochrome_object& monochrome_classification,
					const classification::shape& shape_classification,
					const object_prototype::ConstPtr prototype,
					float min_object_height);
			};

			struct semicylinder
			{
				static classification::semicylinder classify(const pc_segment& seg,
					const classification::bounding_box& box_classification,
					const classification::monochrome_object& monochrome_classification,
					const classification::shape& shape_classification,
					const object_prototype::ConstPtr prototype,
					float min_object_height);

				static transformation::semicylinder get_feasible_transformations(const pc_segment& seg,
					const transformation::bounding_box& feasible_transformations);
			};

			struct triangular_prism
			{
				static classification::triangular_prism  classify(
					const pc_segment& segment,
					const classification::bounding_box& bb_classification,
					const classification::monochrome_object& monochrome_classification,
					const classification::shape& shape_classification,
					const object_prototype::ConstPtr prototype,
					float slant_height,
					float min_object_height);

				static transformation::triangular_prism get_feasible_transformations(
					const pc_segment& seg,
					const aabb& bounding_box,
					float slant_height,
					float min_object_height);
			};

			struct bridge
			{
				static classification::bridge  classify(const pc_segment& segment,
					const classification::bounding_box& box_classification,
					const classification::monochrome_object& monochrome_classification,
					const classification::shape& shape_classification,
					const object_prototype::ConstPtr prototype,
					float min_object_height);
					static transformation::bridge get_feasible_transformations(const pc_segment& seg,
					const transformation::bounding_box& bb_transformations,
					float min_object_height);
			};

		

			struct debug_match
			{
				aabb bounding_box;
				float min_object_height;
				Eigen::Quaternionf rotation;
			};

			struct debug_data
			{/*
				classification::shape shape_clas;
				classification_result bridge_clas;
				classification::monochrome_object color_clas;
				classification::bounding_box bb_clas;*/

				std::vector<float> new_similarities;
				std::vector<float> correct_similarities;

				std::vector<Eigen::Vector3f> new_points;
				std::vector<Eigen::Vector3f> correct_points;

				std::vector<Eigen::Quaternionf> new_prototype_rotation;
				std::vector<Eigen::Quaternionf> correct_prototype_rotation;

				std::vector<Eigen::Quaternionf> new_transformations;
				std::vector<Eigen::Quaternionf> correct_transformations;
				std::unique_ptr<debug_match> match;
				Eigen::Matrix4f transform;
			};

			//static std::unique_ptr<debug_data> debug;
		};

		namespace utility
		{
			/**
		* Projects point q0 onto line p1,p2
		*/
			static Eigen::Vector3f project(const Eigen::Vector3f& q0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2);

			/**
			* Computes the distance from p0 to the triangle p1,p2,p3
			*/
			static float distance(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3);
		}

}//namespace state_observation

#endif //STATE_OBSERVATION__CLASSIFICATION__NEW__HPP
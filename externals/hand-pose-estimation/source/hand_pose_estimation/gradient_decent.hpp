#pragma once

#include <queue>

#include "hand_model.hpp"
#include "parameter_set.hpp"
#include "hand_pose_estimation.h"

namespace hand_pose_estimation
{
	class gradient_decent_optimization_common;
	class gradient_decent_optimization_full;
	class gradient_decent_optimization_incremental;
	class gradient_decent_scheduler;

	/**
	 * @class gradient_decent_parameters
	 * @brief Parameters for gradient descent optimization
	 *
	 * Defines and manages parameters for the gradient descent algorithm used in
	 * hand pose estimation, including learning rates, tolerances, and thresholds.
	 *
	 * Features:
	 * - Learning rate control
	 * - Movement tolerances
	 * - Certainty thresholds
	 * - Parameter serialization
	 */
	class gradient_decent_parameters : public parameter_set {
	public:
		using ConstPtr = std::shared_ptr<const gradient_decent_parameters>;
		
		gradient_decent_parameters();

		~gradient_decent_parameters();

		double min_improvement;

		unsigned int max_steps;

		/*
		 * Impact of the hand certainty for the current frame on the overall hand certainty
		 */
		float learning_rate;

		/*
		 * Since the neural network often fails to properly detect the key points - 
		 * especially when the hand moves - we keep the best certainty value for a longer
		 * period of time. This value is used to compute the hand certainty.
		 * It is slowly decreased by the given factor.
		 */
		float best_net_eval_certainty_decay;

		/* If the absolute value of a joint is below this value, it is considered
		 * to be (roughly) in home pose*/
		float rotational_tolerance_flateness;
		/* If the angle of a rotational joint changed below this value between two
		 * frames, we assume that no movement occurred
		 */
		float rotational_tolerance_movement;
		/*
		 * If the translational movement speed exceeds this threshold, no bone scaling is performed
		 */
		float translational_tolerance_movement;

		/*
		 * If the similarity of the positions of two consecutive segments drops
		 * below the threshold, the hand moves quickly. The time for evaluating the
		 * neural network is saved then.
		 */
		float position_similarity_threshold;
		
		template <typename Archive>
		void serialize(Archive& ar, const unsigned int version) {
			ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
			ar& BOOST_SERIALIZATION_NVP(min_improvement);
			ar& BOOST_SERIALIZATION_NVP(max_steps);
			ar& BOOST_SERIALIZATION_NVP(learning_rate);
			ar& BOOST_SERIALIZATION_NVP(best_net_eval_certainty_decay);
			ar& BOOST_SERIALIZATION_NVP(rotational_tolerance_flateness);
			ar& BOOST_SERIALIZATION_NVP(rotational_tolerance_movement);
			ar& BOOST_SERIALIZATION_NVP(translational_tolerance_movement);
			ar& BOOST_SERIALIZATION_NVP(position_similarity_threshold);
		}

	};

	/**
	 * @class quality_criterion
	 * @brief Base class for quality evaluation criteria
	 *
	 * Abstract base class for evaluating the quality of hand pose estimates.
	 * Provides interface for error calculation and optimal step size determination.
	 *
	 * Features:
	 * - Error evaluation
	 * - Step size optimization
	 * - Weight management
	 * - Clone support
	 */
	class  HANDPOSEESTIMATION_API quality_criterion {
	public:
		typedef std::shared_ptr<quality_criterion> Ptr;
		typedef std::shared_ptr<const quality_criterion> ConstPtr;

		static const float EPSILON;
		
		quality_criterion(double weight);

		quality_criterion(const quality_criterion&) = default;
		quality_criterion(quality_criterion&&) noexcept = default;

		virtual ~quality_criterion() = default;

		const double max_weight;

		quality_criterion& operator=(const quality_criterion&) = default;

		virtual quality_criterion* clone() const = 0;

		/*
		* Returns an (error, weight) pair. Error is within [0,\infty) where 0 corresponds to a perfect match.
		* A value of 1 should roughly correspend to all (key) points being offset
		* by 1 cm. This should ensure that all quality measures have roughly the
		* same magnitude. The weight is applied last before returning the value.
		*
		* In case \infty or NaN is returned, the criterion could not be properly
		* evaluated.
		*/
		virtual std::pair<double, double> evaluate(const visual_input& input,
			const hand_pose_particle_instance& particle,
			const img_segment& seg) const = 0;

		/*
		 * Returs a value within r \in [-1,1].
		 *
		 * If @param{next} was created by adding d to one of the parameters of
		 * @param{current}, then modifiying  @param{current} by r * d yields a
		 * local minimum for this objective function.
		 *
		 * In case of rotational parameters, it is sufficient to estimate the
		 * optimum based on linear interpolation.
		 */
		virtual std::pair<double, double> optimal_stepsize(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance& current,
			const hand_pose_particle_instance& next) const = 0;




		///**
		// * Utility function to compute the step size when the optimum point is known
		// */
		//virtual float optimal_stepsize(const Eigen::Vector3f& current,
		//	const Eigen::Vector3f& next,
		//	const Eigen::Vector3f& best) const;
	};


	/**
	 * @class quality_2d_key_points
	 * @brief 2D key point-based quality evaluation
	 *
	 * Evaluates hand pose quality using neural network heatmaps for 2D key points.
	 *
	 * Features:
	 * - Heatmap-based evaluation
	 * - 2D key point matching
	 * - Quality weight management
	 */
	class  HANDPOSEESTIMATION_API quality_2d_key_points : public  quality_criterion {
	public:

		quality_2d_key_points(double weight) :
			quality_criterion(weight)
		{}

		quality_criterion* clone() const override;
		
		std::pair<double, double> evaluate(const visual_input& input,
	                const hand_pose_particle_instance& particle,
	                const img_segment& seg) const override;

		std::pair<double, double> optimal_stepsize(const visual_input& input,
		                       const img_segment& seg,
		                       const hand_pose_particle_instance& current,
		                       const hand_pose_particle_instance& next) const override;
	};



	/**
	 * @class quality_3d_key_points
	 * @brief 3D key point-based quality evaluation
	 *
	 * Evaluates hand pose quality using neural network pose output for 3D key points.
	 *
	 * Features:
	 * - 3D pose evaluation
	 * - Key point matching
	 * - Quality weight management
	 */
	class  HANDPOSEESTIMATION_API quality_3d_key_points : public  quality_criterion {
	public:



		quality_3d_key_points(double weight) :
			quality_criterion(weight)
		{}

		quality_criterion* clone() const override;

		std::pair<double, double> evaluate(const visual_input& input,
	                const hand_pose_particle_instance& particle,
	                const img_segment& seg) const override;

		std::pair<double, double> optimal_stepsize(const visual_input& input,
		                       const img_segment& seg,
		                       const hand_pose_particle_instance& current,
		                       const hand_pose_particle_instance& next) const override;


	};


	/**
	 * @class quality_key_points_below_surface
	 * @brief Surface constraint quality evaluation
	 *
	 * Ensures key points are positioned below skin-colored surface segments.
	 *
	 * Features:
	 * - Surface constraint enforcement
	 * - Force calculation
	 * - Distance estimation
	 */
	class  HANDPOSEESTIMATION_API quality_key_points_below_surface : public  quality_criterion {
	public:

		quality_key_points_below_surface(double weight) :
			quality_criterion(weight)
		{}

		quality_criterion* clone() const override;

		std::pair<double, double> evaluate(const visual_input& input,
	                const hand_pose_particle_instance& particle,
	                const img_segment& seg) const override;

		std::pair<double, double> optimal_stepsize(const visual_input& input,
		                       const img_segment& seg,
		                       const hand_pose_particle_instance& current,
		                       const hand_pose_particle_instance& next) const override;

		pcl::PointCloud<pcl::PointXYZLNormal>::Ptr get_forces(const visual_input& input,
			const hand_pose_particle_instance& particle,
			const img_segment& seg) const;

		/*
		 * Returns an estimation of the signed distance to the surface (negative if inside)
		 */
		static float distance_to_surface(const visual_input& input,
			const img_segment& seg,
			const Eigen::Vector3f& key_point,
			float radius);

		static float finger_radius(const hand_pose_18DoF& pose);

	private:
		bool creates_force(const Eigen::Vector3f& key_point,
			const Eigen::Vector3f& best,
			const Eigen::Vector3f& neg_normal,
			float radius,
			float surface_distance) const;
	};


	/**
	 * @class quality_key_points_close_to_surface
	 * @brief Proximity-based quality evaluation
	 *
	 * Evaluates quality based on key points' proximity to surface segments.
	 *
	 * Features:
	 * - Proximity evaluation
	 * - Force calculation
	 * - Surface distance metrics
	 */
	class  HANDPOSEESTIMATION_API quality_key_points_close_to_surface : public  quality_criterion {
	public:

		quality_key_points_close_to_surface(double weight) :
			quality_criterion(weight)
		{}

		quality_criterion* clone() const override;

		std::pair<double, double> evaluate(const visual_input& input,
			const hand_pose_particle_instance& particle,
			const img_segment& seg) const override;

		std::pair<double, double> optimal_stepsize(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance& current,
			const hand_pose_particle_instance& next) const override;

		pcl::PointCloud<pcl::PointXYZLNormal>::Ptr get_forces(const visual_input& input,
			const hand_pose_particle_instance& particle,
			const img_segment& seg) const;

	};


	/**
	 * @class quality_boundary_surface
	 * @brief Boundary surface quality evaluation
	 *
	 * Evaluates quality based on hand model's fit to boundary surfaces.
	 *
	 * Features:
	 * - Boundary evaluation
	 * - Surface fitting
	 * - Quality metrics
	 */
	class  HANDPOSEESTIMATION_API quality_boundary_surface : public  quality_criterion {
	public:

		quality_boundary_surface(double weight, Eigen::Hyperplane<float,3> plane) :
			quality_criterion(weight),
			plane(std::move(plane))
		{}

		quality_criterion* clone() const override;

		std::pair<double, double> evaluate(const visual_input& input,
			const hand_pose_particle_instance& particle,
			const img_segment& seg) const override;

		std::pair<double, double> optimal_stepsize(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance& current,
			const hand_pose_particle_instance& next) const override;

		Eigen::Hyperplane<float, 3> plane;
	};


	/**
	 * @class quality_acceleration
	 * @brief Motion-based quality evaluation
	 *
	 * Evaluates quality based on hand motion and acceleration patterns.
	 *
	 * Features:
	 * - Motion evaluation
	 * - Acceleration metrics
	 * - Reference pose comparison
	 */
	class  HANDPOSEESTIMATION_API quality_acceleration : public  quality_criterion {
	public:



		quality_acceleration(double weight) :
			quality_criterion(weight)
		{}

		quality_acceleration(double weight, hand_pose_particle_instance::Ptr reference) :
			quality_criterion(weight),
			reference(std::move(reference))
		{}

		quality_criterion* clone() const override;

		std::pair<double, double> evaluate(const visual_input& input,
			const hand_pose_particle_instance& particle,
			const img_segment& seg) const override;

		std::pair<double, double> optimal_stepsize(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance& current,
			const hand_pose_particle_instance& next) const override;

		hand_pose_particle_instance::Ptr reference;

	private:
		inline static void weight(hand_pose_18DoF::Vector15f& parameters);
	};

	/**
	 * @class quality_fill_mask
	 * @brief Mask-based quality evaluation
	 *
	 * Evaluates quality based on how well the hand model fills the target mask.
	 *
	 * Features:
	 * - Mask evaluation
	 * - Coverage metrics
	 * - Quality assessment
	 */
	class  HANDPOSEESTIMATION_API quality_fill_mask : public  quality_criterion {
	public:

		quality_fill_mask(double weight) :
			quality_criterion(weight)
		{}

		quality_criterion* clone() const override;

		std::pair<double, double> evaluate(const visual_input& input,
			const hand_pose_particle_instance& particle,
			const img_segment& seg) const override;

		std::pair<double, double> optimal_stepsize(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance& current,
			const hand_pose_particle_instance& next) const override;

	};

	/**
	 * @class quality_centroid
	 * @brief Centroid-based quality evaluation
	 *
	 * Evaluates quality based on hand model centroid positioning.
	 *
	 * Features:
	 * - Centroid calculation
	 * - Position evaluation
	 * - Quality metrics
	 */
	class  HANDPOSEESTIMATION_API quality_centroid : public  quality_criterion {
	public:

		quality_centroid(double weight) :
			quality_criterion(weight)
		{}

		quality_criterion* clone() const override;

		std::pair<double, double> evaluate(const visual_input& input,
			const hand_pose_particle_instance& particle,
			const img_segment& seg) const override;

		std::pair<double, double> optimal_stepsize(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance& current,
			const hand_pose_particle_instance& next) const override;

		static Eigen::Vector3f get_centroid(const hand_pose_18DoF& pose);

	};

	/**
	 * @class gradient_decent_optimization_common
	 * @brief Common functionality for gradient descent optimization
	 *
	 * Base class providing shared functionality for gradient descent optimization
	 * of hand poses, including objective evaluation and parameter updates.
	 *
	 * Features:
	 * - Objective evaluation
	 * - Parameter optimization
	 * - Pose extrapolation
	 * - Bone scaling
	 */
	class gradient_decent_optimization_common
	{
	public:
		friend class gradient_decent_scheduler;
		friend class gradient_decent_optimization_incremental;
		friend class gradient_decent_optimization_full;
		
		HANDPOSEESTIMATION_API inline static double bell_curve(double x);
		HANDPOSEESTIMATION_API inline static float bell_curve(float x, float stdev);

		HANDPOSEESTIMATION_API gradient_decent_optimization_common(std::vector<quality_criterion::Ptr> objectives = { std::make_shared< quality_2d_key_points>(1.f) },
			Eigen::Vector3f back_palm_orientation = Eigen::Vector3f::Zero(),
			gradient_decent_parameters::ConstPtr params = nullptr,
			hand_dynamic_parameters::ConstPtr dynamic_params = nullptr);

		HANDPOSEESTIMATION_API gradient_decent_optimization_common(const gradient_decent_optimization_common&);
		HANDPOSEESTIMATION_API gradient_decent_optimization_common(gradient_decent_optimization_common&&) noexcept = default;

		HANDPOSEESTIMATION_API virtual ~gradient_decent_optimization_common() = default;

		HANDPOSEESTIMATION_API gradient_decent_optimization_common& operator=(const gradient_decent_optimization_common&) = default;

		const gradient_decent_parameters::ConstPtr params;
		const hand_dynamic_parameters::ConstPtr dynamic_params;

	protected:

		Eigen::Vector3f back_palm_orientation;

		hand_pose_particle_instance::Ptr best;
		hand_pose_particle_instance::Ptr prev_particle;
		std::chrono::duration<float> timestamp;
		std::chrono::duration<float> prev_timestamp;

		std::vector<quality_criterion::Ptr> objectives;

		hand_pose_particle_instance::Ptr extrapolated_pose;

		void evaluate_objectives(const visual_input& input,
			const img_segment& seg,
			hand_pose_particle_instance& particle) const;

		hand_pose_18DoF::Ptr constrain(const hand_pose_18DoF::Ptr& model);

		hand_pose_18DoF::Ptr disturb(const hand_pose_18DoF& model);


		void best_starting_point(const visual_input& input,
			const img_segment& seg,
			const std::vector<hand_pose_particle_instance::Ptr>& seeds,
			int max_steps = 3);

		void best_starting_point_parallel(const visual_input& input,
			const img_segment& seg,
			const std::vector<hand_pose_particle_instance::Ptr>& seeds);

		/*
		 * Adds up to @param{inc} to parameter @param{index} while ensuring that kinematic
		 * and velocity constraints are met.
		 */
		inline std::pair<hand_pose_particle_instance::Ptr, float> increment_parameter(const visual_input& input,
			const img_segment& seg, 
			const hand_pose_particle_instance& particle,
			int index, 
			float inc) const;

		std::pair<hand_pose_18DoF::Ptr, float> increment_rotation_parameter(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance& particle,
			int index,
			float inc) const;

		std::pair<hand_pose_18DoF::Ptr, float> increment_pose_parameter(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance& particle,
			int index,
			float inc) const;

		
		hand_pose_particle_instance::Ptr gradient_step(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance::Ptr& particle,
			bool dir,
			int count_params = 18) const;

		/*
		 * Checks whether the conditions are met to perform bone scaling:
		 * Non-moving, flat hand facing camera
		 */
		bool can_scale_bones(const hand_pose_particle_instance& particle) const;

		void scale_bones(const visual_input& input, const img_segment& seg, bool smaller);

		float get_hand_certainty(const visual_input& input, const img_segment& seg, 
			const hand_pose_particle_instance& particle) const;

		float get_hand_orientation_fit(const visual_input& input, const img_segment& seg, const hand_pose_particle_instance& particle) const;

		hand_pose_particle_instance::Ptr extrapolate_pose_and_store(const hand_instance& hand,
			bool right_hand);

		void decent_without_net_eval(const visual_input& input,
			const img_segment& seg,
			const hand_pose_particle_instance::Ptr& prev_particle,
			const Eigen::Vector3f& translation);

		void bend_fingers(const visual_input & input, const img_segment & seg);

		std::vector<hand_pose_particle_instance::Ptr> generate_seeds(const visual_input & input,
			const hand_pose_estimation & hand_pose_est,
			const hand_instance & hand,
			bool right_hand);
	};


	/**
	 * @class gradient_decent_optimization_full
	 * @brief Complete gradient descent optimization
	 *
	 * Implements full gradient descent optimization for hand pose estimation,
	 * including neural network evaluation and pose updates.
	 *
	 * Features:
	 * - Complete optimization
	 * - Neural network integration
	 * - Pose refinement
	 * - Timestamp management
	 */
	class gradient_decent_optimization_full : public gradient_decent_optimization_common
	{
	public:
		friend class gradient_decent_scheduler;
		
		HANDPOSEESTIMATION_API gradient_decent_optimization_full(hand_pose_estimation& hand_pose_est,
			std::vector<quality_criterion::Ptr> objectives = { std::make_shared< quality_2d_key_points>(1.f) },
			Eigen::Vector3f back_palm_orientation = Eigen::Vector3f::Zero(),
			gradient_decent_parameters::ConstPtr params = nullptr,
			hand_dynamic_parameters::ConstPtr dynamic_params = nullptr);

		HANDPOSEESTIMATION_API gradient_decent_optimization_full(hand_pose_estimation& hand_pose_est,
			gradient_decent_optimization_common proto);

		HANDPOSEESTIMATION_API virtual ~gradient_decent_optimization_full() = default;

		HANDPOSEESTIMATION_API void update(std::chrono::duration<float> timestamp);
		
		HANDPOSEESTIMATION_API hand_pose_particle_instance::Ptr update(const visual_input& input,
			const img_segment& seg,
			const std::vector<hand_pose_particle_instance::Ptr>& seeds,
			const hand_pose_particle_instance::Ptr& prev_particle = nullptr);

			HANDPOSEESTIMATION_API void update(const visual_input& input,
			hand_instance& hand);

	

	private:
		hand_pose_estimation& hand_pose_est;
	};
		
	
	/**
	 * @class gradient_decent_optimization_incremental
	 * @brief Incremental gradient descent optimization
	 *
	 * Implements incremental optimization for hand poses, focusing on
	 * step-by-step improvements and priority-based processing.
	 *
	 * Features:
	 * - Incremental updates
	 * - Priority management
	 * - Step tracking
	 * - Certainty evaluation
	 */
	class gradient_decent_optimization_incremental : public gradient_decent_optimization_common
	{
	public:
		using Ptr = std::shared_ptr<gradient_decent_optimization_incremental>;

		friend class gradient_decent_scheduler;
		
		HANDPOSEESTIMATION_API gradient_decent_optimization_incremental(std::vector<quality_criterion::Ptr> objectives,
			Eigen::Vector3f back_palm_orientation,
			gradient_decent_parameters::ConstPtr params,
			hand_dynamic_parameters::ConstPtr dynamic_params,
			visual_input::ConstPtr input, 
			hand_instance::Ptr hand, 
			bool right_hand);

		// creates a deep copy of the objective function objects
		HANDPOSEESTIMATION_API gradient_decent_optimization_incremental(gradient_decent_optimization_common common,
			visual_input::ConstPtr input,
			hand_instance::Ptr hand,
			bool right_hand);

		HANDPOSEESTIMATION_API virtual ~gradient_decent_optimization_incremental() = default;

		/*
		 * Priority to update this pose. Zero = highest. 
		 */
		HANDPOSEESTIMATION_API float get_priority() const;

		HANDPOSEESTIMATION_API int update();

	private:
		visual_input::ConstPtr input;
		hand_instance::Ptr hand;
		img_segment::Ptr seg;
		bool right_hand;

		std::atomic_int step;

		float hand_certainty;
		float right_hand_certainty;
		float position_sim;
		Eigen::Vector3f translation;

		bool finished;
		std::atomic_bool net_evaluated;
		std::vector<hand_pose_particle_instance::Ptr> seeds;
	};


	/**
	 * @class gradient_decent_scheduler
	 * @brief Scheduler for gradient descent optimization
	 *
	 * Manages and schedules gradient descent optimization tasks for multiple
	 * hands, handling thread management and priority-based processing.
	 *
	 * Features:
	 * - Task scheduling
	 * - Thread management
	 * - Priority queuing
	 * - Background handling
	 */
	class gradient_decent_scheduler
	{
	public:
		HANDPOSEESTIMATION_API gradient_decent_scheduler(gradient_decent_optimization_common prototype, int max_threads = std::thread::hardware_concurrency());
		HANDPOSEESTIMATION_API ~gradient_decent_scheduler();

		HANDPOSEESTIMATION_API void update(const visual_input::ConstPtr& input, const std::vector<hand_instance::Ptr>& hands);

		HANDPOSEESTIMATION_API const hand_kinematic_parameters& get_hand_kinematic_parameters() const;
		HANDPOSEESTIMATION_API const hand_pose_parameters& get_hand_pose_parameters() const;
		HANDPOSEESTIMATION_API float get_hand_certainty_threshold() const;

		/*
	 * Defines a plane which hands cannot pierce. Points behind this plane are removed from hand candidates,
	 *				normal must point towards origin (= camera position)
	 *
	 */
		HANDPOSEESTIMATION_API void set_background(const Eigen::Hyperplane<float, 3>& plane);

		/*
		 * Sets a vector the back palm should point to.
		 * 	Eigen::Vector3f(0.f,0.f,-1.f) - prefer back hand facing camera
		 *	Eigen::Vector3f(0.f,0.f,1.f) - prefer front hand facing camera
		 *	Eigen::Vector3f(0.f,0.f,0.f) - all orientations equally feasible
		 */
		HANDPOSEESTIMATION_API void set_back_palm_orientation(const Eigen::Vector3f& normal);
		
	private:
		using prio_gd = std::pair<float, gradient_decent_optimization_incremental::Ptr>;
		
		hand_kinematic_parameters hand_kin_params;
		hand_pose_estimation hand_pose_est;
		gradient_decent_optimization_common gd_proto;
		
		std::list<hand_pose_estimation_async> nn_threads;
		std::vector<std::thread> decent_threads;
		std::atomic_bool terminate_flag;

		std::priority_queue< prio_gd, std::vector<prio_gd>, auto(*)(const prio_gd&, const prio_gd&)->bool > queue;
		std::mutex queue_mutex;
		std::condition_variable queue_condition_variable;

		std::chrono::duration<float> timestamp;

		void clear_queue();
	};
}
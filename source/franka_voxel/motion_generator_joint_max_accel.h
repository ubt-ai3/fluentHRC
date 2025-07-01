/**
 *************************************************************************
 *
 * @file motion_generator_joint_max_accel.hpp
 *
 * Motion interpolator for franka movements that always uses maximum
 * acceleration.
 *
 ************************************************************************/

#pragma once

#include <array>
#include <chrono>
#include <ranges>

#include <Eigen/Geometry>

typedef std::tuple<std::array<double, 7>, std::chrono::system_clock::time_point> joints_sync;
typedef std::vector<joints_sync> joints_progress;

namespace franka_proxy
{
	namespace Visualize
	{
		using Vector7d = Eigen::Matrix<double, 7, 1, Eigen::ColMajor>;
		using Vector7i = Eigen::Matrix<int, 7, 1, Eigen::ColMajor>;

		struct JointMovement
		{
			std::array<bool, 7> joint_motion_finished;
			Vector7d delta_q_d;

			[[nodiscard]] bool isMotionFinished() const;
		};

		class franka_joint_motion_generator;
		class franka_joint_motion_sampler
		{
		public:

			/*
			 * \param dt time between steps
			 * \param max_steps maximum points of intermediate steps
			 * 
			 * dt will be reduced if it causes steps to exceed max_steps
			 */
			franka_joint_motion_sampler(
				double dt, size_t max_steps,
				const std::shared_ptr<franka_joint_motion_generator> generator);

			/*
			 * \param dt time between steps
			 * \param max_steps maximum points of intermediate steps
			 * 
			 * dt will be reduced if it causes steps to exceed max_steps
			 * 
			 * generators must be ordered
			 */
			franka_joint_motion_sampler(
				double dt, size_t max_steps,
				const std::list<std::shared_ptr<franka_joint_motion_generator>>& generators);

			class iterator
			{
			public:

				using difference_type = std::ptrdiff_t;
				using value_type = std::tuple<Vector7d, std::chrono::utc_clock::duration>;

				iterator() = default;
				explicit iterator(const franka_joint_motion_sampler* instance, bool end = false);

				iterator& operator++();
				iterator operator++(int);
				
				bool operator==(const iterator& other) const;

				[[nodiscard]] value_type operator*() const;

				[[nodiscard]] bool done() const;

			private:

				const franka_joint_motion_sampler* m_instance;

				std::list<double>::const_iterator current_start_time;
				std::list<double>::const_iterator current_end_time;
				std::list<std::shared_ptr<franka_joint_motion_generator>>::const_iterator current_generator;

				size_t current_step;
			};

			[[nodiscard]] size_t size() const;

			[[nodiscard]] iterator begin();
			[[nodiscard]] iterator end();

			[[nodiscard]] const iterator begin() const;
			[[nodiscard]] const iterator end() const;

		private:

			const std::list<std::shared_ptr<franka_joint_motion_generator>> m_generators;
			std::list<double> accum_start_times;

			size_t m_steps;
			double m_dt;
		};

		/**
		 *************************************************************************
		 *
		 * @class franka_joint_motion_generator
		 *
		 * An example showing how to generate a joint pose motion to a goal
		 * position. Adapted from:
		 * Wisama Khalil and Etienne Dombre. 2002. Modeling, Identification and
		 * Control of Robots (Kogan Page Science Paper edition).
		 *
		 ************************************************************************/
		class franka_joint_motion_generator
		{
		public:

			/**
			 * Creates a new joint_motion_generator instance for a target q.
			 *
			 * @param[in] speed_factor General speed factor in range [0, 1].
			 * @param[in] q_goal Target joint positions.
			 */
			franka_joint_motion_generator(
				double speed_factor, 
				const Vector7d& q_start,
				const Vector7d& q_goal);

			franka_joint_motion_generator(
				double speed_factor,
				const std::array<double, 7>& q_start,
				const std::array<double, 7>& q_goal);

			/**
			 * Returns each joints position and finish state
			 */
			[[nodiscard]] JointMovement calculateDesiredValues(double t) const;

			[[nodiscard]] const Vector7d& q_start() const;

			[[nodiscard]] double end_time() const;

		private:

			void calculateSynchronizedValues();

			static double calculateQuadraticSolution(double a, double b, double c);
			static bool isMotionFinished(double delta);

			static constexpr double kDeltaQMotionFinished = 1e-6;
			const Vector7d q_goal_;

			Vector7d q_start_;
			Vector7d delta_q_;

			Vector7d dq_max_sync_;
			Vector7d t_1_sync_;
			Vector7d t_2_sync_;
			Vector7d t_f_sync_;
			Vector7d q_1_;

			double time_ = 0.0;

			Vector7d dq_max_ = (Vector7d() << 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5).finished();
			Vector7d ddq_max_start_ = (Vector7d() << 5, 5, 5, 5, 5, 5, 5).finished();
			Vector7d ddq_max_goal_ = (Vector7d() << 5, 5, 5, 5, 5, 5, 5).finished();			
		};

		/**
		 * franka_joint_motion_tcps
		 * 
		 * discretizes motion as positions of the tcp
		 */
		class franka_joint_motion_tcps
		{
		public:
			franka_joint_motion_tcps() = delete;

			[[nodiscard]] static std::vector<Eigen::Vector3f> discretize_path(const franka_joint_motion_sampler& sampler);
		};

		/**
		 * franka_joint_motion_tcps
		 * 
		 * discretizes motion as angles of the joints over time from a start point
		 */
		class franka_joint_motion_sync
		{
		public:

			franka_joint_motion_sync(std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now());
			[[nodiscard]] joints_progress discretize_path(const franka_joint_motion_sampler& sampler) const;

		private:

			std::chrono::system_clock::time_point start_time_;
		};
		
		static_assert(std::ranges::range<franka_joint_motion_sampler>);
	} /* namespace Voxel */
} /* namespace franka_proxy */
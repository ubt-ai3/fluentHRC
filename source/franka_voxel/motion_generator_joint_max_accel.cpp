/**
 *************************************************************************
 *
 * @file motion_generator_joint_max_accel.cpp
 *
 * Motion interpolator for franka movements that always uses maximum
 * acceleration, implementation.
 *
 ************************************************************************/

#include "motion_generator_joint_max_accel.h"

#include <cmath>
#include <algorithm>
#include <execution>
#include <iostream>
#include <numeric>

#include <Eigen/Dense>

#include <franka_proxy_share/franka_proxy_util.hpp>

/*
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/robot/dh_robot/KinematicChain.h>
#include <gpu_voxels/robot/dh_robot/KinematicLink.h>
#include <gpu_voxels/robot/robot_interface.h>
*/

namespace franka_proxy
{
	namespace Visualize
	{
		Vector7d cvt_config(const std::array<double, 7>& config)
		{
			auto out = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(config.data());
			return out;
		}

		std::array<double, 7> cvt_config(const Vector7d& config)
		{
			std::array<double, 7> out;
			for (int i = 0; i < 7; i++)
				out[i] = config(i, 0);

			return out;
		}

		bool JointMovement::isMotionFinished() const
		{
			return std::ranges::all_of(joint_motion_finished, [](bool x) { return x; });
		}

		//////////////////////////////////////////////////////////////////////////
		//
		// franka_joint_motion_generator
		//
		//////////////////////////////////////////////////////////////////////////
		franka_joint_motion_generator::franka_joint_motion_generator
		(double speed_factor, 
			const Vector7d& q_start,
			const Vector7d& q_goal)
			: q_goal_(q_goal), q_start_(q_start), delta_q_(q_goal - q_start)
		{
			dq_max_ *= speed_factor;
			ddq_max_start_ *= speed_factor;
			ddq_max_goal_ *= speed_factor;

			dq_max_sync_.setZero();

			t_1_sync_.setZero();
			t_2_sync_.setZero();
			t_f_sync_.setZero();
			q_1_.setZero();

			calculateSynchronizedValues();
		}

		franka_joint_motion_generator::franka_joint_motion_generator(double speed_factor, const std::array<double, 7>& q_start, const std::array<double, 7>& q_goal)
			: q_goal_(cvt_config(q_goal)), q_start_(cvt_config(q_start)), delta_q_(q_goal_ - q_start_)
		{
			dq_max_ *= speed_factor;
			ddq_max_start_ *= speed_factor;
			ddq_max_goal_ *= speed_factor;

			dq_max_sync_.setZero();

			t_1_sync_.setZero();
			t_2_sync_.setZero();
			t_f_sync_.setZero();
			q_1_.setZero();

			calculateSynchronizedValues();
		}


		JointMovement franka_joint_motion_generator::calculateDesiredValues(double t) const
		{
			JointMovement out;

			Vector7i sign_delta_q;
			for (int i = 0; i < 7; i++)
				sign_delta_q[i] = static_cast<int>(std::copysign(1.0, delta_q_[i]));

			for (Eigen::Index i = 0; i < 7; i++)
			{
				if (isMotionFinished(delta_q_[i]))
				{
					out.delta_q_d[i] = 0;
					out.joint_motion_finished[i] = true;
				}
				else
				{
					//uses the Formulas of Modeling, identification and control of robots
					//from page 327 to page 328
					if (t < t_1_sync_[i])
					{
						//formula 13.37
						out.delta_q_d[i] = -1.0 / std::pow(t_1_sync_[i], 3.0) * dq_max_sync_[i] * sign_delta_q[i] *
							(0.5 * t - t_1_sync_[i]) * std::pow(t, 3.0);
					}
					else if (t >= t_1_sync_[i] && t < t_2_sync_[i])
					{
						//13.42
						out.delta_q_d[i] = q_1_[i] + (t - t_1_sync_[i]) * dq_max_sync_[i] * sign_delta_q[i];
					}
					else if (t >= t_2_sync_[i] && t < t_f_sync_[i])
					{
						const double t_d = t_2_sync_[i] - t_1_sync_[i];
						const double delta_t_2_sync_ = t_f_sync_[i] - t_2_sync_[i];
						//13.43
						out.delta_q_d[i] = delta_q_[i] +
							0.5 *
							(1.0 / std::pow(delta_t_2_sync_, 3.0) *
								(t - t_1_sync_[i] - 2.0 * delta_t_2_sync_ - t_d) *
								std::pow((t - t_1_sync_[i] - t_d), 3.0) +
								(2.0 * t - 2.0 * t_1_sync_[i] - delta_t_2_sync_ - 2.0 * t_d)) *
							dq_max_sync_[i] * sign_delta_q[i];
					}
					else
					{
						out.delta_q_d[i] = delta_q_[i];
						out.joint_motion_finished[i] = true;
					}
				}
			}
			return out;
		}

		const Vector7d& franka_joint_motion_generator::q_start() const
		{
			return q_start_;
		}

		double franka_joint_motion_generator::end_time() const
		{
			return t_f_sync_.maxCoeff();
		}

		void franka_joint_motion_generator::calculateSynchronizedValues()
		{
			Vector7i sign_delta_q;
			for (int i = 0; i < 7; i++)
				sign_delta_q[i] = static_cast<int>(std::copysign(1.0, delta_q_[i]));

			constexpr double factor = 1.5;

			Vector7d t_f = Vector7d::Zero();
			for (Eigen::Index i = 0; i < 7; i++)
			{
				if (isMotionFinished(delta_q_[i]))
					continue;

				//max reachable speed
				double dq_max_reach = dq_max_[i];

				if (std::abs(delta_q_[i]) < //if overshooting target distance
					factor / 2.0 * (std::pow(dq_max_[i], 2.0) / ddq_max_start_[i]) +
					factor / 2.0 * (std::pow(dq_max_[i], 2.0) / ddq_max_goal_[i]))
				{
					//reduce max speed
					dq_max_reach = std::sqrt
					(2.0 / factor * delta_q_[i] * sign_delta_q[i] *
						(ddq_max_start_[i] * ddq_max_goal_[i]) /
						(ddq_max_start_[i] + ddq_max_goal_[i]));
				}

				//acceleration time
				const double t_1 = factor * dq_max_reach / ddq_max_start_[i];
				//deceleration time
				const double delta_t_2 = factor * dq_max_reach / ddq_max_goal_[i];

				//final time = acceleration + deceleration + constant speed time?
				t_f[i] = (t_1 + delta_t_2) / 2.0 + std::abs(delta_q_[i]) / dq_max_reach;
			}
			const double max_t_f = t_f.maxCoeff();

			Vector7d delta_t_2_sync = Vector7d::Zero();
			for (Eigen::Index i = 0; i < 7; i++)
			{
				if (isMotionFinished(delta_q_[i]))
					continue;

				const double a = factor / 2.0 * (ddq_max_goal_[i] + ddq_max_start_[i]);
				const double b = -1.0 * max_t_f * ddq_max_goal_[i] * ddq_max_start_[i];
				const double c = std::abs(delta_q_[i]) * ddq_max_goal_[i] * ddq_max_start_[i];

				dq_max_sync_[i] = calculateQuadraticSolution(a, b, c);
				t_1_sync_[i] = factor * dq_max_sync_[i] / ddq_max_start_[i];
				delta_t_2_sync[i] = factor * dq_max_sync_[i] / ddq_max_goal_[i];

				t_f_sync_[i] = (t_1_sync_[i] + delta_t_2_sync[i]) / 2.0 + std::abs(delta_q_[i] / dq_max_sync_[i]);
				t_2_sync_[i] = t_f_sync_[i] - delta_t_2_sync[i];
				q_1_[i] = dq_max_sync_[i] * sign_delta_q[i] * 0.5 * t_1_sync_[i];
			}
		}

		/*

	void franka_joint_motion_generator::calculateSynchronizedValues()
	{
		Vector7i sign_delta_q;
		for (int i = 0; i < 7; i++)
			sign_delta_q[i] = static_cast<int>(std::copysign(1.0, delta_q_[i]));

		//max reachable speed
		Vector7d dq_max_reach(dq_max_);

		Vector7d t_f = Vector7d::Zero();
		for (Eigen::Index i = 0; i < 7; i++)
		{
			if (isMotionFinished(delta_q_[i]))
				continue;

			const double dq_max_p2 = dq_max_[i] * dq_max_[i];
			const double max_a_dist =
				(dq_max_p2 / ddq_max_start_[i] +
				dq_max_p2 / ddq_max_goal_[i]) / 2.0;

			double t_intermediate = 0.0;

			if (std::abs(delta_q_[i]) < //if overshooting target distance
				max_a_dist)
			{
				//reduce max speed V^2 = 2 * a * dx
				dq_max_reach[i] = std::sqrt(2.0 * std::abs(delta_q_[i]) *
					(ddq_max_start_[i] * ddq_max_goal_[i]) /
					(ddq_max_start_[i] + ddq_max_goal_[i]));
			}
			else
				t_intermediate = (max_a_dist - std::abs(delta_q_[i])) / dq_max_reach[i];

			//acceleration time
			const double t_1 = dq_max_reach[i] / ddq_max_start_[i];
			//deceleration time
			const double delta_t_2 = dq_max_reach[i] / ddq_max_goal_[i];

			//final time = acceleration + deceleration + constant speed time?
			t_f[i] = t_1 + delta_t_2 + t_intermediate;
		}
		const double max_t_f = t_f.maxCoeff();

		Vector7d delta_t_2_sync = Vector7d::Zero();
		for (Eigen::Index i = 0; i < 7; i++)
		{
			if (isMotionFinished(delta_q_[i]))
				continue;

			if (max_t_f == t_f[i])
			{
				dq_max_sync_[i] = dq_max_reach[i];
			}
			else //smaller time -> decrease velocity
			{

			}

			const double a = (ddq_max_goal_[i] + ddq_max_start_[i]) / 2.0;
			const double b = -1.0 * max_t_f * ddq_max_goal_[i] * ddq_max_start_[i];
			const double c = std::abs(delta_q_[i]) * ddq_max_goal_[i] * ddq_max_start_[i];

			dq_max_sync_[i] = calculateQuadraticSolution(a, b, c);
			t_1_sync_[i] = dq_max_sync_[i] / ddq_max_start_[i] / 2.0;
			delta_t_2_sync[i] = dq_max_sync_[i] / ddq_max_goal_[i] / 2.0;

			t_f_sync_[i] = t_1_sync_[i] + delta_t_2_sync[i] + std::abs(delta_q_[i] / dq_max_sync_[i]);
			t_2_sync_[i] = t_f_sync_[i] - delta_t_2_sync[i] * 2.0;
			q_1_[i] = dq_max_sync_[i] * sign_delta_q[i] * t_1_sync_[i];
		}
	}
	*/

		double franka_joint_motion_generator::calculateQuadraticSolution(double a, double b, double c)
		{
			double delta = b * b - 4.0 * a * c;
			delta = std::max(delta, 0.0);

			return (-b - std::sqrt(delta)) / (2.0 * a);
		}

		bool franka_joint_motion_generator::isMotionFinished(double delta)
		{
			return std::abs(delta) <= kDeltaQMotionFinished;
		}




		franka_joint_motion_sampler::franka_joint_motion_sampler(double dt, size_t max_steps, const std::shared_ptr<franka_joint_motion_generator> generator)
			: m_generators({ generator })
		{
			const double total_time = generator->end_time();
			accum_start_times = { 0, total_time };
				
			m_steps = static_cast<size_t>(ceil(total_time / dt));
			if (m_steps > max_steps)
			{
				m_steps = max_steps;
				m_dt = total_time / static_cast<double>(max_steps);

				std::cerr << "Exceeded max_steps" << std::endl;
			}
			else
				m_dt = dt;
		}

		franka_joint_motion_sampler::franka_joint_motion_sampler(double dt, size_t max_steps, const std::list<std::shared_ptr<franka_joint_motion_generator>>& generators)
			: m_generators({ generators }), accum_start_times({0})
		{
			std::transform_inclusive_scan(m_generators.begin(), m_generators.end(), std::back_inserter(accum_start_times), std::plus<double>{}, [](const auto& v) {return v->end_time(); });
			const double total_time = accum_start_times.back();

			m_steps = static_cast<size_t>(ceil(total_time / dt));
			if (m_steps > max_steps)
			{
				m_steps = max_steps;
				m_dt = total_time / static_cast<double>(max_steps);

				std::cerr << "Exceeded max_steps" << std::endl;
			}
			else
				m_dt = dt;
		}

		size_t franka_joint_motion_sampler::size() const
		{
			return m_steps;
		}

		franka_joint_motion_sampler::iterator franka_joint_motion_sampler::begin()
		{
			return iterator{ this };
		}

		franka_joint_motion_sampler::iterator franka_joint_motion_sampler::end()
		{
			return iterator { this, true };
		}

		const franka_joint_motion_sampler::iterator franka_joint_motion_sampler::begin() const
		{
			return iterator{ this };
		}

		const franka_joint_motion_sampler::iterator franka_joint_motion_sampler::end() const
		{
			return iterator{ this, true };
		}


		franka_joint_motion_sampler::iterator::iterator(const franka_joint_motion_sampler* instance, bool end)
			: m_instance(instance),
			current_start_time(instance->accum_start_times.begin()),
			current_end_time(++instance->accum_start_times.begin()), current_generator(instance->m_generators.begin()),
			current_step(!end ? 0 : instance->m_steps)
		{}

		franka_joint_motion_sampler::iterator& franka_joint_motion_sampler::iterator::operator++()
		{
			if (!done())
			{
				++current_step;
				const double current_time = static_cast<double>(current_step) * m_instance->m_dt;
				while (current_end_time != m_instance->accum_start_times.end() && current_time > *current_end_time)
				{
					current_start_time = current_end_time;
					++current_end_time;
					++current_generator;
				}
			}
			return *this;
		}

		franka_joint_motion_sampler::iterator franka_joint_motion_sampler::iterator::operator++(int)
		{
			const auto temp = *this;
			++*this;
			return temp;
		}

		bool franka_joint_motion_sampler::iterator::operator==(const iterator& other) const
		{
			return (m_instance == other.m_instance) && (current_step == other.current_step);
		}

		franka_joint_motion_sampler::iterator::value_type franka_joint_motion_sampler::iterator::operator*() const
		{
			const double current_time = static_cast<double>(current_step) * m_instance->m_dt;
			const double generator_time = current_time - *current_start_time;

			return { (*current_generator)->q_start() + (*current_generator)->calculateDesiredValues(generator_time).delta_q_d,
				std::chrono::duration_cast<std::chrono::utc_clock::duration>(std::chrono::duration<double>(current_time))};
		}

		bool franka_joint_motion_sampler::iterator::done() const
		{
			return current_step >= m_instance->m_steps;
		}

		std::vector<Eigen::Vector3f> franka_joint_motion_tcps::discretize_path(const franka_joint_motion_sampler& sampler)
		{
			
			//auto t0 = std::chrono::high_resolution_clock::now();
			std::vector<Eigen::Vector3f> out1(sampler.size());

			auto idxs = std::views::iota(size_t(0), sampler.size());
			std::transform(std::execution::par_unseq, sampler.begin(), sampler.end(), out1.begin(), [](const franka_joint_motion_sampler::iterator::value_type& sample)
				{
					return (franka_proxy::franka_proxy_util::fk(std::get<0>(sample)).back() * Eigen::Translation3d(0., 0., 0.22)).translation().cast<float>();
				});
			//std::cout << std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0) << std::endl;
			return out1;
		/*
			{
			auto t0 = std::chrono::high_resolution_clock::now();
			std::vector<Eigen::Vector3f> out;
			out.reserve(sampler.size());
			for (const auto joints : sampler | std::views::elements<0>)
				out.emplace_back(franka_util::fk(joints).back().translation().cast<float>());

			std::cout << std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0) << std::endl << std::endl;

			for (int i = 0; i < out.size(); ++i)
			{
				if (out1[i] != out[i])
					throw std::exception("!=");
			}

			return out;
			}*/
		}

		franka_joint_motion_sync::franka_joint_motion_sync(std::chrono::system_clock::time_point start_time)
			: start_time_(start_time)
		{
			std::cout << start_time << std::endl;

		}

		joints_progress franka_joint_motion_sync::discretize_path(const franka_joint_motion_sampler& sampler) const
		{
			joints_progress out;
			out.reserve(sampler.size());

			for (const auto [joints, time_offset] : sampler)
				out.emplace_back(joints_sync{ cvt_config(joints), start_time_ + time_offset });

			//auto i = std::chrono::duration_cast<std::chrono::duration<long long, std::ratio<1, 10'000'000>>(std::get<1>(out.front()).time_since_epoch());
			return out;
		}
} /* namespace Voxel */
} /* namespace franka_proxy */

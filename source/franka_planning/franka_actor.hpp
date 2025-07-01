#pragma once
#ifndef STATE_OBSERVATION__FRANKA_ACTOR__HPP
#define STATE_OBSERVATION__FRANKA_ACTOR__HPP
#include <chrono>

#include <enact_priority/signaling_actor.hpp>
#include <state_observation/pn_model.hpp>

namespace franka_control
{
	typedef Eigen::Matrix<double, 7, 1> robot_config_7dof;
}
namespace franka_proxy
{
	typedef std::array<double, 7> robot_config_7dof;
	/*struct vacuum_gripper_state
	{
		uint8_t actual_power_;
		uint8_t vacuum_level;
		bool part_detached_;
		bool part_present_;
		bool in_control_range_;
	};*/
}

namespace state_observation
{

	//helper function to convert between the two possible representations of robot_config: std::array and Eigen::Matrix
	franka_control::robot_config_7dof cvt_config(const franka_proxy::robot_config_7dof& config);
	franka_proxy::robot_config_7dof cvt_config(const franka_control::robot_config_7dof& config);

	/**
	 * Controller
	 * 
	 * Dummy base class to be extended by child classes 
	 */
	class Controller
	{
	public:

		Controller() = default;
		virtual ~Controller() = default;

		virtual void move_to(const franka_proxy::robot_config_7dof& target) = 0;
		[[nodiscard]] virtual franka_proxy::robot_config_7dof current_config() const = 0;
		virtual void set_speed_factor(double speed_factor) = 0;
		virtual void execute_retry(std::function<void()> f);

		virtual bool do_logging();
		virtual std::chrono::high_resolution_clock::time_point start_time();
		[[nodiscard]] virtual bool needs_update_loop() const;

		virtual bool vacuum_gripper_vacuum(std::uint8_t vacuum_strength, std::chrono::milliseconds timeout = std::chrono::milliseconds(100));
		virtual bool vacuum_gripper_stop();
		virtual bool vacuum_gripper_drop(std::chrono::milliseconds timeout = std::chrono::milliseconds(100));
		virtual void update();
		virtual void automatic_error_recovery();
	};

	/*
		handles the interface to the franka proxy
		allows synchronously executing pick'n place tasks
	*/
	class franka_agent
	{

	public:

		
		explicit franka_agent(std::shared_ptr<Controller> controller);

		franka_agent(const franka_agent&) = delete;

		~franka_agent() noexcept;

		[[nodiscard]] bool can_execute_transition(const pn_transition& transition) const;

		bool execute_transition(const pn_transition& transition);

		[[nodiscard]] bool approx_reached(const franka_control::robot_config_7dof&) const noexcept;
		[[nodiscard]] bool approx_reached(const Eigen::Affine3d&) const noexcept;

		// moves robot into rest pose so that it does not cover any objects
		bool rest();

		/*
			performs a picknplace task with the vacuum gripper from position pick to position place in world/robot coordinates
			always grips from above, ignores rotation of objects
			blocks until the motion is finished
			@returns if the task was performed successfully
		*/
		bool picknplace(const obb& pick, const obb& place);

		bool pick(const obb& pick);

		bool place(const obb& place);

		bool approach(const pn_transition& transition);
		bool approach(const obb& box);

		[[nodiscard]] bool has_object_gripped() const noexcept;

		void log(const std::string& line);
		void reset_log(std::chrono::high_resolution_clock::time_point start);

		inline static constexpr double speed = 0.6;
		inline static constexpr double speed_slow = 0.25;
		inline static constexpr double tcp_to_t6_offset = -0.23;
		inline static constexpr double below_suction_cup_height = 0.015;
		inline static constexpr double gripper_radius = 0.065;
		
		//boost::signals2::signal<void(const franka_proxy::robot_config_7dof&)> joint_signal;

		[[nodiscard]] franka_proxy::robot_config_7dof get_config() const;

	private:

		struct FileWrapper
		{
			FileWrapper(bool do_logging)
				: m_do_logging(do_logging)
			{}

			bool is_open() const
			{
				if (!m_do_logging)
					return false;

				return file.is_open();
			}

			void open(const std::string& str)
			{
				if (m_do_logging)
					file.open(str);
			}

			void close()
			{
				if (m_do_logging)
					file.close();
			}

			template<typename T>
			FileWrapper& operator<<(T&& x)
			{
				if (m_do_logging)
					file << std::forward<T>(x);
				return *this;
			}

			typedef FileWrapper& (*MyStreamManipulator)(FileWrapper&);

			// take in a function with the custom signature
			FileWrapper& operator<<(MyStreamManipulator manip)
			{
				if (!m_do_logging)
					return *this;
				// call the function, and return it's value
				return manip(*this);
			}

			static FileWrapper& endl(FileWrapper& stream)
			{
				// print a new line
				stream << std::endl;
				return stream;
			}

			// this is the type of std::cout
			typedef std::basic_ostream<char, std::char_traits<char> > CoutType;

			// this is the function signature of std::endl
			typedef CoutType& (*StandardEndLine)(CoutType&);

			// define an operator<< to take in std::endl
			FileWrapper& operator<<(StandardEndLine manip)
			{
				// call the function, but we cannot return it's value
				if (m_do_logging)
					manip(file);

				return *this;
			}

			bool m_do_logging;
			std::ofstream file;
		};

		bool pick_lockfree(const obb& pick);

		bool place_lockfree(const obb& place);

		void move_to(const Eigen::Matrix<double, 7, 1>& pose);

		//bool move_to_until_contact(const Eigen::Matrix<double, 7, 1>& pose);

		void set_speed(double target_speed);

		void taskmain();

		[[nodiscard]] std::chrono::milliseconds get_time_ms() const;

		static Eigen::Matrix3d get_default_nsa_orientation();

		struct agent_state
		{
			bool on_transfer_height = false;
			bool gripped_object = false;
			bool vacuuming = false;
			bool in_rest_position = false;
		};
		agent_state agent_state_;

		std::shared_ptr<Controller> remote_controller_;
		//doesn't lock franka_lock

		std::mutex franka_lock_;
		std::atomic_bool terminate_ = false;
		std::thread state_update_thread_;
		std::try_to_lock_t lock_type;
		inline static const std::chrono::milliseconds update_interval = std::chrono::milliseconds(10);

		FileWrapper file;
		FileWrapper file_poses;
		std::chrono::high_resolution_clock::time_point start_time;
	};


	/*
		executes transitions which are part of a pickn place task asynchronously
		emits signals and calls callbacks when franka finished a task
	*/
	class franka_async_agent
	{
	public:
		franka_async_agent(std::shared_ptr<Controller>&& controller);
		~franka_async_agent()noexcept;

		template <typename callback>
		void execute_transition(pn_transition::Ptr transition, callback&& call);

		void execute_transition(pn_transition::Ptr transition);

		[[nodiscard]] bool can_execute_transition(const pn_transition& transition)const;

		auto& get_signal()
		{
			return transition_completed_signal;
		}

		boost::signals2::signal<void(pn_transition::Ptr)> transition_completed_signal;

	private:
		franka_agent franka;
		std::map<pn_transition::Ptr, std::function<void()>> callbacks;
		void do_work();

		std::atomic_bool terminate_flag = false;
		std::thread worker;
		std::mutex task_queue_mutex;
		std::queue<pn_transition::Ptr>  task_queue;
	};

	template <typename callback>
	void franka_async_agent::execute_transition(pn_transition::Ptr transition, callback&& call)
	{
		std::scoped_lock lock(task_queue_mutex);
		task_queue.push(transition);

		auto [it,success] = callbacks.emplace(transition,std::forward<callback>(call));
		if (!success)
			throw std::runtime_error("can't execute the same transition multiple times");
	}
}//namespace app_visualization

#endif

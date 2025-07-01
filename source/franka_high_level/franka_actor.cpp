#include "franka_actor.h"

#include <franka_proxy_client/exception.hpp>
#include <franka_proxy_share/franka_proxy_util.hpp>

using namespace std::chrono;
using namespace std::chrono_literals;

remote_controller_wrapper::remote_controller_wrapper(std::chrono::high_resolution_clock::time_point start_time, std::string_view ip_addr)
	: remote_interface_(std::string(ip_addr)), start_time_(start_time)
{}

bool remote_controller_wrapper::do_logging()
{
	return true;
}

std::chrono::high_resolution_clock::time_point remote_controller_wrapper::start_time()
{
	return start_time_;
}

bool remote_controller_wrapper::needs_update_loop() const
{
	return true;
}

void remote_controller_wrapper::move_to(const franka_proxy::robot_config_7dof& target)
{
	remote_interface_.move_to(target);
	/*
	std::this_thread::sleep_for(std::chrono::seconds{ 2 });
	auto should = (franka_control::franka_util::fk(target).back() * Eigen::Translation3d(0., 0., 0.22)).translation();
	auto is = (franka_control::franka_util::fk(current_config()).back() * Eigen::Translation3d(0., 0., 0.22)).translation();
	auto diff = is - should;

	std::cout << "Should: " << should.x() << ", " << should.y() << ", " << should.z() << std::endl;
	std::cout << "Is: " << is.x() << ", " << is.y() << ", " << is.z() << std::endl;
	std::cout << "Diff: " << diff.x() << ", " << diff.y() << ", " << diff.z() << std::endl << std::endl << std::endl;*/
}

franka_proxy::robot_config_7dof remote_controller_wrapper::current_config() const
{
	return remote_interface_.current_config();
}

bool remote_controller_wrapper::vacuum_gripper_vacuum(std::uint8_t vacuum_strength, std::chrono::milliseconds timeout)
{
	return remote_interface_.vacuum_gripper_vacuum(vacuum_strength, timeout);
}

bool remote_controller_wrapper::vacuum_gripper_stop()
{
	return remote_interface_.vacuum_gripper_stop();
}

bool remote_controller_wrapper::vacuum_gripper_drop(std::chrono::milliseconds timeout)
{
	return remote_interface_.vacuum_gripper_drop(timeout);
}

void remote_controller_wrapper::set_speed_factor(double speed_factor)
{
	remote_interface_.set_speed_factor(speed_factor);
}

void remote_controller_wrapper::update()
{
	remote_interface_.update();
}

void remote_controller_wrapper::automatic_error_recovery()
{
	remote_interface_.automatic_error_recovery();
}

void remote_controller_wrapper::execute_retry(std::function<void()> f)
{
	int tries = 0;
	while (true)
	{
		try
		{
			tries++;
			f();
			return;
		}
		catch (const franka_proxy::control_exception&)
		{
			// for some reason, automatic error recovery
			// is only possible after waiting some time...
			std::this_thread::sleep_for(500ms);
			this->automatic_error_recovery();

			if (tries > 3)
				throw;
		}
		catch (const franka_proxy::command_exception& e)
		{
			std::cout << "Encountered command exception. Probably because of wrong working mode. Waiting before retry." << std::endl;
			std::cout << e.what() << "\n";
			std::this_thread::sleep_for(1s);
		}
	}
}

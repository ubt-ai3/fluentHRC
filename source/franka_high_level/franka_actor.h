#pragma once

#include <franka_planning/franka_actor.hpp>

#include <franka_proxy_client/franka_remote_interface.hpp>

/**
 * remote_controller_wrapper
 * 
 * implementation of actual controller communicating with a roboter
 */
class remote_controller_wrapper : public state_observation::Controller
{
public:

	explicit remote_controller_wrapper(std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now(),
		std::string_view ip_addr = "132.180.194.120");

	~remote_controller_wrapper() override = default;

	bool do_logging() override;
	std::chrono::high_resolution_clock::time_point start_time() override;
	bool needs_update_loop() const override;

	void move_to(const franka_proxy::robot_config_7dof& target) override;
	[[nodiscard]] franka_proxy::robot_config_7dof current_config() const override;
	bool vacuum_gripper_vacuum(std::uint8_t vacuum_strength, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) override;
	bool vacuum_gripper_stop() override;
	bool vacuum_gripper_drop(std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) override;
	void set_speed_factor(double speed_factor) override;
	void update() override;
	void automatic_error_recovery() override;

	void execute_retry(std::function<void()> f) override;

private:

	franka_proxy::franka_remote_interface remote_interface_;
	std::chrono::high_resolution_clock::time_point start_time_;
};
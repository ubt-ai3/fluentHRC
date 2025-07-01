#pragma once

#include "framework.hpp"

#include <random>

#include <state_observation/workspace_objects.hpp>
#include <state_observation/pn_model.hpp>

#include "scene.hpp"

namespace simulation
{

/**
 * @class agent
 * @brief Abstract base class for agents that can interact with the simulation environment
 *
 * Represents an agent that can interact with the environment by moving objects
 * and executing actions. When rendered, it is represented by a simulated arm.
 *
 * Features:
 * - Action execution tracking
 * - Object manipulation capabilities
 * - Petri net transition support
 * - State management
 * - Cloning support
 */
class SIMULATION_API agent
{
public:

	typedef std::shared_ptr<agent> Ptr;
	
	state_observation::pn_transition::Ptr executing_action;
	Eigen::Vector3f pick_location;
	const state_observation::pn_place::Ptr place;

	agent(const environment::Ptr& env,
		state_observation::pn_place::Ptr place,
		std::function<double(const state_observation::pn_transition::Ptr&, agent&)> capabilities = [](const state_observation::pn_transition::Ptr&, agent&) {return 1.; });

	virtual ~agent() = default;

	[[nodiscard]] virtual bool is_idle(std::chrono::duration<float> timestamp) const = 0;
	virtual void step(std::chrono::duration<float> timestamp) = 0;

	[[nodiscard]] state_observation::object_prototype::ConstPtr get_grabbed_object() const;

	[[nodiscard]] virtual std::shared_ptr<agent> clone() const = 0;
	
protected:
	environment::Ptr env;
	std::function<double(const state_observation::pn_transition::Ptr&, agent&)> capabilities;

	movable_object::Ptr grabbed_object;

	std::vector<state_observation::pn_transition::Ptr> executed_actions;
};

/**
 * @class human_agent
 * @brief Represents a human agent in the simulation
 *
 * Implements a human agent with specific movement patterns and capabilities.
 * Includes features like transfer height, shoulder height, and home pose management.
 *
 * Features:
 * - Human-like movement patterns
 * - Action generation and execution
 * - State tracking
 * - Home pose management
 * - Action feasibility checking
 */
class SIMULATION_API human_agent : public agent
{
public:

	human_agent(const environment::Ptr& env,
		Eigen::Vector3f base,
		Eigen::Vector3f front,
		state_observation::pn_place::Ptr place,
		std::function<double(const state_observation::pn_transition::Ptr&, agent&)> capabilities = [](const state_observation::pn_transition::Ptr&, agent&) {return 1.; });

	~human_agent() override = default;

	[[nodiscard]] std::shared_ptr<agent> clone() const override;

	const static float transfer_height;
	const static float shoulder_height;

	const Eigen::Vector3f home_pose;

	const simulated_arm::Ptr arm;

	[[nodiscard]] bool is_idle(std::chrono::duration<float> timestamp) const override;
	void step(std::chrono::duration<float> timestamp) override;


	[[nodiscard]] state_observation::pn_transition::Ptr get_put_back_action(const state_observation::pn_binary_marking::ConstPtr& marking) const noexcept;

	std::vector<state_observation::pn_transition::Ptr> generate_feasible_actions();

	void execute_action(const state_observation::pn_transition::Ptr& action, std::chrono::duration<float> timestamp);
	void to_home_pose(std::chrono::duration<float> timestamp);

	[[nodiscard]] bool is_connected_to_agent_place(const state_observation::pn_transition& transition) const;

	void print_actions() const;

private:

	enum class execution_state
	{
		APPROACH_HORIZONTAL,
		APPROACH_VERTICAL,
		LIFT
	};

	execution_state state;
};

/**
 * @class robot_agent
 * @brief Represents a robotic agent in the simulation
 *
 * Implements a robotic agent with specific capabilities and movement patterns.
 * Extends the base agent class with robot-specific functionality.
 *
 * Features:
 * - Robot-specific movement patterns
 * - Action execution
 * - State management
 * - Capability assessment
 */
class SIMULATION_API robot_agent : public agent
{
public:

	robot_agent(const environment::Ptr& env,
		state_observation::pn_place::Ptr place,
		std::function<double(const state_observation::pn_transition::Ptr&, agent&)> capabilities = [](const state_observation::pn_transition::Ptr&, agent&) {return 1.; });

	~robot_agent() override = default;

	[[nodiscard]] std::shared_ptr<agent> clone() const override;

	[[nodiscard]] bool is_idle(std::chrono::duration<float> timestamp) const override;
	void step(std::chrono::duration<float> timestamp) override;
};

/**
 * @class sim_task
 * @brief Represents a simulation task with agents and goals
 *
 * Defines a task that incorporates agents, possible actions, their ordering,
 * and a renderable scene. Manages the overall task structure and goals.
 *
 * Features:
 * - Multiple agent support
 * - Environment management
 * - Goal tracking
 * - Task initialization
 * - Scene rendering
 */
class SIMULATION_API sim_task
{
public:
	typedef std::shared_ptr<sim_task> Ptr;

	const std::string name;
	const environment::Ptr env;
	std::vector<agent::Ptr> agents;
	const state_observation::pn_instance init_goal;
	const state_observation::pn_instance task_goal;

	sim_task(
		const std::string& name,
		const environment::Ptr& env,
		const std::vector<agent::Ptr>& agents,
		const state_observation::pn_instance& init_goal,
		const state_observation::pn_instance& task_goal
	);
};

/**
 * @class task_execution
 * @brief Manages the execution of simulation tasks
 *
 * Coordinates the execution of tasks by allowing agents to choose and execute
 * non-competing actions. The execution logic is stored in the agents.
 *
 * Features:
 * - Action coordination
 * - Timestamp management
 * - FPS control
 * - Random number generation
 * - Step-based execution
 */
class SIMULATION_API task_execution
{
public:
	task_execution(const sim_task::Ptr& task,
		double fps = 30);

	virtual std::chrono::duration<float> step();

	virtual ~task_execution() = default;

protected:
	sim_task::Ptr task;
	std::chrono::duration<float> duration;
	std::chrono::duration<float> timestamp;
	std::random_device rd;
};

/**
 * @class repeated_task_execution
 * @brief Manages repeated execution of simulation tasks
 *
 * Extends task execution to support repeated execution of tasks with
 * initialization and cleanup phases.
 *
 * Features:
 * - Task repetition
 * - Initialization management
 * - Cleanup handling
 * - Object state tracking
 * - Time management
 */
class SIMULATION_API repeated_task_execution : public task_execution
{
public:
	repeated_task_execution(const sim_task::Ptr& task,
		double fps = 30,
		unsigned int count = std::numeric_limits<unsigned int>::max());

	std::chrono::duration<float> step() override;

	~repeated_task_execution() override = default;

protected:
	std::vector<agent::Ptr> initial_agents;
	unsigned int remaining;
	bool cleaned = false;
	std::chrono::duration<float> init_time = std::chrono::duration<float>(0);

	state_observation::pn_transition::Ptr init;
	std::vector<state_observation::pn_transition::Ptr> cleanup;
	std::map<state_observation::pn_instance, movable_object::Ptr> initial_objects;
};

}
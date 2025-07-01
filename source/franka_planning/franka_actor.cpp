#include <boost/date_time/posix_time/posix_time.hpp>

#include <franka_proxy_share/franka_proxy_util.hpp>
#include "franka_actor.hpp"
#include <state_observation/pn_model_extension.hpp>
#include <filesystem>

using namespace state_observation;
using namespace std::chrono;

namespace state_observation
{
franka_agent::franka_agent(std::shared_ptr<Controller> controller) :
	remote_controller_(std::move(controller)),
	state_update_thread_([this]() { taskmain(); }),
	file(remote_controller_->do_logging()),
	file_poses(remote_controller_->do_logging()),
	start_time(remote_controller_->start_time())
{
	set_speed(speed);

	//std::cout << "robot config";
	//for(float joint : remote_controller_.current_config())
	//	std::cout << joint << std::endl;

	//std::cout << std::endl;

	if (remote_controller_->do_logging())
		reset_log(start_time);
}

franka_agent::~franka_agent() noexcept
{
	terminate_ = true;
	try { state_update_thread_.join(); }
	catch (...)
	{
		std::cerr << "franka_state_server::~franka_state_server(): " <<
			"Internal thread of franka ctor threw an exception on joining.";
	}
}

bool franka_agent::can_execute_transition(const pn_transition& transition) const
{
	return dynamic_cast<const stack_action*>(&transition)
		|| dynamic_cast<const pick_action*>(&transition)
		|| dynamic_cast<const place_action*>(&transition)
		|| dynamic_cast<const reverse_stack_action*>(&transition);
}

bool franka_agent::execute_transition(const pn_transition& transition)
{
	file << std::endl
		<< get_time_ms().count()
		<< "," << transition.to_string();

	if (auto* action = dynamic_cast<const stack_action*>(&transition))
	{
		return place(action->to.first->box);
	}
	else if (auto* action = dynamic_cast<const pick_action*>(&transition))
	{
		return pick(action->from->box);

	}
	else if (auto* action = dynamic_cast<const place_action*>(&transition))
	{
		return place(action->to->box);
	}
	else if (auto* action = dynamic_cast<const reverse_stack_action*>(&transition))
	{
		return pick(action->from.first->box);
	}
	else
	{
		throw std::runtime_error("Can not execute this action");
	}

	return false;
}

bool franka_agent::approx_reached(const franka_control::robot_config_7dof& pose) const noexcept
{
	return cvt_config(remote_controller_->current_config()).isApprox(pose, 0.01);
}

bool franka_agent::approx_reached(const Eigen::Affine3d& pose) const noexcept
{
	return (franka_proxy::franka_proxy_util::fk(cvt_config(remote_controller_->current_config())).back().translation() - pose.translation()).norm() < 0.01f;
}

bool franka_agent::rest()
{
	try
	{
		franka_control::robot_config_7dof rest_pose;
		rest_pose << 0.0108909,
			-0.483135,
			-0.0079431,
			-2.81406,
			-0.0382218,
			2.3383,
			0.877388;

		if (approx_reached(rest_pose))
		{
			agent_state_.in_rest_position = true;
			return true;
		}

		set_speed(speed);
		move_to(rest_pose);
		agent_state_.in_rest_position = true;

		return true;
	}
	catch (const std::exception&)
	{
		agent_state_.in_rest_position = false;
		return false;
	}
}

bool franka_agent::picknplace(const obb& pick_loc, const obb& place_loc)
{
	/*
	using  namespace std::chrono_literals;
	std::unique_lock lock(franka_lock_, std::try_to_lock);
	if (!lock.owns_lock())
	{
		std::cout << "robot already in use\n";
		return false;
	}

	Eigen::Matrix3d nsa_orientation = get_default_nsa_orientation();


	Eigen::Vector3d tcp_to_t6 = nsa_orientation.col(2) * tcp_to_t6_offset;//offset between t6 and  vacuum gripper TCP in world coordinates
	Eigen::Vector3d centroid_to_tcp = Eigen::Vector3d{ 0,0,pick.diagonal.z() } *0.5;//offset between centroid of box and TCP in world coordinates
	Eigen::Vector3d safety_offset = { 0,0,0.01 };
	float transfer_height = 0.15;




	Eigen::Affine3d grip_pos;
	grip_pos.setIdentity();
	grip_pos.matrix().block<3, 3>(0, 0) = nsa_orientation;
	grip_pos.matrix().block<3, 1>(0, 3) = pick.translation.cast<double>() + centroid_to_tcp + tcp_to_t6 + safety_offset;
	//grip_pos.matrix()(2, 3) += pick.diagonal.z() * 0.5;

	Eigen::Affine3d hover_grip_pos = grip_pos;
	hover_grip_pos.matrix()(2, 3) += transfer_height;

	Eigen::Affine3d drop_pos;
	drop_pos.setIdentity();
	drop_pos.matrix().block<3, 3>(0, 0) = nsa_orientation;
	drop_pos.matrix().block<3, 1>(0, 3) = place.translation.cast<double>() + centroid_to_tcp + tcp_to_t6 + safety_offset;
	//drop_pos.matrix()(2, 3) += pick.diagonal.z() * 0.5;

	Eigen::Affine3d hover_drop_pos = drop_pos;
	hover_drop_pos.matrix()(2, 3) += transfer_height;

	try {
		remote_controller_.vacuum_gripper_drop();

		//ik throws if pose not reachable
		auto hover_grip_pose = franka_control::franka_util::ik_fast_closest(hover_grip_pos, cvt_config(remote_controller_.current_config()));
		auto grip_pose = franka_control::franka_util::ik_fast_closest(grip_pos, hover_grip_pose);
		auto hover_drop_pose = franka_control::franka_util::ik_fast_closest(hover_drop_pos, hover_grip_pose);
		auto drop_pose = franka_control::franka_util::ik_fast_closest(drop_pos, hover_drop_pose);

		std::cout << "Moving to" << hover_grip_pos.translation();
		move_to(hover_grip_pose);
		move_to(grip_pose);

		//approach the object slowly from the top, while trying to vacuum it
		int tries = 0;
		while(true)
		{
			if (remote_controller_.vacuum_gripper_vacuum(10, 1s))
				break;
			else
				std::cout << "getting closer\n";
			tries++;

			grip_pos.matrix()(2, 3) -= 0.005;
			auto gripping = franka_control::franka_util::ik_fast_closest(grip_pos, cvt_config(remote_controller_.current_config()));
			move_to(gripping);

			if (tries > 10)
				throw std::runtime_error("failed vacuuming");

		}
		move_to(hover_grip_pose);
		move_to(hover_drop_pose);
		move_to(drop_pose);

		//for some reason vacuum_gripper_drop throws command exceptions even tho the command was completed
		//FIXME it works this way
		try
		{
			remote_controller_.vacuum_gripper_drop();
		}
		catch (const std::exception& e)
		{
			std::cout << "vacuum drop error: " << e.what()<<"\n";
		}
		move_to(hover_drop_pose);

		return true;
	}
	catch (const std::exception& e)
	{
		std::this_thread::sleep_for(500ms);
		remote_controller_.automatic_error_recovery();
		std::cout << e.what() << "\n";
		try
		{
			//stop vacuum gripper in case its still active
			remote_controller_.vacuum_gripper_drop();
		}
		catch (const std::exception& e)
		{
		}
		return false;
	}*/
	if (pick_loc.diagonal != place_loc.diagonal)
		throw std::runtime_error("pick and place locations need to have same shape");

	std::unique_lock lock(franka_lock_, lock_type);
	if (!lock.owns_lock())
	{
		std::cout << "robot already in use\n";
		return false;
	}

	if (!pick_lockfree(pick_loc))
		return false;
	return place_lockfree(place_loc);
}

bool franka_agent::pick(const obb& pick)
{
	std::unique_lock lock(franka_lock_, lock_type);
	if (!lock.owns_lock())
	{
		std::cout << "robot already in use\n";
		return false;
	}
	return pick_lockfree(pick);
}

bool franka_agent::place(const obb& place)
{
	std::unique_lock lock(franka_lock_, lock_type);
	if (!lock.owns_lock())
	{
		std::cout << "robot already in use\n";
		return false;
	}
	return place_lockfree(place);
}

bool franka_agent::approach(const pn_transition& transition)
{
	file << std::endl
		<< get_time_ms().count()
		<< "," << transition.id << "," << transition.to_string();

	if (auto* action = dynamic_cast<const stack_action*>(&transition))
	{
		return approach(action->to.first->box);
	}
	else if (auto* action = dynamic_cast<const pick_action*>(&transition))
	{
		return approach(action->from->box);

	}
	else if (auto* action = dynamic_cast<const place_action*>(&transition))
	{
		return approach(action->to->box);
	}
	else if (auto* action = dynamic_cast<const reverse_stack_action*>(&transition))
	{
		return approach(action->from.first->box);
	}
	else
	{
		throw std::runtime_error("Can not execute this action");
	}

	return false;
}

bool franka_agent::approach(const obb& box)
{
	//construct nsa-Rotationmatrix gripping orthogonal from on top
	Eigen::Matrix3d nsa_orientation = get_default_nsa_orientation();

	Eigen::Vector3d tcp_to_t6 = nsa_orientation.col(2) * tcp_to_t6_offset;//offset between t6 and  vacuum gripper TCP in world coordinates in (a)pproach direction
	Eigen::Vector3d centroid_to_tcp = Eigen::Vector3d{ 0,0,box.diagonal.z() } *0.5;//offset between centroid of box and TCP in world coordinates
	float transfer_offset = 0.1;

	Eigen::Affine3d grip_pos;
	grip_pos.setIdentity();
	grip_pos.matrix().block<3, 3>(0, 0) = nsa_orientation;
	grip_pos.matrix().block<3, 1>(0, 3) = box.translation.cast<double>() + centroid_to_tcp + tcp_to_t6;

	Eigen::Affine3d hover_grip_pos = grip_pos;
	hover_grip_pos.matrix()(2, 3) += transfer_offset;

	std::unique_lock lock(franka_lock_, lock_type);
	if (!lock.owns_lock())
	{
		std::cout << "robot already in use\n";
		return false;
	}

	try
	{
		//ik throws if pose not reachable
		auto grip_pose = franka_proxy::franka_proxy_util::ik_fast_closest(grip_pos, cvt_config(remote_controller_->current_config()));
		auto hover_grip_pose = franka_proxy::franka_proxy_util::ik_fast_closest(hover_grip_pos, grip_pose);

		if (!approx_reached(hover_grip_pos))
		{
			file << ",move to hover " << get_time_ms().count();
			set_speed(speed);
			move_to(hover_grip_pose);
			agent_state_.on_transfer_height = true;
		}

		return true;
	}
	catch (const std::exception& e)
	{
		file << ",fail " << get_time_ms().count() << "," << e.what();
		return false;
	}


}

bool franka_agent::has_object_gripped() const noexcept
{
	return agent_state_.gripped_object;
}

void franka_agent::log(const std::string& line)
{
	file << std::endl
		<< get_time_ms().count()
		<< line << std::endl;
}

void franka_agent::reset_log(std::chrono::high_resolution_clock::time_point start)
{
	if (file.is_open())
		file.close();

	if (file_poses.is_open())
		file_poses.close();

	this->start_time = start;

	boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
	std::stringstream stream;
	stream << time.date().year()
		<< "-" << time.date().month().as_number()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes();

	std::filesystem::create_directory(stream.str());

	file.open(stream.str() + "/robot_actions.csv");
	file << "time (ms),transition id,action type,object,place id,status" << std::endl;

	file_poses.open(stream.str() + "/robot_config.csv");
	file_poses << "time (ms),vacuuming,object gripped,joint 0,joint 1,joint 2,joint 3,joint 4,joint 5,joint 6" << std::endl;
}

franka_proxy::robot_config_7dof franka_agent::get_config() const
{
	return remote_controller_->current_config();
}

bool franka_agent::pick_lockfree(const obb& pick)
{
	using namespace std::chrono_literals;

	//construct nsa-Rotationmatrix gripping orthogonal from on top
	Eigen::Matrix3d nsa_orientation = get_default_nsa_orientation();


	Eigen::Vector3d tcp_to_t6 = nsa_orientation.col(2) * tcp_to_t6_offset;//offset between t6 and  vacuum gripper TCP in world coordinates in (a)pproach direction
	Eigen::Vector3d centroid_to_tcp = Eigen::Vector3d{ 0,0,pick.diagonal.z() } *0.5;//offset between centroid of box and TCP in world coordinates
	Eigen::Vector3d safety_offset = { 0,0,-0.004f };
	float transfer_offset = 0.1;

	Eigen::Affine3d grip_pos;
	grip_pos.setIdentity();
	grip_pos.matrix().block<3, 3>(0, 0) = nsa_orientation;
	grip_pos.matrix().block<3, 1>(0, 3) = pick.translation.cast<double>() + centroid_to_tcp + tcp_to_t6 + safety_offset;

	Eigen::Affine3d hover_grip_pos = grip_pos;
	hover_grip_pos.matrix()(2, 3) += transfer_offset;

	using config = franka_control::robot_config_7dof;
	config hover_grip_pose;
	config grip_pose;
	config above_grip_pose;
	config above_grip_pose_2;
	try
	{
		//ik throws if pose not reachable
		grip_pose = franka_proxy::franka_proxy_util::ik_fast_closest(grip_pos, cvt_config(remote_controller_->current_config()));
		hover_grip_pose = franka_proxy::franka_proxy_util::ik_fast_closest(hover_grip_pos, grip_pose);

		auto above_grip_pos = grip_pos;
		above_grip_pos.matrix()(2, 3) += 0.008;
		above_grip_pose = franka_proxy::franka_proxy_util::ik_fast_closest(above_grip_pos, grip_pose);
		
		auto above_grip_pos_2 = grip_pos;
		above_grip_pos_2.matrix()(2, 3) += 0.03;
		above_grip_pose_2 = franka_proxy::franka_proxy_util::ik_fast_closest(above_grip_pos_2, grip_pose);
	}
	catch (const std::exception& e)
	{
		std::cout << "unreachable position commanded:" << e.what();
		file << ",unreachable position commanded" << e.what();
		return false;
	}

	try
	{
		if (!approx_reached(hover_grip_pos))
		{
			file << ",move to hover " << get_time_ms().count();
			set_speed(speed);
			move_to(hover_grip_pose);
		}
		agent_state_.on_transfer_height = true;
	}
	catch (const std::exception& e)
	{
		file << "moving towards object failed " << get_time_ms().count() << "," << e.what();
		return false;
	}

	try
	{
		file << ",approaching vertically " << get_time_ms().count();

		move_to(above_grip_pose);
		agent_state_.on_transfer_height = false;
		set_speed(speed_slow);
		move_to(grip_pose);
		//if (!move_to_until_contact(grip_pose))
		//{
		//	std::cout << "no contact";
		//}

	}
	catch (const std::exception& e)
	{
		file << "fail " << get_time_ms().count() << "," << e.what();
		try {
			move_to(hover_grip_pose);
		}
		catch (...)
		{}
		//move_to_until_contact(hover_grip_pose);
		return false;
	}

	try
	{
		//approach the object slowly from the top, while trying to vacuum it
		int tries = 0;
		while (true)
		{
			//remote_controller_->apply_z_force(0.1, 10.);
			file << ",vacuuming " << get_time_ms().count();
			if (remote_controller_->vacuum_gripper_vacuum(71 /* x10 mbar of pressure*/, 500ms))
			{
				file << ",vacuum success " << get_time_ms().count();
				agent_state_.vacuuming = true;
				break;
			}


			tries++;

			if (tries >= 2)
				throw std::runtime_error("failed vacuuming");

			grip_pos.matrix()(2, 3) -= 0.005;
			auto gripping = franka_proxy::franka_proxy_util::ik_fast_closest(grip_pos, cvt_config(remote_controller_->current_config()));
			file << ",getting closer " << get_time_ms().count();
			move_to(gripping);
			/*if (!move_to_until_contact(gripping))
				std::cout << "contact";*/

		}
		agent_state_.vacuuming = true;
		agent_state_.gripped_object = true;
		file << "returning to hover pose " << get_time_ms().count();
		//if (!approx_reached(hover_grip_pos))
		move_to(above_grip_pose_2);
		set_speed(speed);
		move_to(hover_grip_pose);

		agent_state_.on_transfer_height = true;

		file << ",success " << get_time_ms().count();

		return true;
	}
	catch (const std::exception& e)
	{
		//std::cout << "vacuuming failed: " << e.what();
		file << ",failed vacuuming " << get_time_ms().count() << "," << e.what();
		remote_controller_->vacuum_gripper_stop();
		agent_state_.vacuuming = false;
		agent_state_.gripped_object = false;
		try {
			move_to(hover_grip_pose);
		}
		catch (...) {}
		return false;
	}
}

bool franka_agent::place_lockfree(const obb& place)
{
	using  namespace std::chrono_literals;
	Eigen::Matrix3d nsa_orientation = get_default_nsa_orientation();


	Eigen::Vector3d tcp_to_t6 = nsa_orientation.col(2) * tcp_to_t6_offset;//offset between t6 and  vacuum gripper TCP in world coordinates
	Eigen::Vector3d centroid_to_tcp = Eigen::Vector3d{ 0,0,place.diagonal.z() } *0.5;//offset between centroid of box and TCP in world coordinates
	Eigen::Vector3d safety_offset = { 0,0,-0.005 };
	constexpr double transfer_height = 0.1;

	Eigen::Affine3d drop_pos;
	drop_pos.setIdentity();
	drop_pos.matrix().block<3, 3>(0, 0) = nsa_orientation;
	drop_pos.matrix().block<3, 1>(0, 3) = place.translation.cast<double>() + centroid_to_tcp + tcp_to_t6 + safety_offset;

	Eigen::Affine3d hover_drop_pos = drop_pos;
	hover_drop_pos.matrix()(2, 3) += transfer_height;
	using config = franka_control::robot_config_7dof;
	config hover_drop_pose;
	config drop_pose;
	config above_drop_pose;
	config above_drop_pose_2;
	try
	{
		//ik throws if pose not reachable
		drop_pose = franka_proxy::franka_proxy_util::ik_fast_closest(drop_pos, cvt_config(remote_controller_->current_config()));
		hover_drop_pose = franka_proxy::franka_proxy_util::ik_fast_closest(hover_drop_pos, drop_pose);

		auto above_drop_pos = drop_pos;
		above_drop_pos.matrix()(2, 3) += 0.008;
		above_drop_pose = franka_proxy::franka_proxy_util::ik_fast_closest(above_drop_pos, drop_pose);

		auto above_drop_pos_2 = drop_pos;
		above_drop_pos_2.matrix()(2, 3) += 0.03;
		above_drop_pose_2 = franka_proxy::franka_proxy_util::ik_fast_closest(above_drop_pos_2, drop_pose);
	}
	catch (const std::exception& e)
	{
		//std::cout << "unreachable position commanded:" << e.what();
		file << ",unreachable position commanded " << get_time_ms().count() << "," << e.what();
		return false;
	}

	try
	{
		if (!approx_reached(hover_drop_pos))
		{
			file << ",move to hover " << get_time_ms().count();
			set_speed(speed);
			move_to(hover_drop_pose);
		}
		agent_state_.on_transfer_height = true;
	}
	catch (const std::exception& e)
	{
		file << ",fail " << get_time_ms().count() << "," <<e.what();
		// std::cout << "moving towards object failed:" << e.what();
		return false;
	}

	try
	{
		file << ",vertical approaching " << get_time_ms().count();
		move_to(above_drop_pose_2);
		move_to(above_drop_pose);
		agent_state_.on_transfer_height = false;
		set_speed(speed_slow);
		move_to(drop_pose);

	}
	catch (std::exception& e)
	{
		file << ",fail " << get_time_ms().count() << "," << e.what();
		// std::cout << "vertical approaching failed: " << e.what();
		try {
			move_to(hover_drop_pose);
		}
		catch (...) {}
		return false;
	}

	file << ",stop vacuum " << get_time_ms().count();
	remote_controller_->vacuum_gripper_drop();
	agent_state_.gripped_object = false;
	agent_state_.vacuuming = false;

	try
	{
		/*if (!approx_reached(hover_drop_pos))
		{*/
		file << ",move back to hover " << get_time_ms().count();
		move_to(above_drop_pose);
		set_speed(speed);
		move_to(hover_drop_pose);
		agent_state_.on_transfer_height = true;
		//}

		file << ",success " << get_time_ms().count();
		return true;
	}
	catch (const std::exception& e)
	{
		// std::cout << "move back to hover failed" << e.what() << "\n";
		file << ",fail " << get_time_ms().count() << "," << e.what();
		return false;
	}
}

void franka_agent::move_to(const Eigen::Matrix<double, 7, 1>& pose)
{
	agent_state_.in_rest_position = false;
	remote_controller_->execute_retry([this, &pose]()
		{
			remote_controller_->move_to(cvt_config(pose));
		});
}
/*
bool franka_agent::move_to_until_contact(const Eigen::Matrix<double, 7, 1>& pose)
{
	return execute_retry([&]()
	{
		return remote_controller_->move_to_until_contact(cvt_config(pose));
	}, remote_controller_);
}
*/
void franka_agent::set_speed(double target_speed)
{
	remote_controller_->set_speed_factor(target_speed);
}


Eigen::Matrix3d franka_agent::get_default_nsa_orientation()
{
	//constructs the nsa coordinates for gripping straight from above
	Eigen::Matrix3d nsa_orientation;
	Eigen::Vector3d n = { 1,0,0 };
	Eigen::Vector3d s = { 0,-1,0 };
	Eigen::Vector3d a = { 0,0, -1. };

	nsa_orientation.block<3, 1>(0, 0) = n;
	nsa_orientation.block<3, 1>(0, 1) = s;
	nsa_orientation.block<3, 1>(0, 2) = a;
	return nsa_orientation;
}

void franka_agent::taskmain()
{
	if (!remote_controller_->needs_update_loop())
		return;

	while (!terminate_)
	{
		try
		{
			remote_controller_->update();
		}
		catch (...)
		{
			std::this_thread::sleep_for(std::chrono::seconds{ 1 });
			std::cout << "Robot: trying reconnection\n";
			continue;
		}

		file_poses << get_time_ms().count()
			<< "," << agent_state_.vacuuming
			<< "," << agent_state_.gripped_object;

		auto config = remote_controller_->current_config();
		//joint_signal(config);
		for (const auto& val : config)
			file_poses << "," << val;
		file_poses << std::endl;
		
		std::this_thread::sleep_for(update_interval);
	}
}

std::chrono::milliseconds franka_agent::get_time_ms() const
{
	return duration_cast<milliseconds>(high_resolution_clock::now() - start_time);
}

franka_control::robot_config_7dof state_observation::cvt_config(const franka_proxy::robot_config_7dof& config)
{
	auto out = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(config.data());
	return out;
}

franka_proxy::robot_config_7dof cvt_config(const franka_control::robot_config_7dof& config)
{
	std::array<double, 7> out;
	for (int i = 0; i < 7; i++)
		out[i] = config(i, 0);
	
	return out;
}

franka_async_agent::franka_async_agent(std::shared_ptr<Controller>&& controller)
	:
	franka(std::move(controller)),
	terminate_flag{ false },
	worker([this]() { do_work(); })
{
}

franka_async_agent::~franka_async_agent() noexcept
{
	terminate_flag = true;
	worker.join();
}

void franka_async_agent::execute_transition(pn_transition::Ptr transition)
{
	std::scoped_lock lock(task_queue_mutex);
	task_queue.push(transition);
}

bool franka_async_agent::can_execute_transition(const pn_transition& transition) const
{
	return franka.can_execute_transition(transition);
}

void franka_async_agent::do_work()
{
	using namespace std::literals::chrono_literals;
	while (!terminate_flag)
	{
		pn_transition::Ptr task;
		{
			std::scoped_lock s(task_queue_mutex);
			if (!task_queue.empty())
			{
				task = task_queue.front();
				task_queue.pop();
			}
		}
		if (task)
		{
			//this is the synchronous long lasting task -> dont keep the lock during this
			franka.execute_transition(*task);

			std::scoped_lock lock(task_queue_mutex);
			if (auto fit = callbacks.find(task); fit != callbacks.end())
			{
				fit->second();
				callbacks.erase(fit);//every callback should only be executed once
			}
			transition_completed_signal(task); //send the signal for every completed transition
		}
		else
			std::this_thread::sleep_for(20ms);


	}
}

void Controller::execute_retry(std::function<void()> f)
{
	f();
}

bool Controller::do_logging()
{
	return false;
}

std::chrono::high_resolution_clock::time_point Controller::start_time()
{
	return {};
}

bool Controller::needs_update_loop() const
{
	return false;
}

bool Controller::vacuum_gripper_vacuum(std::uint8_t vacuum_strength, std::chrono::milliseconds timeout)
{
	return true;
}

bool Controller::vacuum_gripper_stop()
{
	return true;
}

bool Controller::vacuum_gripper_drop(std::chrono::milliseconds timeout)
{
	return true;
}

void Controller::update()
{}

void Controller::automatic_error_recovery()
{}


}
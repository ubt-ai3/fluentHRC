#include "franka_visualization_gpu.h"

#include <execution>
#include <numbers>
#include <fstream>

#include <gpu_voxels/GpuVoxels.h>

#undef M_PI_2

namespace franka_proxy
{
	namespace Visualize
	{
		GVLInstanceFranka::GVLInstanceFranka()
			: m_instance(gpu_voxels::GpuVoxels::getInstance())
		{
			using namespace FrankaParams;

			const int dim_xy = static_cast<int>(ceilf((2.f * radius) / voxel_side_length));
			const int dim_z = static_cast<int>(ceilf(vertical / voxel_side_length));

			//defining the dimensions of the gpu_voxels instance
			m_instance->initialize(dim_xy, dim_xy, dim_z, voxel_side_length);
			m_instance->addMap(gpu_voxels::MT_BITVECTOR_VOXELMAP, "robotMap");

			std::vector<std::string> linknames(9);
			std::vector<std::string> paths_to_pointclouds(9);

			//adding linknames aka paths to the pointclouds for the robot model
			linknames[0] = paths_to_pointclouds[0] = "franka/link0.binvox";
			linknames[1] = paths_to_pointclouds[1] = "franka/link1.binvox";
			linknames[2] = paths_to_pointclouds[2] = "franka/link2.binvox";
			linknames[3] = paths_to_pointclouds[3] = "franka/link3.binvox";
			linknames[4] = paths_to_pointclouds[4] = "franka/link4.binvox";
			linknames[5] = paths_to_pointclouds[5] = "franka/link5.binvox";
			linknames[6] = paths_to_pointclouds[6] = "franka/link6.binvox";
			linknames[7] = paths_to_pointclouds[7] = "franka/link7.binvox";
			linknames[8] = paths_to_pointclouds[8] = "franka/cobot_pump.binvox";

			std::vector<gpu_voxels::robot::DHParameters<gpu_voxels::robot::CRAIGS>> dh_params(9);

			constexpr float M_PI_2 = std::numbers::pi_v<float> / 2.f;

			//adding dh_params for the robot model
			// _d,  _theta,  _a,   _alpha, _value, _type
			dh_params[0] = { 0.333f, 0.f, 0.0, 0.f, 0.0, gpu_voxels::robot::REVOLUTE };
			dh_params[1] = { 0.0, 0.0, 0.0, -M_PI_2, 0.0, gpu_voxels::robot::REVOLUTE };
			dh_params[2] = { 0.316f, 0.f, 0.f, M_PI_2, 0.0, gpu_voxels::robot::REVOLUTE };
			dh_params[3] = { 0.0, 0.f, 0.0825f, M_PI_2, 0.0, gpu_voxels::robot::REVOLUTE };
			dh_params[4] = { 0.384f, 0.0, -0.0825f, -M_PI_2, 0.0, gpu_voxels::robot::REVOLUTE };
			dh_params[5] = { 0.0, 0.0, 0.f, M_PI_2, 0.0, gpu_voxels::robot::REVOLUTE };
			dh_params[6] = { 0.0, 0.0, 0.088f, M_PI_2, 0.0, gpu_voxels::robot::REVOLUTE };
			//dh_params[7] = { 0.107f, 0.0, 0.0, 0.f, 0.0, gpu_voxels::robot::REVOLUTE };
			dh_params[7] = { 0.105f, 0.0, 0.0, 0.f, 0.0, gpu_voxels::robot::REVOLUTE };
			dh_params[8] = { 0.f, 0.0, 0.0, 0.f, 0.0, gpu_voxels::robot::REVOLUTE };

			//add robot model to the gpu_voxels instance
			m_instance->addRobot("robot", linknames, dh_params, paths_to_pointclouds, std::filesystem::path("./assets/models"));

			gpu_voxels::robot::JointValueMap min_joint_values;
			min_joint_values["franka/link0.binvox"] = -2.8973f;
			min_joint_values["franka/link1.binvox"] = -1.7628f;
			min_joint_values["franka/link2.binvox"] = -2.8973f;
			min_joint_values["franka/link3.binvox"] = -3.0718f;
			min_joint_values["franka/link4.binvox"] = -2.8973f;
			min_joint_values["franka/link5.binvox"] = -0.0175f;
			min_joint_values["franka/link6.binvox"] = -2.8973f;
			min_joint_values["franka/link7.binvox"] = 0.f;
			min_joint_values["franka/cobot_pump.binvox"] = 0.f;

			gpu_voxels::robot::JointValueMap max_joint_values;
			max_joint_values["franka/link0.binvox"] = 2.8973f;
			max_joint_values["franka/link1.binvox"] = 1.7628f;
			max_joint_values["franka/link2.binvox"] = 2.8973f;
			max_joint_values["franka/link3.binvox"] = -0.0698f;
			max_joint_values["franka/link4.binvox"] = 2.8973f;
			max_joint_values["franka/link5.binvox"] = 3.7525f;
			max_joint_values["franka/link6.binvox"] = 2.8973f;
			max_joint_values["franka/link7.binvox"] = 0.f;
			max_joint_values["franka/cobot_pump.binvox"] = 0.f;

			//set robot origin in the center of the robots extent
			m_instance->setRobotBaseTransformation("robot", Eigen::Affine3f(Eigen::Translation3f(radius, radius, down)).matrix());
		}

		franka_joint_motion_voxelizer::franka_joint_motion_voxelizer(Visualize::GVLInstanceFranka& gpu_instance)
			: m_gpu_instance(gpu_instance)
		{}

		Visualize::VoxelRobot franka_joint_motion_voxelizer::discretize_path(const franka_joint_motion_sampler& sampler) const
		{
			using namespace Visualize::FrankaParams;

			m_gpu_instance.m_instance->clearMap("robotMap");

			Visualize::VoxelRobot res;

			//set joints and insert into voxel map
			for (const auto joints : sampler | std::views::elements<0>)
			{
				const Eigen::Vector<float, 7> joints_f = joints.cast<float>();

				gpu_voxels::robot::JointValueMap joint_values =
				{
					{ "franka/link0.binvox", joints_f[0] },
					{ "franka/link1.binvox", joints_f[1] },
					{ "franka/link2.binvox", joints_f[2] },
					{ "franka/link3.binvox", joints_f[3] },
					{ "franka/link4.binvox", joints_f[4] },
					{ "franka/link5.binvox", joints_f[5] },
					{ "franka/link6.binvox", joints_f[6] }
				};

				m_gpu_instance.m_instance->setRobotConfiguration("robot", joint_values);
				m_gpu_instance.m_instance->insertRobotIntoMap("robot", "robotMap", gpu_voxels::eBVM_OCCUPIED);
			}
			res.robot_origin = origin.matrix();
			res.voxel_length = voxel_side_length;

			//get map the robot was inserted into and extract sparse voxel representation
			const auto robotMap = m_gpu_instance.m_instance->getMap("robotMap")->as<gpu_voxels::voxelmap::BitVectorVoxelMap>();
			res.voxels = gpu_voxels::voxelmap::extract_visual_voxels(robotMap->getDeviceData(), robotMap->getDimensions());

			return res;
		}
	}
}
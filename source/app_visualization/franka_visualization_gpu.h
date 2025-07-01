#pragma once

#include <franka_voxel/motion_generator_joint_max_accel.h>

namespace gpu_voxels
{
	class GpuVoxels;
	typedef std::shared_ptr<gpu_voxels::GpuVoxels> GpuVoxelsSharedPtr;
}

namespace franka_proxy
{
	namespace Visualize
	{
		/**
		 * FrankaParams
		 * 
		 * Parameters describing the extent of the franka panda
		 * and the discretization resolution
		 */
		namespace FrankaParams
		{
			static constexpr float voxel_side_length = 0.018f;

			static constexpr float down = 0.360f;
			static constexpr float up = 1.190f;

			static constexpr float vertical = down + up;

			static constexpr float radius = 0.855f;

			static const Eigen::Affine3f origin = Eigen::Affine3f(Eigen::Translation3f(radius, radius, down));
		}

		struct GVLInstanceFranka
		{
			GVLInstanceFranka();

			gpu_voxels::GpuVoxelsSharedPtr m_instance;
		};

		/**
 		* VoxelRobot
		*
		* Data necessary to sparsely represent robot as voxels in a grid
 		*/
		struct VoxelRobot
		{
			float voxel_length;
			Eigen::Matrix4f robot_origin;
			std::vector<Eigen::Vector<uint32_t, 3>> voxels;
		};

		/**
		 * franka_joint_motion_voxelizer
		 * 
		 * utility class for generating voxel robots from a sampled motion
		 */
		class franka_joint_motion_voxelizer
		{
		public:
			franka_joint_motion_voxelizer(Visualize::GVLInstanceFranka& gpu_instance);

			/**
			 * \brief
			 * \return a discretized path with meta information
			 */
			[[nodiscard]] Visualize::VoxelRobot discretize_path(const franka_joint_motion_sampler& sampler) const;

		private:

			Visualize::GVLInstanceFranka& m_gpu_instance;
		};
	}
}




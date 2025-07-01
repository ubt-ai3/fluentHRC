#pragma once

#include <chrono>
#include <tuple>
#include <variant>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/impl/point_types.hpp>

#include "robot.pb.h"

namespace server
{
	struct holo_pointcloud
	{
		typedef std::shared_ptr<holo_pointcloud> Ptr;

		pcl::PointCloud<pcl::PointXYZ>::Ptr pcl;
		std::chrono::file_clock::time_point timestamp;
		std::chrono::file_clock::time_point recv_timestamp;

		[[nodiscard]] std::chrono::duration<int64_t, std::ratio<1, 10'000'000>> get_latency() const
		{
			return recv_timestamp - timestamp;
		}
	};

	template<typename T>
	using VisualUpdate = std::tuple<T, std::chrono::steady_clock::time_point>;

	template<typename T>
	using MaybeChange = std::variant<T, generated::Visual_Change>;

	typedef MaybeChange<generated::Voxels> VoxelData;
	typedef MaybeChange<generated::Tcps> TcpsData;
	typedef MaybeChange<generated::Sync_Joints_Array> SyncJointsData;

	typedef VisualUpdate<VoxelData> VoxelUpdate;
	typedef VisualUpdate<TcpsData> TcpsUpdate;
	typedef VisualUpdate<SyncJointsData> JointsProgressUpdate;
}
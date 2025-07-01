#pragma once
#include <WinSock2.h>
#include <boost/asio.hpp>
#include <state_observation/object_prototype_loader.hpp>
#include "service_impl.h"

namespace server
{
	class server_module
	{
	public:

		server_module(const state_observation::computed_workspace_parameters& workspace_parameters, const std::string& addr = "0.0.0.0:50051");

		mesh_service meshes_service;
		object_prototype_service proto_service;
		debug_service service_dos;
		pcl_service service_pcl;
		object_service obj_service;
		hand_tracking_service service_hands;
		VoxelService voxel_service;

		grpc::Server& get_instance();

		void load_mesh_data(const state_observation::object_prototype_loader& loader);

	private:

		std::unique_ptr<grpc::Server> server;

	};
}

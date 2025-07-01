#include "server_module.h"

#include "util.h"

namespace server
{
	using namespace generated;
	
	server_module::server_module(const state_observation::computed_workspace_parameters& workspace_parameters, const std::string& addr)
		: service_pcl(workspace_parameters)
	{
		grpc::EnableDefaultHealthCheckService(true);
		grpc::ServerBuilder builder;

		builder.RegisterService(&meshes_service);
		builder.RegisterService(&service_dos);
		builder.RegisterService(&service_pcl);
		builder.RegisterService(&proto_service);
		builder.RegisterService(&obj_service);
		builder.RegisterService(&service_hands);
		builder.RegisterService(&voxel_service);

		builder.AddListeningPort(addr, grpc::InsecureServerCredentials());

		builder.SetSyncServerOption(grpc::ServerBuilder::MAX_POLLERS, 20);
		builder.SetSyncServerOption(grpc::ServerBuilder::MIN_POLLERS, 10);

		server = builder.BuildAndStart();
	}

	void server_module::load_mesh_data(
		const state_observation::object_prototype_loader& loader)
	{
		tinyobj::ObjReaderConfig reader_config;
		reader_config.mtl_search_path = "./";

		std::set<std::string> loaded;
		for (const auto& prototype : loader.get_prototypes())
		{
			const auto& base_mesh = prototype->get_base_mesh();
			if (!base_mesh)
				continue;

			proto_service.object_prototypes.emplace(prototype->get_name(),
				convert<Object_Prototype>(prototype));

			const auto& path = base_mesh->get_path();
			if (!loaded.emplace(path).second) continue;

			tinyobj::ObjReader reader;
			if (!reader.ParseFromFile(path, reader_config))
			{
				std::cerr << reader.Error() << std::endl;
				continue;
			}

			meshes_service.meshes.emplace(
				path, convert<Mesh_Data>(std::make_pair(reader, path)));
		}
	}

	grpc::Server& server_module::get_instance()
	{
		return *server;
	}
}
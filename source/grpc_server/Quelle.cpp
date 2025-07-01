#include <WinSock2.h>
#include <boost/asio.hpp>
#include "object_prototype_loader.hpp"
#include "point_cloud_processing.h"

#include "visualizer.h"
#include "server_module.h"

int main(int argc, char** argv) {

	state_observation::computed_workspace_parameters workspace_params(false);

	server::server_module server(workspace_params);
	visualizer vis;

	state_observation::object_prototype_loader loader;
	server.load_mesh_data(loader);

	server.service_pcl.
	debug_connection.
	connect([&vis](incremental_point_cloud::Ptr pcl)
		{
			vis.set_pcl(pcl);
		});
	
	std::thread t([&server]()
		{
			server.get_instance().Wait();
		});

	vis.start();
	
	server.get_instance().Shutdown(
		std::chrono::system_clock::now() + 
		std::chrono::milliseconds(500));
	t.join();

	return 0;
}
#pragma once
#include <boost/signals2.hpp>
#include <grpcpp/grpcpp.h>
#include <Eigen/Eigen>

#include "wrapper.hpp"
#include "vertex.grpc.pb.h"
#include "object_prototype.grpc.pb.h"
#include "debug.grpc.pb.h"
#include "depth_image.grpc.pb.h"
#include "object.grpc.pb.h"
#include "hand_tracking.grpc.pb.h"
#include "robot.grpc.pb.h"

#include "point_cloud_processing.h"

class incremental_point_cloud;
typedef std::shared_ptr<incremental_point_cloud> Ptr;

namespace server
{
	using namespace generated;
	
	class mesh_service final : public mesh_com::Service
	{
	public:

		grpc::Status transmit_mesh_data(
			grpc::ServerContext* context,
			grpc::ServerReaderWriter<Mesh_Data_TF_Meta, named_request>* stream) override;

		std::map<std::string, Mesh_Data> meshes;

	private:

	};

	class object_prototype_service final : public object_prototype_com::Service
	{
	public:

		grpc::Status transmit_object_prototype(
			grpc::ServerContext* context,
			grpc::ServerReaderWriter<Object_Prototype_TF_Meta, named_request>* stream) override;

		std::map<std::string, Object_Prototype> object_prototypes;

	private:


	};

	class debug_service final : public debug_com::Service
	{
	public:

		grpc::Status transmit_debug_info(
			grpc::ServerContext* context,
			const debug_client* request,
			debug_server* response) override;

	private:


	};

	struct pcl_handle {
		
		typedef std::shared_ptr<pcl_handle> Ptr;

		incremental_point_cloud::Ptr point_cloud;
		std::optional<state_observation::obb> oriented_bb;
	};

	class pcl_service final : public pcl_com::Service
	{
	public:

		pcl_service(const state_observation::computed_workspace_parameters& workspace_parameters);
		~pcl_service() override;
		
		grpc::Status transmit_pcl_data(
			grpc::ServerContext* context,
			grpc::ServerReader<Pcl_Data_Meta>* reader,
			ICP_Result* response) override;
			
		grpc::Status transmit_obb(
			grpc::ServerContext* context, 
			const Obb_Meta* request,
			google::protobuf::Empty* response) override;	

		void set_ref(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr raw = nullptr);

		boost::signals2::signal<void(incremental_point_cloud::Ptr)> debug_connection;
	
	private:

		static void debug_dump(
			const incremental_point_cloud::Ptr& cloud_holo,
			const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_kinect,
			const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_kinect_raw,
			const std::optional<state_observation::obb>& oriented_bb,
			const Eigen::Affine3f& icp_trafo);

		std::mutex ref_mtx;
		pcl::PointCloud<pcl::PointXYZ>::ConstPtr ref_cloud;
		pcl::PointCloud<pcl::PointXYZ>::ConstPtr raw_cloud;
		
		bool stop = false;
		pcl_handle::Ptr get_data_handle(const std::string& uri);
		bool remove_data_handle(const std::string& uri);
		
		std::mutex map_mutex;
		std::map<std::string, pcl_handle::Ptr> data_handles;

		const state_observation::computed_workspace_parameters& workspace_parameters_;
	};
	

	class pending_objects
	{
	public:

		void add_object(const std::shared_ptr<Object_Instance>& object);
		void delete_object(const std::string& id);

		std::vector<std::shared_ptr<Object_Instance>> get_to_send();
		std::vector<std::string> get_to_delete();

		void stop_action();

	private:

		bool stop = false;

		std::set<std::string> to_delete;
		std::map<std::string, std::shared_ptr<Object_Instance>> to_send;

		std::condition_variable to_send_exists;
		std::condition_variable to_delete_exists;

		std::mutex to_delete_mutex;
		std::mutex to_send_mutex;
	};

	class object_service final : public object_com::Service
	{
	public:

		object_service();
		~object_service() override;

		grpc::Status sync_objects(
			grpc::ServerContext* context,
			const google::protobuf::Empty* request,
			grpc::ServerWriter<Object_Instance_TF_Meta>* writer) override;

		grpc::Status transmit_object(
			grpc::ServerContext* context,
			const google::protobuf::Empty* request,
			grpc::ServerWriter<Object_Instance_TF_Meta>* writer) override;

		grpc::Status delete_object(
			grpc::ServerContext* context,
			const google::protobuf::Empty* request,
			grpc::ServerWriter<Delete_Request>* writer) override;


		boost::signals2::slot<void(const std::shared_ptr<Object_Instance>&)> add_object;
		boost::signals2::slot<void(const std::string& id)> add_delete_object;
		boost::signals2::signal<std::vector<std::shared_ptr<Object_Instance>>()> sync_signal;

		bool stop = false;


	private:

		std::shared_ptr<pending_objects> get_handler(const std::string& uri);

		std::mutex map_mutex;
		std::map<std::string, std::shared_ptr<pending_objects>> handlers;
	};

	class hand_tracking_service final : public hand_tracking_com::Service
	{
	public:

		~hand_tracking_service() override;

		grpc::Status transmit_hand_data(
			grpc::ServerContext* context, 
			grpc::ServerReader<generated::Hand_Data_Meta>* reader, 
			google::protobuf::Empty* response) override;
		
		inline static boost::signals2::signal<void(const hand_pose_estimation::hololens::hand_data::ConstPtr&)> on_tracked_hand;

	private:

		std::mutex mtx;
		std::set<grpc::ServerContext*> contexts;
	};

	class VoxelService : public generated::robot_com::Service
	{
	private:

		

		template<typename T>
		struct TransmissionBuffer
		{
			std::unique_ptr<VisualUpdate<MaybeChange<T>>> buffer;

			std::condition_variable cv;
			std::mutex mutex;
		};

		template<typename T>
		bool wait_for_new_data(TransmissionBuffer<T>& t_buffer, const std::chrono::steady_clock::time_point& current_tp);
		

	public:
		
		//typedef std::tuple<std::array<double, 7>, std::chrono::utc_clock::time_point> joints_sync;
		//typedef std::vector<joints_sync> joints_progress;
		//typedef VisualUpdate<Joints> JointUpdate;

		VoxelService();
		~VoxelService() override;

		grpc::Status transmit_voxels(::grpc::ServerContext* context, const ::google::protobuf::Empty* request, ::grpc::ServerWriter<::generated::Voxel_Transmission>* writer) override;
		//grpc::Status transmit_joints(grpc::ServerContext* context, const google::protobuf::Empty* request, grpc::ServerWriter<generated::Joints>* writer) override;
		grpc::Status transmit_tcps(grpc::ServerContext* context, const google::protobuf::Empty* request, grpc::ServerWriter<generated::Tcps_Transmission>* writer) override;
		grpc::Status transmit_sync_joints(grpc::ServerContext* context, const google::protobuf::Empty* request, grpc::ServerWriter<generated::Sync_Joints_Transmission>* writer) override;

		boost::signals2::slot<void(const server::VoxelData&)> voxel_slot;
		//boost::signals2::slot<void(const std::array<double, 7>&)> joint_slot;
		boost::signals2::slot<void(const server::TcpsData&)> tcps_slot;
		boost::signals2::slot<void(const server::SyncJointsData&)> joints_progress_slot;

	private:

		void stop();
		
		TransmissionBuffer<Voxels> voxel_buffer;
		//TransmissionBuffer<generated::Joints> joint_buffer;
		TransmissionBuffer<Tcps> tcp_buffer;
		TransmissionBuffer<Sync_Joints_Array> joints_buffer;

		bool stop_flag = false;
	};
}
#include "service_impl.h"

#include <filesystem>
#include <fstream>

#include <boost/archive/xml_oarchive.hpp>

#include "util.h"

namespace server
{
    grpc::Status mesh_service::transmit_mesh_data(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<Mesh_Data_TF_Meta, named_request>* stream)
    {
        std::list<named_request> requests;
        stream->SendInitialMetadata();

        named_request req;
        while (stream->Read(&req))
            requests.emplace_back(req);

        TF_Stream_Wrapper wrapper(gen_meta());
        for (const auto& request : requests)
        {
            auto it = meshes.find(request.name());
            if (it == meshes.end())
                continue;

            if (!stream->Write(stream_meta<Mesh_Data_TF_Meta>(it->second, wrapper)))
                return grpc::Status::CANCELLED;
        }
        return grpc::Status::OK;
    }

    grpc::Status object_prototype_service::transmit_object_prototype(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<Object_Prototype_TF_Meta, named_request>* stream)
    {
        std::list<named_request> requests;
        stream->SendInitialMetadata();

        named_request req;
        while (stream->Read(&req))
            requests.emplace_back(req);

        TF_Stream_Wrapper wrapper(gen_meta());
        for (const auto& request : requests)
        {
            auto it = object_prototypes.find(request.name());
            if (it == object_prototypes.end())
                continue;

            if (!stream->Write(stream_meta<Object_Prototype_TF_Meta>(it->second, wrapper)))
                return grpc::Status::CANCELLED;
        }
        return grpc::Status::OK;
    }

    grpc::Status debug_service::transmit_debug_info(
        grpc::ServerContext* context,
        const debug_client* request,
        debug_server* response)
    {
        std::cout << "[" + context->peer() + "]: " << request->message() << std::endl;
        return grpc::Status::OK;
    }

    pcl_service::pcl_service(const state_observation::computed_workspace_parameters& workspace_parameters)
	    : workspace_parameters_(workspace_parameters)
    {
    }

    pcl_service::~pcl_service()
    {
        stop = true;
    }

    grpc::Status pcl_service::transmit_pcl_data(
        grpc::ServerContext* context,
        grpc::ServerReader<Pcl_Data_Meta>* reader,
        ICP_Result* response)
    {
        context->set_compression_level(GRPC_COMPRESS_LEVEL_HIGH);
        reader->SendInitialMetadata();

        auto handle = get_data_handle(context->peer());
    	
		auto& [point_cloud, oriented_bb] = *handle;

        TF_Conv_Wrapper tf_wrapper;
        Pcl_Data_Meta read;
        
        while (reader->Read(&read))
        {
            auto cloud = std::make_shared<holo_pointcloud>(convert_meta<holo_pointcloud>(read, tf_wrapper));

            auto latency = cloud->get_latency();
            auto seconds =
                std::chrono::duration_cast<std::chrono::seconds>(latency).count();
            auto millis =
                std::chrono::duration_cast<std::chrono::milliseconds>(latency).count()
                - seconds * 1000;

            std::cout << seconds << "seconds, " << millis << "milliseconds" << std::endl;

            point_cloud->insert(std::move(cloud));            
        }

        if(remove_data_handle(context->peer()))
        {
	        //Point cloud matching
            pcl::PointCloud<pcl::PointXYZ>::Ptr cpy, raw;
            {
                std::unique_lock lock(ref_mtx);
                if (ref_cloud)
                {
                    cpy = ref_cloud->makeShared();
                    if (raw_cloud)
                        raw = raw_cloud->makeShared();
                }
                else
                    return grpc::Status::OK;
            }

            const Eigen::Affine3f matrix = oriented_bb.has_value()
                ? point_cloud->register_pcl_2<pcl::PointXYZ>(raw, oriented_bb.value(), workspace_parameters_)
                : Eigen::Affine3f::Identity();

            /*Eigen::Affine3f matrix = ((oriented_bb.has_value()) 
                ? point_cloud->register_pcl<pcl::PointXYZ>(cpy, oriented_bb.value())
                : point_cloud->register_pcl<pcl::PointXYZ>(cpy));*/

            auto& data = *response->mutable_data();
            //*data.mutable_matrix() = convert<4, 4>((Eigen::Translation3f(0.015f, -0.02f, -0.006f) * matrix).inverse().matrix());
            *data.mutable_matrix() = convert<4, 4>(matrix.inverse().matrix());
            *data.mutable_transformation_meta() = gen_meta();

            debug_dump(point_cloud, cpy, raw, oriented_bb, matrix);
        }      
        
        return grpc::Status::OK;
    }

	grpc::Status pcl_service::transmit_obb(
        grpc::ServerContext* context, 
        const Obb_Meta* request,
        google::protobuf::Empty* response)
    {
        auto handle = get_data_handle(context->peer());

        TF_Conv_Wrapper wrapper;
        handle->oriented_bb = convert_meta<state_observation::obb>(*request, wrapper);

        return grpc::Status::OK;
    }

    void pcl_service::set_ref(
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr raw)
    {
        std::unique_lock lock(ref_mtx);
        ref_cloud = cloud;
        raw_cloud = raw;
    }
	
	void pcl_service::debug_dump(
        const incremental_point_cloud::Ptr& cloud_holo,
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_kinect,
        const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_kinect_raw,
        const std::optional<state_observation::obb>& oriented_bb,
        const Eigen::Affine3f& icp_trafo)
    {
        auto time_id = std::to_string(std::chrono::system_clock::now().
            time_since_epoch().count());

        std::filesystem::path dir_name = "icp_" + time_id;

        std::filesystem::create_directory(dir_name);

        cloud_holo->save((dir_name / "hololens").string());
        pcl::io::savePLYFileBinary((dir_name / "kinect.ply").string(), *cloud_kinect);
        if (cloud_kinect_raw)
			pcl::io::savePLYFileBinary((dir_name / "kinect_raw.ply").string(), *cloud_kinect_raw);

        if (oriented_bb.has_value())
        {
            std::ofstream ofs(dir_name / "obb.xml");
            boost::archive::xml_oarchive ar(ofs);

            ar << boost::serialization::make_nvp("obb", oriented_bb.value());
            ar << boost::serialization::make_nvp("trafo", icp_trafo);
            /*
            auto& bb = oriented_bb.value();

            Eigen::Affine3f obb =
                Eigen::Translation3f(bb.translation) *
                Eigen::Affine3f(bb.rotation);// *
                //Eigen::Affine3f(Eigen::Scaling(bb.diagonal / 2.));

            Eigen::IOFormat format;
            format.coeffSeparator = ',';
            std::ofstream file("obb_affine_" + time_id + ".txt");
            std::ofstream file("obb_affine_diagonal_" + time_id + ".txt");
            file << obb.matrix().format(format);
            file << bb.diagonal().
            file.close();*/
        }
    }
	
    void pending_objects::stop_action()
    {
        stop = true;
        to_send_exists.notify_all();
        to_delete_exists.notify_all();
    }

    pcl_handle::Ptr pcl_service::get_data_handle(const std::string& uri)
    {
        std::unique_lock lock(map_mutex);
        if (stop) return nullptr;

        auto it = data_handles.find(uri);
        if (it != data_handles.end())
            return it->second;

        const auto& temp = data_handles.emplace(
            uri, std::make_shared<pcl_handle>(pcl_handle{
                std::make_shared<incremental_point_cloud>(),
                std::optional<state_observation::obb>()
                }));
        debug_connection(temp.first->second->point_cloud);
    	
        return temp.first->second;
    }

    bool pcl_service::remove_data_handle(const std::string& uri)
    {
        std::unique_lock lock(map_mutex);
        return data_handles.erase(uri);
    }

    void pending_objects::add_object(const std::shared_ptr<Object_Instance>& object)
    {
        {
            std::unique_lock lock(to_delete_mutex);
            to_delete.erase(object->id());
        }

        std::unique_lock lock(to_send_mutex);

        auto it = to_send.emplace(object->id(), object);
        if (!it.second)
            it.first->second = object;
        lock.unlock();

        to_send_exists.notify_all();
    }

    void pending_objects::delete_object(const std::string& id)
    {
        std::unique_lock lock(to_delete_mutex);
        to_delete.emplace(id);

        to_delete_exists.notify_all();
    }

    std::vector<std::shared_ptr<Object_Instance>> pending_objects::get_to_send()
    {
        std::unique_lock lock(to_send_mutex);
        to_send_exists.wait(lock, [this]() { return !to_send.empty() || stop; });
        if (stop) return {};

        std::map<std::string, std::shared_ptr<Object_Instance>> temp;
        to_send.swap(temp);
        lock.unlock();

        std::vector<std::shared_ptr<Object_Instance>> payload;
        payload.reserve(temp.size());
        for (const auto& it : temp)
            payload.emplace_back(it.second);
        return payload;
    }

    std::vector<std::string> pending_objects::get_to_delete()
    {
        std::unique_lock lock(to_delete_mutex);
        to_delete_exists.wait(lock, [this]() { return !to_delete.empty() || stop; });
        if (stop) return {};

        std::set<std::string> temp;
        to_delete.swap(temp);
        lock.unlock();

        std::vector<std::string> payload;
        payload.reserve(temp.size());
        for (const auto& it : temp)
            payload.emplace_back(it);
        return payload;
    }

    object_service::object_service()
        : add_object([this](const std::shared_ptr<Object_Instance>& instance)
            {
                std::unique_lock lock(map_mutex);
                for (const auto& handler : handlers)
                    handler.second->add_object(instance);
            }),
        add_delete_object([this](const std::string& id)
            {
                std::unique_lock lock(map_mutex);
                for (const auto& handler : handlers)
                    handler.second->delete_object(id);
            })
    {}

    object_service::~object_service()
    {
        stop = true;

        std::unique_lock lock(map_mutex);
        for (auto& handler : handlers)
            handler.second->stop_action();
    }

    grpc::Status object_service::sync_objects(
        grpc::ServerContext* context,
        const google::protobuf::Empty* request,
        grpc::ServerWriter<Object_Instance_TF_Meta>* writer)
    {
        writer->SendInitialMetadata();
        auto res = sync_signal();

        if (!res.has_value())
            return grpc::Status::CANCELLED;

        TF_Stream_Wrapper wrapper(gen_meta());
        for (const auto& instance : res.value())
            writer->Write(stream_meta<Object_Instance_TF_Meta>(*instance, wrapper));

        return grpc::Status::OK;
    }

    grpc::Status object_service::transmit_object(
        grpc::ServerContext* context,
        const google::protobuf::Empty* request,
        grpc::ServerWriter<Object_Instance_TF_Meta>* writer)
    {
        writer->SendInitialMetadata();
        auto handler = get_handler(context->peer());

        TF_Stream_Wrapper wrapper(gen_meta());
        bool healthy = true;
        while (healthy && !stop)
        {
            const auto& to_send = handler->get_to_send();
            for (const auto& instance : to_send)
                healthy = writer->Write(stream_meta<Object_Instance_TF_Meta>(*instance, wrapper));
        }
        return grpc::Status::OK;
    }

    grpc::Status object_service::delete_object(
        grpc::ServerContext* context,
        const google::protobuf::Empty* request,
        grpc::ServerWriter<Delete_Request>* writer)
    {
        writer->SendInitialMetadata();
        auto handler = get_handler(context->peer());

        bool healthy = true;
        while (healthy && !stop)
        {
            const auto& to_delete = handler->get_to_delete();
            for (const auto& id : to_delete)
            {
                Delete_Request req;
                req.set_id(id);
                healthy = writer->Write(req);
            }
        }
        return grpc::Status::OK;
    }

    std::shared_ptr<pending_objects> object_service::get_handler(const std::string& uri)
    {
        std::unique_lock lock(map_mutex);
        if (stop) return nullptr;

        auto it = handlers.find(uri);
        if (it != handlers.end())
            return it->second;

        const auto& temp = handlers.emplace(uri, std::make_shared<pending_objects>());
        return temp.first->second;
    }

    hand_tracking_service::~hand_tracking_service()
    {
        std::unique_lock lock(mtx);
    	for (auto& ctx : contexts)
            ctx->TryCancel();
    }

    grpc::Status hand_tracking_service::transmit_hand_data(
        grpc::ServerContext* context,
        grpc::ServerReader<generated::Hand_Data_Meta>* reader,
        google::protobuf::Empty* response)
    {
        {
            std::unique_lock lock(mtx);
            contexts.emplace(context);
        }
        reader->SendInitialMetadata();

        TF_Conv_Wrapper wrapper;
        generated::Hand_Data_Meta data;
        while (reader->Read(&data))
            on_tracked_hand(std::make_shared<const hand_pose_estimation::hololens::hand_data>(convert_meta<hand_pose_estimation::hololens::hand_data>(data, wrapper)));
        {
            std::unique_lock lock(mtx);
        	contexts.erase(context);
        }    	
        return grpc::Status::OK;
    }

    VoxelService::VoxelService()
        : voxel_slot([this](const VoxelData& data)
            {
                const auto tp = std::chrono::steady_clock::now();

        		std::unique_lock lock(voxel_buffer.mutex);
				if (stop_flag)
					return;
			    //overwrite buffer if still didn't change
			    //we don't need outdated voxel visuals anyway
			    voxel_buffer.buffer = std::make_unique<VoxelUpdate>(data, tp);
			    voxel_buffer.cv.notify_one();
            }),
		/*joint_slot([this](const std::array<double, 7>& data)
            {
                const auto tp = std::chrono::steady_clock::now();

                std::unique_lock lock(joint_buffer.mutex);
                if (stop_flag)
                    return;

                joint_buffer.buffer = std::make_unique<JointUpdate>(server::convert<generated::Joints>(data), tp);
                joint_buffer.cv.notify_one();
            }),*/
		tcps_slot([this](const TcpsData& data)
            {
                const auto tp = std::chrono::steady_clock::now();

                std::unique_lock lock(tcp_buffer.mutex);
                if (stop_flag)
                    return;

                tcp_buffer.buffer = std::make_unique<TcpsUpdate>(data, tp);
                tcp_buffer.cv.notify_one();
            }),
		joints_progress_slot([this](const SyncJointsData& data)
		{
            const auto tp = std::chrono::steady_clock::now();

            std::unique_lock lock(joints_buffer.mutex);
            if (stop_flag)
                return;

            //joints_buffer.buffer = std::make_unique<JointsProgressUpdate>(server::convert<generated::Sync_Joints_Array>(data), tp);
            joints_buffer.buffer = std::make_unique<JointsProgressUpdate>(data, tp);
            joints_buffer.cv.notify_one();
		})
    {}

    VoxelService::~VoxelService()
    {
        stop();
    }

    grpc::Status VoxelService::transmit_voxels(
        ::grpc::ServerContext* context,
        const::google::protobuf::Empty* request,
        ::grpc::ServerWriter<::generated::Voxel_Transmission>* writer)
    {
        context->set_compression_algorithm(GRPC_COMPRESS_GZIP);
        writer->SendInitialMetadata();

        std::chrono::steady_clock::time_point current_tp{};

        TF_Stream_Wrapper wrapper(gen_meta_voxels());
        while (true)
        {
            if (!wait_for_new_data(voxel_buffer, current_tp))
                break;

            auto& [payload, new_tp] = *voxel_buffer.buffer;
            current_tp = new_tp;
            
            if (!writer->Write(server::stream_meta<generated::Voxel_Transmission>(payload, wrapper)))
                return grpc::Status::CANCELLED;
        }
        return grpc::Status::OK;
    }
    /*
    grpc::Status VoxelService::transmit_joints(
        grpc::ServerContext* context, 
        const google::protobuf::Empty* request, 
        grpc::ServerWriter<generated::Joints>* writer)
    {
        writer->SendInitialMetadata();

        std::chrono::steady_clock::time_point current_tp{};
        
        while (true)
        {
            if (!wait_for_new_data(joint_buffer, current_tp))
                break;

            auto& [payload, new_tp] = *joint_buffer.buffer;
            current_tp = new_tp;

            if (!writer->Write(payload))
                return grpc::Status::CANCELLED;
        }
        return grpc::Status::OK;
    }*/

    grpc::Status VoxelService::transmit_tcps(
        grpc::ServerContext* context, 
        const google::protobuf::Empty* request, 
        grpc::ServerWriter<generated::Tcps_Transmission>* writer)
    {
        writer->SendInitialMetadata();

        std::chrono::steady_clock::time_point current_tp{};

        TF_Stream_Wrapper wrapper(gen_meta());
        while (true)
        {
            if (!wait_for_new_data(tcp_buffer, current_tp))
                break;

            auto& [payload, new_tp] = *tcp_buffer.buffer;
            current_tp = new_tp;

            if (!writer->Write(server::stream_meta<generated::Tcps_Transmission>(payload, wrapper)))
                return grpc::Status::CANCELLED;
        }
        return grpc::Status::OK;
    }

    grpc::Status VoxelService::transmit_sync_joints(
        grpc::ServerContext* context, 
        const google::protobuf::Empty* request, 
        grpc::ServerWriter<generated::Sync_Joints_Transmission>* writer)
    {
        writer->SendInitialMetadata();

        std::chrono::steady_clock::time_point current_tp{};
        
        while (true)
        {
            if (!wait_for_new_data(joints_buffer, current_tp))
                break;

            auto& [payload, new_tp] = *joints_buffer.buffer;
            current_tp = new_tp;

            if (!writer->Write(server::convert<generated::Sync_Joints_Transmission>(payload)))
                return grpc::Status::CANCELLED;
        }
        return grpc::Status::OK;
    }

    void VoxelService::stop()
    {
        stop_flag = true;
        {
            std::scoped_lock lock(voxel_buffer.mutex);
            voxel_buffer.cv.notify_all();
        }
        /*{
            std::scoped_lock lock(joint_buffer.mutex);
            joint_buffer.cv.notify_all();
        }*/
        {
            std::scoped_lock lock(tcp_buffer.mutex);
            tcp_buffer.cv.notify_all();
        }
        {
            std::scoped_lock lock(joints_buffer.mutex);
            joints_buffer.cv.notify_all();
        }
    }

    template<typename T>
    bool VoxelService::wait_for_new_data(TransmissionBuffer<T>& t_buffer, const std::chrono::steady_clock::time_point& current_tp)
    {
        auto& [buffer, cv, mutex] = t_buffer;

        std::unique_lock lock(mutex);
        cv.wait(lock, [this, &current_tp, &buffer]()
            {
                return (!!buffer && get<1>(*buffer) > current_tp)
                    || stop_flag;
            });

        return !stop_flag;
    }
}
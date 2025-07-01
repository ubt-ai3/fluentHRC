#include "util.h"

#include "proto_plugin.h"
#include <base-transformation/Plugins/eigen.h>
#include <base-transformation/Plugins/pcl.h>

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };

namespace server
{
	inline static Transformation::TransformationMeta CoreMeta(
		{ Transformation::Axis::Y, Transformation::AxisDirection::POSITIVE },
		{ Transformation::Axis::X, Transformation::AxisDirection::NEGATIVE },
		{ Transformation::Axis::Z, Transformation::AxisDirection::POSITIVE }
	);

	TF_Stream_Wrapper::TF_Stream_Wrapper(generated::Transformation_Meta meta)
		: m_meta(std::move(meta))
	{}

	std::optional<generated::Transformation_Meta> TF_Stream_Wrapper::get_meta() const
	{
		if (!first)
			return {};

		first = false;
		return m_meta;
	}

	generated::Transformation_Meta gen_meta()
	{
		generated::Transformation_Meta out;
		auto& right = *out.mutable_right();
		right.set_axis(generated::Y);
		right.set_direction(generated::Axis_Direction::POSITIVE);

		auto& forward = *out.mutable_forward();
		forward.set_axis(generated::X);
		forward.set_direction(generated::Axis_Direction::NEGATIVE);

		auto& up = *out.mutable_up();
		up.set_axis(generated::Z);
		up.set_direction(generated::Axis_Direction::POSITIVE);

		auto& scale = *out.mutable_scale();
		scale.set_num(1);
		scale.set_denom(1);

		return out;
	}

	generated::Transformation_Meta gen_meta_voxels()
	{
		generated::Transformation_Meta out;

		auto& right = *out.mutable_right();
		right.set_axis(generated::X);
		right.set_direction(generated::POSITIVE);

		auto& forward = *out.mutable_forward();
		forward.set_axis(generated::Y);
		forward.set_direction(generated::POSITIVE);

		auto& up = *out.mutable_up();
		up.set_axis(generated::Z);
		up.set_direction(generated::POSITIVE);

		auto& scale = *out.mutable_scale();
		scale.set_num(1);
		scale.set_denom(1);

		return out;
	}

	template<>
	generated::quaternion convert(const Eigen::Quaternionf& in)
	{
		generated::quaternion out;

		out.set_x(in.x());
		out.set_y(in.y());
		out.set_z(in.z());
		out.set_w(in.w());

		return out;
	}

	template<>
	Eigen::Quaternionf convert_meta(const generated::quaternion& in, const Transformation::TransformationConverter* cv)
	{
		if (cv == nullptr)
			return { in.w(), in.x(), in.y(), in.z() };

		return cv->convert_quaternion<QuaternionEigen>(QuaternionProtoConst{ in });
	}

	template<>
	generated::vertex_3d convert(const Eigen::Vector3f& v)
	{
		generated::vertex_3d out;
		out.set_x(v.x());
		out.set_y(v.y());
		out.set_z(v.z());

		return out;
	}

	template<>
	generated::index_3d convert(const Eigen::Vector<uint32_t, 3>& v)
	{
		generated::index_3d out;
		out.set_x(v.x());
		out.set_y(v.y());
		out.set_z(v.z());

		return out;
	}

	template<>
	generated::size_3d convert(const Eigen::Vector3f& in)
	{
		generated::size_3d out;
		out.set_x(in.x());
		out.set_y(in.y());
		out.set_z(in.z());

		return out;
	}

	template<>
	generated::color convert(const pcl::RGB& in)
	{
		generated::color out;

		out.set_r(in.r);
		out.set_g(in.g);
		out.set_b(in.b);
		out.set_a(in.a);

		return out;
	}

	template<>
	generated::aabb convert(const state_observation::aabb& in)
	{
		generated::aabb out;
		*out.mutable_diagonal() =
			convert<generated::size_3d, Eigen::Vector3f>(in.diagonal);

		*out.mutable_translation() =
			convert<generated::vertex_3d, Eigen::Vector3f>(in.translation);

		return out;
	}

	template<>
	google::protobuf::RepeatedField<google::protobuf::uint32> convert(
		const std::vector<tinyobj::index_t>& indices)
	{
		google::protobuf::RepeatedField<uint32_t> out;
		out.Reserve(indices.size());

		for (const auto& index : indices)
			out.AddAlreadyReserved(index.vertex_index);

		return out;
	}

	template<>
	google::protobuf::RepeatedPtrField<generated::vertex_3d> convert(
		const std::vector<tinyobj::real_t>& vertices)
	{
		google::protobuf::RepeatedPtrField<generated::vertex_3d> out;
		out.Reserve(vertices.size());

		for (size_t i = 0; i < vertices.size(); i += 3)
		{
			generated::vertex_3d outVertex;
			outVertex.set_x(vertices[i]);
			outVertex.set_y(vertices[i + 1]);
			outVertex.set_z(vertices[i + 2]);
			out.Add(std::move(outVertex));
		}
		return out;
	}

	template<>
	google::protobuf::RepeatedPtrField<generated::vertex_3d_no_scale> convert(
		const std::vector<tinyobj::real_t>& vertices)
	{
		google::protobuf::RepeatedPtrField<generated::vertex_3d_no_scale> out;
		out.Reserve(vertices.size());

		for (size_t i = 0; i < vertices.size(); i += 3)
		{
			generated::vertex_3d_no_scale outVertex;
			outVertex.set_x(vertices[i]);
			outVertex.set_y(vertices[i + 1]);
			outVertex.set_z(vertices[i + 2]);
			out.Add(std::move(outVertex));
		}
		return out;
	}

	template<>
	generated::Object_Prototype convert(
		const state_observation::object_prototype::ConstPtr& in)
	{
		generated::Object_Prototype proto;
		*proto.mutable_bounding_box() =
			convert<generated::aabb>(in->get_bounding_box());

		*proto.mutable_mean_color() =
			convert<generated::color>(in->get_mean_color());

		auto& col = *proto.mutable_mean_color();
		auto f = [](int c)
			{
				float x = std::exp(10 * c / 255.f - 4.4);
				return static_cast<int>(x / (x + 1) * 255.f);
			};
		col.set_r(f(col.r()));
		col.set_g(f(col.g()));
		col.set_b(f(col.b()));

		proto.set_mesh_name(in->get_base_mesh()->get_path());
		proto.set_name(in->get_name());
		proto.set_type(in->get_type());

		return proto;
	}

	template<>
	generated::Mesh_Data convert(
		const std::pair<tinyobj::ObjReader, std::string>& obj_pair)
	{
		generated::Mesh_Data temp;
		const auto& reader = obj_pair.first;
		const auto& name = obj_pair.second;

		const auto& attr = reader.GetAttrib();
		const auto& vertices = attr.GetVertices();
		const auto& normals = attr.normals;

		const auto& indices = reader.GetShapes()[0].mesh.indices;

		std::vector<std::set<size_t>> vertex_normals_pre;
		vertex_normals_pre.resize(vertices.size() / 3);

		for (const auto& index : indices)
			vertex_normals_pre[index.vertex_index].emplace(index.normal_index);

		const auto m_vertex_normals =
			temp.mutable_vertex_normals()->mutable_vertices();
		m_vertex_normals->Reserve(vertex_normals_pre.size());

		for (const auto& set : vertex_normals_pre)
		{
			Eigen::Matrix<float, 3, 1> normal;
			normal.setZero();

			for (const auto& idx : set)
			{
				normal[0] += normals[3 * idx + 0];
				normal[1] += normals[3 * idx + 1];
				normal[2] += normals[3 * idx + 2];
			}
			normal.normalize();
			m_vertex_normals->Add(convert<generated::vertex_3d>(normal));
		}

		*temp.mutable_vertices() =
			convert<google::protobuf::RepeatedPtrField<generated::vertex_3d_no_scale>>(vertices);

		*temp.mutable_indices() =
			convert<google::protobuf::RepeatedField<google::protobuf::uint32>>(indices);

		temp.set_name(name);

		return temp;
	}

	template<>
	pcl::PointXYZ convert(const Eigen::Vector3f& in)
	{
		return { in.x(), in.y(), in.z() };
	}

	template<>
	pcl::PointXYZ convert_meta(const generated::vertex_3d& in, const Transformation::TransformationConverter* cv)
	{
		if (cv == nullptr)
			return { in.x(), in.y(), in.z() };

		return cv->convert_point<Vector3PCL>(Vector3ProtoConst{ in });
	}

	template<>
	Eigen::Vector4f convert(const pcl::PointXYZ& in)
	{
		return { in.x, in.y, in.z, 1.f };
	}

	template<>
	Eigen::Vector3f convert_meta(const generated::size_3d& in, const Transformation::TransformationConverter* cv)
	{
		if (cv == nullptr)
			return { in.x(), in.y(), in.z() };

		return cv->convert_size<Size3Eigen>(Size3ProtoConst{ in });
	}

	template<>
	Eigen::Vector3f convert_meta(const generated::vertex_3d& in, const Transformation::TransformationConverter* cv)
	{
		if (cv == nullptr)
			return { in.x(), in.y(), in.z() };

		return cv->convert_point<Vector3Eigen>(Vector3ProtoConst{ in });
	}

	/*template<>
	Eigen::Quaternionf convert(const generated::vertex_3d& in)
	{
		return Eigen::Quaternionf(
			Eigen::AngleAxisf(in.x(), Eigen::Vector3f::UnitX()) *
			Eigen::AngleAxisf(in.y(), Eigen::Vector3f::UnitY()) *
			Eigen::AngleAxisf(in.z(), Eigen::Vector3f::UnitZ()));
	}*/

	template<>
	state_observation::obb convert_meta(const generated::Obb_Meta& in, TF_Conv_Wrapper& cv)
	{
		using namespace Transformation;
		if (in.has_transformation_meta())
			cv.set_source(convert<TransformationMeta>(in.transformation_meta()));

		return convert_meta<state_observation::obb>(in.obb(), &cv.converter());
	}

	template<>
	state_observation::obb convert_meta(const generated::Obb& in, const Transformation::TransformationConverter* cv)
	{
		const auto& a_aligned = in.axis_aligned();
		const auto& rot = in.rotation();

		return {
			convert_meta<Eigen::Vector3f>(a_aligned.diagonal(), cv),
			convert_meta<Eigen::Vector3f>(a_aligned.translation(), cv),
			convert_meta<Eigen::Quaternionf>(rot, cv)
		};
	}

	template<>
	hand_pose_estimation::hololens::hand_index convert(const generated::hand_index& in)
	{
		return static_cast<hand_pose_estimation::hololens::hand_index>(in);
	}

	template<>
	hand_pose_estimation::hololens::tracking_status convert(const generated::tracking_status& in)
	{
		return static_cast<hand_pose_estimation::hololens::tracking_status>(in);
	}

	template<>
	hand_pose_estimation::hololens::hand_data convert_meta(const generated::Hand_Data_Meta& in, TF_Conv_Wrapper& cv)
	{
		using namespace Transformation;
		if (in.has_transformation_meta())
			cv.set_source(convert<TransformationMeta>(in.transformation_meta()));

		return convert_meta<hand_pose_estimation::hololens::hand_data>(in.hand_data(), &cv.converter());
	}

	template<>
	hand_pose_estimation::hololens::hand_data convert_meta(const generated::Hand_Data& in, const Transformation::TransformationConverter* cv)
	{
		hand_pose_estimation::hololens::hand_data out;
		out.valid = in.valid();

		out.hand = convert<hand_pose_estimation::hololens::hand_index>(in.hand());
		out.tracking_stat = convert<hand_pose_estimation::hololens::tracking_status>(in.tracking_stat());

		out.grip_position = convert_meta<Eigen::Vector3f>(in.grip_position(), cv);
		out.grip_rotation = convert_meta<Eigen::Quaternionf>(in.grip_rotation(), cv);

		out.aim_position = convert_meta<Eigen::Vector3f>(in.aim_position(), cv);
		out.aim_rotation = convert_meta<Eigen::Quaternionf>(in.aim_rotation(), cv);

		if (in.hand_key_positions_size() != in.hand_key_radii_size() ||
			in.hand_key_radii_size() != in.hand_key_rotations_size())
			throw std::exception("hand_pose_estimation::hololens::hand_data <-> generated::hand_data: size mismatch");

		const size_t size = in.hand_key_radii_size();

		for (size_t i = 0; i < size; ++i)
		{
			out.key_data[i] = hand_pose_estimation::hololens::hand_key_data
			{
				convert_meta<Eigen::Vector3f>(in.hand_key_positions(i), cv),
				convert_meta<Eigen::Quaternionf>(in.hand_key_rotations(i), cv),
				(cv == nullptr) ? in.hand_key_radii(i) : cv->convert_scale(in.hand_key_radii(i))
			};
		}

		out.is_grasped = in.is_grasped();
		out.utc_timestamp = std::chrono::time_point<std::chrono::utc_clock, hundreds_of_nanoseconds>{
			hundreds_of_nanoseconds{in.utc_timestamp()}
		};

		return out;
	}

	template<>
	holo_pointcloud convert_meta(const generated::Pcl_Data_Meta& in, TF_Conv_Wrapper& cv)
	{
		using namespace Transformation;
		if (in.has_transformation_meta())
			cv.set_source(convert<TransformationMeta>(in.transformation_meta()));

		return convert_meta<holo_pointcloud>(in.pcl_data(), &cv.converter());
	}

	template<>
	holo_pointcloud convert_meta(const generated::Pcl_Data& pcl_data, const Transformation::TransformationConverter* cv)
	{
		const auto recv_timestamp = std::chrono::file_clock::now();

		holo_pointcloud out;

		out.recv_timestamp = recv_timestamp;
		out.timestamp = std::chrono::file_clock::time_point(
			hundreds_of_nanoseconds(pcl_data.timestamp()));

		out.pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

		out.pcl->header.stamp =
			std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::file_clock::to_utc(out.timestamp).time_since_epoch()).count();

		out.pcl->reserve(pcl_data.vertices_size());

		for (const auto& p : pcl_data.vertices())
			out.pcl->emplace_back(server::convert_meta<pcl::PointXYZ>(p, cv));

		return out;
	}

	template<>
	generated::Mesh_Data_TF_Meta stream_meta(const generated::Mesh_Data& in, const TF_Stream_Wrapper& wrapper)
	{
		generated::Mesh_Data_TF_Meta out;
		out.mutable_mesh_data()->CopyFrom(in);

		if (const auto meta = wrapper.get_meta(); meta.has_value())
			*out.mutable_transformation_meta() = meta.value();

		return out;
	}

	template<>
	generated::Object_Instance_TF_Meta stream_meta(const generated::Object_Instance& in, const TF_Stream_Wrapper& wrapper)
	{
		generated::Object_Instance_TF_Meta out;
		out.mutable_object_instance()->CopyFrom(in);

		if (const auto meta = wrapper.get_meta(); meta.has_value())
			*out.mutable_transformation_meta() = meta.value();

		return out;
	}

	template<>
	generated::Object_Prototype_TF_Meta stream_meta(const generated::Object_Prototype& in, const TF_Stream_Wrapper& wrapper)
	{
		generated::Object_Prototype_TF_Meta out;
		out.mutable_object_prototype()->CopyFrom(in);

		if (const auto meta = wrapper.get_meta(); meta.has_value())
			*out.mutable_transformation_meta() = meta.value();

		return out;
	}

	template<>
	generated::Voxel_TF_Meta stream_meta(const generated::Voxels& in, const TF_Stream_Wrapper& wrapper)
	{
		generated::Voxel_TF_Meta out;
		out.mutable_voxels()->CopyFrom(in);

		if (const auto meta = wrapper.get_meta(); meta.has_value())
			*out.mutable_transformation_meta() = meta.value();

		return out;
	}

	template<>
	generated::Tcps_TF_Meta stream_meta(const generated::Tcps& v, const TF_Stream_Wrapper& wrapper)
	{
		generated::Tcps_TF_Meta out;
		out.mutable_tcps()->CopyFrom(v);

		if (const auto meta = wrapper.get_meta(); meta.has_value())
			*out.mutable_transformation_meta() = meta.value();

		return out;
	}

	template<>
	generated::Tcps convert(const std::vector<Eigen::Vector3f>& v)
	{
		generated::Tcps out;
		auto& out_data = *out.mutable_points();
		out_data.Reserve(v.size());
		for (const auto& val : v)
			out_data.Add(convert<generated::vertex_3d>(val));

		return out;
	}

	template<>
	generated::Joints convert(const Eigen::Vector<double, 7>& v)
	{
		generated::Joints out;
		out.set_theta_1(v[0]);
		out.set_theta_2(v[1]);
		out.set_theta_3(v[2]);
		out.set_theta_4(v[3]);
		out.set_theta_5(v[4]);
		out.set_theta_6(v[5]);
		out.set_theta_7(v[6]);

		return out;
	}

	template<>
	generated::Joints convert(const std::array<double, 7>& v)
	{
		generated::Joints out;
		out.set_theta_1(v[0]);
		out.set_theta_2(v[1]);
		out.set_theta_3(v[2]);
		out.set_theta_4(v[3]);
		out.set_theta_5(v[4]);
		out.set_theta_6(v[5]);
		out.set_theta_7(v[6]);

		return out;
	}
	
	void TF_Conv_Wrapper::set_source(const Transformation::TransformationMeta& meta)
	{
		m_converter = std::make_unique<Transformation::TransformationConverter>(meta, CoreMeta);
	}

	const Transformation::TransformationConverter& TF_Conv_Wrapper::converter() const
	{
		return *m_converter;
	}

	bool TF_Conv_Wrapper::has_converter() const
	{
		return !!m_converter;
	}

	template<>
	Transformation::AxisAlignment convert(const generated::Axis_Alignment& in)
	{
		return {
			static_cast<Transformation::Axis>(in.axis()),
			static_cast<Transformation::AxisDirection>(in.direction())
		};
	}

	template<>
	Transformation::Ratio convert(const generated::Ratio& in)
	{
		return {
			in.num(),
			in.denom()
		};
	}

	template<>
	Transformation::TransformationMeta convert(const generated::Transformation_Meta& in)
	{
		return {
			convert<Transformation::AxisAlignment>(in.right()),
			convert<Transformation::AxisAlignment>(in.forward()),
			convert<Transformation::AxisAlignment>(in.up()),
			convert<Transformation::Ratio>(in.scale())
		};
	}

	template<>
	generated::Sync_Joints convert(const std::tuple<std::array<double, 7>, std::chrono::system_clock::time_point>& in)
	{
		generated::Sync_Joints out;

		const auto& [joints, tp] = in;

		*out.mutable_joints() = convert<generated::Joints>(joints);
		out.set_utc_timepoint(std::chrono::duration<double>(tp.time_since_epoch()).count());

		return out;
	}

	template<>
	generated::Sync_Joints_Array convert(const std::vector<std::tuple<std::array<double, 7>, std::chrono::system_clock::time_point>>& in)
	{
		generated::Sync_Joints_Array out;
		auto& sync_joints = *out.mutable_sync_joints();
		sync_joints.Reserve(in.size());

		for (const auto& sync_joints_in : in)
			sync_joints.Add(server::convert<generated::Sync_Joints>(sync_joints_in));

		return out;
	}

	template<>
	generated::Voxel_Transmission stream_meta(const MaybeChange<generated::Voxels>& in, const TF_Stream_Wrapper& wrapper)
	{
		generated::Voxel_Transmission out;
		
		std::visit(overloaded{
			[&out, &wrapper](const generated::Voxels& data)
			{
				*out.mutable_voxels_data() = stream_meta<generated::Voxel_TF_Meta>(data, wrapper);
			},
			[&out](const generated::Visual_Change& data)
			{
				out.set_state_update(data);
			}}, in);

		return out;
	}

	template<>
	generated::Sync_Joints_Transmission convert(const MaybeChange<generated::Sync_Joints_Array>& in)
	{
		generated::Sync_Joints_Transmission out;

		std::visit(overloaded{
			[&out](const generated::Sync_Joints_Array& data)
			{
				*out.mutable_sync_joints_data() = data;
			},
			[&out](const generated::Visual_Change& data)
			{
				out.set_state_update(data);
			} }, in);

		return out;
	}

	template<>
	generated::Tcps_Transmission stream_meta(const MaybeChange<generated::Tcps>& in, const TF_Stream_Wrapper& wrapper)
	{
		generated::Tcps_Transmission out;

		std::visit(overloaded{
			[&out, &wrapper](const generated::Tcps& data)
			{
				*out.mutable_tcps_data() = stream_meta<generated::Tcps_TF_Meta>(data, wrapper);
			},
			[&out](const generated::Visual_Change& data)
			{
				out.set_state_update(data);
			} }, in);

		return out;
	}
}
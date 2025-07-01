#pragma once

#include <Eigen/Dense>
#include <pcl/impl/point_types.hpp>
#include <tiny_obj_loader.h>

#include <state_observation/workspace_objects.hpp>

#include "depth_image.pb.h"
#include "vertex.pb.h"
#include "object_prototype.pb.h"
#include "hand_tracking.pb.h"
#include "object.pb.h"
#include "robot.pb.h"

#include "wrapper.hpp"
#include <hand_pose_estimation/hololens_hand_data.hpp>

#include <base-transformation/TransformationHelper.h>

typedef std::chrono::duration<int64_t, std::ratio<1, 10'000'000>> hundreds_of_nanoseconds;

namespace server
{
	/**
	 * \brief Conversion wrapper for outgoing meta streams
	 */
	class TF_Stream_Wrapper
	{
	public:

		TF_Stream_Wrapper(generated::Transformation_Meta meta);

		//only returns valid meta on first stream interaction
		[[nodiscard]] std::optional<generated::Transformation_Meta> get_meta() const;

	private:

		generated::Transformation_Meta m_meta;
		mutable bool first = true;
	};

	/**
	 * \brief Conversion wrapper for incoming meta transformations
	 */
	class TF_Conv_Wrapper
	{
	public:

		TF_Conv_Wrapper() = default;
		void set_source(const Transformation::TransformationMeta& meta);
		[[nodiscard]] const Transformation::TransformationConverter& converter() const;
		[[nodiscard]] bool has_converter() const;

	private:

		std::unique_ptr<Transformation::TransformationConverter> m_converter;
	};

	template<typename out, typename in>
	out convert(const in& v);

	template<typename out, typename in>
	out stream_meta(const in& v, const TF_Stream_Wrapper& wrapper);

	template<typename out, typename in>
	out convert_meta(const in&, TF_Conv_Wrapper& cv);

	template<typename out, typename in>
	out convert_meta(const in&, const Transformation::TransformationConverter* cv = nullptr);

	generated::Transformation_Meta gen_meta();
	generated::Transformation_Meta gen_meta_voxels();

	template<>
	generated::quaternion convert(const Eigen::Quaternionf& in);

	template<>
	Eigen::Quaternionf convert_meta(const generated::quaternion& in, const Transformation::TransformationConverter* cv);
	
	template<>
	generated::vertex_3d convert(const Eigen::Vector3f& v);

	template<>
	generated::index_3d convert(const Eigen::Vector<uint32_t, 3>& v);

	template<>
	generated::size_3d convert_meta(const Eigen::Vector3f& in, const Transformation::TransformationConverter* cv);

	template<>
	generated::color convert(const pcl::RGB& in);

	template<>
	generated::aabb convert(const state_observation::aabb& in);

	template<>
	google::protobuf::RepeatedField<google::protobuf::uint32> convert(
		const std::vector<tinyobj::index_t>& indices);

	template<>
	google::protobuf::RepeatedPtrField<generated::vertex_3d> convert(
		const std::vector<tinyobj::real_t>& vertices);

	template<>
	google::protobuf::RepeatedPtrField<generated::vertex_3d_no_scale> convert(
		const std::vector<tinyobj::real_t>& vertices);

	template<int rows, int cols>
	Eigen::Matrix<float, rows, cols> convert(const generated::Matrix& m)
	{
		Eigen::Matrix<float, rows, cols> matrix(cols, rows);
		if constexpr (rows >= 0)
			if (m.rows() != rows)
				throw std::exception("Invalid rows");
		if constexpr (cols >= 0)
			if (m.rows() != cols)
				throw std::exception("Invalid cols");

		for (size_t y = 0; y < m.rows(); ++y)
			for (size_t x = 0; x < m.cols(); ++x)
				matrix(x, y) = m.data()[y * m.cols() + x];

		return matrix;
	}

	template<>
	generated::Object_Prototype convert(
		const state_observation::object_prototype::ConstPtr& in);

	template<>
	generated::Mesh_Data convert(
		const std::pair<tinyobj::ObjReader, std::string>& obj_pair);

	template<>
	pcl::PointXYZ convert(const Eigen::Vector3f& in);

	template<>
	pcl::PointXYZ convert_meta(const generated::vertex_3d& in, const Transformation::TransformationConverter* cv);
	
	template<>
	Eigen::Vector4f convert(const pcl::PointXYZ& in);

	template<int rows, int cols>
	generated::Matrix convert(const Eigen::Matrix<float, rows, cols>& in)
	{
		generated::Matrix out;
		out.set_rows(in.rows());
		out.set_cols(in.cols());
		auto data = out.mutable_data();
		data->Reserve(in.rows() * in.cols());

		for (size_t y = 0; y < in.rows(); ++y)
			for (size_t x = 0; x < in.cols(); ++x)
				if constexpr (in.Options == Eigen::RowMajor)
				{
					*data->Add() = in(y, x);
				}
				else
					*data->Add() = in(x, y);
		return out;
	}
	
	template<>
	Eigen::Vector3f convert(const generated::size_3d& in);

	template<>
	Eigen::Vector3f convert(const generated::vertex_3d& in);

	/*template<>
	Eigen::Quaternionf convert(const generated::vertex_3d& in)
	{
		return Eigen::Quaternionf(
			Eigen::AngleAxisf(in.x(), Eigen::Vector3f::UnitX()) *
			Eigen::AngleAxisf(in.y(), Eigen::Vector3f::UnitY()) *
			Eigen::AngleAxisf(in.z(), Eigen::Vector3f::UnitZ()));
	}*/

	template<>
	state_observation::obb convert_meta(const generated::Obb_Meta& in, TF_Conv_Wrapper& cv);

	template<>
	state_observation::obb convert_meta(const generated::Obb& in, const Transformation::TransformationConverter* cv);

	template<>
	hand_pose_estimation::hololens::hand_index convert(const generated::hand_index& in);

	template<>
	hand_pose_estimation::hololens::tracking_status convert(const generated::tracking_status& in);
	
	template<>
	hand_pose_estimation::hololens::hand_data convert_meta(const generated::Hand_Data_Meta& in, TF_Conv_Wrapper& cv);

	template<>
	hand_pose_estimation::hololens::hand_data convert_meta(const generated::Hand_Data& in, const Transformation::TransformationConverter* cv);

	template<>
	holo_pointcloud convert_meta(const generated::Pcl_Data_Meta& in, TF_Conv_Wrapper& cv);

	template<>
	holo_pointcloud convert_meta(const generated::Pcl_Data& pcl_data, const Transformation::TransformationConverter* cv);

	template<>
	generated::Mesh_Data_TF_Meta stream_meta(const generated::Mesh_Data& in, const TF_Stream_Wrapper& wrapper);

	template<>
	generated::Object_Instance_TF_Meta stream_meta(const generated::Object_Instance& in, const TF_Stream_Wrapper& wrapper);

	template<>
	generated::Object_Prototype_TF_Meta stream_meta(const generated::Object_Prototype& in, const TF_Stream_Wrapper& wrapper);

	template<>
	generated::Voxel_TF_Meta stream_meta(const generated::Voxels& in, const TF_Stream_Wrapper& wrapper);

	template<>
	generated::Tcps_TF_Meta stream_meta(const generated::Tcps& v, const TF_Stream_Wrapper& wrapper);

	template<>
	generated::Tcps convert(const std::vector<Eigen::Vector3f>& v);

	template<>
	generated::Joints convert(const Eigen::Vector<double, 7>& v);

	template<>
	generated::Joints convert(const std::array<double, 7>& v);

	template<>
	Transformation::AxisAlignment convert(const generated::Axis_Alignment& in);

	template<>
	Transformation::Ratio convert(const generated::Ratio& in);

	template<>
	Transformation::TransformationMeta convert(const generated::Transformation_Meta& in);

	template<>
	generated::Sync_Joints convert(const std::tuple<std::array<double, 7>, std::chrono::system_clock::time_point>& in);

	template<>
	generated::Sync_Joints_Array convert(const std::vector<std::tuple<std::array<double, 7>, std::chrono::system_clock::time_point>>& in);




	template<>
	generated::Voxel_Transmission stream_meta(const MaybeChange<generated::Voxels>& in, const TF_Stream_Wrapper& wrapper);

	template<>
	generated::Sync_Joints_Transmission convert(const MaybeChange<generated::Sync_Joints_Array>& in);

	template<>
	generated::Tcps_Transmission stream_meta(const MaybeChange<generated::Tcps>& in, const TF_Stream_Wrapper& wrapper);
}
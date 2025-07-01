#pragma once
#include "vertex.pb.h"

#include <array>

class Vector3Proto
{
public:

	typedef generated::vertex_3d type;

	Vector3Proto(generated::vertex_3d& vector)
		: vector(vector)
	{}

	void set_x(float x);
	[[nodiscard]] float get_x() const;

	void set_y(float y);
	[[nodiscard]] float get_y() const;

	void set_z(float z);
	[[nodiscard]] float get_z() const;

	void operator()(size_t idx, float value);
	[[nodiscard]] float operator()(size_t idx) const;

	generated::vertex_3d& vector;

private:

	static std::array<void(generated::vertex_3d::*)(float), 3> setter;
};

class Vector3ProtoConst
{
public:

	typedef generated::vertex_3d type;

	Vector3ProtoConst(const generated::vertex_3d& vector)
		: vector(vector)
	{}

	[[nodiscard]] float get_x() const;
	[[nodiscard]] float get_y() const;
	[[nodiscard]] float get_z() const;

	[[nodiscard]] float operator()(size_t idx) const;

	const generated::vertex_3d& vector;
};

class Size3Proto
{
public:

	typedef generated::size_3d type;

	Size3Proto(generated::size_3d& vector)
		: vector(vector)
	{}

	void set_x(float x);
	[[nodiscard]] float get_x() const;

	void set_y(float y);
	[[nodiscard]] float get_y() const;

	void set_z(float z);
	[[nodiscard]] float get_z() const;

	void operator()(size_t idx, float value);
	[[nodiscard]] float operator()(size_t idx) const;

	generated::size_3d& vector;

private:

	static std::array<void(generated::size_3d::*)(float), 3> setter;
};

class Size3ProtoConst
{
public:

	typedef generated::size_3d type;

	Size3ProtoConst(const generated::size_3d& vector)
		: vector(vector)
	{}

	[[nodiscard]] float get_x() const;
	[[nodiscard]] float get_y() const;
	[[nodiscard]] float get_z() const;

	[[nodiscard]] float operator()(size_t idx) const;

	const generated::size_3d& vector;
};

class QuaternionProto
{
public:

	typedef generated::quaternion type;

	QuaternionProto(generated::quaternion& quaternion)
		: quaternion(quaternion)
	{}

	void set_x(float x);
	[[nodiscard]] float get_x() const;

	void set_y(float y);
	[[nodiscard]] float get_y() const;

	void set_z(float z);
	[[nodiscard]] float get_z() const;

	void set_w(float w);
	[[nodiscard]] float get_w() const;

	void operator()(size_t idx, float value);
	[[nodiscard]] float operator()(size_t idx) const;

	generated::quaternion& quaternion;

private:

	static std::array<void(generated::quaternion::*)(float), 4> setter;
};

class QuaternionProtoConst
{
public:

	typedef generated::quaternion type;

	QuaternionProtoConst(const generated::quaternion& quaternion)
		: quaternion(quaternion)
	{}

	[[nodiscard]] float get_x() const;
	[[nodiscard]] float get_y() const;
	[[nodiscard]] float get_z() const;
	[[nodiscard]] float get_w() const;

	[[nodiscard]] float operator()(size_t idx) const;

	const generated::quaternion& quaternion;
};
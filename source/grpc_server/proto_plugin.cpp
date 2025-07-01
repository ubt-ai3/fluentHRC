#include "proto_plugin.h"


std::array<float(generated::vertex_3d::*)() const, 3> vector3proto_getter = {
		&generated::vertex_3d::x,
		&generated::vertex_3d::y,
		&generated::vertex_3d::z
};

std::array<float(generated::size_3d::*)() const, 3> size3proto_getter = {
		&generated::size_3d::x,
		&generated::size_3d::y,
		&generated::size_3d::z
};

std::array<float(generated::quaternion::*)() const, 4> quaternionProto_getter = {
		&generated::quaternion::x,
		&generated::quaternion::y,
		&generated::quaternion::z,
		&generated::quaternion::w
};


void QuaternionProto::set_x(float x)
{
	quaternion.set_x(x);
}

float QuaternionProto::get_x() const
{
	return quaternion.x();
}

void QuaternionProto::set_y(float y)
{
	quaternion.set_y(y);
}

float QuaternionProto::get_y() const
{
	return quaternion.y();
}

void QuaternionProto::set_z(float z)
{
	quaternion.set_z(z);
}

float QuaternionProto::get_z() const
{
	return quaternion.z();
}

void QuaternionProto::set_w(float w)
{
	quaternion.set_w(w);
}

float QuaternionProto::get_w() const
{
	return quaternion.w();
}

void QuaternionProto::operator()(size_t idx, float value)
{
	(quaternion.*setter[idx])(value);
}

float QuaternionProto::operator()(size_t idx) const
{
	return (quaternion.*quaternionProto_getter[idx])();
}

std::array<void(generated::quaternion::*)(float), 4> QuaternionProto::setter = {
		&generated::quaternion::set_x,
		&generated::quaternion::set_y,
		&generated::quaternion::set_z,
		&generated::quaternion::set_w
};

void Vector3Proto::set_x(float x)
{
	vector.set_x(x);
}

float Vector3Proto::get_x() const
{
	return vector.x();
}

void Vector3Proto::set_y(float y)
{
	vector.set_y(y);
}

float Vector3Proto::get_y() const
{
	return vector.y();
}

void Vector3Proto::set_z(float z)
{
	vector.set_z(z);
}

float Vector3Proto::get_z() const
{
	return vector.z();
}

void Vector3Proto::operator()(size_t idx, float value)
{
	(vector.*setter[idx])(value);
}

float Vector3Proto::operator()(size_t idx) const
{
	return (vector.*vector3proto_getter[idx])();
}

std::array<void(generated::vertex_3d::*)(float), 3> Vector3Proto::setter = {
		&generated::vertex_3d::set_x,
		&generated::vertex_3d::set_y,
		&generated::vertex_3d::set_z
};

void Size3Proto::set_x(float x)
{
	vector.set_x(x);
}
float Size3Proto::get_x() const
{
	return vector.x();
}
void Size3Proto::set_y(float y)
{
	vector.set_y(y);
}
float Size3Proto::get_y() const
{
	return vector.y();
}
void Size3Proto::set_z(float z)
{
	vector.set_z(z);
}
float Size3Proto::get_z() const
{
	return vector.z();
}

void Size3Proto::operator()(size_t idx, float value)
{
	(vector.*setter[idx])(value);
}

float Size3Proto::operator()(size_t idx) const
{
	return (vector.*size3proto_getter[idx])();
}

std::array<void(generated::size_3d::*)(float), 3> Size3Proto::setter = {
		&generated::size_3d::set_x,
		&generated::size_3d::set_y,
		&generated::size_3d::set_z
};

float Vector3ProtoConst::get_x() const
{
	return vector.x();
}

float Vector3ProtoConst::get_y() const
{
	return vector.y();
}

float Vector3ProtoConst::get_z() const
{
	return vector.z();
}

float Vector3ProtoConst::operator()(size_t idx) const
{
	return (vector.*vector3proto_getter[idx])();
}

float Size3ProtoConst::get_x() const
{
	return vector.x();
}

float Size3ProtoConst::get_y() const
{
	return vector.y();
}

float Size3ProtoConst::get_z() const
{
	return vector.z();
}

float Size3ProtoConst::operator()(size_t idx) const
{
	return (vector.*size3proto_getter[idx])();
}

float QuaternionProtoConst::get_x() const
{
	return quaternion.x();
}

float QuaternionProtoConst::get_y() const
{
	return quaternion.y();
}

float QuaternionProtoConst::get_z() const
{
	return quaternion.z();
}

float QuaternionProtoConst::get_w() const
{
	return quaternion.w();
}

float QuaternionProtoConst::operator()(size_t idx) const
{
	return (quaternion.*quaternionProto_getter[idx])();
}
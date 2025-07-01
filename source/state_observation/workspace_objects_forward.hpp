#pragma once

#ifndef STATE_OBSERVATION__WORKSPACE_OBJECTS_FORWARD__HPP
#define STATE_OBSERVATION__WORKSPACE_OBJECTS_FORWARD__HPP

#include "framework.hpp"

#include <memory>

namespace state_observation
{

class aabb;
class obb;
class mesh_wrapper;
class pc_segment;
class object_prototype;
class classification_result;
class object_instance;

typedef std::shared_ptr<pc_segment> segmentPtr;
typedef std::shared_ptr<const pc_segment> ConstPtr;

} // namespace state_observation

#endif // !STATE_OBSERVATION__WORKSPACE_OBJECTS_FORWARD__HPP

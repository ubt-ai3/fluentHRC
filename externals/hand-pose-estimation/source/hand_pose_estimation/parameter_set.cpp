#include "parameter_set.hpp"

namespace hand_pose_estimation
{

parameter_set::parameter_set()
	:
	filename_("config.txt"),
	folder_("assets/hand_config/")
{

}

parameter_set::~parameter_set()
{
}

} // namespace hand_pose_estimation

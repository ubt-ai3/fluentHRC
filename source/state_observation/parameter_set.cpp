#include "parameter_set.hpp"

#include <iostream>
#include <fstream>


namespace state_observation
{

parameter_set::parameter_set()
	:
	filename_("config.txt"),
	folder_("assets/config/")
{

}

parameter_set::~parameter_set()
{
}


} //namespace state_observation

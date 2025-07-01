#pragma once

// Windows headers
#include <windows.h>

#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES // required when windows.h and logging.h from glog are included
#endif
#ifndef GLOG_USE_GLOG_EXPORT
#define GLOG_USE_GLOG_EXPORT
#endif
// resolves compiler errors regarding the macro ERROR

// macro HANDPOSEESTIMATION_EXPORTS defined on command line for this project only
// all other projects will treat it as dllimport
// common header for dll definition and usage
#ifdef HANDPOSEESTIMATION_EXPORTS
#define HANDPOSEESTIMATION_API __declspec(dllexport)
#else
#define HANDPOSEESTIMATION_API __declspec(dllimport)
#endif
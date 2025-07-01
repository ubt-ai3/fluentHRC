#pragma once
// macro HANDPOSEESTIMATION_EXPORTS defined on command line for this project only
// all other projects will treat it as dllimport
// common header for dll definition and usage
#ifdef SIMULATION_EXPORTS
#define SIMULATION_API __declspec(dllexport)
#else
#define SIMULATION_API __declspec(dllimport)
#endif

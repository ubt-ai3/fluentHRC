#pragma once

#include "framework.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/simulation/camera.h>
#include <pcl/simulation/range_likelihood.h>
#include <pcl/simulation/scene.h>
#include <memory>
#include <pcl/pcl_config.h>
#include <pcl/point_types.h>

#include <GL/glew.h>

#ifdef OPENGL_IS_A_FRAMEWORK
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#ifdef GLUT_IS_A_FRAMEWORK
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "scene.hpp"

namespace simulation
{

/**
 * @class model_renderer
 * @brief Renders 3D models in a visualization window
 *
 * Provides functionality to render and visualize 3D models in a PCL visualizer.
 * Supports synchronous updates and display of the simulation environment.
 *
 * Features:
 * - Thread-safe rendering
 * - Real-time visualization
 * - Environment state tracking
 * - Synchronous updates
 * - PCL visualization integration
 */
class SIMULATION_API model_renderer
{
public:
	model_renderer(const environment::Ptr& env);

	void update(std::chrono::duration<float> timestamp);

	void run_sync();
	void show();

private:
	std::mutex m;
	pcl::visualization::PCLVisualizer viewer;
	std::chrono::duration<float> timestamp;
	const environment::Ptr& env;
};

/**
 * @class pc_renderer
 * @brief Generates point clouds from simulated tasks
 *
 * Creates point cloud representations of the simulation environment with
 * optional noise. Must be run in a separate thread from PCLVisualizer.
 *
 * Features:
 * - Point cloud generation
 * - Noise simulation
 * - Camera configuration
 * - Range likelihood computation
 * - OpenGL integration
 * - Thread safety considerations
 */
class SIMULATION_API pc_renderer
{
public:
	pc_renderer(int argc, char** argv,
	            environment::Ptr env,
		int height, int width, bool noise = true);

	environment::Ptr env;
	pcl::simulation::Scene::Ptr scene;
	pcl::simulation::Camera::Ptr camera;
	pcl::simulation::RangeLikelihood::Ptr rl;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr render(std::chrono::duration<float> timestamp);

private:
	std::uint16_t t_gamma[2048];

	// of platter, usually 640x480
	int height;
	int width;
	bool noise;

	void initializeGL(int argc, char** argv);
};

}
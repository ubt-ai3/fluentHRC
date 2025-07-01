#ifndef PCL_SIMULATION_IO_
#define PCL_SIMULATION_IO_

#include <boost/shared_ptr.hpp>

#include <GL/glew.h>

#include <pcl/pcl_config.h>
#ifdef OPENGL_IS_A_FRAMEWORK
# include <OpenGL/gl.h>
# include <OpenGL/glu.h>
#else
# include <GL/gl.h>
# include <GL/glu.h>
#endif
#ifdef GLUT_IS_A_FRAMEWORK
# include <GLUT/glut.h>
#else
# include <GL/glut.h>
#endif

// define the following in order to eliminate the deprecated headers warning
#define VTK_EXCLUDE_STRSTREAM_HEADERS
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/vtk_lib_io.h>


#include <pcl/simulation/camera.h>
#include <pcl/simulation/scene.h>
#include <pcl/simulation/range_likelihood.h>

namespace pcl
{
  namespace simulation
  {
    /**
     * @class SimExample
     * @brief Example simulation class for PCL-based simulation
     *
     * Provides a basic simulation environment with a scene, camera, and range likelihood
     * computation. Includes a simulated arm and table for demonstration purposes.
     *
     * Features:
     * - OpenGL initialization
     * - Scene management
     * - Camera configuration
     * - Range likelihood computation
     * - Simulated arm and table
     * - Time-based simulation
     */
    class PCL_EXPORTS SimExample
    {
      public:
        typedef std::shared_ptr<SimExample> Ptr;
        typedef std::shared_ptr<const SimExample> ConstPtr;
    	
        SimExample (int argc, char** argv,
    		int height,int width);
        void initializeGL (int argc, char** argv);
        
        Scene::Ptr scene_;
        Camera::Ptr camera_;
        RangeLikelihood::Ptr rl_; 
        ::simulation::simulated_arm arm_;
        ::simulation::simulated_table table_;
    
        void doSim (float time);
    
  
      private:
        uint16_t t_gamma[2048];  
    
        // of platter, usually 640x480
        int height_;
        int width_;
    };
  }
}




#endif

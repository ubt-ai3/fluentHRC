/**
 * Demo program for simulation library
 * A virtual camera generates simulated point clouds
 * No visual output, point clouds saved to file
 * 
 * three different demo modes:
 * 0 - static camera, 100 poses
 * 1 - circular camera flying around the scene, 16 poses
 * 2 - camera translates between 2 poses using slerp, 20 poses
 * pcl_sim_terminal_demo 2 ../../../../kmcl/models/table_models/meta_model.ply  
 */

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <boost/shared_ptr.hpp>
#ifdef _WIN32
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#endif

#include "simulation_io.hpp"
#include <pcl\visualization\pcl_visualizer.h>

using namespace Eigen;
using namespace pcl;
using namespace pcl::console;
using namespace pcl::io;
using namespace pcl::simulation;
using namespace std;

SimExample::Ptr simexample;



// Output the simulated output to file:
void write_sim_output(string fname_root){ 
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZRGB>);

    // Read Color Buffer from the GPU before creating PointCloud:
    // By default the buffers are not read back from the GPU
    simexample->rl_->getColorBuffer ();
    simexample->rl_->getDepthBuffer ();  
    // Add noise directly to the CPU depth buffer 
    simexample->rl_->addNoise ();


    simexample->rl_->getPointCloud (pc_out,false,simexample->camera_->getPose ());

 
    
    pcl::PCDWriter writer;

    writer.writeBinary (  string (fname_root + ".pcd")  , *pc_out);


}


int
main (int argc, char** argv)
{
    std::mutex m;
 
  pcl::visualization::PCLVisualizer viewer;

  // 2 Construct the simulation method:
  int width = 640;
  int height = 480;
  
 
    std::stringstream ss;
    ss.precision(20);
    ss << "simcloud";// << ".pcd";

    

  

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_out(new pcl::PointCloud<pcl::PointXYZRGB>);
  
  std::thread t([&]() {
      auto simexample = SimExample::Ptr(new SimExample(argc, argv, height, width));
      float time = 0;

      while (!viewer.wasStopped())
      {
          simexample->doSim(time);
          

          // Read Color Buffer from the GPU before creating PointCloud:
// By default the buffers are not read back from the GPU
          simexample->rl_->getColorBuffer();
          simexample->rl_->getDepthBuffer();
          // Add noise directly to the CPU depth buffer 
          simexample->rl_->addNoise();

          {
              std::lock_guard<std::mutex> lock(m);
              simexample->rl_->getPointCloud(pc_out, false, simexample->camera_->getPose());
          }

          pcl_sleep(0.0333);
          time += 0.0333;
            }
      });

  viewer.addCoordinateSystem();
  viewer.setCameraPosition(0, 0, -1, 1, 0, 0);
  //test.arm_.render(viewer, 0);
  while (!viewer.wasStopped())
  {
      {
          std::lock_guard<std::mutex> lock(m);


          //write_sim_output(ss.str());


          if (!viewer.updatePointCloud(pc_out))
              viewer.addPointCloud(pc_out);


          viewer.spinOnce();
      }
      pcl_sleep(0.033);
  }

  if(t.joinable())
    t.join();

  return 0;
}

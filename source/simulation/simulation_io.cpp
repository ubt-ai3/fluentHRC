#include "simulation_io.hpp"
#include <pcl/io/png_io.h>

#include "scene.hpp"

pcl::simulation::SimExample::SimExample(int argc, char** argv,
	int height,int width):
        height_(height), width_(width),
    arm_(Eigen::Vector3f(0.2, 0.2, 0.2), Eigen::Vector3f(0., 0., 0.1))
{

  initializeGL (argc, argv);
  
  // 1. construct member elements:
  camera_ = Camera::Ptr (new Camera ());
  scene_ = Scene::Ptr (new Scene ());

   rl_ = RangeLikelihood::Ptr (new RangeLikelihood (1, 1, height, width, scene_));

  // Actually corresponds to default parameters:
  rl_->setCameraIntrinsicsParameters (width_,height_, 576.09757860,
            576.09757860, 321.06398107, 242.97676897);
  rl_->setComputeOnCPU (false);
  rl_->setSumOnCPU (true);
  rl_->setUseColor (true);
   
  camera_->set(0, 0, 2, 0, 1.5, 0);
  
  for (int i=0; i<2048; i++)
  {
    float v = i/2048.0;
    v = powf(v, 3)* 6;
    t_gamma[i] = v*6*256;
  }  


  arm_.move(Eigen::Vector3f(-0.2, 0.2, 0.1), 0);
}




void 
pcl::simulation::SimExample::initializeGL (int argc, char** argv)
{
  glutInit (&argc, argv);
  glutInitDisplayMode (GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);// was GLUT_RGBA
  glutInitWindowPosition (10, 10);
  glutInitWindowSize (10, 10);
  //glutInitWindowSize (window_width_, window_height_);
  glutCreateWindow ("OpenGL range likelihood");

  GLenum err = glewInit ();
  if (GLEW_OK != err)
  {
    std::cerr << "Error: " << glewGetErrorString (err) << std::endl;
    exit (-1);
  }

  std::cout << "Status: Using GLEW " << glewGetString (GLEW_VERSION) << std::endl;
  if (glewIsSupported ("GL_VERSION_2_0"))
    std::cout << "OpenGL 2.0 supported" << std::endl;
  else
  {
    std::cerr << "Error: OpenGL 2.0 not supported" << std::endl;
    exit(1);
  }
  
  std::cout << "GL_MAX_VIEWPORTS: " << GL_MAX_VIEWPORTS << std::endl;
  const GLubyte* version = glGetString (GL_VERSION);
  std::cout << "OpenGL Version: " << version << std::endl;  
}



void
pcl::simulation::SimExample::doSim (float time)
{
    scene_->clear();
    arm_.render(*scene_, time);
    table_.render(*scene_, time);

  // No reference image - but this is kept for compatibility with range_test_v2:
  float* reference = new float[rl_->getRowHeight() * rl_->getColWidth()];
  const float* depth_buffer = rl_->getDepthBuffer();
  // Copy one image from our last as a reference.
  for (int i=0, n=0; i<rl_->getRowHeight(); ++i)
  {
    for (int j=0; j<rl_->getColWidth(); ++j)
    {
      reference[n++] = depth_buffer[i*rl_->getWidth() + j];
    }
  }

  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d> > poses;
  std::vector<float> scores;
  poses.push_back (camera_->getPose());
  rl_->computeLikelihoods (reference, poses, scores);
  //std::cout << "camera: " << camera_->getX ()
  //     << " " << camera_->getY ()
  //     << " " << camera_->getZ ()
  //     << " " << camera_->getRoll ()
  //     << " " << camera_->getPitch ()
  //     << " " << camera_->getYaw ()
  //     << std::endl;
       
  delete [] reference;
}







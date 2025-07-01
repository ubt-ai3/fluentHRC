#include "rendering.hpp"

#include <ranges>

namespace simulation
{

model_renderer::model_renderer(const environment::Ptr& env)
	:
	viewer("Virtual Scene"),
    timestamp(0),
    env(env)
{
    viewer.addCoordinateSystem();
    viewer.setCameraPosition(0.0, 0.0, 2.0, 0.0, 1.0, 0.0);
}

void model_renderer::update(std::chrono::duration<float> timestamp)
{
    this->timestamp = timestamp;

    if (env->object_traces.empty() && env->additional_scene_objects.empty())
        return;

    std::unique_lock<std::mutex> l(m);

    for (const auto& obj : env->object_traces | std::views::values)
        obj->render(viewer, timestamp);

    for (const scene_object::Ptr& obj : env->additional_scene_objects)
        obj->render(viewer, timestamp);
}

void model_renderer::run_sync()
{
    while (!viewer.wasStopped())
    {
        std::unique_lock<std::mutex> l(m);
        viewer.spinOnce();
    }
}

void model_renderer::show()
{
    std::unique_lock<std::mutex> l(m);
    viewer.spinOnce();
}

pc_renderer::pc_renderer(int argc, char** argv,
                         environment::Ptr env,
    int height, int width, bool noise)
    :
    env(std::move(env)),
    height(height),
    width(width),
    noise(noise)
{
    initializeGL(argc, argv);

    camera = std::make_shared<pcl::simulation::Camera>();
    scene = std::make_shared<pcl::simulation::Scene>();

    rl = std::make_shared<pcl::simulation::RangeLikelihood>(1, 1, height, width, scene);

    // Actually corresponds to default parameters:
    rl->setCameraIntrinsicsParameters(
        width, height, 576.09757860f, 576.09757860f, 321.06398107f, 242.97676897f);
    rl->setComputeOnCPU(false);
    rl->setSumOnCPU(true);
    rl->setUseColor(true);


    camera->set(0, 0, 2, 0, 1.5, -1.5);

    for (int i = 0; i < 2048; i++) {
        float v = i / 2048.0;
        v = powf(v, 3) * 6;
        t_gamma[i] = v * 6 * 256;
    }
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_renderer::render(std::chrono::duration<float> timestamp)
{
    scene->clear();
    for (const auto& object : env->object_traces | std::views::values)
        object->render(*scene, timestamp);

    for (const scene_object::Ptr& obj : env->additional_scene_objects)
        obj->render(*scene, timestamp);


    // No reference image - but this is kept for compatibility with range_test_v2:
    std::vector<float> reference;
    reference.resize(static_cast<size_t>(rl->getRowHeight()) * rl->getColWidth());
    std::memcpy(reference.data(), rl->getDepthBuffer(), sizeof(float) * reference.size());
    
    std::vector<float> scores;
    rl->computeLikelihoods(reference.data(), { camera->getPose() }, scores);


    // Read Color Buffer from the GPU before creating PointCloud:
	// By default the buffers are not read back from the GPU

    //rl->getColorBuffer();

    //rl->getDepthBuffer();

    if(noise)
        rl->addNoise();

    auto pc = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    std::cout.setstate(std::ios_base::failbit); //hacky workaround for cout spam in pcl library
    rl->getPointCloud(pc, false, camera->getPose(), true);
    std::cout.clear();

    //pcl::transformPointCloud(*pc, *pc, Eigen::Affine3d(camera->getPose().affine()).inverse() * Eigen::Translation3d(0, 0, -0.009));
    pc->header.stamp = std::chrono::duration_cast<std::chrono::microseconds>(timestamp).count();
    pc->sensor_orientation_ = Eigen::Quaternionf::Identity();
    pc->sensor_origin_ = Eigen::Vector4f(0.f, 0.f, 0.f, 1.f);
    
    return pc;
}

void pc_renderer::initializeGL(int argc, char** argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB); // was GLUT_RGBA
    glutInitWindowPosition(10, 10);
    glutInitWindowSize(10, 10);
    // glutInitWindowSize (window_width_, window_height_);
    glutCreateWindow("OpenGL range likelihood");

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error: " << glewGetErrorString(err) << std::endl;
        std::exit(-1);
    }

    std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
    if (glewIsSupported("GL_VERSION_2_0"))
        std::cout << "OpenGL 2.0 supported" << std::endl;
    else {
        std::cerr << "Error: OpenGL 2.0 not supported" << std::endl;
        std::exit(1);
    }

    std::cout << "GL_MAX_VIEWPORTS: " << GL_MAX_VIEWPORTS << std::endl;
    const GLubyte* version = glGetString(GL_VERSION);
    std::cout << "OpenGL Version: " << version << std::endl;
}

}

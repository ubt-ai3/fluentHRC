#include "baraglia17.hpp"
#include "building.hpp"
#include "mogaze.hpp"
#include "rendering.hpp"

using namespace simulation;
using namespace state_observation;

int main(int argc, char** argv)
{
    building_simulation_test sth;
    mogaze::task_execution execution({}, 2);

    std::mutex m;

    pcl::visualization::PCLVisualizer viewer;


    double fps = 30;
    auto duration = std::chrono::milliseconds(static_cast<int>(1000 / fps));
    pn_transition::Ptr transition;


    std::thread t([&]() {
        std::cout << "Press enter to execute one action." << std::endl;

        do
        {
            std::string s;
            std::getline(std::cin, s, '\n');

            std::lock_guard<std::mutex> lock(m);
            transition = execution.next();
        } while (!viewer.wasStopped() && transition);
    });



    auto rendered_state = transition;
    viewer.addCoordinateSystem();
    viewer.setCameraPosition(4, 2, 4,-1, 0, 2);
    while (!viewer.wasStopped())
    {
        auto frame_start = std::chrono::high_resolution_clock::now();

        if(rendered_state != transition)
        {
            std::lock_guard<std::mutex> lock(m);
            execution.render(viewer);
            rendered_state = transition;
        }

        try {
            viewer.spinOnce();
        }
        catch (const std::exception&)
        {
        }

        std::this_thread::sleep_until(frame_start + duration);
    }
    

    if (t.joinable())
        t.join();
}
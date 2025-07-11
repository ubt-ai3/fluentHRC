[![DOI](https://zenodo.org/badge/1011842209.svg)](https://doi.org/10.5281/zenodo.15783113) 

Fluency in Dynamic Human-Robot Teaming with Intention Prediction - Main Application
===================================================================================

[![Video Preview](https://raw.githubusercontent.com/NiHoel/fluentHRC/main/WISEL_promo.webp)](https://resy-server.inf.uni-bayreuth.de/resypub/files/hllerich2024a.Fluency.in.Dynamic.HumanRobot.Teaming.with.Intention.Prediction.mp4)

# TODO
* Update link to Franka Proxy in line 45 of overlay-ports\franka-proxy\portfile.cmake and in the README

# Getting Started
1. Clone Repository to a short path (< 30 characters including repository folder): `git clone https://resy-gitlab.inf.uni-bayreuth.de/flexcobot/core.git`.
2. Clone Sub-Repositories `git submodule update --init --recursive`.
3. Download and install Kinect SDK v2 from: [https://www.microsoft.com/en-us/download/details.aspx?id=44561](https://www.microsoft.com/en-us/download/details.aspx?id=44561)
4. Install Visual Studio: select the "Desktop development with C++" workload and additionally install:
    * .NET Framework SDK 4.7.2
    * .NET Framework SDK 4.7.2 targeting pack
5. Download and install CUDA 12.8.1 from: https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
6. Download CUDNN (Version: tarball): https://developer.nvidia.com/cudnn
7. Unzip into the installation directory of CUDA (%CUDA_PATH%), overwrite existing files if asked.
8. Open Visual Studio (CMake Cache generation should trigger. Otherwise right click on "CMakeLists.txt" in Solution Explorer and select "Generate Cache")
9. Wait for cmake to finish (may take some time)
10. Download the CNN models for hand tracking [here](https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/) and copy `merged_net.prototxt` and `merged_snapshot_iter_300000.caffemodel` from `GanHandsAPI.zip/data/CNNClassifier/rgb-crop_232_FINAL_synth+GAN_ProjLayer` to `externals/hand-pose-estimation/assets/network_models`
11. Open solution file in build/default, compile the program and run it.

# Related Resources
* (Video)[https://resy-server.inf.uni-bayreuth.de/resypub/files/hllerich2024a.Fluency.in.Dynamic.HumanRobot.Teaming.with.Intention.Prediction.mp4]
* (HoloLens Application)[https://github.com/Olli1080/ar_integration/tree/5.4.3]
* (Franka Proxy)[] # TODO
* (Data and Plots)[https://codeocean.com/capsule/2820731/tree]

When using this code or related resources please mention the following paper in your work
```bibtex
@PhdThesis{Hoellerich25,
  author       = {Höllerich, Nico},
  school       = {Universit{\"a}t Bayreuth},
  title        = {Fluency in Dynamic Human-Robot Teaming with Intention Prediction},
  year         = {2025},
  address      = {Bayreuth},
  type         = {Dissertation},
  journal      = {Dissertation},
}
```
For the hardware setup, see Appendix B.1 in the dissertation. To calibrate Kinect 2 and the robot:
1. Prepare the mat (`assets\mat2.pdf`) and objects.
2. run `app_visualization` and press `l`.
3. Repeatedly press the suction cup onto the highlighted blocks and press `l` (see at the end of `source/app_visualization/module_manager.cpp`).

# Code Structure Overview

The main source code is located in the `source/` directory, organized into the following modules:

## Core Modules

### state_observation/
Core module for state estimation and tracking:
- **Petri Net Modeling**: Implements Petri net-based state representation and reasoning
- **Object Tracking**: Handles object detection, tracking, and classification
- **Building Estimation**: Manages spatial relationships and building structures
- **World Traceability**: Links physical objects to Petri net tokens
- **Reasoning**: Provides transition extraction and optimization

### intention_prediction/
Predicts human intentions and actions:
- **Agent Management**: Tracks and manages observed agents
- **Intention Prediction**: Predicts future actions based on current state
- **Agent Observation**: Monitors and analyzes agent behavior

### simulation/
Provides simulation capabilities:
- **Scene Management**: Handles virtual environment setup
- **Task Simulation**: Simulates task execution
- **Behavior Testing**: Tests various behaviors and scenarios
- **Franka Simulation**: Simulates Franka robot behavior

## Robot Control Modules

### franka_high_level/
High-level robot control:
- **Motion Control**: Manages robot movement and trajectories
- **Task Execution**: Handles high-level task commands
- **State Management**: Tracks robot state and configuration

### franka_planning/
Robot planning and decision making:
- **Motion Planning**: Generates robot motion plans
- **Task Planning**: Plans sequences of robot actions
- **Agent Logic**: Implements robot decision-making

### franka_voxel/
Motion controller to sample poses for vizualization as voxel grid.


## Visualization and Interface Modules

### app_visualization/
Visualization and user interface:
- **Task Progress**: Visualizes task execution progress
- **Intention Display**: Shows predicted intentions
- **Robot State**: Displays robot state and actions
- **Petri Net Rendering**: Visualizes Petri net states
- **Module Management**: Coordinates visualization components

### grpc_server/
Network communication:
- **Server Implementation**: Handles gRPC server setup
- **Service Logic**: Implements remote procedure calls
- **Point Cloud Processing**: Processes point cloud data
- **HoloLens Integration**: Manages HoloLens communication

## Support Modules

### kinect2_grabber/
Kinect v2 integration:
- **Data Acquisition**: Captures Kinect sensor data
- **Point Cloud Processing**: Processes depth and color data
- **Calibration**: Handles sensor calibration

### csv_reader/
Data import and processing:
- **Data Import**: Reads CSV data files
- **Tracking Data**: Processes tracking information
- **Data Validation**: Validates imported data

### sample_data_registration/
Sample data handling:
- **Data Registration**: Registers sample data
- **Evaluation**: Evaluates data quality
- **Hand/Object Analysis**: Analyzes hand and object data

## Module Interactions

The system's architecture follows a layered approach:

1. **Sensor Layer**
   - `kinect2_grabber` provides raw sensor data
   - Data flows to `state_observation` for processing

2. **State Layer**
   - `state_observation` processes sensor data
   - Updates Petri net models and object states
   - Feeds information to `intention_prediction`

3. **Prediction Layer**
   - `intention_prediction` analyzes current state
   - Generates predictions for human actions
   - Informs `franka_planning` for robot responses

4. **Control Layer**
   - `franka_planning` generates robot plans
   - `franka_high_level` executes plans
   - `franka_voxel` handles motion details

5. **Visualization Layer**
   - `app_visualization` displays system state
   - `grpc_server` provides remote visualization
   - Integrates with HoloLens for AR display

6. **Simulation Layer**
   - `simulation` provides virtual environment
   - Supports testing and development for task state tracking
   - Can replace sensor input for testing

## Additional Directories

- **externals/**: Third-party libraries and dependencies
- **assets/**: Configuration files, models, and mesh data
- **unit_tests/**: Unit tests for dedicate algorithms (task state tracking, neighbrourhood calculation, occlusion detection)

## Description of the Code in the Dissertation
Parameters mentioned throught the paper are stored as `.xml` files inside the `assets` folder.
* Section 4: `externals/hand-pose-estimation/source/hand_pose_estimation`
   * Section 4.3: `bounding_box_tracking` (Equations 4.1 and 4.2), `ra_point_cloud_classifier`, `ra_skin_color_classifier`, and `skin_detection` 
   * Section 4.4: `hand_model` and `hand_pose_estimation`
   * Section 4.5: `hand_tracker` 
* Section 5: `source/state_observation`
   * Section 5.2: `classification_handler`
   * Section 5.3: `building_estimation` (constructs Petri net), `pn_model`, and `pn_model_extension`
   * Section 5.4: `classification_handler` and `pn_reasoning` in `source/state_observation`; `task_progress_visualizer` in `source/app_visualization`
   * Section 5.5: `get_feasible_actions` in `source/franka_planning/robot_agent`
   * Section 5.6: `module_manager` in `source/app_visualization`
   * Section 5.7: all files in `source/simulation`
* Section 6: `source/intention_prediction`
   * Section 6.2: `apply_general_prediction_rules` in `observed_agent`
   * Section 6.3: `transition_context` in `observed_agent`
   * Section 6.4: `assets/prediction`
   * Section 6.5: `agent_manager` and remaining methods in `observed_agent`
* Section 7.2: `robot_agent` (prediction and scheduling) and `franka_actor` (motion execution) - both in `source/franka_planning`

# Troubleshooting
* If the CMake cache generation does not trigger in step 8, open the Solution Explorere, right click on the CMakeLists.txt and select Configure CMake. See for further information: https://learn.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170
* If vcpkg fails to build because it cannot download dependencies from servers, retry with an newer commit (though this can result in new errors due to incompatible package versions)

# Contact
Nico Höllerich (nico.hoellerich@uni-bayreuth.de)

# License
* The code in this repository is licensed under the [MIT License](LICENSE.txt) except for the files `externals\hand-pose-estimation\source\hand_pose_estimation` prefixed with `ra_` these are licensed under a [derivateive of MIT for non-commercial usage](LICENSE-skin-segmentation.txt).
* An overview of the third-party libraries used in this project can be found in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
* Detailled copyright and licensing information for each third-party library `lib` can be found in `externals/vcpkg/packages/{lib}_x64-windows/share` after step 8 of the Getting Started section.

# Contributors
* Axel Wimmer (building_estimation)
* R. A. (skin_color_detector)
* Oliver Zahn (franka_high_level, franka_planning, franka_voxel, grpc_server, interface for HoloLens 2, rework of build system)
* Nico Höllerich (all other modules)
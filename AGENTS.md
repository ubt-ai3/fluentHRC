# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Summary

FlexCobot is a human-robot collaboration system with Petri net-based state tracking. Key compile-time configuration switches in `source/app_visualization/module_manager.hpp`:

- **`USE_HOLOLENS`** (lines 9,28,36,43,53,154,217,225,731,742,756,760,788) - Enables/disables HoloLens AR integration and gRPC server
- **`USE_ROBOT`** (lines 10,43,85,150,180,193,728,756,776,830,849,885,930,942,947,1076,1092,1124,1195) - Enables/disables Franka robot control and planning  
- **Camera types** (`module_manager.cpp:1214`): `SIMULATION`, `KINECT_V2`, `REALSENSE`

Main executable: `app_visualization.exe`. System processes Kinect sensor data through Petri nets for task state tracking, predicts human intentions, and plans robot actions. Supports simulation mode, HoloLens AR visualization, and real robot control.

## Project Overview

FlexCobot is a human-robot collaboration (HRC) system that uses Petri nets for task state tracking, intention prediction, and robot planning. The system integrates Kinect v2 sensor data, hand pose estimation, object tracking, and Franka robot control with HoloLens AR visualization via gRPC.

## Build and Development Commands

### Initial Setup
```bash
# Clone with submodules
git clone https://resy-gitlab.inf.uni-bayreuth.de/flexcobot/core.git
git submodule update --init --recursive

# Generate CMake cache (in Visual Studio)
# Right-click CMakeLists.txt → "Generate Cache"
# Or use CMake preset:
cmake --preset default
```

### Building
```bash
# Build using Visual Studio solution
# Open build/default/*.sln and build in Visual Studio

# Or build via CMake preset
cmake --build --preset default
```

### Running
The main executable is `app_visualization.exe`, located in:
- `build/default/source/app_visualization/Debug/app_visualization.exe` (Debug)
- `build/default/source/app_visualization/Release/app_visualization.exe` (Release)

Other executables:
- `app_evaluation_net.exe` - Network evaluation utilities
- `app_util.exe` - Utility functions
- `sample_data_registration.exe` - Sample data handling

### Testing
Unit tests are built as a shared library in `source/unit_tests/`:
- `box_overlap.cpp` - Spatial overlap detection tests
- `pn_differ_test.cpp` - Petri net difference calculation tests
- `pn_reasoning_test.cpp` - Petri net reasoning tests
- `prediction_neighborhood.cpp` - Neighborhood calculation tests

## Architecture

### Core Data Flow

1. **Sensor Input** → `kinect2_grabber` captures depth/color data
2. **State Tracking** → `state_observation` processes sensor data, updates Petri net models, tracks objects
3. **Intention Prediction** → `intention_prediction` analyzes current state to predict human actions
4. **Robot Planning** → `franka_planning` generates robot plans based on predictions
5. **Robot Control** → `franka_high_level` executes motion plans
6. **Visualization** → `app_visualization` renders system state; `grpc_server` provides remote AR visualization

### Key Modules

#### state_observation/
The core state estimation module implementing Petri net-based task modeling:

- **Petri Net Implementation**:
  - `pn_model.hpp/cpp` - Core Petri net data structures and operations
  - `pn_model_extension.hpp/cpp` - Extended Petri net functionality
  - `pn_reasoning.hpp/cpp` - Transition extraction and optimization
  - `pn_world_traceability.hpp/cpp` - Links physical objects to Petri net tokens

- **Object Tracking**:
  - `object_detection.hpp/cpp` - Detects objects in point clouds
  - `object_tracking.hpp/cpp` - Tracks objects over time
  - `object_prototype_loader.hpp/cpp` - Loads object models from assets

- **Spatial Reasoning**:
  - `building_estimation.hpp/cpp` - Manages spatial relationships and building structures
  - `workspace_objects.hpp/cpp` - Workspace object management
  - `workspace_calibration.h/cpp` - Sensor calibration

- **Selection Management**:
  - `selection_registry.hpp/cpp` - Manages unique IDs for selected objects (resource pool instances)
  - Enables HoloLens to select blocks and transmit selection IDs back to server
  - Links user selections to Petri net logic

#### grpc_server/
Implements gRPC-based communication with HoloLens client:

- **Proto Definitions** (in `assets/grpc/proto/`):
  - `services.proto` - Main gRPC service definitions
  - `object.proto` - Object data structures
  - `hand_tracking.proto` - Hand tracking data
  - `robot.proto` - Robot state and commands
  - `selection.proto` - Selection/interaction data
  - `debug.proto` - Debug information
  - `meta_data.proto` - Metadata structures

- **Server Implementation**:
  - `service_impl.h/cpp` - gRPC service implementation
  - `server_module.h/cpp` - Server lifecycle management
  - `point_cloud_processing.h/cpp` - Point cloud processing for transmission
  - `proto_plugin.h/cpp` - Protocol buffer utilities

Proto files are compiled via CMake using `protobuf_generate()` with gRPC plugin. Generated files are placed in `assets/grpc/generated/`.

#### intention_prediction/
Predicts human intentions based on observed behavior:

- `agent_manager.hpp/cpp` - Manages tracked agents
- `observed_agent.hpp/cpp` - Individual agent state and behavior tracking

#### franka_planning/
Robot planning and scheduling system with a three-layer architecture:

**Layer 1: Agent Orchestration** (`robot_agent.hpp/cpp`)
- `robot::agent` class runs in dedicated thread executing `update()` loop
- Manages current Petri net marking (system state)
- Holds pluggable `action_planner` instance (defines robot behavior strategy)
- Queries planner for next action via `behaviour->next(franka, failed_action)`
- Executes actions through `franka_agent` and updates internal marking on success
- Maintains `next_action` queue and `failed_action` exclusion list
- Location: source/franka_planning/robot_agent.cpp:1172-1314

**Layer 2: Action Planners** (scheduling strategies)
Multiple planner implementations inherit from `action_planner` base class:

- **`null_planner`**: Robot remains passive, no actions selected
- **`planner_layerwise_rtl`** (Right-to-Left):
  - Builds structures layer-wise from right to left (from user perspective)
  - Creates sorted `place_order` list of all placement actions
  - Sorting criteria: Z-height (bottom first), then Y, then X coordinates
  - Considers gripper collisions: if block A blocks access to B, B is placed first
  - `select_action()` (line 351): Iterates through `place_order`, selects first executable action or finds enabling pick action

- **`planner_adaptive`** (Cooperative):
  - Selects actions the human is **least likely** to execute next (minimizes interference)
  - Uses `intention_prediction/agent_manager` for human action prediction
  - `predict_consecutive()` (line 551): 2-step lookahead to compute action probabilities
  - Chooses action with **minimum probability** from human predictions

- **`planner_adversarial`** (Competitive):
  - Inherits from `planner_adaptive` but selects **most likely** human actions
  - Used for competitive scenarios or testing

**Layer 3: Action Execution** (`franka_actor.hpp/cpp`)
- `franka_agent` class handles physical robot motion
- `execute_transition()` (line 51-80): Dispatches action types:
  - `stack_action` → `place(action->to.first->box)`
  - `pick_action` → `pick(action->from->box)`
  - `place_action` → `place(action->to->box)`
  - `reverse_stack_action` → `pick(action->from.first->box)`
- Pick/place implementation (lines 389-643):
  - Computes gripper pose from target OBB (Oriented Bounding Box)
  - Trajectory: hover → approach → grasp/release → return to hover
  - Uses inverse kinematics (IK) for joint angle computation
  - Vacuum gripper control for object manipulation

**Action Data Structures** (in `state_observation/pn_model_extension.hpp`):
- **`pn_boxed_place`** (line 79): Represents 3D spatial location
  - Contains `obb box` field with `translation` (3D position), `diagonal` (dimensions), `rotation`
  - Tracks overlapping places for collision detection
- **`place_action`** (line 155): Transition for placing object at location
  - Has `from` (source place) and `to` (target `pn_boxed_place`)
- **`stack_action`** (line 194): Transition for stacking objects
  - Has `to` field of type `pn_object_instance` (pair of `pn_boxed_place::Ptr` and `pn_object_token::Ptr`)
- **`pick_action`** (line 118): Transition for picking object
  - Has `from` (`pn_boxed_place`) specifying pick location

**Creating Custom Planners**:
To implement runtime rescheduling with custom placement logic:
1. Create new class inheriting from `action_planner` in `source/franka_planning/robot_agent.hpp`
2. Override `select_action(const std::vector<pn_transition::Ptr>& feasible_actions, const franka_agent&)`:
   - Access target positions via `get_target_place(transition)->box.translation`
   - Filter/sort `feasible_actions` based on desired criteria
   - Return `std::pair<pn_transition::Ptr, pn_transition::Ptr>` (current action, next action)
3. Optionally override `compute_forward_transitions()` to filter available actions
4. Instantiate planner and set via `agent::update_behaviour(std::make_unique<my_planner>(...))`
   - Typically done in `app_visualization/module_manager.cpp` where agent is initialized

**Key Insight**: Target positions for placement are encoded in the Petri net transitions themselves (via `pn_boxed_place` OBB), not computed at runtime. The planner's role is to **select** which transition to execute from the available set, not to generate new positions.

#### franka_high_level/
Higher-level robot control abstractions and state management

#### franka_voxel/
Motion controller for sampling poses and visualizing as voxel grids (used with GPU-Voxels library).

#### app_visualization/
Main visualization application:

- `module_manager.hpp/cpp` - Coordinates all visualization components
- `viewer.hpp/cpp` - Main rendering window
- `petri_net_rendering.hpp/cpp` - Visualizes Petri net states
- `task_progress_visualizer.hpp/cpp` - Shows task execution progress
- `intention_visualizer.hpp/cpp` - Displays predicted intentions
- `franka_visualization.hpp/cpp` - Robot state visualization
- `franka_visualization_gpu.cu/h` - CUDA-accelerated visualization

#### simulation/
Virtual environment for testing without hardware:

- Simulates sensor input, task execution, and robot behavior
- Replaces physical Kinect/robot with simulated equivalents

## Important Implementation Details

### Petri Net System
The Petri net implementation is central to the system's task state tracking:
- Places represent states (e.g., "block at position A")
- Transitions represent actions (e.g., "move block from A to B")
- Tokens represent physical objects tracked in the world
- `pn_world_traceability` maintains bidirectional mapping between Petri net tokens and physical objects

### Selection Registry & HoloLens Integration
Recent work (see git commits) implements a selection system:
- `selection_registry` assigns unique IDs (`pn_ids`) to resource pool instances
- HoloLens users can select objects in AR; selections are sent via gRPC with their IDs
- Server uses IDs to identify which Petri net elements correspond to user selections
- This links UI interactions to underlying Petri net logic

### Asset Management
Assets are linked (not copied) to build output using junction points (Windows) or symlinks (Linux):
- `LinkAssets()` macro in CMakeLists.txt creates junctions/symlinks
- Assets include: config files, 3D models, object meshes, network models, shaders
- Hand pose estimation assets from `externals/hand-pose-estimation/assets/`

### CMake Macros
- `LinkAssets(target)` - Creates asset junctions/symlinks for target
- `CopyRuntimeDeps(target)` - Copies required DLLs to output directory
- `FixExecutionContext(target)` - Sets working directory for debugging

### Dependencies
Key dependencies (managed via vcpkg):
- **Point Cloud Library (PCL)** - Point cloud processing
- **VTK** - Visualization Toolkit
- **OpenCV** - Computer vision
- **Caffe** - Neural network inference (with CUDA)
- **gRPC + Protobuf** - Network communication
- **Boost** - Various utilities (serialization, signals, timers)
- **Eigen3** - Linear algebra
- **OGDF** - Graph algorithms (for Petri net layout)
- **GPU-Voxels** - Voxel-based collision detection
- **Cairo/CairoMM** - 2D rendering
- **Kinect SDK v2** - Sensor integration
- **Franka libraries** - Robot control

### CUDA Support
The project uses CUDA for:
- Neural network inference (Caffe)
- GPU-accelerated visualization (`franka_visualization_gpu.cu`)
- Voxel processing (GPU-Voxels)

CUDA flags are set in root `CMakeLists.txt`:
```cmake
CMAKE_CUDA_FLAGS: --expt-extended-lambda --expt-relaxed-constexpr
```

## File Organization

### Source Code Structure
- `source/` - All C++ source modules (each is a CMake subdirectory)
- `externals/` - Third-party dependencies and submodules
  - `hand-pose-estimation/` - Hand tracking submodule
  - `KinectGrabber/` - Kinect integration
  - `vcpkg/` - Package manager
- `assets/` - Runtime assets (models, configs, shaders, proto files)
  - `assets/grpc/proto/` - gRPC protocol definitions
  - `assets/grpc/generated/` - Generated protobuf/gRPC code
  - `assets/config/` - Configuration files
  - `assets/models/` - 3D models
  - `assets/object_meshes/` - Object mesh files
- `build/` - Build output directory (CMake default preset)
- `overlay-ports/` - Custom vcpkg port overlays

### Configuration Files
- `CMakeLists.txt` - Root build configuration
- `CMakePresets.json` - CMake preset configuration (Visual Studio 2022)
- `vcpkg.json` - Dependency manifest
- `vcpkg-configuration.json` - vcpkg settings

## Development Notes

### Path Length Limitation
Clone to a short path (< 30 characters including repository folder) due to Windows path length limitations with vcpkg.

### Visual Studio Setup
Required workload: "Desktop development with C++"
Additional components:
- .NET Framework SDK 4.7.2
- .NET Framework SDK 4.7.2 targeting pack

### CUDA/CUDNN Setup
- CUDA 12.8.1 required
- CUDNN must be unzipped into CUDA installation directory (%CUDA_PATH%)

### Working with Proto Files
To regenerate gRPC/protobuf code:
1. Modify `.proto` files in `assets/grpc/proto/`
2. Rebuild the `proto-objects` target (happens automatically on full build)
3. Generated files appear in `assets/grpc/generated/`

Alternatively, use the manual build script:
```bash
cd assets/grpc
python build_proto_x64.py
```

### Recent Development Focus (Per Git History)
- Integration of `pn_ids` for unique identification of Petri net objects
- Implementation of `selection_registry` to manage resource pool instances
- gRPC server-side mesh selection from HoloLens
- Transmission of selection IDs between HoloLens and server
- Linking Petri net logic to user selections

## Common Issues

### CMake Cache Generation
If CMake cache doesn't auto-generate in Visual Studio:
- Right-click `CMakeLists.txt` in Solution Explorer
- Select "Configure CMake" or "Generate Cache"

### vcpkg Download Failures
If vcpkg fails to download dependencies:
- Retry with a newer vcpkg commit (may cause compatibility issues)
- Check network connectivity
- Verify proxy settings if behind corporate firewall

### Missing DLLs at Runtime
If executable fails with missing DLL errors:
- Ensure `CopyRuntimeDeps()` macro is called for the target
- Check that Visual Studio debugger working directory is set correctly
- Verify CUDA/CUDNN DLLs are in system PATH or copied to output directory

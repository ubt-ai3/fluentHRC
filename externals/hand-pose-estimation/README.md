# Hand Tracker

Plugin-project to detect and track multiple hands. Outputs a skeleton model.

## Features
* No skeleton tracking required.
* Simultaneous detection of multiple hands supported.
* Realtime

## Demo 
The branch demo contains a demo application. To run it:
1. Download and install Kinect SDK v2 from: [https://www.microsoft.com/en-us/download/details.aspx?id=44561](https://www.microsoft.com/en-us/download/details.aspx?id=44561)
2. Download and install CUDA from: https://developer.nvidia.com/cuda-downloads
3. Download and install Ensenso SDK from: https://www.ensenso.com/support/sdk-download/
4. Check out the branch 'demo' and recurisvely checkout all submodules
5. Go to tools/vc19, open hand_pose_estimation.sln and compile the demo project
6. Run the project, connect a Kinect v2 (depth and color stream required) and see the result 

## Usage
### Project structure
The main project must have the following structure and folder names:

* assets
* build <- place where the compiled library files go
* externals
  * hand_pose_estimation [place where this repository is added as a submodule]
* tools
   * vcXX [Visual Studio version]
      * vs-props [Visual Studio property sheets from https://resy-gitlab.inf.uni-bayreuth.de/flexcobot/vs-props.git]
      * XX.sln [Visual Studio solution file]
    
### Import steps
1. Clone repositories into specified locations.
2. Add property sheets to your projects: View -> Other Windows -> Property Manager -> Right click project -> Add existing property sheet -> select vs-props/build_common.props (ensures that output is copied to the correct place)
3. Import plugin-project: Right click solution -> Add -> Existing project -> Select externals/hand_pose_estimation/tools/vc19/projects/hand_pose_estimation.vcxproj
4. Register build library: Right click main project -> Properties -> Configuration Properties -> Linker -> Input ->Additional Dependencies -> Add hand_pose_estimation.lib

### Dependencies
1. Get the AI3 package repository from https://resy-gitlab.inf.uni-bayreuth.de/tools/vcpkg (branch: ai3) or get the pre-built binaries from https://resy-gitlab.inf.uni-bayreuth.de/tools/vcpkg-binaries (branch: binaries)
2. The following packages must be installed: opencv, cuda, pcl, caffe[cuda]
3. To install a package, enter: `vcpkg install <package-name>:x64-windows`

### Training the skin detector
In case the detection result is poor, several steps can be taken:
1. Relevant parameters are found in: assets/hand_config/hand_pose_parameters.xml (e.g. hand_probability_threshold) and assets/hand_config/hand_dynamic_parameters.xml
2. Check that the skin detection does a proper job, call hand_tracker::show_skin_regions at the beginning of the code
2.1. Train the classifier by cropping images that are either skin or non-skin and place them in assets/skin_detection_training/skin|non_skin
2.2. Add entries to skin_file_paths.txt or non_skin_file_paths.txt, adding a file multiple times gives more weight to it
2.3. Delete assets/hand_config/probability_nskin.txt and assets/hand_config/probability_skin.txt
2.4. Run and close the application. This will recreate the above files with udpated values.

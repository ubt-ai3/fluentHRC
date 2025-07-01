add_executable(app_evaluation_net)

target_include_directories(app_evaluation_net PUBLIC ../)

target_sources(app_evaluation_net
  PRIVATE      
    module_manager.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    module_manager.hpp
    signal_types.hpp
)

target_link_libraries(app_evaluation_net
    PUBLIC
        state_observation
        simulation
)

target_compile_features(app_evaluation_net PUBLIC cxx_std_20)

LinkAssets(app_evaluation_net)
CopyRuntimeDeps(app_evaluation_net)
FixExecutionContext(app_evaluation_net)

install(
  TARGETS 
    app_evaluation_net
)
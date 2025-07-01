add_executable(app_util)

target_include_directories(app_util PUBLIC ../)

target_sources(app_util
  PRIVATE      
    extract_goal-oriented_actions.cpp
)

target_link_libraries(app_util
    PUBLIC
        state_observation
        simulation
)

target_compile_features(app_util PUBLIC cxx_std_20)

LinkAssets(app_util)
CopyRuntimeDeps(app_util)
FixExecutionContext(app_util)

install(
  TARGETS 
    app_util
)
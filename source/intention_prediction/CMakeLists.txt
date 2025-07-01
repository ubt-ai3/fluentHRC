add_library(intention_prediction SHARED)

target_include_directories(intention_prediction PUBLIC ../)
target_compile_definitions(intention_prediction PRIVATE INTENTIONPREDICTION_EXPORTS)

target_sources(intention_prediction
  PRIVATE    
    agent_manager.cpp
    observed_agent.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ./
  FILES
    agent_manager.hpp
    observed_agent.hpp
)

target_link_libraries(intention_prediction
    PUBLIC
        hand-pose-estimation
        state_observation
)

target_compile_features(intention_prediction PUBLIC cxx_std_20)

install(
  TARGETS 
    intention_prediction
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION include/intention_prediction
)
cmake_minimum_required(VERSION 3.25)

project(Flexcobot 
    LANGUAGES CXX CUDA)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -rdc=true")
#TODO::
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

add_compile_options(
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/MP>
)

if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
  add_compile_definitions(UNICODE _WIN32_WINNT=0x0A00 WIN32_LEAN_AND_MEAN BOOST_USE_WINDOWS_H NOMINMAX)
else ()
  add_compile_options(-Wa,-mbig-obj)
endif ()

find_package (Python COMPONENTS Interpreter REQUIRED)

set(AssetsDir ${CMAKE_SOURCE_DIR}/assets)
set(GrpcDir ${AssetsDir}/grpc)
list(APPEND CopyAbleAssets config models mogaze object_meshes prediction)

set(HandposeAssetsDir ${CMAKE_SOURCE_DIR}/externals/hand-pose-estimation/assets)
list(APPEND HandposeCopyAbleAssets hand_config hand_part_meshes network_models)

macro(LinkAssets target_name)
  add_custom_command(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${target_name}>/assets
    COMMAND_EXPAND_LISTS
  )
  foreach(asset IN LISTS CopyAbleAssets)
    if (MSVC)
      add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND cmd /C IF NOT EXIST "\"$<TARGET_FILE_DIR:${target_name}>/assets/${asset}\"" mklink /J "\"$<TARGET_FILE_DIR:${target_name}>/assets/${asset}\"" "\"${AssetsDir}/${asset}\"" 
        COMMAND_EXPAND_LISTS
      )
    else()
      add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink "\"${AssetsDir}/${asset}\"" "\"$<TARGET_FILE_DIR:${target_name}>/assets/${asset}\""
        COMMAND_EXPAND_LISTS
      )
      message(ERROR "not implemented for other platforms")
    endif()
  endforeach()
  foreach(asset IN LISTS HandposeCopyAbleAssets)
    if (MSVC)
      add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND cmd /C IF NOT EXIST "\"$<TARGET_FILE_DIR:${target_name}>/assets/${asset}\"" mklink /J "\"$<TARGET_FILE_DIR:${target_name}>/assets/${asset}\"" "\"${HandposeAssetsDir}/${asset}\"" 
        COMMAND_EXPAND_LISTS
      )
    else()
      add_custom_command(TARGET ${target_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink "\"${HandposeAssetsDir}/${asset}\"" "\"$<TARGET_FILE_DIR:${target_name}>/assets/${asset}\""
        COMMAND_EXPAND_LISTS
      )
      message(ERROR "not implemented for other platforms")
    endif()
  endforeach()
  file(GLOB shaders "${AssetsDir}/shaders/*.*")
  add_custom_command(TARGET ${target_name} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${shaders}" $<TARGET_FILE_DIR:${target_name}> 
    COMMAND_EXPAND_LISTS
  )
endmacro()

macro(CopyRuntimeDeps target_name)
  file(GENERATE 
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_$<CONFIG>_runtimedeps.txt"
    CONTENT "$<TARGET_RUNTIME_DLLS:${target_name}>"
  )

  add_custom_command(TARGET ${target_name} POST_BUILD
    #COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target_name}> $<TARGET_FILE_DIR:${target_name}>
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/util.py "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_$<CONFIG>_runtimedeps.txt" "$<TARGET_FILE_DIR:${target_name}>"
    #COMMAND ${CMAKE_COMMAND} -E echo $<TARGET_RUNTIME_DLLS:${target_name}>
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_$<CONFIG>_runtimedeps.txt"
    COMMAND_EXPAND_LISTS
  )
endmacro()



#macro(CopyCudnn target_name)
#  find_package(CUDNN REQUIRED)
#
#  cmake_path(GET CUDNN_INCLUDE_DIRS FILENAME folder_version)
#  file(GLOB cudnn_dlls "${CUDNN_INCLUDE_DIRS}/../../bin/${folder_version}/*.*")
#  #message("${CUDNN_INCLUDE_DIRS}/../bin/${folder_version}/*.*")
#  #message(folder_version=${folder_version})
#  #message(full_folder=${CUDNN_INCLUDE_DIRS}/../bin/${folder_version})
#
#  add_custom_command(TARGET ${target_name} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${cudnn_dlls} $<TARGET_FILE_DIR:${target_name}>
#    COMMAND_EXPAND_LISTS
#  )
#endmacro()

macro(FixExecutionContext target_name)
set_target_properties(
  ${target_name} PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${target_name}>)
endmacro()


add_subdirectory(assets/grpc)
add_subdirectory(externals)
add_subdirectory(source)
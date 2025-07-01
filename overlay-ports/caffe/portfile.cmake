vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO  Olli1080/caffe
    REF 1c2eeb4dfa603a6a328ee6e0a6478832bb08a4cd
    SHA512 710f18b98b96078f669b269de74c88e37a3c60d0a2a3681a553337e817fe352f81d86c3a9c9b2e53538397ea491cbf492737f6293c9a5445b8272b21c2cc464d
)

# Cannot use vcpkg_check_features because the template CaffeConfig.cmake.in is not correctly initiated otherwise

if("cuda" IN_LIST FEATURES)
    set(CPU_ONLY OFF)
else()
    set(CPU_ONLY ON)
endif()

if("mkl" IN_LIST FEATURES)
    set(BLAS MKL)
else()
    set(BLAS Open)
endif()

if("opencv")
    set(USE_OPENCV ON)
else()
    set(USE_OPENCV OFF)
endif()

if("lmdb" IN_LIST FEATURES)
    set(USE_LMDB ON)
else()
    set(USE_LMDB OFF)
endif()

if("leveldb" IN_LIST FEATURES)
    set(USE_LEVELDB ON)
else()
    set(USE_LEVELDB OFF)
endif()

if("python" IN_LIST FEATURES)
    set(USE_PYTHON ON)
else()
    set(USE_PYTHON OFF)
endif()

if("hdf5" IN_LIST FEATURES)
    set(USE_HDF5 ON)
else()
    set(USE_HDF5 OFF)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
    OPTIONS
    -DCOPY_PREREQUISITES=OFF
    -DINSTALL_PREREQUISITES=OFF
    # Set to ON to use python
    -DBUILD_python=${USE_PYTHON}
    -DBUILD_python_layer=${USE_PYTHON}
    -Dpython_version=3
    -DBUILD_matlab=OFF
    -DBUILD_docs=OFF
    -DBLAS=${BLAS}
    -DCPU_ONLY=${CPU_ONLY}
    #-DBUILD_TEST=OFF
    -DUSE_LEVELDB=${USE_LEVELDB}
    -DUSE_OPENCV=${USE_OPENCV}
    -DUSE_LMDB=${USE_LMDB}
    -DUSE_NCCL=OFF
    -DUSE_HDF5=${USE_HDF5}
    -DNVCC_THREADS=1
    -DBUILD_EXAMPLES=OFF
    #-DUSE_CUDNN=${}
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()

vcpkg_copy_pdbs()

# Move bin to tools
file(MAKE_DIRECTORY ${CURRENT_PACKAGES_DIR}/tools)
file(GLOB BINARIES ${CURRENT_PACKAGES_DIR}/bin/*.exe)
foreach(binary ${BINARIES})
    get_filename_component(binary_name ${binary} NAME)
    file(RENAME ${binary} ${CURRENT_PACKAGES_DIR}/tools/${binary_name})
endforeach()


file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/include)

if(USE_PYTHON)
    vcpkg_copy_tool_dependencies(${CURRENT_PACKAGES_DIR}/python)
else()
    file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/python)
    file(REMOVE_RECURSE ${CURRENT_PACKAGES_DIR}/debug/python)
endif()

file(GLOB DEBUG_BINARIES ${CURRENT_PACKAGES_DIR}/debug/bin/*.exe)
file(REMOVE ${DEBUG_BINARIES})

file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)
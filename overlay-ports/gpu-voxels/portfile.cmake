vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Olli1080/gpu-voxels
    REF 82379b9a0ba28a440b1f0a3c5557e8aee6693d70
    HEAD_REF ar_integration
    SHA512 27ff6d3e4fea9fb8f8ef885844252d2fd7261892ebf302f274c25ad453832a2c5b8db4f2a1603c49385c22848917b97db49f40d3e88410f137e99c3321daa1ee
)

vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DUSE_ZLIB=ON
        -DNVCC_THREADS=1
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(
    PACKAGE_NAME gpu_voxels
    CONFIG_PATH share/gpu_voxels
)
vcpkg_cmake_config_fixup(
    PACKAGE_NAME icl_core
    CONFIG_PATH share/icl_core
)
vcpkg_copy_pdbs()

file(REMOVE_RECURSE 
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/bin"
    "${CURRENT_PACKAGES_DIR}/debug/bin"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt" "${SOURCE_PATH}/LICENSE_BSD.txt" "${SOURCE_PATH}/LICENSE_CDDL.txt" "${SOURCE_PATH}/LICENSE_PBA.txt")

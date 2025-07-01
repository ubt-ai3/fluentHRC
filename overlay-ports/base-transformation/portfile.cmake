vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Olli1080/CoordTrafoUtil
    REF 3575a14a90bf2e53dd5fe029f826c2993f6eae82
    HEAD_REF remove_vtable
    SHA512 74aa55ff948469987042acba06f1d7de63e963df98f1bf0508403f9812834579629d5c24c9804d3cb37b223b2469e90c856c3ddd3fdbf4795fb17187bddd823d
)

vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(
    PACKAGE_NAME base-transformation
    CONFIG_PATH share/base-transformation
)

vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

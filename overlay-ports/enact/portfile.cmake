vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_git(
    OUT_SOURCE_PATH SOURCE_PATH
    URL https://resy-gitlab.inf.uni-bayreuth.de/enact/enact
    REF 21e5a0e9f7b30c074b787382223b2c3a5840ddfa
    HEAD_REF building_agent
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/enact")
vcpkg_copy_pdbs()

file(
    INSTALL "${CMAKE_CURRENT_LIST_DIR}/License.txt" 
    DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
    RENAME copyright
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
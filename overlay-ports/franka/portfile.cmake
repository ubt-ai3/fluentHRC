# Common Ambient Variables:
#   CURRENT_BUILDTREES_DIR    = ${VCPKG_ROOT_DIR}\buildtrees\${PORT}
#   CURRENT_PACKAGES_DIR      = ${VCPKG_ROOT_DIR}\packages\${PORT}_${TARGET_TRIPLET}
#   CURRENT_PORT_DIR          = ${VCPKG_ROOT_DIR}\ports\${PORT}
#   CURRENT_INSTALLED_DIR     = ${VCPKG_ROOT_DIR}\installed\${TRIPLET}
#   DOWNLOADS                 = ${VCPKG_ROOT_DIR}\downloads
#   PORT                      = current port name (zlib, etc)
#   TARGET_TRIPLET            = current triplet (x86-windows, x64-windows-static, etc)
#   VCPKG_CRT_LINKAGE         = C runtime linkage type (static, dynamic)
#   VCPKG_LIBRARY_LINKAGE     = target library linkage type (static, dynamic)
#   VCPKG_ROOT_DIR            = <C:\path\to\current\vcpkg>
#   VCPKG_TARGET_ARCHITECTURE = target architecture (x64, x86, arm)
#   VCPKG_TOOLCHAIN           = ON OFF
#   TRIPLET_SYSTEM_ARCH       = arm x86 x64
#   BUILD_ARCH                = "Win32" "x64" "ARM"
#   MSBUILD_PLATFORM          = "Win32"/"x64"/${TRIPLET_SYSTEM_ARCH}
#   DEBUG_CONFIG              = "Debug Static" "Debug Dll"
#   RELEASE_CONFIG            = "Release Static"" "Release DLL"
#   VCPKG_TARGET_IS_WINDOWS
#   VCPKG_TARGET_IS_UWP
#   VCPKG_TARGET_IS_LINUX
#   VCPKG_TARGET_IS_OSX
#   VCPKG_TARGET_IS_FREEBSD
#   VCPKG_TARGET_IS_ANDROID
#   VCPKG_TARGET_IS_MINGW
#   VCPKG_TARGET_EXECUTABLE_SUFFIX
#   VCPKG_TARGET_STATIC_LIBRARY_SUFFIX
#   VCPKG_TARGET_SHARED_LIBRARY_SUFFIX
#
# 	See additional helpful variables in /docs/maintainers/vcpkg_common_definitions.md

# # Specifies if the port install should fail immediately given a condition
# vcpkg_fail_port_install(MESSAGE "libfranka currently only supports Linux and Mac platforms" ON_TARGET "Windows")

# Download and Patch
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO frankaemika/libfranka
    REF f3b8d775a9c847cab32684c8a316f67867761674
    SHA512 68050fa184f8a94d47165f4747d889b8647c65f3d9fb028f748aba7239d2ee2057089abaa4497b51757f535666404b19031c84887f0d111400356af881a19874
	PATCHES
      libfranka_1.patch
      #0001-windows_includes.patch
      #0002-create_project_file.patch
	  #0003-missing_includes.patch
	  #0004-fix_cmake_lists.patch
)

vcpkg_from_github(
    OUT_SOURCE_PATH COMMON_SOURCE_PATH
    REPO frankaemika/libfranka-common
    REF e6aa0fc210d93fe618bfd8956829a264d5476ba8
    SHA512 225300c47be41a180da6eaabe28a9e69f6a940fb42a2752eb27d0053e6fdcd05d003da0a3987f0e88d9a15a6fc9677c815460d3155526685012f06e37a5b9542
	PATCHES
      common_1.patch
      #0001-fix_cmake.patch
)

file(COPY ${COMMON_SOURCE_PATH}/ DESTINATION ${SOURCE_PATH}/common)


# Build and Install
vcpkg_configure_cmake(
    SOURCE_PATH ${SOURCE_PATH}
    PREFER_NINJA # Disable this option if project cannot be built with Ninja
    OPTIONS
        -DBUILD_COVERAGE=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_DOCUMENTATION=OFF
    # OPTIONS -DUSE_THIS_IN_ALL_BUILDS=1 -DUSE_THIS_TOO=2
    # OPTIONS_RELEASE -DOPTIMIZE=1
    # OPTIONS_DEBUG -DDEBUGGABLE=1
)

vcpkg_install_cmake()

# Fix cmake structure
vcpkg_fixup_cmake_targets(CONFIG_PATH lib/cmake/Franka)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

# Handle copyright
file(INSTALL ${SOURCE_PATH}/LICENSE DESTINATION ${CURRENT_PACKAGES_DIR}/share/${PORT} RENAME copyright)

# Post-build test for cmake libraries
SET(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)
#vcpkg_test_cmake(PACKAGE_NAME franka)

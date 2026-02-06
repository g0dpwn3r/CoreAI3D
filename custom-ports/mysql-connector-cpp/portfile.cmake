vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO g0dpwn3r/mysql-connector-cpp
    # Make sure this REF matches the commit SHA you want to build
    REF 6ebe27f3176b38b68830613933d960e963eb9310 
    SHA512 2cb514a175835684297bebc5930ff482b8f6825273e24d6957a53061b945c05e6e95bd83a09325b400ddf2cfec0267503dd75066754d87bcec286844d5d9ccf2
    HEAD_REF master
    PATCHES
        # List any external .patch files here if you have them, otherwise leave empty
)

# --- CRITICAL FIX: Patch value.h to include <cstdint> ---
message(STATUS "Patching include/mysqlx/common/value.h to fix uint64_t error...")
vcpkg_replace_string(
    "${SOURCE_PATH}/include/mysqlx/common/value.h"
    "#include \"error.h\""
    "#include \"error.h\"\n#include <cstdint>"
)
# --------------------------------------------------------

# Clean up any leftover build artifacts from the tarball to ensure a fresh configure
file(REMOVE_RECURSE "${SOURCE_PATH}/build")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DWITH_JDBC=ON
        -DWITH_X=ON
        # Force the connector to use the system (vcpkg) OpenSSL and Protobuf
        # instead of building its own bundled versions
        -DWITH_SSL=system
        -DBUNDLE_DEPENDENCIES=OFF 
        -DINFO_SRC=OFF
        -DINFO_BIN=OFF
        # Ensure we are building static libs if vcpkg triplet is static
        -DBUILD_STATIC=ON 
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()

# -------------------------------------------------------------------------
# CRITICAL FIX: Move libraries from 'debug/lib/debug' to 'debug/lib'
# -------------------------------------------------------------------------
if(EXISTS "${CURRENT_PACKAGES_DIR}/debug/lib/debug")
    message(STATUS "Fixing nested debug library paths...")
    file(GLOB_RECURSE NESTED_DEBUG_LIBS "${CURRENT_PACKAGES_DIR}/debug/lib/debug/*.a")
    foreach(LIB_PATH ${NESTED_DEBUG_LIBS})
        get_filename_component(LIB_NAME "${LIB_PATH}" NAME)
        file(RENAME "${LIB_PATH}" "${CURRENT_PACKAGES_DIR}/debug/lib/${LIB_NAME}")
    endforeach()
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/lib/debug")
endif()

# -------------------------------------------------------------------------
# CRITICAL FIX: Standardize Library Names
# CMake looks for 'mysqlcppconn', 'mysqlcppconn8', etc.
# We rename '*-static.a' to '.a' and create copies for compatibility.
# -------------------------------------------------------------------------
file(GLOB_RECURSE ALL_STATIC_LIBS "${CURRENT_PACKAGES_DIR}/*-static.a")

foreach(LIB_PATH ${ALL_STATIC_LIBS})
    get_filename_component(DIR_PATH "${LIB_PATH}" DIRECTORY)
    get_filename_component(LIB_NAME "${LIB_PATH}" NAME)
    
    # 1. Rename libmysqlcppconn-static.a -> libmysqlcppconn.a
    string(REGEX REPLACE "-static\\.a$" ".a" NEW_NAME "${LIB_NAME}")
    set(NEW_PATH "${DIR_PATH}/${NEW_NAME}")
    
    if(NOT EXISTS "${NEW_PATH}")
        file(RENAME "${LIB_PATH}" "${NEW_PATH}")
    endif()

    # 2. Create '8' alias (e.g. libmysqlcppconn.a -> libmysqlcppconn8.a)
    # Many CMake find scripts specifically look for the '8' version.
    string(REGEX REPLACE "mysqlcppconn" "mysqlcppconn8" LIB_NAME_8 "${NEW_NAME}")
    if(NOT "${NEW_NAME}" STREQUAL "${LIB_NAME_8}")
        file(COPY "${NEW_PATH}" DESTINATION "${DIR_PATH}")
        file(RENAME "${DIR_PATH}/${NEW_NAME}" "${DIR_PATH}/${LIB_NAME_8}")
    endif()
endforeach()

# Clean up garbage
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/README")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
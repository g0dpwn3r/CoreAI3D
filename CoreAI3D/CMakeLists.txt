cmake_minimum_required(VERSION 3.10)
project(CoreAI3D CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Find MySQL Connector/C++ (X DevAPI) via vcpkg's export
# This package typically provides the mysql-connector-cpp::mysqlx target.
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    find_package(PkgConfig REQUIRED)    
    find_package(mysql-concpp CONFIG REQUIRED)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    find_package(mysql-concpp CONFIG REQUIRED)
endif()

find_package(Boost REQUIRED COMPONENTS system)

# Find cURL
find_package(CURL REQUIRED)

# Find nlohmann_json
find_package(nlohmann_json CONFIG REQUIRED) # Assumes nlohmann_json is installed via vcpkg and provides a config file

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(INCLUDE_DIR /usr/include)
    set(LIBRARY_DIR /usr/lib/x86_64-linux-gnu)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(INCLUDE_DIR D:/code/CoreAI3D/vcpkg_installed/x64-windows/include)
    set(LIBRARY_DIR D:/code/CoreAI3D/vcpkg_installed/x64-windows/lib)
endif()


# Include directories for your project headers
include_directories(. include ${INCLUDE_DIR})
link_directories(${LIBRARY_DIR})

# Source files
set(SOURCE_FILES
    main.cpp
    src/Core.cpp
    src/Train.cpp
    src/Database.cpp
    src/Language.cpp
)

add_executable(CoreAI3D ${SOURCE_FILES})
# Detect Windows host building for WSL (e.g., syncing to /home/user)
# Link libraries conditionally based on OS (though vcpkg usually abstracts this)
# CMAKE_SYSTEM_NAME will be "Windows" on Windows, and "Linux" on WSL.
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(STATUS "Building for Windows.")
    target_link_libraries(CoreAI3D PRIVATE
        Boost::system  
        mysql::concpp # Use the target provided by mysql-connector-cpp
        CURL::libcurl
        nlohmann_json::nlohmann_json
        # Add any other Windows-specific libraries if needed
    )
elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
    link_directories(/usr/lib/x86_64-linux-gnu)
    message(STATUS "Building for Linux (WSL).")
    target_link_libraries(CoreAI3D PRIVATE
        Boost::system  
        mysqlcppconnx   # Use the target provided by mysql-connector-cpp
        CURL::libcurl
        nlohmann_json::nlohmann_json
        # Add any other Linux-specific libraries if needed
    )
else()
    message(FATAL_ERROR "Unsupported operating system: ${CMAKE_SYSTEM_NAME}")
endif()


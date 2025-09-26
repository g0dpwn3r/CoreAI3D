
cmake_minimum_required(VERSION 3.16)
project(CoreAI3D)

# Paths for protobuf imports (relative to project root)
set(GOOGLEAPIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../googleapis")
set(GOOGLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../google")

# Zorg dat protoc beide dirs kan zien
set(PROTOBUF_IMPORT_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GOOGLEAPIS_DIR}
    ${GOOGLE_DIR}
)
# --- Find and Link Boost Components ---
# Find all required Boost components using the CONFIG mode.
find_package(Boost CONFIG REQUIRED)
find_package(unofficial-mysql-connector-cpp CONFIG REQUIRED)
# Find Abseil (absl) - protobuf's dependency
find_package(absl CONFIG REQUIRED)

# Find utf8_range - protobuf's dependency
find_package(utf8_range CONFIG REQUIRED)
# Find ZLIB (compatible with built-in CMake targets)
find_package(ZLIB REQUIRED)

# Find CURL (compatible with built-in CMake targets)
find_package(CURL REQUIRED)

# Find zstd (provides CMake targets via CONFIG)
find_package(zstd CONFIG REQUIRED)

# Find protobuf (provides CMake targets via CONFIG)
# Note: This is heuristically generated; you might need to adjust components
# based on your specific protobuf usage (e.g., if you only need libprotobuf-lite).
find_package(protobuf CONFIG REQUIRED)

# Find nlohmann-json (provides CMake targets via CONFIG)
find_package(nlohmann_json CONFIG REQUIRED)


# --- Define Executable and Link Libraries ---

# Add your executable target.
# The name 'CoreAI3D' here will be the name of your .exe file (e.g., CoreAI3D.exe).
# 'main.cpp' is assumed to be in the same directory as this CMakeLists.txt.
# Source files
# Gebruik protobuf_generate_cpp (van FindProtobuf)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
    ${CMAKE_CURRENT_SOURCE_DIR}/cloud_speech.proto
    IMPORT_DIRS ${PROTOBUF_IMPORT_DIRS}
)

add_library(proto_objs ${PROTO_SRCS} ${PROTO_HDRS})

# Copy protobuf include directories to build directory for easier access
add_custom_command(TARGET proto_objs PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${GOOGLE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/google
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${GOOGLEAPIS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/googleapis
    COMMENT "Copying protobuf include directories to build directory"
)

target_include_directories(proto_objs PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/google
    ${CMAKE_CURRENT_BINARY_DIR}/googleapis
    ${PROTOBUF_INCLUDE_DIR}
)

set(SOURCE_FILES
    main.cpp
    src/Core.cpp
    src/Train.cpp
    src/Database.cpp
    src/Language.cpp
    src/APIServer.cpp
    src/AudioModule.cpp
    src/MathModule.cpp
    src/SystemModule.cpp
    src/WebModule.cpp
    src/ModuleOrchestrator.cpp
    src/WebSocketServer.cpp
    src/VisionModule.cpp
    src/Gemini.cpp
)

# Add Linux-specific modules only on Linux
if(NOT WIN32)
    list(APPEND SOURCE_FILES src/LinuxModule.cpp)
endif()

add_executable(CoreAI3D ${PROTO_SRCS} ${PROTO_HDRS} ${SOURCE_FILES})

# Include directories for protobuf headers
target_include_directories(CoreAI3D PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # Include current directory for main.cpp
    include  # Include directory for your project headers
    ${CMAKE_CURRENT_BINARY_DIR}  # Include generated protobuf headers
    ${CMAKE_CURRENT_BINARY_DIR}/google  # Include copied google directory
    ${CMAKE_CURRENT_BINARY_DIR}/googleapis  # Include copied googleapis directory
    ${PROTOBUF_INCLUDE_DIR}  # Include protobuf library headers
    ${Boost_INCLUDE_DIRS}
)

# Find protobuf library include directory
find_path(PROTOBUF_INCLUDE_DIR google/protobuf/runtime_version.h
    PATHS
        "D:/code/vcpkg/installed/x64-windows/include"
        "C:/vcpkg/installed/x64-windows/include"
        "$ENV{VCPKG_ROOT}/installed/x64-windows/include"
        "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/include"
        "D:/vcpkg/installed/x64-windows/include"
        "C:/Program Files/vcpkg/installed/x64-windows/include"
    REQUIRED
)

if(PROTOBUF_INCLUDE_DIR)
    target_include_directories(CoreAI3D PRIVATE ${PROTOBUF_INCLUDE_DIR})
    message(STATUS "Found protobuf include dir: ${PROTOBUF_INCLUDE_DIR}")
else()
    message(WARNING "Could not find protobuf include directory")
endif()

# Add protobuf include directories
target_link_libraries(CoreAI3D PRIVATE protobuf::libprotobuf)

# Define _WIN32_WINNT and WIN32_LEAN_AND_MEAN directly for this target.
# _WIN32_WINNT=0x0A00 targets Windows 10.
# WIN32_LEAN_AND_MEAN reduces the size of windows.h, helping to prevent WinSock.h conflicts.
if(WIN32)
    add_compile_definitions(
        WIN32_LEAN_AND_MEAN  # Excludes less-used headers from windows.h to speed up compilation and reduce conflicts
        NOMINMAX             # Prevents min/max macros in windows.h from conflicting with std::min/std::max
        _WINSOCKAPI_         # Prevents winsock.h from being included by windows.h, forcing winsock2.h
    )
    target_compile_definitions(CoreAI3D PRIVATE
        _WIN32_WINNT=0x0A00
        WIN32_LEAN_AND_MEAN
    )
endif()
# Link your executable with all the required libraries.
# The targets (e.g., Boost::program_options, ZLIB::ZLIB) are "imported targets"
# provided by the respective find_package calls.
target_link_libraries(CoreAI3D PRIVATE
    Boost::boost
    absl::absl_check # Example Abseil targets, link what's needed by your code/protobuf
    absl::absl_log
    absl::algorithm
    absl::base
    absl::bind_front
    absl::bits
    absl::btree
    absl::cleanup
    absl::cord
    absl::core_headers
    absl::debugging
    absl::die_if_null
    absl::dynamic_annotations
    absl::flags
    absl::flat_hash_map
    absl::flat_hash_set
    absl::function_ref
    absl::hash
    absl::layout
    absl::log_initialize
    absl::log_globals
    absl::log_severity
    absl::memory
    absl::node_hash_map
    absl::node_hash_set
    absl::random_distributions
    absl::random_random
    absl::span
    absl::status
    absl::statusor
    absl::strings
    absl::synchronization
    absl::time
    absl::type_traits
    absl::utility
    utf8_range::utf8_validity
    utf8_range::utf8_range
    ZLIB::ZLIB
    CURL::libcurl
    zstd::libzstd
    protobuf::libprotoc
    protobuf::libprotobuf
    protobuf::libprotobuf-lite
    unofficial::mysql-connector-cpp::connector
    nlohmann_json::nlohmann_json
)
# If you have other source files in this subdirectory, add them to add_executable:
# add_executable(CoreAI3D main.cpp another_file.cpp yet_another.cpp)

# If you have custom include directories for this specific subdirectory, add them here:
# target_include_directories(CoreAI3D PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

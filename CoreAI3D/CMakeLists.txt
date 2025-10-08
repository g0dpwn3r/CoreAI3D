cmake_minimum_required(VERSION 3.16)
project(CoreAI3D)

# If vcpkg toolchain exists next to the repo, use it so find_package(CONFIG) can find packages installed by vcpkg
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake")
    message(STATUS "Using vcpkg toolchain: ${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "Vcpkg toolchain file" FORCE)
endif()

# Ensure there's a sensible vcpkg triplet defined for constructing prefixes
if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    if(WIN32)
        set(VCPKG_TARGET_TRIPLET "x64-windows")
    else()
        set(VCPKG_TARGET_TRIPLET "x64-linux")
    endif()
endif()

# If a vcpkg install prefix exists, add it to CMAKE_PREFIX_PATH so find_package can locate packages like Boost/Protobuf
if(NOT DEFINED _VCPKG_PROBED)
    set(_VCPKG_PROBED TRUE CACHE INTERNAL "internal flag")
    if(DEFINED ENV{VCPKG_ROOT})
        set(_vcpkg_root $ENV{VCPKG_ROOT})
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg")
        set(_vcpkg_root "${CMAKE_SOURCE_DIR}/vcpkg")
    else()
        set(_vcpkg_root "")
    endif()

    if(_vcpkg_root)
        set(_vcpkg_installed "${_vcpkg_root}/installed/${VCPKG_TARGET_TRIPLET}")
        if(EXISTS "${_vcpkg_installed}")
            list(APPEND CMAKE_PREFIX_PATH "${_vcpkg_installed}")
            message(STATUS "Appended vcpkg installed prefix to CMAKE_PREFIX_PATH: ${_vcpkg_installed}")
            # set Boost_ROOT to help FindBoost in CONFIG mode
            set(Boost_ROOT "${_vcpkg_installed}")
            set(Boost_NO_SYSTEM_PATHS ON)
        endif()
    endif()
endif()

# Zorg dat protoc beide dirs kan zien
set(PROTOBUF_IMPORT_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# If local google/googleapis directories exist inside this subproject, add them to import dirs
set(LOCAL_GOOGLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/google")
set(LOCAL_GOOGLEAPIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/googleapis")
if(EXISTS "${LOCAL_GOOGLE_DIR}")
    list(APPEND PROTOBUF_IMPORT_DIRS "${LOCAL_GOOGLE_DIR}")
    message(STATUS "Adding local google proto dir to import dirs: ${LOCAL_GOOGLE_DIR}")
endif()
if(EXISTS "${LOCAL_GOOGLEAPIS_DIR}")
    list(APPEND PROTOBUF_IMPORT_DIRS "${LOCAL_GOOGLEAPIS_DIR}")
    message(STATUS "Adding local googleapis proto dir to import dirs: ${LOCAL_GOOGLEAPIS_DIR}")
    # Copy local googleapis into the binary dir at configure time so protoc can find imports during code generation
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/googleapis")
        file(COPY "${LOCAL_GOOGLEAPIS_DIR}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
        message(STATUS "Copied local googleapis to build dir: ${CMAKE_CURRENT_BINARY_DIR}/googleapis")
    else()
        message(STATUS "Build dir already contains googleapis: ${CMAKE_CURRENT_BINARY_DIR}/googleapis")
    endif()
endif()

# --- Find and Link Boost Components ---
# Use FindBoost (module mode) requesting compiled components; asio/beast are header-only so not requested here.
find_package(Boost REQUIRED COMPONENTS program_options filesystem system regex)
if(Boost_FOUND)
    message(STATUS "Found Boost include dirs: ${Boost_INCLUDE_DIRS}")
else()
    message(WARNING "Boost not found. Install Boost via vcpkg (e.g. 'vcpkg install boost-program-options boost-filesystem boost-system boost-regex') and ensure the vcpkg toolchain is used.")
endif()

find_package(unofficial-mysql-connector-cpp CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(utf8_range CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(CURL REQUIRED)
find_package(zstd CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

# Prefer Protobuf include dir provided by the CMake package
if(NOT DEFINED PROTOBUF_INCLUDE_DIR OR NOT PROTOBUF_INCLUDE_DIR)
    if(DEFINED Protobuf_INCLUDE_DIR)
        set(PROTOBUF_INCLUDE_DIR ${Protobuf_INCLUDE_DIR})
    elseif(DEFINED Protobuf_INCLUDE_DIRS)
        # take first entry
        list(GET Protobuf_INCLUDE_DIRS 0 _first_proto_include)
        set(PROTOBUF_INCLUDE_DIR ${_first_proto_include})
    elseif(TARGET protobuf::libprotobuf)
        # extract include dirs from the imported target
        get_target_property(_pb_includes protobuf::libprotobuf INTERFACE_INCLUDE_DIRECTORIES)
        if(_pb_includes)
            list(GET _pb_includes 0 _first_proto_include)
            set(PROTOBUF_INCLUDE_DIR ${_first_proto_include})
        endif()
    endif()
endif()

# If still not found, probe common vcpkg locations as a fallback
if(NOT PROTOBUF_INCLUDE_DIR)
    find_path(PROTOBUF_INCLUDE_DIR google/protobuf/runtime_version.h
        PATHS
            "D:/code/vcpkg/installed/x64-windows/include"
            "C:/vcpkg/installed/x64-windows/include"
            "$ENV{VCPKG_ROOT}/installed/x64-windows/include"
            "${CMAKE_SOURCE_DIR}/vcpkg_installed/installed/${VCPKG_TARGET_TRIPLET}/include"
            "$ENV{USERPROFILE}/vcpkg/installed/x64-windows/include"
            "D:/vcpkg/installed/x64-windows/include"
            "C:/Program Files/vcpkg/installed/x64-windows/include"
        NO_DEFAULT_PATH
    )
endif()

if(PROTOBUF_INCLUDE_DIR)
    message(STATUS "Found protobuf include dir: ${PROTOBUF_INCLUDE_DIR}")
    list(APPEND PROTOBUF_IMPORT_DIRS ${PROTOBUF_INCLUDE_DIR})
    list(APPEND PROTOBUF_IMPORT_DIRS ${PROTOBUF_INCLUDE_DIR}/google ${PROTOBUF_INCLUDE_DIR}/googleapis)
else()
    message(WARNING "Could not find protobuf include directory; proto import may fail if google/googleapis are not available. Install protobuf via vcpkg and ensure the vcpkg toolchain is used.")
endif()

# Build explicit import dirs list with preferred order (local googleapis first)
set(IMPORT_DIRS_FOR_PROTO "")
# prefer local googleapis then local google, then protobuf include dir, then project root
if(EXISTS "${LOCAL_GOOGLEAPIS_DIR}")
    list(APPEND IMPORT_DIRS_FOR_PROTO "${LOCAL_GOOGLEAPIS_DIR}")
    # also prefer the copied binary location if present
    list(APPEND IMPORT_DIRS_FOR_PROTO "${CMAKE_CURRENT_BINARY_DIR}/googleapis")
endif()
if(EXISTS "${LOCAL_GOOGLE_DIR}")
    list(APPEND IMPORT_DIRS_FOR_PROTO "${LOCAL_GOOGLE_DIR}")
    list(APPEND IMPORT_DIRS_FOR_PROTO "${CMAKE_CURRENT_BINARY_DIR}/google")
endif()
if(PROTOBUF_INCLUDE_DIR)
    list(APPEND IMPORT_DIRS_FOR_PROTO "${PROTOBUF_INCLUDE_DIR}")
    list(APPEND IMPORT_DIRS_FOR_PROTO "${PROTOBUF_INCLUDE_DIR}/google")
    list(APPEND IMPORT_DIRS_FOR_PROTO "${PROTOBUF_INCLUDE_DIR}/googleapis")
endif()
list(APPEND IMPORT_DIRS_FOR_PROTO "${CMAKE_CURRENT_SOURCE_DIR}")

# Remove any import dirs that do not exist
set(VALID_PROTOBUF_IMPORT_DIRS "")
foreach(_dir IN LISTS IMPORT_DIRS_FOR_PROTO)
    if(_dir AND EXISTS "${_dir}")
        list(APPEND VALID_PROTOBUF_IMPORT_DIRS "${_dir}")
    else()
        message(VERBOSE "Skipping non-existing proto import dir: ${_dir}")
    endif()
endforeach()

message(STATUS "Final protobuf import dirs: ${VALID_PROTOBUF_IMPORT_DIRS}")

# Check proto source exists
set(CLOUD_SPEECH_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/cloud_speech.proto")
if(NOT EXISTS "${CLOUD_SPEECH_PROTO}")
    message(FATAL_ERROR "Missing proto file: ${CLOUD_SPEECH_PROTO}. Ensure cloud_speech.proto is present in ${CMAKE_CURRENT_SOURCE_DIR}.")
endif()

# --- Define Executable and Link Libraries ---

# Use protobuf_generate_cpp (from FindProtobuf)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
    ${CLOUD_SPEECH_PROTO}
    IMPORT_DIRS ${VALID_PROTOBUF_IMPORT_DIRS}
)

add_library(proto_objs ${PROTO_SRCS} ${PROTO_HDRS})

# Copy protobuf include directories to build directory for easier access only when local dirs exist
if(EXISTS "${LOCAL_GOOGLE_DIR}" AND EXISTS "${LOCAL_GOOGLEAPIS_DIR}")
    add_custom_command(TARGET proto_objs PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${LOCAL_GOOGLE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/google
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${LOCAL_GOOGLEAPIS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/googleapis
        COMMENT "Copying protobuf include directories to build directory"
    )
else()
    message(STATUS "Skipping copy of google/googleapis into build tree because local directories are missing; using Protobuf include paths from vcpkg/system")
endif()

target_include_directories(proto_objs PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)
if(PROTOBUF_INCLUDE_DIR)
    target_include_directories(proto_objs PUBLIC ${PROTOBUF_INCLUDE_DIR})
endif()

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
)

# If we copied local google/googleapis into the build tree, include them too
if(EXISTS "${LOCAL_GOOGLE_DIR}" AND EXISTS "${LOCAL_GOOGLEAPIS_DIR}")
    target_include_directories(CoreAI3D PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/google
        ${CMAKE_CURRENT_BINARY_DIR}/googleapis
    )
else()
    # Otherwise include protobuf include dir detected earlier (from vcpkg/system)
    if(PROTOBUF_INCLUDE_DIR)
        target_include_directories(CoreAI3D PRIVATE ${PROTOBUF_INCLUDE_DIR})
    endif()
endif()

# Also add the generic protobuf include if we have it
if(PROTOBUF_INCLUDE_DIR)
    target_include_directories(CoreAI3D PRIVATE ${PROTOBUF_INCLUDE_DIR})
endif()

# Build library list conditionally to avoid referencing missing Boost targets
set(PROJECT_LIBS)
if(TARGET Boost::program_options)
    list(APPEND PROJECT_LIBS Boost::program_options)
endif()
if(TARGET Boost::filesystem)
    list(APPEND PROJECT_LIBS Boost::filesystem)
endif()
if(TARGET Boost::system)
    list(APPEND PROJECT_LIBS Boost::system)
endif()

# Add protobuf include directories
target_link_libraries(CoreAI3D PRIVATE
    ${PROJECT_LIBS}
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
    gRPC::grpc++
    zstd::libzstd
    protobuf::libprotoc
    protobuf::libprotobuf
    protobuf::libprotobuf-lite
    unofficial::mysql-connector-cpp::connector
    nlohmann_json::nlohmann_json
)

# After computing VALID_PROTOBUF_IMPORT_DIRS, print them for diagnostics
message(STATUS "Proto import dirs used for generation: ${VALID_PROTOBUF_IMPORT_DIRS}")

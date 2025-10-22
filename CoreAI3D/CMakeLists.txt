cmake_minimum_required(VERSION 3.16)
project(CoreAI3D)


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

# Set Boost policy to avoid version mismatch warnings
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# --- Find and Link Boost Components ---
# Use only vcpkg Boost to avoid version conflicts
find_package(boost_program_options CONFIG REQUIRED)
find_package(boost_filesystem CONFIG REQUIRED)
find_package(boost_system CONFIG REQUIRED)
find_package(boost_regex CONFIG REQUIRED)
# Use vcpkg Boost CONFIG mode only
if(Boost_FOUND)
    message(STATUS "Found Boost include dirs: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Found Boost libraries: ${Boost_LIBRARIES}")
else()
    message(WARNING "Boost not found. Install Boost via vcpkg or ensure Boost is available in your system paths.")
endif()
# Force vcpkg paths to come first for MySQL connector and other vcpkg packages
list(INSERT CMAKE_PREFIX_PATH 0 "/home/carlon/code/CoreAI3D/vcpkg_installed/x64-linux/share")
list(INSERT CMAKE_PREFIX_PATH 0 "/home/carlon/code/CoreAI3D/vcpkg_installed/x64-linux")
# Find MySQL connector using vcpkg
find_package(unofficial-mysql-connector-cpp CONFIG)
find_package(absl CONFIG REQUIRED)
find_package(utf8_range CONFIG)
find_package(ZLIB REQUIRED)
find_package(CURL REQUIRED)
find_package(zstd CONFIG)
if(NOT zstd_FOUND)
    message(WARNING "zstd CONFIG not found, trying MODULE mode")
    find_package(zstd MODULE)
    if(NOT zstd_FOUND)
        message(WARNING "zstd not found, skipping zstd dependency")
    endif()
endif()
find_package(nlohmann_json CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(google_cloud_cpp_speech CONFIG REQUIRED)
find_package(google_cloud_cpp_common CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(lz4 CONFIG)
if(NOT lz4_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(lz4 REQUIRED liblz4)
endif()

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


if(PROTOBUF_INCLUDE_DIR)
    message(STATUS "Found protobuf include dir: ${PROTOBUF_INCLUDE_DIR}")
    list(APPEND PROTOBUF_IMPORT_DIRS ${PROTOBUF_INCLUDE_DIR})
    list(APPEND PROTOBUF_IMPORT_DIRS ${PROTOBUF_INCLUDE_DIR}/google ${PROTOBUF_INCLUDE_DIR}/googleapis)
else()
    message(WARNING "Could not find protobuf include directory; proto import may fail if google/googleapis are not available. Install protobuf via system package manager.")
endif()

# Find google-cloud-cpp include dirs
if(TARGET google-cloud-cpp::speech)
    get_target_property(GOOGLE_CLOUD_CPP_INCLUDE_DIRS google-cloud-cpp::speech INTERFACE_INCLUDE_DIRECTORIES)
    if(GOOGLE_CLOUD_CPP_INCLUDE_DIRS)
        message(STATUS "Found google-cloud-cpp include dirs: ${GOOGLE_CLOUD_CPP_INCLUDE_DIRS}")
        list(APPEND IMPORT_DIRS_FOR_PROTO ${GOOGLE_CLOUD_CPP_INCLUDE_DIRS})
    endif()
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
    message(STATUS "Skipping copy of google/googleapis into build tree because local directories are missing; using Protobuf include paths from system")
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

# Create library source files (exclude main.cpp)
set(LIB_SOURCE_FILES ${SOURCE_FILES})
list(REMOVE_ITEM LIB_SOURCE_FILES main.cpp)

# Create CoreAI3DLib library
add_library(CoreAI3DLib ${LIB_SOURCE_FILES})

# Set include directories for CoreAI3DLib
target_include_directories(CoreAI3DLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Add USE_MYSQL preprocessor definition conditionally
option(USE_MYSQL "Enable MySQL support" OFF)
if(USE_MYSQL)
    target_compile_definitions(CoreAI3DLib PUBLIC USE_MYSQL)
endif()

# If we copied local google/googleapis into the build tree, include them too
if(EXISTS "${LOCAL_GOOGLE_DIR}" AND EXISTS "${LOCAL_GOOGLEAPIS_DIR}")
    target_include_directories(CoreAI3DLib PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/google
        ${CMAKE_CURRENT_BINARY_DIR}/googleapis
    )
else()
    # Otherwise include protobuf include dir detected earlier (from system)
    if(PROTOBUF_INCLUDE_DIR)
        target_include_directories(CoreAI3DLib PUBLIC ${PROTOBUF_INCLUDE_DIR})
    endif()
endif()

# Also add the generic protobuf include if we have it
if(PROTOBUF_INCLUDE_DIR)
    target_include_directories(CoreAI3DLib PUBLIC ${PROTOBUF_INCLUDE_DIR})
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
if(TARGET Boost::regex)
    list(APPEND PROJECT_LIBS Boost::regex)
endif()

# Link libraries to CoreAI3DLib
target_link_libraries(CoreAI3DLib PUBLIC
    proto_objs
    ${PROJECT_LIBS}
    unofficial::mysql-connector-cpp::connector
    mysqlcppconn
    mysqlcppconnx
    Boost::program_options
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
    absl::log_internal_message
    absl::log_internal_check_op
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
    $<$<TARGET_EXISTS:zstd::libzstd>:zstd::libzstd>
    $<$<TARGET_EXISTS:zstd::libzstd_shared>:zstd::libzstd_shared>
    $<$<TARGET_EXISTS:zstd::libzstd_static>:zstd::libzstd_static>
    $<$<BOOL:${zstd_FOUND}>:zstd::libzstd>
    protobuf::libprotoc
    protobuf::libprotobuf
    protobuf::libprotobuf-lite
    nlohmann_json::nlohmann_json
    google-cloud-cpp::speech
    google-cloud-cpp::common
    ${OpenCV_LIBS}
    resolv
    $<$<TARGET_EXISTS:lz4::lz4>:lz4::lz4>
    $<$<TARGET_EXISTS:lz4::lz4_static>:lz4::lz4_static>
)

# Add MySQL libraries if found
if(TARGET unofficial::mysql-connector-cpp::connector)
    target_link_libraries(CoreAI3DLib PUBLIC unofficial::mysql-connector-cpp::connector)
    message(STATUS "MySQL connector-cpp found and linked")
else()
    message(WARNING "MySQL connector-cpp not found - database functionality may not work")
endif()

add_executable(CoreAI3D main.cpp)

# Include directories for the executable
target_include_directories(CoreAI3D PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}  # Include current directory for main.cpp
    include  # Include directory for your project headers
    ${CMAKE_CURRENT_BINARY_DIR}  # Include generated protobuf headers
)

# Link the executable against CoreAI3DLib
target_link_libraries(CoreAI3D PRIVATE CoreAI3DLib Boost::program_options)

# Add post-build automation script
add_custom_command(TARGET CoreAI3D POST_BUILD
    COMMAND python ${CMAKE_SOURCE_DIR}/post_build_automation.py --source-dir ${CMAKE_SOURCE_DIR} --build-dir ${CMAKE_BINARY_DIR} --target-executable $<TARGET_FILE:CoreAI3D>
    COMMENT "Running post-build automation script"
    VERBATIM
)

# After computing VALID_PROTOBUF_IMPORT_DIRS, print them for diagnostics
message(STATUS "Proto import dirs used for generation: ${VALID_PROTOBUF_IMPORT_DIRS}")

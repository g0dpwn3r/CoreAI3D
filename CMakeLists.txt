# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.20)



# Suppress problematic Windows macros and ensure Winsock 2 is prioritized
if(WIN32)
    add_compile_definitions(
        WIN32_LEAN_AND_MEAN  # Excludes less-used headers from windows.h to speed up compilation and reduce conflicts
        NOMINMAX             # Prevents min/max macros in windows.h from conflicting with std::min/std::max
        _WINSOCKAPI_         # Prevents winsock.h from being included by windows.h, forcing winsock2.h
    )
endif()

if(WIN32)
    set(VCPKG_TARGET_TRIPLET x64-windows)
else()
    set(VCPKG_TARGET_TRIPLET x64-linux)
endif()

if(DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_TOOLCHAIN_FILE}")
    include("${CMAKE_TOOLCHAIN_FILE}")
endif()

message("CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
message("CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("CMAKE_MAKE_PROGRAM: ${CMAKE_MAKE_PROGRAM}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/CoreAI3D")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/CoreAI3D")

if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "D:/code/CoreAI3D/out/build/linux-x64-debug-wsl/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "D:/code/CoreAI3D/out/build/linux-x64-debug-wsl/lib")
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

if (POLICY CMP0167)
	cmake_policy(SET CMP0167 NEW)
endif()
if(WIN32)
    add_definitions(-D_WIN32_WINNT=0x0A00 -DWIN32_LEAN_AND_MEAN -D_WINSOCKAPI_)
else()
    add_definitions(-D_POSIX_C_SOURCE=200809L)
endif()

project ("CoreAI3D")

# Include sub-projects.
add_subdirectory ("CoreAI3D")

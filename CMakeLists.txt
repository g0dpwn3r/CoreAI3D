# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.20)

# Suppress problematic Windows macros and ensure Winsock 2 is prioritized
add_compile_definitions(
    WIN32_LEAN_AND_MEAN  # Excludes less-used headers from windows.h to speed up compilation and reduce conflicts
    NOMINMAX             # Prevents min/max macros in windows.h from conflicting with std::min/std::max
    _WINSOCKAPI_         # Prevents winsock.h from being included by windows.h, forcing winsock2.h
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/CoreAI3D")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/CoreAI3D")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/CoreAI3D")

set(CMAKE_CXX_STANDARD_REQUIRED True)
# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

if (POLICY CMP0167)
	cmake_policy(SET CMP0167 NEW)
endif()




project ("CoreAI3D")

# Include sub-projects.
add_subdirectory ("CoreAI3D")

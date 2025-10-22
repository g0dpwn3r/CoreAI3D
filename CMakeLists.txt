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

if(MSVC)
    add_compile_options(/FS)
endif()

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

# ======================================
#  CPack Configuration for CoreAI3D
# ======================================

include(CPack)

# ---- Basic metadata ----
set(CPACK_PACKAGE_NAME "CoreAI3D")
set(CPACK_PACKAGE_VENDOR "Wizit")
set(CPACK_PACKAGE_CONTACT "Carlon <carlonvanspijker@gmail.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CoreAI3D — Neural Core Engine for C++ AI Training")
set(CPACK_PACKAGE_VERSION "2.1.0")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/g0dpwn3r/CoreAI3D")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

# ---- Installation root ----
set(CPACK_PACKAGE_INSTALL_DIRECTORY "CoreAI3D")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}")

# ---- Include executable and dependencies ----
install(TARGETS CoreAI3D
    RUNTIME DESTINATION bin
)

install(DIRECTORY assets/ DESTINATION share/CoreAI3D/assets OPTIONAL)
install(FILES README.md DESTINATION share/CoreAI3D)

# ======================================
#  Platform-Specific Packaging
# ======================================

# 🐧 Debian / Ubuntu (WSL or Linux)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Carlon")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libstdc++6, libmysqlcppconn8.4.0")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_SECTION "science")
set(CPACK_DEBIAN_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})

# 🪟 Windows NSIS installer
set(CPACK_NSIS_DISPLAY_NAME "CoreAI3D")
set(CPACK_NSIS_PACKAGE_NAME "CoreAI3D Installer")
set(CPACK_NSIS_CONTACT "Carlon <carlonvanspijker@gmail.com>")
set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/g0dpwn3r/CoreAI3D")
set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/assets/icon.ico")
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
set(CPACK_NSIS_MODIFY_PATH ON)

# ---- Multi-generator ----
set(CPACK_GENERATOR "DEB;NSIS")

message(STATUS "CPack configured for generators: ${CPACK_GENERATOR}")
enable_testing()
add_subdirectory(tests)
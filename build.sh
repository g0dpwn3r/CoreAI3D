#!/bin/bash

# CoreAI3D Build Script
# Builds the project using system packages (no vcpkg)

set -e

# Default build type
BUILD_TYPE="Release"
USE_MYSQL="ON"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --release)
      BUILD_TYPE="Release"
      shift
      ;;
    --no-mysql)
      USE_MYSQL="OFF"
      shift
      ;;
    --help)
      echo "Usage: $0 [--debug|--release] [--no-mysql]"
      echo "  --debug    Build in Debug mode (default: Release)"
      echo "  --release  Build in Release mode (default)"
      echo "  --no-mysql Disable MySQL support"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage"
      exit 1
      ;;
  esac
done

echo "Building CoreAI3D in $BUILD_TYPE mode with MySQL=$USE_MYSQL"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Configure with CMake using system packages
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DUSE_MYSQL=$USE_MYSQL \
    -G Ninja

# Build the project
cmake --build . --parallel $(nproc)

echo "Build completed successfully!"
echo "Executable location: $(pwd)/bin/CoreAI3D"
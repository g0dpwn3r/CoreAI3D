# CoreAI3D Build Instructions

This document provides instructions for building CoreAI3D from the command line using system packages (no vcpkg toolchain).

## Prerequisites

Before building, ensure you have the following system packages installed:

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libboost-all-dev \
    libmysqlcppconn-dev \
    libabsl-dev \
    libutf8-range-dev \
    zlib1g-dev \
    libcurl4-openssl-dev \
    libzstd-dev \
    nlohmann-json3-dev \
    libgrpc++-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-cloud-cpp-speech-dev \
    libgoogle-cloud-cpp-common-dev \
    liblz4-dev \
    resolvconf
```

### CentOS/RHEL/Fedora
```bash
# Install dependencies (adjust for your distribution)
sudo dnf install -y \
    gcc-c++ \
    cmake \
    ninja-build \
    pkgconfig \
    boost-devel \
    mysql-connector-c++-devel \
    abseil-cpp-devel \
    utf8cpp-devel \
    zlib-devel \
    libcurl-devel \
    libzstd-devel \
    json-devel \
    grpc-devel \
    protobuf-devel \
    google-cloud-cpp-speech-devel \
    google-cloud-cpp-common-devel \
    lz4-devel
```

## Building

### Using the Build Script (Recommended)

The easiest way to build is using the provided `build.sh` script:

```bash
# Build in release mode (default)
./build.sh

# Build in debug mode
./build.sh --debug

# Build without MySQL support
./build.sh --no-mysql

# Show help
./build.sh --help
```

### Using Make

Alternatively, you can use the provided Makefile:

```bash
# Build in release mode (default)
make

# Build in debug mode
make debug

# Build without MySQL support
make no-mysql

# Clean build directory
make clean

# Show help
make help
```

### Manual CMake Build

If you prefer to build manually:

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MYSQL=ON -G Ninja

# Build
cmake --build . --parallel $(nproc)
```

## Build Options

- `CMAKE_BUILD_TYPE`: Set to `Release` or `Debug`
- `USE_MYSQL`: Set to `ON` or `OFF` to enable/disable MySQL support
- `CMAKE_BUILD_PARALLEL_LEVEL`: Number of parallel build jobs (default: number of CPU cores)

## Output

After successful build, the executable will be located at:
```
build/bin/CoreAI3D
```

## Troubleshooting

### Missing Dependencies

If you encounter missing dependencies, install them using your system's package manager. The CMake configuration will warn about missing optional packages but will continue building with reduced functionality.

### Protobuf Issues

If you have protobuf-related errors, ensure you have both `libprotobuf-dev` and `protobuf-compiler` installed, and that the versions are compatible.

### MySQL Issues

If MySQL support is not working, you can disable it by setting `USE_MYSQL=OFF` during configuration.

## Testing

To run the tests after building:

```bash
cd build
ctest
```

## Packaging

To create packages (deb/rpm):

```bash
cd build
cpack
```

This will create `.deb` packages for Debian/Ubuntu systems and `.rpm` for Red Hat-based systems.
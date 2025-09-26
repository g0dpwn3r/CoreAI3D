FROM ubuntu:latest

# Install build dependencies and runtime libraries
RUN apt update -y && apt upgrade -y && apt dist-upgrade -y && apt install -y \
    build-essential \
    cmake \
    git \
    libboost-all-dev \
    libmysqlcppconn-dev \
    libmysqlcppconn7t64 \
    libcurl4-openssl-dev \
    libssl-dev \
    wget \
    zlib1g \
    nlohmann-json3-dev \
    pkg-config \
    && apt clean

WORKDIR /app

# Install vcpkg
RUN git clone https://github.com/g0dpwn3r/vcpkg && \
    cd vcpkg && \
    ./bootstrap-vcpkg.sh && \
    ./vcpkg integrate install

# Copy source code
COPY . /app

# Build the application
RUN mkdir -p build && \
    cd build && \
    cmake -DCMAKE_TOOLCHAIN_FILE=/app/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release .. && \
    cmake --build . --config Release

# Make executable
RUN chmod +x /app/build/CoreAI3D

# Default command
CMD ["/app/build/CoreAI3D"]
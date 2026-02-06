FROM ubuntu:latest

# Install build dependencies and runtime libraries
RUN apt update -y && apt upgrade -y && apt dist-upgrade -y && apt install -y \
    build-essential \
    cmake \
    curl \
    git \
    git-lfs \
    libboost-all-dev \
    libmysqlcppconn-dev \
    libmysqlcppconn7t64 \
    libcurl4-openssl-dev \
    libssl-dev \
    pkg-config \
    unzip \
    wget \
    zip \
    zlib1g \
    nlohmann-json3-dev \
    && apt clean

WORKDIR /app

# Initialize git-lfs
RUN git lfs install

# Clone vcpkg and CoreAI3D repositories
RUN git clone https://github.com/g0dpwn3r/vcpkg.git
RUN git clone -b v0.8 https://github.com/g0dpwn3r/CoreAI3D.git

# Initialize and update git-lfs in CoreAI3D repository
RUN cd CoreAI3D && git lfs pull

# Initialize and update vcpkg
RUN cd vcpkg && ./bootstrap-vcpkg.sh && ./vcpkg integrate install

# Create build directory and configure CMake for release build
RUN mkdir -p CoreAI3D/build && cd CoreAI3D/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/app/vcpkg/scripts/buildsystems/vcpkg.cmake

# Build the release version
RUN cd CoreAI3D/build && make -j$(nproc)

# Default command
CMD ["/app/CoreAI3D/build/CoreAI3D"]
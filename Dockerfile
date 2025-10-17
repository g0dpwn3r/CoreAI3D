FROM ubuntu:latest

# Install build dependencies and runtime libraries
RUN apt update -y && apt upgrade -y && apt dist-upgrade -y && apt install -y \
    build-essential \
    cmake \
    curl \
    git \
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

# Install vcpkg
RUN git clone https://github.com/g0dpwn3r/vcpkg && \
    cd vcpkg && \
    ./bootstrap-vcpkg.sh && \
    ./vcpkg integrate install

# Default command
CMD ["/app/build/CoreAI3D"]
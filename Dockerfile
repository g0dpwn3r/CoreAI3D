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
COPY out/buid/linux-release/CoreAI3D/* /app/build/CoreAI3D

# Default command
CMD ["/app/build/CoreAI3D"]
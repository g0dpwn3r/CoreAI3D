FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git libmysqlcppconn-dev mysql-client \
    && rm -rf /var/lib/apt/lists/*

# Create app dir
WORKDIR /app

# Copy source (assumed to be copied in docker-compose)
COPY . .

# Build
RUN cmake -S . -B build && cmake --build build

# Entry point
CMD ["./build/CoreAI3D"]

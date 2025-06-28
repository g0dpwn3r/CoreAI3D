FROM ubuntu:22.04

# Install required runtime dependencies
RUN apt update && apt install -y \
    libboost-system-dev \
    libmysqlcppconn-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    zlib1g \
    && apt clean

WORKDIR /app
COPY CoreAI3D .

# Default command can be empty or help-related
CMD ["./CoreAI3D"]
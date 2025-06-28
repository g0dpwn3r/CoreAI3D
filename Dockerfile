FROM ubuntu:latest

RUN apt update -y && apt upgrade -y && apt dist-upgrade -y && apt install -y \
    libboost-system-dev \
    libmysqlcppconn-dev \
    libmysqlcppconn7t64 \
    libcurl4-openssl-dev \
    libssl-dev \
    wget \
    zlib1g \
    && apt clean

WORKDIR /app
#install mysql-connector-cpp-x
RUN wget -O libmysqlcppconx.deb https://dev.mysql.com/get/Downloads/Connector-C++/libmysqlcppconnx2_9.3.0-1ubuntu25.04_amd64.deb && dpkg -i libmysqlcppconx.deb
#Change to release
COPY out/build/linux-x64-debug-wsl/CoreAI3D/CoreAI3D /app/CoreAI3D
RUN chmod +x /app/CoreAI3D


# Default command can be empty or help-related
CMD ["./CoreAI3D"]
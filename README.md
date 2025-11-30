# CoreAI3D: Multimodal AI Platform

CoreAI3D is a powerful multimodal AI platform that combines computer vision, natural language processing, audio processing, and neural network capabilities into a single, easy-to-use system. Whether you're building intelligent applications, analyzing data, or creating interactive AI experiences, CoreAI3D provides the tools you need to harness the power of machine learning across multiple modalities.

## Key Features

- **Multimodal AI Capabilities**: Seamlessly integrate vision, audio, language, and neural network processing
- **Computer Vision**: Object detection, image classification, OCR, facial recognition, and medical imaging analysis
- **Natural Language Processing**: Text encoding, chat interfaces, and language detection with embedding support
- **Audio Processing**: Speech recognition, audio analysis, synthesis, music analysis, and speaker identification
- **Neural Networks**: Customizable neural network training and prediction with flexible architecture
- **Advanced Activation Functions**: Traditional activations plus calculus-based functions using numerical integration
- **Multiple Interfaces**: Command-line interface, REST API server, and graphical user interfaces
- **Database Integration**: MySQL support for data persistence and model storage
- **Cross-Platform**: Built with C++ for Windows and Linux compatibility
- **Extensible Architecture**: Modular design for easy integration and customization

## How It Works

CoreAI3D operates through specialized modules that handle different AI tasks:

- **Vision Module**: Processes images and videos using neural networks for tasks like object detection and classification
- **Audio Module**: Analyzes and synthesizes audio, including speech recognition and music analysis
- **Language Module**: Handles text processing with word embeddings and conversational AI capabilities
- **Core AI Engine**: Provides the underlying neural network infrastructure for training and inference
- **API Server**: Exposes functionality through HTTP endpoints for easy integration
- **Database Layer**: Manages data storage and retrieval for models and results

The system uses a modular orchestrator to coordinate between different AI capabilities, allowing you to combine vision, audio, and language processing in powerful ways.

## Calculus-Based Activation Functions

CoreAI3D features advanced calculus-based activation functions that use numerical integration for more sophisticated neural network behavior:

### Available Calculus Activations

- **Integral Sigmoid**: ∫ sigmoid(t) dt from 0 to x - Provides smooth, integrated activation
- **Integral ReLU**: ∫ max(0, t) dt from 0 to x - Creates quadratic activation patterns
- **Integral Tanh**: ∫ tanh(t) dt from 0 to x - Offers hyperbolic tangent integration

### Configuration Parameters

Each calculus activation can be configured with:
- **N**: Number of integration points (default: 10, range: 1-1000)
- **SM**: Simpson's method multiplier (default: 1.0)
- **a**: Alpha scaling parameter (default: 1.0)

### API Configuration

Configure calculus activations via REST API:

```bash
# Set calculus activation for a specific layer
curl -X POST http://localhost:8080/api/neural/set_calculus_config \
  -H "Content-Type: application/json" \
  -d '{
    "layer_index": 1,
    "N": 20,
    "SM": 1.5,
    "a": 2.0
  }'

# Set calculus activation for all layers
curl -X POST http://localhost:8080/api/neural/set_all_calculus_activations \
  -H "Content-Type: application/json" \
  -d '{
    "activation_type": "INTEGRAL_SIGMOID",
    "N": 15,
    "SM": 1.0,
    "a": 1.5
  }'

# Test calculus activation function
curl -X POST http://localhost:8080/api/neural/test_calculus_activation \
  -H "Content-Type: application/json" \
  -d '{
    "input": 1.5,
    "activation_type": "INTEGRAL_SIGMOID",
    "N": 10,
    "SM": 1.0,
    "a": 1.0
  }'
```

### Benefits

- **Smoother Gradients**: Integrated functions provide more stable gradient flow
- **Complex Patterns**: Quadratic and integrated behaviors enable richer representations
- **Numerical Precision**: Configurable integration points balance accuracy and performance
- **Research Applications**: Enables exploration of advanced activation function research

## Getting Started

### Prerequisites

#### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+, Fedora 30+) or Windows (Visual Studio 2022) or WSL
- **Build Tools**: CMake 3.20+ with Ninja build system, C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2022+)
- **Memory**: Minimum 4GB RAM, 8GB recommended for compilation
- **Storage**: 2GB free space for build artifacts and dependencies

#### Dependencies

CoreAI3D uses system packages for optimal performance and compatibility. Install the following dependencies based on your operating system:

**Ubuntu/Debian:**
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
    resolvconf \
    python3 \
    python3-pip
```

**CentOS/RHEL/Fedora:**
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
    lz4-devel \
    python3 \
    python3-pip
```

**Windows (Visual Studio 2022):**
- Install Visual Studio 2022 with C++ development workload
- Install CMake from https://cmake.org/download/
- Use vcpkg for dependency management (recommended):
  ```powershell
  git clone https://github.com/Microsoft/vcpkg.git
  cd vcpkg
  .\bootstrap-vcpkg.bat
  .\vcpkg integrate install
  .\vcpkg install boost mysql-connector-cpp absl zlib curl zstd nlohmann-json grpc protobuf
  ```

#### Optional Dependencies
- **MySQL Server**: For data persistence and model storage
  - Ubuntu: `sudo apt install mysql-server`
  - CentOS: `sudo dnf install mysql-server`
  - Windows: Download from https://dev.mysql.com/downloads/mysql/
- **Docker**: For containerized deployment
- **Python 3.8+**: For build automation and testing scripts

### Installation and Build

#### 1. Clone the Repository
```bash
git clone https://github.com/g0dpwn3r/CoreAI3D.git
cd CoreAI3D
```

#### 2. Build Methods

CoreAI3D supports multiple build methods. Choose the one that best fits your workflow:

##### Method 1: Automated Build Script (Recommended)
The easiest way to build CoreAI3D is using the provided `build.sh` script:

```bash
# Build in release mode (default)
./build.sh

# Build in debug mode
./build.sh --debug

# Build without MySQL support
./build.sh --no-mysql

# Build with post-build automation
./build.sh --post

# Clean build artifacts
./build.sh --clean

# Show help
./build.sh --help
```

##### Method 2: Manual CMake Build
For more control over the build process:

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MYSQL=ON -G Ninja

# Build the project
cmake --build . --parallel $(nproc)

# Optional: Run tests
ctest

# Optional: Create packages
cpack
```

##### Method 3: Windows with Visual Studio
```powershell
# Configure for Visual Studio 2022
cmake -S . -B out/build/vs -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake

# Build
cmake --build out/build/vs --config Release
```

##### Method 4: Docker Build
```bash
# Build Docker image
docker build -t coreai3d:latest .

# Run in container
docker run -p 8080:8080 coreai3d:latest
```

#### 3. Build Options

- `CMAKE_BUILD_TYPE`: `Release` (optimized) or `Debug` (with debug symbols)
- `USE_MYSQL`: `ON` (enable database support) or `OFF` (disable)
- `CMAKE_BUILD_PARALLEL_LEVEL`: Number of parallel jobs (default: CPU count)
- `BUILD_TESTING`: `ON` to build and run tests

#### 4. Post-Build Setup

After successful compilation, the executable will be located at:
- Linux: `build/bin/CoreAI3D` or `out/build/linux-release/CoreAI3D/CoreAI3D`
- Windows: `out/build/vs/Release/CoreAI3D.exe`

The build process automatically copies `config.json` to the executable directory.

#### 5. Optional: MySQL Database Setup
If you want to use database features:

1. **Install MySQL Server**:
   - Ubuntu: `sudo apt install mysql-server`
   - CentOS: `sudo dnf install mysql-server`
   - Windows: Download from MySQL website

2. **Create Database**:
   ```sql
   CREATE DATABASE coreai3d;
   CREATE USER 'coreai3d'@'localhost' IDENTIFIED BY 'your_password';
   GRANT ALL PRIVILEGES ON coreai3d.* TO 'coreai3d'@'localhost';
   FLUSH PRIVILEGES;
   ```

3. **Configure Connection**: Edit `config.json` with your database credentials (see Configuration section below)

## Configuration

CoreAI3D uses a JSON configuration file (`config.json`) to customize various aspects of the system. The configuration file is automatically copied to the build directory during compilation.

### Configuration File Structure

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "threads": 4,
    "api_version": "v1",
    "cors_origin": "*",
    "request_timeout": 30,
    "max_request_size": 1048576,
    "enable_logging": true,
    "log_file": "api_server.log"
  },
  "database": {
    "host": "localhost",
    "port": 33060,
    "user": "root",
    "password": "your_password",
    "schema": "coreai3d",
    "ssl_mode": "DISABLED",
    "create_tables": false
  },
  "math": {
    "precision": 1e-10,
    "max_iterations": 1000,
    "convergence_threshold": 1e-8,
    "learning_rate": 0.01,
    "batch_size": 32,
    "epochs": 100,
    "checkpointing": {
      "enabled": true,
      "checkpoint_frequency_epochs": 10,
      "best_model_tracking": {
        "enabled": true,
        "patience_epochs": 20,
        "min_improvement": 1e-4
      },
      "checkpoint_directory": "checkpoints",
      "checkpoint_prefix": "model_checkpoint",
      "best_model_prefix": "best_model",
      "max_checkpoints_to_keep": 5,
      "save_optimizer_state": true,
      "save_training_metadata": true
    },
    "constants": {
      "pi": 3.14159265358979323846,
      "e": 2.71828182845904523536,
      "golden_ratio": 1.618033988749895
    },
    "algorithms": {
      "optimization": "gradient_descent",
      "integration": "trapezoidal",
      "differentiation": "finite_difference",
      "random_distribution": "uniform"
    },
    "precision_settings": {
      "numerical_derivative_step": 1e-5,
      "integration_steps": 1000,
      "random_seed": 42
    }
  }
}
```

### Server Configuration

- `host`: IP address to bind the API server (default: "0.0.0.0" for all interfaces)
- `port`: Port number for the API server (default: 8080)
- `threads`: Number of worker threads for handling requests (default: 4)
- `api_version`: API version prefix (default: "v1")
- `cors_origin`: CORS allowed origin (default: "*" for all)
- `request_timeout`: Request timeout in seconds (default: 30)
- `max_request_size`: Maximum request body size in bytes (default: 1MB)
- `enable_logging`: Enable/disable request logging (default: true)
- `log_file`: Path to log file (default: "api_server.log")

### Database Configuration

- `host`: MySQL server hostname (default: "localhost")
- `port`: MySQL server port (default: 33060)
- `user`: MySQL username
- `password`: MySQL password
- `schema`: Database name (default: "coreai3d")
- `ssl_mode`: SSL connection mode - "DISABLED", "PREFERRED", or "REQUIRED"
- `create_tables`: Auto-create database tables on startup (default: false)

### Math/Training Configuration

- `precision`: Numerical precision threshold (default: 1e-10)
- `max_iterations`: Maximum iterations for algorithms (default: 1000)
- `convergence_threshold`: Training convergence threshold (default: 1e-8)
- `learning_rate`: Default learning rate for training (default: 0.01)
- `batch_size`: Training batch size (default: 32)
- `epochs`: Default number of training epochs (default: 100)

#### Checkpointing

- `enabled`: Enable model checkpointing (default: true)
- `checkpoint_frequency_epochs`: Save checkpoint every N epochs (default: 10)
- `best_model_tracking`: Track and save best performing model
  - `enabled`: Enable best model tracking (default: true)
  - `patience_epochs`: Stop if no improvement for N epochs (default: 20)
  - `min_improvement`: Minimum improvement threshold (default: 1e-4)
- `checkpoint_directory`: Directory for checkpoints (default: "checkpoints")
- `max_checkpoints_to_keep`: Maximum number of checkpoints to retain (default: 5)

### Runtime Configuration Override

You can override configuration values using command-line arguments or environment variables:

```bash
# Override server port
./CoreAI3D --api-port 9090

# Override database settings
./CoreAI3D --db-host mysql.example.com --db-user myuser --db-password mypass

# Override training parameters
./CoreAI3D --learning-rate 0.001 --batch-size 64 --epochs 200
```

## Usage

CoreAI3D supports multiple interaction modes to fit your workflow. Below are detailed usage examples for each mode.

### Command Line Interface

#### Basic Usage
```bash
# Show help
./CoreAI3D --help

# Show version
./CoreAI3D --version
```

#### Chat Mode
Start an interactive AI chat session with various configuration options:
```bash
# Basic chat with English language support
./CoreAI3D --start-chat --language en

# Chat with custom embeddings and database
./CoreAI3D --start-chat \
  --language en \
  --embedding-file en_embeddings.csv \
  --db-host localhost \
  --db-user your_user \
  --db-password your_password \
  --db-schema coreai3d

# Chat with calculus activation functions
./CoreAI3D --start-chat \
  --calculus-activation INTEGRAL_SIGMOID \
  --calculus-n 20 \
  --calculus-sm 1.5 \
  --calculus-a 2.0
```

#### Prediction Mode
Train neural networks and make predictions on CSV data:
```bash
# Basic prediction with default settings
./CoreAI3D --start-predict \
  --input-file data.csv \
  --output-csv predictions.csv

# Advanced training configuration
./CoreAI3D --start-predict \
  --input-file training_data.csv \
  --output-csv predictions.csv \
  --layers 3 \
  --neurons 64 \
  --epochs 500 \
  --learning-rate 0.001 \
  --batch-size 32 \
  --activation relu \
  --optimizer adam \
  --loss mse

# Use calculus-based activation functions
./CoreAI3D --start-predict \
  --input-file data.csv \
  --output-csv predictions.csv \
  --calculus-activation INTEGRAL_TANH \
  --calculus-n 50 \
  --calculus-sm 2.0 \
  --calculus-a 1.5

# Load and use a pre-trained model
./CoreAI3D --start-predict \
  --input-file test_data.csv \
  --output-csv results.csv \
  --load-model saved_model.json
```

#### Training Mode
Train models without immediate prediction:
```bash
# Train a model and save it
./CoreAI3D --start-train \
  --input-file training_data.csv \
  --model-output trained_model.json \
  --layers 4 \
  --neurons 128 \
  --epochs 1000 \
  --learning-rate 0.0001 \
  --validation-split 0.2 \
  --early-stopping \
  --checkpoint-dir checkpoints/
```

#### API Server Mode
Start the HTTP REST API server:
```bash
# Start with default settings (port 8080)
./CoreAI3D --api-port 8080

# Start with custom configuration
./CoreAI3D --api-port 9090 \
  --api-threads 8 \
  --api-timeout 60 \
  --cors-origin "https://yourdomain.com"

# Start with SSL/TLS (if configured)
./CoreAI3D --api-port 8443 \
  --ssl-cert server.crt \
  --ssl-key server.key
```

### REST API Usage

#### Neural Network Operations
```bash
# Train a model via API
curl -X POST http://localhost:8080/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1,2,3], [4,5,6], [7,8,9]],
    "targets": [0, 1, 0],
    "layers": 2,
    "neurons": 16,
    "epochs": 100,
    "learning_rate": 0.01
  }'

# Make predictions
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "input": [1.5, 2.3, 3.1],
    "model_id": "optional_model_id"
  }'

# Get model information
curl http://localhost:8080/api/models
```

#### Calculus Activation Configuration
```bash
# Configure calculus activation for specific layer
curl -X POST http://localhost:8080/api/neural/set_calculus_config \
  -H "Content-Type: application/json" \
  -d '{
    "layer_index": 1,
    "activation_type": "INTEGRAL_SIGMOID",
    "N": 25,
    "SM": 1.5,
    "a": 2.0
  }'

# Set calculus activation for all layers
curl -X POST http://localhost:8080/api/neural/set_all_calculus_activations \
  -H "Content-Type: application/json" \
  -d '{
    "activation_type": "INTEGRAL_RELU",
    "N": 30,
    "SM": 2.0,
    "a": 1.0
  }'

# Test calculus activation function
curl -X POST http://localhost:8080/api/neural/test_calculus_activation \
  -H "Content-Type: application/json" \
  -d '{
    "input": 1.5,
    "activation_type": "INTEGRAL_SIGMOID",
    "N": 20,
    "SM": 1.0,
    "a": 1.0
  }'
```

#### Vision, Audio, and Language APIs
```bash
# Image classification
curl -X POST http://localhost:8080/api/vision/classify \
  -F "image=@photo.jpg"

# Speech recognition
curl -X POST http://localhost:8080/api/audio/recognize \
  -F "audio=@speech.wav"

# Text analysis
curl -X POST http://localhost:8080/api/language/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "language": "en"}'
```

### Python Client
Use the included Python client for programmatic access:
```python
from coreai3d_client import CoreAI3DClient

# Initialize client
client = CoreAI3DClient('http://localhost:8080')

# Train a model
training_data = [[1, 2], [3, 4], [5, 6]]
targets = [0, 1, 0]
model_id = client.train(training_data, targets, layers=2, neurons=16, epochs=100)

# Make predictions
predictions = client.predict([[7, 8], [9, 10]], model_id=model_id)
print(predictions)

# Chat interaction
response = client.chat("Hello, how are you?")
print(response)

# Configure calculus activations
client.set_calculus_config(layer_index=0,
                          activation_type="INTEGRAL_SIGMOID",
                          N=20, SM=1.5, a=2.0)
```

### Docker Usage
Run CoreAI3D in a containerized environment:
```bash
# Build the image
docker build -t coreai3d:latest .

# Run with default settings
docker run -p 8080:8080 coreai3d:latest

# Run with custom configuration
docker run -p 8080:8080 \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/data:/app/data \
  coreai3d:latest

# Run with database access
docker run -p 8080:8080 \
  --network host \
  -e DB_HOST=host.docker.internal \
  coreai3d:latest
```

### Web Interface
Access the web-based GUI:
```bash
# For development (if web interface exists)
python -m http.server 3000
# Then open http://localhost:3000 in browser

# Windows dashboard (if available)
python windows_gui/run_dashboard.py
```

## API Reference

CoreAI3D provides a REST API for programmatic access to AI capabilities:

### Core Endpoints

- `POST /api/predict` - Neural network prediction
- `POST /api/train` - Train models on data
- `POST /api/chat` - Interactive chat with AI
- `GET /api/models` - List available models

### Neural Network Configuration Endpoints

- `POST /api/neural/set_calculus_config` - Configure calculus activation for specific layer
- `GET /api/neural/get_calculus_config` - Get calculus configuration for specific layer
- `POST /api/neural/set_all_calculus_activations` - Set calculus activation for all layers
- `POST /api/neural/test_calculus_activation` - Test calculus activation function output

### Vision Endpoints

- `POST /api/vision/detect` - Object detection in images
- `POST /api/vision/classify` - Image classification
- `POST /api/vision/ocr` - Optical character recognition

### Audio Endpoints

- `POST /api/audio/recognize` - Speech-to-text
- `POST /api/audio/analyze` - Audio analysis
- `POST /api/audio/synthesize` - Text-to-speech

### Language Endpoints

- `POST /api/language/encode` - Text encoding with embeddings
- `POST /api/language/detect` - Language detection

For detailed API documentation, see the [API Documentation](api-docs.md) or explore the endpoints interactively when the server is running.

## Troubleshooting

### Build Issues

#### CMake Configuration Errors
**Problem**: CMake fails to find dependencies
```bash
CMake Error: Could not find package Boost
```

**Solutions**:
- Ensure all system packages are installed (see Prerequisites section)
- On Ubuntu/Debian: `sudo apt install libboost-all-dev`
- On CentOS/RHEL: `sudo dnf install boost-devel`
- Clear CMake cache: `rm -rf build/CMakeCache.txt`

#### Protobuf Compilation Errors
**Problem**: Protocol buffer generation fails
```bash
protoc: command not found
```

**Solutions**:
- Install protobuf compiler: `sudo apt install protobuf-compiler`
- Ensure protobuf version compatibility (3.6+ required)
- Check protobuf include paths in CMake output

#### MySQL Connection Issues
**Problem**: MySQL support disabled despite installation
```bash
MySQL Connector/C++ not found - MySQL support disabled
```

**Solutions**:
- Install MySQL connector: `sudo apt install libmysqlcppconn-dev`
- Verify pkg-config can find it: `pkg-config --libs libmysqlcppconn`
- Build with MySQL enabled: `cmake -DUSE_MYSQL=ON`

#### Linker Errors
**Problem**: Undefined reference errors during linking
```bash
undefined reference to `boost::system::error_code::error_code()'
```

**Solutions**:
- Ensure all Boost components are installed
- Check library search paths: `ldconfig -p | grep boost`
- Rebuild with correct library paths

### Runtime Issues

#### API Server Won't Start
**Problem**: Server fails to bind to port
```bash
Failed to bind to port 8080: Address already in use
```

**Solutions**:
- Check if port is in use: `netstat -tlnp | grep 8080`
- Kill conflicting process: `sudo kill -9 <PID>`
- Use different port: `./CoreAI3D --api-port 9090`

#### Database Connection Failures
**Problem**: Cannot connect to MySQL database
```bash
Database connection failed: Access denied for user
```

**Solutions**:
- Verify database credentials in `config.json`
- Ensure MySQL server is running: `sudo systemctl status mysql`
- Create database and user:
  ```sql
  CREATE DATABASE coreai3d;
  CREATE USER 'coreai3d'@'localhost' IDENTIFIED BY 'password';
  GRANT ALL PRIVILEGES ON coreai3d.* TO 'coreai3d'@'localhost';
  ```

#### Memory Issues During Training
**Problem**: Out of memory errors during model training
```bash
std::bad_alloc: out of memory
```

**Solutions**:
- Reduce batch size: `--batch-size 16`
- Decrease model complexity: `--layers 2 --neurons 32`
- Use smaller datasets or data augmentation
- Monitor memory usage: `htop` or `free -h`

#### Calculus Activation Performance
**Problem**: Calculus activations are slow
```bash
Training taking too long with calculus activations
```

**Solutions**:
- Reduce integration points: `--calculus-n 10`
- Use simpler activation: `--calculus-activation INTEGRAL_SIGMOID`
- Consider standard activations for large models

### Docker Issues

#### Container Build Failures
**Problem**: Docker build fails with missing dependencies
```bash
E: Unable to locate package libboost-all-dev
```

**Solutions**:
- Update Docker image base: Use `ubuntu:22.04` instead of older versions
- Check network connectivity during build
- Clear Docker cache: `docker system prune -a`

#### Volume Mount Permissions
**Problem**: Permission denied on mounted volumes
```bash
Permission denied accessing /app/data
```

**Solutions**:
- Set correct user permissions: `chmod 755 data/`
- Use `--user $(id -u):$(id -g)` in docker run
- Mount volumes with correct ownership

### Performance Optimization

#### Slow Compilation
- Use Ninja build system: `cmake -G Ninja`
- Enable parallel builds: `cmake --build . --parallel $(nproc)`
- Use ccache for faster recompilation: `sudo apt install ccache`

#### Slow Training
- Use GPU acceleration (if available)
- Optimize batch size and learning rate
- Enable checkpointing to resume training
- Use pre-compiled headers (enabled by default)

#### High Memory Usage
- Reduce model size: fewer layers/neurons
- Use smaller data types where possible
- Implement data streaming for large datasets
- Monitor with `valgrind --tool=massif`

### Getting Help

If you encounter issues not covered here:
1. Check the [Issues](https://github.com/g0dpwn3r/CoreAI3D/issues) page for similar problems
2. Enable verbose logging: `cmake -DCMAKE_VERBOSE_MAKEFILE=ON`
3. Run diagnostics: `./CoreAI3D --diagnostic`
4. Provide system information and error logs when reporting issues

## Contributing

We welcome contributions to CoreAI3D! Here's how you can help:

1. **Report Issues**: Found a bug? Open an issue on GitHub
2. **Feature Requests**: Have an idea? Let us know in the issues
3. **Code Contributions**: Fork the repo, make changes, and submit a pull request
4. **Documentation**: Help improve documentation and examples

### Development Setup

1. Follow the installation steps above
2. Enable testing: `cmake -S . -B out/build -DBUILD_TESTING=ON`
3. Run tests: `ctest --test-dir out/build -C Debug -V`

## License

CoreAI3D is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

**Built with ❤️ using C++20, CMake, and modern AI techniques**

*Experience the future of multimodal AI with CoreAI3D - where vision, voice, and language come together seamlessly.*

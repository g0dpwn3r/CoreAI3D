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

- **Operating System**: Windows (Visual Studio 2022) or Linux/WSL
- **Build Tools**: CMake 3.20+, C++20 compiler
- **Dependencies**: vcpkg for dependency management (recommended)
- **Database**: MySQL server (optional, for data persistence)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/g0dpwn3r/CoreAI3D.git
   cd CoreAI3D
   ```

2. **Install dependencies with vcpkg**:
   ```bash
   # Set VCPKG_ROOT environment variable or place vcpkg in repo root
   cmake -S . -B out/build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
   ```

3. **Build the project**:

   **Linux/WSL**:
   ```bash
   mkdir -p out/build/linux
   cd out/build/linux
   cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../.. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
   cmake --build . --config Release
   ```

   **Windows (Visual Studio)**:
   ```powershell
   cmake -S . -B out/build/vs -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
   cmake --build out/build/vs --config Release
   ```

4. **Optional: Set up MySQL database**:
   - Install MySQL server
   - Create a database for CoreAI3D
   - Configure connection settings in your application

## Usage

CoreAI3D supports multiple interaction modes to fit your workflow:

### Command Line Interface

#### Chat Mode
Start an interactive AI chat session:
```bash
./CoreAI3D --start-chat --language en --embedding-file en_embeddings.csv --db-host localhost --db-user your_user --db-password your_password
```

#### Prediction Mode
Train and predict on CSV data:
```bash
./CoreAI3D --start-predict --input-file data.csv --output-csv predictions.csv --layers 3 --neurons 25 --epochs 100 --learning-rate 0.01
```

#### API Server Mode
Start the HTTP API server:
```bash
./CoreAI3D --api-port 8080
```

### Python Client
Use the included Python client for easy integration:
```python
from python.coreai3d_client import CoreAI3DClient

client = CoreAI3DClient('http://localhost:8080')
result = client.predict(data)
```

### Web Interface
Access the web-based GUI by opening `web/gui.jsx` in a modern browser or use the Windows dashboard:
```bash
python windows_gui/run_dashboard.py
```

### Docker
Run CoreAI3D in a containerized environment:
```bash
docker build -t coreai3d:latest .
docker run -p 8080:8080 coreai3d:latest
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

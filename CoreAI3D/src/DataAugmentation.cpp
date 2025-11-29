#include "DataAugmentation.hpp"
#include <algorithm>
#include <iostream>

// Default constructor
DataAugmentation::DataAugmentation()
    : gen_(std::random_device{}()),
      noiseDist_(0.0f, 0.01f), uniformDist_(0.0f, 1.0f) {
}

// Constructor with config
DataAugmentation::DataAugmentation(const AugmentationConfig& config)
    : config_(config), gen_(std::random_device{}()),
      noiseDist_(0.0f, config.noiseStdDev),
      uniformDist_(0.0f, 1.0f) {
}

std::vector<float> DataAugmentation::augmentCoordinates(const std::vector<float>& coordinates) {
    if (coordinates.size() % 3 != 0) {
        std::cerr << "Warning: Coordinate data size (" << coordinates.size()
                  << ") is not divisible by 3. Assuming incomplete coordinate triplets." << std::endl;
    }

    std::vector<float> augmented = coordinates;

    // Apply augmentations in sequence
    if (config_.enableRotation) {
        augmented = applyRotation(augmented, config_.rotationMaxAngle);
    }

    if (config_.enableScaling) {
        augmented = applyScaling(augmented, config_.scalingMin, config_.scalingMax);
    }

    if (config_.enableNoise) {
        augmented = applyNoise(augmented, config_.noiseStdDev);
    }

    if (config_.enableSymmetry) {
        augmented = applySymmetry(augmented, config_.symmetryProbability);
    }

    return augmented;
}

std::vector<float> DataAugmentation::applyRotation(const std::vector<float>& coordinates, float maxAngle) {
    // Generate random rotation angles
    float angleX = uniformDist_(gen_) * 2.0f * maxAngle - maxAngle;
    float angleY = uniformDist_(gen_) * 2.0f * maxAngle - maxAngle;
    float angleZ = uniformDist_(gen_) * 2.0f * maxAngle - maxAngle;

    auto rotationMatrix = createRotationMatrix(angleX, angleY, angleZ);
    return applyTransformationMatrix(coordinates, rotationMatrix);
}

std::vector<float> DataAugmentation::applyScaling(const std::vector<float>& coordinates, float minScale, float maxScale) {
    // Generate random scaling factors for each axis
    float scaleX = minScale + uniformDist_(gen_) * (maxScale - minScale);
    float scaleY = minScale + uniformDist_(gen_) * (maxScale - minScale);
    float scaleZ = minScale + uniformDist_(gen_) * (maxScale - minScale);

    auto scalingMatrix = createScalingMatrix(scaleX, scaleY, scaleZ);
    return applyTransformationMatrix(coordinates, scalingMatrix);
}

std::vector<float> DataAugmentation::applyNoise(const std::vector<float>& coordinates, float stdDev) {
    std::vector<float> noisy = coordinates;

    // Update noise distribution with current stdDev
    noiseDist_ = std::normal_distribution<float>(0.0f, stdDev);

    // Add Gaussian noise to each coordinate
    for (size_t i = 0; i < noisy.size(); ++i) {
        noisy[i] += noiseDist_(gen_);
    }

    return noisy;
}

std::vector<float> DataAugmentation::applySymmetry(const std::vector<float>& coordinates, float probability) {
    std::vector<float> symmetric = coordinates;

    // Randomly decide whether to apply symmetry
    if (uniformDist_(gen_) < probability) {
        // Choose random axis for reflection (0=X, 1=Y, 2=Z)
        int axis = static_cast<int>(uniformDist_(gen_) * 3.0f);
        auto reflectionMatrix = createReflectionMatrix(axis);
        symmetric = applyTransformationMatrix(symmetric, reflectionMatrix);
    }

    return symmetric;
}

std::vector<float> DataAugmentation::applyTransformationMatrix(const std::vector<float>& coordinates,
                                                              const std::vector<std::vector<float>>& matrix) {
    std::vector<float> transformed;

    // Process coordinates in triplets (x, y, z)
    for (size_t i = 0; i < coordinates.size(); i += 3) {
        std::vector<float> point = {coordinates[i],
                                   (i + 1 < coordinates.size()) ? coordinates[i + 1] : 0.0f,
                                   (i + 2 < coordinates.size()) ? coordinates[i + 2] : 0.0f};

        auto transformedPoint = matrixVectorMultiply(matrix, point);

        transformed.insert(transformed.end(), transformedPoint.begin(), transformedPoint.end());
    }

    return transformed;
}

// Configuration methods
void DataAugmentation::setConfig(const AugmentationConfig& config) {
    config_ = config;
    noiseDist_ = std::normal_distribution<float>(0.0f, config.noiseStdDev);
}

const DataAugmentation::AugmentationConfig& DataAugmentation::getConfig() const {
    return config_;
}

void DataAugmentation::enableRotation(bool enable) {
    config_.enableRotation = enable;
}

void DataAugmentation::enableScaling(bool enable) {
    config_.enableScaling = enable;
}

void DataAugmentation::enableNoise(bool enable) {
    config_.enableNoise = enable;
}

void DataAugmentation::enableSymmetry(bool enable) {
    config_.enableSymmetry = enable;
}

void DataAugmentation::setRotationMaxAngle(float angle) {
    config_.rotationMaxAngle = angle;
}

void DataAugmentation::setScalingRange(float minScale, float maxScale) {
    config_.scalingMin = minScale;
    config_.scalingMax = maxScale;
}

void DataAugmentation::setNoiseStdDev(float stdDev) {
    config_.noiseStdDev = stdDev;
    noiseDist_ = std::normal_distribution<float>(0.0f, stdDev);
}

void DataAugmentation::setSymmetryProbability(float probability) {
    config_.symmetryProbability = probability;
}

// Matrix creation methods
std::vector<std::vector<float>> DataAugmentation::createRotationMatrix(float angleX, float angleY, float angleZ) {
    // Convert angles to radians
    float radX = degreesToRadians(angleX);
    float radY = degreesToRadians(angleY);
    float radZ = degreesToRadians(angleZ);

    // Rotation matrices for each axis
    auto rotX = createIdentityMatrix();
    auto rotY = createIdentityMatrix();
    auto rotZ = createIdentityMatrix();

    // X-axis rotation
    rotX[1][1] = cos(radX); rotX[1][2] = -sin(radX);
    rotX[2][1] = sin(radX); rotX[2][2] = cos(radX);

    // Y-axis rotation
    rotY[0][0] = cos(radY); rotY[0][2] = sin(radY);
    rotY[2][0] = -sin(radY); rotY[2][2] = cos(radY);

    // Z-axis rotation
    rotZ[0][0] = cos(radZ); rotZ[0][1] = -sin(radZ);
    rotZ[1][0] = sin(radZ); rotZ[1][1] = cos(radZ);

    // Combine rotations: R = Rz * Ry * Rx
    auto temp = matrixMultiply(rotY, rotX);
    return matrixMultiply(rotZ, temp);
}

std::vector<std::vector<float>> DataAugmentation::createScalingMatrix(float scaleX, float scaleY, float scaleZ) {
    auto matrix = createIdentityMatrix();
    matrix[0][0] = scaleX;
    matrix[1][1] = scaleY;
    matrix[2][2] = scaleZ;
    return matrix;
}

std::vector<std::vector<float>> DataAugmentation::createReflectionMatrix(int axis) {
    auto matrix = createIdentityMatrix();

    switch (axis) {
        case 0: // X-axis reflection
            matrix[0][0] = -1.0f;
            break;
        case 1: // Y-axis reflection
            matrix[1][1] = -1.0f;
            break;
        case 2: // Z-axis reflection
            matrix[2][2] = -1.0f;
            break;
    }

    return matrix;
}

std::vector<std::vector<float>> DataAugmentation::matrixMultiply(const std::vector<std::vector<float>>& a,
                                                                const std::vector<std::vector<float>>& b) {
    size_t rows = a.size();
    size_t cols = b[0].size();
    size_t inner = a[0].size();

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

// Helper methods
float DataAugmentation::degreesToRadians(float degrees) {
    return degrees * M_PI / 180.0f;
}

std::vector<float> DataAugmentation::matrixVectorMultiply(const std::vector<std::vector<float>>& matrix,
                                                         const std::vector<float>& vector) {
    std::vector<float> result(matrix.size(), 0.0f);

    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size() && j < vector.size(); ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }

    return result;
}

std::vector<std::vector<float>> DataAugmentation::createIdentityMatrix() {
    return {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}
    };
}
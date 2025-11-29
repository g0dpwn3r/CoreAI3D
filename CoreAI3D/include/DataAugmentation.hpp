#pragma once
#include "CoreAI3DCommon.hpp"
#include <vector>
#include <random>
#include <cmath>

class DataAugmentation {
public:
    // Configuration struct for augmentation parameters
    struct AugmentationConfig {
        // Rotation parameters
        bool enableRotation = true;
        float rotationMaxAngle = 30.0f; // Maximum rotation angle in degrees

        // Scaling parameters
        bool enableScaling = true;
        float scalingMin = 0.8f; // Minimum scaling factor
        float scalingMax = 1.2f; // Maximum scaling factor

        // Noise parameters
        bool enableNoise = true;
        float noiseStdDev = 0.01f; // Standard deviation for Gaussian noise

        // Symmetry parameters
        bool enableSymmetry = true;
        float symmetryProbability = 0.3f; // Probability of applying symmetry operations
    };

    // Constructor
    DataAugmentation();
    explicit DataAugmentation(const AugmentationConfig& config);

    // Main augmentation method - applies all enabled augmentations
    std::vector<float> augmentCoordinates(const std::vector<float>& coordinates);

    // Individual augmentation methods
    std::vector<float> applyRotation(const std::vector<float>& coordinates, float maxAngle);
    std::vector<float> applyScaling(const std::vector<float>& coordinates, float minScale, float maxScale);
    std::vector<float> applyNoise(const std::vector<float>& coordinates, float stdDev);
    std::vector<float> applySymmetry(const std::vector<float>& coordinates, float probability);

    // Configuration methods
    void setConfig(const AugmentationConfig& config);
    const AugmentationConfig& getConfig() const;

    // Enable/disable individual augmentations
    void enableRotation(bool enable);
    void enableScaling(bool enable);
    void enableNoise(bool enable);
    void enableSymmetry(bool enable);

    // Set intensity parameters
    void setRotationMaxAngle(float angle);
    void setScalingRange(float minScale, float maxScale);
    void setNoiseStdDev(float stdDev);
    void setSymmetryProbability(float probability);

    // Matrix operations for coordinate transformations
    std::vector<float> applyTransformationMatrix(const std::vector<float>& coordinates,
                                                 const std::vector<std::vector<float>>& matrix);

    // Utility methods for matrix operations
    std::vector<std::vector<float>> createRotationMatrix(float angleX, float angleY, float angleZ);
    std::vector<std::vector<float>> createScalingMatrix(float scaleX, float scaleY, float scaleZ);
    std::vector<std::vector<float>> createReflectionMatrix(int axis); // 0=X, 1=Y, 2=Z
    std::vector<std::vector<float>> matrixMultiply(const std::vector<std::vector<float>>& a,
                                                   const std::vector<std::vector<float>>& b);

private:
    AugmentationConfig config_;
    std::mt19937 gen_;
    std::normal_distribution<float> noiseDist_;
    std::uniform_real_distribution<float> uniformDist_;

    // Helper methods
    float degreesToRadians(float degrees);
    std::vector<float> matrixVectorMultiply(const std::vector<std::vector<float>>& matrix,
                                           const std::vector<float>& vector);
    std::vector<std::vector<float>> createIdentityMatrix();
};
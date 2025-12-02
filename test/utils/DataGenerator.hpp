#ifndef DATA_GENERATOR_HPP
#define DATA_GENERATOR_HPP

#include <vector>
#include <string>
#include <random>
#include <functional>

class DataGenerator {
public:
    // Regression data generation
    static std::vector<std::vector<float>> generateRegressionData(
        size_t numSamples,
        size_t numFeatures,
        std::function<float(const std::vector<float>&)> targetFunction = nullptr,
        float noiseLevel = 0.1f
    );

    // Classification data generation
    static std::pair<std::vector<std::vector<float>>, std::vector<int>> generateClassificationData(
        size_t numSamples,
        size_t numFeatures,
        size_t numClasses,
        std::function<int(const std::vector<float>&)> labelFunction = nullptr
    );

    // Text data generation
    static std::vector<std::string> generateTextData(
        size_t numSamples,
        size_t minLength = 10,
        size_t maxLength = 100,
        const std::string& charset = "abcdefghijklmnopqrstuvwxyz "
    );

    // Generate random features
    static std::vector<float> generateRandomFeatures(size_t numFeatures, float minVal = -1.0f, float maxVal = 1.0f);

    // Generate Gaussian noise
    static float generateGaussianNoise(float mean = 0.0f, float stddev = 1.0f);

private:
    static std::mt19937& getRandomEngine();
    static std::normal_distribution<float>& getNormalDistribution(float mean, float stddev);
};

#endif // DATA_GENERATOR_HPP
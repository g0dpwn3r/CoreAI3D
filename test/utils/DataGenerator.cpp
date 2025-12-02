#include "DataGenerator.hpp"
#include <algorithm>
#include <cmath>
#include <thread>

std::vector<std::vector<float>> DataGenerator::generateRegressionData(
    size_t numSamples,
    size_t numFeatures,
    std::function<float(const std::vector<float>&)> targetFunction,
    float noiseLevel
) {
    std::vector<std::vector<float>> data;
    data.reserve(numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        std::vector<float> features = generateRandomFeatures(numFeatures);
        float target = 0.0f;

        if (targetFunction) {
            target = targetFunction(features);
        } else {
            // Default: linear combination with some weights
            std::vector<float> weights = generateRandomFeatures(numFeatures, 0.5f, 2.0f);
            for (size_t j = 0; j < numFeatures; ++j) {
                target += features[j] * weights[j];
            }
        }

        // Add noise
        target += generateGaussianNoise(0.0f, noiseLevel);

        features.push_back(target); // Append target as last element
        data.push_back(std::move(features));
    }

    return data;
}

std::pair<std::vector<std::vector<float>>, std::vector<int>> DataGenerator::generateClassificationData(
    size_t numSamples,
    size_t numFeatures,
    size_t numClasses,
    std::function<int(const std::vector<float>&)> labelFunction
) {
    std::vector<std::vector<float>> features;
    std::vector<int> labels;
    features.reserve(numSamples);
    labels.reserve(numSamples);

    for (size_t i = 0; i < numSamples; ++i) {
        std::vector<float> sampleFeatures = generateRandomFeatures(numFeatures);
        int label = 0;

        if (labelFunction) {
            label = labelFunction(sampleFeatures);
        } else {
            // Default: classify based on first feature
            label = static_cast<int>(std::floor(sampleFeatures[0] * numClasses));
            label = std::max(0, std::min(static_cast<int>(numClasses - 1), label));
        }

        features.push_back(std::move(sampleFeatures));
        labels.push_back(label);
    }

    return {features, labels};
}

std::vector<std::string> DataGenerator::generateTextData(
    size_t numSamples,
    size_t minLength,
    size_t maxLength,
    const std::string& charset
) {
    std::vector<std::string> texts;
    texts.reserve(numSamples);

    auto& engine = getRandomEngine();
    std::uniform_int_distribution<size_t> lengthDist(minLength, maxLength);
    std::uniform_int_distribution<size_t> charDist(0, charset.size() - 1);

    for (size_t i = 0; i < numSamples; ++i) {
        size_t length = lengthDist(engine);
        std::string text;
        text.reserve(length);

        for (size_t j = 0; j < length; ++j) {
            text += charset[charDist(engine)];
        }

        texts.push_back(std::move(text));
    }

    return texts;
}

std::vector<float> DataGenerator::generateRandomFeatures(size_t numFeatures, float minVal, float maxVal) {
    std::vector<float> features;
    features.reserve(numFeatures);

    auto& engine = getRandomEngine();
    std::uniform_real_distribution<float> dist(minVal, maxVal);

    for (size_t i = 0; i < numFeatures; ++i) {
        features.push_back(dist(engine));
    }

    return features;
}

float DataGenerator::generateGaussianNoise(float mean, float stddev) {
    auto& dist = getNormalDistribution(mean, stddev);
    return dist(getRandomEngine());
}

std::mt19937& DataGenerator::getRandomEngine() {
    static thread_local std::mt19937 engine(std::random_device{}());
    return engine;
}

std::normal_distribution<float>& DataGenerator::getNormalDistribution(float mean, float stddev) {
    static thread_local std::normal_distribution<float> dist(mean, stddev);
    dist = std::normal_distribution<float>(mean, stddev); // Reset parameters
    return dist;
}
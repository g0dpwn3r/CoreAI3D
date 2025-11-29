#include "Ensemble.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>

Ensemble::Ensemble(int inputSize, int outputSize, int layers, int neurons,
                   float minVal, float maxVal, const EnsembleConfig& config)
    : config_(config), inputSize_(inputSize), outputSize_(outputSize),
      layers_(layers), neurons_(neurons), minVal_(minVal), maxVal_(maxVal),
      rng_(42), uniformDist_(0.0f, 1.0f) {

    initializeModels();

    if (config_.verbose) {
        std::cout << "[ENSEMBLE] Initialized ensemble with " << config_.numModels << " models" << std::endl;
        std::cout << "[ENSEMBLE] Method: " << static_cast<int>(config_.method) << std::endl;
    }
}

Ensemble::~Ensemble() {
    // Models are automatically cleaned up by unique_ptr
}

void Ensemble::initializeModels() {
    models_.clear();
    modelWeights_.clear();
    sampleWeights_.clear();

    modelWeights_.resize(config_.numModels, 1.0f / config_.numModels); // Equal weights initially

    for (int i = 0; i < config_.numModels; ++i) {
        auto model = std::make_unique<CoreAI>(inputSize_, layers_, neurons_, outputSize_,
                                             minVal_, maxVal_, config_.verbose);
        models_.push_back(std::move(model));
    }

    if (config_.verbose) {
        std::cout << "[ENSEMBLE] Created " << models_.size() << " CoreAI models" << std::endl;
    }
}

void Ensemble::setConfig(const EnsembleConfig& config) {
    config_ = config;
    // Reinitialize models if necessary
    if (models_.size() != static_cast<size_t>(config_.numModels)) {
        initializeModels();
    }
}

const EnsembleConfig& Ensemble::getConfig() const {
    return config_;
}

bool Ensemble::train(const std::vector<std::vector<float>>& inputs,
                    const std::vector<std::vector<float>>& targets,
                    double learningRate, int epochs) {

    prepareTrainingData(inputs, targets);

    switch (config_.method) {
        case EnsembleMethod::SIMPLE_AVERAGING:
            return trainSimpleAveraging(inputs, targets, learningRate, epochs);
        case EnsembleMethod::WEIGHTED_AVERAGING:
            return trainWeightedAveraging(inputs, targets, learningRate, epochs);
        case EnsembleMethod::BAGGING:
            return trainBagging(inputs, targets, learningRate, epochs);
        case EnsembleMethod::BOOSTING:
            return trainBoosting(inputs, targets, learningRate, epochs);
        default:
            return false;
    }
}

bool Ensemble::trainSimpleAveraging(const std::vector<std::vector<float>>& inputs,
                                  const std::vector<std::vector<float>>& targets,
                                  double learningRate, int epochs) {

    if (config_.verbose) {
        std::cout << "[ENSEMBLE] Training with simple averaging..." << std::endl;
    }

    // Train all models on the full dataset
    for (auto& model : models_) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            int numSamples = inputs.size();
            model->train(inputs, targets, learningRate, numSamples);
        }
    }

    // Update weights to be equal
    std::fill(modelWeights_.begin(), modelWeights_.end(), 1.0f / models_.size());

    return true;
}

bool Ensemble::trainWeightedAveraging(const std::vector<std::vector<float>>& inputs,
                                    const std::vector<std::vector<float>>& targets,
                                    double learningRate, int epochs) {

    if (config_.verbose) {
        std::cout << "[ENSEMBLE] Training with weighted averaging..." << std::endl;
    }

    // Train all models on the full dataset
    for (auto& model : models_) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            int numSamples = inputs.size();
            model->train(inputs, targets, learningRate, numSamples);
        }
    }

    // Calculate model weights based on performance
    updateModelWeights();

    return true;
}

bool Ensemble::trainBagging(const std::vector<std::vector<float>>& inputs,
                           const std::vector<std::vector<float>>& targets,
                           double learningRate, int epochs) {

    if (config_.verbose) {
        std::cout << "[ENSEMBLE] Training with bagging..." << std::endl;
    }

    int sampleSize = static_cast<int>(inputs.size() * config_.subsetRatio);

    for (size_t i = 0; i < models_.size(); ++i) {
        // Create bootstrap sample
        auto bootstrapInputs = createBootstrapSample(inputs, sampleSize);
        auto bootstrapTargets = createBootstrapSample(targets, sampleSize);

        // Train model on bootstrap sample
        for (int epoch = 0; epoch < epochs; ++epoch) {
            int numSamples = bootstrapInputs.size();
            models_[i]->train(bootstrapInputs, bootstrapTargets, learningRate, numSamples);
        }
    }

    // Update weights based on performance
    updateModelWeights();

    return true;
}

bool Ensemble::trainBoosting(const std::vector<std::vector<float>>& inputs,
                           const std::vector<std::vector<float>>& targets,
                           double learningRate, int epochs) {

    if (config_.verbose) {
        std::cout << "[ENSEMBLE] Training with boosting..." << std::endl;
    }

    // Initialize sample weights uniformly
    std::vector<float> currentSampleWeights(inputs.size(), 1.0f / inputs.size());
    sampleWeights_.clear();
    sampleWeights_.resize(models_.size());

    for (size_t i = 0; i < models_.size(); ++i) {
        // Store current sample weights for this model
        sampleWeights_[i] = currentSampleWeights;

        // Create weighted sample based on current weights
        auto weightedInputs = createWeightedSample(inputs, currentSampleWeights);
        auto weightedTargets = createWeightedSample(targets, currentSampleWeights);

        // Train model on weighted sample
        for (int epoch = 0; epoch < epochs; ++epoch) {
            int numSamples = weightedInputs.size();
            models_[i]->train(weightedInputs, weightedTargets, learningRate, numSamples);
        }

        // Update sample weights based on model performance
        currentSampleWeights = calculateSampleWeights(i, weightedInputs, weightedTargets);

        // Check for early stopping
        float totalWeight = std::accumulate(currentSampleWeights.begin(), currentSampleWeights.end(), 0.0f);
        if (totalWeight < config_.earlyStopThreshold) {
            if (config_.verbose) {
                std::cout << "[ENSEMBLE] Early stopping at model " << i + 1 << std::endl;
            }
            break;
        }

        // Limit iterations
        if (i >= static_cast<size_t>(config_.maxBoostingIterations) - 1) {
            break;
        }
    }

    // Update model weights
    updateModelWeights();

    return true;
}

std::vector<std::vector<float>> Ensemble::predict(const std::vector<std::vector<float>>& inputs) {
    switch (config_.method) {
        case EnsembleMethod::SIMPLE_AVERAGING:
            return predictSimpleAverage(inputs);
        case EnsembleMethod::WEIGHTED_AVERAGING:
        case EnsembleMethod::BAGGING:
        case EnsembleMethod::BOOSTING:
            return predictWeightedAverage(inputs);
        default:
            return predictSimpleAverage(inputs);
    }
}

std::vector<std::vector<float>> Ensemble::predictSimpleAverage(const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<std::vector<float>>> allPredictions;

    // Get predictions from all models
    for (const auto& model : models_) {
        auto predictions = model->forward(inputs, false); // training = false
        allPredictions.push_back(predictions);
    }

    return combinePredictions(allPredictions);
}

std::vector<std::vector<float>> Ensemble::predictWeightedAverage(const std::vector<std::vector<float>>& inputs) {
    std::vector<std::vector<std::vector<float>>> allPredictions;

    // Get predictions from all models
    for (const auto& model : models_) {
        auto predictions = model->forward(inputs, false); // training = false
        allPredictions.push_back(predictions);
    }

    // Apply weights to predictions
    for (size_t modelIdx = 0; modelIdx < allPredictions.size(); ++modelIdx) {
        float weight = modelWeights_[modelIdx];
        for (auto& samplePred : allPredictions[modelIdx]) {
            for (float& pred : samplePred) {
                pred *= weight;
            }
        }
    }

    return combinePredictions(allPredictions);
}

std::vector<std::vector<float>> Ensemble::predictVoting(const std::vector<std::vector<float>>& inputs) {
    // For regression tasks, voting is similar to averaging
    // For classification, this would implement majority voting
    return predictWeightedAverage(inputs);
}

std::vector<std::vector<float>> Ensemble::createBootstrapSample(
    const std::vector<std::vector<float>>& data, int sampleSize) {

    std::vector<std::vector<float>> sample;
    sample.reserve(sampleSize);

    for (int i = 0; i < sampleSize; ++i) {
        int randomIndex = std::uniform_int_distribution<int>(0, data.size() - 1)(rng_);
        sample.push_back(data[randomIndex]);
    }

    return sample;
}

std::vector<std::vector<float>> Ensemble::createWeightedSample(
    const std::vector<std::vector<float>>& data, const std::vector<float>& weights) {

    std::vector<std::vector<float>> sample;
    sample.reserve(data.size());

    // Create discrete distribution from weights
    std::discrete_distribution<int> dist(weights.begin(), weights.end());

    for (size_t i = 0; i < data.size(); ++i) {
        int selectedIndex = dist(rng_);
        sample.push_back(data[selectedIndex]);
    }

    return sample;
}

std::vector<float> Ensemble::calculateSampleWeights(int modelIndex,
    const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets) {

    std::vector<float> newWeights(predictions.size(), 0.0f);

    // Calculate error for each sample
    for (size_t i = 0; i < predictions.size(); ++i) {
        float error = 0.0f;
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            float diff = predictions[i][j] - targets[i][j];
            error += diff * diff;
        }
        error = std::sqrt(error / predictions[i].size()); // RMSE for this sample

        // Update weight based on error (AdaBoost-style)
        float currentWeight = sampleWeights_[modelIndex][i];
        newWeights[i] = currentWeight * std::exp(config_.learningRateBoost * error);
    }

    // Normalize weights
    float totalWeight = std::accumulate(newWeights.begin(), newWeights.end(), 0.0f);
    if (totalWeight > 0) {
        for (float& w : newWeights) {
            w /= totalWeight;
        }
    }

    return newWeights;
}

float Ensemble::calculateModelError(const std::vector<std::vector<float>>& predictions,
                                  const std::vector<std::vector<float>>& targets,
                                  const std::vector<float>& sampleWeights) {

    float totalError = 0.0f;
    float totalWeight = 0.0f;

    for (size_t i = 0; i < predictions.size(); ++i) {
        float sampleError = 0.0f;
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            float diff = predictions[i][j] - targets[i][j];
            sampleError += diff * diff;
        }
        sampleError = std::sqrt(sampleError / predictions[i].size()); // RMSE

        float weight = sampleWeights.empty() ? 1.0f : sampleWeights[i];
        totalError += sampleError * weight;
        totalWeight += weight;
    }

    return totalWeight > 0 ? totalError / totalWeight : totalError;
}

void Ensemble::updateModelWeights() {
    std::vector<float> modelErrors;

    // Calculate error for each model
    for (const auto& model : models_) {
        auto predictions = model->forward(trainingInputs_, false);
        float error = calculateModelError(predictions, trainingTargets_);
        modelErrors.push_back(error);
    }

    // Convert errors to weights (lower error = higher weight)
    float totalWeight = 0.0f;
    for (size_t i = 0; i < modelErrors.size(); ++i) {
        // Avoid division by zero and negative weights
        float error = std::max(modelErrors[i], 1e-6f);
        modelWeights_[i] = 1.0f / error;
        totalWeight += modelWeights_[i];
    }

    // Normalize weights
    if (totalWeight > 0) {
        for (float& w : modelWeights_) {
            w /= totalWeight;
        }
    }

    if (config_.verbose) {
        std::cout << "[ENSEMBLE] Updated model weights: ";
        for (float w : modelWeights_) {
            std::cout << w << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<float>> Ensemble::combinePredictions(
    const std::vector<std::vector<std::vector<float>>>& allPredictions) {

    if (allPredictions.empty() || allPredictions[0].empty()) {
        return {};
    }

    size_t numSamples = allPredictions[0].size();
    size_t numOutputs = allPredictions[0][0].size();
    std::vector<std::vector<float>> combinedPredictions(numSamples, std::vector<float>(numOutputs, 0.0f));

    // Average predictions across all models
    for (size_t sampleIdx = 0; sampleIdx < numSamples; ++sampleIdx) {
        for (size_t outputIdx = 0; outputIdx < numOutputs; ++outputIdx) {
            float sum = 0.0f;
            for (const auto& modelPredictions : allPredictions) {
                sum += modelPredictions[sampleIdx][outputIdx];
            }
            combinedPredictions[sampleIdx][outputIdx] = sum / allPredictions.size();
        }
    }

    return combinedPredictions;
}

void Ensemble::prepareTrainingData(const std::vector<std::vector<float>>& inputs,
                                 const std::vector<std::vector<float>>& targets) {
    trainingInputs_ = inputs;
    trainingTargets_ = targets;
}

void Ensemble::addModel(std::unique_ptr<CoreAI> model, float weight) {
    models_.push_back(std::move(model));
    modelWeights_.push_back(weight);

    // Renormalize weights
    float totalWeight = std::accumulate(modelWeights_.begin(), modelWeights_.end(), 0.0f);
    if (totalWeight > 0) {
        for (float& w : modelWeights_) {
            w /= totalWeight;
        }
    }
}

void Ensemble::removeModel(int index) {
    if (index >= 0 && index < static_cast<int>(models_.size())) {
        models_.erase(models_.begin() + index);
        modelWeights_.erase(modelWeights_.begin() + index);

        // Renormalize remaining weights
        float totalWeight = std::accumulate(modelWeights_.begin(), modelWeights_.end(), 0.0f);
        if (totalWeight > 0) {
            for (float& w : modelWeights_) {
                w /= totalWeight;
            }
        }
    }
}

void Ensemble::clearModels() {
    models_.clear();
    modelWeights_.clear();
    sampleWeights_.clear();
}

size_t Ensemble::getNumModels() const {
    return models_.size();
}

void Ensemble::setModelWeights(const std::vector<float>& weights) {
    if (weights.size() == modelWeights_.size()) {
        modelWeights_ = weights;

        // Normalize weights
        float totalWeight = std::accumulate(modelWeights_.begin(), modelWeights_.end(), 0.0f);
        if (totalWeight > 0) {
            for (float& w : modelWeights_) {
                w /= totalWeight;
            }
        }
    }
}

const std::vector<float>& Ensemble::getModelWeights() const {
    return modelWeights_;
}

float Ensemble::calculateEnsembleError(const std::vector<std::vector<float>>& inputs,
                                     const std::vector<std::vector<float>>& targets) {
    auto predictions = predict(inputs);
    return calculateModelError(predictions, targets);
}

std::vector<float> Ensemble::getModelErrors(const std::vector<std::vector<float>>& inputs,
                                          const std::vector<std::vector<float>>& targets) {
    std::vector<float> errors;

    for (const auto& model : models_) {
        auto predictions = model->forward(inputs, false);
        float error = calculateModelError(predictions, targets);
        errors.push_back(error);
    }

    return errors;
}

void Ensemble::saveEnsemble(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for saving ensemble: " + filename);
    }

    // Save configuration
    file << "CONFIG\n";
    file << static_cast<int>(config_.method) << "\n";
    file << config_.numModels << "\n";
    file << config_.subsetRatio << "\n";
    file << config_.learningRateBoost << "\n";
    file << config_.maxBoostingIterations << "\n";
    file << config_.earlyStopThreshold << "\n";
    file << (config_.useDataAugmentation ? 1 : 0) << "\n";
    file << (config_.verbose ? 1 : 0) << "\n";

    // Save model weights
    file << "WEIGHTS\n";
    for (float w : modelWeights_) {
        file << w << "\n";
    }

    // Note: Individual model states would need to be saved separately
    // This is a simplified implementation

    file.close();
}

bool Ensemble::loadEnsemble(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::getline(file, line);
    if (line != "CONFIG") return false;

    // Load configuration
    std::getline(file, line);
    config_.method = static_cast<EnsembleMethod>(std::stoi(line));
    std::getline(file, line);
    config_.numModels = std::stoi(line);
    std::getline(file, line);
    config_.subsetRatio = std::stof(line);
    std::getline(file, line);
    config_.learningRateBoost = std::stof(line);
    std::getline(file, line);
    config_.maxBoostingIterations = std::stoi(line);
    std::getline(file, line);
    config_.earlyStopThreshold = std::stof(line);
    std::getline(file, line);
    config_.useDataAugmentation = std::stoi(line) != 0;
    std::getline(file, line);
    config_.verbose = std::stoi(line) != 0;

    std::getline(file, line);
    if (line != "WEIGHTS") return false;

    // Load model weights
    modelWeights_.clear();
    while (std::getline(file, line)) {
        modelWeights_.push_back(std::stof(line));
    }

    // Reinitialize models with loaded configuration
    initializeModels();

    file.close();
    return true;
}

void Ensemble::printEnsembleInfo() const {
    std::cout << "=== Ensemble Information ===" << std::endl;
    std::cout << "Method: " << static_cast<int>(config_.method) << std::endl;
    std::cout << "Number of models: " << models_.size() << std::endl;
    std::cout << "Model weights: ";
    for (float w : modelWeights_) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    std::cout << "Input size: " << inputSize_ << std::endl;
    std::cout << "Output size: " << outputSize_ << std::endl;
    std::cout << "Layers: " << layers_ << std::endl;
    std::cout << "Neurons per layer: " << neurons_ << std::endl;
    std::cout << "============================" << std::endl;
}
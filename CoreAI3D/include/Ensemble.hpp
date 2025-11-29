#ifndef ENSEMBLE_HPP
#define ENSEMBLE_HPP

#include "CoreAI3DCommon.hpp"
#include "Core.hpp"
#include "Train.hpp"
#include <memory>
#include <vector>
#include <string>
#include <random>
#include <functional>

// Ensemble method types
enum class EnsembleMethod {
    SIMPLE_AVERAGING,    // Simple average of all model predictions
    WEIGHTED_AVERAGING,  // Weighted average based on model performance
    BAGGING,            // Bootstrap aggregation with random subsets
    BOOSTING            // Sequential model improvement with sample reweighting
};

// Voting strategy for classification tasks
enum class VotingStrategy {
    MAJORITY_VOTE,      // Majority voting
    WEIGHTED_VOTE,      // Weighted voting based on confidence
    PROBABILITY_AVERAGE // Average predicted probabilities
};

// Ensemble configuration structure
struct EnsembleConfig {
    EnsembleMethod method = EnsembleMethod::SIMPLE_AVERAGING;
    VotingStrategy voting = VotingStrategy::MAJORITY_VOTE;
    int numModels = 5;                          // Number of models in ensemble
    float subsetRatio = 0.8f;                   // Ratio of training data for bagging (0.0-1.0)
    float learningRateBoost = 1.1f;             // Learning rate multiplier for boosting
    bool useDataAugmentation = false;           // Enable data augmentation for diversity
    int maxBoostingIterations = 10;             // Maximum boosting iterations
    float earlyStopThreshold = 0.001f;          // Early stopping threshold for boosting
    bool verbose = false;                       // Enable verbose output
};

class Ensemble {
private:
    // Ensemble models
    std::vector<std::unique_ptr<CoreAI>> models_;
    std::vector<float> modelWeights_;           // Weights for weighted averaging/voting
    std::vector<std::vector<std::vector<float>>> sampleWeights_; // For boosting (per model, per sample)

    // Configuration
    EnsembleConfig config_;
    int inputSize_;
    int outputSize_;
    int layers_;
    int neurons_;
    float minVal_;
    float maxVal_;

    // Training data (stored for bagging and boosting)
    std::vector<std::vector<float>> trainingInputs_;
    std::vector<std::vector<float>> trainingTargets_;

    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniformDist_;

    // Helper methods
    void initializeModels();
    std::vector<std::vector<float>> createBootstrapSample(const std::vector<std::vector<float>>& data, int sampleSize);
    std::vector<float> calculateSampleWeights(int modelIndex, const std::vector<std::vector<float>>& predictions,
                                            const std::vector<std::vector<float>>& targets);
    float calculateModelError(const std::vector<std::vector<float>>& predictions,
                            const std::vector<std::vector<float>>& targets,
                            const std::vector<float>& sampleWeights = {});
    void updateModelWeights();
    std::vector<std::vector<float>> combinePredictions(const std::vector<std::vector<std::vector<float>>>& allPredictions);

public:
    // Constructor
    Ensemble(int inputSize, int outputSize, int layers = 3, int neurons = 10,
             float minVal = 0.0f, float maxVal = 1.0f, const EnsembleConfig& config = EnsembleConfig());

    // Destructor
    ~Ensemble();

    // Configuration methods
    void setConfig(const EnsembleConfig& config);
    const EnsembleConfig& getConfig() const;

    // Training methods
    bool trainSimpleAveraging(const std::vector<std::vector<float>>& inputs,
                            const std::vector<std::vector<float>>& targets,
                            double learningRate, int epochs);
    bool trainWeightedAveraging(const std::vector<std::vector<float>>& inputs,
                              const std::vector<std::vector<float>>& targets,
                              double learningRate, int epochs);
    bool trainBagging(const std::vector<std::vector<float>>& inputs,
                    const std::vector<std::vector<float>>& targets,
                    double learningRate, int epochs);
    bool trainBoosting(const std::vector<std::vector<float>>& inputs,
                     const std::vector<std::vector<float>>& targets,
                     double learningRate, int epochs);

    // Unified training method
    bool train(const std::vector<std::vector<float>>& inputs,
              const std::vector<std::vector<float>>& targets,
              double learningRate, int epochs);

    // Prediction methods
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& inputs);
    std::vector<std::vector<float>> predictSimpleAverage(const std::vector<std::vector<float>>& inputs);
    std::vector<std::vector<float>> predictWeightedAverage(const std::vector<std::vector<float>>& inputs);
    std::vector<std::vector<float>> predictVoting(const std::vector<std::vector<float>>& inputs);

    // Model management
    void addModel(std::unique_ptr<CoreAI> model, float weight = 1.0f);
    void removeModel(int index);
    void clearModels();
    size_t getNumModels() const;

    // Weight management
    void setModelWeights(const std::vector<float>& weights);
    const std::vector<float>& getModelWeights() const;

    // Evaluation methods
    float calculateEnsembleError(const std::vector<std::vector<float>>& inputs,
                               const std::vector<std::vector<float>>& targets);
    std::vector<float> getModelErrors(const std::vector<std::vector<float>>& inputs,
                                    const std::vector<std::vector<float>>& targets);

    // Utility methods
    void saveEnsemble(const std::string& filename);
    bool loadEnsemble(const std::string& filename);
    void printEnsembleInfo() const;

    // Data preprocessing for ensemble methods
    void prepareTrainingData(const std::vector<std::vector<float>>& inputs,
                           const std::vector<std::vector<float>>& targets);
};

#endif // ENSEMBLE_HPP
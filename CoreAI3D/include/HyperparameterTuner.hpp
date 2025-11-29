#pragma once

#include "CoreAI3DCommon.hpp"
#include "Train.hpp"
#include "Database.hpp"
#include <nlohmann/json.hpp>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <random>
#include <functional>
#include <atomic>
#include <future>

// Forward declarations
class Training;
class Database;

// Hyperparameter search methods
enum class SearchMethod {
    GRID_SEARCH,
    RANDOM_SEARCH,
    BAYESIAN_OPTIMIZATION
};

// Parameter types for different hyperparameters
enum class ParameterType {
    FLOAT_RANGE,
    INT_RANGE,
    CATEGORICAL,
    LOG_SCALE
};

// Parameter definition structure
struct ParameterDefinition {
    std::string name;
    ParameterType type;
    std::vector<float> float_range; // [min, max] for float ranges
    std::vector<int> int_range;     // [min, max] for int ranges
    std::vector<std::string> categories; // for categorical parameters
    float log_base = 10.0f;         // base for log scale
    int num_samples = 10;           // number of samples for random search
};

// Cross-validation configuration
struct CrossValidationConfig {
    int k_folds = 5;
    bool stratified = false;
    std::string metric = "mse"; // mse, rmse, accuracy, etc.
    bool shuffle = true;
};

// Early stopping configuration
struct EarlyStoppingConfig {
    bool enabled = true;
    int patience = 10;
    float min_delta = 1e-4f;
    std::string monitor = "val_loss";
    bool restore_best_weights = true;
};

// Bayesian optimization configuration
struct BayesianConfig {
    int initial_points = 5;
    int n_iter = 25;
    float kappa = 2.576f; // exploration-exploitation trade-off
    std::string kernel = "matern"; // matern, rbf, etc.
};

// Tuning configuration
struct TuningConfig {
    SearchMethod method = SearchMethod::GRID_SEARCH;
    CrossValidationConfig cv_config;
    EarlyStoppingConfig early_stop_config;
    BayesianConfig bayesian_config;
    int max_evaluations = 100;
    int num_random_starts = 10;
    bool parallel_evaluation = true;
    int max_parallel_jobs = 4;
    std::string save_path = "tuning_results.json";
    bool verbose = true;
};

// Parameter set for evaluation
struct ParameterSet {
    std::map<std::string, float> float_params;
    std::map<std::string, int> int_params;
    std::map<std::string, std::string> categorical_params;
    float score = 0.0f;
    float std_dev = 0.0f;
    int evaluation_time = 0;
    bool evaluated = false;
};

// Tuning results
struct TuningResults {
    std::vector<ParameterSet> parameter_sets;
    ParameterSet best_parameters;
    std::vector<std::pair<int, float>> convergence_history;
    double total_time = 0.0;
    int total_evaluations = 0;
    std::string timestamp;
    nlohmann::json metadata;
};

// Hyperparameter Tuner class
class HyperparameterTuner {
private:
    // Core components
    std::unique_ptr<Training> training_template_;
    std::map<std::string, ParameterDefinition> parameter_space_;
    TuningConfig config_;

    // Database integration
    std::unique_ptr<Database> database_;
    bool use_database_ = false;

    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<float> uniform_dist_;

    // Tuning state
    std::atomic<bool> tuning_active_;
    std::atomic<bool> stop_requested_;
    std::future<TuningResults> tuning_future_;
    std::string current_status_;
    int current_evaluation_;
    int total_evaluations_;

    // Bayesian optimization state
    std::vector<std::vector<float>> observed_points_;
    std::vector<float> observed_scores_;
    std::vector<std::vector<float>> candidate_points_;

    // Progress tracking
    std::chrono::steady_clock::time_point tuning_start_time_;

public:
    // Constructor
    HyperparameterTuner(std::unique_ptr<Training> training_template,
                       const std::map<std::string, ParameterDefinition>& parameter_space,
                       const TuningConfig& config = TuningConfig());

    // Database integration
    void setDatabase(std::unique_ptr<Database> database);

    // Parameter space management
    void addParameter(const std::string& name, const ParameterDefinition& param_def);
    void removeParameter(const std::string& name);
    void clearParameterSpace();
    const std::map<std::string, ParameterDefinition>& getParameterSpace() const;

    // Tuning configuration
    void setTuningConfig(const TuningConfig& config);
    const TuningConfig& getTuningConfig() const;

    // Grid search implementation
    std::vector<ParameterSet> generateGridSearchSpace() const;
    TuningResults performGridSearch();

    // Random search implementation
    std::vector<ParameterSet> generateRandomSearchSpace(int num_samples) const;
    TuningResults performRandomSearch(int num_samples = 100);

    // Bayesian optimization implementation
    TuningResults performBayesianOptimization();
    float expectedImprovement(const std::vector<float>& point) const;
    std::vector<float> suggestNextPoint() const;

    // Cross-validation
    float crossValidateParameters(const ParameterSet& params, const CrossValidationConfig& cv_config);
    std::vector<float> kFoldSplit(int k, int total_samples) const;

    // Early stopping
    bool shouldStopEarly(const std::vector<float>& validation_losses,
                        const EarlyStoppingConfig& config) const;

    // Main tuning interface
    TuningResults tuneHyperparameters();
    bool startTuning();
    bool isTuning() const;
    std::string getTuningStatus() const;
    bool stopTuning();
    void resetTuningState();

    // Results management
    TuningResults getCurrentResults() const;
    bool saveResults(const TuningResults& results, const std::string& filepath) const;
    TuningResults loadResults(const std::string& filepath) const;

    // Database operations
    bool saveTuningResultsToDB(const TuningResults& results, int dataset_id);
    bool loadTuningResultsFromDB(int tuning_session_id, TuningResults& results);

    // Utility functions
    ParameterSet createParameterSetFromMaps(
        const std::map<std::string, float>& float_params,
        const std::map<std::string, int>& int_params,
        const std::map<std::string, std::string>& categorical_params) const;

    std::map<std::string, float> extractFloatParams(const ParameterSet& params) const;
    std::map<std::string, int> extractIntParams(const ParameterSet& params) const;
    std::map<std::string, std::string> extractCategoricalParams(const ParameterSet& params) const;

    // Parameter sampling
    float sampleFloatParameter(const ParameterDefinition& param) const;
    int sampleIntParameter(const ParameterDefinition& param) const;
    std::string sampleCategoricalParameter(const ParameterDefinition& param) const;

    // Gaussian process utilities for Bayesian optimization
    float gaussianKernel(const std::vector<float>& x1, const std::vector<float>& x2, float length_scale = 1.0f) const;
    std::vector<std::vector<float>> computeCovarianceMatrix(const std::vector<std::vector<float>>& points, float length_scale = 1.0f) const;
    float predictMean(const std::vector<float>& point, float length_scale = 1.0f) const;
    float predictVariance(const std::vector<float>& point, float length_scale = 1.0f) const;

private:
    // Internal evaluation function
    float evaluateParameterSet(const ParameterSet& params);

    // Parallel evaluation
    std::vector<float> evaluateParameterSetsParallel(const std::vector<ParameterSet>& param_sets);

    // Utility functions
    std::vector<float> parameterSetToVector(const ParameterSet& params) const;
    ParameterSet vectorToParameterSet(const std::vector<float>& vec) const;
    float calculateScore(const std::vector<float>& predictions, const std::vector<float>& targets) const;

    // Progress callbacks
    void updateProgress(int current, int total, const std::string& status);
    void logMessage(const std::string& message) const;
};
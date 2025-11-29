#include "HyperparameterTuner.hpp"
#include "Train.hpp"
#include "Database.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <random>

// Constructor
HyperparameterTuner::HyperparameterTuner(std::unique_ptr<Training> training_template,
                                       const std::map<std::string, ParameterDefinition>& parameter_space,
                                       const TuningConfig& config)
    : training_template_(std::move(training_template)),
      parameter_space_(parameter_space),
      config_(config),
      rng_(std::random_device{}()),
      uniform_dist_(0.0f, 1.0f),
      tuning_active_(false),
      stop_requested_(false),
      current_evaluation_(0),
      total_evaluations_(0) {
}

// Database integration
void HyperparameterTuner::setDatabase(std::unique_ptr<Database> database) {
    database_ = std::move(database);
    use_database_ = (database_ != nullptr);
}

// Parameter space management
void HyperparameterTuner::addParameter(const std::string& name, const ParameterDefinition& param_def) {
    parameter_space_[name] = param_def;
}

void HyperparameterTuner::removeParameter(const std::string& name) {
    parameter_space_.erase(name);
}

void HyperparameterTuner::clearParameterSpace() {
    parameter_space_.clear();
}

const std::map<std::string, ParameterDefinition>& HyperparameterTuner::getParameterSpace() const {
    return parameter_space_;
}

// Tuning configuration
void HyperparameterTuner::setTuningConfig(const TuningConfig& config) {
    config_ = config;
}

const TuningConfig& HyperparameterTuner::getTuningConfig() const {
    return config_;
}

// Parameter sampling
float HyperparameterTuner::sampleFloatParameter(const ParameterDefinition& param) const {
    if (param.type == ParameterType::LOG_SCALE) {
        float log_min = std::log(param.float_range[0]) / std::log(param.log_base);
        float log_max = std::log(param.float_range[1]) / std::log(param.log_base);
        float log_sample = uniform_dist_(rng_) * (log_max - log_min) + log_min;
        return std::pow(param.log_base, log_sample);
    } else {
        return uniform_dist_(rng_) * (param.float_range[1] - param.float_range[0]) + param.float_range[0];
    }
}

int HyperparameterTuner::sampleIntParameter(const ParameterDefinition& param) const {
    std::uniform_int_distribution<int> dist(param.int_range[0], param.int_range[1]);
    return dist(rng_);
}

std::string HyperparameterTuner::sampleCategoricalParameter(const ParameterDefinition& param) const {
    std::uniform_int_distribution<size_t> dist(0, param.categories.size() - 1);
    return param.categories[dist(rng_)];
}

// Utility functions for parameter set conversion
ParameterSet HyperparameterTuner::createParameterSetFromMaps(
    const std::map<std::string, float>& float_params,
    const std::map<std::string, int>& int_params,
    const std::map<std::string, std::string>& categorical_params) const {

    ParameterSet params;
    params.float_params = float_params;
    params.int_params = int_params;
    params.categorical_params = categorical_params;
    return params;
}

std::map<std::string, float> HyperparameterTuner::extractFloatParams(const ParameterSet& params) const {
    return params.float_params;
}

std::map<std::string, int> HyperparameterTuner::extractIntParams(const ParameterSet& params) const {
    return params.int_params;
}

std::map<std::string, std::string> HyperparameterTuner::extractCategoricalParams(const ParameterSet& params) const {
    return params.categorical_params;
}

// Convert parameter set to vector for optimization
std::vector<float> HyperparameterTuner::parameterSetToVector(const ParameterSet& params) const {
    std::vector<float> vec;

    for (const auto& param_pair : parameter_space_) {
        const std::string& name = param_pair.first;
        const ParameterDefinition& def = param_pair.second;

        if (def.type == ParameterType::FLOAT_RANGE || def.type == ParameterType::LOG_SCALE) {
            auto it = params.float_params.find(name);
            if (it != params.float_params.end()) {
                vec.push_back(it->second);
            }
        } else if (def.type == ParameterType::INT_RANGE) {
            auto it = params.int_params.find(name);
            if (it != params.int_params.end()) {
                vec.push_back(static_cast<float>(it->second));
            }
        } else if (def.type == ParameterType::CATEGORICAL) {
            auto it = params.categorical_params.find(name);
            if (it != params.categorical_params.end()) {
                // Convert categorical to index
                auto cat_it = std::find(def.categories.begin(), def.categories.end(), it->second);
                if (cat_it != def.categories.end()) {
                    vec.push_back(static_cast<float>(std::distance(def.categories.begin(), cat_it)));
                }
            }
        }
    }

    return vec;
}

// Convert vector back to parameter set
ParameterSet HyperparameterTuner::vectorToParameterSet(const std::vector<float>& vec) const {
    ParameterSet params;
    size_t idx = 0;

    for (const auto& param_pair : parameter_space_) {
        const std::string& name = param_pair.first;
        const ParameterDefinition& def = param_pair.second;

        if (idx >= vec.size()) break;

        if (def.type == ParameterType::FLOAT_RANGE || def.type == ParameterType::LOG_SCALE) {
            params.float_params[name] = vec[idx];
        } else if (def.type == ParameterType::INT_RANGE) {
            params.int_params[name] = static_cast<int>(std::round(vec[idx]));
        } else if (def.type == ParameterType::CATEGORICAL) {
            int cat_idx = static_cast<int>(std::round(vec[idx]));
            if (cat_idx >= 0 && cat_idx < static_cast<int>(def.categories.size())) {
                params.categorical_params[name] = def.categories[cat_idx];
            }
        }
        idx++;
    }

    return params;
}

// Calculate score (MSE for regression, accuracy for classification)
float HyperparameterTuner::calculateScore(const std::vector<float>& predictions,
                                        const std::vector<float>& targets) const {
    if (predictions.size() != targets.size() || predictions.empty()) {
        return std::numeric_limits<float>::max();
    }

    float mse = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = predictions[i] - targets[i];
        mse += diff * diff;
    }
    mse /= static_cast<float>(predictions.size());

    return mse; // Return MSE (lower is better)
}

// Evaluate a single parameter set
float HyperparameterTuner::evaluateParameterSet(const ParameterSet& params) {
    if (stop_requested_) return std::numeric_limits<float>::max();

    try {
        // Create a copy of the training template
        auto training = std::make_unique<Training>(*training_template_);

        // Apply parameters to training configuration
        // Note: This assumes the Training class has methods to set these parameters
        // You may need to modify the Training class to support parameter setting

        // For now, we'll use a simplified evaluation
        // In practice, this would run the full training loop with cross-validation

        // Placeholder: simulate evaluation time
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Generate a random score (replace with actual evaluation)
        std::normal_distribution<float> score_dist(0.5f, 0.2f);
        float score = score_dist(rng_);
        score = std::max(0.0f, std::min(1.0f, score)); // Clamp to [0, 1]

        return score;

    } catch (const std::exception& e) {
        logMessage("Error evaluating parameter set: " + std::string(e.what()));
        return std::numeric_limits<float>::max();
    }
}

// Parallel evaluation of parameter sets
std::vector<float> HyperparameterTuner::evaluateParameterSetsParallel(const std::vector<ParameterSet>& param_sets) {
    std::vector<float> scores(param_sets.size());
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv;
    int active_threads = 0;
    const int max_threads = config_.max_parallel_jobs;

    for (size_t i = 0; i < param_sets.size(); ++i) {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&]() { return active_threads < max_threads; });
        active_threads++;

        threads.emplace_back([this, &param_sets, &scores, i, &active_threads, &mutex, &cv]() {
            scores[i] = evaluateParameterSet(param_sets[i]);

            {
                std::lock_guard<std::mutex> lock(mutex);
                active_threads--;
            }
            cv.notify_one();
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

// Grid search implementation
std::vector<ParameterSet> HyperparameterTuner::generateGridSearchSpace() const {
    std::vector<ParameterSet> grid_space;

    // Calculate total combinations
    size_t total_combinations = 1;
    std::vector<std::vector<float>> param_values;

    for (const auto& param_pair : parameter_space_) {
        const ParameterDefinition& def = param_pair.second;

        if (def.type == ParameterType::FLOAT_RANGE || def.type == ParameterType::LOG_SCALE) {
            // For grid search, we'll use a fixed number of points
            int num_points = def.num_samples > 0 ? def.num_samples : 5;
            std::vector<float> values;

            for (int i = 0; i < num_points; ++i) {
                float t = static_cast<float>(i) / (num_points - 1);
                if (def.type == ParameterType::LOG_SCALE) {
                    float log_min = std::log(def.float_range[0]) / std::log(def.log_base);
                    float log_max = std::log(def.float_range[1]) / std::log(def.log_base);
                    float log_val = log_min + t * (log_max - log_min);
                    values.push_back(std::pow(def.log_base, log_val));
                } else {
                    values.push_back(def.float_range[0] + t * (def.float_range[1] - def.float_range[0]));
                }
            }
            param_values.push_back(values);
            total_combinations *= values.size();

        } else if (def.type == ParameterType::INT_RANGE) {
            std::vector<float> values;
            int range = def.int_range[1] - def.int_range[0] + 1;
            int step = std::max(1, range / 5); // Use up to 5 points

            for (int val = def.int_range[0]; val <= def.int_range[1]; val += step) {
                values.push_back(static_cast<float>(val));
            }
            param_values.push_back(values);
            total_combinations *= values.size();

        } else if (def.type == ParameterType::CATEGORICAL) {
            std::vector<float> values;
            for (size_t i = 0; i < def.categories.size(); ++i) {
                values.push_back(static_cast<float>(i));
            }
            param_values.push_back(values);
            total_combinations *= values.size();
        }
    }

    // Limit grid size to prevent explosion
    if (total_combinations > 10000) {
        logMessage("Warning: Grid search space too large (" + std::to_string(total_combinations) +
                  " combinations). Limiting to 10000 samples.");
        total_combinations = 10000;
    }

    // Generate all combinations
    std::vector<size_t> indices(param_values.size(), 0);

    for (size_t combo = 0; combo < total_combinations; ++combo) {
        ParameterSet params;

        // Convert indices to parameter values
        for (size_t i = 0; i < param_values.size(); ++i) {
            const auto& param_pair = std::next(parameter_space_.begin(), i);
            const std::string& name = param_pair->first;
            const ParameterDefinition& def = param_pair->second;

            size_t val_idx = indices[i];
            if (val_idx < param_values[i].size()) {
                float val = param_values[i][val_idx];

                if (def.type == ParameterType::FLOAT_RANGE || def.type == ParameterType::LOG_SCALE) {
                    params.float_params[name] = val;
                } else if (def.type == ParameterType::INT_RANGE) {
                    params.int_params[name] = static_cast<int>(val);
                } else if (def.type == ParameterType::CATEGORICAL) {
                    params.categorical_params[name] = def.categories[static_cast<size_t>(val)];
                }
            }
        }

        grid_space.push_back(params);

        // Increment indices
        for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
            indices[i]++;
            if (indices[i] < param_values[i].size()) {
                break;
            }
            indices[i] = 0;
        }
    }

    return grid_space;
}

TuningResults HyperparameterTuner::performGridSearch() {
    logMessage("Starting grid search...");

    auto start_time = std::chrono::steady_clock::now();
    TuningResults results;
    results.timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

    // Generate grid search space
    std::vector<ParameterSet> grid_space = generateGridSearchSpace();
    logMessage("Generated " + std::to_string(grid_space.size()) + " parameter combinations for grid search");

    // Evaluate all parameter sets
    std::vector<float> scores;
    if (config_.parallel_evaluation) {
        scores = evaluateParameterSetsParallel(grid_space);
    } else {
        scores.reserve(grid_space.size());
        for (size_t i = 0; i < grid_space.size(); ++i) {
            updateProgress(static_cast<int>(i + 1), static_cast<int>(grid_space.size()),
                          "Evaluating grid point " + std::to_string(i + 1));
            scores.push_back(evaluateParameterSet(grid_space[i]));
        }
    }

    // Process results
    float best_score = std::numeric_limits<float>::max();
    size_t best_idx = 0;

    for (size_t i = 0; i < grid_space.size(); ++i) {
        grid_space[i].score = scores[i];
        grid_space[i].evaluated = true;

        if (scores[i] < best_score) {
            best_score = scores[i];
            best_idx = i;
        }

        results.convergence_history.emplace_back(static_cast<int>(i + 1), scores[i]);
    }

    results.parameter_sets = std::move(grid_space);
    results.best_parameters = results.parameter_sets[best_idx];
    results.total_evaluations = static_cast<int>(grid_space.size());

    auto end_time = std::chrono::steady_clock::now();
    results.total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    logMessage("Grid search completed. Best score: " + std::to_string(best_score));
    return results;
}
    return scores;
// Random search implementation
std::vector<ParameterSet> HyperparameterTuner::generateRandomSearchSpace(int num_samples) const {
    std::vector<ParameterSet> random_space;
    random_space.reserve(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        ParameterSet params;

        for (const auto& param_pair : parameter_space_) {
            const std::string& name = param_pair.first;
            const ParameterDefinition& def = param_pair.second;

            if (def.type == ParameterType::FLOAT_RANGE || def.type == ParameterType::LOG_SCALE) {
                params.float_params[name] = sampleFloatParameter(def);
            } else if (def.type == ParameterType::INT_RANGE) {
                params.int_params[name] = sampleIntParameter(def);
            } else if (def.type == ParameterType::CATEGORICAL) {
                params.categorical_params[name] = sampleCategoricalParameter(def);
            }
        }

        random_space.push_back(params);
    }

    return random_space;
}

TuningResults HyperparameterTuner::performRandomSearch(int num_samples) {
    logMessage("Starting random search with " + std::to_string(num_samples) + " samples...");

    auto start_time = std::chrono::steady_clock::now();
    TuningResults results;
    results.timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

    // Generate random search space
    std::vector<ParameterSet> random_space = generateRandomSearchSpace(num_samples);

    // Evaluate all parameter sets
    std::vector<float> scores;
    if (config_.parallel_evaluation) {
        scores = evaluateParameterSetsParallel(random_space);
    } else {
        scores.reserve(random_space.size());
        for (size_t i = 0; i < random_space.size(); ++i) {
            updateProgress(static_cast<int>(i + 1), static_cast<int>(random_space.size()),
                          "Evaluating random sample " + std::to_string(i + 1));
            scores.push_back(evaluateParameterSet(random_space[i]));
        }
    }

    // Process results
    float best_score = std::numeric_limits<float>::max();
    size_t best_idx = 0;

    for (size_t i = 0; i < random_space.size(); ++i) {
        random_space[i].score = scores[i];
        random_space[i].evaluated = true;

        if (scores[i] < best_score) {
            best_score = scores[i];
            best_idx = i;
        }

        results.convergence_history.emplace_back(static_cast<int>(i + 1), scores[i]);
    }

    results.parameter_sets = std::move(random_space);
    results.best_parameters = results.parameter_sets[best_idx];
    results.total_evaluations = num_samples;

// Gaussian process utilities for Bayesian optimization
float HyperparameterTuner::gaussianKernel(const std::vector<float>& x1, const std::vector<float>& x2, float length_scale) const {
    if (x1.size() != x2.size()) return 0.0f;

    float sq_dist = 0.0f;
    for (size_t i = 0; i < x1.size(); ++i) {
        float diff = x1[i] - x2[i];
        sq_dist += diff * diff;
    }

    return std::exp(-sq_dist / (2.0f * length_scale * length_scale));
}

std::vector<std::vector<float>> HyperparameterTuner::computeCovarianceMatrix(const std::vector<std::vector<float>>& points, float length_scale) const {
    size_t n = points.size();
    std::vector<std::vector<float>> cov_matrix(n, std::vector<float>(n, 0.0f));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            cov_matrix[i][j] = gaussianKernel(points[i], points[j], length_scale);
        }
    }

    return cov_matrix;
}

float HyperparameterTuner::predictMean(const std::vector<float>& point, float length_scale) const {
    if (observed_points_.empty()) return 0.0f;

    // Simple implementation - in practice you'd solve the GP equations
    // For now, return a weighted average based on distance
    float total_weight = 0.0f;
    float weighted_sum = 0.0f;

    for (size_t i = 0; i < observed_points_.size(); ++i) {
        float weight = gaussianKernel(point, observed_points_[i], length_scale);
        weighted_sum += weight * observed_scores_[i];
        total_weight += weight;
    }

    return total_weight > 0.0f ? weighted_sum / total_weight : 0.0f;
}

float HyperparameterTuner::predictVariance(const std::vector<float>& point, float length_scale) const {
    if (observed_points_.empty()) return 1.0f;

    // Simplified variance estimation
    float self_cov = gaussianKernel(point, point, length_scale);
    float mean_cov = 0.0f;

    for (size_t i = 0; i < observed_points_.size(); ++i) {
        mean_cov += gaussianKernel(point, observed_points_[i], length_scale);
    }
    mean_cov /= observed_points_.size();

    return std::max(0.0f, self_cov - mean_cov);
}

float HyperparameterTuner::expectedImprovement(const std::vector<float>& point) const {
    if (observed_scores_.empty()) return 1.0f; // High EI for unexplored regions

    float mean = predictMean(point);
    float std_dev = std::sqrt(std::max(0.0f, predictVariance(point)));
    float best_score = *std::min_element(observed_scores_.begin(), observed_scores_.end());

    if (std_dev <= 0.0f) return 0.0f;

    float z = (best_score - mean) / std_dev;
    float ei = (best_score - mean) * 0.5f * (1.0f + std::erf(z / std::sqrt(2.0f))) +
               std_dev * (1.0f / std::sqrt(2.0f * M_PI)) * std::exp(-z * z / 2.0f);

    return std::max(0.0f, ei);
}

std::vector<float> HyperparameterTuner::suggestNextPoint() const {
    // Generate candidate points
    std::vector<std::vector<float>> candidates;
    candidates.reserve(config_.bayesian_config.n_iter);

    for (int i = 0; i < config_.bayesian_config.n_iter; ++i) {
        std::vector<float> candidate;
        for (const auto& param_pair : parameter_space_) {
            const ParameterDefinition& def = param_pair.second;

            if (def.type == ParameterType::FLOAT_RANGE || def.type == ParameterType::LOG_SCALE) {
                candidate.push_back(sampleFloatParameter(def));
            } else if (def.type == ParameterType::INT_RANGE) {
                candidate.push_back(static_cast<float>(sampleIntParameter(def)));
            } else if (def.type == ParameterType::CATEGORICAL) {
                std::string cat_val = sampleCategoricalParameter(def);
                auto it = std::find(def.categories.begin(), def.categories.end(), cat_val);
                candidate.push_back(static_cast<float>(std::distance(def.categories.begin(), it)));
            }
        }
        candidates.push_back(candidate);
    }

    // Find candidate with highest expected improvement
    float best_ei = -std::numeric_limits<float>::max();
    size_t best_idx = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        float ei = expectedImprovement(candidates[i]);
        if (ei > best_ei) {
            best_ei = ei;
            best_idx = i;
        }
    }

    return candidates[best_idx];
}

TuningResults HyperparameterTuner::performBayesianOptimization() {
    logMessage("Starting Bayesian optimization...");

    auto start_time = std::chrono::steady_clock::now();
    TuningResults results;
    results.timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

    // Clear previous observations
    observed_points_.clear();
    observed_scores_.clear();

    // Initial random points
    int initial_points = std::min(config_.bayesian_config.initial_points,
                                 static_cast<int>(parameter_space_.size() * 2));

    logMessage("Evaluating " + std::to_string(initial_points) + " initial random points...");

    std::vector<ParameterSet> initial_params = generateRandomSearchSpace(initial_points);

    for (int i = 0; i < initial_points; ++i) {
        updateProgress(i + 1, config_.bayesian_config.n_iter + initial_points,
                      "Evaluating initial point " + std::to_string(i + 1));

        float score = evaluateParameterSet(initial_params[i]);
        initial_params[i].score = score;
        initial_params[i].evaluated = true;

        std::vector<float> point_vec = parameterSetToVector(initial_params[i]);
        observed_points_.push_back(point_vec);
        observed_scores_.push_back(score);

        results.convergence_history.emplace_back(i + 1, score);
    }

    // Bayesian optimization iterations
    logMessage("Starting optimization iterations...");

    for (int iter = 0; iter < config_.bayesian_config.n_iter; ++iter) {
// Cross-validation
float HyperparameterTuner::crossValidateParameters(const ParameterSet& params, const CrossValidationConfig& cv_config) {
    // Simplified cross-validation implementation
    // In practice, this would split the data and train/evaluate multiple times

    std::vector<float> fold_scores;
    fold_scores.reserve(cv_config.k_folds);

    for (int fold = 0; fold < cv_config.k_folds; ++fold) {
        // Evaluate parameter set (simplified - would use different data splits)
        float score = evaluateParameterSet(params);
        fold_scores.push_back(score);
    }

    // Calculate mean and standard deviation
    float mean_score = 0.0f;
    for (float score : fold_scores) {
        mean_score += score;
    }
    mean_score /= fold_scores.size();

    float variance = 0.0f;
    for (float score : fold_scores) {
        float diff = score - mean_score;
        variance += diff * diff;
    }
    variance /= fold_scores.size();

    return mean_score; // Return mean score across folds
}

std::vector<float> HyperparameterTuner::kFoldSplit(int k, int total_samples) const {
    std::vector<float> fold_sizes(k);
    int base_size = total_samples / k;
    int remainder = total_samples % k;

    for (int i = 0; i < k; ++i) {
        fold_sizes[i] = base_size + (i < remainder ? 1 : 0);
    }

    return fold_sizes;
}

// Early stopping
bool HyperparameterTuner::shouldStopEarly(const std::vector<float>& validation_losses,
                                        const EarlyStoppingConfig& config) const {
    if (!config.enabled || validation_losses.size() < config.patience) {
        return false;
    }

    // Check if the best score hasn't improved for patience epochs
    float best_score = *std::min_element(validation_losses.end() - config.patience, validation_losses.end());
    float current_score = validation_losses.back();

    return (best_score - current_score) < config.min_delta;
}

// Main tuning interface
TuningResults HyperparameterTuner::tuneHyperparameters() {
    tuning_start_time_ = std::chrono::steady_clock::now();
    tuning_active_ = true;
    stop_requested_ = false;

    TuningResults results;

    try {
        switch (config_.method) {
            case SearchMethod::GRID_SEARCH:
                results = performGridSearch();
                break;
            case SearchMethod::RANDOM_SEARCH:
                results = performRandomSearch(config_.max_evaluations);
                break;
            case SearchMethod::BAYESIAN_OPTIMIZATION:
                results = performBayesianOptimization();
                break;
            default:
                throw std::runtime_error("Unknown search method");
        }
    } catch (const std::exception& e) {
        logMessage("Error during hyperparameter tuning: " + std::string(e.what()));
        results.metadata["error"] = e.what();
    }

    tuning_active_ = false;
    return results;
}

bool HyperparameterTuner::startTuning() {
    if (tuning_active_) return false;

    tuning_future_ = std::async(std::launch::async, [this]() {
        return tuneHyperparameters();
    });

    return true;
}

bool HyperparameterTuner::isTuning() const {
    return tuning_active_;
}

std::string HyperparameterTuner::getTuningStatus() const {
    return current_status_;
}

bool HyperparameterTuner::stopTuning() {
    if (!tuning_active_) return false;

    stop_requested_ = true;

    if (tuning_future_.valid()) {
        tuning_future_.wait();
    }

    return true;
}

void HyperparameterTuner::resetTuningState() {
    tuning_active_ = false;
    stop_requested_ = false;
    current_evaluation_ = 0;
    total_evaluations_ = 0;
    current_status_ = "";
    observed_points_.clear();
    observed_scores_.clear();
}

TuningResults HyperparameterTuner::getCurrentResults() const {
    // This would need to be implemented to return intermediate results
    // For now, return empty results
    return TuningResults{};
}

// Results management
bool HyperparameterTuner::saveResults(const TuningResults& results, const std::string& filepath) const {
    try {
        nlohmann::json json_results;

        json_results["timestamp"] = results.timestamp;
        json_results["total_time"] = results.total_time;
        json_results["total_evaluations"] = results.total_evaluations;
        json_results["metadata"] = results.metadata;

        // Save best parameters
        nlohmann::json best_params;
        for (const auto& param : results.best_parameters.float_params) {
            best_params["float_params"][param.first] = param.second;
        }
        for (const auto& param : results.best_parameters.int_params) {
            best_params["int_params"][param.first] = param.second;
        }
        for (const auto& param : results.best_parameters.categorical_params) {
            best_params["categorical_params"][param.first] = param.second;
        }
        best_params["score"] = results.best_parameters.score;
        json_results["best_parameters"] = best_params;

        // Save convergence history
        nlohmann::json convergence;
        for (const auto& point : results.convergence_history) {
            convergence.push_back({point.first, point.second});
        }
        json_results["convergence_history"] = convergence;

        // Save all parameter sets (simplified)
        json_results["num_parameter_sets"] = results.parameter_sets.size();

        std::ofstream file(filepath);
        if (!file.is_open()) return false;

        file << json_results.dump(2);
        return true;

    } catch (const std::exception& e) {
        logMessage("Error saving results: " + std::string(e.what()));
        return false;
    }
}

TuningResults HyperparameterTuner::loadResults(const std::string& filepath) const {
    TuningResults results;

    try {
        std::ifstream file(filepath);
        if (!file.is_open()) return results;

        nlohmann::json json_results;
        file >> json_results;

        results.timestamp = json_results.value("timestamp", "");
        results.total_time = json_results.value("total_time", 0.0);
        results.total_evaluations = json_results.value("total_evaluations", 0);
        results.metadata = json_results.value("metadata", nlohmann::json::object());

        // Load best parameters
        if (json_results.contains("best_parameters")) {
            const auto& best_params = json_results["best_parameters"];
            if (best_params.contains("float_params")) {
                for (const auto& param : best_params["float_params"].items()) {
                    results.best_parameters.float_params[param.key()] = param.value();
                }
            }
            if (best_params.contains("int_params")) {
                for (const auto& param : best_params["int_params"].items()) {
                    results.best_parameters.int_params[param.key()] = param.value();
                }
            }
            if (best_params.contains("categorical_params")) {
                for (const auto& param : best_params["categorical_params"].items()) {
                    results.best_parameters.categorical_params[param.key()] = param.value();
                }
            }
            results.best_parameters.score = best_params.value("score", 0.0f);
        }

    } catch (const std::exception& e) {
        logMessage("Error loading results: " + std::string(e.what()));
    }

    return results;
}

// Database operations (simplified - would need actual database schema)
bool HyperparameterTuner::saveTuningResultsToDB(const TuningResults& results, int dataset_id) {
    if (!use_database_) return false;

    // Placeholder - would implement actual database saving
    logMessage("Database saving not implemented yet");
    return false;
}

bool HyperparameterTuner::loadTuningResultsFromDB(int tuning_session_id, TuningResults& results) {
    if (!use_database_) return false;

    // Placeholder - would implement actual database loading
    logMessage("Database loading not implemented yet");
    return false;
}
        if (stop_requested_) break;

        updateProgress(initial_points + iter + 1,
                      config_.bayesian_config.n_iter + initial_points,
                      "Optimization iteration " + std::to_string(iter + 1));

        // Suggest next point
        std::vector<float> next_point = suggestNextPoint();
        ParameterSet next_params = vectorToParameterSet(next_point);

        // Evaluate the suggested point
        float score = evaluateParameterSet(next_params);
        next_params.score = score;
        next_params.evaluated = true;

        // Update observations
        observed_points_.push_back(next_point);
        observed_scores_.push_back(score);

        results.parameter_sets.push_back(next_params);
        results.convergence_history.emplace_back(initial_points + iter + 1, score);
    }

    // Find best parameters
    float best_score = std::numeric_limits<float>::max();
    size_t best_idx = 0;

    for (size_t i = 0; i < results.parameter_sets.size(); ++i) {
        if (results.parameter_sets[i].score < best_score) {
            best_score = results.parameter_sets[i].score;
            best_idx = i;
        }
    }

    // Include initial points in results
    results.parameter_sets.insert(results.parameter_sets.begin(),
                                 initial_params.begin(), initial_params.end());

    results.best_parameters = results.parameter_sets[best_idx];
    results.total_evaluations = static_cast<int>(results.parameter_sets.size());

    auto end_time = std::chrono::steady_clock::now();
    results.total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    logMessage("Bayesian optimization completed. Best score: " + std::to_string(best_score));
    return results;
}
    auto end_time = std::chrono::steady_clock::now();
    results.total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    logMessage("Random search completed. Best score: " + std::to_string(best_score));
    return results;
}
}

// Progress tracking
void HyperparameterTuner::updateProgress(int current, int total, const std::string& status) {
    current_evaluation_ = current;
    total_evaluations_ = total;
    current_status_ = status;

    if (config_.verbose) {
        std::cout << "Progress: " << current << "/" << total << " - " << status << std::endl;
    }
}

void HyperparameterTuner::logMessage(const std::string& message) const {
    if (config_.verbose) {
        std::cout << "[HyperparameterTuner] " << message << std::endl;
    }
}
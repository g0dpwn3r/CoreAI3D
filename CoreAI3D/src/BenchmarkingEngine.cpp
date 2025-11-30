#include "ValidationFramework.hpp"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <chrono>
#include <thread>

// Cross-validation implementation
std::vector<PerformanceMetrics> ValidationFramework::kFoldCrossValidation(
    std::unique_ptr<Training> trainer,
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& targets,
    int k, double learningRate, int epochs) {
    
    std::vector<PerformanceMetrics> results;
    
    if (k <= 1) {
        // Single fold (no cross-validation)
        auto result = evaluateModel(std::move(trainer), data, targets);
        results.push_back(result);
        return results;
    }
    
    size_t totalSize = data.size();
    size_t foldSize = totalSize / k;
    
    // Shuffle indices for random fold assignment
    std::vector<size_t> indices(totalSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), randomGen);
    
    for (int fold = 0; fold < k; fold++) {
        // Create train/test split for this fold
        std::vector<std::vector<float>> trainData, testData;
        std::vector<std::vector<float>> trainTargets, testTargets;
        
        size_t testStart = fold * foldSize;
        size_t testEnd = (fold == k - 1) ? totalSize : (fold + 1) * foldSize;
        
        for (size_t i = 0; i < totalSize; i++) {
            if (i >= testStart && i < testEnd) {
                testData.push_back(data[indices[i]]);
                testTargets.push_back(targets[indices[i]]);
            } else {
                trainData.push_back(data[indices[i]]);
                trainTargets.push_back(targets[indices[i]]);
            }
        }
        
        // Create new trainer for each fold
        auto foldTrainer = std::make_unique<Training>(true, verbose); // offline mode
        
        // Load data into trainer
        if (!foldTrainer->loadCSVFromData(trainData, trainTargets, "cv_fold_" + std::to_string(fold))) {
            continue;
        }
        
        // Preprocess and train
        foldTrainer->preprocess(0.0f, 1.0f);
        foldTrainer->train(learningRate, epochs);
        
        // Evaluate on test set
        auto result = evaluateModel(std::move(foldTrainer), testData, testTargets);
        results.push_back(result);
        
        if (verbose) {
            std::cout << "Fold " << (fold + 1) << "/" << k << " completed" << std::endl;
        }
    }
    
    return results;
}

// Performance evaluation implementation
PerformanceMetrics ValidationFramework::evaluateModel(
    std::unique_ptr<Training> trainer,
    const std::vector<std::vector<float>>& testData,
    const std::vector<std::vector<float>>& testTargets) {
    
    PerformanceMetrics metrics;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Generate predictions using the CoreAI forward method
    auto predictions_matrix = trainer->getCore()->forward(testData, false); // false = not training
    std::vector<float> predictions;
    for (const auto& pred_row : predictions_matrix) {
        if (!pred_row.empty()) {
            predictions.push_back(pred_row[0]); // Assume single output
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    metrics.inferenceTime = duration.count() / 1000.0; // Convert to milliseconds
    
    // Flatten targets for metrics calculation
    std::vector<float> targets;
    for (const auto& targetRow : testTargets) {
        targets.push_back(targetRow[0]); // Assume single output
    }
    
    // Calculate metrics
    metrics.accuracy = calculateAccuracy(predictions, targets);
    metrics.precision = calculatePrecision(predictions, targets);
    metrics.recall = calculateRecall(predictions, targets);
    metrics.f1_score = calculateF1Score(predictions, targets);
    metrics.rmse = calculateRMSE(predictions, targets);
    metrics.mae = calculateMAE(predictions, targets);
    metrics.r2_score = calculateR2Score(predictions, targets);
    
    // Set default values for metrics not computed
    metrics.auc_roc = 0.0f;
    metrics.trainingTime = 0.0; // This would be set during training
    metrics.memoryUsage = 0; // This would be tracked during execution
    metrics.parameters = 0; // This would be calculated based on model architecture
    
    return metrics;
}

// Benchmark against baselines implementation
std::vector<BenchmarkResult> ValidationFramework::benchmarkAgainstBaselines(
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& targets,
    float testSize) {
    
    std::vector<BenchmarkResult> results;
    
    // Split data into train/test
    std::vector<std::vector<float>> trainData, testData;
    std::vector<std::vector<float>> trainTargets, testTargets;
    splitData(data, targets, 1.0f - testSize, trainData, testData, trainTargets, testTargets);
    
    if (verbose) {
        std::cout << "Benchmarking with " << trainData.size() << " training samples and " 
                  << testData.size() << " test samples" << std::endl;
    }
    
    // CoreAI3D Neural Network
    try {
        auto coreAIResult = runCoreAI3D(trainData, trainTargets, testData, testTargets);
        results.push_back(coreAIResult);
    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "CoreAI3D benchmark failed: " << e.what() << std::endl;
        }
    }
    
    // Linear Regression
    try {
        auto linearResult = runLinearRegression(trainData, trainTargets, testData, testTargets);
        results.push_back(linearResult);
    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "Linear Regression benchmark failed: " << e.what() << std::endl;
        }
    }
    
    // Random Forest
    try {
        auto forestResult = runRandomForest(trainData, trainTargets, testData, testTargets);
        results.push_back(forestResult);
    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "Random Forest benchmark failed: " << e.what() << std::endl;
        }
    }
    
    // K-Nearest Neighbors
    try {
        auto knnResult = runKNN(trainData, trainTargets, testData, testTargets);
        results.push_back(knnResult);
    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "KNN benchmark failed: " << e.what() << std::endl;
        }
    }
    
    // Logistic Regression (if classification problem)
    bool isClassification = std::all_of(targets.begin(), targets.end(), 
                                       [](const std::vector<float>& t) { return t[0] == static_cast<int>(t[0]); });
    if (isClassification) {
        try {
            auto logisticResult = runLogisticRegression(trainData, trainTargets, testData, testTargets);
            results.push_back(logisticResult);
        } catch (const std::exception& e) {
            if (verbose) {
                std::cout << "Logistic Regression benchmark failed: " << e.what() << std::endl;
            }
        }
    }
    
    return results;
}

// CoreAI3D benchmark implementation
BenchmarkResult ValidationFramework::runCoreAI3D(
    const std::vector<std::vector<float>>& trainData,
    const std::vector<std::vector<float>>& trainTargets,
    const std::vector<std::vector<float>>& testData,
    const std::vector<std::vector<float>>& testTargets) {
    
    BenchmarkResult result;
    result.modelName = "CoreAI3D Neural Network";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Create and train CoreAI3D model
        auto trainer = std::make_unique<Training>(true, verbose); // offline mode
        
        // Load training data
        if (!trainer->loadCSVFromData(trainData, trainTargets, "coreai3d_benchmark")) {
            throw std::runtime_error("Failed to load training data");
        }
        
        // Auto-tune hyperparameters
        auto complexity = trainer->detectProblemComplexity();
        trainer->applyAutoTuning(complexity);
        
        // Train the model
        auto trainingStart = std::chrono::high_resolution_clock::now();
        trainer->preprocess(0.0f, 1.0f);
        trainer->train(trainer->learningRate, 50); // 50 epochs
        auto trainingEnd = std::chrono::high_resolution_clock::now();
        
        // Evaluate on test set
        auto metrics = evaluateModel(std::move(trainer), testData, testTargets);
        result.metrics = metrics;
        
        auto endTime = std::chrono::high_resolution_clock::now();
        result.totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        result.trainingTime = std::chrono::duration_cast<std::chrono::microseconds>(trainingEnd - trainingStart);
        
        // Extract predictions for comparison
        auto testTrainer = std::make_unique<Training>(true, verbose);
        testTrainer->loadCSVFromData(trainData, trainTargets, "coreai3d_test");
        testTrainer->preprocess(0.0f, 1.0f);

        // Use forward method for predictions
        auto predictions_matrix = testTrainer->getCore()->forward(testData, false);
        for (const auto& pred_row : predictions_matrix) {
            if (!pred_row.empty()) {
                result.predictions.push_back(pred_row[0]);
            }
        }
        
        for (const auto& target : testTargets) {
            result.targets.push_back(target[0]);
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error("CoreAI3D execution failed: " + std::string(e.what()));
    }
    
    return result;
}

// Linear Regression baseline implementation
BenchmarkResult ValidationFramework::runLinearRegression(
    const std::vector<std::vector<float>>& trainData,
    const std::vector<std::vector<float>>& trainTargets,
    const std::vector<std::vector<float>>& testData,
    const std::vector<std::vector<float>>& testTargets) {
    
    BenchmarkResult result;
    result.modelName = "Linear Regression";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        // Prepare data for linear regression
        size_t nFeatures = trainData[0].size();
        size_t nSamples = trainData.size();
        
        // Add bias term (column of ones)
        std::vector<std::vector<float>> X_train(nSamples, std::vector<float>(nFeatures + 1, 1.0f));
        for (size_t i = 0; i < nSamples; i++) {
            for (size_t j = 0; j < nFeatures; j++) {
                X_train[i][j + 1] = trainData[i][j];
            }
        }
        
        // Extract targets
        std::vector<float> y_train(nSamples);
        for (size_t i = 0; i < nSamples; i++) {
            y_train[i] = trainTargets[i][0];
        }
        
        auto trainingStart = std::chrono::high_resolution_clock::now();
        
        // Simple gradient descent for linear regression
        std::vector<float> weights(nFeatures + 1, 0.0f);
        float learningRate = 0.01f;
        int epochs = 1000;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            std::vector<float> gradients(nFeatures + 1, 0.0f);
            
            // Compute gradients
            for (size_t i = 0; i < nSamples; i++) {
                float prediction = 0.0f;
                for (size_t j = 0; j < nFeatures + 1; j++) {
                    prediction += weights[j] * X_train[i][j];
                }
                float error = prediction - y_train[i];
                
                for (size_t j = 0; j < nFeatures + 1; j++) {
                    gradients[j] += error * X_train[i][j];
                }
            }
            
            // Update weights
            for (size_t j = 0; j < weights.size(); j++) {
                weights[j] -= learningRate * gradients[j] / nSamples;
            }
        }
        
        auto trainingEnd = std::chrono::high_resolution_clock::now();
        
        // Make predictions on test set
        for (const auto& testSample : testData) {
            float prediction = weights[0]; // bias term
            for (size_t j = 0; j < nFeatures; j++) {
                prediction += weights[j + 1] * testSample[j];
            }
            result.predictions.push_back(prediction);
        }
        
        // Extract test targets
        for (const auto& testTarget : testTargets) {
            result.targets.push_back(testTarget[0]);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        result.totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        result.trainingTime = std::chrono::duration_cast<std::chrono::microseconds>(trainingEnd - trainingStart);
        
        // Calculate metrics
        result.metrics.accuracy = calculateAccuracy(result.predictions, result.targets);
        result.metrics.precision = calculatePrecision(result.predictions, result.targets);
        result.metrics.recall = calculateRecall(result.predictions, result.targets);
        result.metrics.f1_score = calculateF1Score(result.predictions, result.targets);
        result.metrics.rmse = calculateRMSE(result.predictions, result.targets);
        result.metrics.mae = calculateMAE(result.predictions, result.targets);
        result.metrics.r2_score = calculateR2Score(result.predictions, result.targets);
        result.metrics.parameters = weights.size();
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Linear Regression execution failed: " + std::string(e.what()));
    }
    
    return result;
}

// Random Forest baseline implementation (simplified)
BenchmarkResult ValidationFramework::runRandomForest(
    const std::vector<std::vector<float>>& trainData,
    const std::vector<std::vector<float>>& trainTargets,
    const std::vector<std::vector<float>>& testData,
    const std::vector<std::vector<float>>& testTargets,
    int numTrees, int maxDepth) {
    
    BenchmarkResult result;
    result.modelName = "Random Forest";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        size_t nFeatures = trainData[0].size();
        size_t nSamples = trainData.size();
        
        auto trainingStart = std::chrono::high_resolution_clock::now();
        
        // Simple decision tree implementation (very basic)
        struct DecisionTree {
            struct Node {
                int featureIndex;
                float threshold;
                std::unique_ptr<Node> left;
                std::unique_ptr<Node> right;
                float prediction;
                bool isLeaf;
            };
            
            std::unique_ptr<Node> root;
            int maxDepth;
            
            void train(const std::vector<std::vector<float>>& data, 
                      const std::vector<float>& targets, int depth = 0) {
                maxDepth = 5; // Limit depth for simplicity
                root = buildTree(data, targets, depth);
            }
            
            std::unique_ptr<Node> buildTree(const std::vector<std::vector<float>>& data,
                                          const std::vector<float>& targets, int depth) {
                auto node = std::make_unique<Node>();
                
                // Check stopping criteria
                if (depth >= maxDepth || data.size() < 2) {
                    node->isLeaf = true;
                    node->prediction = std::accumulate(targets.begin(), targets.end(), 0.0f) / targets.size();
                    return node;
                }
                
                // Find best split (simplified - just use first feature)
                node->featureIndex = 0;
                
                // Find threshold (mean of feature values)
                float sum = 0.0f;
                for (const auto& sample : data) {
                    sum += sample[0];
                }
                node->threshold = sum / data.size();
                
                // Split data
                std::vector<std::vector<float>> leftData, rightData;
                std::vector<float> leftTargets, rightTargets;
                
                for (size_t i = 0; i < data.size(); i++) {
                    if (data[i][0] <= node->threshold) {
                        leftData.push_back(data[i]);
                        leftTargets.push_back(targets[i]);
                    } else {
                        rightData.push_back(data[i]);
                        rightTargets.push_back(targets[i]);
                    }
                }
                
                // Create children
                if (!leftData.empty() && !rightData.empty()) {
                    node->left = buildTree(leftData, leftTargets, depth + 1);
                    node->right = buildTree(rightData, rightTargets, depth + 1);
                } else {
                    node->isLeaf = true;
                    node->prediction = std::accumulate(targets.begin(), targets.end(), 0.0f) / targets.size();
                }
                
                return node;
            }
            
            float predict(const std::vector<float>& sample) const {
                const Node* current = root.get();
                
                while (current && !current->isLeaf) {
                    if (sample[current->featureIndex] <= current->threshold) {
                        current = current->left.get();
                    } else {
                        current = current->right.get();
                    }
                }
                
                return current ? current->prediction : 0.0f;
            }
        };
        
        // Extract targets as vector<float> for training
        std::vector<float> targets_flat;
        for (const auto& target_row : trainTargets) {
            if (!target_row.empty()) {
                targets_flat.push_back(target_row[0]);
            }
        }

        // Train multiple trees (simplified - same data for all trees with some randomness)
        std::vector<DecisionTree> trees;
        for (int t = 0; t < std::min(numTrees, 3); t++) { // Limit to 3 trees for simplicity
            DecisionTree tree;
            tree.train(trainData, targets_flat);
            trees.push_back(std::move(tree));
        }
        
        auto trainingEnd = std::chrono::high_resolution_clock::now();
        
        // Make predictions (average across trees)
        for (const auto& testSample : testData) {
            float sumPrediction = 0.0f;
            for (const auto& tree : trees) {
                sumPrediction += tree.predict(testSample);
            }
            result.predictions.push_back(sumPrediction / trees.size());
        }
        
        // Extract test targets
        for (const auto& testTarget : testTargets) {
            result.targets.push_back(testTarget[0]);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        result.totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        result.trainingTime = std::chrono::duration_cast<std::chrono::microseconds>(trainingEnd - trainingStart);
        
        // Calculate metrics
        result.metrics.accuracy = calculateAccuracy(result.predictions, result.targets);
        result.metrics.precision = calculatePrecision(result.predictions, result.targets);
        result.metrics.recall = calculateRecall(result.predictions, result.targets);
        result.metrics.f1_score = calculateF1Score(result.predictions, result.targets);
        result.metrics.rmse = calculateRMSE(result.predictions, result.targets);
        result.metrics.mae = calculateMAE(result.predictions, result.targets);
        result.metrics.r2_score = calculateR2Score(result.predictions, result.targets);
        result.metrics.parameters = trees.size() * 10; // Approximate parameter count
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Random Forest execution failed: " + std::string(e.what()));
    }
    
    return result;
}

// K-Nearest Neighbors baseline implementation
BenchmarkResult ValidationFramework::runKNN(
    const std::vector<std::vector<float>>& trainData,
    const std::vector<std::vector<float>>& trainTargets,
    const std::vector<std::vector<float>>& testData,
    const std::vector<std::vector<float>>& testTargets,
    int k) {
    
    BenchmarkResult result;
    result.modelName = "K-Nearest Neighbors";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        size_t nFeatures = trainData[0].size();
        
        auto trainingStart = std::chrono::high_resolution_clock::now();
        // KNN has no training phase - just store the data
        auto trainingEnd = std::chrono::high_resolution_clock::now();
        
        // Make predictions for each test sample
        for (const auto& testSample : testData) {
            // Calculate distances to all training samples
            std::vector<std::pair<float, float>> distances; // (distance, target)
            
            for (size_t i = 0; i < trainData.size(); i++) {
                float distance = 0.0f;
                for (size_t j = 0; j < nFeatures; j++) {
                    float diff = testSample[j] - trainData[i][j];
                    distance += diff * diff;
                }
                distance = std::sqrt(distance);
                distances.push_back({distance, trainTargets[i][0]});
            }
            
            // Sort by distance and take k nearest neighbors
            std::partial_sort(distances.begin(), distances.begin() + k, distances.end(),
                            [](const auto& a, const auto& b) { return a.first < b.first; });
            
            // Average the targets of k nearest neighbors
            float sum = 0.0f;
            for (int i = 0; i < std::min(k, (int)distances.size()); i++) {
                sum += distances[i].second;
            }
            result.predictions.push_back(sum / std::min(k, (int)distances.size()));
        }
        
        // Extract test targets
        for (const auto& testTarget : testTargets) {
            result.targets.push_back(testTarget[0]);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        result.totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        result.trainingTime = std::chrono::duration_cast<std::chrono::microseconds>(trainingEnd - trainingStart);
        
        // Calculate metrics
        result.metrics.accuracy = calculateAccuracy(result.predictions, result.targets);
        result.metrics.precision = calculatePrecision(result.predictions, result.targets);
        result.metrics.recall = calculateRecall(result.predictions, result.targets);
        result.metrics.f1_score = calculateF1Score(result.predictions, result.targets);
        result.metrics.rmse = calculateRMSE(result.predictions, result.targets);
        result.metrics.mae = calculateMAE(result.predictions, result.targets);
        result.metrics.r2_score = calculateR2Score(result.predictions, result.targets);
        result.metrics.parameters = trainData.size() * nFeatures; // All training data stored
        
    } catch (const std::exception& e) {
        throw std::runtime_error("KNN execution failed: " + std::string(e.what()));
    }
    
    return result;
}

// Logistic Regression baseline implementation
BenchmarkResult ValidationFramework::runLogisticRegression(
    const std::vector<std::vector<float>>& trainData,
    const std::vector<std::vector<float>>& trainTargets,
    const std::vector<std::vector<float>>& testData,
    const std::vector<std::vector<float>>& testTargets) {
    
    BenchmarkResult result;
    result.modelName = "Logistic Regression";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    try {
        size_t nFeatures = trainData[0].size();
        size_t nSamples = trainData.size();
        
        // Add bias term
        std::vector<std::vector<float>> X_train(nSamples, std::vector<float>(nFeatures + 1, 1.0f));
        for (size_t i = 0; i < nSamples; i++) {
            for (size_t j = 0; j < nFeatures; j++) {
                X_train[i][j + 1] = trainData[i][j];
            }
        }
        
        // Convert targets to binary (0 or 1)
        std::vector<float> y_train(nSamples);
        for (size_t i = 0; i < nSamples; i++) {
            y_train[i] = trainTargets[i][0] > 0.5f ? 1.0f : 0.0f;
        }
        
        auto trainingStart = std::chrono::high_resolution_clock::now();
        
        // Initialize weights
        std::vector<float> weights(nFeatures + 1, 0.0f);
        float learningRate = 0.01f;
        int epochs = 1000;
        
        auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            std::vector<float> gradients(nFeatures + 1, 0.0f);
            
            // Compute gradients
            for (size_t i = 0; i < nSamples; i++) {
                float linear = 0.0f;
                for (size_t j = 0; j < nFeatures + 1; j++) {
                    linear += weights[j] * X_train[i][j];
                }
                float prediction = sigmoid(linear);
                float error = prediction - y_train[i];
                
                for (size_t j = 0; j < nFeatures + 1; j++) {
                    gradients[j] += error * X_train[i][j];
                }
            }
            
            // Update weights
            for (size_t j = 0; j < weights.size(); j++) {
                weights[j] -= learningRate * gradients[j] / nSamples;
            }
        }
        
        auto trainingEnd = std::chrono::high_resolution_clock::now();
        
        // Make predictions on test set
        for (const auto& testSample : testData) {
            float linear = weights[0]; // bias term
            for (size_t j = 0; j < nFeatures; j++) {
                linear += weights[j + 1] * testSample[j];
            }
            result.predictions.push_back(sigmoid(linear));
        }
        
        // Extract test targets (convert to binary)
        for (const auto& testTarget : testTargets) {
            result.targets.push_back(testTarget[0] > 0.5f ? 1.0f : 0.0f);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        result.totalTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        result.trainingTime = std::chrono::duration_cast<std::chrono::microseconds>(trainingEnd - trainingStart);
        
        // Calculate metrics
        result.metrics.accuracy = calculateAccuracy(result.predictions, result.targets);
        result.metrics.precision = calculatePrecision(result.predictions, result.targets);
        result.metrics.recall = calculateRecall(result.predictions, result.targets);
        result.metrics.f1_score = calculateF1Score(result.predictions, result.targets);
        result.metrics.rmse = calculateRMSE(result.predictions, result.targets);
        result.metrics.mae = calculateMAE(result.predictions, result.targets);
        result.metrics.r2_score = calculateR2Score(result.predictions, result.targets);
        result.metrics.parameters = weights.size();
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Logistic Regression execution failed: " + std::string(e.what()));
    }
    
    return result;
}
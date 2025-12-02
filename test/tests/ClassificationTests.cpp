// test/tests/ClassificationTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>

#include "Core.hpp"
#include "../utils/TestHelpers.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// Classification metrics structure
struct ClassificationMetrics {
    float accuracy;
    float precision;
    float recall;
    float f1_score;
    std::vector<std::vector<int>> confusion_matrix;
};

// Calculate classification metrics for binary classification
ClassificationMetrics calculateBinaryMetrics(const std::vector<float>& predictions, const std::vector<float>& targets, float threshold = 0.5f) {
    ClassificationMetrics metrics;
    int tp = 0, tn = 0, fp = 0, fn = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        bool pred_positive = predictions[i] >= threshold;
        bool true_positive = targets[i] >= threshold;

        if (pred_positive && true_positive) tp++;
        else if (!pred_positive && !true_positive) tn++;
        else if (pred_positive && !true_positive) fp++;
        else if (!pred_positive && true_positive) fn++;
    }

    metrics.confusion_matrix = {{tp, fp}, {fn, tn}};
    metrics.accuracy = static_cast<float>(tp + tn) / (tp + tn + fp + fn);
    metrics.precision = tp + fp > 0 ? static_cast<float>(tp) / (tp + fp) : 0.0f;
    metrics.recall = tp + fn > 0 ? static_cast<float>(tp) / (tp + fn) : 0.0f;
    metrics.f1_score = metrics.precision + metrics.recall > 0 ?
                      2.0f * metrics.precision * metrics.recall / (metrics.precision + metrics.recall) : 0.0f;

    return metrics;
}

// Calculate accuracy for multi-class classification
float calculateMultiClassAccuracy(const std::vector<std::vector<float>>& predictions, const std::vector<std::vector<float>>& targets) {
    int correct = 0;
    int total = predictions.size();

    for (size_t i = 0; i < predictions.size(); ++i) {
        // Find predicted class (highest output)
        size_t pred_class = 0;
        float max_pred = predictions[i][0];
        for (size_t j = 1; j < predictions[i].size(); ++j) {
            if (predictions[i][j] > max_pred) {
                max_pred = predictions[i][j];
                pred_class = j;
            }
        }

        // Find true class (highest target)
        size_t true_class = 0;
        float max_true = targets[i][0];
        for (size_t j = 1; j < targets[i].size(); ++j) {
            if (targets[i][j] > max_true) {
                max_true = targets[i][j];
                true_class = j;
            }
        }

        if (pred_class == true_class) correct++;
    }

    return static_cast<float>(correct) / total;
}

// Generate binary classification data: separable by x > 0
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generateBinaryData(int num_samples = 200) {
    std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(2));
    std::vector<std::vector<float>> targets(num_samples, std::vector<float>(1));

    for (int i = 0; i < num_samples; ++i) {
        float x1 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
        float x2 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.2f;

        // Class 1 if x1 + x2 + noise > 0, else class 0
        float decision = x1 + x2 + noise;
        float target = decision > 0 ? 1.0f : 0.0f;

        inputs[i][0] = x1;
        inputs[i][1] = x2;
        targets[i][0] = target;
    }

    return {inputs, targets};
}

// Generate multi-class classification data (3 classes)
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generateMultiClassData(int num_samples = 300) {
    std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(2));
    std::vector<std::vector<float>> targets(num_samples, std::vector<float>(3)); // One-hot encoded

    for (int i = 0; i < num_samples; ++i) {
        float x1 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 6.0f;
        float x2 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 6.0f;

        // Classify based on angle and distance
        float angle = std::atan2(x2, x1);
        float distance = std::sqrt(x1*x1 + x2*x2);

        int class_label;
        if (distance < 2.0f) {
            class_label = 0; // Center class
        } else if (angle > 0) {
            class_label = 1; // Upper half
        } else {
            class_label = 2; // Lower half
        }

        inputs[i][0] = x1;
        inputs[i][1] = x2;

        // One-hot encoding
        targets[i][0] = (class_label == 0) ? 1.0f : 0.0f;
        targets[i][1] = (class_label == 1) ? 1.0f : 0.0f;
        targets[i][2] = (class_label == 2) ? 1.0f : 0.0f;
    }

    return {inputs, targets};
}

// Generate imbalanced binary classification data
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generateImbalancedData(int num_samples = 1000) {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    // 90% class 0, 10% class 1
    int class0_samples = static_cast<int>(num_samples * 0.9f);
    int class1_samples = num_samples - class0_samples;

    // Generate class 0 samples (majority class)
    for (int i = 0; i < class0_samples; ++i) {
        float x1 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;
        float x2 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f;

        // Class 0: around origin
        inputs.push_back({x1 * 0.5f, x2 * 0.5f});
        targets.push_back({0.0f});
    }

    // Generate class 1 samples (minority class)
    for (int i = 0; i < class1_samples; ++i) {
        float x1 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f + 3.0f; // Shifted right
        float x2 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 4.0f + 3.0f; // Shifted up

        inputs.push_back({x1, x2});
        targets.push_back({1.0f});
    }

    return {inputs, targets};
}

TEST_CASE("Classification Tests: Binary Classification", "[classification]") {
    CoreAI core(2, 2, 12, 1, -5.0f, 5.0f);

    SECTION("Basic binary classification") {
        auto [inputs, targets] = generateBinaryData(300);

        // Split into train/test
        std::vector<std::vector<float>> train_inputs(inputs.begin(), inputs.begin() + 200);
        std::vector<std::vector<float>> train_targets(targets.begin(), targets.begin() + 200);
        std::vector<std::vector<float>> test_inputs(inputs.begin() + 200, inputs.end());
        std::vector<std::vector<float>> test_targets(targets.begin() + 200, targets.end());

        // Set sigmoid activation for output layer
        core.setLayerActivation(1, ActivationType::SIGMOID);

        // Train the model
        int num_samples = train_inputs.size();
        core.train(train_inputs, train_targets, 0.01, num_samples, 32);

        // Make predictions
        auto predictions = core.forward(test_inputs);

        // Flatten predictions and targets
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(test_targets[i][0]);
        }

        // Calculate metrics
        auto metrics = calculateBinaryMetrics(flat_predictions, flat_targets);

        INFO("Binary Classification Metrics:");
        INFO("Accuracy: " << metrics.accuracy);
        INFO("Precision: " << metrics.precision);
        INFO("Recall: " << metrics.recall);
        INFO("F1-Score: " << metrics.f1_score);

        // Should achieve reasonable performance
        REQUIRE(metrics.accuracy > 0.75f);
        REQUIRE(metrics.precision > 0.7f);
        REQUIRE(metrics.recall > 0.7f);
        REQUIRE(metrics.f1_score > 0.7f);
    }

    SECTION("Binary classification with different thresholds") {
        auto [inputs, targets] = generateBinaryData(200);

        CoreAI test_core(2, 2, 12, 1, -5.0f, 5.0f);
        test_core.setLayerActivation(1, ActivationType::SIGMOID);

        int num_samples = inputs.size();
        test_core.train(inputs, targets, 0.01, num_samples, 32);

        auto predictions = test_core.forward(inputs);
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(targets[i][0]);
        }

        // Test different thresholds
        std::vector<float> thresholds = {0.3f, 0.5f, 0.7f};
        for (float threshold : thresholds) {
            auto metrics = calculateBinaryMetrics(flat_predictions, flat_targets, threshold);
            REQUIRE(metrics.accuracy > 0.6f); // Should work reasonably at different thresholds
        }
    }
}

TEST_CASE("Classification Tests: Multi-class Classification", "[classification]") {
    CoreAI core(2, 3, 15, 3, -8.0f, 8.0f);

    SECTION("Multi-class classification with 3 classes") {
        auto [inputs, targets] = generateMultiClassData(400);

        // Split into train/test
        std::vector<std::vector<float>> train_inputs(inputs.begin(), inputs.begin() + 300);
        std::vector<std::vector<float>> train_targets(targets.begin(), targets.begin() + 300);
        std::vector<std::vector<float>> test_inputs(inputs.begin() + 300, inputs.end());
        std::vector<std::vector<float>> test_targets(targets.begin() + 300, targets.end());

        // Set sigmoid activations for output layer (multi-label approach)
        core.setAllLayerActivations(ActivationType::SIGMOID);

        // Train the model
        int num_samples = train_inputs.size();
        core.train(train_inputs, train_targets, 0.005, num_samples, 32);

        // Make predictions
        auto predictions = core.forward(test_inputs);

        // Calculate accuracy
        float accuracy = calculateMultiClassAccuracy(predictions, test_targets);

        INFO("Multi-class Accuracy: " << accuracy);

        // Should achieve reasonable multi-class performance
        REQUIRE(accuracy > 0.65f);
    }

    SECTION("Multi-class with different architectures") {
        auto [inputs, targets] = generateMultiClassData(200);

        // Test with deeper network
        CoreAI deep_core(2, 4, 20, 3, -8.0f, 8.0f);
        deep_core.setAllLayerActivations(ActivationType::SIGMOID);

        int num_samples = inputs.size();
        deep_core.train(inputs, targets, 0.005, num_samples, 16);

        auto predictions = deep_core.forward(inputs);
        float accuracy = calculateMultiClassAccuracy(predictions, targets);

        REQUIRE(accuracy > 0.6f);
    }
}

TEST_CASE("Classification Tests: Imbalanced Datasets", "[classification]") {
    CoreAI core(2, 2, 12, 1, -10.0f, 10.0f);

    SECTION("Handling imbalanced binary classification") {
        auto [inputs, targets] = generateImbalancedData(800);

        // Count class distribution
        int class0_count = 0, class1_count = 0;
        for (const auto& target : targets) {
            if (target[0] < 0.5f) class0_count++;
            else class1_count++;
        }

        INFO("Class distribution - Class 0: " << class0_count << ", Class 1: " << class1_count);
        REQUIRE(class0_count > class1_count * 5); // Confirm imbalance

        // Split into train/test
        std::vector<std::vector<float>> train_inputs(inputs.begin(), inputs.begin() + 600);
        std::vector<std::vector<float>> train_targets(targets.begin(), targets.begin() + 600);
        std::vector<std::vector<float>> test_inputs(inputs.begin() + 600, inputs.end());
        std::vector<std::vector<float>> test_targets(targets.begin() + 600, targets.end());

        core.setLayerActivation(1, ActivationType::SIGMOID);

        // Train the model
        int num_samples = train_inputs.size();
        core.train(train_inputs, train_targets, 0.01, num_samples, 32);

        // Make predictions
        auto predictions = core.forward(test_inputs);
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(test_targets[i][0]);
        }

        auto metrics = calculateBinaryMetrics(flat_predictions, flat_targets);

        INFO("Imbalanced Dataset Metrics:");
        INFO("Accuracy: " << metrics.accuracy);
        INFO("Precision: " << metrics.precision);
        INFO("Recall: " << metrics.recall);
        INFO("F1-Score: " << metrics.f1_score);

        // With imbalance, accuracy might be high due to majority class
        // but precision/recall for minority class should be reasonable
        REQUIRE(metrics.accuracy > 0.7f); // High accuracy due to majority class
        REQUIRE(metrics.precision > 0.5f); // Should detect minority class reasonably
        REQUIRE(metrics.recall > 0.4f); // Should find some minority class instances
    }

    SECTION("Imbalanced classification with class weights") {
        auto [inputs, targets] = generateImbalancedData(400);

        // For now, just test that the model can still learn
        // (Implementing class weights would require modifying the training loop)
        CoreAI weighted_core(2, 2, 12, 1, -10.0f, 10.0f);
        weighted_core.setLayerActivation(1, ActivationType::SIGMOID);

        int num_samples = inputs.size();
        weighted_core.train(inputs, targets, 0.005, num_samples, 32);

        auto predictions = weighted_core.forward(inputs);
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(targets[i][0]);
        }

        auto metrics = calculateBinaryMetrics(flat_predictions, flat_targets);
        REQUIRE(metrics.accuracy > 0.65f);
    }
}

TEST_CASE("Classification Tests: Realistic Use Cases", "[classification]") {
    SECTION("Medical diagnosis simulation (binary)") {
        // Simulate medical diagnosis: predict disease based on symptoms
        // Features: fever, cough, fatigue, age
        // Target: has_disease (0/1)

        int num_samples = 500;
        std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(4));
        std::vector<std::vector<float>> targets(num_samples, std::vector<float>(1));

        for (int i = 0; i < num_samples; ++i) {
            float age = 20.0f + (static_cast<float>(rand()) / RAND_MAX) * 60.0f;
            float fever = (static_cast<float>(rand()) / RAND_MAX) * 5.0f; // 0-5 fever level
            float cough = static_cast<float>(rand()) / RAND_MAX; // 0-1 cough severity
            float fatigue = static_cast<float>(rand()) / RAND_MAX; // 0-1 fatigue level

            // Disease probability based on symptoms
            float disease_prob = 0.1f + 0.3f * (fever / 5.0f) + 0.4f * cough + 0.2f * fatigue + 0.1f * (age / 80.0f);
            float has_disease = (static_cast<float>(rand()) / RAND_MAX) < disease_prob ? 1.0f : 0.0f;

            inputs[i][0] = age;
            inputs[i][1] = fever;
            inputs[i][2] = cough;
            inputs[i][3] = fatigue;
            targets[i][0] = has_disease;
        }

        CoreAI medical_core(4, 2, 16, 1, 0.0f, 100.0f);
        medical_core.setLayerActivation(1, ActivationType::SIGMOID);

        // Normalize inputs
        medical_core.normalizeInput(0.0f, 100.0f);

        int num_samples_train = 350;
        std::vector<std::vector<float>> train_inputs(inputs.begin(), inputs.begin() + num_samples_train);
        std::vector<std::vector<float>> train_targets(targets.begin(), targets.begin() + num_samples_train);
        std::vector<std::vector<float>> test_inputs(inputs.begin() + num_samples_train, inputs.end());
        std::vector<std::vector<float>> test_targets(targets.begin() + num_samples_train, targets.end());

        medical_core.train(train_inputs, train_targets, 0.005, num_samples_train, 32);

        auto predictions = medical_core.forward(test_inputs);
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(test_targets[i][0]);
        }

        auto metrics = calculateBinaryMetrics(flat_predictions, flat_targets);

        INFO("Medical Diagnosis Metrics:");
        INFO("Accuracy: " << metrics.accuracy);
        INFO("Precision: " << metrics.precision);
        INFO("Recall: " << metrics.recall);
        INFO("F1-Score: " << metrics.f1_score);

        // Medical diagnosis should have reasonable performance
        REQUIRE(metrics.accuracy > 0.7f);
        REQUIRE(metrics.precision > 0.6f);
        REQUIRE(metrics.recall > 0.6f);
    }

    SECTION("Customer churn prediction (binary)") {
        // Simulate customer churn prediction
        // Features: account_age, monthly_spend, support_calls, satisfaction
        // Target: will_churn (0/1)

        int num_samples = 600;
        std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(4));
        std::vector<std::vector<float>> targets(num_samples, std::vector<float>(1));

        for (int i = 0; i < num_samples; ++i) {
            float account_age = (static_cast<float>(rand()) / RAND_MAX) * 60.0f; // months
            float monthly_spend = 20.0f + (static_cast<float>(rand()) / RAND_MAX) * 180.0f; // $20-$200
            float support_calls = (static_cast<float>(rand()) / RAND_MAX) * 10.0f; // 0-10 calls
            float satisfaction = (static_cast<float>(rand()) / RAND_MAX) * 5.0f; // 0-5 rating

            // Churn probability
            float churn_prob = 0.05f + 0.2f * (support_calls / 10.0f) - 0.15f * (satisfaction / 5.0f) +
                             0.1f * (account_age / 60.0f) + 0.05f * (monthly_spend / 200.0f);
            churn_prob = std::max(0.0f, std::min(1.0f, churn_prob));
            float will_churn = (static_cast<float>(rand()) / RAND_MAX) < churn_prob ? 1.0f : 0.0f;

            inputs[i][0] = account_age;
            inputs[i][1] = monthly_spend;
            inputs[i][2] = support_calls;
            inputs[i][3] = satisfaction;
            targets[i][0] = will_churn;
        }

        CoreAI churn_core(4, 2, 16, 1, 0.0f, 200.0f);
        churn_core.setLayerActivation(1, ActivationType::SIGMOID);

        int num_samples_train = 400;
        std::vector<std::vector<float>> train_inputs(inputs.begin(), inputs.begin() + num_samples_train);
        std::vector<std::vector<float>> train_targets(targets.begin(), targets.begin() + num_samples_train);
        std::vector<std::vector<float>> test_inputs(inputs.begin() + num_samples_train, inputs.end());
        std::vector<std::vector<float>> test_targets(targets.begin() + num_samples_train, targets.end());

        churn_core.train(train_inputs, train_targets, 0.005, num_samples_train, 32);

        auto predictions = churn_core.forward(test_inputs);
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(test_targets[i][0]);
        }

        auto metrics = calculateBinaryMetrics(flat_predictions, flat_targets);

        INFO("Customer Churn Prediction Metrics:");
        INFO("Accuracy: " << metrics.accuracy);
        INFO("Precision: " << metrics.precision);
        INFO("Recall: " << metrics.recall);
        INFO("F1-Score: " << metrics.f1_score);

        // Churn prediction should have reasonable performance
        REQUIRE(metrics.accuracy > 0.65f);
        REQUIRE(metrics.precision > 0.5f);
        REQUIRE(metrics.recall > 0.5f);
    }
}
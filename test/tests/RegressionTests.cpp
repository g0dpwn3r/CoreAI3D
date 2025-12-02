// test/tests/RegressionTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>

#include "Core.hpp"
#include "../utils/TestHelpers.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// Helper function to calculate Mean Squared Error
float calculateMSE(const std::vector<float>& predictions, const std::vector<float>& targets) {
    float mse = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float diff = predictions[i] - targets[i];
        mse += diff * diff;
    }
    return mse / predictions.size();
}

// Helper function to calculate Mean Absolute Error
float calculateMAE(const std::vector<float>& predictions, const std::vector<float>& targets) {
    float mae = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        mae += std::abs(predictions[i] - targets[i]);
    }
    return mae / predictions.size();
}

// Helper function to calculate R-squared
float calculateRSquared(const std::vector<float>& predictions, const std::vector<float>& targets) {
    float mean_target = std::accumulate(targets.begin(), targets.end(), 0.0f) / targets.size();

    float ss_res = 0.0f;
    float ss_tot = 0.0f;

    for (size_t i = 0; i < predictions.size(); ++i) {
        ss_res += (predictions[i] - targets[i]) * (predictions[i] - targets[i]);
        ss_tot += (targets[i] - mean_target) * (targets[i] - mean_target);
    }

    return 1.0f - (ss_res / ss_tot);
}

// Generate synthetic linear regression data: y = 2*x + 1 + noise
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generateLinearData(int num_samples = 100) {
    std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(1));
    std::vector<std::vector<float>> targets(num_samples, std::vector<float>(1));

    for (int i = 0; i < num_samples; ++i) {
        float x = static_cast<float>(i) / num_samples * 10.0f - 5.0f; // x in [-5, 5]
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.5f; // small noise
        float y = 2.0f * x + 1.0f + noise;

        inputs[i][0] = x;
        targets[i][0] = y;
    }

    return {inputs, targets};
}

// Generate synthetic polynomial regression data: y = x^2 + 2*x + 1 + noise
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generatePolynomialData(int num_samples = 100) {
    std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(1));
    std::vector<std::vector<float>> targets(num_samples, std::vector<float>(1));

    for (int i = 0; i < num_samples; ++i) {
        float x = static_cast<float>(i) / num_samples * 8.0f - 4.0f; // x in [-4, 4]
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 1.0f;
        float y = x*x + 2.0f*x + 1.0f + noise;

        inputs[i][0] = x;
        targets[i][0] = y;
    }

    return {inputs, targets};
}

// Generate synthetic multi-variable regression data: y = 2*x1 + 3*x2 - x3 + 1 + noise
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generateMultiVariableData(int num_samples = 100) {
    std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(3));
    std::vector<std::vector<float>> targets(num_samples, std::vector<float>(1));

    for (int i = 0; i < num_samples; ++i) {
        float x1 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
        float x2 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
        float x3 = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 10.0f;
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f;
        float y = 2.0f*x1 + 3.0f*x2 - x3 + 1.0f + noise;

        inputs[i][0] = x1;
        inputs[i][1] = x2;
        inputs[i][2] = x3;
        targets[i][0] = y;
    }

    return {inputs, targets};
}

TEST_CASE("Regression Tests: Linear Regression", "[regression]") {
    CoreAI core(1, 2, 10, 1, -10.0f, 10.0f);

    SECTION("Basic linear regression training and prediction") {
        auto [inputs, targets] = generateLinearData(200);

        // Split into train/test
        std::vector<std::vector<float>> train_inputs(inputs.begin(), inputs.begin() + 150);
        std::vector<std::vector<float>> train_targets(targets.begin(), targets.begin() + 150);
        std::vector<std::vector<float>> test_inputs(inputs.begin() + 150, inputs.end());
        std::vector<std::vector<float>> test_targets(targets.begin() + 150, targets.end());

        // Train the model
        int num_samples = train_inputs.size();
        core.train(train_inputs, train_targets, 0.01, num_samples, 32);

        // Make predictions on test set
        auto predictions = core.forward(test_inputs);

        // Flatten predictions and targets for metric calculation
        std::vector<float> flat_predictions;
        std::vector<float> flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(test_targets[i][0]);
        }

        // Calculate performance metrics
        float mse = calculateMSE(flat_predictions, flat_targets);
        float mae = calculateMAE(flat_predictions, flat_targets);
        float r2 = calculateRSquared(flat_predictions, flat_targets);

        // Check that model learned something reasonable
        REQUIRE(mse < 5.0f); // Should be reasonably low
        REQUIRE(mae < 2.0f); // Should be reasonably low
        REQUIRE(r2 > 0.8f);  // Should explain most variance
    }

    SECTION("Linear regression with different learning rates") {
        auto [inputs, targets] = generateLinearData(100);

        std::vector<double> learning_rates = {0.001, 0.01, 0.1};

        for (double lr : learning_rates) {
            CoreAI test_core(1, 2, 10, 1, -10.0f, 10.0f);
            int num_samples = inputs.size();
            test_core.train(inputs, targets, lr, num_samples, 32);

            auto predictions = test_core.forward(inputs);
            std::vector<float> flat_predictions, flat_targets;
            for (size_t i = 0; i < predictions.size(); ++i) {
                flat_predictions.push_back(predictions[i][0]);
                flat_targets.push_back(targets[i][0]);
            }

            float mse = calculateMSE(flat_predictions, flat_targets);
            REQUIRE(mse < 10.0f); // Should converge reasonably
        }
    }
}

TEST_CASE("Regression Tests: Polynomial Regression", "[regression]") {
    CoreAI core(1, 3, 20, 1, -10.0f, 10.0f);

    SECTION("Polynomial regression with quadratic data") {
        auto [inputs, targets] = generatePolynomialData(200);

        // Split into train/test
        std::vector<std::vector<float>> train_inputs(inputs.begin(), inputs.begin() + 150);
        std::vector<std::vector<float>> train_targets(targets.begin(), targets.begin() + 150);
        std::vector<std::vector<float>> test_inputs(inputs.begin() + 150, inputs.end());
        std::vector<std::vector<float>> test_targets(targets.begin() + 150, targets.end());

        // Enable polynomial transforms for better fitting
        core.set3DTransformConfig(true, false, false, false, false, false, false, false);
        core.set3DTransformWeights(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

        // Train the model
        int num_samples = train_inputs.size();
        core.train(train_inputs, train_targets, 0.005, num_samples, 16);

        // Make predictions
        auto predictions = core.forward(test_inputs);

        // Calculate metrics
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(test_targets[i][0]);
        }

        float mse = calculateMSE(flat_predictions, flat_targets);
        float r2 = calculateRSquared(flat_predictions, flat_targets);

        // Polynomial should fit better than linear
        REQUIRE(mse < 8.0f);
        REQUIRE(r2 > 0.7f);
    }
}

TEST_CASE("Regression Tests: Multi-variable Regression", "[regression]") {
    CoreAI core(3, 2, 15, 1, -20.0f, 20.0f);

    SECTION("Multi-variable linear regression") {
        auto [inputs, targets] = generateMultiVariableData(300);

        // Split into train/test
        std::vector<std::vector<float>> train_inputs(inputs.begin(), inputs.begin() + 200);
        std::vector<std::vector<float>> train_targets(targets.begin(), targets.begin() + 200);
        std::vector<std::vector<float>> test_inputs(inputs.begin() + 200, inputs.end());
        std::vector<std::vector<float>> test_targets(targets.begin() + 200, targets.end());

        // Train the model
        int num_samples = train_inputs.size();
        core.train(train_inputs, train_targets, 0.005, num_samples, 32);

        // Make predictions
        auto predictions = core.forward(test_inputs);

        // Calculate metrics
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(test_targets[i][0]);
        }

        float mse = calculateMSE(flat_predictions, flat_targets);
        float mae = calculateMAE(flat_predictions, flat_targets);
        float r2 = calculateRSquared(flat_predictions, flat_targets);

        // Should achieve good fit for linear relationship
        REQUIRE(mse < 15.0f);
        REQUIRE(mae < 3.0f);
        REQUIRE(r2 > 0.85f);
    }

    SECTION("Multi-variable regression with regularization") {
        auto [inputs, targets] = generateMultiVariableData(200);

        CoreAI regularized_core(3, 2, 15, 1, -20.0f, 20.0f);
        regularized_core.setRegularizationParams(0.01f, 0.1f, 0.1f, 1.0f);
        regularized_core.enableRegularization(true, true, true, true);

        int num_samples = inputs.size();
        regularized_core.train(inputs, targets, 0.01, num_samples, 32);

        auto predictions = regularized_core.forward(inputs);
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(targets[i][0]);
        }

        float mse = calculateMSE(flat_predictions, flat_targets);
        REQUIRE(mse < 20.0f); // Should still converge with regularization
    }
}

TEST_CASE("Regression Tests: Performance Metrics Evaluation", "[regression]") {
    CoreAI core(1, 2, 10, 1, -10.0f, 10.0f);

    SECTION("Comprehensive metrics evaluation") {
        auto [inputs, targets] = generateLinearData(100);

        int num_samples = inputs.size();
        core.train(inputs, targets, 0.01, num_samples, 32);

        auto predictions = core.forward(inputs);

        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(targets[i][0]);
        }

        float mse = calculateMSE(flat_predictions, flat_targets);
        float mae = calculateMAE(flat_predictions, flat_targets);
        float r2 = calculateRSquared(flat_predictions, flat_targets);

        // Log metrics for analysis
        INFO("MSE: " << mse);
        INFO("MAE: " << mae);
        INFO("R²: " << r2);

        // Verify metrics are reasonable
        REQUIRE(mse >= 0.0f);
        REQUIRE(mae >= 0.0f);
        REQUIRE(r2 <= 1.0f);
        REQUIRE(r2 >= -1.0f); // R² can be negative for very poor fits
    }

    SECTION("Realistic use case: House price prediction") {
        // Simulate house price data: price = 100000 + 50000*size + 20000*bedrooms + 10000*age + noise
        int num_samples = 150;
        std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(3)); // size, bedrooms, age
        std::vector<std::vector<float>> targets(num_samples, std::vector<float>(1)); // price

        for (int i = 0; i < num_samples; ++i) {
            float size = 1000.0f + (static_cast<float>(rand()) / RAND_MAX) * 3000.0f; // 1000-4000 sq ft
            float bedrooms = 1.0f + (static_cast<float>(rand()) / RAND_MAX) * 4.0f; // 1-5 bedrooms
            float age = (static_cast<float>(rand()) / RAND_MAX) * 50.0f; // 0-50 years
            float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 20000.0f;

            float price = 100000.0f + 50000.0f * (size / 2000.0f - 1.0f) +
                         20000.0f * (bedrooms - 3.0f) - 10000.0f * (age / 25.0f) + noise;

            inputs[i][0] = size;
            inputs[i][1] = bedrooms;
            inputs[i][2] = age;
            targets[i][0] = price;
        }

        // Normalize inputs
        core.normalizeInput(0.0f, 4000.0f); // Approximate ranges

        int train_samples = 100;
        std::vector<std::vector<float>> train_inputs(inputs.begin(), inputs.begin() + train_samples);
        std::vector<std::vector<float>> train_targets(targets.begin(), targets.begin() + train_samples);
        std::vector<std::vector<float>> test_inputs(inputs.begin() + train_samples, inputs.end());
        std::vector<std::vector<float>> test_targets(targets.begin() + train_samples, targets.end());

        core.train(train_inputs, train_targets, 0.001, train_samples, 16);

        auto predictions = core.forward(test_inputs);
        std::vector<float> flat_predictions, flat_targets;
        for (size_t i = 0; i < predictions.size(); ++i) {
            flat_predictions.push_back(predictions[i][0]);
            flat_targets.push_back(test_targets[i][0]);
        }

        float mse = calculateMSE(flat_predictions, flat_targets);
        float r2 = calculateRSquared(flat_predictions, flat_targets);

        INFO("House Price Prediction - MSE: " << mse << ", R²: " << r2);

        // Realistic performance expectations
        REQUIRE(mse < 500000000.0f); // Less than $500k squared error on average
        REQUIRE(r2 > 0.5f); // Should explain more than 50% of variance
    }
}
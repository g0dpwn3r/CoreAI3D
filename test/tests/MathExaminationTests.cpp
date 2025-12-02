// test/tests/MathExaminationTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>

#include "Core.hpp"
#include "../utils/TestHelpers.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// Helper function to calculate numerical derivative
float numericalDerivative(float (*func)(float), float x, float h = 1e-5f) {
    return (func(x + h) - func(x - h)) / (2.0f * h);
}

// Helper function to calculate integral using trapezoidal rule
float trapezoidalIntegral(float (*func)(float), float a, float b, int n = 1000) {
    float h = (b - a) / n;
    float sum = 0.5f * (func(a) + func(b));

    for (int i = 1; i < n; ++i) {
        sum += func(a + i * h);
    }

    return sum * h;
}

// Generate 3D coordinate data for spatial testing
std::vector<std::vector<float>> generate3DCoordinates(int num_samples = 100) {
    std::vector<std::vector<float>> coordinates(num_samples, std::vector<float>(3));

    for (int i = 0; i < num_samples; ++i) {
        // Generate points in a sphere
        float theta = (static_cast<float>(rand()) / RAND_MAX) * 2.0f * M_PI;
        float phi = std::acos(2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f);
        float r = (static_cast<float>(rand()) / RAND_MAX) * 2.0f;

        coordinates[i][0] = r * std::sin(phi) * std::cos(theta); // x
        coordinates[i][1] = r * std::sin(phi) * std::sin(theta); // y
        coordinates[i][2] = r * std::cos(phi); // z
    }

    return coordinates;
}

// Generate data for calculus-based testing
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generateCalculusTestData(int num_samples = 200) {
    std::vector<std::vector<float>> inputs(num_samples, std::vector<float>(1));
    std::vector<std::vector<float>> targets(num_samples, std::vector<float>(1));

    for (int i = 0; i < num_samples; ++i) {
        float x = (static_cast<float>(i) / num_samples) * 4.0f - 2.0f; // x in [-2, 2]
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;

        // Target is integral of sigmoid from 0 to x
        float integral_sigmoid = trapezoidalIntegral([](float t) {
            return 1.0f / (1.0f + std::exp(-t));
        }, 0.0f, x);

        targets[i][0] = integral_sigmoid + noise;
        inputs[i][0] = x;
    }

    return {inputs, targets};
}

TEST_CASE("Math Examination: Calculus-based Activations", "[math][calculus]") {
    CoreAI core(1, 2, 10, 1, -5.0f, 5.0f);

    SECTION("Integral Sigmoid activation") {
        CalculusConfig config{20, 1.0f, 1.0f};

        // Test basic functionality
        float result = core.integral_sigmoid(1.0f, config);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
        REQUIRE(result > 0.0f);

        // Test derivative
        float derivative = core.integral_sigmoid_derivative(1.0f, config);
        REQUIRE(!std::isnan(derivative));
        REQUIRE(std::isfinite(derivative));

        // Derivative should be approximately equal to sigmoid
        float expected_sigmoid = core.sigmoid(1.0f);
        REQUIRE(std::abs(derivative - expected_sigmoid) < 0.1f);
    }

    SECTION("Integral ReLU activation") {
        CalculusConfig config{20, 1.0f, 1.0f};

        // Test positive input
        float result_pos = core.integral_relu(1.0f, config);
        REQUIRE(result_pos > 0.0f);

        // Test negative input
        float result_neg = core.integral_relu(-1.0f, config);
        REQUIRE(result_neg == 0.0f); // Integral of ReLU for negative x is 0

        // Test derivative
        float deriv_pos = core.integral_relu_derivative(1.0f, config);
        float deriv_neg = core.integral_relu_derivative(-1.0f, config);

        REQUIRE(deriv_pos >= 0.9f); // Should be close to 1
        REQUIRE(deriv_neg == 0.0f); // Derivative of ReLU is 0 for negative x
    }

    SECTION("Integral Tanh activation") {
        CalculusConfig config{20, 1.0f, 1.0f};

        float result = core.integral_tanh(1.0f, config);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));

        // Test derivative
        float derivative = core.integral_tanh_derivative(1.0f, config);
        float expected_tanh = std::tanh(1.0f);
        REQUIRE(std::abs(derivative - expected_tanh) < 0.1f);
    }

    SECTION("Calculus-based activation in network") {
        auto [inputs, targets] = generateCalculusTestData(150);

        CoreAI calculus_core(1, 2, 15, 1, -3.0f, 3.0f);
        calculus_core.setLayerActivation(1, ActivationType::INTEGRAL_SIGMOID);

        int num_samples = inputs.size();
        calculus_core.train(inputs, targets, 0.01, num_samples, 32);

        auto predictions = calculus_core.forward(inputs);

        // Calculate MSE
        float mse = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float diff = predictions[i][0] - targets[i][0];
            mse += diff * diff;
        }
        mse /= predictions.size();

        INFO("Calculus-based network MSE: " << mse);
        REQUIRE(mse < 1.0f); // Should learn the integral relationship
    }
}

TEST_CASE("Math Examination: Spatial Transforms", "[math][spatial]") {
    CoreAI core(3, 2, 12, 1, -10.0f, 10.0f);

    SECTION("Spatial ELU activation") {
        float distance = 2.0f;
        float alpha = 1.0f;
        float spatial_weight = 0.5f;

        // Test basic functionality
        float result = core.spatial_elu(1.0f, distance, alpha, spatial_weight);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));

        // Test derivative
        float derivative = core.spatial_elu_derivative(1.0f, distance, alpha, spatial_weight);
        REQUIRE(!std::isnan(derivative));
        REQUIRE(std::isfinite(derivative));
    }

    SECTION("Spatial Leaky ReLU activation") {
        float distance = 1.5f;
        float alpha = 0.1f;
        float spatial_weight = 0.3f;

        float result = core.spatial_leaky_relu(1.0f, distance, alpha, spatial_weight);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));

        float derivative = core.spatial_leaky_relu_derivative(1.0f, distance, alpha, spatial_weight);
        REQUIRE(!std::isnan(derivative));
        REQUIRE(std::isfinite(derivative));
    }

    SECTION("Spatial activation in 3D coordinate processing") {
        auto coordinates = generate3DCoordinates(100);
        std::vector<std::vector<float>> targets(coordinates.size(), std::vector<float>(1));

        // Create targets based on distance from origin
        for (size_t i = 0; i < coordinates.size(); ++i) {
            float distance = std::sqrt(coordinates[i][0]*coordinates[i][0] +
                                     coordinates[i][1]*coordinates[i][1] +
                                     coordinates[i][2]*coordinates[i][2]);
            targets[i][0] = distance > 1.0f ? 1.0f : 0.0f; // Classify based on distance
        }

        CoreAI spatial_core(3, 2, 15, 1, -5.0f, 5.0f);
        spatial_core.setLayerActivation(1, ActivationType::SPATIAL_ELU);

        int num_samples = coordinates.size();
        spatial_core.train(coordinates, targets, 0.01, num_samples, 32);

        auto predictions = spatial_core.forward(coordinates);

        // Calculate accuracy
        int correct = 0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            bool pred_positive = predictions[i][0] > 0.5f;
            bool true_positive = targets[i][0] > 0.5f;
            if (pred_positive == true_positive) correct++;
        }
        float accuracy = static_cast<float>(correct) / predictions.size();

        INFO("Spatial activation accuracy: " << accuracy);
        REQUIRE(accuracy > 0.7f);
    }
}

TEST_CASE("Math Examination: 3D Feature Engineering", "[math][3d]") {
    CoreAI core(3, 3, 20, 1, -20.0f, 20.0f);

    SECTION("Angular feature engineering") {
        core.set3DTransformConfig(false, false, false, false, true, false, false, false);
        core.set3DTransformWeights(0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        auto coordinates = generate3DCoordinates(50);

        // Test that angular transforms don't crash
        auto results = core.forward(coordinates);
        REQUIRE(results.size() == coordinates.size());
        for (const auto& result : results) {
            REQUIRE(result.size() == 1);
            REQUIRE(!std::isnan(result[0]));
            REQUIRE(std::isfinite(result[0]));
        }
    }

    SECTION("Projection feature engineering") {
        core.set3DTransformConfig(false, false, false, false, false, true, false, false);
        core.set3DTransformWeights(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

        auto coordinates = generate3DCoordinates(50);

        auto results = core.forward(coordinates);
        REQUIRE(results.size() == coordinates.size());
        for (const auto& result : results) {
            REQUIRE(!std::isnan(result[0]));
            REQUIRE(std::isfinite(result[0]));
        }
    }

    SECTION("Ratio feature engineering") {
        core.set3DTransformConfig(false, false, false, false, false, false, true, false);
        core.set3DTransformWeights(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

        auto coordinates = generate3DCoordinates(50);

        auto results = core.forward(coordinates);
        REQUIRE(results.size() == coordinates.size());
        for (const auto& result : results) {
            REQUIRE(!std::isnan(result[0]));
            REQUIRE(std::isfinite(result[0]));
        }
    }

    SECTION("Transformation feature engineering") {
        core.set3DTransformConfig(false, false, false, false, false, false, false, true);
        core.set3DTransformWeights(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

        auto coordinates = generate3DCoordinates(50);

        auto results = core.forward(coordinates);
        REQUIRE(results.size() == coordinates.size());
        for (const auto& result : results) {
            REQUIRE(!std::isnan(result[0]));
            REQUIRE(std::isfinite(result[0]));
        }
    }

    SECTION("Combined 3D feature engineering") {
        core.set3DTransformConfig(true, true, true, true, true, true, true, true);
        core.set3DTransformWeights(0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f, 0.2f);

        auto coordinates = generate3DCoordinates(80);
        std::vector<std::vector<float>> targets(coordinates.size(), std::vector<float>(1));

        // Create targets based on complex 3D relationship
        for (size_t i = 0; i < coordinates.size(); ++i) {
            float x = coordinates[i][0], y = coordinates[i][1], z = coordinates[i][2];
            float distance = std::sqrt(x*x + y*y + z*z);
            float angle_xy = std::atan2(y, x);
            float angle_xz = std::atan2(z, x);

            // Complex target based on multiple geometric properties
            targets[i][0] = std::sin(distance) + 0.5f * std::cos(angle_xy) + 0.3f * std::sin(angle_xz);
        }

        int num_samples = coordinates.size();
        core.train(coordinates, targets, 0.005, num_samples, 16);

        auto predictions = core.forward(coordinates);

        // Calculate correlation
        float mean_pred = 0.0f, mean_target = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            mean_pred += predictions[i][0];
            mean_target += targets[i][0];
        }
        mean_pred /= predictions.size();
        mean_target /= targets.size();

        float numerator = 0.0f, denom_pred = 0.0f, denom_target = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float pred_diff = predictions[i][0] - mean_pred;
            float target_diff = targets[i][0] - mean_target;
            numerator += pred_diff * target_diff;
            denom_pred += pred_diff * pred_diff;
            denom_target += target_diff * target_diff;
        }

        float correlation = numerator / std::sqrt(denom_pred * denom_target);

        INFO("3D Feature Engineering Correlation: " << correlation);
        REQUIRE(std::abs(correlation) > 0.5f); // Should capture the geometric relationships
    }
}

TEST_CASE("Math Examination: Realistic Mathematical Applications", "[math][realistic]") {
    SECTION("Physics simulation: Orbital mechanics") {
        // Simulate gravitational potential field learning
        int num_samples = 200;
        std::vector<std::vector<float>> positions(num_samples, std::vector<float>(3));
        std::vector<std::vector<float>> potentials(num_samples, std::vector<float>(1));

        for (int i = 0; i < num_samples; ++i) {
            // Generate positions in 3D space
            float r = 1.0f + (static_cast<float>(rand()) / RAND_MAX) * 9.0f; // Distance 1-10
            float theta = (static_cast<float>(rand()) / RAND_MAX) * 2.0f * M_PI;
            float phi = std::acos(2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f);

            float x = r * std::sin(phi) * std::cos(theta);
            float y = r * std::sin(phi) * std::sin(theta);
            float z = r * std::cos(phi);

            positions[i][0] = x;
            positions[i][1] = y;
            positions[i][2] = z;

            // Gravitational potential: V = -GM/r
            potentials[i][0] = -1.0f / r; // Simplified (GM=1)
        }

        CoreAI physics_core(3, 3, 25, 1, -15.0f, 15.0f);

        // Enable spatial and transformation features
        physics_core.set3DTransformConfig(false, false, false, false, true, true, true, true);
        physics_core.set3DTransformWeights(0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.3f, 0.2f);

        // Use spatial activation
        physics_core.setLayerActivation(2, ActivationType::SPATIAL_ELU);

        int num_train = 150;
        std::vector<std::vector<float>> train_positions(positions.begin(), positions.begin() + num_train);
        std::vector<std::vector<float>> train_potentials(potentials.begin(), potentials.begin() + num_train);
        std::vector<std::vector<float>> test_positions(positions.begin() + num_train, positions.end());
        std::vector<std::vector<float>> test_potentials(potentials.begin() + num_train, potentials.end());

        physics_core.train(train_positions, train_potentials, 0.005, num_train, 16);

        auto predictions = physics_core.forward(test_positions);

        // Calculate MSE
        float mse = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float diff = predictions[i][0] - test_potentials[i][0];
            mse += diff * diff;
        }
        mse /= predictions.size();

        INFO("Orbital Mechanics MSE: " << mse);
        REQUIRE(mse < 0.5f); // Should learn the 1/r relationship reasonably well
    }

    SECTION("Signal processing: Calculus-based filtering") {
        // Simulate learning integral transforms for signal processing
        int num_samples = 150;
        std::vector<std::vector<float>> signals(num_samples, std::vector<float>(1));
        std::vector<std::vector<float>> transforms(num_samples, std::vector<float>(1));

        for (int i = 0; i < num_samples; ++i) {
            float t = (static_cast<float>(i) / num_samples) * 4.0f * M_PI; // Time from 0 to 4π
            float signal = std::sin(t) + 0.5f * std::cos(2.0f * t); // Mixed frequency signal

            signals[i][0] = signal;

            // Target is cumulative integral (running sum approximation)
            float integral = 0.0f;
            for (int j = 0; j <= i; ++j) {
                float tj = (static_cast<float>(j) / num_samples) * 4.0f * M_PI;
                integral += std::sin(tj) + 0.5f * std::cos(2.0f * tj);
            }
            integral *= (4.0f * M_PI / num_samples); // Scale by time step

            transforms[i][0] = integral;
        }

        CoreAI signal_core(1, 2, 20, 1, -50.0f, 50.0f);
        signal_core.setLayerActivation(1, ActivationType::INTEGRAL_SIGMOID);

        int num_train = 100;
        std::vector<std::vector<float>> train_signals(signals.begin(), signals.begin() + num_train);
        std::vector<std::vector<float>> train_transforms(transforms.begin(), transforms.begin() + num_train);
        std::vector<std::vector<float>> test_signals(signals.begin() + num_train, signals.end());
        std::vector<std::vector<float>> test_transforms(transforms.begin() + num_train, transforms.end());

        signal_core.train(train_signals, train_transforms, 0.002, num_train, 16);

        auto predictions = signal_core.forward(test_signals);

        // Calculate relative error
        float total_error = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float error = std::abs(predictions[i][0] - test_transforms[i][0]);
            float scale = std::abs(test_transforms[i][0]) + 1.0f; // Avoid division by zero
            total_error += error / scale;
        }
        float avg_relative_error = total_error / predictions.size();

        INFO("Signal Processing Relative Error: " << avg_relative_error);
        REQUIRE(avg_relative_error < 0.3f); // Should learn integral transform reasonably
    }

    SECTION("Computer graphics: 3D surface reconstruction") {
        // Simulate learning height field from 2D coordinates
        int grid_size = 10;
        int num_samples = grid_size * grid_size;
        std::vector<std::vector<float>> coordinates(num_samples, std::vector<float>(2));
        std::vector<std::vector<float>> heights(num_samples, std::vector<float>(1));

        int idx = 0;
        for (int i = 0; i < grid_size; ++i) {
            for (int j = 0; j < grid_size; ++j) {
                float x = (static_cast<float>(i) / (grid_size - 1)) * 4.0f - 2.0f;
                float y = (static_cast<float>(j) / (grid_size - 1)) * 4.0f - 2.0f;

                coordinates[idx][0] = x;
                coordinates[idx][1] = y;

                // Complex surface: combination of hills and valleys
                float r = std::sqrt(x*x + y*y);
                heights[idx][0] = std::sin(r) * std::exp(-r*r*0.1f) + 0.5f * std::cos(x) * std::sin(y);

                idx++;
            }
        }

        // Add z-coordinate (height) to make it 3D
        for (auto& coord : coordinates) {
            coord.push_back(0.0f); // Add z=0 for input
        }

        CoreAI graphics_core(3, 3, 30, 1, -5.0f, 5.0f);

        // Enable comprehensive 3D transforms
        graphics_core.set3DTransformConfig(true, true, true, true, true, true, true, true);
        graphics_core.set3DTransformWeights(0.1f, 0.1f, 0.1f, 0.1f, 0.15f, 0.15f, 0.15f);

        graphics_core.train(coordinates, heights, 0.001, num_samples, 16);

        auto predictions = graphics_core.forward(coordinates);

        // Calculate surface reconstruction error
        float mse = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float diff = predictions[i][0] - heights[i][0];
            mse += diff * diff;
        }
        mse /= predictions.size();

        INFO("3D Surface Reconstruction MSE: " << mse);
        REQUIRE(mse < 0.2f); // Should reconstruct complex surface reasonably well
    }
}
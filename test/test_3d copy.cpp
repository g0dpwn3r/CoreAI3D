#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "CoreAI3D/include/Core.hpp"

int main() {
    // Set fixed seed for reproducibility
    std::srand(42);

    std::cout << "Testing CoreAI3D with 3D spatial data..." << std::endl;

    try {
        // Create neural network for 3D input (x,y,z coordinates)
        CoreAI network(3, 3, 20, 1, 0.0f, 1.0f, true);

        // Generate 3D coordinate data
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> targets;

        // Create samples: predict distance from origin
        for (float x = -1.0f; x <= 1.0f; x += 0.2f) {
            for (float y = -1.0f; y <= 1.0f; y += 0.2f) {
                for (float z = -1.0f; z <= 1.0f; z += 0.2f) {
                    inputs.push_back({x, y, z});
                    float distance = std::sqrt(x*x + y*y + z*z);
                    targets.push_back({distance});
                }
            }
        }

        std::cout << "Generated " << inputs.size() << " 3D coordinate samples" << std::endl;

        // Test initial predictions
        std::cout << "\n=== Initial Predictions ===" << std::endl;
        auto initial_predictions = network.forward(inputs);
        std::cout << "First 5 predictions:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), inputs.size()); ++i) {
            std::cout << "Input: (" << inputs[i][0] << "," << inputs[i][1] << "," << inputs[i][2]
                      << ") -> Prediction: " << initial_predictions[i][0]
                      << " (Target: " << targets[i][0] << ")" << std::endl;
        }

        // Train the network
        std::cout << "\n=== Training ===" << std::endl;
        int num_samples = inputs.size();
        double learning_rate = 0.1;

        for (int epoch = 0; epoch < 200; ++epoch) {
            network.train(inputs, targets, learning_rate, num_samples);

            if (epoch % 20 == 0) {
                auto predictions = network.forward(inputs);
                float mse = 0.0f;
                for (size_t i = 0; i < predictions.size(); ++i) {
                    float error = targets[i][0] - predictions[i][0];
                    mse += error * error;
                }
                mse /= predictions.size();
                std::cout << "Epoch " << epoch << " - MSE: " << mse << std::endl;
            }
        }

        // Final assessment
        std::cout << "\n=== Final Assessment ===" << std::endl;
        auto final_predictions = network.forward(inputs);

        // Denormalize predictions back to original target range
        auto min_it = std::min_element(targets.begin(), targets.end(),
            [](const std::vector<float>& a, const std::vector<float>& b) { return a[0] < b[0]; });
        auto max_it = std::max_element(targets.begin(), targets.end(),
            [](const std::vector<float>& a, const std::vector<float>& b) { return a[0] < b[0]; });
        float target_min = (*min_it)[0];
        float target_max = (*max_it)[0];

        for (auto& pred : final_predictions) {
            pred[0] = pred[0] * (target_max - target_min) + target_min;
        }

        std::cout << "Final predictions (denormalized):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), inputs.size()); ++i) {
            std::cout << "Input: (" << inputs[i][0] << "," << inputs[i][1] << "," << inputs[i][2]
                      << ") -> Prediction: " << final_predictions[i][0]
                      << " (Target: " << targets[i][0] << ")" << std::endl;
        }

        // Check if predictions vary
        bool predictions_vary = false;
        float first_pred = final_predictions[0][0];
        for (size_t i = 1; i < final_predictions.size(); ++i) {
            if (std::abs(final_predictions[i][0] - first_pred) > 0.01f) {
                predictions_vary = true;
                break;
            }
        }

        if (predictions_vary) {
            std::cout << "✓ SUCCESS: Predictions vary with 3D coordinates (non-constant)" << std::endl;
        } else {
            std::cout << "✗ FAILED: Predictions are constant across all 3D inputs" << std::endl;
        }

        std::cout << "\n=== Test Completed ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
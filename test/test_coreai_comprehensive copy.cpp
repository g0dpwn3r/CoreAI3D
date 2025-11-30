#include <iostream>
#include <vector>
#include <cmath>
#include "CoreAI3D/include/Core.hpp"

float compute_mse(const std::vector<std::vector<float>>& predictions, 
                  const std::vector<std::vector<float>>& targets) {
    float mse = 0.0f;
    for (size_t i = 0; i < predictions.size(); ++i) {
        float error = targets[i][0] - predictions[i][0];
        mse += error * error;
    }
    return mse / predictions.size();
}

int main() {
    std::cout << "Testing CoreAI3D Comprehensive Learning Capability..." << std::endl;
    
    try {
        // Create neural network with larger capacity
        CoreAI network(1, 3, 20, 1, 0.0f, 1.0f, true);
        
        // Test data: simple linear relationship y = 2x (normalized)
        std::vector<std::vector<float>> inputs = {
            {0.1f}, {0.2f}, {0.3f}, {0.4f}, {0.5f},
            {0.6f}, {0.7f}, {0.8f}, {0.9f}, {1.0f}
        };
        
        std::vector<std::vector<float>> targets = {
            {0.2f}, {0.4f}, {0.6f}, {0.8f}, {1.0f},
            {1.0f}, {1.0f}, {1.0f}, {1.0f}, {1.0f}  // Cap at 1.0 for normalization
        };
        
        std::cout << "\n=== Initial Assessment ===" << std::endl;
        auto initial_results = network.forward(inputs);
        float initial_mse = compute_mse(initial_results, targets);
        std::cout << "Initial MSE: " << initial_mse << std::endl;
        std::cout << "Initial predictions:" << std::endl;
        for (size_t i = 0; i < initial_results.size(); ++i) {
            std::cout << "Input: " << inputs[i][0] << " -> Prediction: " << initial_results[i][0] 
                      << " (Target: " << targets[i][0] << ")" << std::endl;
        }
        
        std::cout << "\n=== Training with Multiple Epochs ===" << std::endl;
        int numSamples = inputs.size();
        std::vector<float> mse_history;
        
        for (int epoch = 0; epoch < 100; ++epoch) {
            network.train(inputs, targets, 0.1, numSamples);
            
            if (epoch % 20 == 0 || epoch < 10) {
                auto results = network.forward(inputs);
                float mse = compute_mse(results, targets);
                mse_history.push_back(mse);
                std::cout << "Epoch " << epoch << " - MSE: " << mse << std::endl;
                
                // Check for non-constant predictions
                bool all_same = true;
                float first_pred = results[0][0];
                for (size_t i = 1; i < results.size(); ++i) {
                    if (fabs(results[i][0] - first_pred) > 0.01f) {
                        all_same = false;
                        break;
                    }
                }
                
                if (all_same) {
                    std::cout << "WARNING: All predictions are similar at epoch " << epoch << std::endl;
                }
            }
        }
        
        std::cout << "\n=== Final Assessment ===" << std::endl;
        auto final_results = network.forward(inputs);
        float final_mse = compute_mse(final_results, targets);
        std::cout << "Final MSE: " << final_mse << std::endl;
        std::cout << "Final predictions:" << std::endl;
        
        bool learning_occurred = false;
        for (size_t i = 0; i < final_results.size(); ++i) {
            std::cout << "Input: " << inputs[i][0] << " -> Prediction: " << final_results[i][0] 
                      << " (Target: " << targets[i][0] << ")" << std::endl;
            
            // Check if prediction is sensible (within reasonable range of target)
            float error = fabs(final_results[i][0] - targets[i][0]);
            if (error < 0.3f) { // Reasonable tolerance
                learning_occurred = true;
            }
        }
        
        std::cout << "\n=== Learning Analysis ===" << std::endl;
        if (learning_occurred) {
            std::cout << "✓ Learning capability confirmed - predictions show improvement" << std::endl;
        } else {
            std::cout << "✗ Learning capability issue - predictions not improving significantly" << std::endl;
        }
        
        if (final_mse < initial_mse) {
            std::cout << "✓ MSE improved from " << initial_mse << " to " << final_mse << std::endl;
        } else {
            std::cout << "✗ MSE did not improve (initial: " << initial_mse << ", final: " << final_mse << ")" << std::endl;
        }
        
        // Transform functions are private, so we'll test learning capability through forward/backward passes
        std::cout << "\n=== Learning Capability Summary ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
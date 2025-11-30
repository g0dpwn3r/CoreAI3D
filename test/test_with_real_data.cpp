#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include "CoreAI3D/include/Core.hpp"

std::vector<std::vector<float>> load_csv_data(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;
        
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stof(value));
            } catch (const std::exception& e) {
                std::cerr << "Error parsing value: " << value << std::endl;
            }
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    return data;
}

float calculate_mse(const std::vector<std::vector<float>>& predictions, 
                   const std::vector<std::vector<float>>& targets) {
    if (predictions.size() != targets.size()) {
        throw std::invalid_argument("Predictions and targets must have same size");
    }
    
    float sum_squared_error = 0.0f;
    size_t count = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i].size() != targets[i].size()) {
            throw std::invalid_argument("Prediction and target vectors must have same size");
        }
        
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            float error = targets[i][j] - predictions[i][j];
            sum_squared_error += error * error;
            count++;
        }
    }
    
    return sum_squared_error / count;
}

int main() {
    std::cout << "CoreAI3D Test with Real Data (test_data.csv)" << std::endl;
    std::cout << "============================================" << std::endl;
    
    try {
        // Load test data
        auto data = load_csv_data("test_data.csv");
        std::cout << "Loaded " << data.size() << " samples from test_data.csv" << std::endl;
        
        if (data.empty()) {
            std::cerr << "No data loaded!" << std::endl;
            return 1;
        }
        
        // Separate inputs (x) and targets (y)
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> targets;
        
        for (const auto& row : data) {
            if (row.size() >= 2) {
                inputs.push_back({row[0]});  // x values
                targets.push_back({row[1]}); // y values
            }
        }
        
        std::cout << "Input range: " << inputs[0][0] << " to " << inputs.back()[0] << std::endl;
        std::cout << "Target range: " << targets[0][0] << " to " << targets.back()[0] << std::endl;
        
        // Create and test neural network
        CoreAI network(1, 3, 20, 1, 0.0f, 1.0f, true);
        
        // Test initial predictions
        std::cout << "\n=== Initial Prediction Test ===" << std::endl;
        auto initial_predictions = network.forward(inputs);
        float initial_mse = calculate_mse(initial_predictions, targets);
        std::cout << "Initial MSE: " << initial_mse << std::endl;
        
        // Show first few predictions vs targets
        std::cout << "First 5 predictions vs targets:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), inputs.size()); ++i) {
            std::cout << "Input: " << inputs[i][0] << " -> Prediction: " << initial_predictions[i][0] 
                      << " (Target: " << targets[i][0] << ")" << std::endl;
        }
        
        // Train the network
        std::cout << "\n=== Training Test ===" << std::endl;
        int num_samples = inputs.size();
        double learning_rate = 0.1;
        
        std::cout << "Training for 100 epochs with learning rate: " << learning_rate << std::endl;
        
        std::vector<float> mse_history;
        mse_history.push_back(initial_mse);
        
        for (int epoch = 0; epoch < 100; ++epoch) {
            network.train(inputs, targets, learning_rate, num_samples);
            
            // Get predictions after training
            auto predictions = network.forward(inputs);
            float mse = calculate_mse(predictions, targets);
            mse_history.push_back(mse);
            
            if (epoch % 20 == 0) {
                std::cout << "Epoch " << epoch << " - MSE: " << mse << std::endl;
                
                // Check if predictions are still constant
                bool all_similar = true;
                float first_pred = predictions[0][0];
                for (size_t i = 1; i < predictions.size(); ++i) {
                    if (std::abs(predictions[i][0] - first_pred) > 0.01f) {
                        all_similar = false;
                        break;
                    }
                }
                
                if (all_similar) {
                    std::cout << "WARNING: All predictions are still similar at epoch " << epoch << std::endl;
                }
            }
        }
        
        // Final assessment
        std::cout << "\n=== Final Assessment ===" << std::endl;
        auto final_predictions = network.forward(inputs);
        float final_mse = calculate_mse(final_predictions, targets);
        
        std::cout << "Final MSE: " << final_mse << std::endl;
        std::cout << "MSE improvement: " << (initial_mse - final_mse) << std::endl;
        
        std::cout << "\nFinal predictions vs targets:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), inputs.size()); ++i) {
            std::cout << "Input: " << inputs[i][0] << " -> Prediction: " << final_predictions[i][0] 
                      << " (Target: " << targets[i][0] << ")" << std::endl;
        }
        
        // Learning capability analysis
        std::cout << "\n=== Learning Capability Analysis ===" << std::endl;
        if (final_mse < initial_mse) {
            std::cout << "✓ Learning confirmed - MSE decreased from " << initial_mse << " to " << final_mse << std::endl;
        } else {
            std::cout << "✗ Learning failed - MSE did not improve (initial: " << initial_mse 
                      << ", final: " << final_mse << ")" << std::endl;
        }
        
        // Check if predictions vary with inputs
        bool predictions_vary = false;
        float first_pred = final_predictions[0][0];
        for (size_t i = 1; i < final_predictions.size(); ++i) {
            if (std::abs(final_predictions[i][0] - first_pred) > 0.01f) {
                predictions_vary = true;
                break;
            }
        }
        
        if (predictions_vary) {
            std::cout << "✓ Predictions vary with inputs (non-constant)" << std::endl;
        } else {
            std::cout << "✗ Predictions are constant across all inputs" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
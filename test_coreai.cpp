#include <iostream>
#include <vector>
#include "CoreAI3D/include/Core.hpp"

int main() {
    std::cout << "Testing CoreAI3D neural network implementation..." << std::endl;
    
    try {
        // Create a simple neural network
        CoreAI network(1, 3, 10, 1, 0.0f, 1.0f, true); // inputSize=1, layers=3, neurons=10, outputSize=1
        
        // Test data: simple linear relationship y = 2x
        std::vector<std::vector<float>> inputs = {
            {1.0f}, {2.0f}, {3.0f}, {4.0f}, {5.0f}
        };
        
        std::vector<std::vector<float>> targets = {
            {2.0f}, {4.0f}, {6.0f}, {8.0f}, {10.0f}
        };
        
        std::cout << "\n=== Testing Forward Pass ===" << std::endl;
        auto results = network.forward(inputs);
        std::cout << "Initial predictions (before training):" << std::endl;
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "Input: " << inputs[i][0] << " -> Prediction: " << results[i][0] << std::endl;
        }
        
        std::cout << "\n=== Testing Training ===" << std::endl;
        int numSamples = inputs.size();
        network.train(inputs, targets, 0.01, numSamples);
        
        std::cout << "\n=== Testing Forward Pass After Training ===" << std::endl;
        results = network.forward(inputs);
        std::cout << "Predictions after training:" << std::endl;
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "Input: " << inputs[i][0] << " -> Prediction: " << results[i][0] << std::endl;
        }
        
        std::cout << "\n=== Testing Error Handling ===" << std::endl;
        try {
            std::vector<std::vector<float>> empty_input;
            network.forward(empty_input);
            std::cout << "ERROR: Should have thrown exception for empty input" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "✓ Correctly caught exception: " << e.what() << std::endl;
        }
        
        try {
            std::vector<std::vector<float>> wrong_size = {{1.0f, 2.0f}}; // Wrong input size
            network.forward(wrong_size);
            std::cout << "ERROR: Should have thrown exception for wrong input size" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "✓ Correctly caught exception: " << e.what() << std::endl;
        }
        
        std::cout << "\n=== Test Completed Successfully ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
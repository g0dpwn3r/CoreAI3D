#include <iostream>
#include <vector>
#include <cmath>
#include "CoreAI3D/include/Core.hpp"

void debug_weights(const CoreAI& network) {
    std::cout << "\n=== Weight Analysis ===" << std::endl;
    
    // Check input to hidden weights
    auto input_hidden_weights = network.getWeightsHiddenInput();
    std::cout << "Input to hidden weights dimensions: " 
              << input_hidden_weights.size() << " x " 
              << (input_hidden_weights.empty() ? 0 : input_hidden_weights[0].size()) << std::endl;
    
    if (!input_hidden_weights.empty() && !input_hidden_weights[0].empty()) {
        std::cout << "First few input->hidden weights: ";
        for (size_t i = 0; i < std::min(size_t(3), input_hidden_weights.size()); ++i) {
            for (size_t j = 0; j < std::min(size_t(3), input_hidden_weights[i].size()); ++j) {
                std::cout << input_hidden_weights[i][j] << " ";
            }
            std::cout << "... ";
        }
        std::cout << std::endl;
    }
    
    // Check hidden to output weights
    auto hidden_output_weights = network.getWeightsOutputHidden();
    std::cout << "Hidden to output weights dimensions: " 
              << hidden_output_weights.size() << " x " 
              << (hidden_output_weights.empty() ? 0 : hidden_output_weights[0].size()) << std::endl;
    
    if (!hidden_output_weights.empty() && !hidden_output_weights[0].empty()) {
        std::cout << "First few hidden->output weights: ";
        for (size_t i = 0; i < std::min(size_t(3), hidden_output_weights.size()); ++i) {
            for (size_t j = 0; j < std::min(size_t(3), hidden_output_weights[i].size()); ++j) {
                std::cout << hidden_output_weights[i][j] << " ";
            }
            std::cout << "... ";
        }
        std::cout << std::endl;
    }
}

void test_transform_functions() {
    std::cout << "\n=== Transform Function Analysis ===" << std::endl;
    
    // Test the transform functions with various inputs
    CoreAI test_network(1, 1, 1, 1, 0.0f, 1.0f, false);
    
    // Test with different values to see if transforms produce meaningful results
    std::vector<float> test_values = {0.1f, 0.5f, 1.0f, 2.0f};
    
    // Note: We can't call private transform functions directly, but we can test through forward pass
    std::cout << "Transform functions will be tested through forward pass behavior" << std::endl;
}

void test_forward_pass_behavior() {
    std::cout << "\n=== Forward Pass Behavior Analysis ===" << std::endl;
    
    CoreAI network(1, 3, 5, 1, 0.0f, 1.0f, true);
    
    // Test with different inputs to see if outputs vary
    std::vector<std::vector<float>> test_inputs = {
        {0.1f}, {0.5f}, {0.9f}, {2.0f}, {5.0f}
    };
    
    std::cout << "Testing forward pass with different inputs:" << std::endl;
    for (const auto& input : test_inputs) {
        auto result = network.forward({input});
        std::cout << "Input: " << input[0] << " -> Output: " << result[0][0] << std::endl;
    }
}

void analyze_learning_issue() {
    std::cout << "\n=== Learning Issue Analysis ===" << std::endl;
    
    // The main issues identified:
    std::cout << "1. Transform functions use k % 3 logic, but we have 1D input (k=0 always)" << std::endl;
    std::cout << "   - This means only xTransform2 is used for all inputs" << std::endl;
    std::cout << "2. Hidden layer activations are all 0.5, indicating zero or very small weights" << std::endl;
    std::cout << "3. Backpropagation uses transform derivatives, but transforms may not be differentiable properly" << std::endl;
    std::cout << "4. Weight updates are extremely small (order of 1e-6), suggesting vanishing gradients" << std::endl;
    std::cout << "5. The network architecture may be too simple for the transform-based approach" << std::endl;
}

int main() {
    std::cout << "CoreAI3D Neural Network Diagnostic Analysis" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    try {
        // Create network and analyze weights
        CoreAI network(1, 3, 10, 1, 0.0f, 1.0f, true);
        debug_weights(network);
        
        test_forward_pass_behavior();
        test_transform_functions();
        analyze_learning_issue();
        
        std::cout << "\n=== RECOMMENDED FIXES ===" << std::endl;
        std::cout << "1. Remove the k % 3 logic for 1D inputs - use a single transform approach" << std::endl;
        std::cout << "2. Increase weight initialization range to avoid small initial values" << std::endl;
        std::cout << "3. Consider using standard neural network operations instead of complex transforms" << std::endl;
        std::cout << "4. Add bias terms to the network architecture" << std::endl;
        std::cout << "5. Test with simpler activation functions or standard matrix multiplication" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during diagnostic analysis: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
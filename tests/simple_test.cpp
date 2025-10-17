#include <iostream>
#include <vector>
#include "../CoreAI3D/include/Core.hpp"

int main() {
    // Test CoreAI forward
    CoreAI core(3, 1, 4, 2, 0.0f, 1.0f);
    
    std::vector<std::vector<float>> inputs = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f}
    };
    
    auto results = core.forward(inputs);
    std::cout << "Forward test: results.size() = " << results.size() << std::endl;
    if (results.size() == 2) {
        std::cout << "SUCCESS: Forward returns correct number of samples" << std::endl;
    } else {
        std::cout << "FAIL: Expected 2 samples, got " << results.size() << std::endl;
    }
    
    // Test training
    std::vector<std::vector<float>> targets = {
        {0.5f, 0.7f},
        {0.6f, 0.8f}
    };
    
    int numSamples = 2;
    try {
        core.train(inputs, targets, 0.01, numSamples);
        std::cout << "SUCCESS: Training completed without vector index errors" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAIL: Training threw exception: " << e.what() << std::endl;
    }
    
    return 0;
}

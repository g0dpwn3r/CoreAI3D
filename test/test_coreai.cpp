#include <iostream>
#include <vector>
#include <cmath>
#include "CoreAI3D/include/Core.hpp"

// Minimal test for calculus-based activation functions
float test_simpsons_rule(float a, float b, int n, std::function<float(float)> f) {
    // Simpson's rule for numerical integration
    if (n <= 0) n = 2;
    if (n > 1000) n = 1000;

    // Ensures n is even
    if (n % 2 != 0) n++;

    // Handle edge case where a == b
    if (std::abs(b - a) < std::numeric_limits<float>::epsilon()) {
        return 0.0f;
    }

    float h = (b - a) / n;
    float sum = f(a) + f(b);

    // Sum for odd indices (4x multiplier)
    for (int i = 1; i < n; i += 2) {
        float x = a + i * h;
        x = std::max(-100.0f, std::min(100.0f, x));
        sum += 4.0f * f(x);
    }

    // Sum for even indices (2x multiplier)
    for (int i = 2; i < n; i += 2) {
        float x = a + i * h;
        x = std::max(-100.0f, std::min(100.0f, x));
        sum += 2.0f * f(x);
    }

    float result = (h / 3.0f) * sum;
    return std::max(-1e6f, std::min(1e6f, result));
}

float test_integral_sigmoid(float x, int N = 10, float SM = 1.0f, float a = 1.0f) {
    auto sigmoid_func = [](float t) { return 1.0f / (1.0f + std::exp(-t)); };
    float clamped_x = std::max(-10.0f, std::min(10.0f, x));
    float integral = test_simpsons_rule(0.0f, clamped_x, N, sigmoid_func);
    float result = a * SM * integral;
    return std::max(-1e3f, std::min(1e3f, result));
}

int main() {
    std::cout << "Testing calculus-based activation functions..." << std::endl;

    try {
        
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
        
        std::cout << "\n=== Testing Calculus-Based Activation Functions ===" << std::endl;

        // Test values
        std::vector<float> test_values = {0.0f, 0.5f, 1.0f, -0.5f, -1.0f};

        std::cout << "Testing calculus-based activation functions:" << std::endl;
        for (float x : test_values) {
            float integral_sigmoid = test_integral_sigmoid(x, 10, 1.0f, 1.0f);
            float integral_sigmoid_deriv = 1.0f * 1.0f * (1.0f / (1.0f + std::exp(-x))); // a * SM * sigmoid(x)

            std::cout << "x=" << x << ":" << std::endl;
            std::cout << "  Integral Sigmoid: " << integral_sigmoid << " (deriv: " << integral_sigmoid_deriv << ")" << std::endl;
        }

        std::cout << "\n=== Test Completed Successfully ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
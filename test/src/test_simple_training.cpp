#include "Train.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

int main() {
    // Create a simple synthetic dataset: y = 2*x + 1
    // This should be very easy to learn
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> targets;

    // Generate 100 samples
    for (int i = 0; i < 100; ++i) {
        float x = static_cast<float>(i) / 10.0f; // 0.0, 0.1, 0.2, ..., 9.9
        float y = 2.0f * x + 1.0f; // y = 2*x + 1

        inputs.push_back({x});
        targets.push_back({y});
    }

    std::cout << "Testing simple linear regression: y = 2*x + 1" << std::endl;
    std::cout << "Generated " << inputs.size() << " training samples" << std::endl;

    // Test with offline training
    Training trainer(false, true); // offline mode, verbose

    // Create temporary CSV files for inputs and targets
    std::ofstream inputFile("temp_inputs.csv");
    std::ofstream targetFile("temp_targets.csv");

    // Write headers
    inputFile << "x\n";
    targetFile << "y\n";

    // Write data
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputFile << inputs[i][0] << "\n";
        targetFile << targets[i][0] << "\n";
    }
    inputFile.close();
    targetFile.close();

    // Load data using the API
    if (!trainer.loadCSV("temp_inputs.csv", inputs.size(), 0, true, false, ',', "test_input")) {
        std::cerr << "Failed to load input data" << std::endl;
        return 1;
    }

    int datasetId;
    if (!trainer.loadTargetsCSV("temp_targets.csv", ',', true, false, datasetId)) {
        std::cerr << "Failed to load target data" << std::endl;
        return 1;
    }

    // Set network architecture
    trainer.layers = 1;     // Single hidden layer
    trainer.neurons = 4;    // 4 neurons in hidden layer

    // Preprocess with normalization
    trainer.preprocess(0.0f, 1.0f); // normalize to [0,1]

    // Train for a few epochs
    std::cout << "\nStarting training..." << std::endl;
    trainer.train(0.1, 10); // learning rate 0.1, 10 epochs

    // Test prediction on a few samples
    std::cout << "\nTesting predictions:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::vector<std::vector<float>> test_input = {inputs[i]};
        auto predictions = trainer.getCore()->forward(test_input, false); // false = inference mode

        float actual_x = inputs[i][0];
        float actual_y = targets[i][0];
        float predicted_y = predictions[0][0];

        std::cout << "x=" << actual_x << ", actual y=" << actual_y
                  << ", predicted y=" << predicted_y
                  << ", error=" << std::abs(predicted_y - actual_y) << std::endl;
    }

    // Calculate final MSE
    float mse = trainer.calculateMSE();
    std::cout << "\nFinal MSE: " << mse << std::endl;

    if (mse < 0.01) {
        std::cout << "SUCCESS: Model learned the simple relationship!" << std::endl;
    } else {
        std::cout << "FAILURE: Model failed to learn simple relationship. MSE: " << mse << std::endl;
    }

    return 0;
}
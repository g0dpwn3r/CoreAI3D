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

    std::cout << "Testing gradient accumulation: y = 2*x + 1" << std::endl;
    std::cout << "Generated " << inputs.size() << " training samples" << std::endl;

    // Test with offline training
    Training trainer(false, true); // offline mode, verbose

    // Create temporary CSV files for inputs and targets
    std::ofstream inputFile("temp_inputs_ga.csv");
    std::ofstream targetFile("temp_targets_ga.csv");

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
    if (!trainer.loadCSV("temp_inputs_ga.csv", inputs.size(), 0, true, false, ',', "test_input_ga")) {
        std::cerr << "Failed to load input data" << std::endl;
        return 1;
    }

    int datasetId;
    if (!trainer.loadTargetsCSV("temp_targets_ga.csv", ',', true, false, datasetId)) {
        std::cerr << "Failed to load target data" << std::endl;
        return 1;
    }

    // Set network architecture
    trainer.layers = 1;     // Single hidden layer
    trainer.neurons = 4;    // 4 neurons in hidden layer

    // Preprocess with normalization
    trainer.preprocess(0.0f, 1.0f); // normalize to [0,1]

    // Enable gradient accumulation on the CoreAI instance
    auto core = trainer.getCore();
    core->enableGradientAccumulation(true, 4, 50); // accumulate over 4 batches, keep 50 historical entries

    std::cout << "Gradient accumulation enabled with 4 steps and 50 max history" << std::endl;

    // Train for a few epochs
    std::cout << "\nStarting training with gradient accumulation..." << std::endl;
    trainer.train(0.1, 5, 16); // learning rate 0.1, 5 epochs, batch size 16

    // Check gradient history
    size_t historySize = core->getGradientHistorySize();
    std::cout << "\nGradient history size: " << historySize << std::endl;

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

    // Get accumulated gradients for inspection
    auto acc_out_grads = core->getAccumulatedGradientsOutputHidden();
    auto acc_in_grads = core->getAccumulatedGradientsInputHidden();

    std::cout << "Accumulated output-hidden gradients size: " << acc_out_grads.size()
              << " x " << (acc_out_grads.empty() ? 0 : acc_out_grads[0].size()) << std::endl;
    std::cout << "Accumulated input-hidden gradients size: " << acc_in_grads.size()
              << " x " << (acc_in_grads.empty() ? 0 : acc_in_grads[0].size()) << std::endl;

    if (mse < 0.1) {
        std::cout << "SUCCESS: Model with gradient accumulation learned the relationship! MSE: " << mse << std::endl;
    } else {
        std::cout << "Model performance with gradient accumulation: MSE: " << mse << std::endl;
    }

    // Clean up temporary files
    std::remove("temp_inputs_ga.csv");
    std::remove("temp_targets_ga.csv");

    return 0;
}
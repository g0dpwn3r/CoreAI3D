#include "Train.hpp"
#include <iostream>
#include <filesystem>

int main() {
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    Training trainer(true, true, nullptr); // offline, verbose

    // Load input data with outputSize=0 (all columns as inputs)
    if (!trainer.loadCSV("../../../regression_test_sample.csv", 10, 0, true, false, ',', "test")) {
        std::cerr << "Failed to load input data" << std::endl;
        return 1;
    }

    // Load target data
    int datasetId;
    if (!trainer.loadTargetsCSV("../../../regression_test_target.csv", ',', true, false, datasetId)) {
        std::cerr << "Failed to load target data" << std::endl;
        return 1;
    }

    std::cout << "Input size: " << trainer.inputSize << std::endl;
    std::cout << "Output size: " << trainer.outputSize << std::endl;
    std::cout << "Num samples: " << trainer.numSamples << std::endl;

    auto inputs = trainer.getInputs();
    auto targets = trainer.getTargets();

    std::cout << "Inputs size: " << inputs.size() << std::endl;
    std::cout << "Targets size: " << targets.size() << std::endl;

    if (!inputs.empty()) {
        std::cout << "First input sample size: " << inputs[0].size() << std::endl;
        std::cout << "First input values: ";
        for (size_t i = 0; i < inputs[0].size(); ++i) {
            std::cout << inputs[0][i] << " ";
        }
        std::cout << std::endl;
    }

    if (!targets.empty()) {
        std::cout << "First target sample size: " << targets[0].size() << std::endl;
        std::cout << "First target values: ";
        for (size_t i = 0; i < targets[0].size(); ++i) {
            std::cout << targets[0][i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
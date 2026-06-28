// End-to-end diagnostic using actual Training class
#include "Train.hpp"
#include <iostream>
#include <filesystem>

int main() {
    std::cout << "Current directory: " << std::filesystem::current_path() << std::endl;

    Training trainer(true, true, nullptr); // offline, verbose=true

    std::cout << "\n[STEP 1] Loading gold.csv with outputSize=1 (last column is target)..." << std::endl;
    if (!trainer.loadCSV("gold.csv", 1000, 1, true, false, ',', "test_sliding")) {
        std::cerr << "FAILED: loadCSV returned false" << std::endl;
        return 1;
    }

    std::cout << "\n[RESULT] After loadCSV:" << std::endl;
    std::cout << "  inputSize: " << trainer.inputSize << std::endl;
    std::cout << "  outputSize: " << trainer.outputSize << std::endl;
    std::cout << "  numSamples: " << trainer.numSamples << std::endl;

    auto inputsBefore = trainer.getInputs();
    auto targetsBefore = trainer.getTargets();

    std::cout << "  inputsBefore size: " << inputsBefore.size() << std::endl;
    if (!inputsBefore.empty()) {
        std::cout << "  First input sample size: " << inputsBefore[0].size() << std::endl;
        std::cout << "  First input values: ";
        for (size_t i = 0; i < std::min(size_t(5), inputsBefore[0].size()); ++i) {
            std::cout << inputsBefore[0][i] << " ";
        }
        std::cout << "...";
        // Check for zeros
        int zeroCount = 0;
        for (float v : inputsBefore[0]) if (v == 0.0f) zeroCount++;
        std::cout << " (" << zeroCount << " zeros)" << std::endl;
    }

    std::cout << "  targetsBefore size: " << targetsBefore.size() << std::endl;
    if (!targetsBefore.empty()) {
        std::cout << "  First target: " << targetsBefore[0][0] << std::endl;
    }

    std::cout << "\n[STEP 2] Applying sliding window (windowSize=12)..." << std::endl;
    if (!trainer.applySlidingWindows(12)) {
        std::cerr << "FAILED: applySlidingWindows returned false" << std::endl;
        return 1;
    }

    std::cout << "\n[RESULT] After applySlidingWindows:" << std::endl;
    std::cout << "  inputSize: " << trainer.inputSize << std::endl;
    std::cout << "  outputSize: " << trainer.outputSize << std::endl;
    std::cout << "  numSamples: " << trainer.numSamples << std::endl;

    auto inputsAfter = trainer.getInputs();
    auto targetsAfter = trainer.getTargets();

    std::cout << "  inputsAfter size: " << inputsAfter.size() << std::endl;
    if (!inputsAfter.empty()) {
        std::cout << "  First input sample size: " << inputsAfter[0].size() << std::endl;
        std::cout << "  First input values: ";
        for (size_t i = 0; i < inputsAfter[0].size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << inputsAfter[0][i] << " ";
        }
        std::cout << std::endl;
        // Check for all zeros
        bool allZeros = true;
        int totalZeros = 0;
        for (const auto& row : inputsAfter) {
            for (float v : row) {
                if (v != 0.0f) { allZeros = false; }
                if (v == 0.0f) { totalZeros++; }
            }
        }
        std::cout << "  Total zero values in ALL inputs: " << totalZeros << std::endl;
        std::cout << "  All inputs are zero: " << (allZeros ? "YES (PROBLEM!)" : "NO") << std::endl;
    }

    std::cout << "  targetsAfter size: " << targetsAfter.size() << std::endl;
    if (!targetsAfter.empty()) {
        std::cout << "  First target: " << targetsAfter[0][0] << std::endl;
    }

    std::cout << "\n[STEP 3] Checking normalization bounds..." << std::endl;
    std::cout << "  input_min size: " << trainer.input_min.size() << std::endl;
    std::cout << "  input_max size: " << trainer.input_max.size() << std::endl;
    std::cout << "  target_mins size: " << trainer.target_mins.size() << std::endl;
    std::cout << "  target_maxs size: " << trainer.target_maxs.size() << std::endl;

    if (!trainer.input_min.empty()) {
        std::cout << "  input_min[0]: " << trainer.input_min[0] << std::endl;
        std::cout << "  input_max[0]: " << trainer.input_max[0] << std::endl;
    }

    return 0;
}
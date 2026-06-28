// Diagnostic test for sliding window data pipeline
// This test isolates and traces the data flow to identify why inputs become all zeros

#include "Train.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>

int main() {
    std::cout << "=== SLIDING WINDOW DATA PIPELINE DIAGNOSTIC ===" << std::endl;
    std::cout << "CWD: " << std::filesystem::current_path() << std::endl;

    Training trainer(true, true, nullptr); // offline, verbose

    // Step 1: Load data with outputSize=1 (last column is target)
    std::cout << "\n[STEP 1] Loading data with outputSize=1..." << std::endl;
    if (!trainer.loadCSV("gold.csv", 100, 1, true, false, ',', "gold_dataset")) {
        std::cerr << "FAILED: loadCSV returned false" << std::endl;
        return 1;
    }

    std::cout << "After loadCSV:" << std::endl;
    std::cout << "  inputSize: " << trainer.inputSize << std::endl;
    std::cout << "  outputSize: " << trainer.outputSize << std::endl;
    std::cout << "  numSamples: " << trainer.numSamples << std::endl;

    auto inputsBefore = trainer.getInputs();
    auto targetsBefore = trainer.getTargets();

    if (!inputsBefore.empty()) {
        std::cout << "  First input sample: ";
        for (size_t i = 0; i < std::min(size_t(5), inputsBefore[0].size()); ++i) {
            std::cout << std::fixed << std::setprecision(4) << inputsBefore[0][i] << " ";
        }
        std::cout << "... (size=" << inputsBefore[0].size() << ")" << std::endl;
    } else {
        std::cout << "  inputsBefore is EMPTY!" << std::endl;
    }

    if (!targetsBefore.empty()) {
        std::cout << "  First target sample: ";
        for (size_t i = 0; i < targetsBefore[0].size(); ++i) {
            std::cout << targetsBefore[0][i] << " ";
        }
        std::cout << "... (size=" << targetsBefore[0].size() << ")" << std::endl;
    } else {
        std::cout << "  targetsBefore is EMPTY!" << std::endl;
    }

    // Step 2: Apply sliding window
    std::cout << "\n[STEP 2] Applying sliding window (windowSize=12)..." << std::endl;
    if (!trainer.applySlidingWindows(12)) {
        std::cerr << "FAILED: applySlidingWindows returned false" << std::endl;
        return 1;
    }

    std::cout << "After applySlidingWindows:" << std::endl;
    std::cout << "  inputSize: " << trainer.inputSize << std::endl;
    std::cout << "  outputSize: " << trainer.outputSize << std::endl;
    std::cout << "  numSamples: " << trainer.numSamples << std::endl;

    auto inputsAfter = trainer.getInputs();
    auto targetsAfter = trainer.getTargets();

    if (!inputsAfter.empty()) {
        std::cout << "  First input sample (after window): ";
        for (size_t i = 0; i < inputsAfter[0].size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << inputsAfter[0][i] << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "  inputsAfter is EMPTY!" << std::endl;
    }

    if (!targetsAfter.empty()) {
        std::cout << "  First target sample (after window): " << targetsAfter[0][0] << std::endl;
    } else {
        std::cout << "  targetsAfter is EMPTY!" << std::endl;
    }

    // Step 3: Check normalization bounds
    std::cout << "\n[STEP 3] Checking normalization bounds..." << std::endl;
    std::cout << "  input_min size: " << trainer.input_min.size() << std::endl;
    std::cout << "  input_max size: " << trainer.input_max.size() << std::endl;
    std::cout << "  target_mins size: " << trainer.target_mins.size() << std::endl;
    std::cout << "  target_maxs size: " << trainer.target_maxs.size() << std::endl;

    if (!trainer.input_min.empty()) {
        std::cout << "  input_min[0]: " << trainer.input_min[0] << std::endl;
        std::cout << "  input_max[0]: " << trainer.input_max[0] << std::endl;
    }
    if (!trainer.target_mins.empty()) {
        std::cout << "  target_mins[0]: " << trainer.target_mins[0] << std::endl;
        std::cout << "  target_maxs[0]: " << trainer.target_maxs[0] << std::endl;
    }

    // Step 4: Check for all-zero inputs
    std::cout << "\n[STEP 4] Checking for all-zero input values..." << std::endl;
    if (!inputsAfter.empty()) {
        bool allZeros = true;
        for (const auto& row : inputsAfter) {
            for (float val : row) {
                if (val != 0.0f) {
                    allZeros = false;
                    break;
                }
            }
            if (!allZeros) break;
        }
        std::cout << "  All inputs are zero: " << (allZeros ? "YES (PROBLEM!)" : "NO") << std::endl;
    }

    // Step 5: Check for all-zero targets
    std::cout << "\n[STEP 5] Checking for all-zero target values..." << std::endl;
    if (!targetsAfter.empty()) {
        bool allZeros = true;
        for (const auto& row : targetsAfter) {
            for (float val : row) {
                if (val != 0.0f) {
                    allZeros = false;
                    break;
                }
            }
            if (!allZeros) break;
        }
        std::cout << "  All targets are zero: " << (allZeros ? "YES (PROBLEM!)" : "NO") << std::endl;
    }

    // Step 6: Test CoreAI forward pass with manual input
    std::cout << "\n[STEP 6] Testing CoreAI with manually populated input..." << std::endl;
    std::vector<std::vector<float>> manualInput = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}};
    std::vector<std::vector<float>> manualTargets = {{13.0f}};

    CoreAI core(12, 3, 10, 1, 0.0f, 1.0f, true);
    core.setInput(manualInput);
    core.setTarget(manualTargets);

    auto manualResult = core.forward(manualInput, false);
    std::cout << "  Manual forward result: ";
    for (float val : manualResult[0]) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
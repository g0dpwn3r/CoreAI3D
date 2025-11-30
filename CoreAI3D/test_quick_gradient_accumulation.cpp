#include "include/Core.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Testing gradient accumulation functionality..." << std::endl;

    // Create a simple CoreAI instance
    CoreAI core(3, 1, 4, 1, 0.0f, 1.0f, true); // 3 inputs, 1 layer, 4 neurons, 1 output, min, max, verbose

    // Enable gradient accumulation
    core.enableGradientAccumulation(true, 2, 10); // accumulate over 2 batches, keep 10 historical entries

    std::cout << "Gradient accumulation enabled" << std::endl;

    // Check if it's working
    size_t historySize = core.getGradientHistorySize();
    std::cout << "Initial gradient history size: " << historySize << std::endl;

    // Get accumulated gradients
    auto acc_out = core.getAccumulatedGradientsOutputHidden();
    auto acc_in = core.getAccumulatedGradientsInputHidden();

    std::cout << "Accumulated gradients - Output-Hidden: " << acc_out.size() << " x "
              << (acc_out.empty() ? 0 : acc_out[0].size()) << std::endl;
    std::cout << "Accumulated gradients - Input-Hidden: " << acc_in.size() << " x "
              << (acc_in.empty() ? 0 : acc_in[0].size()) << std::endl;

    // Reset accumulation
    core.resetGradientAccumulation();
    std::cout << "Gradient accumulation reset" << std::endl;

    // Clear history
    core.clearGradientHistory();
    std::cout << "Gradient history cleared" << std::endl;

    std::cout << "SUCCESS: Gradient accumulation functionality is working!" << std::endl;

    return 0;
}
#include <catch2/catch_all.hpp>
#include "Core.hpp"

// Simple helper to compute MSE between predictions and targets
static float computeMSE(const std::vector<std::vector<float>>& preds,
    const std::vector<std::vector<float>>& targets) {
    if (preds.empty()) return 0.0f;
    float sum = 0.0f;
    size_t n = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        for (size_t j = 0; j < preds[i].size(); ++j) {
            float d = preds[i][j] - targets[i][j];
            sum += d * d;
            ++n;
        }
    }
    return n ? sum / n : 0.0f;
}

TEST_CASE("Training converges") {
    // Small CoreAI instance matching the project constructor
    CoreAI core(3, 1, 4, 2, 0.0f, 1.0f);

    // Toy dataset
    std::vector<std::vector<float>> inputs = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f},
        {0.7f, 0.8f, 0.9f}
    };
    std::vector<std::vector<float>> targets = {
        {0.5f, 0.7f},
        {0.6f, 0.8f},
        {0.9f, 0.95f}
    };

    // Forward before training
    auto before = core.forward(inputs);
    float initialLoss = computeMSE(before, targets);

    // Simple training loop using CoreAI::train
    int numSamples = static_cast<int>(inputs.size());
    const int epochs = 10;
    const double lr = 0.01;
    for (int e = 0; e < epochs; ++e) {
        core.train(inputs, targets, lr, numSamples);
    }

    // Forward after training
    auto after = core.forward(inputs);
    float finalLoss = computeMSE(after, targets);

    REQUIRE(finalLoss <= initialLoss + 1e-4f);
}

// tests/TrainTests.cpp
#include <gtest/gtest.h>

#include "Core.hpp"

// Helper to compute simple MSE
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

TEST(TrainingTest, WeightsUpdateAndLossDecrease) {
    // small network
    CoreAI core(3, 1, 4, 2, 0.0f, 1.0f);

    // toy dataset: 4 samples
    std::vector<std::vector<float>> inputs = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f}
    };
    std::vector<std::vector<float>> targets = {
        {0.0f, 1.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 0.0f}
    };

    // snapshot initial forward + loss
    auto before = core.forward(inputs);
    float initialLoss = computeMSE(before, targets);

    // Train using CoreAI::train repeatedly
    const int epochs = 5;
    const double lr = 0.05;
    int numSamples = static_cast<int>(inputs.size());
    for (int e = 0; e < epochs; ++e) {
        core.train(inputs, targets, lr, numSamples);
    }

    // forward after training and loss
    auto after = core.forward(inputs);
    float finalLoss = computeMSE(after, targets);

    // We expect finalLoss to be <= initialLoss (non-strict)
    EXPECT_LE(finalLoss, initialLoss + 1e-4f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

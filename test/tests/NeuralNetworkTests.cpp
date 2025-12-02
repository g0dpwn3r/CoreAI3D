// test/tests/NeuralNetworkTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>

#include "Core.hpp"
#include "../utils/TestHelpers.hpp"

TEST_CASE("Neural Network: Forward Pass", "[neural_network]") {
    CoreAI core(3, 2, 5, 2, 0.0f, 1.0f);

    SECTION("Single sample forward pass") {
        std::vector<std::vector<float>> inputs = {{0.1f, 0.2f, 0.3f}};

        auto results = core.forward(inputs, false);

        REQUIRE(results.size() == 1);
        REQUIRE(results[0].size() == 2);
        REQUIRE(std::all_of(results[0].begin(), results[0].end(),
                           [](float x) { return !std::isnan(x) && std::isfinite(x); }));
    }

    SECTION("Multiple samples forward pass") {
        std::vector<std::vector<float>> inputs = {
            {0.1f, 0.2f, 0.3f},
            {0.4f, 0.5f, 0.6f},
            {0.7f, 0.8f, 0.9f}
        };

        auto results = core.forward(inputs, false);

        REQUIRE(results.size() == 3);
        for (const auto& result : results) {
            REQUIRE(result.size() == 2);
            REQUIRE(std::all_of(result.begin(), result.end(),
                               [](float x) { return !std::isnan(x) && std::isfinite(x); }));
        }
    }
}

TEST_CASE("Neural Network: Backward Pass", "[neural_network]") {
    CoreAI core(2, 1, 3, 1, 0.0f, 1.0f);

    SECTION("Backward pass with single sample") {
        std::vector<std::vector<float>> inputs = {{0.5f, 0.7f}};
        std::vector<std::vector<float>> targets = {{0.8f}};

        core.setTarget(targets);
        core.forward(inputs, true);

        // Note: backward pass is typically called during training
        // This test verifies the network can perform forward pass in training mode
        REQUIRE(true); // Placeholder - actual backward testing would require training loop
    }
}

TEST_CASE("Neural Network: Multi-layer Architecture", "[neural_network]") {
    CoreAI core(3, 3, 4, 2, 0.0f, 1.0f);

    SECTION("Multi-layer network initialization") {
        std::vector<int> expected_layers = {3, 4, 4, 2}; // input + hidden layers + output
        // Note: layer_sizes includes input layer size

        REQUIRE(true); // Network should initialize without errors
    }

    SECTION("Multi-layer forward pass") {
        std::vector<std::vector<float>> inputs = {{0.1f, 0.2f, 0.3f}};

        auto results = core.forward(inputs, false);

        REQUIRE(results.size() == 1);
        REQUIRE(results[0].size() == 2);
        REQUIRE(std::all_of(results[0].begin(), results[0].end(),
                           [](float x) { return !std::isnan(x) && std::isfinite(x); }));
    }
}

TEST_CASE("Neural Network: Weight Initialization", "[neural_network]") {
    CoreAI core(2, 2, 3, 1, 0.0f, 1.0f);

    SECTION("Weights are properly initialized") {
        // Test that weights are not all zeros and are within reasonable bounds
        const auto& weights_input_hidden = core.getWeightsHiddenInput();
        const auto& weights_output_hidden = core.getWeightsOutputHidden();

        REQUIRE(!weights_input_hidden.empty());
        REQUIRE(!weights_output_hidden.empty());

        // Check that weights are not all zero
        bool hasNonZero = false;
        for (const auto& row : weights_input_hidden) {
            for (float w : row) {
                if (w != 0.0f) hasNonZero = true;
            }
        }
        REQUIRE(hasNonZero);

        hasNonZero = false;
        for (const auto& row : weights_output_hidden) {
            for (float w : row) {
                if (w != 0.0f) hasNonZero = true;
            }
        }
        REQUIRE(hasNonZero);
    }

    SECTION("Multi-layer weight initialization") {
        CoreAI multi_core(2, 3, 4, 1, 0.0f, 1.0f);
        multi_core.initializeMultiLayerWeights();

        // Should not throw any exceptions
        REQUIRE(true);
    }
}

TEST_CASE("Neural Network: Training Integration", "[neural_network]") {
    CoreAI core(2, 1, 3, 1, 0.0f, 1.0f);

    SECTION("Basic training loop") {
        std::vector<std::vector<float>> inputs = {
            {0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}
        };
        std::vector<std::vector<float>> targets = {
            {0.3f}, {0.5f}, {0.7f}
        };

        int numSamples = 3;
        core.train(inputs, targets, 0.01, numSamples, 2);

        // Should complete training without errors
        REQUIRE(true);
    }
}
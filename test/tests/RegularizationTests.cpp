// test/tests/RegularizationTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>

#include "Core.hpp"
#include "../utils/TestHelpers.hpp"

TEST_CASE("Regularization: Dropout", "[regularization]") {
    CoreAI core(3, 1, 4, 2, 0.0f, 1.0f);

    SECTION("Enable dropout") {
        core.enableRegularization(true, false, false, false);
        REQUIRE(core.use_dropout == true);
        REQUIRE(core.dropout_rate == 0.2f); // Default rate
    }

    SECTION("Set dropout parameters") {
        core.setRegularizationParams(0.01f, 0.5f, 0.1f, 1.0f);
        REQUIRE(core.dropout_rate == 0.5f);
    }

    SECTION("Dropout masks generation") {
        core.enableRegularization(true, false, false, false);
        core.setRegularizationParams(0.01f, 0.5f, 0.1f, 1.0f);

        std::vector<std::vector<float>> inputs = {{0.1f, 0.2f, 0.3f}};
        core.forward(inputs, true); // Training mode

        // Should have dropout masks after forward pass in training mode
        REQUIRE(!core.dropout_masks.empty());
    }

    SECTION("Dropout disabled in inference") {
        core.enableRegularization(true, false, false, false);

        std::vector<std::vector<float>> inputs = {{0.1f, 0.2f, 0.3f}};
        core.forward(inputs, false); // Inference mode

        // Dropout masks should be empty in inference mode
        REQUIRE(core.dropout_masks.empty());
    }
}

TEST_CASE("Regularization: L2 Regularization", "[regularization]") {
    CoreAI core(2, 1, 3, 1, 0.0f, 1.0f);

    SECTION("Enable L2 regularization") {
        core.enableRegularization(false, false, true, false);
        REQUIRE(core.use_l2_reg == true);
        REQUIRE(core.l2_lambda == 0.01f); // Default lambda
    }

    SECTION("Set L2 parameters") {
        core.setRegularizationParams(0.05f, 0.2f, 0.1f, 1.0f);
        REQUIRE(core.l2_lambda == 0.05f);
    }

    SECTION("L2 regularization in training") {
        core.enableRegularization(false, false, true, false);
        core.setRegularizationParams(0.1f, 0.2f, 0.1f, 1.0f);

        std::vector<std::vector<float>> inputs = {{0.5f, 0.7f}};
        std::vector<std::vector<float>> targets = {{0.8f}};

        // Training should complete with L2 regularization
        int numSamples = 1;
        core.train(inputs, targets, 0.01, numSamples, 1);
        REQUIRE(true);
    }
}

TEST_CASE("Regularization: Batch Normalization", "[regularization]") {
    CoreAI core(2, 1, 3, 1, 0.0f, 1.0f);

    SECTION("Enable batch normalization") {
        core.enableRegularization(false, true, false, false);
        REQUIRE(core.use_batch_norm == true);
        REQUIRE(core.bn_momentum == 0.1f); // Default momentum
    }

    SECTION("Set batch normalization parameters") {
        core.setRegularizationParams(0.01f, 0.2f, 0.2f, 1.0f);
        REQUIRE(core.bn_momentum == 0.2f);
    }

    SECTION("Batch normalization parameters initialization") {
        core.enableRegularization(false, true, false, false);

        // Should have batch norm parameters initialized (non-empty)
        REQUIRE(!core.bn_running_mean.empty());
        REQUIRE(!core.bn_running_var.empty());
        REQUIRE(!core.bn_gamma.empty());
        REQUIRE(!core.bn_beta.empty());
    }

    SECTION("Batch normalization in training") {
        core.enableRegularization(false, true, false, false);

        std::vector<std::vector<float>> inputs = {{0.5f, 0.7f}};
        std::vector<std::vector<float>> targets = {{0.8f}};

        int numSamples = 1;
        core.train(inputs, targets, 0.01, numSamples, 1);
        REQUIRE(true);
    }
}

TEST_CASE("Regularization: Gradient Clipping", "[regularization]") {
    CoreAI core(2, 1, 3, 1, 0.0f, 1.0f);

    SECTION("Enable gradient clipping") {
        core.enableRegularization(false, false, false, true);
        REQUIRE(core.use_grad_clip == true);
        REQUIRE(core.grad_clip_threshold == 1.0f); // Default threshold
    }

    SECTION("Set gradient clipping parameters") {
        core.setRegularizationParams(0.01f, 0.2f, 0.1f, 5.0f);
        REQUIRE(core.grad_clip_threshold == 5.0f);
    }

    SECTION("Gradient clipping in training") {
        core.enableRegularization(false, false, false, true);
        core.setRegularizationParams(0.01f, 0.2f, 0.1f, 0.1f); // Very low threshold

        std::vector<std::vector<float>> inputs = {{0.5f, 0.7f}};
        std::vector<std::vector<float>> targets = {{0.8f}};

        int numSamples = 1;
        core.train(inputs, targets, 0.01, numSamples, 1);
        REQUIRE(true);
    }
}

TEST_CASE("Regularization: Combined Regularization", "[regularization]") {
    CoreAI core(3, 2, 4, 2, 0.0f, 1.0f);

    SECTION("Enable all regularization methods") {
        core.enableRegularization(true, true, true, true);
        REQUIRE(core.use_dropout == true);
        REQUIRE(core.use_batch_norm == true);
        REQUIRE(core.use_l2_reg == true);
        REQUIRE(core.use_grad_clip == true);
    }

    SECTION("Set all regularization parameters") {
        core.setRegularizationParams(0.02f, 0.3f, 0.15f, 2.0f);
        REQUIRE(core.l2_lambda == 0.02f);
        REQUIRE(core.dropout_rate == 0.3f);
        REQUIRE(core.bn_momentum == 0.15f);
        REQUIRE(core.grad_clip_threshold == 2.0f);
    }

    SECTION("Combined regularization in training") {
        core.enableRegularization(true, true, true, true);
        core.setRegularizationParams(0.01f, 0.2f, 0.1f, 1.0f);

        std::vector<std::vector<float>> inputs = {
            {0.1f, 0.2f, 0.3f},
            {0.4f, 0.5f, 0.6f}
        };
        std::vector<std::vector<float>> targets = {
            {0.3f, 0.7f},
            {0.6f, 0.9f}
        };

        int numSamples = 2;
        core.train(inputs, targets, 0.01, numSamples, 2);
        REQUIRE(true);
    }
}

TEST_CASE("Regularization: Gradient Accumulation", "[regularization]") {
    CoreAI core(2, 1, 3, 1, 0.0f, 1.0f);

    SECTION("Enable gradient accumulation") {
        core.enableGradientAccumulation(true, 4, 10);
        REQUIRE(true); // Should not throw
    }

    SECTION("Gradient accumulation parameters") {
        core.enableGradientAccumulation(true, 2, 5);

        REQUIRE(core.enable_gradient_accumulation == true);
        REQUIRE(core.gradient_accumulation_steps == 2);
        REQUIRE(core.max_historical_gradients == 5);
    }

    SECTION("Reset gradient accumulation") {
        core.enableGradientAccumulation(true, 4, 10);
        core.resetGradientAccumulation();

        REQUIRE(core.current_accumulation_step == 0);
        // Check that gradients are zeroed but not empty
        REQUIRE(!core.accumulated_input_hidden_gradients.empty());
        REQUIRE(!core.accumulated_output_hidden_gradients.empty());
    }

    SECTION("Gradient accumulation history") {
        core.enableGradientAccumulation(true, 2, 3);

        // Should have accumulated gradients initialized
        REQUIRE(!core.getAccumulatedGradientsOutputHidden().empty());
        REQUIRE(!core.getAccumulatedGradientsInputHidden().empty());
        REQUIRE(core.getGradientHistorySize() == 0); // History starts empty
    }

    SECTION("Clear gradient history") {
        core.enableGradientAccumulation(true, 2, 5);
        core.clearGradientHistory();

        REQUIRE(core.historical_output_hidden_gradients.empty());
        REQUIRE(core.historical_input_hidden_gradients.empty());
    }
}

TEST_CASE("Regularization: Edge Cases", "[regularization]") {
    CoreAI core(1, 1, 2, 1, 0.0f, 1.0f);

    SECTION("Zero regularization parameters") {
        core.setRegularizationParams(0.0f, 0.0f, 0.0f, 0.0f);
        REQUIRE(core.l2_lambda == 0.0f);
        REQUIRE(core.dropout_rate == 0.0f);
        REQUIRE(core.bn_momentum == 0.0f);
        REQUIRE(core.grad_clip_threshold == 0.0f);
    }

    SECTION("Extreme regularization parameters") {
        core.setRegularizationParams(10.0f, 0.99f, 0.99f, 100.0f);
        REQUIRE(core.l2_lambda == 10.0f);
        REQUIRE(core.dropout_rate == 0.99f);
        REQUIRE(core.bn_momentum == 0.99f);
        REQUIRE(core.grad_clip_threshold == 100.0f);
    }

    SECTION("Disable all regularization") {
        core.enableRegularization(false, false, false, false);
        REQUIRE(core.use_dropout == false);
        REQUIRE(core.use_batch_norm == false);
        REQUIRE(core.use_l2_reg == false);
        REQUIRE(core.use_grad_clip == false);
    }
}
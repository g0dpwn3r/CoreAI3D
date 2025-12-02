// test/tests/OptimizerTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>

#include "Core.hpp"
#include "Optimizer.hpp"
#include "../utils/TestHelpers.hpp"

TEST_CASE("Optimizer: SGD", "[optimizer]") {
    std::vector<std::vector<float>> weights = {{0.1f, 0.2f}, {0.3f, 0.4f}};
    std::vector<std::vector<float>> gradients = {{0.01f, 0.02f}, {0.03f, 0.04f}};

    SECTION("SGD without momentum") {
        SGD sgd(0.1f, 0.0f);

        auto original_weights = weights;
        sgd.update(weights, gradients);

        // Check that weights have been updated
        REQUIRE(weights[0][0] != original_weights[0][0]);
        REQUIRE(weights[0][1] != original_weights[0][1]);
        REQUIRE(weights[1][0] != original_weights[1][0]);
        REQUIRE(weights[1][1] != original_weights[1][1]);

        // Check correct SGD update: w = w - lr * grad
        REQUIRE(weights[0][0] == Catch::Approx(original_weights[0][0] - 0.1f * gradients[0][0]));
        REQUIRE(weights[0][1] == Catch::Approx(original_weights[0][1] - 0.1f * gradients[0][1]));
    }

    SECTION("SGD with momentum") {
        SGD sgd(0.1f, 0.9f);

        auto original_weights = weights;
        sgd.update(weights, gradients);

        // Should not throw and weights should change
        REQUIRE(weights[0][0] != original_weights[0][0]);
        REQUIRE(weights[0][1] != original_weights[0][1]);
    }

    SECTION("SGD reset") {
        SGD sgd(0.1f, 0.5f);
        sgd.reset();
        // Should not throw
        REQUIRE(sgd.getName() == "SGD");
    }
}

TEST_CASE("Optimizer: Adam", "[optimizer]") {
    std::vector<std::vector<float>> weights = {{0.1f, 0.2f}, {0.3f, 0.4f}};
    std::vector<std::vector<float>> gradients = {{0.01f, 0.02f}, {0.03f, 0.04f}};

    SECTION("Adam basic update") {
        Adam adam(0.001f, 0.9f, 0.999f, 1e-8f);

        auto original_weights = weights;
        adam.update(weights, gradients);

        // Check that weights have been updated
        REQUIRE(weights[0][0] != original_weights[0][0]);
        REQUIRE(weights[0][1] != original_weights[0][1]);
        REQUIRE(weights[1][0] != original_weights[1][0]);
        REQUIRE(weights[1][1] != original_weights[1][1]);
    }

    SECTION("Adam with different parameters") {
        Adam adam(0.01f, 0.8f, 0.95f, 1e-7f);

        auto original_weights = weights;
        adam.update(weights, gradients);

        REQUIRE(weights[0][0] != original_weights[0][0]);
        REQUIRE(weights[1][1] != original_weights[1][1]);
    }

    SECTION("Adam reset") {
        Adam adam(0.001f);
        adam.reset();
        REQUIRE(adam.getName() == "Adam");
    }

    SECTION("Adam multiple updates") {
        Adam adam(0.01f);

        auto weights1 = weights;
        adam.update(weights1, gradients);

        auto weights2 = weights1;
        adam.update(weights2, gradients);

        // Second update should be different from first
        REQUIRE(weights2[0][0] != weights1[0][0]);
        REQUIRE(weights2[1][1] != weights1[1][1]);
    }
}

TEST_CASE("Optimizer: RMSprop", "[optimizer]") {
    std::vector<std::vector<float>> weights = {{0.1f, 0.2f}, {0.3f, 0.4f}};
    std::vector<std::vector<float>> gradients = {{0.01f, 0.02f}, {0.03f, 0.04f}};

    SECTION("RMSprop basic update") {
        RMSprop rmsprop(0.01f, 0.9f, 1e-8f);

        auto original_weights = weights;
        rmsprop.update(weights, gradients);

        REQUIRE(weights[0][0] != original_weights[0][0]);
        REQUIRE(weights[0][1] != original_weights[0][1]);
        REQUIRE(weights[1][0] != original_weights[1][0]);
        REQUIRE(weights[1][1] != original_weights[1][1]);
    }

    SECTION("RMSprop with different parameters") {
        RMSprop rmsprop(0.001f, 0.8f, 1e-7f);

        auto original_weights = weights;
        rmsprop.update(weights, gradients);

        REQUIRE(weights[0][0] != original_weights[0][0]);
        REQUIRE(weights[1][1] != original_weights[1][1]);
    }

    SECTION("RMSprop reset") {
        RMSprop rmsprop(0.01f);
        rmsprop.reset();
        REQUIRE(rmsprop.getName() == "RMSprop");
    }

    SECTION("RMSprop multiple updates") {
        RMSprop rmsprop(0.01f);

        auto weights1 = weights;
        rmsprop.update(weights1, gradients);

        auto weights2 = weights1;
        rmsprop.update(weights2, gradients);

        // Updates should be different
        REQUIRE(weights2[0][0] != weights1[0][0]);
        REQUIRE(weights2[1][1] != weights1[1][1]);
    }
}

TEST_CASE("Optimizer: Integration with CoreAI", "[optimizer]") {
    CoreAI core(2, 1, 3, 1, 0.0f, 1.0f);

    SECTION("Set SGD optimizer") {
        core.setOptimizer("SGD", 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.9f);
        REQUIRE(core.optimizer != nullptr);
        REQUIRE(core.optimizer->getName() == "SGD");
    }

    SECTION("Set Adam optimizer") {
        core.setOptimizer("Adam", 0.001f, 0.9f, 0.999f, 1e-8f);
        REQUIRE(core.optimizer != nullptr);
        REQUIRE(core.optimizer->getName() == "Adam");
    }

    SECTION("Set RMSprop optimizer") {
        core.setOptimizer("RMSprop", 0.01f, 0.9f, 0.999f, 1e-8f, 0.0f, 0.9f);
        REQUIRE(core.optimizer != nullptr);
        REQUIRE(core.optimizer->getName() == "RMSprop");
    }

    SECTION("Reset optimizer state") {
        core.setOptimizer("Adam", 0.001f);
        core.resetOptimizerState();
        REQUIRE(core.optimizer != nullptr);
    }
}

TEST_CASE("Optimizer: Edge Cases", "[optimizer]") {
    std::vector<std::vector<float>> weights = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    std::vector<std::vector<float>> zero_gradients = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    SECTION("Zero gradients") {
        SGD sgd(0.1f);
        auto original_weights = weights;
        sgd.update(weights, zero_gradients);

        // Weights should remain unchanged with zero gradients
        REQUIRE(weights[0][0] == original_weights[0][0]);
        REQUIRE(weights[1][1] == original_weights[1][1]);
    }

    SECTION("Large gradients") {
        std::vector<std::vector<float>> large_gradients = {{100.0f, 200.0f}, {300.0f, 400.0f}};

        Adam adam(0.001f); // Small learning rate to prevent explosion
        adam.update(weights, large_gradients);

        // Should handle large gradients without NaN
        REQUIRE(!std::isnan(weights[0][0]));
        REQUIRE(!std::isnan(weights[1][1]));
        REQUIRE(std::isfinite(weights[0][0]));
        REQUIRE(std::isfinite(weights[1][1]));
    }
}
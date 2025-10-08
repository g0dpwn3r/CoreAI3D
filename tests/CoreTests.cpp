// tests/CoreTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>

#include "Core.hpp"

// Basic smoke tests for CoreAI
TEST_CASE("CoreAI: forward and sigmoid basic checks", "[core]") {
    // Create a small CoreAI instance
    CoreAI core(3, 1, 4, 2, 0.0f, 1.0f);

    SECTION("Forward returns correct shape") {
        std::vector<std::vector<float>> inputs = {
            {0.1f, 0.2f, 0.3f},
            {0.4f, 0.5f, 0.6f}
        };

        auto results = core.forward(inputs);
        // results should have same number of rows as inputs
        REQUIRE(results.size() == inputs.size());
        // each output row should have outputSize entries (2 in our constructor)
        for (auto& r : results) {
            REQUIRE(r.size() == static_cast<size_t>(2));
        }
    }

    SECTION("Sigmoid sanity") {
        REQUIRE(core.sigmoid(0.0f) == Approx(0.5f));
        REQUIRE(core.sigmoid(100.0f) > 0.999f);
        REQUIRE(core.sigmoid(-100.0f) < 0.001f);
    }
}

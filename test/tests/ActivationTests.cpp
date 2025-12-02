// test/tests/ActivationTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>

#include "Core.hpp"
#include "../utils/TestHelpers.hpp"

TEST_CASE("Activation Functions: Basic Functions", "[activation]") {
    CoreAI core(1, 1, 2, 1, 0.0f, 1.0f);

    SECTION("Sigmoid function") {
        REQUIRE(core.sigmoid(0.0f) == Catch::Approx(0.5f));
        REQUIRE(core.sigmoid(1.0f) == Catch::Approx(0.7310585786f).epsilon(1e-6));
        REQUIRE(core.sigmoid(-1.0f) == Catch::Approx(0.2689414214f).epsilon(1e-6));
        REQUIRE(core.sigmoid(100.0f) > 0.999f);
        REQUIRE(core.sigmoid(-100.0f) < 0.001f);
    }

    SECTION("ELU function") {
        REQUIRE(core.elu(0.0f, 1.0f) == 0.0f);
        REQUIRE(core.elu(1.0f, 1.0f) == 1.0f);
        REQUIRE(core.elu(-1.0f, 1.0f) == Catch::Approx(-0.6321205588f).epsilon(1e-6));
        REQUIRE(core.elu(-2.0f, 2.0f) == Catch::Approx(-1.7293294335f).epsilon(1e-6));
    }

    SECTION("Leaky ReLU function") {
        REQUIRE(core.leaky_relu(0.0f, 0.1f) == 0.0f);
        REQUIRE(core.leaky_relu(1.0f, 0.1f) == 1.0f);
        REQUIRE(core.leaky_relu(-1.0f, 0.1f) == -0.1f);
        REQUIRE(core.leaky_relu(-2.0f, 0.2f) == -0.4f);
    }

    SECTION("Parametric ReLU function") {
        REQUIRE(core.parametric_relu(0.0f, 0.1f) == 0.0f);
        REQUIRE(core.parametric_relu(1.0f, 0.1f) == 1.0f);
        REQUIRE(core.parametric_relu(-1.0f, 0.1f) == -0.1f);
        REQUIRE(core.parametric_relu(-2.0f, 0.2f) == -0.4f);
    }

    SECTION("Swish function") {
        REQUIRE(core.swish(0.0f) == 0.0f);
        REQUIRE(core.swish(1.0f) == Catch::Approx(0.7310585786f).epsilon(1e-6));
        REQUIRE(core.swish(-1.0f) == Catch::Approx(-0.2689414214f).epsilon(1e-6));
    }

    SECTION("GELU function") {
        REQUIRE(core.gelu(0.0f) == 0.0f);
        REQUIRE(core.gelu(1.0f) == Catch::Approx(0.8413f).epsilon(1e-4));
        REQUIRE(core.gelu(-1.0f) == Catch::Approx(-0.1587f).epsilon(1e-4));
    }
}

TEST_CASE("Activation Functions: Derivatives", "[activation]") {
    CoreAI core(1, 1, 2, 1, 0.0f, 1.0f);

    SECTION("ELU derivative") {
        REQUIRE(core.elu_derivative(1.0f, 1.0f) == 1.0f);
        REQUIRE(core.elu_derivative(-1.0f, 1.0f) == Catch::Approx(0.3678794412f).epsilon(1e-6));
    }

    SECTION("Leaky ReLU derivative") {
        REQUIRE(core.leaky_relu_derivative(1.0f, 0.1f) == 1.0f);
        REQUIRE(core.leaky_relu_derivative(-1.0f, 0.1f) == 0.1f);
    }

    SECTION("Parametric ReLU derivative") {
        REQUIRE(core.parametric_relu_derivative(1.0f, 0.1f) == 1.0f);
        REQUIRE(core.parametric_relu_derivative(-1.0f, 0.1f) == 0.1f);
    }

    SECTION("Swish derivative") {
        float swish_val = core.swish(1.0f);
        float sigmoid_val = core.sigmoid(1.0f);
        float expected = swish_val + sigmoid_val * (1.0f - swish_val);
        REQUIRE(core.swish_derivative(1.0f) == Catch::Approx(expected).epsilon(1e-6));
    }

    SECTION("GELU derivative") {
        // Test that GELU derivative is positive and reasonable
        float deriv = core.gelu_derivative(1.0f);
        REQUIRE(deriv > 0.0f);
        REQUIRE(deriv < 2.0f); // Should be less than 2
        REQUIRE(!std::isnan(deriv));
        REQUIRE(std::isfinite(deriv));
    }
}

TEST_CASE("Activation Functions: Calculus-based", "[activation]") {
    CoreAI core(1, 1, 2, 1, 0.0f, 1.0f);

    CalculusConfig config{10, 1.0f, 1.0f};

    SECTION("Integral Sigmoid") {
        float result = core.integral_sigmoid(1.0f, config);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
        REQUIRE(result > 0.0f);
    }

    SECTION("Integral Sigmoid derivative") {
        float result = core.integral_sigmoid_derivative(1.0f, config);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
    }

    SECTION("Integral ReLU") {
        float result = core.integral_relu(1.0f, config);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
        REQUIRE(result >= 0.0f);
    }

    SECTION("Integral ReLU derivative") {
        float result = core.integral_relu_derivative(1.0f, config);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
    }

    SECTION("Integral Tanh") {
        float result = core.integral_tanh(1.0f, config);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
    }

    SECTION("Integral Tanh derivative") {
        float result = core.integral_tanh_derivative(1.0f, config);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
    }
}

TEST_CASE("Activation Functions: Spatial Variants", "[activation]") {
    CoreAI core(1, 1, 2, 1, 0.0f, 1.0f);

    SECTION("Spatial ELU") {
        float distance = 1.0f;
        float result = core.spatial_elu(1.0f, distance, 1.0f, 1.0f);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
    }

    SECTION("Spatial ELU derivative") {
        float distance = 1.0f;
        float result = core.spatial_elu_derivative(1.0f, distance, 1.0f, 1.0f);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
    }

    SECTION("Spatial Leaky ReLU") {
        float distance = 1.0f;
        float result = core.spatial_leaky_relu(1.0f, distance, 0.1f, 1.0f);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
    }

    SECTION("Spatial Leaky ReLU derivative") {
        float distance = 1.0f;
        float result = core.spatial_leaky_relu_derivative(1.0f, distance, 0.1f, 1.0f);
        REQUIRE(!std::isnan(result));
        REQUIRE(std::isfinite(result));
    }
}

TEST_CASE("Activation Functions: Configuration and Activation", "[activation]") {
    CoreAI core(2, 2, 3, 1, 0.0f, 1.0f);

    SECTION("Set layer activation - Sigmoid") {
        core.setLayerActivation(0, ActivationType::SIGMOID);
        // Should not throw
        REQUIRE(true);
    }

    SECTION("Set layer activation - ELU") {
        core.setLayerActivation(0, ActivationType::ELU, 1.0f);
        REQUIRE(true);
    }

    SECTION("Set all layer activations") {
        core.setAllLayerActivations(ActivationType::GELU);
        REQUIRE(true);
    }

    SECTION("Activate function with different types") {
        float distance = 0.0f;

        ActivationConfig sigmoid_config{ActivationType::SIGMOID};
        float sigmoid_result = core.activate(1.0f, sigmoid_config, distance);
        REQUIRE(sigmoid_result == Catch::Approx(core.sigmoid(1.0f)));

        ActivationConfig elu_config{ActivationType::ELU, 1.0f};
        float elu_result = core.activate(1.0f, elu_config, distance);
        REQUIRE(elu_result == Catch::Approx(core.elu(1.0f, 1.0f)));
    }

    SECTION("Activate derivative function") {
        float distance = 0.0f;

        ActivationConfig config{ActivationType::SIGMOID};
        float derivative = core.activate_derivative(1.0f, config, distance);
        REQUIRE(!std::isnan(derivative));
        REQUIRE(std::isfinite(derivative));
    }
}
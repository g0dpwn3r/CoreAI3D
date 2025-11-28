#include "Optimizer.hpp"
#include <cmath>
#include <algorithm>

// SGD Implementation
SGD::SGD(float learningRate, float momentum)
    : Optimizer(learningRate), momentum(momentum) {}

void SGD::update(std::vector<std::vector<float>>& weights,
                 const std::vector<std::vector<float>>& gradients) {
    // Initialize velocity if not done
    if (velocity.empty()) {
        velocity.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            velocity[i].resize(weights[i].size(), 0.0f);
        }
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            // Update velocity: v = momentum * v - learning_rate * gradient
            velocity[i][j] = momentum * velocity[i][j] - learningRate * gradients[i][j];
            // Update weights: w = w + v
            weights[i][j] += velocity[i][j];
        }
    }
}

void SGD::reset() {
    velocity.clear();
}

// Adam Implementation
Adam::Adam(float learningRate, float beta1, float beta2, float epsilon)
    : Optimizer(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

void Adam::update(std::vector<std::vector<float>>& weights,
                 const std::vector<std::vector<float>>& gradients) {
    t++; // Increment timestep

    // Initialize moment estimates if not done
    if (m.empty()) {
        m.resize(weights.size());
        v.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            m[i].resize(weights[i].size(), 0.0f);
            v[i].resize(weights[i].size(), 0.0f);
        }
    }

    float beta1_t = std::pow(beta1, t);
    float beta2_t = std::pow(beta2, t);

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            // Update biased first moment estimate
            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * gradients[i][j];
            // Update biased second raw moment estimate
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * gradients[i][j] * gradients[i][j];

            // Compute bias-corrected first moment estimate
            float m_hat = m[i][j] / (1.0f - beta1_t);
            // Compute bias-corrected second raw moment estimate
            float v_hat = v[i][j] / (1.0f - beta2_t);

            // Update weights
            weights[i][j] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
}

void Adam::reset() {
    m.clear();
    v.clear();
    t = 0;
}

// RMSprop Implementation
RMSprop::RMSprop(float learningRate, float rho, float epsilon)
    : Optimizer(learningRate), rho(rho), epsilon(epsilon) {}

void RMSprop::update(std::vector<std::vector<float>>& weights,
                    const std::vector<std::vector<float>>& gradients) {
    // Initialize cache if not done
    if (cache.empty()) {
        cache.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            cache[i].resize(weights[i].size(), 0.0f);
        }
    }

    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < weights[i].size(); ++j) {
            // Update cache: cache = rho * cache + (1 - rho) * gradient^2
            cache[i][j] = rho * cache[i][j] + (1.0f - rho) * gradients[i][j] * gradients[i][j];
            // Update weights: w = w - learning_rate * gradient / sqrt(cache + epsilon)
            weights[i][j] -= learningRate * gradients[i][j] / (std::sqrt(cache[i][j]) + epsilon);
        }
    }
}

void RMSprop::reset() {
    cache.clear();
}
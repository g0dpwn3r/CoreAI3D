#include "LearningRateScheduler.hpp"
#include <cmath>

// ExponentialDecay Implementation
ExponentialDecay::ExponentialDecay(float initialLearningRate, float decayRate, int decaySteps)
    : LearningRateScheduler(initialLearningRate), decayRate(decayRate), decaySteps(decaySteps) {}

float ExponentialDecay::getLearningRate(int epoch) {
    // lr = initial_lr * decay_rate^(epoch / decay_steps)
    currentLearningRate = initialLearningRate * std::pow(decayRate, static_cast<float>(epoch) / decaySteps);
    return currentLearningRate;
}

// StepDecay Implementation
StepDecay::StepDecay(float initialLearningRate, float decayFactor, int decaySteps)
    : LearningRateScheduler(initialLearningRate), decayFactor(decayFactor), decaySteps(decaySteps) {}

float StepDecay::getLearningRate(int epoch) {
    // lr = initial_lr * decay_factor^(floor(epoch / decay_steps))
    int decay_epochs = epoch / decaySteps;
    currentLearningRate = initialLearningRate * std::pow(decayFactor, decay_epochs);
    return currentLearningRate;
}

// CosineAnnealing Implementation
CosineAnnealing::CosineAnnealing(float initialLearningRate, int totalEpochs, float minLearningRate)
    : LearningRateScheduler(initialLearningRate), totalEpochs(totalEpochs), minLearningRate(minLearningRate) {}

float CosineAnnealing::getLearningRate(int epoch) {
    // lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / total_epochs))
    float cos_term = std::cos(M_PI * epoch / static_cast<float>(totalEpochs));
    currentLearningRate = minLearningRate + 0.5f * (initialLearningRate - minLearningRate) * (1.0f + cos_term);
    return currentLearningRate;
}
#ifndef LEARNING_RATE_SCHEDULER_HPP
#define LEARNING_RATE_SCHEDULER_HPP

#include <memory>
#include <string>

class LearningRateScheduler {
public:
    LearningRateScheduler(float initialLearningRate) : initialLearningRate(initialLearningRate), currentLearningRate(initialLearningRate) {}
    virtual ~LearningRateScheduler() = default;

    virtual float getLearningRate(int epoch) = 0;
    virtual std::string getName() const = 0;
    virtual void reset() { currentLearningRate = initialLearningRate; }

protected:
    float initialLearningRate;
    float currentLearningRate;
};

class ExponentialDecay : public LearningRateScheduler {
public:
    ExponentialDecay(float initialLearningRate, float decayRate, int decaySteps);
    ~ExponentialDecay() override = default;

    float getLearningRate(int epoch) override;
    std::string getName() const override { return "ExponentialDecay"; }

private:
    float decayRate;
    int decaySteps;
};

class StepDecay : public LearningRateScheduler {
public:
    StepDecay(float initialLearningRate, float decayFactor, int decaySteps);
    ~StepDecay() override = default;

    float getLearningRate(int epoch) override;
    std::string getName() const override { return "StepDecay"; }

private:
    float decayFactor;
    int decaySteps;
};

class CosineAnnealing : public LearningRateScheduler {
public:
    CosineAnnealing(float initialLearningRate, int totalEpochs, float minLearningRate = 0.0f);
    ~CosineAnnealing() override = default;

    float getLearningRate(int epoch) override;
    std::string getName() const override { return "CosineAnnealing"; }

private:
    int totalEpochs;
    float minLearningRate;
};

#endif // LEARNING_RATE_SCHEDULER_HPP
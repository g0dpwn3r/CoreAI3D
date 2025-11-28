#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include <string>
#include <memory>

class Optimizer {
public:
    Optimizer(float learningRate) : learningRate(learningRate) {}
    virtual ~Optimizer() = default;

    virtual void update(std::vector<std::vector<float>>& weights,
                       const std::vector<std::vector<float>>& gradients) = 0;

    virtual std::string getName() const = 0;
    virtual void reset() = 0; // Reset optimizer state (e.g., for new training run)

protected:
    float learningRate;
};

class SGD : public Optimizer {
public:
    SGD(float learningRate, float momentum = 0.0f);
    ~SGD() override = default;

    void update(std::vector<std::vector<float>>& weights,
               const std::vector<std::vector<float>>& gradients) override;

    std::string getName() const override { return "SGD"; }
    void reset() override;

private:
    float momentum;
    std::vector<std::vector<float>> velocity;
};

class Adam : public Optimizer {
public:
    Adam(float learningRate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    ~Adam() override = default;

    void update(std::vector<std::vector<float>>& weights,
               const std::vector<std::vector<float>>& gradients) override;

    std::string getName() const override { return "Adam"; }
    void reset() override;

private:
    float beta1, beta2, epsilon;
    std::vector<std::vector<float>> m, v; // First and second moment estimates
    int t; // Timestep
};

class RMSprop : public Optimizer {
public:
    RMSprop(float learningRate, float rho = 0.9f, float epsilon = 1e-8f);
    ~RMSprop() override = default;

    void update(std::vector<std::vector<float>>& weights,
               const std::vector<std::vector<float>>& gradients) override;

    std::string getName() const override { return "RMSprop"; }
    void reset() override;

private:
    float rho, epsilon;
    std::vector<std::vector<float>> cache;
};

#endif // OPTIMIZER_HPP
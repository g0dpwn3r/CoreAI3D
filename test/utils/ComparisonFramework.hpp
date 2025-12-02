#ifndef COMPARISON_FRAMEWORK_HPP
#define COMPARISON_FRAMEWORK_HPP

#include "BenchmarkUtils.hpp"
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <memory>

struct SystemResult {
    std::string systemName;
    BenchmarkResult benchmarkResult;
    std::vector<MetricResult> metrics;
    std::string notes;
};

struct ComparisonReport {
    std::string testName;
    std::vector<SystemResult> results;
    std::map<std::string, std::string> summary;
};

class AISystem {
public:
    virtual ~AISystem() = default;
    virtual std::string getName() const = 0;
    virtual void train(const std::vector<std::vector<float>>& inputs,
                      const std::vector<std::vector<float>>& targets) = 0;
    virtual std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& inputs) = 0;
    virtual void reset() = 0; // Reset model state for fair comparison
};

class ComparisonFramework {
public:
    // Register AI systems for comparison
    void registerSystem(std::unique_ptr<AISystem> system);

    // Run comparison on regression task
    ComparisonReport compareRegression(
        const std::string& testName,
        const std::vector<std::vector<float>>& trainInputs,
        const std::vector<std::vector<float>>& trainTargets,
        const std::vector<std::vector<float>>& testInputs,
        const std::vector<std::vector<float>>& testTargets,
        size_t benchmarkIterations = 10
    );

    // Run comparison on classification task
    ComparisonReport compareClassification(
        const std::string& testName,
        const std::vector<std::vector<float>>& trainInputs,
        const std::vector<int>& trainTargets,
        const std::vector<std::vector<float>>& testInputs,
        const std::vector<int>& testTargets,
        size_t benchmarkIterations = 10
    );

    // Generate performance summary
    std::string generateSummary(const ComparisonReport& report);

    // Export results to file
    bool exportResults(const ComparisonReport& report, const std::string& filename);

private:
    std::vector<std::unique_ptr<AISystem>> systems_;

    BenchmarkResult benchmarkTraining(AISystem& system,
                                    const std::vector<std::vector<float>>& inputs,
                                    const std::vector<std::vector<float>>& targets,
                                    size_t iterations);

    BenchmarkResult benchmarkPrediction(AISystem& system,
                                      const std::vector<std::vector<float>>& inputs,
                                      size_t iterations);

    std::vector<MetricResult> calculateMetricsRegression(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets
    );

    std::vector<MetricResult> calculateMetricsClassification(
        const std::vector<int>& predictions,
        const std::vector<int>& targets
    );

    std::vector<int> predictionsToLabels(const std::vector<std::vector<float>>& predictions);
};

#endif // COMPARISON_FRAMEWORK_HPP
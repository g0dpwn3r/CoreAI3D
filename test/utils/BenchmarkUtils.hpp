#ifndef BENCHMARK_UTILS_HPP
#define BENCHMARK_UTILS_HPP

#include <vector>
#include <string>
#include <chrono>
#include <functional>
#include <memory>

struct BenchmarkResult {
    std::string name;
    double durationMs;
    size_t iterations;
    double avgTimePerIterationMs;
    double minTimeMs;
    double maxTimeMs;
    double stdDevMs;
};

struct MetricResult {
    double value;
    std::string unit;
    std::string description;
};

class BenchmarkUtils {
public:
    // Timing utilities
    static BenchmarkResult benchmarkFunction(
        const std::string& name,
        std::function<void()> func,
        size_t iterations = 100,
        size_t warmupIterations = 10
    );

    template<typename Func, typename... Args>
    static BenchmarkResult benchmarkFunction(
        const std::string& name,
        Func&& func,
        Args&&... args,
        size_t iterations = 100,
        size_t warmupIterations = 10
    );

    // Regression metrics
    static MetricResult calculateMSE(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets
    );

    static MetricResult calculateMAE(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets
    );

    static MetricResult calculateRMSE(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets
    );

    static MetricResult calculateR2Score(
        const std::vector<std::vector<float>>& predictions,
        const std::vector<std::vector<float>>& targets
    );

    // Classification metrics
    static MetricResult calculateAccuracy(
        const std::vector<int>& predictions,
        const std::vector<int>& targets
    );

    static MetricResult calculatePrecision(
        const std::vector<int>& predictions,
        const std::vector<int>& targets,
        int positiveClass = 1
    );

    static MetricResult calculateRecall(
        const std::vector<int>& predictions,
        const std::vector<int>& targets,
        int positiveClass = 1
    );

    static MetricResult calculateF1Score(
        const std::vector<int>& predictions,
        const std::vector<int>& targets,
        int positiveClass = 1
    );

    // Memory usage estimation
    static size_t estimateMemoryUsage(const std::vector<std::vector<float>>& data);

    // Statistical utilities
    static double calculateMean(const std::vector<double>& values);
    static double calculateStdDev(const std::vector<double>& values);
    static double calculateMedian(std::vector<double> values);

private:
    static std::vector<double> runBenchmarkIterations(
        std::function<void()> func,
        size_t iterations
    );
};

#include "BenchmarkUtils.tpp"

#endif // BENCHMARK_UTILS_HPP
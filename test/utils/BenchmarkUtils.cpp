#include "BenchmarkUtils.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

BenchmarkResult BenchmarkUtils::benchmarkFunction(
    const std::string& name,
    std::function<void()> func,
    size_t iterations,
    size_t warmupIterations
) {
    // Warmup
    for (size_t i = 0; i < warmupIterations; ++i) {
        func();
    }

    // Benchmark
    auto times = runBenchmarkIterations(func, iterations);

    // Calculate statistics
    double totalTime = 0.0;
    double minTime = *std::min_element(times.begin(), times.end());
    double maxTime = *std::max_element(times.begin(), times.end());

    for (double t : times) {
        totalTime += t;
    }

    double avgTime = totalTime / iterations;
    double variance = 0.0;
    for (double t : times) {
        double diff = t - avgTime;
        variance += diff * diff;
    }
    double stdDev = std::sqrt(variance / iterations);

    return BenchmarkResult{
        name,
        totalTime,
        iterations,
        avgTime,
        minTime,
        maxTime,
        stdDev
    };
}

std::vector<double> BenchmarkUtils::runBenchmarkIterations(
    std::function<void()> func,
    size_t iterations
) {
    std::vector<double> times;
    times.reserve(iterations);

    for (size_t i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        times.push_back(duration.count());
    }

    return times;
}

MetricResult BenchmarkUtils::calculateMSE(
    const std::vector<std::vector<float>>& predictions,
    const std::vector<std::vector<float>>& targets
) {
    if (predictions.empty() || predictions.size() != targets.size()) {
        return {0.0, "", "MSE - Invalid input dimensions"};
    }

    double sum = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i].size() != targets[i].size()) {
            return {0.0, "", "MSE - Mismatched dimensions"};
        }

        for (size_t j = 0; j < predictions[i].size(); ++j) {
            double diff = predictions[i][j] - targets[i][j];
            sum += diff * diff;
            ++count;
        }
    }

    double mse = count > 0 ? sum / count : 0.0;
    return {mse, "", "Mean Squared Error"};
}

MetricResult BenchmarkUtils::calculateMAE(
    const std::vector<std::vector<float>>& predictions,
    const std::vector<std::vector<float>>& targets
) {
    if (predictions.empty() || predictions.size() != targets.size()) {
        return {0.0, "", "MAE - Invalid input dimensions"};
    }

    double sum = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i].size() != targets[i].size()) {
            return {0.0, "", "MAE - Mismatched dimensions"};
        }

        for (size_t j = 0; j < predictions[i].size(); ++j) {
            sum += std::abs(predictions[i][j] - targets[i][j]);
            ++count;
        }
    }

    double mae = count > 0 ? sum / count : 0.0;
    return {mae, "", "Mean Absolute Error"};
}

MetricResult BenchmarkUtils::calculateRMSE(
    const std::vector<std::vector<float>>& predictions,
    const std::vector<std::vector<float>>& targets
) {
    auto mse = calculateMSE(predictions, targets);
    double rmse = std::sqrt(mse.value);
    return {rmse, mse.unit, "Root Mean Squared Error"};
}

MetricResult BenchmarkUtils::calculateR2Score(
    const std::vector<std::vector<float>>& predictions,
    const std::vector<std::vector<float>>& targets
) {
    if (predictions.empty() || predictions.size() != targets.size()) {
        return {0.0, "", "R² - Invalid input dimensions"};
    }

    // Calculate mean of targets
    double targetMean = 0.0;
    size_t count = 0;

    for (const auto& target : targets) {
        for (float val : target) {
            targetMean += val;
            ++count;
        }
    }
    targetMean /= count;

    // Calculate SS_res and SS_tot
    double ssRes = 0.0;
    double ssTot = 0.0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        for (size_t j = 0; j < predictions[i].size(); ++j) {
            double pred = predictions[i][j];
            double target = targets[i][j];

            ssRes += (pred - target) * (pred - target);
            ssTot += (target - targetMean) * (target - targetMean);
        }
    }

    double r2 = ssTot > 0 ? 1.0 - (ssRes / ssTot) : 0.0;
    return {r2, "", "R² Score"};
}

MetricResult BenchmarkUtils::calculateAccuracy(
    const std::vector<int>& predictions,
    const std::vector<int>& targets
) {
    if (predictions.size() != targets.size()) {
        return {0.0, "", "Accuracy - Mismatched sizes"};
    }

    size_t correct = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == targets[i]) {
            ++correct;
        }
    }

    double accuracy = predictions.empty() ? 0.0 : static_cast<double>(correct) / predictions.size();
    return {accuracy, "", "Accuracy"};
}

MetricResult BenchmarkUtils::calculatePrecision(
    const std::vector<int>& predictions,
    const std::vector<int>& targets,
    int positiveClass
) {
    if (predictions.size() != targets.size()) {
        return {0.0, "", "Precision - Mismatched sizes"};
    }

    size_t truePositives = 0;
    size_t falsePositives = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == positiveClass) {
            if (targets[i] == positiveClass) {
                ++truePositives;
            } else {
                ++falsePositives;
            }
        }
    }

    double precision = (truePositives + falsePositives) > 0 ?
        static_cast<double>(truePositives) / (truePositives + falsePositives) : 0.0;
    return {precision, "", "Precision (class " + std::to_string(positiveClass) + ")"};
}

MetricResult BenchmarkUtils::calculateRecall(
    const std::vector<int>& predictions,
    const std::vector<int>& targets,
    int positiveClass
) {
    if (predictions.size() != targets.size()) {
        return {0.0, "", "Recall - Mismatched sizes"};
    }

    size_t truePositives = 0;
    size_t falseNegatives = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        if (targets[i] == positiveClass) {
            if (predictions[i] == positiveClass) {
                ++truePositives;
            } else {
                ++falseNegatives;
            }
        }
    }

    double recall = (truePositives + falseNegatives) > 0 ?
        static_cast<double>(truePositives) / (truePositives + falseNegatives) : 0.0;
    return {recall, "", "Recall (class " + std::to_string(positiveClass) + ")"};
}

MetricResult BenchmarkUtils::calculateF1Score(
    const std::vector<int>& predictions,
    const std::vector<int>& targets,
    int positiveClass
) {
    auto precision = calculatePrecision(predictions, targets, positiveClass);
    auto recall = calculateRecall(predictions, targets, positiveClass);

    double f1 = (precision.value + recall.value) > 0 ?
        2.0 * (precision.value * recall.value) / (precision.value + recall.value) : 0.0;
    return {f1, "", "F1 Score (class " + std::to_string(positiveClass) + ")"};
}

size_t BenchmarkUtils::estimateMemoryUsage(const std::vector<std::vector<float>>& data) {
    size_t totalBytes = 0;

    // Size of the outer vector
    totalBytes += sizeof(std::vector<std::vector<float>>);

    for (const auto& inner : data) {
        // Size of each inner vector
        totalBytes += sizeof(std::vector<float>);
        // Size of the float data
        totalBytes += inner.size() * sizeof(float);
    }

    return totalBytes;
}

double BenchmarkUtils::calculateMean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double BenchmarkUtils::calculateStdDev(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;

    double mean = calculateMean(values);
    double variance = 0.0;

    for (double val : values) {
        double diff = val - mean;
        variance += diff * diff;
    }

    variance /= (values.size() - 1); // Sample standard deviation
    return std::sqrt(variance);
}

double BenchmarkUtils::calculateMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;

    std::sort(values.begin(), values.end());
    size_t n = values.size();

    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    } else {
        return values[n/2];
    }
}
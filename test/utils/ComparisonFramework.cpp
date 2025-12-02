#include "ComparisonFramework.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

void ComparisonFramework::registerSystem(std::unique_ptr<AISystem> system) {
    systems_.push_back(std::move(system));
}

ComparisonReport ComparisonFramework::compareRegression(
    const std::string& testName,
    const std::vector<std::vector<float>>& trainInputs,
    const std::vector<std::vector<float>>& trainTargets,
    const std::vector<std::vector<float>>& testInputs,
    const std::vector<std::vector<float>>& testTargets,
    size_t benchmarkIterations
) {
    ComparisonReport report;
    report.testName = testName;

    for (auto& system : systems_) {
        system->reset();

        // Benchmark training
        auto trainBenchmark = benchmarkTraining(*system, trainInputs, trainTargets, benchmarkIterations);

        // Benchmark prediction
        auto predictBenchmark = benchmarkPrediction(*system, testInputs, benchmarkIterations);

        // Get predictions for metrics
        auto predictions = system->predict(testInputs);

        // Calculate metrics
        auto metrics = calculateMetricsRegression(predictions, testTargets);

        // Combine benchmarks (use prediction time as primary metric)
        BenchmarkResult combinedBenchmark = predictBenchmark;
        combinedBenchmark.name = system->getName() + " (prediction)";

        SystemResult result{
            system->getName(),
            combinedBenchmark,
            metrics,
            "Training time: " + std::to_string(trainBenchmark.avgTimePerIterationMs) + "ms"
        };

        report.results.push_back(std::move(result));
    }

    // Generate summary
    report.summary = generateSummaryMap(report);

    return report;
}

ComparisonReport ComparisonFramework::compareClassification(
    const std::string& testName,
    const std::vector<std::vector<float>>& trainInputs,
    const std::vector<int>& trainTargets,
    const std::vector<std::vector<float>>& testInputs,
    const std::vector<int>& testTargets,
    size_t benchmarkIterations
) {
    ComparisonReport report;
    report.testName = testName;

    // Convert targets to float vectors for training
    std::vector<std::vector<float>> trainTargetsFloat;
    std::vector<std::vector<float>> testTargetsFloat;

    for (int target : trainTargets) {
        trainTargetsFloat.push_back({static_cast<float>(target)});
    }

    for (int target : testTargets) {
        testTargetsFloat.push_back({static_cast<float>(target)});
    }

    for (auto& system : systems_) {
        system->reset();

        // Benchmark training
        auto trainBenchmark = benchmarkTraining(*system, trainInputs, trainTargetsFloat, benchmarkIterations);

        // Benchmark prediction
        auto predictBenchmark = benchmarkPrediction(*system, testInputs, benchmarkIterations);

        // Get predictions for metrics
        auto predictionsFloat = system->predict(testInputs);
        auto predictions = predictionsToLabels(predictionsFloat);

        // Calculate metrics
        auto metrics = calculateMetricsClassification(predictions, testTargets);

        // Combine benchmarks
        BenchmarkResult combinedBenchmark = predictBenchmark;
        combinedBenchmark.name = system->getName() + " (prediction)";

        SystemResult result{
            system->getName(),
            combinedBenchmark,
            metrics,
            "Training time: " + std::to_string(trainBenchmark.avgTimePerIterationMs) + "ms"
        };

        report.results.push_back(std::move(result));
    }

    // Generate summary
    report.summary = generateSummaryMap(report);

    return report;
}

std::string ComparisonFramework::generateSummary(const ComparisonReport& report) {
    std::stringstream ss;
    ss << "Comparison Report: " << report.testName << "\n";
    ss << "================================\n\n";

    for (const auto& result : report.results) {
        ss << "System: " << result.systemName << "\n";
        ss << "  Benchmark: " << result.benchmarkResult.avgTimePerIterationMs << "ms avg\n";
        ss << "  Metrics:\n";

        for (const auto& metric : result.metrics) {
            ss << "    " << metric.description << ": " << std::fixed << std::setprecision(4) << metric.value;
            if (!metric.unit.empty()) ss << " " << metric.unit;
            ss << "\n";
        }

        if (!result.notes.empty()) {
            ss << "  Notes: " << result.notes << "\n";
        }
        ss << "\n";
    }

    // Summary
    ss << "Summary:\n";
    for (const auto& pair : report.summary) {
        ss << "  " << pair.first << ": " << pair.second << "\n";
    }

    return ss.str();
}

bool ComparisonFramework::exportResults(const ComparisonReport& report, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << generateSummary(report);
    return true;
}

BenchmarkResult ComparisonFramework::benchmarkTraining(
    AISystem& system,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& targets,
    size_t iterations
) {
    return BenchmarkUtils::benchmarkFunction(
        system.getName() + " training",
        [&]() { system.train(inputs, targets); },
        iterations
    );
}

BenchmarkResult ComparisonFramework::benchmarkPrediction(
    AISystem& system,
    const std::vector<std::vector<float>>& inputs,
    size_t iterations
) {
    return BenchmarkUtils::benchmarkFunction(
        system.getName() + " prediction",
        [&]() { system.predict(inputs); },
        iterations
    );
}

std::vector<MetricResult> ComparisonFramework::calculateMetricsRegression(
    const std::vector<std::vector<float>>& predictions,
    const std::vector<std::vector<float>>& targets
) {
    std::vector<MetricResult> metrics;

    metrics.push_back(BenchmarkUtils::calculateMSE(predictions, targets));
    metrics.push_back(BenchmarkUtils::calculateMAE(predictions, targets));
    metrics.push_back(BenchmarkUtils::calculateRMSE(predictions, targets));
    metrics.push_back(BenchmarkUtils::calculateR2Score(predictions, targets));

    return metrics;
}

std::vector<MetricResult> ComparisonFramework::calculateMetricsClassification(
    const std::vector<int>& predictions,
    const std::vector<int>& targets
) {
    std::vector<MetricResult> metrics;

    metrics.push_back(BenchmarkUtils::calculateAccuracy(predictions, targets));

    // Calculate metrics for each class (assuming binary classification for simplicity)
    int numClasses = 0;
    for (int label : targets) {
        numClasses = std::max(numClasses, label + 1);
    }

    for (int c = 0; c < numClasses; ++c) {
        metrics.push_back(BenchmarkUtils::calculatePrecision(predictions, targets, c));
        metrics.push_back(BenchmarkUtils::calculateRecall(predictions, targets, c));
        metrics.push_back(BenchmarkUtils::calculateF1Score(predictions, targets, c));
    }

    return metrics;
}

std::vector<int> ComparisonFramework::predictionsToLabels(const std::vector<std::vector<float>>& predictions) {
    std::vector<int> labels;
    labels.reserve(predictions.size());

    for (const auto& pred : predictions) {
        if (pred.empty()) {
            labels.push_back(0);
        } else {
            // For multi-class, take argmax; for binary, threshold at 0.5
            if (pred.size() == 1) {
                labels.push_back(pred[0] >= 0.5f ? 1 : 0);
            } else {
                auto maxIt = std::max_element(pred.begin(), pred.end());
                labels.push_back(static_cast<int>(maxIt - pred.begin()));
            }
        }
    }

    return labels;
}

std::map<std::string, std::string> ComparisonFramework::generateSummaryMap(const ComparisonReport& report) {
    std::map<std::string, std::string> summary;

    if (report.results.empty()) return summary;

    // Find best system by average time
    const SystemResult* fastest = &report.results[0];
    for (const auto& result : report.results) {
        if (result.benchmarkResult.avgTimePerIterationMs < fastest->benchmarkResult.avgTimePerIterationMs) {
            fastest = &result;
        }
    }

    summary["Fastest System"] = fastest->systemName;
    summary["Best Avg Time"] = std::to_string(fastest->benchmarkResult.avgTimePerIterationMs) + "ms";

    // Find system with best primary metric (first metric)
    if (!report.results[0].metrics.empty()) {
        const SystemResult* best = &report.results[0];
        for (const auto& result : report.results) {
            if (!result.metrics.empty() &&
                result.metrics[0].value > best->metrics[0].value) {
                best = &result;
            }
        }

        summary["Best Performance"] = best->systemName + " (" +
            std::to_string(best->metrics[0].value) + ")";
    }

    return summary;
}
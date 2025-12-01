#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <algorithm>
#include "CoreAI3D/include/ValidationFramework.hpp"
#include "CoreAI3D/include/Train.hpp"

void createSampleDataset(const std::string& filename, int numSamples = 1000, int numFeatures = 5) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot create file " << filename << std::endl;
        return;
    }

    // Write header
    for (int i = 0; i < numFeatures; i++) {
        file << "feature_" << i;
        if (i < numFeatures - 1) file << ",";
    }
    file << ",target\n";

    // Generate synthetic data with known relationships
    for (int i = 0; i < numSamples; i++) {
        std::vector<float> features;
        float target = 0.0f;

        for (int j = 0; j < numFeatures; j++) {
            // Generate features with some correlation
            float feature = (rand() % 1000) / 100.0f; // 0.0 to 10.0

            // Create linear relationship with target
            target += feature * (0.5f + (j % 3) * 0.3f); // Different weights

            features.push_back(feature);
        }

        // Add noise
        target += (rand() % 200 - 100) / 100.0f * 0.1f; // ±1.0 noise

        // Write features
        for (size_t j = 0; j < features.size(); j++) {
            file << features[j];
            if (j < features.size() - 1) file << ",";
        }
        file << "," << target << "\n";
    }

    file.close();
    std::cout << "Created sample dataset: " << filename << " (" << numSamples << " samples, " << numFeatures << " features)" << std::endl;
}

void demonstrateDatasetValidation() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "DATASET VALIDATION DEMONSTRATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    ValidationFramework validator(true);

    // Create and validate a dataset
    std::string datasetFile = "demo_dataset.csv";
    createSampleDataset(datasetFile, 500, 4);

    auto validationResult = validator.validateDataset(datasetFile, true);

    std::cout << "\nValidation Result:" << std::endl;
    std::cout << "Valid: " << (validationResult.isValid ? "YES" : "NO") << std::endl;
    std::cout << "Quality Score: " << validationResult.qualityScore << std::endl;

    if (!validationResult.errorMessage.empty()) {
        std::cout << "Error: " << validationResult.errorMessage << std::endl;
    }

    if (!validationResult.warningMessage.empty()) {
        std::cout << "Warning: " << validationResult.warningMessage << std::endl;
    }

    std::cout << "\nValidation Checks:" << std::endl;
    for (const auto& check : validationResult.validationChecks) {
        std::cout << "  " << check << std::endl;
    }

    std::cout << "\nDataset Statistics:" << std::endl;
    for (const auto& stat : validationResult.statistics) {
        std::cout << "  " << stat.first << ": " << stat.second << std::endl;
    }

    // Generate validation report
    std::string report = validator.generateValidationReport(validationResult);
    validator.saveReportToFile("validation_report.txt", report);
    std::cout << "\nValidation report saved to: validation_report.txt" << std::endl;
}

void demonstrateBenchmarking() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "COMPREHENSIVE BENCHMARKING DEMONSTRATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    ValidationFramework validator(true);

    // Load data for benchmarking
    Training loader(false, false);
    std::string datasetFile = "benchmark_dataset.csv";
    createSampleDataset(datasetFile, 800, 3);

    bool loaded = loader.loadCSV(datasetFile, 800, 1, true, false, ',', "benchmark");
    if (!loaded) {
        std::cout << "Failed to load dataset for benchmarking" << std::endl;
        return;
    }

    const auto& inputs = loader.getInputs();
    const auto& targets = loader.getTargets();

    std::cout << "Loaded dataset: " << inputs.size() << " samples, " << inputs[0].size() << " features" << std::endl;

    // Run comprehensive benchmarking
    auto startTime = std::chrono::high_resolution_clock::now();
    auto results = validator.benchmarkAgainstBaselines(inputs, targets, 0.2f);
    auto endTime = std::chrono::high_resolution_clock::now();

    auto totalBenchmarkTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "\nBenchmarking completed in " << totalBenchmarkTime.count() << " ms" << std::endl;
    std::cout << "Models evaluated: " << results.size() << std::endl;

    // Display results
    std::cout << "\n" << std::string(100, '-') << std::endl;
    std::cout << std::left << std::setw(25) << "Model"
              << std::setw(10) << "R² Score"
              << std::setw(10) << "RMSE"
              << std::setw(10) << "MAE"
              << std::setw(12) << "Train Time"
              << std::setw(12) << "Infer Time"
              << std::setw(10) << "Params" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    // Sort by R² score for regression
    std::sort(results.begin(), results.end(), [](const BenchmarkResult& a, const BenchmarkResult& b) {
        return a.metrics.r2_score > b.metrics.r2_score;
    });

    for (const auto& result : results) {
        std::cout << std::left << std::setw(25) << result.modelName.substr(0, 24)
                  << std::fixed << std::setprecision(4)
                  << std::setw(10) << result.metrics.r2_score
                  << std::setw(10) << result.metrics.rmse
                  << std::setw(10) << result.metrics.mae
                  << std::setw(12) << std::chrono::duration_cast<std::chrono::milliseconds>(result.trainingTime).count()
                  << std::setw(12) << std::chrono::duration_cast<std::chrono::milliseconds>(result.inferenceTime).count()
                  << std::setw(10) << result.metrics.parameters << std::endl;
    }

    // Generate benchmark report
    std::string benchmarkReport = validator.generateBenchmarkReport(results);
    validator.saveReportToFile("benchmark_report.txt", benchmarkReport);
    std::cout << "\nBenchmark report saved to: benchmark_report.txt" << std::endl;
}

void demonstrateCrossValidation() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CROSS-VALIDATION DEMONSTRATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    ValidationFramework validator(true);

    // Create dataset for cross-validation
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> targets;

    for (int i = 0; i < 200; i++) {
        float x1 = (rand() % 100) / 10.0f;
        float x2 = (rand() % 100) / 10.0f;
        float y = 2.0f * x1 + 1.5f * x2 + (rand() % 20 - 10) / 100.0f;

        data.push_back({x1, x2});
        targets.push_back({y});
    }

    std::cout << "Running 5-fold cross-validation..." << std::endl;

    auto trainer = std::make_unique<Training>(false, true);
    auto cvResults = validator.kFoldCrossValidation(std::move(trainer), data, targets, 5, 0.01, 50);

    std::cout << "\nCross-validation Results:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(8) << "Fold"
              << std::setw(12) << "R² Score"
              << std::setw(12) << "RMSE"
              << std::setw(12) << "MAE"
              << std::setw(12) << "Train Time"
              << std::setw(12) << "Infer Time" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    float avgR2 = 0.0f, avgRMSE = 0.0f, avgMAE = 0.0f;

    for (size_t i = 0; i < cvResults.size(); i++) {
        const auto& metrics = cvResults[i];
        std::cout << std::left << std::setw(8) << ("Fold " + std::to_string(i + 1))
                  << std::fixed << std::setprecision(4)
                  << std::setw(12) << metrics.r2_score
                  << std::setw(12) << metrics.rmse
                  << std::setw(12) << metrics.mae
                  << std::setw(12) << metrics.trainingTime
                  << std::setw(12) << metrics.inferenceTime << std::endl;

        avgR2 += metrics.r2_score;
        avgRMSE += metrics.rmse;
        avgMAE += metrics.mae;
    }

    avgR2 /= cvResults.size();
    avgRMSE /= cvResults.size();
    avgMAE /= cvResults.size();

    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(8) << "Average"
              << std::fixed << std::setprecision(4)
              << std::setw(12) << avgR2
              << std::setw(12) << avgRMSE
              << std::setw(12) << avgMAE
              << std::setw(12) << "-" << std::setw(12) << "-" << std::endl;
}

void demonstrateScikitLearnIntegration() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "SCIKIT-LEARN INTEGRATION DEMONSTRATION" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    ValidationFramework validator(true);

    // Create test data
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> targets;

    for (int i = 0; i < 300; i++) {
        float x1 = (rand() % 100) / 10.0f;
        float x2 = (rand() % 100) / 10.0f;
        float x3 = (rand() % 100) / 10.0f;
        float y = 1.5f * x1 + 2.0f * x2 - 0.5f * x3 + (rand() % 20 - 10) / 100.0f;

        data.push_back({x1, x2, x3});
        targets.push_back({y});
    }

    std::cout << "Testing scikit-learn integration..." << std::endl;

    try {
        auto sklearnResults = validator.benchmarkAgainstScikitLearn(data, targets, 0.25f);

        if (sklearnResults.empty()) {
            std::cout << "Scikit-learn integration not available (Python/scikit-learn not found)" << std::endl;
            std::cout << "To enable scikit-learn benchmarking:" << std::endl;
            std::cout << "1. Install Python 3.x" << std::endl;
            std::cout << "2. Install scikit-learn: pip install scikit-learn" << std::endl;
            std::cout << "3. Ensure python command is in PATH" << std::endl;
            return;
        }

        std::cout << "Scikit-learn benchmarking successful!" << std::endl;
        std::cout << "Models evaluated: " << sklearnResults.size() << std::endl;

        // Display top 5 models
        std::cout << "\nTop 5 Scikit-learn Models:" << std::endl;
        std::cout << std::string(90, '-') << std::endl;
        std::cout << std::left << std::setw(25) << "Model"
                  << std::setw(10) << "R² Score"
                  << std::setw(10) << "RMSE"
                  << std::setw(10) << "MAE"
                  << std::setw(12) << "Train Time"
                  << std::setw(12) << "Infer Time" << std::endl;
        std::cout << std::string(90, '-') << std::endl;

        // Sort by R² score
        std::sort(sklearnResults.begin(), sklearnResults.end(),
                 [](const BenchmarkResult& a, const BenchmarkResult& b) {
                     return a.metrics.r2_score > b.metrics.r2_score;
                 });

        for (size_t i = 0; i < std::min(size_t(5), sklearnResults.size()); i++) {
            const auto& result = sklearnResults[i];
            std::cout << std::left << std::setw(25) << result.modelName.substr(0, 24)
                      << std::fixed << std::setprecision(4)
                      << std::setw(10) << result.metrics.r2_score
                      << std::setw(10) << result.metrics.rmse
                      << std::setw(10) << result.metrics.mae
                      << std::setw(12) << std::chrono::duration_cast<std::chrono::milliseconds>(result.trainingTime).count()
                      << std::setw(12) << std::chrono::duration_cast<std::chrono::milliseconds>(result.inferenceTime).count() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Scikit-learn integration failed: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "CoreAI3D Validation Framework - Comprehensive Demonstration" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Seed random number generator
    srand(static_cast<unsigned int>(time(nullptr)));

    try {
        // Demonstrate dataset validation
        demonstrateDatasetValidation();

        // Demonstrate comprehensive benchmarking
        demonstrateBenchmarking();

        // Demonstrate cross-validation
        demonstrateCrossValidation();

        // Demonstrate scikit-learn integration
        demonstrateScikitLearnIntegration();

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "DEMONSTRATION COMPLETED SUCCESSFULLY!" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::cout << "\nGenerated files:" << std::endl;
        std::cout << "- validation_report.txt" << std::endl;
        std::cout << "- benchmark_report.txt" << std::endl;
        std::cout << "- demo_dataset.csv" << std::endl;
        std::cout << "- benchmark_dataset.csv" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during demonstration: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
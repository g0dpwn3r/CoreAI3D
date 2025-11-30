#include "CoreAI3D/include/ValidationFramework.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

// Demo function to create sample datasets
void createSampleDataset(const std::string& filename, int numSamples, int numFeatures, bool isClassification = false) {
    std::ofstream file(filename);
    
    // Write header
    for (int i = 0; i < numFeatures; i++) {
        file << "feature_" << i;
        if (i < numFeatures - 1) file << ",";
    }
    file << (isClassification ? ",class" : ",target") << "\n";
    
    for (int i = 0; i < numSamples; i++) {
        // Generate features
        std::vector<float> features;
        for (int j = 0; j < numFeatures; j++) {
            float feature = (rand() % 1000) / 100.0f; // 0.0 to 10.0
            features.push_back(feature);
        }
        
        float target;
        if (isClassification) {
            // Binary classification: target based on sum of features
            float sum = 0.0f;
            for (float f : features) sum += f;
            target = sum > features.size() * 5.0f ? 1 : 0;
        } else {
            // Regression: y = weighted sum with noise
            target = 0.0f;
            for (int j = 0; j < numFeatures; j++) {
                target += features[j] * (0.5f / numFeatures);
            }
            target += (rand() % 200 - 100) / 100.0f * 0.1f; // Add noise
        }
        
        // Write to file
        for (int j = 0; j < numFeatures; j++) {
            file << features[j] << ",";
        }
        file << target << "\n";
    }
    
    file.close();
    std::cout << "Created dataset: " << filename << " with " << numSamples 
              << " samples and " << numFeatures << " features" << std::endl;
}

void demonstrateDatasetValidation() {
    std::cout << "\n=== DATASET VALIDATION DEMONSTRATION ===" << std::endl;
    
    ValidationFramework validator(true);
    
    // Create sample datasets
    createSampleDataset("demo_regression.csv", 200, 5);
    createSampleDataset("demo_classification.csv", 150, 4, true);
    
    // Validate regression dataset
    std::cout << "\n1. Validating Regression Dataset:" << std::endl;
    auto regressionResult = validator.validateDataset("demo_regression.csv", true);
    
    std::cout << "Validation Status: " << (regressionResult.isValid ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Quality Score: " << regressionResult.qualityScore << std::endl;
    
    if (!regressionResult.warningMessage.empty()) {
        std::cout << "Warning: " << regressionResult.warningMessage << std::endl;
    }
    
    // Validate classification dataset
    std::cout << "\n2. Validating Classification Dataset:" << std::endl;
    auto classificationResult = validator.validateDataset("demo_classification.csv", true);
    
    std::cout << "Validation Status: " << (classificationResult.isValid ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Quality Score: " << classificationResult.qualityScore << std::endl;
    
    // Generate validation reports
    std::cout << "\n3. Generating Validation Reports:" << std::endl;
    auto regressionReport = validator.generateValidationReport(regressionResult);
    auto classificationReport = validator.generateValidationReport(classificationResult);
    
    validator.saveReportToFile("regression_validation_report.txt", regressionReport);
    validator.saveReportToFile("classification_validation_report.txt", classificationReport);
    
    std::cout << "Validation reports saved to files." << std::endl;
    
    // Clean up demo files
    std::remove("demo_regression.csv");
    std::remove("demo_classification.csv");
}

void demonstrateBenchmarking() {
    std::cout << "\n=== PERFORMANCE BENCHMARKING DEMONSTRATION ===" << std::endl;
    
    ValidationFramework validator(true);
    
    // Create datasets for benchmarking
    createSampleDataset("benchmark_regression.csv", 300, 6);
    createSampleDataset("benchmark_classification.csv", 200, 5, true);
    
    // Benchmark regression dataset
    std::cout << "\n1. Regression Dataset Benchmark:" << std::endl;
    
    // Load data using Training class
    Training loader(false, false); // offline mode
    bool loaded = loader.loadCSV("benchmark_regression.csv", 300, 1, true, false, ',', "benchmark_regression");
    
    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();
        
        std::cout << "Dataset loaded: " << inputs.size() << " samples, " 
                  << inputs[0].size() << " features" << std::endl;
        
        // Run comprehensive benchmark
        auto startTime = std::chrono::high_resolution_clock::now();
        auto results = validator.benchmarkAgainstBaselines(inputs, targets, 0.2f);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << "Benchmark completed in " << totalDuration.count() << " ms" << std::endl;
        
        // Display results
        std::cout << "\nBenchmark Results:" << std::endl;
        for (const auto& result : results) {
            std::cout << "\n" << result.modelName << ":" << std::endl;
            std::cout << "  Accuracy: " << std::fixed << std::setprecision(4) << result.metrics.accuracy << std::endl;
            std::cout << "  RMSE: " << result.metrics.rmse << std::endl;
            std::cout << "  R² Score: " << result.metrics.r2_score << std::endl;
            std::cout << "  Training Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(result.trainingTime).count() << " ms" << std::endl;
            std::cout << "  Parameters: " << result.metrics.parameters << std::endl;
        }
        
        // Generate benchmark report
        auto benchmarkReport = validator.generateBenchmarkReport(results);
        validator.saveReportToFile("benchmark_report.txt", benchmarkReport);
        std::cout << "\nBenchmark report saved to benchmark_report.txt" << std::endl;
    }
    
    // Benchmark classification dataset
    std::cout << "\n2. Classification Dataset Benchmark:" << std::endl;
    
    Training classifierLoader(false, false);
    bool classificationLoaded = classifierLoader.loadCSV("benchmark_classification.csv", 200, 1, true, false, ',', "benchmark_classification");
    
    if (classificationLoaded) {
        const auto& inputs = classifierLoader.getInputs();
        const auto& targets = classifierLoader.getTargets();
        
        auto classificationResults = validator.benchmarkAgainstBaselines(inputs, targets, 0.25f);
        
        std::cout << "Classification Benchmark Results:" << std::endl;
        for (const auto& result : classificationResults) {
            std::cout << "\n" << result.modelName << ":" << std::endl;
            std::cout << "  Accuracy: " << std::fixed << std::setprecision(4) << result.metrics.accuracy << std::endl;
            std::cout << "  Precision: " << result.metrics.precision << std::endl;
            std::cout << "  Recall: " << result.metrics.recall << std::endl;
            std::cout << "  F1-Score: " << result.metrics.f1_score << std::endl;
        }
    }
    
    // Clean up demo files
    std::remove("benchmark_regression.csv");
    std::remove("benchmark_classification.csv");
}

void demonstrateCrossValidation() {
    std::cout << "\n=== CROSS-VALIDATION DEMONSTRATION ===" << std::endl;
    
    ValidationFramework validator(true);
    
    // Create small dataset for cross-validation
    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}, {4.0f, 5.0f},
        {5.0f, 6.0f}, {6.0f, 7.0f}, {7.0f, 8.0f}, {8.0f, 9.0f},
        {9.0f, 10.0f}, {10.0f, 11.0f}, {11.0f, 12.0f}, {12.0f, 13.0f}
    };
    
    std::vector<std::vector<float>> targets = {
        {3.0f}, {5.0f}, {7.0f}, {9.0f}, {11.0f}, {13.0f},
        {15.0f}, {17.0f}, {19.0f}, {21.0f}, {23.0f}, {25.0f}
    };
    
    std::cout << "Running 5-fold cross-validation on synthetic dataset..." << std::endl;
    
    auto trainer = std::make_unique<Training>(false, true);
    auto cvResults = validator.kFoldCrossValidation(std::move(trainer), data, targets, 5, 0.01, 50);
    
    std::cout << "\nCross-Validation Results:" << std::endl;
    std::cout << "Number of folds: " << cvResults.size() << std::endl;
    
    float avgAccuracy = 0.0f, avgRMSE = 0.0f, avgMAE = 0.0f;
    for (size_t i = 0; i < cvResults.size(); i++) {
        std::cout << "Fold " << (i + 1) << ": "
                  << "Accuracy=" << cvResults[i].accuracy << ", "
                  << "RMSE=" << cvResults[i].rmse << ", "
                  << "MAE=" << cvResults[i].mae << std::endl;
        
        avgAccuracy += cvResults[i].accuracy;
        avgRMSE += cvResults[i].rmse;
        avgMAE += cvResults[i].mae;
    }
    
    avgAccuracy /= cvResults.size();
    avgRMSE /= cvResults.size();
    avgMAE /= cvResults.size();
    
    std::cout << "\nAverage Performance:" << std::endl;
    std::cout << "Average Accuracy: " << avgAccuracy << std::endl;
    std::cout << "Average RMSE: " << avgRMSE << std::endl;
    std::cout << "Average MAE: " << avgMAE << std::endl;
}

void demonstrateDataPreprocessing() {
    std::cout << "\n=== DATA PREPROCESSING DEMONSTRATION ===" << std::endl;
    
    ValidationFramework validator;
    
    // Create sample data with different scales
    std::vector<std::vector<float>> rawData = {
        {1.0f, 100.0f, 0.01f},
        {2.0f, 200.0f, 0.02f},
        {3.0f, 300.0f, 0.03f},
        {4.0f, 400.0f, 0.04f},
        {5.0f, 500.0f, 0.05f}
    };
    
    std::cout << "Original data (different scales):" << std::endl;
    for (const auto& row : rawData) {
        std::cout << "[";
        for (size_t i = 0; i < row.size(); i++) {
            std::cout << row[i];
            if (i < row.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // Normalize data
    std::vector<float> means, stds;
    auto normalizedData = validator.normalizeData(rawData, means, stds);
    
    std::cout << "\nComputed normalization parameters:" << std::endl;
    for (size_t i = 0; i < means.size(); i++) {
        std::cout << "Feature " << i << ": mean=" << means[i] << ", std=" << stds[i] << std::endl;
    }
    
    std::cout << "\nNormalized data (mean=0, std=1):" << std::endl;
    for (const auto& row : normalizedData) {
        std::cout << "[";
        for (size_t i = 0; i < row.size(); i++) {
            std::cout << std::fixed << std::setprecision(3) << row[i];
            if (i < row.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // Demonstrate data splitting
    std::vector<std::vector<float>> trainData, testData, trainTargets, testTargets;
    std::vector<std::vector<float>> dummyTargets = {{1.0f}, {2.0f}, {3.0f}, {4.0f}, {5.0f}};
    
    validator.splitData(rawData, dummyTargets, 0.6f, trainData, testData, trainTargets, testTargets);
    
    std::cout << "\nData split (60% train, 40% test):" << std::endl;
    std::cout << "Training samples: " << trainData.size() << std::endl;
    std::cout << "Test samples: " << testData.size() << std::endl;
}

void demonstratePerformanceComparison() {
    std::cout << "\n=== PERFORMANCE COMPARISON DEMONSTRATION ===" << std::endl;
    
    ValidationFramework validator(true);
    
    // Create linear dataset (should favor linear regression)
    std::vector<std::vector<float>> linearData;
    std::vector<std::vector<float>> linearTargets;
    
    for (int i = 0; i < 100; i++) {
        float x = (rand() % 1000) / 100.0f;
        float noise = (rand() % 20 - 10) / 100.0f;
        float y = 2.0f * x + 3.0f + noise; // y = 2x + 3 + noise
        
        linearData.push_back({x});
        linearTargets.push_back({y});
    }
    
    std::cout << "Testing on linear relationship: y = 2x + 3 + noise" << std::endl;
    
    auto results = validator.benchmarkAgainstBaselines(linearData, linearTargets, 0.2f);
    
    std::cout << "\nPerformance Comparison on Linear Data:" << std::endl;
    
    // Find best performing models
    BenchmarkResult* bestRMSE = nullptr;
    BenchmarkResult* bestR2 = nullptr;
    BenchmarkResult* fastest = nullptr;
    
    for (auto& result : results) {
        if (!bestRMSE || result.metrics.rmse < bestRMSE->metrics.rmse) {
            bestRMSE = &result;
        }
        if (!bestR2 || result.metrics.r2_score > bestR2->metrics.r2_score) {
            bestR2 = &result;
        }
        if (!fastest || result.trainingTime.count() < fastest->trainingTime.count()) {
            fastest = &result;
        }
    }
    
    if (bestRMSE) {
        std::cout << "Best RMSE: " << bestRMSE->modelName << " (RMSE=" << bestRMSE->metrics.rmse << ")" << std::endl;
    }
    if (bestR2) {
        std::cout << "Best R²: " << bestR2->modelName << " (R²=" << bestR2->metrics.r2_score << ")" << std::endl;
    }
    if (fastest) {
        std::cout << "Fastest Training: " << fastest->modelName 
                  << " (" << std::chrono::duration_cast<std::chrono::milliseconds>(fastest->trainingTime).count() << " ms)" << std::endl;
    }
    
    // Summary comparison
    std::cout << "\nModel Comparison Summary:" << std::endl;
    std::cout << std::setw(20) << "Model" << std::setw(10) << "RMSE" 
              << std::setw(10) << "R²" << std::setw(15) << "Training(ms)" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(20) << result.modelName 
                  << std::setw(10) << std::fixed << std::setprecision(4) << result.metrics.rmse
                  << std::setw(10) << result.metrics.r2_score
                  << std::setw(15) << std::chrono::duration_cast<std::chrono::milliseconds>(result.trainingTime).count() << std::endl;
    }
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              CoreAI3D Validation & Benchmarking Demo              ║" << std::endl;
    std::cout << "║        Comprehensive Dataset Validation and ML Model Comparison   ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝" << std::endl;
    
    try {
        // Run all demonstrations
        demonstrateDatasetValidation();
        demonstrateBenchmarking();
        demonstrateCrossValidation();
        demonstrateDataPreprocessing();
        demonstratePerformanceComparison();
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Demo completed successfully!" << std::endl;
        std::cout << "Check the generated report files for detailed results." << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
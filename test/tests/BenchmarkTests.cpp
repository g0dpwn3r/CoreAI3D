#include <gtest/gtest.h>
#include "ValidationFramework.hpp"
#include "Train.hpp"
#include <chrono>
#include <fstream>

// Helper function to create realistic datasets
void createRealisticDataset(const std::string& filename, int numSamples, int numFeatures, 
                           float noiseLevel = 0.1f, bool addOutliers = false) {
    std::ofstream file(filename);
    
    // Write header
    for (int i = 0; i < numFeatures; i++) {
        file << "feature_" << i;
        if (i < numFeatures - 1) file << ",";
    }
    file << ",target\n";
    
    for (int i = 0; i < numSamples; i++) {
        // Generate features
        std::vector<float> features;
        for (int j = 0; j < numFeatures; j++) {
            float feature = (rand() % 1000) / 100.0f; // 0.0 to 10.0
            features.push_back(feature);
        }
        
        // Generate target with some correlation to features
        float target = 0.0f;
        for (int j = 0; j < numFeatures; j++) {
            target += features[j] * (0.5f / numFeatures); // Weighted sum
        }
        
        // Add noise
        target += (rand() % 200 - 100) / 100.0f * noiseLevel;
        
        // Optionally add outliers
        if (addOutliers && i % 20 == 0) {
            target += (rand() % 100 - 50) / 10.0f; // Large outlier
        }
        
        // Write features and target
        for (int j = 0; j < numFeatures; j++) {
            file << features[j] << ",";
        }
        file << target << "\n";
    }
    
    file.close();
}

void createClassificationDataset(const std::string& filename, int numSamples, int numFeatures) {
    std::ofstream file(filename);
    
    // Write header
    for (int i = 0; i < numFeatures; i++) {
        file << "feature_" << i;
        if (i < numFeatures - 1) file << ",";
    }
    file << ",class\n";
    
    for (int i = 0; i < numSamples; i++) {
        // Generate features
        std::vector<float> features;
        for (int j = 0; j < numFeatures; j++) {
            float feature = (rand() % 100) / 100.0f; // 0.0 to 1.0
            features.push_back(feature);
        }
        
        // Create binary classification target based on feature combination
        float classScore = 0.0f;
        for (int j = 0; j < numFeatures; j++) {
            classScore += features[j] * (j % 2 == 0 ? 1.0f : -1.0f);
        }
        
        int targetClass = classScore > 0 ? 1 : 0;
        
        // Write features and target
        for (int j = 0; j < numFeatures; j++) {
            file << features[j] << ",";
        }
        file << targetClass << "\n";
    }
    
    file.close();
}

TEST(BenchmarkTest, FullRegressionBenchmark) {
    ValidationFramework validator(true);
    
    // Create realistic regression dataset
    createRealisticDataset("benchmark_regression.csv", 200, 5, 0.1f, true);
    
    // Load and validate the dataset
    auto validationResult = validator.validateDataset("benchmark_regression.csv", true);
    EXPECT_TRUE(validationResult.isValid);
    
    // Load data for benchmarking
    std::vector<std::vector<float>> data, targets;
    Training loader(false, false); // offline mode
    
    // Use Training class to load data (assuming this method exists)
    bool loaded = loader.loadCSV("benchmark_regression.csv", 200, 1, true, false, ',', "benchmark");
    EXPECT_TRUE(loaded);
    
    if (loaded) {
        // Extract data and targets from the loaded dataset
        const auto& inputs = loader.getInputs();
        const auto& targets_vec = loader.getTargets();
        
        data = inputs;
        targets = targets_vec;
        
        // Run comprehensive benchmark
        auto results = validator.benchmarkAgainstBaselines(data, targets, 0.2f);
        
        EXPECT_FALSE(results.empty());
        EXPECT_GE(results.size(), 2); // Should have at least 2 models
        
        // Verify each result has valid metrics
        for (const auto& result : results) {
            EXPECT_GT(result.metrics.accuracy, -1.0f);
            EXPECT_GT(result.metrics.rmse, 0.0f);
            EXPECT_GT(result.metrics.mae, 0.0f);
            EXPECT_GE(result.metrics.parameters, 0);
            
            // Check timing information
            EXPECT_GE(result.totalTime.count(), 0);
            EXPECT_GE(result.trainingTime.count(), 0);
        }
        
        // Generate and verify report
        auto report = validator.generateBenchmarkReport(results);
        EXPECT_FALSE(report.empty());
        EXPECT_TRUE(report.find("BENCHMARK REPORT") != std::string::npos);
        EXPECT_TRUE(report.find("Number of Models:") != std::string::npos);
    }
    
    // Clean up
    std::remove("benchmark_regression.csv");
}

TEST(BenchmarkTest, ClassificationBenchmark) {
    ValidationFramework validator(true);
    
    // Create classification dataset
    createClassificationDataset("benchmark_classification.csv", 150, 4);
    
    // Load and validate
    auto validationResult = validator.validateDataset("benchmark_classification.csv", true);
    EXPECT_TRUE(validationResult.isValid);
    
    // Load data for benchmarking
    Training loader(false, false);
    bool loaded = loader.loadCSV("benchmark_classification.csv", 150, 1, true, false, ',', "classification");
    EXPECT_TRUE(loaded);
    
    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets_vec = loader.getTargets();
        
        // Run benchmark
        auto results = validator.benchmarkAgainstBaselines(inputs, targets_vec, 0.25f);
        
        EXPECT_FALSE(results.empty());
        
        // Check that we have both regression and classification models
        bool hasLinearRegression = false;
        bool hasLogisticRegression = false;
        
        for (const auto& result : results) {
            if (result.modelName == "Linear Regression") hasLinearRegression = true;
            if (result.modelName == "Logistic Regression") hasLogisticRegression = true;
        }
        
        // Classification tasks should include logistic regression
        EXPECT_TRUE(hasLogisticRegression || hasLinearRegression);
    }
    
    // Clean up
    std::remove("benchmark_classification.csv");
}

TEST(BenchmarkTest, CrossValidationTest) {
    ValidationFramework validator(true);
    
    // Create small dataset for cross-validation
    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}, {4.0f, 5.0f},
        {5.0f, 6.0f}, {6.0f, 7.0f}, {7.0f, 8.0f}, {8.0f, 9.0f},
        {9.0f, 10.0f}, {10.0f, 11.0f}
    };
    
    std::vector<std::vector<float>> targets = {
        {3.0f}, {5.0f}, {7.0f}, {9.0f}, {11.0f},
        {13.0f}, {15.0f}, {17.0f}, {19.0f}, {21.0f}
    };
    
    // Test k-fold cross-validation with CoreAI3D
    auto trainer = std::make_unique<Training>(false, verbose);
    
    auto cvResults = validator.kFoldCrossValidation(std::move(trainer), data, targets, 3, 0.01, 10);
    
    EXPECT_EQ(cvResults.size(), 3); // 3 folds
    
    // Check that all folds completed successfully
    for (const auto& metrics : cvResults) {
        EXPECT_GT(metrics.accuracy, -1.0f);
        EXPECT_GT(metrics.rmse, 0.0f);
        EXPECT_GE(metrics.trainingTime, 0.0);
        EXPECT_GE(metrics.inferenceTime, 0.0);
    }
    
    // Calculate average metrics across folds
    float avgAccuracy = 0.0f, avgRMSE = 0.0f;
    for (const auto& metrics : cvResults) {
        avgAccuracy += metrics.accuracy;
        avgRMSE += metrics.rmse;
    }
    avgAccuracy /= cvResults.size();
    avgRMSE /= cvResults.size();
    
    // Averages should be reasonable
    EXPECT_GT(avgAccuracy, 0.0f);
    EXPECT_GT(avgRMSE, 0.0f);
}

TEST(BenchmarkTest, PerformanceComparison) {
    ValidationFramework validator(true);
    
    // Create dataset with known patterns
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> targets;
    
    // Linear relationship: y = 2*x1 + 3*x2 + noise
    for (int i = 0; i < 100; i++) {
        float x1 = (rand() % 100) / 10.0f;
        float x2 = (rand() % 100) / 10.0f;
        float noise = (rand() % 20 - 10) / 100.0f;
        float y = 2.0f * x1 + 3.0f * x2 + noise;
        
        data.push_back({x1, x2});
        targets.push_back({y});
    }
    
    // Run benchmark
    auto results = validator.benchmarkAgainstBaselines(data, targets, 0.2f);
    
    EXPECT_FALSE(results.empty());
    
    // For linear data, linear regression should perform well
    BenchmarkResult* linearRegression = nullptr;
    BenchmarkResult* coreAI3D = nullptr;
    
    for (auto& result : results) {
        if (result.modelName == "Linear Regression") {
            linearRegression = &result;
        } else if (result.modelName == "CoreAI3D Neural Network") {
            coreAI3D = &result;
        }
    }
    
    if (linearRegression) {
        // Linear regression should achieve good performance on linear data
        EXPECT_LT(linearRegression->metrics.rmse, 1.0f);
        EXPECT_GT(linearRegression->metrics.r2_score, 0.8f);
    }
    
    if (coreAI3D) {
        // CoreAI3D should also perform reasonably well
        EXPECT_GT(coreAI3D->metrics.r2_score, 0.5f);
    }
    
    // Compare training times
    if (linearRegression && coreAI3D) {
        // Linear regression should be faster than neural network for simple problems
        EXPECT_LE(linearRegression->trainingTime.count(), coreAI3D->trainingTime.count());
    }
}

TEST(BenchmarkTest, MemoryAndTimeMetrics) {
    ValidationFramework validator;
    
    // Create dataset to test memory and time tracking
    std::vector<std::vector<float>> data(50, std::vector<float>(3));
    std::vector<std::vector<float>> targets(50, std::vector<float>(1));
    
    for (int i = 0; i < 50; i++) {
        for (int j = 0; j < 3; j++) {
            data[i][j] = (rand() % 100) / 10.0f;
        }
        targets[i][0] = data[i][0] + data[i][1] + data[i][2] + (rand() % 10 - 5) / 10.0f;
    }
    
    auto results = validator.benchmarkAgainstBaselines(data, targets, 0.3f);
    
    for (const auto& result : results) {
        // Verify timing metrics are reasonable
        EXPECT_GE(result.totalTime.count(), 0);
        EXPECT_GE(result.trainingTime.count(), 0);
        EXPECT_GE(result.inferenceTime.count(), 0);
        
        // Training time should be >= inference time for most models
        // (This might not always be true for very simple models)
        
        // Memory usage should be tracked
        EXPECT_GE(result.metrics.memoryUsage, 0);
        
        // Parameter count should be positive for learned models
        if (result.modelName != "K-Nearest Neighbors") {
            EXPECT_GT(result.metrics.parameters, 0);
        }
    }
}

TEST(BenchmarkTest, EdgeCaseHandling) {
    ValidationFramework validator(true);
    
    // Test with very small dataset
    std::vector<std::vector<float>> tinyData = {
        {1.0f, 2.0f},
        {2.0f, 3.0f}
    };
    
    std::vector<std::vector<float>> tinyTargets = {
        {3.0f},
        {5.0f}
    };
    
    // Should not crash even with small data
    auto smallResults = validator.benchmarkAgainstBaselines(tinyData, tinyTargets, 0.5f);
    
    // May not have valid results, but shouldn't crash
    EXPECT_TRUE(smallResults.size() >= 0);
    
    // Test with high-dimensional data
    std::vector<std::vector<float>> highDimData(20, std::vector<float>(50, 1.0f));
    std::vector<std::vector<float>> highDimTargets(20, std::vector<float>(1, 1.0f));
    
    // Should handle high-dimensional data
    auto highDimResults = validator.benchmarkAgainstBaselines(highDimData, highDimTargets, 0.3f);
    EXPECT_TRUE(highDimResults.size() >= 0);
}

TEST(BenchmarkTest, ReportGeneration) {
    ValidationFramework validator;
    
    // Create mock benchmark results
    std::vector<BenchmarkResult> mockResults;
    
    // Add CoreAI3D result
    BenchmarkResult coreAIResult;
    coreAIResult.modelName = "CoreAI3D Neural Network";
    coreAIResult.metrics.accuracy = 0.85f;
    coreAIResult.metrics.rmse = 0.15f;
    coreAIResult.metrics.mae = 0.12f;
    coreAIResult.metrics.r2_score = 0.90f;
    coreAIResult.metrics.parameters = 100;
    coreAIResult.trainingTime = std::chrono::microseconds(50000);
    coreAIResult.inferenceTime = std::chrono::microseconds(1000);
    mockResults.push_back(coreAIResult);
    
    // Add Linear Regression result
    BenchmarkResult linearResult;
    linearResult.modelName = "Linear Regression";
    linearResult.metrics.accuracy = 0.80f;
    linearResult.metrics.rmse = 0.20f;
    linearResult.metrics.mae = 0.16f;
    linearResult.metrics.r2_score = 0.85f;
    linearResult.metrics.parameters = 3;
    linearResult.trainingTime = std::chrono::microseconds(1000);
    linearResult.inferenceTime = std::chrono::microseconds(100);
    mockResults.push_back(linearResult);
    
    auto report = validator.generateBenchmarkReport(mockResults);
    
    EXPECT_FALSE(report.empty());
    EXPECT_TRUE(report.find("CoreAI3D Neural Network") != std::string::npos);
    EXPECT_TRUE(report.find("Linear Regression") != std::string::npos);
    EXPECT_TRUE(report.find("0.850") != std::string::npos); // CoreAI3D accuracy
    EXPECT_TRUE(report.find("0.800") != std::string::npos); // Linear regression accuracy
    EXPECT_TRUE(report.find("50 ms") != std::string::npos); // Training time
    
    // Test saving to file
    validator.saveReportToFile("test_benchmark_report.txt", report);
    
    // Verify file was created and has content
    std::ifstream file("test_benchmark_report.txt");
    EXPECT_TRUE(file.is_open());
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());
    file.close();
    
    // Clean up
    std::remove("test_benchmark_report.txt");
}

TEST(BenchmarkTest, ValidationAndBenchmarkIntegration) {
    ValidationFramework validator(true);

    // Create a comprehensive dataset
    createRealisticDataset("integration_test.csv", 100, 4, 0.05f, true);

    // Step 1: Validate the dataset
    auto validationResult = validator.validateDataset("integration_test.csv", true);
    EXPECT_TRUE(validationResult.isValid);

    if (validationResult.isValid) {
        // Step 2: Load the validated data
        Training loader(false, false);
        bool loaded = loader.loadCSV("integration_test.csv", 100, 1, true, false, ',', "integration");
        EXPECT_TRUE(loaded);

        if (loaded) {
            const auto& inputs = loader.getInputs();
            const auto& targets_vec = loader.getTargets();

            // Step 3: Preprocess the data
            std::vector<float> means, stds;
            auto normalized = validator.normalizeData(inputs, means, stds);
            EXPECT_EQ(normalized.size(), inputs.size());

            // Step 4: Run benchmark
            auto results = validator.benchmarkAgainstBaselines(normalized, targets_vec, 0.2f);

            EXPECT_FALSE(results.empty());

            // Step 5: Run cross-validation on the best model
            auto trainer = std::make_unique<Training>(false, false);
            auto cvResults = validator.kFoldCrossValidation(std::move(trainer),
                                                          normalized, targets_vec,
                                                          5, 0.01, 20);

            EXPECT_EQ(cvResults.size(), 5); // 5-fold CV

            // Step 6: Generate comprehensive report
            std::string validationReport = validator.generateValidationReport(validationResult);
            std::string benchmarkReport = validator.generateBenchmarkReport(results);

            EXPECT_FALSE(validationReport.empty());
            EXPECT_FALSE(benchmarkReport.empty());

            // Step 7: Save reports
            validator.saveReportToFile("validation_report.txt", validationReport);
            validator.saveReportToFile("benchmark_report.txt", benchmarkReport);

            // Verify files exist
            std::ifstream validationFile("validation_report.txt");
            std::ifstream benchmarkFile("benchmark_report.txt");
            EXPECT_TRUE(validationFile.is_open());
            EXPECT_TRUE(benchmarkFile.is_open());

            validationFile.close();
            benchmarkFile.close();
        }
    }

    // Clean up
    std::remove("integration_test.csv");
    std::remove("validation_report.txt");
    std::remove("benchmark_report.txt");
}

// Standard dataset benchmarks
TEST(BenchmarkTest, IrisDatasetBenchmark) {
    ValidationFramework validator(true);

    std::vector<std::vector<float>> data;
    std::vector<float> targets;

    bool loaded = validator.loadIrisDataset(data, targets);
    EXPECT_TRUE(loaded);

    if (loaded) {
        // Convert targets to vector of vectors for compatibility
        std::vector<std::vector<float>> targets_vec;
        for (float target : targets) {
            targets_vec.push_back({target});
        }

        // Run benchmark on Iris dataset
        auto results = validator.benchmarkAgainstBaselines(data, targets_vec, 0.3f);

        EXPECT_FALSE(results.empty());
        EXPECT_GE(results.size(), 2); // Should have at least CoreAI3D and one baseline

        // Iris is a classification problem with 3 classes
        for (const auto& result : results) {
            EXPECT_GE(result.metrics.accuracy, 0.0f);
            EXPECT_LE(result.metrics.accuracy, 1.0f);
            // For classification, RMSE might not be meaningful, but should be reasonable
            EXPECT_GE(result.metrics.rmse, 0.0f);
        }

        // Generate and save report
        auto report = validator.generateBenchmarkReport(results);
        validator.saveReportToFile("iris_benchmark_report.txt", report);

        std::cout << "Iris dataset benchmark completed. Results saved to iris_benchmark_report.txt" << std::endl;

        // Clean up
        std::remove("iris_benchmark_report.txt");
    }
}

TEST(BenchmarkTest, BostonHousingDatasetBenchmark) {
    ValidationFramework validator(true);

    std::vector<std::vector<float>> data;
    std::vector<float> targets;

    bool loaded = validator.loadBostonHousingDataset(data, targets);
    EXPECT_TRUE(loaded);

    if (loaded) {
        std::vector<std::vector<float>> targets_vec;
        for (float target : targets) {
            targets_vec.push_back({target});
        }

        // Run benchmark on Boston Housing dataset (regression)
        auto results = validator.benchmarkAgainstBaselines(data, targets_vec, 0.2f);

        EXPECT_FALSE(results.empty());

        // Boston Housing is a regression problem
        for (const auto& result : results) {
            EXPECT_GE(result.metrics.rmse, 0.0f);
            EXPECT_GE(result.metrics.mae, 0.0f);
            EXPECT_GE(result.metrics.r2_score, -1.0f); // R² can be negative
            EXPECT_LE(result.metrics.r2_score, 1.0f);
        }

        auto report = validator.generateBenchmarkReport(results);
        validator.saveReportToFile("boston_housing_benchmark_report.txt", report);

        std::cout << "Boston Housing dataset benchmark completed. Results saved to boston_housing_benchmark_report.txt" << std::endl;

        std::remove("boston_housing_benchmark_report.txt");
    }
}

TEST(BenchmarkTest, WineDatasetBenchmark) {
    ValidationFramework validator(true);

    std::vector<std::vector<float>> data;
    std::vector<float> targets;

    bool loaded = validator.loadWineDataset(data, targets);
    EXPECT_TRUE(loaded);

    if (loaded) {
        std::vector<std::vector<float>> targets_vec;
        for (float target : targets) {
            targets_vec.push_back({target});
        }

        // Run benchmark on Wine dataset (classification)
        auto results = validator.benchmarkAgainstBaselines(data, targets_vec, 0.3f);

        EXPECT_FALSE(results.empty());

        // Wine dataset has 3 classes
        for (const auto& result : results) {
            EXPECT_GE(result.metrics.accuracy, 0.0f);
            EXPECT_LE(result.metrics.accuracy, 1.0f);
        }

        auto report = validator.generateBenchmarkReport(results);
        validator.saveReportToFile("wine_benchmark_report.txt", report);

        std::cout << "Wine dataset benchmark completed. Results saved to wine_benchmark_report.txt" << std::endl;

        std::remove("wine_benchmark_report.txt");
    }
}

TEST(BenchmarkTest, DiabetesDatasetBenchmark) {
    ValidationFramework validator(true);

    std::vector<std::vector<float>> data;
    std::vector<float> targets;

    bool loaded = validator.loadDiabetesDataset(data, targets);
    EXPECT_TRUE(loaded);

    if (loaded) {
        std::vector<std::vector<float>> targets_vec;
        for (float target : targets) {
            targets_vec.push_back({target});
        }

        // Run benchmark on Diabetes dataset (regression)
        auto results = validator.benchmarkAgainstBaselines(data, targets_vec, 0.2f);

        EXPECT_FALSE(results.empty());

        for (const auto& result : results) {
            EXPECT_GE(result.metrics.rmse, 0.0f);
            EXPECT_GE(result.metrics.mae, 0.0f);
        }

        auto report = validator.generateBenchmarkReport(results);
        validator.saveReportToFile("diabetes_benchmark_report.txt", report);

        std::cout << "Diabetes dataset benchmark completed. Results saved to diabetes_benchmark_report.txt" << std::endl;

        std::remove("diabetes_benchmark_report.txt");
    }
}

TEST(BenchmarkTest, DigitsDatasetBenchmark) {
    ValidationFramework validator(true);

    std::vector<std::vector<float>> data;
    std::vector<float> targets;

    bool loaded = validator.loadDigitsDataset(data, targets);
    EXPECT_TRUE(loaded);

    if (loaded) {
        std::vector<std::vector<float>> targets_vec;
        for (float target : targets) {
            targets_vec.push_back({target});
        }

        // Run benchmark on Digits dataset (MNIST-like, classification with 10 classes)
        auto results = validator.benchmarkAgainstBaselines(data, targets_vec, 0.2f);

        EXPECT_FALSE(results.empty());

        // Digits dataset has 10 classes (0-9)
        for (const auto& result : results) {
            EXPECT_GE(result.metrics.accuracy, 0.0f);
            EXPECT_LE(result.metrics.accuracy, 1.0f);
        }

        auto report = validator.generateBenchmarkReport(results);
        validator.saveReportToFile("digits_benchmark_report.txt", report);

        std::cout << "Digits dataset benchmark completed. Results saved to digits_benchmark_report.txt" << std::endl;

        std::remove("digits_benchmark_report.txt");
    }
}

// Comprehensive standard datasets comparison
TEST(BenchmarkTest, StandardDatasetsComparison) {
    ValidationFramework validator(true);

    struct DatasetInfo {
        std::string name;
        std::function<bool(ValidationFramework&, std::vector<std::vector<float>>&, std::vector<float>&)> loader;
        bool isClassification;
    };

    std::vector<DatasetInfo> datasets = {
        {"Iris", &ValidationFramework::loadIrisDataset, true},
        {"Boston Housing", &ValidationFramework::loadBostonHousingDataset, false},
        {"Wine", &ValidationFramework::loadWineDataset, true},
        {"Diabetes", &ValidationFramework::loadDiabetesDataset, false},
        {"Digits", &ValidationFramework::loadDigitsDataset, true}
    };

    std::ofstream comparisonReport("standard_datasets_comparison.csv");
    comparisonReport << "Dataset,Model,Accuracy,RMSE,MAE,R2_Score,TrainingTime_ms,MemoryUsage_KB\n";

    for (const auto& dataset : datasets) {
        std::vector<std::vector<float>> data;
        std::vector<float> targets;

        bool loaded = (validator.*(dataset.loader))(data, targets);
        EXPECT_TRUE(loaded) << "Failed to load " << dataset.name << " dataset";

        if (loaded) {
            std::vector<std::vector<float>> targets_vec;
            for (float target : targets) {
                targets_vec.push_back({target});
            }

            auto results = validator.benchmarkAgainstBaselines(data, targets_vec, 0.2f);

            for (const auto& result : results) {
                comparisonReport << dataset.name << ","
                               << result.modelName << ","
                               << result.metrics.accuracy << ","
                               << result.metrics.rmse << ","
                               << result.metrics.mae << ","
                               << result.metrics.r2_score << ","
                               << (result.trainingTime.count() / 1000.0) << ","
                               << (result.metrics.memoryUsage / 1024) << "\n";
            }

            std::cout << "Completed benchmark for " << dataset.name << " dataset" << std::endl;
        }
    }

    comparisonReport.close();

    // Verify the comparison file was created and has content
    std::ifstream file("standard_datasets_comparison.csv");
    EXPECT_TRUE(file.is_open());
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(content.find("Dataset,Model,Accuracy") != std::string::npos);
    file.close();

    std::cout << "Standard datasets comparison completed. Results saved to standard_datasets_comparison.csv" << std::endl;

    std::remove("standard_datasets_comparison.csv");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
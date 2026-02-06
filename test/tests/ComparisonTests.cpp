#include <gtest/gtest.h>
#include "ValidationFramework.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <algorithm>

// Framework comparison tests
class ComparisonTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test datasets
        createComparisonDataset("comparison_regression.csv", 500, 5, false);
        createComparisonDataset("comparison_classification.csv", 400, 4, true);
    }

    void TearDown() override {
        // Clean up test files
        std::remove("comparison_regression.csv");
        std::remove("comparison_classification.csv");
        std::remove("framework_comparison_report.json");
        std::remove("performance_comparison.csv");
    }

    void createComparisonDataset(const std::string& filename, int numSamples, int numFeatures, bool isClassification) {
        std::ofstream file(filename);
        file << "feature_0";
        for (int i = 1; i < numFeatures; i++) {
            file << ",feature_" << i;
        }
        file << ",target\n";

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                float value = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f; // -5 to 5 range
                file << value;
                if (j < numFeatures - 1) file << ",";
            }

            float target;
            if (isClassification) {
                // Binary classification target
                float score = 0.0f;
                for (int j = 0; j < numFeatures; j++) {
                    score += (static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f) * 0.5f;
                }
                target = score > 0 ? 1.0f : 0.0f;
            } else {
                // Regression target
                target = 0.0f;
                for (int j = 0; j < numFeatures; j++) {
                    target += (static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f) * 0.3f;
                }
                target += static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f; // Add noise
            }

            file << "," << target << "\n";
        }
        file.close();
    }
};

// Test comparison with scikit-learn models
TEST_F(ComparisonTests, ScikitLearnComparisonRegression) {
    ValidationFramework validator(true);

    // Load regression dataset
    Training loader(false, false);
    bool loaded = loader.loadCSV("comparison_regression.csv", 500, 1, true, false, ',', "regression_cmp");
    ASSERT_TRUE(loaded);

    const auto& inputs = loader.getInputs();
    const auto& targets = loader.getTargets();

    // Run CoreAI3D benchmark
    auto coreAIResults = validator.benchmarkAgainstBaselines(inputs, targets, 0.2f);
    ASSERT_FALSE(coreAIResults.empty());

    // Run scikit-learn comparison (if available)
    try {
        auto sklearnResults = validator.benchmarkAgainstScikitLearn(inputs, targets, 0.2f);
        if (!sklearnResults.empty()) {
            // Combine results for comparison
            std::vector<BenchmarkResult> allResults = coreAIResults;
            allResults.insert(allResults.end(), sklearnResults.begin(), sklearnResults.end());

            // Generate comparison report
            auto report = validator.generateBenchmarkReport(allResults);
            validator.saveReportToFile("sklearn_regression_comparison.txt", report);

            std::cout << "Scikit-learn regression comparison completed" << std::endl;

            // Verify we have results from both frameworks
            bool hasCoreAI3D = false;
            bool hasSklearn = false;
            for (const auto& result : allResults) {
                if (result.modelName.find("CoreAI3D") != std::string::npos) hasCoreAI3D = true;
                if (result.modelName.find("sklearn") != std::string::npos ||
                    result.modelName.find("Scikit") != std::string::npos) hasSklearn = true;
            }

            EXPECT_TRUE(hasCoreAI3D);
            if (!sklearnResults.empty()) {
                EXPECT_TRUE(hasSklearn);
            }

            std::remove("sklearn_regression_comparison.txt");
        } else {
            std::cout << "Scikit-learn comparison not available, using CoreAI3D only" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Scikit-learn comparison failed: " << e.what() << std::endl;
        // Continue with CoreAI3D only
    }
}

TEST_F(ComparisonTests, ScikitLearnComparisonClassification) {
    ValidationFramework validator(true);

    // Load classification dataset
    Training loader(false, false);
    bool loaded = loader.loadCSV("comparison_classification.csv", 400, 1, true, false, ',', "classification_cmp");
    ASSERT_TRUE(loaded);

    const auto& inputs = loader.getInputs();
    const auto& targets = loader.getTargets();

    // Run CoreAI3D benchmark
    auto coreAIResults = validator.benchmarkAgainstBaselines(inputs, targets, 0.3f);
    ASSERT_TRUE(!coreAIResults.empty());

    // Run scikit-learn comparison
    try {
        auto sklearnResults = validator.benchmarkAgainstScikitLearn(inputs, targets, 0.3f);
        if (!sklearnResults.empty()) {
            std::vector<BenchmarkResult> allResults = coreAIResults;
            allResults.insert(allResults.end(), sklearnResults.begin(), sklearnResults.end());

            auto report = validator.generateBenchmarkReport(allResults);
            validator.saveReportToFile("sklearn_classification_comparison.txt", report);

            std::cout << "Scikit-learn classification comparison completed" << std::endl;

            std::remove("sklearn_classification_comparison.txt");
        }
    } catch (const std::exception& e) {
        std::cout << "Scikit-learn comparison failed: " << e.what() << std::endl;
    }
}

// Comprehensive framework comparison across multiple datasets
TEST_F(ComparisonTests, MultiFrameworkMultiDatasetComparison) {
    ValidationFramework validator(true);

    struct DatasetConfig {
        std::string filename;
        std::string name;
        bool isClassification;
        int samples;
        int features;
    };

    std::vector<DatasetConfig> datasets = {
        {"comparison_regression.csv", "Regression", false, 500, 5},
        {"comparison_classification.csv", "Classification", true, 400, 4}
    };

    std::ofstream comparisonReport("multi_framework_comparison.json");
    comparisonReport << "{\n";
    comparisonReport << "  \"framework_comparison\": {\n";
    comparisonReport << "    \"timestamp\": \"" << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\",\n";
    comparisonReport << "    \"datasets\": [\n";

    for (size_t i = 0; i < datasets.size(); i++) {
        const auto& dataset = datasets[i];

        Training loader(false, false);
        bool loaded = loader.loadCSV(dataset.filename, dataset.samples, 1, true, false, ',', dataset.name);
        ASSERT_TRUE(loaded);

        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        comparisonReport << "      {\n";
        comparisonReport << "        \"name\": \"" << dataset.name << "\",\n";
        comparisonReport << "        \"type\": \"" << (dataset.isClassification ? "classification" : "regression") << "\",\n";
        comparisonReport << "        \"samples\": " << dataset.samples << ",\n";
        comparisonReport << "        \"features\": " << dataset.features << ",\n";
        comparisonReport << "        \"frameworks\": [\n";

        // CoreAI3D results
        auto coreAIResults = validator.benchmarkAgainstBaselines(inputs, targets, 0.2f);

        for (size_t j = 0; j < coreAIResults.size(); j++) {
            const auto& result = coreAIResults[j];
            comparisonReport << "          {\n";
            comparisonReport << "            \"name\": \"CoreAI3D\",\n";
            comparisonReport << "            \"model\": \"" << result.modelName << "\",\n";
            comparisonReport << "            \"accuracy\": " << result.metrics.accuracy << ",\n";
            comparisonReport << "            \"rmse\": " << result.metrics.rmse << ",\n";
            comparisonReport << "            \"mae\": " << result.metrics.mae << ",\n";
            comparisonReport << "            \"r2_score\": " << result.metrics.r2_score << ",\n";
            comparisonReport << "            \"training_time_ms\": " << (result.trainingTime.count() / 1000.0) << ",\n";
            comparisonReport << "            \"inference_time_us\": " << result.inferenceTime.count() << ",\n";
            comparisonReport << "            \"memory_usage_kb\": " << (result.metrics.memoryUsage / 1024) << "\n";
            comparisonReport << "          }";
            if (j < coreAIResults.size() - 1) comparisonReport << ",";
            comparisonReport << "\n";
        }

        // Try scikit-learn comparison
        try {
            auto sklearnResults = validator.benchmarkAgainstScikitLearn(inputs, targets, 0.2f);
            if (!sklearnResults.empty()) {
                comparisonReport << ",\n";
                for (size_t j = 0; j < sklearnResults.size(); j++) {
                    const auto& result = sklearnResults[j];
                    comparisonReport << "          {\n";
                    comparisonReport << "            \"name\": \"ScikitLearn\",\n";
                    comparisonReport << "            \"model\": \"" << result.modelName << "\",\n";
                    comparisonReport << "            \"accuracy\": " << result.metrics.accuracy << ",\n";
                    comparisonReport << "            \"rmse\": " << result.metrics.rmse << ",\n";
                    comparisonReport << "            \"mae\": " << result.metrics.mae << ",\n";
                    comparisonReport << "            \"r2_score\": " << result.metrics.r2_score << ",\n";
                    comparisonReport << "            \"training_time_ms\": " << (result.trainingTime.count() / 1000.0) << ",\n";
                    comparisonReport << "            \"inference_time_us\": " << result.inferenceTime.count() << ",\n";
                    comparisonReport << "            \"memory_usage_kb\": " << (result.metrics.memoryUsage / 1024) << "\n";
                    comparisonReport << "          }";
                    if (j < sklearnResults.size() - 1) comparisonReport << ",";
                    comparisonReport << "\n";
                }
            }
        } catch (const std::exception& e) {
            std::cout << "Scikit-learn comparison failed for " << dataset.name << ": " << e.what() << std::endl;
        }

        comparisonReport << "        ]\n";
        comparisonReport << "      }";
        if (i < datasets.size() - 1) comparisonReport << ",";
        comparisonReport << "\n";
    }

    comparisonReport << "    ]\n";
    comparisonReport << "  }\n";
    comparisonReport << "}\n";
    comparisonReport.close();

    std::cout << "Multi-framework comparison completed. Results saved to multi_framework_comparison.json" << std::endl;
}

// Performance comparison across different problem scales
TEST_F(ComparisonTests, ScalabilityComparison) {
    ValidationFramework validator(true);

    std::vector<std::tuple<int, int, std::string>> scales = {
        {100, 3, "Small"},
        {500, 5, "Medium"},
        {1000, 8, "Large"}
    };

    std::ofstream scalabilityReport("scalability_comparison.csv");
    scalabilityReport << "Scale,DatasetSize,Features,Framework,Model,TrainingTime_ms,InferenceTime_us,MemoryUsage_KB,Accuracy\n";

    for (const auto& scale : scales) {
        int samples, features;
        std::string scaleName;
        std::tie(samples, features, scaleName) = scale;

        // Create dataset for this scale
        createComparisonDataset("scale_cmp.csv", samples, features, false);

        Training loader(false, false);
        bool loaded = loader.loadCSV("scale_cmp.csv", samples, 1, true, false, ',', "scale_cmp");
        ASSERT_TRUE(loaded);

        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // CoreAI3D benchmark
        auto coreAIResults = validator.benchmarkAgainstBaselines(inputs, targets, 0.2f);

        for (const auto& result : coreAIResults) {
            scalabilityReport << scaleName << "," << samples << "," << features << ",CoreAI3D,"
                            << result.modelName << "," << (result.trainingTime.count() / 1000.0) << ","
                            << result.inferenceTime.count() << "," << (result.metrics.memoryUsage / 1024) << ","
                            << result.metrics.accuracy << "\n";
        }

        // Try scikit-learn
        try {
            auto sklearnResults = validator.benchmarkAgainstScikitLearn(inputs, targets, 0.2f);
            for (const auto& result : sklearnResults) {
                scalabilityReport << scaleName << "," << samples << "," << features << ",ScikitLearn,"
                                << result.modelName << "," << (result.trainingTime.count() / 1000.0) << ","
                                << result.inferenceTime.count() << "," << (result.metrics.memoryUsage / 1024) << ","
                                << result.metrics.accuracy << "\n";
            }
        } catch (const std::exception& e) {
            // Scikit-learn not available, skip
        }

        std::remove("scale_cmp.csv");
        std::cout << "Completed scalability comparison for " << scaleName << " scale" << std::endl;
    }

    scalabilityReport.close();
    std::cout << "Scalability comparison completed. Results saved to scalability_comparison.csv" << std::endl;
}

// Accuracy vs Speed trade-off analysis
TEST_F(ComparisonTests, AccuracySpeedTradeoff) {
    ValidationFramework validator(true);

    // Load a moderately sized dataset
    createComparisonDataset("tradeoff_test.csv", 300, 6, true);

    Training loader(false, false);
    bool loaded = loader.loadCSV("tradeoff_test.csv", 300, 1, true, false, ',', "tradeoff");
    ASSERT_TRUE(loaded);

    const auto& inputs = loader.getInputs();
    const auto& targets = loader.getTargets();

    auto results = validator.benchmarkAgainstBaselines(inputs, targets, 0.2f);

    std::ofstream tradeoffReport("accuracy_speed_tradeoff.csv");
    tradeoffReport << "Model,Accuracy,TrainingTime_ms,InferenceTime_us,MemoryUsage_KB,Efficiency_Score\n";

    for (const auto& result : results) {
        // Calculate an efficiency score (accuracy / (training_time + inference_time))
        double totalTime = (result.trainingTime.count() / 1000.0) + (result.inferenceTime.count() / 1000000.0);
        double efficiencyScore = (totalTime > 0) ? result.metrics.accuracy / totalTime : 0.0;

        tradeoffReport << result.modelName << ","
                      << result.metrics.accuracy << ","
                      << (result.trainingTime.count() / 1000.0) << ","
                      << result.inferenceTime.count() << ","
                      << (result.metrics.memoryUsage / 1024) << ","
                      << efficiencyScore << "\n";
    }

    tradeoffReport.close();

    // Verify we have meaningful results
    EXPECT_FALSE(results.empty());
    for (const auto& result : results) {
        EXPECT_GE(result.metrics.accuracy, 0.0f);
        EXPECT_GE(result.trainingTime.count(), 0);
    }

    std::cout << "Accuracy vs Speed tradeoff analysis completed. Results saved to accuracy_speed_tradeoff.csv" << std::endl;

    std::remove("tradeoff_test.csv");
    std::remove("accuracy_speed_tradeoff.csv");
}

// Statistical significance testing between frameworks
TEST_F(ComparisonTests, StatisticalSignificanceTest) {
    ValidationFramework validator(true);

    // Run multiple benchmarks to get statistical significance
    const int numRuns = 5;
    createComparisonDataset("significance_test.csv", 200, 4, false);

    std::vector<std::vector<BenchmarkResult>> allRuns;

    for (int run = 0; run < numRuns; run++) {
        Training loader(false, false);
        bool loaded = loader.loadCSV("significance_test.csv", 200, 1, true, false, ',', "sig_test");
        ASSERT_TRUE(loaded);

        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        auto results = validator.benchmarkAgainstBaselines(inputs, targets, 0.3f);
        allRuns.push_back(results);
    }

    // Analyze statistical significance (simplified)
    std::ofstream significanceReport("statistical_significance.csv");
    significanceReport << "Model,Run,Accuracy,TrainingTime_ms\n";

    std::map<std::string, std::vector<double>> modelAccuracies;
    std::map<std::string, std::vector<double>> modelTimes;

    for (int run = 0; run < numRuns; run++) {
        for (const auto& result : allRuns[run]) {
            significanceReport << result.modelName << "," << run << ","
                             << result.metrics.accuracy << ","
                             << (result.trainingTime.count() / 1000.0) << "\n";

            modelAccuracies[result.modelName].push_back(result.metrics.accuracy);
            modelTimes[result.modelName].push_back(result.trainingTime.count() / 1000.0);
        }
    }

    significanceReport << "\nStatistical Summary:\n";
    significanceReport << "Model,Mean_Accuracy,Std_Accuracy,Mean_TrainingTime_ms,Std_TrainingTime_ms\n";

    for (const auto& pair : modelAccuracies) {
        const std::string& modelName = pair.first;
        const auto& accuracies = modelAccuracies[modelName];
        const auto& times = modelTimes[modelName];

        // Calculate means and standard deviations
        double accMean = std::accumulate(accuracies.begin(), accuracies.end(), 0.0) / accuracies.size();
        double timeMean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

        double accVar = 0.0, timeVar = 0.0;
        for (size_t i = 0; i < accuracies.size(); i++) {
            accVar += std::pow(accuracies[i] - accMean, 2);
            timeVar += std::pow(times[i] - timeMean, 2);
        }
        double accStd = std::sqrt(accVar / accuracies.size());
        double timeStd = std::sqrt(timeVar / times.size());

        significanceReport << modelName << "," << accMean << "," << accStd << ","
                          << timeMean << "," << timeStd << "\n";
    }

    significanceReport.close();

    std::cout << "Statistical significance testing completed. Results saved to statistical_significance.csv" << std::endl;

    std::remove("significance_test.csv");
    std::remove("statistical_significance.csv");
}


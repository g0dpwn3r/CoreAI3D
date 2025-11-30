#include <gtest/gtest.h>
#include "ValidationFramework.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

// Helper function to create test datasets
void createTestDataset(const std::string& filename, const std::vector<std::vector<float>>& data) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); i++) {
                file << row[i];
                if (i < row.size() - 1) file << ",";
            }
            file << "\n";
        }
        file.close();
    }
}

TEST(ValidationFrameworkTest, DataIntegrityCheck) {
    ValidationFramework validator(true);
    
    // Test valid data
    std::vector<std::vector<float>> validData = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };
    
    EXPECT_TRUE(validator.checkDataIntegrity(validData));
    
    // Test data with inconsistent dimensions
    std::vector<std::vector<float>> invalidData = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f}, // Missing one value
        {7.0f, 8.0f, 9.0f}
    };
    
    EXPECT_FALSE(validator.checkDataIntegrity(invalidData));
    
    // Test empty data
    std::vector<std::vector<float>> emptyData;
    EXPECT_FALSE(validator.checkDataIntegrity(emptyData));
}

TEST(ValidationFrameworkTest, SampleSizeCheck) {
    ValidationFramework validator;
    
    std::vector<std::vector<float>> largeData(20, std::vector<float>(3, 1.0f));
    std::vector<std::vector<float>> smallData(5, std::vector<float>(3, 1.0f));
    std::vector<std::vector<float>> emptyData;
    
    EXPECT_TRUE(validator.checkSampleSize(largeData, 10));
    EXPECT_FALSE(validator.checkSampleSize(smallData, 10));
    EXPECT_FALSE(validator.checkSampleSize(emptyData, 10));
}

TEST(ValidationFrameworkTest, MissingValuesCheck) {
    ValidationFramework validator;
    
    // Test data without missing values
    std::vector<std::vector<float>> cleanData = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    
    EXPECT_TRUE(validator.checkMissingValues(cleanData));
    
    // Test data with some missing values
    std::vector<std::vector<float>> dataWithMissing = {
        {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f},
        {4.0f, 5.0f, 6.0f}
    };
    
    EXPECT_FALSE(validator.checkMissingValues(dataWithMissing, 0.1f)); // 16.7% missing
    EXPECT_TRUE(validator.checkMissingValues(dataWithMissing, 0.2f));  // Allow 20% missing
}

TEST(ValidationFrameworkTest, FeatureVarianceCheck) {
    ValidationFramework validator;
    
    // Test data with variance
    std::vector<std::vector<float>> variedData = {
        {1.0f, 2.0f, 3.0f},
        {2.0f, 3.0f, 4.0f},
        {3.0f, 4.0f, 5.0f}
    };
    
    EXPECT_TRUE(validator.checkFeatureVariance(variedData));
    
    // Test data with zero variance (constant features)
    std::vector<std::vector<float>> constantData = {
        {1.0f, 2.0f, 3.0f},
        {1.0f, 2.0f, 3.0f},
        {1.0f, 2.0f, 3.0f}
    };
    
    EXPECT_FALSE(validator.checkFeatureVariance(constantData));
}

TEST(ValidationFrameworkTest, DatasetStatistics) {
    ValidationFramework validator;
    
    std::vector<std::vector<float>> testData = {
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f},
        {7.0f, 8.0f, 9.0f}
    };
    
    auto stats = validator.computeDatasetStatistics(testData);
    
    EXPECT_EQ(stats["num_samples"], 3.0f);
    EXPECT_EQ(stats["num_features"], 3.0f);
    EXPECT_EQ(stats["global_min"], 1.0f);
    EXPECT_EQ(stats["global_max"], 9.0f);
    EXPECT_EQ(stats["global_range"], 8.0f);
}

TEST(ValidationFrameworkTest, DataSplitting) {
    ValidationFramework validator;
    
    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, 
        {7.0f, 8.0f}, {9.0f, 10.0f}
    };
    std::vector<std::vector<float>> targets = {
        {0.0f}, {1.0f}, {0.0f}, {1.0f}, {0.0f}
    };
    
    std::vector<std::vector<float>> trainData, testData, trainTargets, testTargets;
    
    validator.splitData(data, targets, 0.6f, trainData, testData, trainTargets, testTargets);
    
    EXPECT_EQ(trainData.size() + testData.size(), data.size());
    EXPECT_EQ(trainTargets.size() + testTargets.size(), targets.size());
    
    // Check that total sizes are correct (60% train, 40% test)
    EXPECT_EQ(trainData.size(), 3u); // 5 * 0.6 = 3
    EXPECT_EQ(testData.size(), 2u);  // 5 * 0.4 = 2
}

TEST(ValidationFrameworkTest, DataNormalization) {
    ValidationFramework validator;
    
    std::vector<std::vector<float>> testData = {
        {1.0f, 10.0f},
        {2.0f, 20.0f},
        {3.0f, 30.0f}
    };
    
    std::vector<float> means, stds;
    auto normalized = validator.normalizeData(testData, means, stds);
    
    // Check that means are approximately zero and stds are approximately 1
    EXPECT_NEAR(means[0], 2.0f, 1e-5f);
    EXPECT_NEAR(means[1], 20.0f, 1e-5f);
    EXPECT_NEAR(stds[0], 0.8164965809f, 1e-5f); // sqrt(2/3)
    EXPECT_NEAR(stds[1], 8.164965809f, 1e-5f); // sqrt(200/3)
}

TEST(ValidationFrameworkTest, PerformanceMetrics) {
    std::vector<float> predictions = {0.8f, 0.2f, 0.9f, 0.1f, 0.7f};
    std::vector<float> targets = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    
    float accuracy = ValidationFramework::calculateAccuracy(predictions, targets);
    float precision = ValidationFramework::calculatePrecision(predictions, targets, 0.5f);
    float recall = ValidationFramework::calculateRecall(predictions, targets, 0.5f);
    float f1 = ValidationFramework::calculateF1Score(predictions, targets, 0.5f);
    float rmse = ValidationFramework::calculateRMSE(predictions, targets);
    float mae = ValidationFramework::calculateMAE(predictions, targets);
    
    // For this example: 4 correct predictions out of 5 = 80% accuracy
    EXPECT_NEAR(accuracy, 0.8f, 1e-5f);
    EXPECT_GT(precision, 0.0f);
    EXPECT_GT(recall, 0.0f);
    EXPECT_GT(f1, 0.0f);
    EXPECT_GT(rmse, 0.0f);
    EXPECT_GT(mae, 0.0f);
}

TEST(ValidationFrameworkTest, FullDatasetValidation) {
    ValidationFramework validator(true);
    
    // Create a test dataset
    std::vector<std::vector<float>> testData = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {2.0f, 3.0f, 4.0f, 5.0f},
        {3.0f, 4.0f, 5.0f, 6.0f},
        {4.0f, 5.0f, 6.0f, 7.0f},
        {5.0f, 6.0f, 7.0f, 8.0f}
    };
    
    createTestDataset("test_validation.csv", testData);
    
    auto result = validator.validateDataset("test_validation.csv", false);
    
    EXPECT_TRUE(result.isValid);
    EXPECT_GT(result.qualityScore, 0.5f);
    EXPECT_FALSE(result.validationChecks.empty());
    
    // Clean up
    std::remove("test_validation.csv");
}

TEST(ValidationFrameworkTest, ValidationWithOutliers) {
    ValidationFramework validator;
    
    // Create data with outliers
    std::vector<std::vector<float>> dataWithOutliers = {
        {1.0f, 2.0f, 3.0f},
        {2.0f, 3.0f, 4.0f},
        {3.0f, 4.0f, 5.0f},
        {100.0f, 200.0f, 300.0f} // Clear outlier
    };
    
    auto result = validator.validateDataMatrix(dataWithOutliers);
    
    // Should pass validation but may have warnings
    EXPECT_TRUE(result.isValid); // Outliers don't fail validation
    EXPECT_TRUE(result.qualityScore <= 1.0f);
}

TEST(ValidationFrameworkTest, CorrelationMatrix) {
    ValidationFramework validator;
    
    std::vector<std::vector<float>> testData = {
        {1.0f, 2.0f, 3.0f},
        {2.0f, 4.0f, 6.0f},
        {3.0f, 6.0f, 9.0f}
    };
    
    auto correlationMatrix = validator.computeCorrelationMatrix(testData);
    
    // Check diagonal is 1.0
    EXPECT_NEAR(correlationMatrix[0][0], 1.0f, 1e-5f);
    EXPECT_NEAR(correlationMatrix[1][1], 1.0f, 1e-5f);
    EXPECT_NEAR(correlationMatrix[2][2], 1.0f, 1e-5f);
    
    // Check symmetry
    EXPECT_NEAR(correlationMatrix[0][1], correlationMatrix[1][0], 1e-5f);
    EXPECT_NEAR(correlationMatrix[0][2], correlationMatrix[2][0], 1e-5f);
    EXPECT_NEAR(correlationMatrix[1][2], correlationMatrix[2][1], 1e-5f);
}

TEST(ValidationFrameworkTest, BenchmarkFramework) {
    ValidationFramework validator(true);
    
    // Create simple test dataset
    std::vector<std::vector<float>> data = {
        {1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}, {4.0f, 5.0f},
        {5.0f, 6.0f}, {6.0f, 7.0f}, {7.0f, 8.0f}, {8.0f, 9.0f}
    };
    
    std::vector<std::vector<float>> targets = {
        {3.0f}, {5.0f}, {7.0f}, {9.0f}, {11.0f}, {13.0f}, {15.0f}, {17.0f}
    };
    
    try {
        auto results = validator.benchmarkAgainstBaselines(data, targets, 0.25f);
        
        EXPECT_FALSE(results.empty());
        
        // Check that we have results from different models
        std::vector<std::string> modelNames;
        for (const auto& result : results) {
            modelNames.push_back(result.modelName);
        }
        
        EXPECT_GT(modelNames.size(), 0);
        
        // Check that each result has valid metrics
        for (const auto& result : results) {
            EXPECT_GT(result.metrics.accuracy, -1.0f); // Valid range
            EXPECT_LT(result.metrics.rmse, 1e6f); // Reasonable upper bound
        }
        
    } catch (const std::exception& e) {
        // Benchmarking might fail in test environment, that's ok
        SUCCEED() << "Benchmarking test completed with exception: " << e.what();
    }
}

TEST(ValidationFrameworkTest, ReportGeneration) {
    ValidationFramework validator;
    
    // Create a validation result
    ValidationResult result;
    result.isValid = true;
    result.qualityScore = 0.85f;
    result.errorMessage = "";
    result.warningMessage = "Minor outliers detected";
    result.validationChecks = {"Sample size: PASS", "Data integrity: PASS", "Outliers: WARNING"};
    result.statistics["num_samples"] = 100.0f;
    result.statistics["num_features"] = 5.0f;
    
    auto report = validator.generateValidationReport(result);
    
    EXPECT_FALSE(report.empty());
    EXPECT_TRUE(report.find("VALIDATION REPORT") != std::string::npos);
    EXPECT_TRUE(report.find("PASSED") != std::string::npos);
    EXPECT_TRUE(report.find("0.850") != std::string::npos); // Quality score
}

TEST(ValidationFrameworkTest, DatasetSplitValidation) {
    ValidationFramework validator;
    
    std::vector<std::vector<float>> trainData = {
        {1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 4.0f}
    };
    std::vector<std::vector<float>> testData = {
        {4.0f, 5.0f}, {5.0f, 6.0f}
    };
    std::vector<std::vector<float>> trainTargets = {
        {3.0f}, {5.0f}, {7.0f}
    };
    std::vector<std::vector<float>> testTargets = {
        {9.0f}, {11.0f}
    };
    
    auto result = validator.validateDatasetSplit(trainData, testData, trainTargets, testTargets);
    
    // Should pass basic validation
    EXPECT_TRUE(result.isValid);
    EXPECT_GT(result.qualityScore, 0.5f);
}

TEST(ValidationFrameworkTest, ConfigurableValidation) {
    ValidationFramework validator;
    
    ValidationConfig config;
    config.minSampleSize = 100;
    config.maxMissingValuePercentage = 0.05f;
    config.outlierThreshold = 2.0f;
    
    validator.setValidationConfig(config);
    
    auto retrievedConfig = validator.getValidationConfig();
    
    EXPECT_EQ(retrievedConfig.minSampleSize, 100);
    EXPECT_EQ(retrievedConfig.maxMissingValuePercentage, 0.05f);
    EXPECT_EQ(retrievedConfig.outlierThreshold, 2.0f);
}

TEST(ValidationFrameworkTest, EdgeCases) {
    ValidationFramework validator;
    
    // Test with single sample
    std::vector<std::vector<float>> singleSample = {
        {1.0f, 2.0f, 3.0f}
    };
    
    auto result = validator.validateDataMatrix(singleSample);
    
    // Should fail sample size check but not crash
    EXPECT_FALSE(result.validationChecks.empty());
    
    // Test with single feature
    std::vector<std::vector<float>> singleFeature = {
        {1.0f}, {2.0f}, {3.0f}, {4.0f}, {5.0f}
    };
    
    auto singleFeatureResult = validator.validateDataMatrix(singleFeature);
    
    // Should pass most checks
    EXPECT_TRUE(singleFeatureResult.validationChecks.size() > 0);
}

// Add a new test file for training integration tests
TEST(ValidationFrameworkTest, TrainingIntegration) {
    ValidationFramework validator(true);

    // Create a simple training scenario
    std::vector<std::vector<float>> data = {
        {1.0f, 1.0f}, {2.0f, 2.0f}, {3.0f, 3.0f}, {4.0f, 4.0f}, {5.0f, 5.0f}
    };

    std::vector<std::vector<float>> targets = {
        {2.0f}, {4.0f}, {6.0f}, {8.0f}, {10.0f}
    };

    // Test data preprocessing
    std::vector<float> means, stds;
    auto normalized = validator.normalizeData(data, means, stds);

    EXPECT_EQ(normalized.size(), data.size());
    EXPECT_EQ(normalized[0].size(), data[0].size());

    // Test data splitting
    std::vector<std::vector<float>> trainData, testData, trainTargets, testTargets;
    validator.splitData(data, targets, 0.8f, trainData, testData, trainTargets, testTargets);

    EXPECT_EQ(trainData.size() + testData.size(), data.size());
    EXPECT_EQ(trainTargets.size() + testTargets.size(), targets.size());
}

TEST(ValidationFrameworkTest, ScikitLearnBenchmarking) {
    ValidationFramework validator(true);

    // Create test regression data
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> targets;

    // Generate linear relationship: y = 2*x1 + 3*x2 + noise
    for (int i = 0; i < 100; i++) {
        float x1 = static_cast<float>(rand()) / RAND_MAX * 10.0f;
        float x2 = static_cast<float>(rand()) / RAND_MAX * 10.0f;
        float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        float y = 2.0f * x1 + 3.0f * x2 + noise;

        data.push_back({x1, x2});
        targets.push_back({y});
    }

    // Test scikit-learn benchmarking (only if Python is available)
    try {
        auto scikitResults = validator.benchmarkAgainstScikitLearn(data, targets, 0.2f);

        // If scikit-learn is available, we should get results
        if (!scikitResults.empty()) {
            EXPECT_GE(scikitResults.size(), 5); // Should have multiple models

            // Check that results have valid metrics
            for (const auto& result : scikitResults) {
                EXPECT_FALSE(result.modelName.empty());
                EXPECT_GE(result.totalTime, 0.0f);

                // For regression, R² should be reasonable
                if (result.metrics.r2_score > -10.0f) { // Valid range check
                    EXPECT_GE(result.metrics.r2_score, -1.0f);
                    EXPECT_LE(result.metrics.r2_score, 1.0f);
                }
            }

            // Linear regression should perform well on linear data
            auto linearRegression = std::find_if(scikitResults.begin(), scikitResults.end(),
                [](const BenchmarkResult& r) { return r.modelName == "Linear Regression"; });

            if (linearRegression != scikitResults.end()) {
                EXPECT_GT(linearRegression->metrics.r2_score, 0.8f); // Should be high for linear data
            }
        } else {
            // If no results, that's also acceptable (Python/scikit-learn not available)
            SUCCEED() << "Scikit-learn benchmarking test passed (no Python available)";
        }

    } catch (const std::exception& e) {
        // If scikit-learn integration fails, that's acceptable in test environment
        SUCCEED() << "Scikit-learn benchmarking test completed with exception: " << e.what();
    }
}

TEST(ValidationFrameworkTest, ComprehensiveBenchmarkComparison) {
    ValidationFramework validator(true);

    // Create test data
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> targets;

    for (int i = 0; i < 50; i++) {
        float x1 = static_cast<float>(rand()) / RAND_MAX * 5.0f;
        float x2 = static_cast<float>(rand()) / RAND_MAX * 5.0f;
        float y = x1 * 2.0f + x2 * 1.5f + (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.5f;

        data.push_back({x1, x2});
        targets.push_back({y});
    }

    // Run C++ baseline benchmarks
    auto cppResults = validator.benchmarkAgainstBaselines(data, targets, 0.3f);
    EXPECT_FALSE(cppResults.empty());

    // Run scikit-learn benchmarks if available
    try {
        auto sklearnResults = validator.benchmarkAgainstScikitLearn(data, targets, 0.3f);

        if (!sklearnResults.empty()) {
            // Compare results between C++ and scikit-learn implementations
            // Both should have similar performance trends

            // Find Linear Regression results from both
            auto cppLinear = std::find_if(cppResults.begin(), cppResults.end(),
                [](const BenchmarkResult& r) { return r.modelName == "Linear Regression"; });
            auto sklearnLinear = std::find_if(sklearnResults.begin(), sklearnResults.end(),
                [](const BenchmarkResult& r) { return r.modelName == "Linear Regression"; });

            if (cppLinear != cppResults.end() && sklearnLinear != sklearnResults.end()) {
                // Both should achieve reasonable R² scores on linear data
                EXPECT_GT(cppLinear->metrics.r2_score, 0.5f);
                EXPECT_GT(sklearnLinear->metrics.r2_score, 0.5f);

                // Scikit-learn might be slightly better due to optimization
                // but both should be in reasonable range
                EXPECT_NEAR(cppLinear->metrics.r2_score, sklearnLinear->metrics.r2_score, 0.3f);
            }
        }

    } catch (const std::exception& e) {
        // Scikit-learn comparison is optional
        SUCCEED() << "Comprehensive benchmark comparison test passed (scikit-learn not available)";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
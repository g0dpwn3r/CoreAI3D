#include <gtest/gtest.h>
#include "ValidationFramework.hpp"
#include "Train.hpp"
#include <thread>
#include <mutex>
#include <atomic>
#include <limits>
#include <cmath>
#include <vector>
#include <memory>

// Helper function to create large datasets
void createLargeDataset(const std::string& filename, int numSamples, int numFeatures) {
    std::ofstream file(filename);

    // Write header
    for (int i = 0; i < numFeatures; i++) {
        file << "feature_" << i;
        if (i < numFeatures - 1) file << ",";
    }
    file << ",target\n";

    for (int i = 0; i < numSamples; i++) {
        // Generate features
        for (int j = 0; j < numFeatures; j++) {
            float feature = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // -1 to 1
            file << feature << ",";
        }
        // Generate target
        float target = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        file << target << "\n";
    }

    file.close();
}

// Helper function to create dataset with extreme values
void createExtremeValueDataset(const std::string& filename, int numSamples, int numFeatures) {
    std::ofstream file(filename);

    // Write header
    for (int i = 0; i < numFeatures; i++) {
        file << "feature_" << i;
        if (i < numFeatures - 1) file << ",";
    }
    file << ",target\n";

    for (int i = 0; i < numSamples; i++) {
        // Generate features with extreme values
        for (int j = 0; j < numFeatures; j++) {
            float feature;
            int type = rand() % 5;
            switch (type) {
                case 0: feature = std::numeric_limits<float>::max(); break;
                case 1: feature = std::numeric_limits<float>::lowest(); break;
                case 2: feature = std::numeric_limits<float>::infinity(); break;
                case 3: feature = -std::numeric_limits<float>::infinity(); break;
                case 4: feature = std::numeric_limits<float>::quiet_NaN(); break;
                default: feature = 0.0f;
            }
            file << feature << ",";
        }
        // Generate target with extreme values
        float target;
        int type = rand() % 5;
        switch (type) {
            case 0: target = std::numeric_limits<float>::max(); break;
            case 1: target = std::numeric_limits<float>::lowest(); break;
            case 2: target = std::numeric_limits<float>::infinity(); break;
            case 3: target = -std::numeric_limits<float>::infinity(); break;
            case 4: target = std::numeric_limits<float>::quiet_NaN(); break;
            default: target = 0.0f;
        }
        file << target << "\n";
    }

    file.close();
}

TEST(EdgeCaseTest, LargeDatasetHandling) {
    ValidationFramework validator(true);

    // Test with very large dataset (100k samples, 50 features)
    const int LARGE_SAMPLE_COUNT = 100000;
    const int FEATURE_COUNT = 50;

    createLargeDataset("large_dataset.csv", LARGE_SAMPLE_COUNT, FEATURE_COUNT);

    // Load and validate the dataset
    auto validationResult = validator.validateDataset("large_dataset.csv", true);
    EXPECT_TRUE(validationResult.isValid);

    // Load data for training
    Training loader(false, false);
    bool loaded = loader.loadCSV("large_dataset.csv", LARGE_SAMPLE_COUNT, 1, true, false, ',', "large_test");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        EXPECT_EQ(inputs.size(), LARGE_SAMPLE_COUNT);
        EXPECT_EQ(targets.size(), LARGE_SAMPLE_COUNT);

        // Test training on large dataset (limit epochs to avoid excessive time)
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setLearningRate(0.01f);
        trainer->setEpochs(1); // Only 1 epoch for large dataset

        bool trained = trainer->train(inputs, targets);
        EXPECT_TRUE(trained);

        if (trained) {
            // Test inference on subset
            std::vector<std::vector<float>> testInputs(inputs.begin(), inputs.begin() + 100);
            auto predictions = trainer->predict(testInputs);
            EXPECT_EQ(predictions.size(), 100);
        }
    }

    // Clean up
    std::remove("large_dataset.csv");
}

TEST(EdgeCaseTest, ExtremeValueHandling) {
    ValidationFramework validator(true);

    // Create dataset with extreme values
    createExtremeValueDataset("extreme_values.csv", 1000, 10);

    // Load and validate the dataset
    auto validationResult = validator.validateDataset("extreme_values.csv", true);
    // Dataset with NaN/inf might not validate as "valid" but should not crash
    EXPECT_NO_THROW({
        auto result = validator.validateDataset("extreme_values.csv", true);
    });

    // Load data for training
    Training loader(false, false);
    bool loaded = loader.loadCSV("extreme_values.csv", 1000, 1, true, false, ',', "extreme_test");

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // Test training with extreme values (should handle gracefully)
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setLearningRate(0.001f); // Lower learning rate for stability
        trainer->setEpochs(5);

        // Should not crash even with extreme values
        EXPECT_NO_THROW({
            bool trained = trainer->train(inputs, targets);
            // Training might fail with extreme values, but shouldn't crash
        });
    }

    // Clean up
    std::remove("extreme_values.csv");
}

TEST(EdgeCaseTest, MemoryLimitTesting) {
    ValidationFramework validator(true);

    // Test with progressively larger datasets to approach memory limits
    std::vector<int> datasetSizes = {10000, 50000, 100000};

    for (int size : datasetSizes) {
        createLargeDataset("memory_test_" + std::to_string(size) + ".csv", size, 20);

        Training loader(false, false);
        bool loaded = loader.loadCSV("memory_test_" + std::to_string(size) + ".csv", size, 1, true, false, ',', "memory_test");

        if (loaded) {
            const auto& inputs = loader.getInputs();
            const auto& targets = loader.getTargets();

            // Test memory usage during training
            auto trainer = std::make_unique<Training>(false, false);
            trainer->setEpochs(1);

            // Measure memory before training
            auto memBefore = validator.getCurrentMemoryUsage();

            bool trained = trainer->train(inputs, targets);

            // Measure memory after training
            auto memAfter = validator.getCurrentMemoryUsage();

            if (trained) {
                // Memory usage should be reasonable (not excessive growth)
                EXPECT_LT(memAfter - memBefore, 1024 * 1024 * 100); // Less than 100MB increase
            }
        }

        // Clean up
        std::remove(("memory_test_" + std::to_string(size) + ".csv").c_str());
    }
}

TEST(EdgeCaseTest, ConcurrentAccessTest) {
    ValidationFramework validator(true);

    // Create a moderate-sized dataset
    createLargeDataset("concurrent_test.csv", 5000, 10);

    Training loader(false, false);
    bool loaded = loader.loadCSV("concurrent_test.csv", 5000, 1, true, false, ',', "concurrent_test");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // Train a model first
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setEpochs(10);
        bool trained = trainer->train(inputs, targets);
        EXPECT_TRUE(trained);

        if (trained) {
            // Test concurrent predictions
            const int NUM_THREADS = 4;
            const int PREDICTIONS_PER_THREAD = 100;

            std::vector<std::thread> threads;
            std::atomic<int> successfulPredictions(0);
            std::mutex predictionMutex;

            for (int t = 0; t < NUM_THREADS; t++) {
                threads.emplace_back([&]() {
                    try {
                        // Each thread makes predictions
                        std::vector<std::vector<float>> threadInputs;
                        for (int i = 0; i < PREDICTIONS_PER_THREAD; i++) {
                            int idx = rand() % inputs.size();
                            threadInputs.push_back(inputs[idx]);
                        }

                        auto predictions = trainer->predict(threadInputs);

                        std::lock_guard<std::mutex> lock(predictionMutex);
                        if (predictions.size() == PREDICTIONS_PER_THREAD) {
                            successfulPredictions += PREDICTIONS_PER_THREAD;
                        }
                    } catch (...) {
                        // Thread failed
                    }
                });
            }

            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }

            // Should have successful predictions from all threads
            EXPECT_EQ(successfulPredictions, NUM_THREADS * PREDICTIONS_PER_THREAD);
        }
    }

    // Clean up
    std::remove("concurrent_test.csv");
}

TEST(EdgeCaseTest, ZeroAndNegativeDimensionHandling) {
    // Test edge cases with invalid dimensions
    EXPECT_THROW({
        CoreAI core(0, 1, 4, 2, 0.0f, 1.0f); // Zero input size
    }, std::exception);

    EXPECT_THROW({
        CoreAI core(3, 0, 4, 2, 0.0f, 1.0f); // Zero hidden size
    }, std::exception);

    EXPECT_THROW({
        CoreAI core(3, 1, 0, 2, 0.0f, 1.0f); // Zero layers
    }, std::exception);

    EXPECT_THROW({
        CoreAI core(3, 1, 4, 0, 0.0f, 1.0f); // Zero output size
    }, std::exception);
}

TEST(EdgeCaseTest, EmptyDatasetHandling) {
    ValidationFramework validator(true);

    // Create empty dataset
    std::ofstream emptyFile("empty_dataset.csv");
    emptyFile << "feature_0,target\n"; // Only header
    emptyFile.close();

    Training loader(false, false);
    bool loaded = loader.loadCSV("empty_dataset.csv", 0, 1, true, false, ',', "empty_test");

    // Should handle empty dataset gracefully
    EXPECT_FALSE(loaded);

    // Clean up
    std::remove("empty_dataset.csv");
}

TEST(EdgeCaseTest, SingleSampleTraining) {
    ValidationFramework validator(true);

    // Create dataset with single sample
    std::ofstream singleFile("single_sample.csv");
    singleFile << "feature_0,feature_1,target\n";
    singleFile << "1.0,2.0,3.0\n";
    singleFile.close();

    Training loader(false, false);
    bool loaded = loader.loadCSV("single_sample.csv", 1, 1, true, false, ',', "single_test");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        EXPECT_EQ(inputs.size(), 1);
        EXPECT_EQ(targets.size(), 1);

        // Training with single sample should work
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setEpochs(1);

        bool trained = trainer->train(inputs, targets);
        EXPECT_TRUE(trained);

        if (trained) {
            // Test prediction
            auto predictions = trainer->predict(inputs);
            EXPECT_EQ(predictions.size(), 1);
        }
    }

    // Clean up
    std::remove("single_sample.csv");
}

TEST(EdgeCaseTest, HighDimensionalData) {
    ValidationFramework validator(true);

    // Test with very high-dimensional data (1000 features)
    const int HIGH_DIM_FEATURES = 1000;
    createLargeDataset("high_dim_dataset.csv", 100, HIGH_DIM_FEATURES);

    Training loader(false, false);
    bool loaded = loader.loadCSV("high_dim_dataset.csv", 100, 1, true, false, ',', "high_dim_test");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        EXPECT_EQ(inputs.size(), 100);
        EXPECT_EQ(inputs[0].size(), HIGH_DIM_FEATURES);

        // Training on high-dimensional data
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setEpochs(5);

        bool trained = trainer->train(inputs, targets);
        EXPECT_TRUE(trained);

        if (trained) {
            // Test prediction on subset
            std::vector<std::vector<float>> testInputs(inputs.begin(), inputs.begin() + 10);
            auto predictions = trainer->predict(testInputs);
            EXPECT_EQ(predictions.size(), 10);
        }
    }

    // Clean up
    std::remove("high_dim_dataset.csv");
}
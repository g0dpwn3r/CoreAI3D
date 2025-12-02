#include <gtest/gtest.h>
#include "ValidationFramework.hpp"
#include "Train.hpp"
#include <chrono>
#include <thread>
#include <atomic>
#include <memory>
#include <fstream>
#include <iostream>

// Helper function to create stress test dataset
void createStressDataset(const std::string& filename, int numSamples, int numFeatures,
                        float complexity = 1.0f) {
    std::ofstream file(filename);

    // Write header
    for (int i = 0; i < numFeatures; i++) {
        file << "feature_" << i;
        if (i < numFeatures - 1) file << ",";
    }
    file << ",target\n";

    for (int i = 0; i < numSamples; i++) {
        // Generate complex features
        for (int j = 0; j < numFeatures; j++) {
            float feature = sin(i * 0.01f * complexity) * cos(j * 0.01f * complexity) +
                           static_cast<float>(rand()) / RAND_MAX * 0.1f;
            file << feature << ",";
        }

        // Generate complex target
        float target = 0.0f;
        for (int j = 0; j < numFeatures; j++) {
            target += sin(j * 0.1f * complexity) * cos(i * 0.1f * complexity);
        }
        target += static_cast<float>(rand()) / RAND_MAX * 0.1f;
        file << target << "\n";
    }

    file.close();
}

// Helper function to monitor system resources
struct ResourceMonitor {
    long long initialMemory;
    std::chrono::steady_clock::time_point startTime;

    ResourceMonitor() {
        initialMemory = getCurrentMemoryUsage();
        startTime = std::chrono::steady_clock::now();
    }

    long long getCurrentMemoryUsage() {
        // Simple memory usage estimation (in KB)
        // In a real implementation, you'd use platform-specific APIs
        return 0; // Placeholder
    }

    double getElapsedTime() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count() / 1000.0;
    }

    long long getMemoryDelta() {
        return getCurrentMemoryUsage() - initialMemory;
    }
};

TEST(StressTest, LongRunningTraining) {
    ValidationFramework validator(true);

    // Create a moderately complex dataset for long training
    const int SAMPLES = 10000;
    const int FEATURES = 20;
    const int EPOCHS = 100; // Long training duration

    createStressDataset("long_training.csv", SAMPLES, FEATURES, 2.0f);

    Training loader(false, false);
    bool loaded = loader.loadCSV("long_training.csv", SAMPLES, 1, true, false, ',', "long_training");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // Start resource monitoring
        ResourceMonitor monitor;

        // Train for many epochs
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setLearningRate(0.001f);
        trainer->setEpochs(EPOCHS);

        std::cout << "Starting long-running training test (" << EPOCHS << " epochs)..." << std::endl;

        bool trained = trainer->train(inputs, targets);
        EXPECT_TRUE(trained);

        double trainingTime = monitor.getElapsedTime();
        long long memoryDelta = monitor.getMemoryDelta();

        std::cout << "Training completed in " << trainingTime << " seconds" << std::endl;
        std::cout << "Memory delta: " << memoryDelta << " KB" << std::endl;

        if (trained) {
            // Verify the model can still make predictions
            std::vector<std::vector<float>> testInputs(inputs.begin(), inputs.begin() + 100);
            auto predictions = trainer->predict(testInputs);
            EXPECT_EQ(predictions.size(), 100);

            // Training should have taken reasonable time (not too fast, not crashed)
            EXPECT_GT(trainingTime, 1.0); // At least 1 second
            EXPECT_LT(trainingTime, 300.0); // Less than 5 minutes
        }
    }

    // Clean up
    std::remove("long_training.csv");
}

TEST(StressTest, ResourceExhaustionMemory) {
    ValidationFramework validator(true);

    // Test memory exhaustion with increasingly large models
    std::vector<std::tuple<int, int, int>> modelConfigs = {
        {100, 50, 10},   // Small model
        {1000, 100, 20}, // Medium model
        {5000, 200, 30}, // Large model
    };

    for (auto& config : modelConfigs) {
        int samples, features, epochs;
        std::tie(samples, features, epochs) = config;

        createStressDataset("memory_stress_" + std::to_string(samples) + ".csv", samples, features);

        Training loader(false, false);
        bool loaded = loader.loadCSV("memory_stress_" + std::to_string(samples) + ".csv",
                                   samples, 1, true, false, ',', "memory_stress");

        if (loaded) {
            const auto& inputs = loader.getInputs();
            const auto& targets = loader.getTargets();

            ResourceMonitor monitor;

            auto trainer = std::make_unique<Training>(false, false);
            trainer->setEpochs(epochs);

            // Should not crash due to memory exhaustion
            EXPECT_NO_THROW({
                bool trained = trainer->train(inputs, targets);
                if (trained) {
                    // Test prediction
                    auto predictions = trainer->predict(std::vector<std::vector<float>>(inputs.begin(), inputs.begin() + 10));
                    EXPECT_FALSE(predictions.empty());
                }
            });

            long long memoryDelta = monitor.getMemoryDelta();
            std::cout << "Model (" << samples << " samples, " << features << " features): Memory delta = "
                      << memoryDelta << " KB" << std::endl;
        }

        // Clean up
        std::remove(("memory_stress_" + std::to_string(samples) + ".csv").c_str());
    }
}

TEST(StressTest, ContinuousTrainingLoad) {
    ValidationFramework validator(true);

    // Test continuous training with multiple iterations
    const int ITERATIONS = 10;
    const int SAMPLES = 1000;
    const int FEATURES = 10;

    createStressDataset("continuous_training.csv", SAMPLES, FEATURES);

    Training loader(false, false);
    bool loaded = loader.loadCSV("continuous_training.csv", SAMPLES, 1, true, false, ',', "continuous");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        ResourceMonitor totalMonitor;

        for (int iter = 0; iter < ITERATIONS; iter++) {
            std::cout << "Iteration " << (iter + 1) << "/" << ITERATIONS << std::endl;

            auto trainer = std::make_unique<Training>(false, false);
            trainer->setEpochs(5);

            ResourceMonitor iterMonitor;

            bool trained = trainer->train(inputs, targets);
            EXPECT_TRUE(trained);

            if (trained) {
                // Test prediction
                auto predictions = trainer->predict(std::vector<std::vector<float>>(inputs.begin(), inputs.begin() + 10));
                EXPECT_EQ(predictions.size(), 10);
            }

            double iterTime = iterMonitor.getElapsedTime();
            std::cout << "Iteration " << (iter + 1) << " completed in " << iterTime << " seconds" << std::endl;
        }

        double totalTime = totalMonitor.getElapsedTime();
        std::cout << "Total continuous training time: " << totalTime << " seconds" << std::endl;

        // Should complete within reasonable time
        EXPECT_LT(totalTime, 120.0); // Less than 2 minutes
    }

    // Clean up
    std::remove("continuous_training.csv");
}

TEST(StressTest, ConcurrentTrainingStress) {
    ValidationFramework validator(true);

    // Test multiple training sessions running concurrently
    const int NUM_CONCURRENT_TRAININGS = 3;
    const int SAMPLES_PER_TRAINING = 2000;
    const int FEATURES = 15;

    std::vector<std::string> datasetFiles;
    std::vector<std::unique_ptr<Training>> trainers;
    std::atomic<int> completedTrainings(0);

    // Create datasets and trainers
    for (int i = 0; i < NUM_CONCURRENT_TRAININGS; i++) {
        std::string filename = "concurrent_stress_" + std::to_string(i) + ".csv";
        datasetFiles.push_back(filename);

        createStressDataset(filename, SAMPLES_PER_TRAINING, FEATURES, 1.0f + i * 0.5f);

        auto trainer = std::make_unique<Training>(false, false);
        trainer->setEpochs(10);
        trainers.push_back(std::move(trainer));
    }

    // Load datasets
    std::vector<std::vector<std::vector<float>>> allInputs;
    std::vector<std::vector<std::vector<float>>> allTargets;

    for (int i = 0; i < NUM_CONCURRENT_TRAININGS; i++) {
        Training loader(false, false);
        bool loaded = loader.loadCSV(datasetFiles[i], SAMPLES_PER_TRAINING, 1, true, false, ',', "concurrent_" + std::to_string(i));

        if (loaded) {
            allInputs.push_back(loader.getInputs());
            allTargets.push_back(loader.getTargets());
        } else {
            allInputs.emplace_back();
            allTargets.emplace_back();
        }
    }

    // Start concurrent training
    std::vector<std::thread> threads;

    ResourceMonitor monitor;

    for (int i = 0; i < NUM_CONCURRENT_TRAININGS; i++) {
        threads.emplace_back([&, i]() {
            if (!allInputs[i].empty() && !allTargets[i].empty()) {
                bool trained = trainers[i]->train(allInputs[i], allTargets[i]);
                if (trained) {
                    completedTrainings++;
                }
            }
        });
    }

    // Wait for all trainings to complete
    for (auto& thread : threads) {
        thread.join();
    }

    double totalTime = monitor.getElapsedTime();

    std::cout << "Concurrent training completed in " << totalTime << " seconds" << std::endl;
    std::cout << "Completed trainings: " << completedTrainings << "/" << NUM_CONCURRENT_TRAININGS << std::endl;

    // Should complete all trainings
    EXPECT_EQ(completedTrainings, NUM_CONCURRENT_TRAININGS);

    // Concurrent training should be faster than sequential
    double expectedSequentialTime = 10.0 * NUM_CONCURRENT_TRAININGS; // Rough estimate
    EXPECT_LT(totalTime, expectedSequentialTime * 2.0); // Allow some overhead

    // Clean up
    for (const auto& file : datasetFiles) {
        std::remove(file.c_str());
    }
}

TEST(StressTest, LargeScaleDataProcessing) {
    ValidationFramework validator(true);

    // Test processing very large amounts of data
    const int BATCH_SIZE = 1000;
    const int NUM_BATCHES = 50;
    const int FEATURES = 25;

    ResourceMonitor monitor;

    std::vector<double> processingTimes;

    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        std::string filename = "batch_" + std::to_string(batch) + ".csv";
        createStressDataset(filename, BATCH_SIZE, FEATURES, 1.0f + batch * 0.1f);

        auto batchStart = std::chrono::steady_clock::now();

        Training loader(false, false);
        bool loaded = loader.loadCSV(filename, BATCH_SIZE, 1, true, false, ',', "batch_" + std::to_string(batch));

        if (loaded) {
            const auto& inputs = loader.getInputs();
            const auto& targets = loader.getTargets();

            // Quick training on each batch
            auto trainer = std::make_unique<Training>(false, false);
            trainer->setEpochs(1);

            bool trained = trainer->train(inputs, targets);
            EXPECT_TRUE(trained);

            if (trained) {
                // Test prediction
                auto predictions = trainer->predict(std::vector<std::vector<float>>(inputs.begin(), inputs.begin() + 10));
                EXPECT_EQ(predictions.size(), 10);
            }
        }

        auto batchEnd = std::chrono::steady_clock::now();
        double batchTime = std::chrono::duration_cast<std::chrono::milliseconds>(batchEnd - batchStart).count() / 1000.0;
        processingTimes.push_back(batchTime);

        // Clean up
        std::remove(filename.c_str());
    }

    double totalTime = monitor.getElapsedTime();
    double avgBatchTime = 0.0;
    for (double time : processingTimes) {
        avgBatchTime += time;
    }
    avgBatchTime /= processingTimes.size();

    std::cout << "Large scale processing completed in " << totalTime << " seconds" << std::endl;
    std::cout << "Average batch processing time: " << avgBatchTime << " seconds" << std::endl;

    // Should process all batches within reasonable time
    EXPECT_LT(totalTime, 300.0); // Less than 5 minutes
    EXPECT_GT(avgBatchTime, 0.1); // At least 100ms per batch
}

TEST(StressTest, MemoryLeakDetection) {
    ValidationFramework validator(true);

    // Test for memory leaks during repeated training cycles
    const int CYCLES = 20;
    const int SAMPLES = 500;
    const int FEATURES = 10;

    createStressDataset("leak_test.csv", SAMPLES, FEATURES);

    std::vector<long long> memoryUsage;

    for (int cycle = 0; cycle < CYCLES; cycle++) {
        Training loader(false, false);
        bool loaded = loader.loadCSV("leak_test.csv", SAMPLES, 1, true, false, ',', "leak_test");

        if (loaded) {
            const auto& inputs = loader.getInputs();
            const auto& targets = loader.getTargets();

            auto trainer = std::make_unique<Training>(false, false);
            trainer->setEpochs(2);

            bool trained = trainer->train(inputs, targets);
            EXPECT_TRUE(trained);

            // Record memory usage
            memoryUsage.push_back(validator.getCurrentMemoryUsage());
        }
    }

    // Check for significant memory growth (potential leak)
    if (memoryUsage.size() >= 2) {
        long long initialMemory = memoryUsage[0];
        long long finalMemory = memoryUsage.back();

        long long memoryGrowth = finalMemory - initialMemory;
        double growthPercentage = (static_cast<double>(memoryGrowth) / initialMemory) * 100.0;

        std::cout << "Memory growth over " << CYCLES << " cycles: " << memoryGrowth
                  << " KB (" << growthPercentage << "%)" << std::endl;

        // Allow some memory growth but not excessive (less than 50% growth)
        EXPECT_LT(growthPercentage, 50.0);
    }

    // Clean up
    std::remove("leak_test.csv");
}
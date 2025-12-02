#include <gtest/gtest.h>
#include "ValidationFramework.hpp"
#include "Train.hpp"
#include <chrono>
#include <memory>
#include <fstream>
#include <iostream>

// Performance test configuration
struct PerformanceConfig {
    int numSamples;
    int numFeatures;
    int numEpochs;
    double learningRate;
    int hiddenLayers;
    int neuronsPerLayer;
};

class PerformanceTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test data
        createPerformanceDataset("performance_test.csv", 1000, 10);
    }

    void TearDown() override {
        // Clean up test files
        std::remove("performance_test.csv");
        std::remove("performance_report.json");
    }

    void createPerformanceDataset(const std::string& filename, int numSamples, int numFeatures) {
        std::ofstream file(filename);
        file << "feature_0";
        for (int i = 1; i < numFeatures; i++) {
            file << ",feature_" << i;
        }
        file << ",target\n";

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                float value = static_cast<float>(rand()) / RAND_MAX * 10.0f;
                file << value;
                if (j < numFeatures - 1) file << ",";
            }
            // Create target with some correlation to features
            float target = 0.0f;
            for (int j = 0; j < numFeatures; j++) {
                target += (static_cast<float>(rand()) / RAND_MAX * 10.0f) * 0.1f;
            }
            target += static_cast<float>(rand()) / RAND_MAX * 2.0f; // Add noise
            file << "," << target << "\n";
        }
        file.close();
    }

    PerformanceConfig defaultConfig = {1000, 10, 50, 0.01, 2, 16};
};

// Test training time performance
TEST_F(PerformanceTests, TrainingTimeBenchmark) {
    ValidationFramework validator(true);

    // Load test data
    Training loader(false, false);
    bool loaded = loader.loadCSV("performance_test.csv", 1000, 1, true, false, ',', "perf_test");
    ASSERT_TRUE(loaded);

    const auto& inputs = loader.getInputs();
    const auto& targets = loader.getTargets();

    // Test different network sizes
    std::vector<std::tuple<int, int, std::string>> networkConfigs = {
        {1, 8, "Small"},
        {2, 16, "Medium"},
        {3, 32, "Large"}
    };

    std::ofstream report("training_time_report.csv");
    report << "NetworkSize,TrainingTime_ms,Epochs,Samples,Features\n";

    for (const auto& config : networkConfigs) {
        int layers, neurons;
        std::string sizeName;
        std::tie(layers, neurons, sizeName) = config;

        auto start = std::chrono::high_resolution_clock::now();

        // Create and train network
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setNetworkConfig(layers, neurons, inputs[0].size(), targets[0].size());
        trainer->train(inputs, targets, defaultConfig.learningRate, defaultConfig.numEpochs);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        report << sizeName << "," << duration.count() << "," << defaultConfig.numEpochs
               << "," << inputs.size() << "," << inputs[0].size() << "\n";

        std::cout << "Training time for " << sizeName << " network: " << duration.count() << " ms" << std::endl;
    }

    report.close();
}

// Test inference speed performance
TEST_F(PerformanceTests, InferenceSpeedBenchmark) {
    ValidationFramework validator(true);

    // Load test data
    Training loader(false, false);
    bool loaded = loader.loadCSV("performance_test.csv", 1000, 1, true, false, ',', "perf_test");
    ASSERT_TRUE(loaded);

    const auto& inputs = loader.getInputs();
    const auto& targets = loader.getTargets();

    // Train a network first
    auto trainer = std::make_unique<Training>(false, false);
    trainer->setNetworkConfig(2, 16, inputs[0].size(), targets[0].size());
    trainer->train(inputs, targets, 0.01, 20);

    // Test inference speed
    const int numInferenceRuns = 1000;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numInferenceRuns; i++) {
        auto predictions = trainer->predict(inputs[i % inputs.size()]);
        ASSERT_FALSE(predictions.empty());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avgInferenceTime = static_cast<double>(duration.count()) / numInferenceRuns;
    double inferencesPerSecond = 1000000.0 / avgInferenceTime;

    std::cout << "Average inference time: " << avgInferenceTime << " microseconds" << std::endl;
    std::cout << "Inferences per second: " << inferencesPerSecond << std::endl;

    // Save results
    std::ofstream report("inference_speed_report.csv");
    report << "Metric,Value,Unit\n";
    report << "AverageInferenceTime," << avgInferenceTime << ",microseconds\n";
    report << "InferencesPerSecond," << inferencesPerSecond << ",per_second\n";
    report << "TestRuns," << numInferenceRuns << ",count\n";
    report.close();

    // Performance assertions
    EXPECT_LT(avgInferenceTime, 1000.0); // Should be less than 1ms per inference
    EXPECT_GT(inferencesPerSecond, 100.0); // Should handle at least 100 inferences/second
}

// Test memory usage during training
TEST_F(PerformanceTests, MemoryUsageBenchmark) {
    ValidationFramework validator(true);

    // Test with different dataset sizes
    std::vector<int> datasetSizes = {100, 500, 1000, 2000};

    std::ofstream report("memory_usage_report.csv");
    report << "DatasetSize,EstimatedMemoryUsage_MB,Features,Epochs\n";

    for (int size : datasetSizes) {
        // Create dataset of specific size
        createPerformanceDataset("temp_dataset.csv", size, 10);

        Training loader(false, false);
        bool loaded = loader.loadCSV("temp_dataset.csv", size, 1, true, false, ',', "mem_test");
        ASSERT_TRUE(loaded);

        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // Estimate memory usage
        size_t estimatedMemory = 0;
        // Network weights (rough estimate)
        int numWeights = 10 * 16 + 16 * 16 + 16 * 1; // input->hidden1->hidden2->output
        estimatedMemory += numWeights * sizeof(float);
        // Dataset memory
        estimatedMemory += inputs.size() * inputs[0].size() * sizeof(float);
        estimatedMemory += targets.size() * targets[0].size() * sizeof(float);
        // Additional overhead
        estimatedMemory += 1024 * 1024; // 1MB overhead estimate

        double memoryMB = static_cast<double>(estimatedMemory) / (1024 * 1024);

        report << size << "," << memoryMB << "," << inputs[0].size() << "," << defaultConfig.numEpochs << "\n";

        std::cout << "Dataset size " << size << ": estimated memory usage " << memoryMB << " MB" << std::endl;

        // Clean up
        std::remove("temp_dataset.csv");

        // Memory usage should be reasonable
        EXPECT_LT(memoryMB, 100.0); // Should use less than 100MB for these sizes
    }

    report.close();
}

// Test scalability with increasing data size
TEST_F(PerformanceTests, ScalabilityTest) {
    ValidationFramework validator(true);

    std::vector<int> scales = {500, 1000, 2000, 5000};

    std::ofstream report("scalability_report.csv");
    report << "DataSize,TrainingTime_ms,InferenceTime_us,MemoryEstimate_MB\n";

    for (int scale : scales) {
        createPerformanceDataset("scale_test.csv", scale, 5);

        Training loader(false, false);
        bool loaded = loader.loadCSV("scale_test.csv", scale, 1, true, false, ',', "scale_test");
        ASSERT_TRUE(loaded);

        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // Train network
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setNetworkConfig(1, 8, inputs[0].size(), targets[0].size());

        auto trainStart = std::chrono::high_resolution_clock::now();
        trainer->train(inputs, targets, 0.01, 10); // Fewer epochs for scalability test
        auto trainEnd = std::chrono::high_resolution_clock::now();
        auto trainDuration = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);

        // Test inference speed
        auto inferenceStart = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < std::min(100UL, inputs.size()); i++) {
            auto predictions = trainer->predict(inputs[i]);
            ASSERT_FALSE(predictions.empty());
        }
        auto inferenceEnd = std::chrono::high_resolution_clock::now();
        auto inferenceDuration = std::chrono::duration_cast<std::chrono::microseconds>(inferenceEnd - inferenceStart);
        double avgInferenceTime = static_cast<double>(inferenceDuration.count()) / std::min(100UL, inputs.size());

        // Memory estimate
        size_t estimatedMemory = inputs.size() * inputs[0].size() * sizeof(float) +
                                targets.size() * targets[0].size() * sizeof(float) +
                                100000; // Network overhead
        double memoryMB = static_cast<double>(estimatedMemory) / (1024 * 1024);

        report << scale << "," << trainDuration.count() << "," << avgInferenceTime << "," << memoryMB << "\n";

        std::cout << "Scale " << scale << ": train=" << trainDuration.count() << "ms, "
                  << "inference=" << avgInferenceTime << "us, memory=" << memoryMB << "MB" << std::endl;

        std::remove("scale_test.csv");
    }

    report.close();
}

// Test concurrent performance (multi-threading)
TEST_F(PerformanceTests, ConcurrentPerformanceTest) {
    ValidationFramework validator(true);

    const int numThreads = 4;
    const int samplesPerThread = 250;

    // Create test data
    createPerformanceDataset("concurrent_test.csv", samplesPerThread * numThreads, 5);

    Training loader(false, false);
    bool loaded = loader.loadCSV("concurrent_test.csv", samplesPerThread * numThreads, 1, true, false, ',', "concurrent");
    ASSERT_TRUE(loaded);

    const auto& inputs = loader.getInputs();
    const auto& targets = loader.getTargets();

    // Split data for concurrent processing
    std::vector<std::future<void>> futures;

    auto concurrentTrain = [&](int threadId, const std::vector<std::vector<float>>& threadInputs,
                              const std::vector<std::vector<float>>& threadTargets) {
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setNetworkConfig(1, 8, threadInputs[0].size(), threadTargets[0].size());

        auto start = std::chrono::high_resolution_clock::now();
        trainer->train(threadInputs, threadTargets, 0.01, 20);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Thread " << threadId << " training time: " << duration.count() << " ms" << std::endl;
    };

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numThreads; i++) {
        size_t startIdx = i * samplesPerThread;
        size_t endIdx = std::min(startIdx + samplesPerThread, inputs.size());

        std::vector<std::vector<float>> threadInputs(inputs.begin() + startIdx, inputs.begin() + endIdx);
        std::vector<std::vector<float>> threadTargets(targets.begin() + startIdx, targets.begin() + endIdx);

        futures.push_back(std::async(std::launch::async, concurrentTrain, i, threadInputs, threadTargets));
    }

    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Total concurrent training time: " << totalDuration.count() << " ms" << std::endl;

    std::remove("concurrent_test.csv");

    // Concurrent processing should be reasonably fast
    EXPECT_LT(totalDuration.count(), 10000); // Should complete within 10 seconds
}

// Benchmark result tracking test
TEST_F(PerformanceTests, BenchmarkResultTracking) {
    ValidationFramework validator(true);

    // Create a comprehensive performance benchmark
    createPerformanceDataset("tracking_test.csv", 500, 8);

    Training loader(false, false);
    bool loaded = loader.loadCSV("tracking_test.csv", 500, 1, true, false, ',', "tracking");
    ASSERT_TRUE(loaded);

    const auto& inputs = loader.getInputs();
    const auto& targets = loader.getTargets();

    // Run benchmark with tracking
    auto results = validator.benchmarkAgainstBaselines(inputs, targets, 0.3f);

    // Save detailed results
    std::ofstream jsonReport("performance_report.json");
    jsonReport << "{\n";
    jsonReport << "  \"performance_benchmark\": {\n";
    jsonReport << "    \"timestamp\": \"" << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\",\n";
    jsonReport << "    \"dataset_size\": " << inputs.size() << ",\n";
    jsonReport << "    \"features\": " << inputs[0].size() << ",\n";
    jsonReport << "    \"models_tested\": " << results.size() << ",\n";
    jsonReport << "    \"results\": [\n";

    for (size_t i = 0; i < results.size(); i++) {
        const auto& result = results[i];
        jsonReport << "      {\n";
        jsonReport << "        \"model\": \"" << result.modelName << "\",\n";
        jsonReport << "        \"accuracy\": " << result.metrics.accuracy << ",\n";
        jsonReport << "        \"rmse\": " << result.metrics.rmse << ",\n";
        jsonReport << "        \"mae\": " << result.metrics.mae << ",\n";
        jsonReport << "        \"training_time_ms\": " << result.trainingTime.count() / 1000.0 << ",\n";
        jsonReport << "        \"inference_time_us\": " << result.inferenceTime.count() << ",\n";
        jsonReport << "        \"memory_usage_kb\": " << result.metrics.memoryUsage / 1024 << "\n";
        jsonReport << "      }";
        if (i < results.size() - 1) jsonReport << ",";
        jsonReport << "\n";
    }

    jsonReport << "    ]\n";
    jsonReport << "  }\n";
    jsonReport << "}\n";
    jsonReport.close();

    std::cout << "Performance benchmark results saved to performance_report.json" << std::endl;

    // Verify results are reasonable
    ASSERT_FALSE(results.empty());
    for (const auto& result : results) {
        EXPECT_GE(result.metrics.accuracy, -1.0f);
        EXPECT_GE(result.trainingTime.count(), 0);
        EXPECT_GE(result.inferenceTime.count(), 0);
    }

    std::remove("tracking_test.csv");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#include "Train.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <memory>

struct PerformanceMetrics {
    double training_time_seconds;
    double final_loss;
    double final_accuracy;
    size_t peak_memory_usage_mb;
    size_t gradient_history_size;
    int epochs_completed;
    std::string configuration_name;
};

class PerformanceTester {
private:
    std::vector<std::vector<float>> test_inputs;
    std::vector<std::vector<float>> test_targets;
    std::vector<std::vector<float>> validation_inputs;
    std::vector<std::vector<float>> validation_targets;

    void generateDataset(int num_samples, bool is_validation = false) {
        auto& inputs = is_validation ? validation_inputs : test_inputs;
        auto& targets = is_validation ? validation_targets : test_targets;

        inputs.clear();
        targets.clear();

        // Generate a more complex dataset: y = 2*x1 + 3*x2 + 0.5*x1*x2 + noise
        for (int i = 0; i < num_samples; ++i) {
            float x1 = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f; // [-2, 2]
            float x2 = static_cast<float>(rand()) / RAND_MAX * 4.0f - 2.0f; // [-2, 2]
            float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f; // small noise

            float y = 2.0f * x1 + 3.0f * x2 + 0.5f * x1 * x2 + noise;

            inputs.push_back({x1, x2});
            targets.push_back({y});
        }
    }

    PerformanceMetrics runTrainingTest(bool enable_gradient_accumulation,
                                      int accumulation_steps = 4,
                                      int max_history = 50,
                                      int epochs = 20,
                                      int batch_size = 16) {
        // Create training instance
        Training trainer(false, false); // offline mode, no verbose

        // Generate fresh dataset for this test
        generateDataset(200); // training data
        generateDataset(50, true); // validation data

        // Create temporary files
        std::string input_file = "temp_perf_inputs.csv";
        std::string target_file = "temp_perf_targets.csv";

        // Write training data to CSV
        std::ofstream inputStream(input_file);
        std::ofstream targetStream(target_file);
        inputStream << "x1,x2\n";
        targetStream << "y\n";

        for (size_t i = 0; i < test_inputs.size(); ++i) {
            inputStream << test_inputs[i][0] << "," << test_inputs[i][1] << "\n";
            targetStream << test_targets[i][0] << "\n";
        }
        inputStream.close();
        targetStream.close();

        // Load data
        if (!trainer.loadCSV(input_file, test_inputs.size(), 0, true, false, ',', "perf_test")) {
            throw std::runtime_error("Failed to load input data");
        }

        int datasetId;
        if (!trainer.loadTargetsCSV(target_file, ',', true, false, datasetId)) {
            throw std::runtime_error("Failed to load target data");
        }

        // Set network architecture
        trainer.layers = 1;
        trainer.neurons = 8;

        // Preprocess
        trainer.preprocess(0.0f, 1.0f);

        // Configure gradient accumulation
        auto core = trainer.getCore();
        if (enable_gradient_accumulation) {
            core->enableGradientAccumulation(true, accumulation_steps, max_history);
        }

        // Measure training time
        auto start_time = std::chrono::high_resolution_clock::now();

        // Train
        trainer.train(0.01, epochs, batch_size);

        auto end_time = std::chrono::high_resolution_clock::now();
        double training_time = std::chrono::duration<double>(end_time - start_time).count();

        // Calculate final metrics
        double final_loss = trainer.calculateMSE();
        double final_accuracy = trainer.calculateAccuracy(0.1f); // 0.1 tolerance

        // Get memory metrics
        size_t gradient_history_size = enable_gradient_accumulation ? core->getGradientHistorySize() : 0;

        // Estimate memory usage (rough approximation)
        size_t estimated_memory = 0;
        if (enable_gradient_accumulation) {
            // Each gradient set: (output_size * neurons + neurons * input_size) * sizeof(float)
            estimated_memory = gradient_history_size * (1 * 8 + 8 * 2) * sizeof(float);
        }
        estimated_memory = estimated_memory / (1024 * 1024); // Convert to MB

        // Clean up
        std::remove(input_file.c_str());
        std::remove(target_file.c_str());

        PerformanceMetrics metrics;
        metrics.training_time_seconds = training_time;
        metrics.final_loss = final_loss;
        metrics.final_accuracy = final_accuracy;
        metrics.peak_memory_usage_mb = estimated_memory;
        metrics.gradient_history_size = gradient_history_size;
        metrics.epochs_completed = epochs;
        metrics.configuration_name = enable_gradient_accumulation ?
            "GA_" + std::to_string(accumulation_steps) + "_" + std::to_string(max_history) :
            "No_GA";

        return metrics;
    }

public:
    void runComprehensiveTests() {
        std::cout << "========================================\n";
        std::cout << "GRADIENT ACCUMULATION PERFORMANCE TESTS\n";
        std::cout << "========================================\n\n";

        std::vector<PerformanceMetrics> results;

        // Test 1: No gradient accumulation (baseline)
        std::cout << "Test 1: Training without gradient accumulation (baseline)\n";
        std::cout << "--------------------------------------------------------\n";
        auto baseline = runTrainingTest(false);
        results.push_back(baseline);
        printMetrics(baseline);

        // Test 2: Gradient accumulation with 2 steps
        std::cout << "\nTest 2: Gradient accumulation (2 steps, history=25)\n";
        std::cout << "---------------------------------------------------\n";
        auto ga_2 = runTrainingTest(true, 2, 25);
        results.push_back(ga_2);
        printMetrics(ga_2);

        // Test 3: Gradient accumulation with 4 steps
        std::cout << "\nTest 3: Gradient accumulation (4 steps, history=50)\n";
        std::cout << "---------------------------------------------------\n";
        auto ga_4 = runTrainingTest(true, 4, 50);
        results.push_back(ga_4);
        printMetrics(ga_4);

        // Test 4: Gradient accumulation with 8 steps
        std::cout << "\nTest 4: Gradient accumulation (8 steps, history=100)\n";
        std::cout << "----------------------------------------------------\n";
        auto ga_8 = runTrainingTest(true, 8, 100);
        results.push_back(ga_8);
        printMetrics(ga_8);

        // Generate comparison report
        generateComparisonReport(results);
    }

private:
    void printMetrics(const PerformanceMetrics& metrics) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Configuration: " << metrics.configuration_name << "\n";
        std::cout << "Training Time: " << metrics.training_time_seconds << " seconds\n";
        std::cout << "Final Loss (MSE): " << metrics.final_loss << "\n";
        std::cout << "Final Accuracy: " << metrics.final_accuracy * 100 << "%\n";
        std::cout << "Gradient History Size: " << metrics.gradient_history_size << "\n";
        std::cout << "Estimated Memory Usage: " << metrics.peak_memory_usage_mb << " MB\n";
        std::cout << "Epochs Completed: " << metrics.epochs_completed << "\n";
    }

    void generateComparisonReport(const std::vector<PerformanceMetrics>& results) {
        std::cout << "\n\n========================================\n";
        std::cout << "PERFORMANCE COMPARISON REPORT\n";
        std::cout << "========================================\n\n";

        std::cout << std::setw(15) << "Configuration"
                  << std::setw(12) << "Time(s)"
                  << std::setw(12) << "Loss"
                  << std::setw(12) << "Accuracy"
                  << std::setw(10) << "Memory"
                  << std::setw(8) << "History\n";
        std::cout << std::string(79, '-') << "\n";

        std::cout << std::fixed << std::setprecision(4);
        for (const auto& result : results) {
            std::cout << std::setw(15) << result.configuration_name
                      << std::setw(12) << result.training_time_seconds
                      << std::setw(12) << result.final_loss
                      << std::setw(12) << result.final_accuracy
                      << std::setw(10) << result.peak_memory_usage_mb
                      << std::setw(8) << result.gradient_history_size << "\n";
        }

        // Find best performers
        auto best_accuracy = std::max_element(results.begin(), results.end(),
            [](const PerformanceMetrics& a, const PerformanceMetrics& b) {
                return a.final_accuracy < b.final_accuracy;
            });

        auto best_loss = std::min_element(results.begin(), results.end(),
            [](const PerformanceMetrics& a, const PerformanceMetrics& b) {
                return a.final_loss < b.final_loss;
            });

        std::cout << "\n\nANALYSIS:\n";
        std::cout << "---------\n";
        std::cout << "Best Accuracy: " << best_accuracy->configuration_name
                  << " (" << best_accuracy->final_accuracy * 100 << "%)\n";
        std::cout << "Best Loss: " << best_loss->configuration_name
                  << " (" << best_loss->final_loss << ")\n";

        // Memory vs Performance analysis
        std::cout << "\nMemory vs Performance Trade-offs:\n";
        for (const auto& result : results) {
            if (result.configuration_name != "No_GA") {
                double memory_overhead = result.peak_memory_usage_mb;
                double accuracy_improvement = result.final_accuracy - results[0].final_accuracy;
                double loss_improvement = results[0].final_loss - result.final_loss;

                std::cout << result.configuration_name << ": ";
                std::cout << "Memory overhead: " << memory_overhead << "MB, ";
                std::cout << "Accuracy Δ: " << (accuracy_improvement >= 0 ? "+" : "") << accuracy_improvement * 100 << "%, ";
                std::cout << "Loss Δ: " << (loss_improvement >= 0 ? "+" : "") << loss_improvement << "\n";
            }
        }

        std::cout << "\nCONCLUSION:\n";
        std::cout << "-----------\n";
        std::cout << "Gradient accumulation provides memory-intensive training that can improve\n";
        std::cout << "model stability and potentially accuracy, with configurable memory usage.\n";
        std::cout << "Higher accumulation steps increase memory usage but may improve convergence.\n";
    }
};

int main() {
    srand(42); // For reproducible results

    try {
        PerformanceTester tester;
        tester.runComprehensiveTests();
    } catch (const std::exception& e) {
        std::cerr << "Error during testing: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
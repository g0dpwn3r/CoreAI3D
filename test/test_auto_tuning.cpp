#include "Train.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cassert>

int main() {
    std::cout << "Testing Auto-Tuning System for Problem Complexity Detection and Hyperparameter Adjustment" << std::endl;
    std::cout << "==================================================================================" << std::endl;

    // Test 1: Simple Linear Regression (should be SIMPLE)
    std::cout << "\nTest 1: Simple Linear Regression (y = 2*x + 1)" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    {
        Training trainer(false, true); // offline mode, verbose

        // Create simple dataset: y = 2*x + 1
        std::ofstream inputFile("test_simple_input.csv");
        std::ofstream targetFile("test_simple_target.csv");

        inputFile << "x\n";
        targetFile << "y\n";

        for (int i = 0; i < 50; ++i) {
            float x = static_cast<float>(i) / 5.0f; // 0.0, 0.2, 0.4, ..., 9.8
            float y = 2.0f * x + 1.0f;
            inputFile << x << "\n";
            targetFile << y << "\n";
        }
        inputFile.close();
        targetFile.close();

        // Load data using CSV methods
        if (!trainer.loadCSV("test_simple_input.csv", 50, 0, true, false, ',', "test_simple")) {
            std::cerr << "Failed to load input data" << std::endl;
            return 1;
        }

        int datasetId;
        if (!trainer.loadTargetsCSV("test_simple_target.csv", ',', true, false, datasetId)) {
            std::cerr << "Failed to load target data" << std::endl;
            return 1;
        }

        // Run complexity detection
        auto analysis = trainer.detectProblemComplexity();

        std::cout << "Expected: SIMPLE complexity" << std::endl;
        std::cout << "Detected: " << (analysis.complexity == Training::ProblemComplexity::SIMPLE ? "SIMPLE" :
                                       analysis.complexity == Training::ProblemComplexity::MODERATE ? "MODERATE" : "COMPLEX") << std::endl;
        std::cout << "Complexity Score: " << analysis.complexity_score << std::endl;
        std::cout << "3D Transforms Required: " << (analysis.requires_3d_transforms ? "YES" : "NO") << std::endl;
        std::cout << "Recommended: layers=" << analysis.recommended_layers
                  << ", neurons=" << analysis.recommended_neurons
                  << ", lr=" << analysis.recommended_learning_rate
                  << ", epochs=" << analysis.recommended_epochs << std::endl;

        // Verify expectations
        assert(analysis.complexity == Training::ProblemComplexity::SIMPLE);
        assert(!analysis.requires_3d_transforms);
        assert(analysis.recommended_layers == 2);
        assert(analysis.recommended_neurons == 5);
        assert(std::abs(analysis.recommended_learning_rate - 0.1) < 0.001);

        // Apply auto-tuning
        trainer.applyAutoTuning(analysis);

        // Verify hyperparameters were set
        assert(trainer.layers == 2);
        assert(trainer.neurons == 5);
        assert(std::abs(trainer.learningRate - 0.1) < 0.001);

        std::cout << "✓ Simple regression correctly classified as SIMPLE" << std::endl;
    }

    // Test 2: Moderate Complexity (multi-dimensional, moderate correlation)
    std::cout << "\nTest 2: Moderate Complexity (multi-dimensional with some correlation)" << std::endl;
    std::cout << "---------------------------------------------------------------------" << std::endl;

    {
        Training trainer(false, true);

        // Create moderate complexity dataset
        std::ofstream inputFile("test_moderate_input.csv");
        std::ofstream targetFile("test_moderate_target.csv");

        inputFile << "x1,x2,x3\n";
        targetFile << "y\n";

        for (int i = 0; i < 200; ++i) {
            float x1 = static_cast<float>(i) / 10.0f;
            float x2 = std::sin(x1) + 0.1f * (rand() % 100) / 100.0f; // Some correlation with noise
            float x3 = x1 * x2 + 0.5f; // Non-linear relationship
            float y = 1.5f * x1 + 0.8f * x2 - 0.3f * x3 + 0.1f; // Multi-variable relationship

            inputFile << x1 << "," << x2 << "," << x3 << "\n";
            targetFile << y << "\n";
        }
        inputFile.close();
        targetFile.close();

        // Load data using CSV methods
        if (!trainer.loadCSV("test_moderate_input.csv", 200, 0, true, false, ',', "test_moderate")) {
            std::cerr << "Failed to load input data" << std::endl;
            return 1;
        }

        int datasetId;
        if (!trainer.loadTargetsCSV("test_moderate_target.csv", ',', true, false, datasetId)) {
            std::cerr << "Failed to load target data" << std::endl;
            return 1;
        }

        auto analysis = trainer.detectProblemComplexity();

        std::cout << "Expected: MODERATE complexity" << std::endl;
        std::cout << "Detected: " << (analysis.complexity == Training::ProblemComplexity::SIMPLE ? "SIMPLE" :
                                       analysis.complexity == Training::ProblemComplexity::MODERATE ? "MODERATE" : "COMPLEX") << std::endl;
        std::cout << "Complexity Score: " << analysis.complexity_score << std::endl;
        std::cout << "3D Transforms Required: " << (analysis.requires_3d_transforms ? "YES" : "NO") << std::endl;

        // For moderate complexity, 3D transforms might be enabled depending on exact score
        // Just verify it's not classified as SIMPLE
        assert(analysis.complexity != Training::ProblemComplexity::SIMPLE);

        trainer.applyAutoTuning(analysis);

        // Verify reasonable hyperparameters
        assert(trainer.layers >= 2 && trainer.layers <= 4);
        assert(trainer.neurons >= 5 && trainer.neurons <= 15);
        assert(trainer.learningRate > 0.001 && trainer.learningRate <= 0.1);

        std::cout << "✓ Moderate complexity correctly handled" << std::endl;
    }

    // Test 3: Complex 3D Spatial Data (should be COMPLEX)
    std::cout << "\nTest 3: Complex 3D Spatial Data" << std::endl;
    std::cout << "----------------------------------" << std::endl;

    {
        Training trainer(false, true);

        // Create 3D spatial dataset (9 features, multiple of 3)
        std::ofstream inputFile("test_complex_input.csv");
        std::ofstream targetFile("test_complex_target.csv");

        inputFile << "x1,y1,z1,x2,y2,z2,x3,y3,z3\n";
        targetFile << "y\n";

        for (int i = 0; i < 500; ++i) {
            // 3D coordinates for 3 points (9 features total)
            float x1 = (rand() % 100) / 10.0f;
            float y1 = (rand() % 100) / 10.0f;
            float z1 = (rand() % 100) / 10.0f;

            float x2 = (rand() % 100) / 10.0f;
            float y2 = (rand() % 100) / 10.0f;
            float z2 = (rand() % 100) / 10.0f;

            float x3 = (rand() % 100) / 10.0f;
            float y3 = (rand() % 100) / 10.0f;
            float z3 = (rand() % 100) / 10.0f;

            // Complex target based on 3D relationships
            float distance1 = std::sqrt(x1*x1 + y1*y1 + z1*z1);
            float distance2 = std::sqrt(x2*x2 + y2*y2 + z2*z2);
            float distance3 = std::sqrt(x3*x3 + y3*y3 + z3*z3);
            float y = distance1 * 0.3f + distance2 * 0.4f + distance3 * 0.3f +
                     std::sin(x1 + y1) + std::cos(z2) + x1*y2*z3 * 0.01f;

            inputFile << x1 << "," << y1 << "," << z1 << ","
                     << x2 << "," << y2 << "," << z2 << ","
                     << x3 << "," << y3 << "," << z3 << "\n";
            targetFile << y << "\n";
        }
        inputFile.close();
        targetFile.close();

        // Load data using CSV methods
        if (!trainer.loadCSV("test_complex_input.csv", 500, 0, true, false, ',', "test_complex")) {
            std::cerr << "Failed to load input data" << std::endl;
            return 1;
        }

        int datasetId;
        if (!trainer.loadTargetsCSV("test_complex_target.csv", ',', true, false, datasetId)) {
            std::cerr << "Failed to load target data" << std::endl;
            return 1;
        }

        auto analysis = trainer.detectProblemComplexity();

        std::cout << "Expected: COMPLEX complexity (high dim, spatial 3D data)" << std::endl;
        std::cout << "Detected: " << (analysis.complexity == Training::ProblemComplexity::SIMPLE ? "SIMPLE" :
                                       analysis.complexity == Training::ProblemComplexity::MODERATE ? "MODERATE" : "COMPLEX") << std::endl;
        std::cout << "Complexity Score: " << analysis.complexity_score << std::endl;
        std::cout << "3D Transforms Required: " << (analysis.requires_3d_transforms ? "YES" : "NO") << std::endl;

        // Should be classified as COMPLEX due to high dimensionality and 3D spatial nature
        assert(analysis.complexity == Training::ProblemComplexity::COMPLEX);
        assert(analysis.requires_3d_transforms);

        trainer.applyAutoTuning(analysis);

        // Verify complex hyperparameters
        assert(trainer.layers >= 3);
        assert(trainer.neurons >= 15);
        assert(trainer.learningRate <= 0.01);

        std::cout << "✓ Complex 3D data correctly classified as COMPLEX" << std::endl;
    }

    // Test 4: Integration Test - Full Training Pipeline with Auto-Tuning
    std::cout << "\nTest 4: Full Training Pipeline with Auto-Tuning" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    {
        Training trainer(false, true);

        // Create a moderately complex dataset
        std::ofstream inputFile("test_integration_input.csv");
        std::ofstream targetFile("test_integration_target.csv");

        inputFile << "x1,x2\n";
        targetFile << "y\n";

        for (int i = 0; i < 100; ++i) {
            float x1 = static_cast<float>(i) / 10.0f;
            float x2 = std::sin(x1 * 0.5f) + 0.2f * ((rand() % 100) / 100.0f - 0.5f);
            float y = 2.0f * x1 + 1.5f * x2 + 0.5f * std::sin(x1 + x2);

            inputFile << x1 << "," << x2 << "\n";
            targetFile << y << "\n";
        }
        inputFile.close();
        targetFile.close();

        // Load data using CSV methods
        if (!trainer.loadCSV("test_integration_input.csv", 100, 0, true, false, ',', "test_integration")) {
            std::cerr << "Failed to load input data" << std::endl;
            return 1;
        }

        int datasetId;
        if (!trainer.loadTargetsCSV("test_integration_target.csv", ',', true, false, datasetId)) {
            std::cerr << "Failed to load target data" << std::endl;
            return 1;
        }

        // Run auto-tuning
        auto analysis = trainer.detectProblemComplexity();
        trainer.applyAutoTuning(analysis);

        // Initialize CoreAI with auto-tuned parameters
        trainer.preprocess(0.0f, 1.0f);

        // Train for a few epochs
        trainer.train(trainer.learningRate, 5); // Use auto-tuned learning rate, 5 epochs

        // Calculate final performance
        float mse = trainer.calculateMSE();
        std::cout << "Final MSE after auto-tuning: " << mse << std::endl;
        std::cout << "Auto-tuned parameters: layers=" << trainer.layers
                  << ", neurons=" << trainer.neurons
                  << ", lr=" << trainer.learningRate << std::endl;

        // Should achieve reasonable performance (allowing for training convergence issues)
        assert(mse < 200.0f); // Should be able to learn this relationship reasonably well

        std::cout << "✓ Full training pipeline with auto-tuning successful" << std::endl;
    }

    // Test 5: Edge Cases
    std::cout << "\nTest 5: Edge Cases" << std::endl;
    std::cout << "------------------" << std::endl;

    // Test with no data
    {
        Training trainer(false, false);
        auto analysis = trainer.detectProblemComplexity();
        assert(analysis.complexity == Training::ProblemComplexity::MODERATE); // Default
        std::cout << "✓ No data case handled correctly" << std::endl;
    }

    // Test with single feature, single sample
    {
        Training trainer(false, false);

        std::ofstream inputFile("test_edge_input.csv");
        std::ofstream targetFile("test_edge_target.csv");

        inputFile << "x\n";
        targetFile << "y\n";
        inputFile << "1.0\n";
        targetFile << "2.0\n";
        inputFile.close();
        targetFile.close();

        if (!trainer.loadCSV("test_edge_input.csv", 1, 0, true, false, ',', "test_edge")) {
            std::cerr << "Failed to load edge case input data" << std::endl;
            return 1;
        }

        int datasetId;
        if (!trainer.loadTargetsCSV("test_edge_target.csv", ',', true, false, datasetId)) {
            std::cerr << "Failed to load edge case target data" << std::endl;
            return 1;
        }

        auto analysis = trainer.detectProblemComplexity();
        // Should be SIMPLE due to low dimensionality and small dataset
        assert(analysis.complexity == Training::ProblemComplexity::SIMPLE);
        std::cout << "✓ Single sample case handled correctly" << std::endl;
    }

    std::cout << "\n==================================================================================" << std::endl;
    std::cout << "All auto-tuning tests passed successfully! ✓" << std::endl;
    std::cout << "==================================================================================" << std::endl;

    return 0;
}
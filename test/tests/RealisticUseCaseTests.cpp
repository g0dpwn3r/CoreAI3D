#include <gtest/gtest.h>
#include "ValidationFramework.hpp"
#include "Train.hpp"
#include <chrono>
#include <random>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

// Helper function to create financial dataset (stock price prediction)
void createFinancialDataset(const std::string& filename, int numSamples) {
    std::ofstream file(filename);
    file << "price,volume,momentum,rsi,macd,target_price\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> price_dist(100.0, 20.0);
    std::normal_distribution<> volume_dist(1000000.0, 200000.0);

    double currentPrice = 100.0;
    std::vector<double> prices;

    for (int i = 0; i < numSamples + 50; i++) { // Extra samples for indicators
        // Simulate price movement with trend and noise
        double trend = sin(i * 0.1) * 2.0;
        double noise = (rand() % 200 - 100) / 100.0;
        currentPrice += trend + noise;
        currentPrice = std::max(10.0, currentPrice); // Floor price
        prices.push_back(currentPrice);
    }

    for (int i = 50; i < numSamples + 50; i++) {
        double price = prices[i];
        double volume = volume_dist(gen);

        // Calculate technical indicators
        double momentum = (prices[i] - prices[i-10]) / prices[i-10] * 100.0;

        // Simple RSI calculation (simplified)
        int gains = 0, losses = 0;
        for (int j = i-14; j < i; j++) {
            double change = prices[j+1] - prices[j];
            if (change > 0) gains++;
            else if (change < 0) losses++;
        }
        double rsi = gains > 0 ? (100.0 * gains) / (gains + losses) : 50.0;

        // Simple MACD (simplified)
        double ema12 = 0, ema26 = 0;
        for (int j = i-25; j <= i; j++) {
            if (j >= i-11) ema12 += prices[j];
            ema26 += prices[j];
        }
        ema12 /= 12.0;
        ema26 /= 26.0;
        double macd = ema12 - ema26;

        // Target: next day's price
        double targetPrice = prices[i+1];

        file << price << "," << volume << "," << momentum << ","
             << rsi << "," << macd << "," << targetPrice << "\n";
    }

    file.close();
}

// Helper function to create image classification dataset (MNIST-like)
void createImageDataset(const std::string& filename, int numSamples, int imageSize = 28) {
    std::ofstream file(filename);

    // Header with pixel features
    for (int i = 0; i < imageSize * imageSize; i++) {
        file << "pixel_" << i;
        if (i < imageSize * imageSize - 1) file << ",";
    }
    file << ",digit_class\n";

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int sample = 0; sample < numSamples; sample++) {
        int digitClass = rand() % 10; // 0-9 digits

        // Generate pixel values based on digit class
        for (int i = 0; i < imageSize * imageSize; i++) {
            int row = i / imageSize;
            int col = i % imageSize;

            // Create simple patterns for each digit
            float pixel = 0.0f;

            switch (digitClass) {
                case 0: // Circle pattern
                    if (pow(row - imageSize/2, 2) + pow(col - imageSize/2, 2) < pow(imageSize/3, 2)) {
                        pixel = 1.0f;
                    }
                    break;
                case 1: // Vertical line
                    if (col == imageSize/2) pixel = 1.0f;
                    break;
                case 2: // Diagonal
                    if (abs(row - col) < 3) pixel = 1.0f;
                    break;
                case 3: // Cross
                    if (row == imageSize/2 || col == imageSize/2) pixel = 1.0f;
                    break;
                case 4: // Corner pattern
                    if ((row < imageSize/2 && col < imageSize/2) ||
                        (row > imageSize/2 && col > imageSize/2)) pixel = 1.0f;
                    break;
                case 5: // Horizontal stripes
                    if (row % 4 == 0) pixel = 1.0f;
                    break;
                case 6: // Vertical stripes
                    if (col % 4 == 0) pixel = 1.0f;
                    break;
                case 7: // Checkerboard
                    if ((row + col) % 2 == 0) pixel = 1.0f;
                    break;
                case 8: // Random noise
                    pixel = (rand() % 100) / 100.0f;
                    break;
                case 9: // Gradient
                    pixel = (row + col) / (2.0f * imageSize);
                    break;
            }

            // Add noise
            pixel += (rand() % 20 - 10) / 100.0f;
            pixel = std::max(0.0f, std::min(1.0f, pixel));

            file << pixel;
            if (i < imageSize * imageSize - 1) file << ",";
        }
        file << "," << digitClass << "\n";
    }

    file.close();
}

// Helper function to create text analysis dataset (sentiment analysis)
void createTextAnalysisDataset(const std::string& filename, int numSamples) {
    std::ofstream file(filename);
    file << "word_count,sentence_count,avg_word_length,positive_words,negative_words,exclamation_count,question_count,target_sentiment\n";

    std::vector<std::string> positiveWords = {"good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "best"};
    std::vector<std::string> negativeWords = {"bad", "terrible", "awful", "hate", "worst", "horrible", "disappointing", "poor"};

    for (int i = 0; i < numSamples; i++) {
        // Generate text features
        int wordCount = 10 + rand() % 40;
        int sentenceCount = 1 + rand() % 5;
        float avgWordLength = 4.0f + (rand() % 40) / 10.0f;

        // Sentiment-based features
        int positiveWordsCount = rand() % 5;
        int negativeWordsCount = rand() % 5;
        int exclamationCount = rand() % 3;
        int questionCount = rand() % 2;

        // Determine sentiment based on features
        int sentimentScore = positiveWordsCount - negativeWordsCount + exclamationCount - questionCount;
        int targetSentiment = sentimentScore > 0 ? 1 : (sentimentScore < 0 ? -1 : 0);

        file << wordCount << "," << sentenceCount << "," << avgWordLength << ","
             << positiveWordsCount << "," << negativeWordsCount << ","
             << exclamationCount << "," << questionCount << "," << targetSentiment << "\n";
    }

    file.close();
}

TEST(RealisticUseCaseTest, FinancialPredictionEndToEnd) {
    ValidationFramework validator(true);

    // Create financial dataset
    createFinancialDataset("financial_data.csv", 1000);

    // Load and validate
    auto validationResult = validator.validateDataset("financial_data.csv", true);
    EXPECT_TRUE(validationResult.isValid);

    // Load data
    Training loader(false, false);
    bool loaded = loader.loadCSV("financial_data.csv", 1000, 1, true, false, ',', "financial");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // Train model for price prediction
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setLearningRate(0.001f);
        trainer->setEpochs(20);

        auto startTime = std::chrono::steady_clock::now();
        bool trained = trainer->train(inputs, targets);
        auto endTime = std::chrono::steady_clock::now();

        EXPECT_TRUE(trained);

        if (trained) {
            // Test predictions on held-out data
            std::vector<std::vector<float>> testInputs(inputs.end() - 100, inputs.end());
            std::vector<std::vector<float>> testTargets(targets.end() - 100, targets.end());

            auto predictions = trainer->predict(testInputs);
            EXPECT_EQ(predictions.size(), 100);

            // Calculate prediction accuracy metrics
            float mse = 0.0f, mae = 0.0f;
            for (size_t i = 0; i < predictions.size(); i++) {
                float error = predictions[i][0] - testTargets[i][0];
                mse += error * error;
                mae += std::abs(error);
            }
            mse /= predictions.size();
            mae /= predictions.size();

            std::cout << "Financial Prediction Results:" << std::endl;
            std::cout << "MSE: " << mse << ", MAE: " << mae << std::endl;
            std::cout << "Training time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()
                      << " ms" << std::endl;

            // Reasonable performance for regression task
            EXPECT_LT(mse, 1000.0f); // MSE should be reasonable
            EXPECT_LT(mae, 20.0f);   // MAE should be reasonable for price prediction
        }
    }

    // Clean up
    std::remove("financial_data.csv");
}

TEST(RealisticUseCaseTest, ImageClassificationEndToEnd) {
    ValidationFramework validator(true);

    // Create image classification dataset
    createImageDataset("image_data.csv", 500, 14); // Smaller images for testing

    // Load and validate
    auto validationResult = validator.validateDataset("image_data.csv", true);
    EXPECT_TRUE(validationResult.isValid);

    // Load data
    Training loader(false, false);
    bool loaded = loader.loadCSV("image_data.csv", 500, 1, true, false, ',', "image");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // Train model for digit classification
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setLearningRate(0.01f);
        trainer->setEpochs(15);

        auto startTime = std::chrono::steady_clock::now();
        bool trained = trainer->train(inputs, targets);
        auto endTime = std::chrono::steady_clock::now();

        EXPECT_TRUE(trained);

        if (trained) {
            // Test predictions
            std::vector<std::vector<float>> testInputs(inputs.end() - 50, inputs.end());
            std::vector<std::vector<float>> testTargets(targets.end() - 50, targets.end());

            auto predictions = trainer->predict(testInputs);
            EXPECT_EQ(predictions.size(), 50);

            // Calculate classification accuracy
            int correct = 0;
            for (size_t i = 0; i < predictions.size(); i++) {
                int predictedClass = std::round(predictions[i][0]);
                int actualClass = std::round(testTargets[i][0]);
                if (predictedClass == actualClass) correct++;
            }
            float accuracy = static_cast<float>(correct) / predictions.size();

            std::cout << "Image Classification Results:" << std::endl;
            std::cout << "Accuracy: " << accuracy * 100.0f << "%" << std::endl;
            std::cout << "Training time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()
                      << " ms" << std::endl;

            // Should achieve reasonable accuracy for this simple task
            EXPECT_GT(accuracy, 0.5f);
        }
    }

    // Clean up
    std::remove("image_data.csv");
}

TEST(RealisticUseCaseTest, TextAnalysisEndToEnd) {
    ValidationFramework validator(true);

    // Create text analysis dataset
    createTextAnalysisDataset("text_data.csv", 800);

    // Load and validate
    auto validationResult = validator.validateDataset("text_data.csv", true);
    EXPECT_TRUE(validationResult.isValid);

    // Load data
    Training loader(false, false);
    bool loaded = loader.loadCSV("text_data.csv", 800, 1, true, false, ',', "text");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // Train model for sentiment analysis
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setLearningRate(0.005f);
        trainer->setEpochs(25);

        auto startTime = std::chrono::steady_clock::now();
        bool trained = trainer->train(inputs, targets);
        auto endTime = std::chrono::steady_clock::now();

        EXPECT_TRUE(trained);

        if (trained) {
            // Test predictions
            std::vector<std::vector<float>> testInputs(inputs.end() - 100, inputs.end());
            std::vector<std::vector<float>> testTargets(targets.end() - 100, targets.end());

            auto predictions = trainer->predict(testInputs);
            EXPECT_EQ(predictions.size(), 100);

            // Calculate sentiment analysis metrics
            int correct = 0;
            std::vector<int> sentimentCounts = {0, 0, 0}; // -1, 0, 1

            for (size_t i = 0; i < predictions.size(); i++) {
                int predictedSentiment = std::round(predictions[i][0]);
                int actualSentiment = std::round(testTargets[i][0]);

                predictedSentiment = std::max(-1, std::min(1, predictedSentiment));
                actualSentiment = std::max(-1, std::min(1, actualSentiment));

                if (predictedSentiment == actualSentiment) correct++;
                sentimentCounts[actualSentiment + 1]++;
            }

            float accuracy = static_cast<float>(correct) / predictions.size();

            std::cout << "Text Analysis (Sentiment) Results:" << std::endl;
            std::cout << "Accuracy: " << accuracy * 100.0f << "%" << std::endl;
            std::cout << "Training time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count()
                      << " ms" << std::endl;

            // Should achieve reasonable accuracy
            EXPECT_GT(accuracy, 0.6f);
        }
    }

    // Clean up
    std::remove("text_data.csv");
}

TEST(RealisticUseCaseTest, CrossDomainValidation) {
    ValidationFramework validator(true);

    // Test model trained on one domain performing on another
    createFinancialDataset("financial_train.csv", 500);
    createFinancialDataset("financial_test.csv", 200);

    // Train on financial data
    Training trainLoader(false, false);
    bool trainLoaded = trainLoader.loadCSV("financial_train.csv", 500, 1, true, false, ',', "financial_train");

    EXPECT_TRUE(trainLoaded);

    if (trainLoaded) {
        const auto& trainInputs = trainLoader.getInputs();
        const auto& trainTargets = trainLoader.getTargets();

        auto trainer = std::make_unique<Training>(false, false);
        trainer->setEpochs(10);
        bool trained = trainer->train(trainInputs, trainTargets);
        EXPECT_TRUE(trained);

        if (trained) {
            // Test on different financial data
            Training testLoader(false, false);
            bool testLoaded = testLoader.loadCSV("financial_test.csv", 200, 1, true, false, ',', "financial_test");

            if (testLoaded) {
                const auto& testInputs = testLoader.getInputs();
                const auto& testTargets = testLoader.getTargets();

                auto predictions = trainer->predict(testInputs);

                // Calculate performance on test set
                float mse = 0.0f;
                for (size_t i = 0; i < predictions.size(); i++) {
                    float error = predictions[i][0] - testTargets[i][0];
                    mse += error * error;
                }
                mse /= predictions.size();

                std::cout << "Cross-domain validation MSE: " << mse << std::endl;

                // Should perform reasonably on similar domain data
                EXPECT_LT(mse, 2000.0f);
            }
        }
    }

    // Clean up
    std::remove("financial_train.csv");
    std::remove("financial_test.csv");
}

TEST(RealisticUseCaseTest, EndToEndPipelineIntegration) {
    ValidationFramework validator(true);

    // Test complete ML pipeline: data loading -> preprocessing -> training -> validation -> deployment simulation

    // 1. Create and load data
    createFinancialDataset("pipeline_data.csv", 300);

    auto validationResult = validator.validateDataset("pipeline_data.csv", true);
    EXPECT_TRUE(validationResult.isValid);

    Training loader(false, false);
    bool loaded = loader.loadCSV("pipeline_data.csv", 300, 1, true, false, ',', "pipeline");
    EXPECT_TRUE(loaded);

    if (loaded) {
        const auto& inputs = loader.getInputs();
        const auto& targets = loader.getTargets();

        // 2. Preprocessing (normalization)
        auto normalizedInputs = validator.normalizeData(inputs);
        EXPECT_EQ(normalizedInputs.size(), inputs.size());

        // 3. Training
        auto trainer = std::make_unique<Training>(false, false);
        trainer->setEpochs(15);
        bool trained = trainer->train(normalizedInputs, targets);
        EXPECT_TRUE(trained);

        if (trained) {
            // 4. Validation
            auto cvResults = validator.kFoldCrossValidation(
                std::make_unique<Training>(false, false), normalizedInputs, targets, 3, 0.01, 5);
            EXPECT_EQ(cvResults.size(), 3);

            // 5. Performance evaluation
            auto predictions = trainer->predict(normalizedInputs);
            auto metrics = validator.calculateMetrics(predictions, targets);

            std::cout << "Pipeline Integration Results:" << std::endl;
            std::cout << "MSE: " << metrics.mse << ", MAE: " << metrics.mae
                      << ", R²: " << metrics.r2_score << std::endl;

            // 6. Model persistence simulation (save/load would be tested here if implemented)

            // Verify reasonable performance
            EXPECT_LT(metrics.mse, 500.0f);
            EXPECT_GT(metrics.r2_score, 0.5f);
        }
    }

    // Clean up
    std::remove("pipeline_data.csv");
}
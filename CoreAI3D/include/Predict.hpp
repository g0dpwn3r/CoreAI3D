#pragma once

#include "CoreAI3DCommon.hpp"
#include "Train.hpp"
#include "Ensemble.hpp"
#include <string>
#include <vector>
#include <memory>

class WebSocketServer;

class Predict {
public:
    Predict(bool verbose = false, WebSocketServer* wsServer = nullptr);
    ~Predict();

    // Main prediction execution method
    bool runPrediction(const std::string& inputFile,
                       const std::string& outputCsvFile,
                       const std::string& targetFile = "",
                       const std::string& loadModelId = "",
                       int datasetId = -1,
                       int layers = 3,
                       int neurons = 10,
                       float minRange = 0.0f,
                       float maxRange = 1.0f,
                       int inputSize = 1,
                       int outputSize = 1,
                       int numSamples = 0,
                       int batchSize = 32,
                       double learningRate = 0.01,
                       int epochs = 10,
                       const std::string& embeddingFile = "embedding.txt",
                       const std::string& language = "en",
                       const std::string& datasetName = "online-1a",
                       bool hasHeader = true,
                       bool containsText = false,
                       char delimiter = ',',
                       bool isOfflineMode = false,
                       const std::string& dbHost = "",
                       int dbPort = 0,
                       const std::string& dbUser = "",
                       const std::string& dbPassword = "",
                       const std::string& dbSchema = "");

    // Ensemble prediction method
    // bool runEnsemblePrediction(const std::string& inputFile,
    //                            const std::string& outputCsvFile,
    //                            const std::string& targetFile = "",
    //                            int datasetId = -1,
    //                            int layers = 3,
    //                            int neurons = 10,
    //                            float minRange = 0.0f,
    //                            float maxRange = 1.0f,
    //                            int inputSize = 1,
    //                            int outputSize = 1,
    //                            int numSamples = 0,
    //                            double learningRate = 0.01,
    //                            int epochs = 10,
    //                            const std::string& datasetName = "online-1a",
    //                            bool hasHeader = true,
    //                            char delimiter = ',',
    //                            EnsembleMethod ensembleMethod = EnsembleMethod::SIMPLE_AVERAGING,
    //                            int numEnsembleModels = 5,
    //                            const std::string& dbHost = "",
    //                            int dbPort = 0,
    //                            const std::string& dbUser = "",
    //                            const std::string& dbPassword = "",
    //                            const std::string& dbSchema = "");

    // Get evaluation metrics
    float getRMSE() const { return rmse_; }
    float getMSE() const { return mse_; }
    float getAccuracy() const { return accuracy_; }
    double getTrainingTime() const { return trainingTime_; }

private:
    bool verbose_;
    Training* trainer_;
    WebSocketServer* webSocketServer_;
    // std::unique_ptr<Ensemble> ensemble_;

    // Evaluation results
    float rmse_;
    float mse_;
    float accuracy_;
    double trainingTime_;

    // Helper methods
    bool initializeTrainer(const std::string& dbHost,
                           int dbPort,
                           const std::string& dbUser,
                           const std::string& dbPassword,
                           const std::string& dbSchema,
                           bool isOfflineMode);
    bool loadData(const std::string& inputFile,
                  const std::string& targetFile,
                  int numSamples,
                  int outputSize,
                  bool hasHeader,
                  bool containsText,
                  char delimiter,
                  const std::string& datasetName);
    bool loadModel(const std::string& loadModelId, int datasetId);
    bool preprocessData(float minRange, float maxRange);
    bool trainModel(double learningRate, int epochs, int batchSize);
    bool evaluateModel();
    bool saveResults(const std::string& outputCsvFile,
                     const std::string& inputFile,
                     bool hasHeader,
                     char delimiter);
    // bool saveEnsembleResults(const std::string& outputCsvFile,
    //                         const std::string& inputFile,
    //                         bool hasHeader,
    //                         char delimiter);
};
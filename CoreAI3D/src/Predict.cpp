#include "Predict.hpp"
#include "Train.hpp"
#include <iostream>
#include <chrono>
#include <stdexcept>

Predict::Predict(bool verbose, WebSocketServer* wsServer)
    : verbose_(verbose), trainer_(nullptr), webSocketServer_(wsServer),
      rmse_(0.0f), mse_(0.0f), accuracy_(0.0f), trainingTime_(0.0) {
}

Predict::~Predict() {
    if (trainer_) {
        delete trainer_;
        trainer_ = nullptr;
    }
}

bool Predict::runPrediction(const std::string& inputFile,
                           const std::string& outputCsvFile,
                           const std::string& targetFile,
                           const std::string& loadModelId,
                           int datasetId,
                           int layers,
                           int neurons,
                           float minRange,
                           float maxRange,
                           int inputSize,
                           int outputSize,
                           int numSamples,
                           double learningRate,
                           int epochs,
                           const std::string& embeddingFile,
                           const std::string& language,
                           const std::string& datasetName,
                           bool hasHeader,
                           bool containsText,
                           char delimiter,
                           bool isOfflineMode,
                           const std::string& dbHost,
                           int dbPort,
                           const std::string& dbUser,
                           const std::string& dbPassword,
                           const std::string& dbSchema) {
    try {
        if (verbose_) {
            std::cout << "[PREDICT] Starting prediction process..." << std::endl;
        }

        // Initialize trainer with database if parameters provided
        if (!initializeTrainer(dbHost, dbPort, dbUser, dbPassword, dbSchema, isOfflineMode)) {
            std::cerr << "[PREDICT] Failed to initialize trainer." << std::endl;
            return false;
        }

        bool useOnlineMode = !trainer_->isOfflineMode;
        if (verbose_) {
            std::cout << "[PREDICT] Running in " << (useOnlineMode ? "online" : "offline") << " mode" << std::endl;
        }

        // Set training parameters
        trainer_->layers = layers;
        trainer_->embedding_file = embeddingFile;
        trainer_->language = language;
        trainer_->neurons = neurons;
        trainer_->min = minRange;
        trainer_->max = maxRange;
        trainer_->outputSize = outputSize;
        trainer_->inputSize = inputSize;
        trainer_->numSamples = numSamples;

        // Load data
        if (!loadData(inputFile, targetFile, numSamples, outputSize, hasHeader, containsText, delimiter, datasetName)) {
            std::cerr << "[PREDICT] Failed to load data." << std::endl;
            return false;
        }

        // Load model if specified
        if (!loadModelId.empty() || datasetId != -1) {
            if (!loadModel(loadModelId, datasetId)) {
                std::cerr << "[PREDICT] Failed to load model, proceeding with new model initialization." << std::endl;
            }
        }

        // Preprocess data
        if (!preprocessData(minRange, maxRange)) {
            std::cerr << "[PREDICT] Failed to preprocess data." << std::endl;
            return false;
        }

        // Train model
        if (!trainModel(learningRate, epochs)) {
            std::cerr << "[PREDICT] Failed to train model." << std::endl;
            return false;
        }

#ifdef USE_MYSQL
        // Save model if in online mode
        if (useOnlineMode && datasetId != -1) {
            if (verbose_) {
                std::cout << "[PREDICT] Saving trained model to database..." << std::endl;
            }
            if (!trainer_->saveModel(datasetId)) {
                std::cerr << "[PREDICT] Failed to save model to database." << std::endl;
            }
        } else if (useOnlineMode && trainer_->currentDatasetId != -1) {
             // If datasetId was not passed but created during loadData
             int currentId = trainer_->currentDatasetId;
             if (verbose_) {
                std::cout << "[PREDICT] Saving trained model to database (ID: " << currentId << ")..." << std::endl;
            }
             if (!trainer_->saveModel(currentId)) {
                std::cerr << "[PREDICT] Failed to save model to database." << std::endl;
            }
        }
#endif

        // Evaluate model
        if (!evaluateModel()) {
            std::cerr << "[PREDICT] Failed to evaluate model." << std::endl;
            return false;
        }

        // Save results
        if (!saveResults(outputCsvFile, inputFile, hasHeader, delimiter)) {
            std::cerr << "[PREDICT] Failed to save results." << std::endl;
            return false;
        }

        if (verbose_) {
            std::cout << "[PREDICT] Prediction process completed successfully." << std::endl;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[PREDICT] Exception during prediction: " << e.what() << std::endl;
        return false;
    }
}

bool Predict::initializeTrainer(const std::string& dbHost,
                                int dbPort,
                                const std::string& dbUser,
                                const std::string& dbPassword,
                                const std::string& dbSchema,
                                bool isOfflineMode) {
    try {
        bool hasDbParams = !dbHost.empty() && !dbUser.empty() && !dbSchema.empty() && !isOfflineMode;

        if (hasDbParams) {
            // Try online mode with database first (but MySQL is disabled, so skip)
            std::cerr << "[PREDICT] MySQL support is disabled, skipping database connection attempt." << std::endl;
            std::cerr << "[PREDICT] Using offline mode." << std::endl;
            // Fall through to offline mode
        }

        // Offline mode (fallback or when no database params provided)
        trainer_ = new Training(true, verbose_, webSocketServer_);
        if (verbose_) {
            std::cout << "[PREDICT] Trainer initialized in offline mode." << std::endl;
        }
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[PREDICT] Failed to initialize trainer in offline mode: " << e.what() << std::endl;
        trainer_ = nullptr; // Ensure trainer_ is null on failure
        return false;
    }
}

bool Predict::loadData(const std::string& inputFile,
                      const std::string& targetFile,
                      int numSamples,
                      int outputSize,
                      bool hasHeader,
                      bool containsText,
                      char delimiter,
                      const std::string& datasetName) {
    if (verbose_) {
        std::cout << "[PREDICT] Loading data from: " << inputFile << std::endl;
    }

    // Load main input data
    if (!trainer_->loadCSV(inputFile, numSamples, outputSize, hasHeader, containsText, delimiter, datasetName)) {
        std::cerr << "[PREDICT] Failed to load CSV data from " << inputFile << std::endl;
        return false;
    }

    // Load target data if provided
    if (!targetFile.empty()) {
        if (verbose_) {
            std::cout << "[PREDICT] Loading targets from: " << targetFile << std::endl;
        }
        if (!trainer_->loadTargetsCSV(targetFile, delimiter, hasHeader, containsText, trainer_->currentDatasetId)) {
            std::cerr << "[PREDICT] Failed to load target CSV data from " << targetFile << std::endl;
            return false;
        }
    }

    if (verbose_) {
        std::cout << "[PREDICT] Data loaded successfully." << std::endl;
    }
    return true;
}

bool Predict::loadModel(const std::string& loadModelId, int datasetId) {
    if (verbose_) {
        std::cout << "[PREDICT] Attempting to load model..." << std::endl;
    }

    bool modelLoaded = false;
    int modelIdToLoad = datasetId != -1 ? datasetId : std::stoi(loadModelId);

    if (verbose_) {
        std::cout << "[PREDICT] Using model ID " << modelIdToLoad << std::endl;
    }

    // Try to load dataset from database
    if (trainer_->loadDatasetFromDB(modelIdToLoad)) {
        if (verbose_) {
            std::cout << "[PREDICT] Dataset loaded from database with " << trainer_->numSamples << " samples." << std::endl;
        }
    } else {
        std::cerr << "[PREDICT] Failed to load dataset from database." << std::endl;
        return false;
    }

#ifdef USE_MYSQL
    try {
        if (!loadModelId.empty()) {
            modelLoaded = trainer_->loadModel(loadModelId); // Load by name
        } else {
            modelLoaded = trainer_->loadModel(modelIdToLoad); // Load by ID
        }
    } catch (const std::exception& e) {
        std::cerr << "[PREDICT] Exception during model loading: " << e.what() << std::endl;
        modelLoaded = false;
    }

    if (modelLoaded) {
        if (verbose_) {
            std::cout << "[PREDICT] Model loaded from database for ID " << modelIdToLoad << std::endl;
        }
    } else {
        std::cerr << "[PREDICT] Failed to load model from database." << std::endl;
    }
#endif

    return modelLoaded;
}

bool Predict::preprocessData(float minRange, float maxRange) {
    if (verbose_) {
        std::cout << "[PREDICT] Preprocessing data..." << std::endl;
    }

    try {
        trainer_->preprocess(minRange, maxRange);
        if (verbose_) {
            std::cout << "[PREDICT] Data preprocessing complete." << std::endl;
            std::cout << "[PREDICT] After preprocessing - numSamples: " << trainer_->numSamples
                      << ", inputSize: " << trainer_->inputSize << ", outputSize: " << trainer_->outputSize << std::endl;
            if (!trainer_->getTargets().empty()) {
                std::cout << "[PREDICT] First target value: " << trainer_->getTargets()[0][0] << std::endl;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[PREDICT] Failed to preprocess data: " << e.what() << std::endl;
        return false;
    }
}

bool Predict::trainModel(double learningRate, int epochs) {
    if (verbose_) {
        std::cout << "[PREDICT] Starting model training..." << std::endl;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        trainer_->train(learningRate, epochs);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> training_duration = end_time - start_time;
        trainingTime_ = training_duration.count();

        if (verbose_) {
            std::cout << "[PREDICT] Model training complete." << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[PREDICT] Failed to train model: " << e.what() << std::endl;
        return false;
    }
}

bool Predict::evaluateModel() {
    if (verbose_) {
        std::cout << "[PREDICT] Evaluating model..." << std::endl;
    }

    try {
        rmse_ = trainer_->calculateRMSE();
        mse_ = trainer_->calculateMSE();
        accuracy_ = trainer_->calculateAccuracy();

        if (verbose_) {
            std::cout << "\n[PREDICT] Evaluation Metrics:" << std::endl;
            std::cout << "RMSE: " << rmse_ << std::endl;
            std::cout << "MSE: " << mse_ << std::endl;
            std::cout << "Accuracy: " << accuracy_ * 100.0f << "%" << std::endl;
            std::cout << "Training Execution Time: " << trainingTime_ << " seconds" << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[PREDICT] Failed to evaluate model: " << e.what() << std::endl;
        return false;
    }
}

bool Predict::saveResults(const std::string& outputCsvFile,
                         const std::string& inputFile,
                         bool hasHeader,
                         char delimiter) {
    if (verbose_) {
        std::cout << "[PREDICT] Saving prediction results to: " << outputCsvFile << std::endl;
    }

    try {
        if (trainer_->saveResultsToCSV(outputCsvFile, inputFile, hasHeader, delimiter)) {
            if (verbose_) {
                std::cout << "[PREDICT] Results saved successfully." << std::endl;
            }
            return true;
        } else {
            std::cerr << "[PREDICT] Failed to save results to CSV." << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "[PREDICT] Failed to save results: " << e.what() << std::endl;
        return false;
    }
}
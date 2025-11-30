#include "Predict.hpp"
#include "Train.hpp"
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <fstream>

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
                           int batchSize,
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

        bool useOnlineMode = !trainer_->getIsOfflineMode();
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
        trainer_->outputSize = outputSize;  // Fixed to 1 for regression
        trainer_->inputSize = inputSize;    // Will be set dynamically if 0
        trainer_->numSamples = numSamples;

        // Load data
        if (!loadData(inputFile, targetFile, numSamples, outputSize, hasHeader, containsText, delimiter, datasetName)) {
            std::cerr << "[PREDICT] Failed to load data." << std::endl;
            return false;
        }

        // Perform automatic complexity detection and hyperparameter tuning
        if (verbose_) {
            std::cout << "[PREDICT] Performing automatic complexity detection and hyperparameter tuning..." << std::endl;
        }
        auto complexityAnalysis = trainer_->detectProblemComplexity();
        trainer_->applyAutoTuning(complexityAnalysis);

        // Override user-provided hyperparameters with auto-tuned values if not explicitly set
        // (This allows users to override auto-tuning by providing specific values)
        if (layers == 3 && neurons == 10 && learningRate == 0.01) { // Default values
            layers = complexityAnalysis.recommended_layers;
            neurons = complexityAnalysis.recommended_neurons;
            learningRate = complexityAnalysis.recommended_learning_rate;
            epochs = complexityAnalysis.recommended_epochs;

            if (verbose_) {
                std::cout << "[PREDICT] Auto-tuning applied: layers=" << layers
                          << ", neurons=" << neurons << ", learning_rate=" << learningRate
                          << ", epochs=" << epochs << std::endl;
            }
        } else if (verbose_) {
            std::cout << "[PREDICT] Using user-provided hyperparameters (auto-tuning recommendations available)" << std::endl;
        }

        // Update trainer parameters with final values
        trainer_->layers = layers;
        trainer_->neurons = neurons;
        trainer_->learningRate = learningRate;

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
        if (!trainModel(learningRate, epochs, batchSize)) {
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
        } else if (useOnlineMode && trainer_->getCurrentDatasetId() != -1) {
             // If datasetId was not passed but created during loadData
             int currentId = trainer_->getCurrentDatasetId();
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
#ifdef USE_MYSQL

        if (hasDbParams) {
            // Try online mode with database first
            try {
                trainer_ = new Training(dbHost, dbPort, dbUser, dbPassword, dbSchema, 0, true, verbose_, webSocketServer_);
                if (verbose_) {
                    std::cout << "[PREDICT] Trainer initialized in online mode with database." << std::endl;
                }
                return true;
            } catch (const std::exception& dbEx) {
                std::cerr << "[PREDICT] Database connection failed: " << dbEx.what() << std::endl;
                std::cerr << "[PREDICT] Falling back to offline mode." << std::endl;
                // Fall through to offline mode
            }
        }
#endif

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

    // Check if input file exists
    std::ifstream checkFile(inputFile);
    if (!checkFile.good()) {
        std::cerr << "[PREDICT] ERROR: Input file does not exist or cannot be opened: " << inputFile << std::endl;
        return false;
    }
    checkFile.close();

    if (verbose_) {
        std::cout << "[PREDICT] Input file exists and is accessible: " << inputFile << std::endl;
    }

    // Determine outputSizeParam for inputFile loading
    // If targetFile is provided, inputFile contains only features (outputSizeParam=0)
    // If no targetFile, inputFile contains features + target (outputSizeParam=1)
    int outputSizeParam = targetFile.empty() ? 1 : 0;

    if (verbose_) {
        std::cout << "[PREDICT] Loading CSV with outputSizeParam=" << outputSizeParam
                  << ", numSamples=" << numSamples << ", hasHeader=" << hasHeader
                  << ", containsText=" << containsText << ", delimiter='" << delimiter << "'" << std::endl;
    }

    if (!trainer_->loadCSV(inputFile, numSamples, outputSizeParam, hasHeader, containsText, delimiter, datasetName)) {
        std::cerr << "[PREDICT] Failed to load CSV data from " << inputFile << std::endl;
        return false;
    }

    if (verbose_) {
        std::cout << "[PREDICT] After loadCSV: numSamples=" << trainer_->numSamples
                  << ", inputSize=" << trainer_->inputSize << ", outputSize=" << trainer_->outputSize << std::endl;
        std::cout << "[PREDICT] Inputs vector size: " << trainer_->getInputs().size() << std::endl;
        std::cout << "[PREDICT] Targets vector size: " << trainer_->getTargets().size() << std::endl;
        if (!trainer_->getInputs().empty()) {
            std::cout << "[PREDICT] First input sample size: " << trainer_->getInputs()[0].size() << std::endl;
            if (!trainer_->getInputs()[0].empty()) {
                std::cout << "[PREDICT] First few input values: "
                          << trainer_->getInputs()[0][0];
                for (size_t i = 1; i < std::min(size_t(3), trainer_->getInputs()[0].size()); ++i) {
                    std::cout << ", " << trainer_->getInputs()[0][i];
                }
                std::cout << std::endl;
            }
        }
    }

    // For regression, outputSize is always 1
    trainer_->outputSize = 1;
    // inputSize is set by loadCSV to columns in inputFile (since outputSizeParam=0 when targetFile provided)

    // Load target data if provided
    if (!targetFile.empty()) {
        if (verbose_) {
            std::cout << "[PREDICT] Loading targets from: " << targetFile << std::endl;
        }

        // Check if target file exists
        std::ifstream checkTargetFile(targetFile);
        if (!checkTargetFile.good()) {
            std::cerr << "[PREDICT] ERROR: Target file does not exist or cannot be opened: " << targetFile << std::endl;
            return false;
        }
        checkTargetFile.close();

        int datasetId = trainer_->getCurrentDatasetId();
        if (!trainer_->loadTargetsCSV(targetFile, delimiter, hasHeader, containsText, datasetId)) {
            std::cerr << "[PREDICT] Failed to load target CSV data from " << targetFile << std::endl;
            return false;
        }

        if (verbose_) {
            std::cout << "[PREDICT] After loading targets: Targets vector size: " << trainer_->getTargets().size() << std::endl;
            if (!trainer_->getTargets().empty()) {
                std::cout << "[PREDICT] First target sample size: " << trainer_->getTargets()[0].size() << std::endl;
                if (!trainer_->getTargets()[0].empty()) {
                    std::cout << "[PREDICT] First few target values: "
                              << trainer_->getTargets()[0][0];
                    for (size_t i = 1; i < std::min(size_t(3), trainer_->getTargets()[0].size()); ++i) {
                        std::cout << ", " << trainer_->getTargets()[0][i];
                    }
                    std::cout << std::endl;
                }
            }
        }
    }

    if (verbose_) {
        std::cout << "[PREDICT] Data loaded successfully. Input size: " << trainer_->inputSize
                  << ", Output size: " << trainer_->outputSize
                  << ", Final inputs size: " << trainer_->getInputs().size()
                  << ", Final targets size: " << trainer_->getTargets().size() << std::endl;
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
        std::cout << "[PREDICT] Before preprocessing - numSamples: " << trainer_->numSamples
                  << ", inputSize: " << trainer_->inputSize << ", outputSize: " << trainer_->outputSize << std::endl;
        std::cout << "[PREDICT] Before preprocessing - inputs size: " << trainer_->getInputs().size()
                  << ", targets size: " << trainer_->getTargets().size() << std::endl;
        if (!trainer_->getInputs().empty() && !trainer_->getInputs()[0].empty()) {
            std::cout << "[PREDICT] Before preprocessing - first input values: "
                      << trainer_->getInputs()[0][0];
            for (size_t i = 1; i < std::min(size_t(3), trainer_->getInputs()[0].size()); ++i) {
                std::cout << ", " << trainer_->getInputs()[0][i];
            }
            std::cout << std::endl;
        }
        if (!trainer_->getTargets().empty() && !trainer_->getTargets()[0].empty()) {
            std::cout << "[PREDICT] Before preprocessing - first target values: "
                      << trainer_->getTargets()[0][0];
            for (size_t i = 1; i < std::min(size_t(3), trainer_->getTargets()[0].size()); ++i) {
                std::cout << ", " << trainer_->getTargets()[0][i];
            }
            std::cout << std::endl;
        }
    }

    try {
        trainer_->preprocess(minRange, maxRange);
        if (verbose_) {
            std::cout << "[PREDICT] Data preprocessing complete." << std::endl;
            std::cout << "[PREDICT] After preprocessing - numSamples: " << trainer_->numSamples
                      << ", inputSize: " << trainer_->inputSize << ", outputSize: " << trainer_->outputSize << std::endl;
            std::cout << "[PREDICT] After preprocessing - inputs size: " << trainer_->getInputs().size()
                      << ", targets size: " << trainer_->getTargets().size() << std::endl;
            if (!trainer_->getInputs().empty() && !trainer_->getInputs()[0].empty()) {
                std::cout << "[PREDICT] After preprocessing - first input values: "
                          << trainer_->getInputs()[0][0];
                for (size_t i = 1; i < std::min(size_t(3), trainer_->getInputs()[0].size()); ++i) {
                    std::cout << ", " << trainer_->getInputs()[0][i];
                }
                std::cout << std::endl;
            }
            if (!trainer_->getTargets().empty() && !trainer_->getTargets()[0].empty()) {
                std::cout << "[PREDICT] After preprocessing - first target values: "
                          << trainer_->getTargets()[0][0];
                for (size_t i = 1; i < std::min(size_t(3), trainer_->getTargets()[0].size()); ++i) {
                    std::cout << ", " << trainer_->getTargets()[0][i];
                }
                std::cout << std::endl;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[PREDICT] Failed to preprocess data: " << e.what() << std::endl;
        return false;
    }
}

bool Predict::trainModel(double learningRate, int epochs, int batchSize) {
    if (verbose_) {
        std::cout << "[PREDICT] Starting model training..." << std::endl;
    }

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        trainer_->train(learningRate, epochs, batchSize);
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
// bool Predict::runEnsemblePrediction(const std::string& inputFile,
//                                    const std::string& outputCsvFile,
//                                    const std::string& targetFile,
//                                    int datasetId,
//                                    int layers,
//                                    int neurons,
//                                    float minRange,
//                                    float maxRange,
//                                    int inputSize,
//                                    int outputSize,
//                                    int numSamples,
//                                    double learningRate,
//                                    int epochs,
//                                    const std::string& datasetName,
//                                    bool hasHeader,
//                                    char delimiter,
//                                    EnsembleMethod ensembleMethod,
//                                    int numEnsembleModels,
//                                    const std::string& dbHost,
//                                    int dbPort,
//                                    const std::string& dbUser,
//                                    const std::string& dbPassword,
//                                    const std::string& dbSchema) {
//     try {
//         if (verbose_) {
//             std::cout << "[ENSEMBLE PREDICT] Starting ensemble prediction process..." << std::endl;
//         }

//         // Initialize trainer (offline mode for ensemble)
//         if (!initializeTrainer(dbHost, dbPort, dbUser, dbPassword, dbSchema, true)) {
//             std::cerr << "[ENSEMBLE PREDICT] Failed to initialize trainer." << std::endl;
//             return false;
//         }

//         // Set training parameters
//         trainer_->layers = layers;
//         trainer_->neurons = neurons;
//         trainer_->min = minRange;
//         trainer_->max = maxRange;
//         trainer_->outputSize = outputSize;
//         trainer_->inputSize = inputSize;
//         trainer_->numSamples = numSamples;

//         // Load data
//         if (!loadData(inputFile, targetFile, numSamples, outputSize, hasHeader, false, delimiter, datasetName)) {
//             std::cerr << "[ENSEMBLE PREDICT] Failed to load data." << std::endl;
//             return false;
//         }

//         // Preprocess data
//         if (!preprocessData(minRange, maxRange)) {
//             std::cerr << "[ENSEMBLE PREDICT] Failed to preprocess data." << std::endl;
//             return false;
//         }

//         // Initialize ensemble
//         EnsembleConfig config;
//         config.method = ensembleMethod;
//         config.numModels = numEnsembleModels;
//         config.verbose = verbose_;

//         ensemble_ = std::make_unique<Ensemble>(inputSize, outputSize, layers, neurons,
//                                              minRange, maxRange, config);

//         // Train ensemble
//         auto start_time = std::chrono::high_resolution_clock::now();
//         if (!ensemble_->train(trainer_->getInputs(), trainer_->getTargets(), learningRate, epochs)) {
//             std::cerr << "[ENSEMBLE PREDICT] Failed to train ensemble." << std::endl;
//             return false;
//         }
//         auto end_time = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> training_duration = end_time - start_time;
//         trainingTime_ = training_duration.count();

//         // Evaluate ensemble
//         if (!trainer_->getTargets().empty()) {
//             rmse_ = ensemble_->calculateEnsembleError(trainer_->getInputs(), trainer_->getTargets());
//             mse_ = rmse_ * rmse_;
//             accuracy_ = 1.0f - rmse_; // Simple accuracy approximation

//             if (verbose_) {
//                 std::cout << "\n[ENSEMBLE PREDICT] Evaluation Metrics:" << std::endl;
//                 std::cout << "RMSE: " << rmse_ << std::endl;
//                 std::cout << "MSE: " << mse_ << std::endl;
//                 std::cout << "Accuracy: " << accuracy_ * 100.0f << "%" << std::endl;
//                 std::cout << "Training Execution Time: " << trainingTime_ << " seconds" << std::endl;
//             }
//         }

//         // Save results
//         if (!saveEnsembleResults(outputCsvFile, inputFile, hasHeader, delimiter)) {
//             std::cerr << "[ENSEMBLE PREDICT] Failed to save results." << std::endl;
//             return false;
//         }

//         if (verbose_) {
//             std::cout << "[ENSEMBLE PREDICT] Ensemble prediction process completed successfully." << std::endl;
//         }

//         return true;
//     } catch (const std::exception& e) {
//         std::cerr << "[ENSEMBLE PREDICT] Exception during ensemble prediction: " << e.what() << std::endl;
//         return false;
//     }
// }

// bool Predict::saveEnsembleResults(const std::string& outputCsvFile,
//                                  const std::string& inputFile,
//                                  bool hasHeader,
//                                  char delimiter) {
//     if (verbose_) {
//         std::cout << "[ENSEMBLE PREDICT] Saving ensemble prediction results to: " << outputCsvFile << std::endl;
//     }

//     try {
//         // Get ensemble predictions
//         auto predictions = ensemble_->predict(trainer_->getInputs());

//         // Denormalize predictions if trainer has normalization info
//         if (!predictions.empty() && trainer_->original_data_global_min != trainer_->original_data_global_max) {
//             for (auto& sample : predictions) {
//                 for (float& val : sample) {
//                     val = ((val - trainer_->min) / (trainer_->max - trainer_->min)) *
//                           (trainer_->original_data_global_max - trainer_->original_data_global_min) +
//                           trainer_->original_data_global_min;
//                 }
//             }
//         }

//         // Save to CSV (simplified version - you might want to extend this)
//         std::ofstream file(outputCsvFile);
//         if (!file.is_open()) {
//             std::cerr << "[ENSEMBLE PREDICT] Cannot open output file: " << outputCsvFile << std::endl;
//             return false;
//         }

//         // Write header if needed
//         if (hasHeader) {
//             file << "prediction_1";
//             for (size_t i = 1; i < predictions[0].size(); ++i) {
//                 file << delimiter << "prediction_" << (i + 1);
//             }
//             file << "\n";
//         }

//         // Write predictions
//         for (const auto& sample : predictions) {
//             if (!sample.empty()) {
//                 file << sample[0];
//                 for (size_t i = 1; i < sample.size(); ++i) {
//                     file << delimiter << sample[i];
//                 }
//                 file << "\n";
//             }
//         }

//         file.close();

//         if (verbose_) {
//             std::cout << "[ENSEMBLE PREDICT] Ensemble results saved successfully." << std::endl;
//         }
//         return true;
//     } catch (const std::exception& e) {
//         std::cerr << "[ENSEMBLE PREDICT] Failed to save ensemble results: " << e.what() << std::endl;
//         return false;
//     }
// }
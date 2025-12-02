#pragma once
#define EMBEDDING_DIM 50

#include "CoreAI3DCommon.hpp"

#include "Core.hpp"
#include "Language.hpp"
#include "Database.hpp"
#include "VisionModule.hpp"
#include "WebSocketServer.hpp"
#include "DataAugmentation.hpp"
#include <nlohmann/json.hpp>

class Language;
class CoreAI;
class Database;
class VisionModule;

// Checkpoint configuration structure
struct CheckpointConfig {
    int checkpointFrequencyEpochs = 10;        // Save checkpoint every N epochs
    bool bestModelTrackingEnabled = true;      // Enable best model tracking
    int bestModelPatienceEpochs = 5;           // Patience for no improvement before saving best model
    float bestModelMinImprovement = 0.001f;    // Minimum improvement threshold for best model
    std::string checkpointDirectory = "checkpoints"; // Directory for checkpoints
    std::string checkpointPrefix = "checkpoint";     // Prefix for checkpoint files
    std::string bestModelPrefix = "best_model";      // Prefix for best model files
    int maxCheckpointsToKeep = 5;              // Maximum number of checkpoints to keep
    bool saveOptimizerState = true;            // Save optimizer state in checkpoints
    bool saveTrainingMetadata = true;          // Save training metadata
};

class Training {
public:
    // Constructor for online mode (with database) - only if USE_MYSQL is defined
#ifdef USE_MYSQL
    Training(const std::string& dbHost, unsigned int dbPort, const std::string& dbUser,
        const std::string dbPassword, const std::string& dbSchema, int sslDummy, bool createTables, bool verbose = true, WebSocketServer* wsServer = nullptr);
#endif

    // Constructor for offline mode (no database) - always available
    Training(bool isOffline, bool verbose = true, WebSocketServer* wsServer = nullptr);

    void initializeLanguageProcessor(std::string& embedingFile, int& embeddingDim, std::string& dbHost, int& dbPort, std::string& dbUser, std::string& dbPassword, std::string& dbSchema, int sslDummy, std::string& lang, int& inputSize, int& outputSize, int& layers, int& neurons);


    float convertDateTimeToTimestamp(const std::string& datetime);
    std::string timestampToDateTimeString(float timestamp);
    // Modified loadCSV to accept datasetName for database saving
    bool loadCSV(const std::string& filename, long long numSamplesToLoad, int outputSize, bool hasHeader,
        bool containsText, const char& delimitor, const std::string& datasetName);

    // Modified loadTargetsCSV to save to database if dbManager is active
    bool loadTargetsCSV(const std::string& filename, const char& delim,
        bool hasHeader, bool containsText, int& datasetId); // Added datasetId

    // New method to load dataset from database
    bool loadDatasetFromDB(int& datasetId);

    void splitInputOutput(int outputSize); // This might become less relevant if inputs/targets are directly loaded
    void normalize(float minRange, float maxRange);
    int getMinInputColumns() const;
    void preprocess(float minRange, float maxRange);
    void train(double learningRate, int epochs, int batchSize = 32); // Changed samples to epochs, added batch size

    // Optimized batch size selection methods
    int getOptimalBatchSize(int datasetSize, size_t availableMemoryMB = 1024);
    int getAdaptiveBatchSize(int currentEpoch, int totalEpochs, int baseBatchSize);
    void setBatchSizeStrategy(const std::string& strategy); // "fixed", "adaptive", "memory_based"

    // Enhanced training with adaptive hyperparameters
    void trainAdaptive(double initialLearningRate, int maxEpochs, int batchSize = 32,
                      double minLearningRate = 1e-6, int patience = 10, double minImprovement = 1e-4);

    int numSamples; // Renamed from 'samples' for clarity
    int inputSize; // Will be set dynamically after loading data
    int outputSize; // Will be set dynamically after loading data
    int layers;
    int neurons;
    int batchSize; // Batch size for training (adaptive)
    double learningRate;
    float min, max;

    // Adaptive hyperparameters
    int adaptiveBatchSize;
    double adaptiveLearningRateValue;
    int optimalEpochs;
    std::string optimalOptimizer;

    // Convergence and early stopping variables
    double convergenceLossThreshold;
    int convergencePatienceEpochs;
    double convergenceMinImprovement;
    bool earlyStoppingEnabled;

    // Gradient clipping variables
    bool gradientClippingEnabled;
    double maxGradientNormThreshold;

    // Learning rate decay variables
    std::string lrDecayType;
    double lrDecayRate;
    int lrDecaySteps;

    // Optimizer configuration variables
    std::string currentOptimizerType;
    std::map<std::string, double> optimizerParams;

    // Epoch management variables
    int minTrainingEpochs;
    int maxTrainingEpochs;

    // Auto-tuning variables
    std::map<std::string, std::pair<double, double>> tunableParameters;
    bool autoTuningEnabled;
    std::map<std::string, double> bestHyperparameters;

    std::string embedding_file;
    std::string language;
    // New evaluation methods
    float calculateRMSE();
    float calculateMSE();
    float calculateAccuracy(float threshold = 0.1f);
    const std::vector<std::vector<float>>& getTargets() const { return targets; }
    const std::vector<std::vector<float>>& getInputs() const { return inputs; }
    // K-fold cross-validation structure
    struct CrossValidationResults {
        float avgRMSE;
        float avgMSE;
        float avgAccuracy;
        std::vector<float> foldRMSEs;
        std::vector<float> foldMSEs;
        std::vector<float> foldAccuracies;
    };

    // K-fold cross-validation method
    CrossValidationResults performKFoldCrossValidation(int k = 5, double learningRate = 0.01, int epochs = 100);

    // Convergence and early stopping structures
    struct ConvergenceCriteria {
        double minLossImprovement = 1e-4;     // Minimum improvement to continue training
        int patienceEpochs = 10;              // Epochs to wait for improvement before early stopping
        double minLearningRate = 1e-6;        // Minimum learning rate before stopping
        int maxEpochs = 1000;                 // Maximum epochs regardless of convergence
        bool useEarlyStopping = true;         // Enable early stopping
        bool useLossPlateauDetection = true;  // Detect loss plateaus
        int plateauPatience = 5;              // Epochs to wait after plateau detection
    };

    struct TrainingMetrics {
        double currentLoss = 0.0;
        double bestLoss = std::numeric_limits<double>::max();
        double previousLoss = std::numeric_limits<double>::max();
        int epochsWithoutImprovement = 0;
        int currentEpoch = 0;
        int totalEpochs = 0;
        double currentLearningRate = 0.0;
        bool converged = false;
        bool earlyStopped = false;
        std::vector<double> lossHistory;
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point endTime;
    };

    CoreAI* getCore();
    Language* getLanguage();
    nlohmann::json getNetworkTopology();
    nlohmann::json getNetworkActivity();
    // Database interaction methods - only if USE_MYSQL is defined
#ifdef USE_MYSQL
    bool saveModel(int& datasetId);
    bool loadModel(const int& datasetId);
    bool loadModel(const std::string& datasetId);
#endif
    void printFullMatrix(std::vector<std::vector<float>>& data, int len, int precision = 6);
    void printDenormalizedAsOriginalMatrix(std::vector<std::vector<float>>& normalized_data, int len, int precision = 4);
    float original_data_global_min;                 // Stores global min from original data for denorm
    float original_data_global_max;                 // Stores global max from original data for denorm
    float original_input_global_min;                // Stores input min for separate scaling
    float original_input_global_max;                // Stores input max for separate scaling
    float original_target_global_min;               // Stores target min for separate scaling
    float original_target_global_max;               // Stores target max for separate scaling


    // New method to save results to CSV
    bool saveResultsToCSV(const std::string& filename, const std::string& inputFilename, bool hasHeader, const char& delimiter); // New parameter for input filename and header
    std::vector<std::vector<float>> denormalizeMatrix(const std::vector<std::vector<float>>& normalized_data) const;

    // Asynchronous training methods
    bool startTraining(double learningRate, int epochs);
    bool isTraining() const;
    std::string getTrainingStatus() const;
    bool stopTraining();
    void resetTrainingState();
    // Checkpointing methods
    void configureCheckpointing(const CheckpointConfig& config);
    void enableCheckpointing(bool enable);
    bool saveCheckpoint(int epoch, float loss);
    bool saveBestModel(float currentLoss);
    bool loadCheckpoint(const std::string& checkpointPath);
    std::string getCheckpointFilename(int epoch) const;
    std::string getBestModelFilename() const;
    void initializeCheckpointing();

    // Utility methods
    std::string getTimestampString() const;

    // Data augmentation methods
    void initializeDataAugmentation(const DataAugmentation::AugmentationConfig& config = DataAugmentation::AugmentationConfig());
    void enableRotation(bool enable);
    void enableScaling(bool enable);
    void enableNoise(bool enable);
    void enableSymmetry(bool enable);
    void setRotationMaxAngle(float angle);
    void setScalingRange(float minScale, float maxScale);
    void setNoiseStdDev(float stdDev);
    void setSymmetryProbability(float probability);
    const DataAugmentation::AugmentationConfig& getAugmentationConfig() const;
    void applyDataAugmentation();

    // Memory-intensive data augmentation methods
    void createMultipleAugmentedCopies(int numCopies = 10);
    void clearAugmentedCopies();
    size_t getAugmentedDataSize() const;

    // Data buffering system
    struct BatchBuffer {
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> targets;
        int batchIndex;
        bool isProcessed;
        std::chrono::steady_clock::time_point createdTime;
    };

    void initializeDataBuffering(size_t maxBuffers = 100, size_t batchSize = 32);
    void addBatchToBuffer(const std::vector<std::vector<float>>& batchInputs,
                         const std::vector<std::vector<float>>& batchTargets);
    bool getNextBatchFromBuffer(std::vector<std::vector<float>>& batchInputs,
                               std::vector<std::vector<float>>& batchTargets);
    void clearOldBuffers(std::chrono::seconds maxAge = std::chrono::seconds(300));
    size_t getBufferMemoryUsage() const;
    void enableMemoryIntensiveBuffering(bool enable);

    // Historical data storage
    struct HistoricalDataPoint {
        std::vector<std::vector<float>> inputs;
        std::vector<std::vector<float>> targets;
        float loss;
        int epoch;
        std::chrono::steady_clock::time_point timestamp;
        std::string metadata;
    };

    void initializeHistoricalStorage(size_t maxHistorySize = 1000);
    void storeHistoricalDataPoint(const std::vector<std::vector<float>>& inputs,
                                 const std::vector<std::vector<float>>& targets,
                                 float loss, int epoch, const std::string& metadata = "");
    const std::vector<HistoricalDataPoint>& getHistoricalData() const;
    void clearHistoricalData();
    size_t getHistoricalMemoryUsage() const;
    std::vector<HistoricalDataPoint> getHistoricalDataInRange(int startEpoch, int endEpoch) const;

    // Memory usage monitoring
    struct MemoryStats {
        size_t baseDataSize;
        size_t augmentedDataSize;
        size_t bufferSize;
        size_t historicalSize;
        size_t totalSize;
        double memoryUsageMB;
    };

    MemoryStats getMemoryStats() const;
    void logMemoryUsage() const;
    void enableMemoryMonitoring(bool enable);

    // Batch processing methods for multiple files
    bool loadBatchFiles(const std::vector<std::string>& filenames, int maxSeqLen, int embeddingDim);
    bool loadDirectoryFiles(const std::string& directoryPath, const std::vector<std::string>& extensions, int maxSeqLen, int embeddingDim);

    // General file loading method
    bool loadFile(const std::string& filename, int maxSeqLen, int embeddingDim); // General method that detects file type and routes to appropriate loader

    bool loadCSVFromData(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& targets, const std::string& datasetName); // Load data directly into the training system

 private:
   std::mt19937 gen;
   std::vector<std::vector<float>> raw_data; // Might be less used now
   std::vector<std::vector<float>> inputs;
   std::vector<std::vector<float>> targets;
   std::vector<std::vector<float>> embeddingLayer;
   std::map<std::string, float> textEncodingMap;
   std::default_random_engine randomEngine;

   std::unique_ptr<CoreAI> core;
   std::unique_ptr<Language> langProc;
   std::unique_ptr<DataAugmentation> dataAugmentation;
   WebSocketServer* webSocketServer;

   // Memory-intensive data augmentation storage
   std::vector<std::vector<std::vector<float>>> augmentedInputsCopies;  // Multiple copies of augmented inputs
   std::vector<std::vector<std::vector<float>>> augmentedTargetsCopies; // Multiple copies of augmented targets

   // Data buffering system
   std::deque<BatchBuffer> batchBuffers;  // Queue of processed batches in memory
   size_t maxBatchBuffers;                // Maximum number of buffers to keep
   size_t bufferBatchSize;                // Size of each batch in buffer
   bool memoryIntensiveBufferingEnabled;  // Flag to enable memory-intensive buffering
   std::mutex bufferMutex;                // Mutex for thread-safe buffer access

   // Historical data storage
   std::vector<HistoricalDataPoint> historicalData;  // Historical training data points
   size_t maxHistoricalSize;              // Maximum number of historical points to keep
   bool memoryMonitoringEnabled;          // Flag to enable memory monitoring

   // Training state for async operations
   std::atomic<bool> trainingActive;
   std::atomic<bool> trainingStopRequested;
   std::future<void> trainingFuture;
   std::string trainingStatus;
   int currentEpoch;
   int totalEpochs;
   float currentLoss;
   std::chrono::steady_clock::time_point trainingStartTime;
   // Checkpointing configuration and state
   CheckpointConfig checkpointConfig;
   bool checkpointingEnabled;

   // Checkpointing state variables
   float bestValidationLoss;
   int epochsSinceBestModel;
   int lastCheckpointEpoch;
   std::vector<std::string> checkpointFiles;
    // Database manager - only if USE_MYSQL is defined
 #ifdef USE_MYSQL
    std::unique_ptr<Database> dbManager; // Changed to unique_ptr
 #endif
    std::vector<std::vector<float>> normalizeData(const std::vector<std::vector<float>>& data_to_normalize,
        float original_min, float original_max,
        float target_min, float target_max) const;

    float last_known_timestamp;
    bool loadTextCSV(const std::string& filename, int maxSeqLen, int embeddingDim);

    // New methods for learning from various file types
    bool loadTextFile(const std::string& filename, int maxSeqLen, int embeddingDim);
    bool loadJSONFile(const std::string& filename, const std::string& dataPath, int maxSeqLen, int embeddingDim);
    bool loadXMLFile(const std::string& filename, const std::string& xpath, int maxSeqLen, int embeddingDim);
    bool loadDocumentFile(const std::string& filename, int maxSeqLen, int embeddingDim);

    // New methods for learning from web content
    bool loadFromWebURL(const std::string& url, int maxSeqLen, int embeddingDim);
    bool loadFromWebSearch(const std::string& query, int maxResults, int maxSeqLen, int embeddingDim);
    bool trainOnWebContent(const std::string& url, int epochs = 10);
    bool trainOnWebSearchResults(const std::string& query, int maxResults, int epochs = 10);

    // New methods for learning from video content
    bool loadFromVideoFile(const std::string& videoPath, int maxSeqLen, int embeddingDim, int frameSamplingRate = 30);
    bool loadFromVideoAnalysis(const std::string& videoPath, int maxSeqLen, int embeddingDim, bool includeOCR = true, bool includeDetections = true);
    bool trainOnVideoContent(const std::string& videoPath, int epochs = 10, int frameSamplingRate = 30);
    bool trainOnVideoDataset(const std::string& videoPath, int epochs = 10, int frameSamplingRate = 30);
    bool analyzeVideoContent(const std::string& videoPath);
    std::vector<std::vector<float>> extractVideoFeatures(const std::string& videoPath, int temporalWindow = 5);
    std::vector<std::vector<float>> extractVideoFeaturesBuiltin(const std::string& videoPath, int frameSamplingRate);

    // New methods for learning from image content
    bool loadFromImageFile(const std::string& imagePath, int maxSeqLen, int embeddingDim);
    bool loadFromImageDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim);
    bool trainOnImageContent(const std::string& imagePath, int epochs = 10);
    bool trainOnImageDataset(const std::string& datasetPath, int epochs = 10);
    bool analyzeImageContent(const std::string& imagePath);
    std::vector<float> extractImageFeatures(const std::string& imagePath);
    std::vector<float> extractImageFeaturesBuiltin(const std::string& imagePath);

    // Document processing helper methods
    std::string extractTextFromPDF(const std::string& filename);
    std::string extractTextFromRawPDF(const std::string& filename);
    std::string decodePDFText(const std::string& pdfText);
    std::string extractTextFromDOCX(const std::string& filename);
    std::string extractTextFromRawDOCX(const std::string& filename);
    std::string extractTextFromDOC(const std::string& filename);
    std::string extractTextFromTXT(const std::string& filename);
    bool processExtractedText(const std::string& extractedText, int maxSeqLen, int embeddingDim, const std::string& filename);

    // New methods for learning from audio content
    bool loadFromAudioFile(const std::string& audioPath, int maxSeqLen, int embeddingDim);
    bool loadFromAudioTranscription(const std::string& audioPath, int maxSeqLen, int embeddingDim);
    bool trainOnAudioContent(const std::string& audioPath, int epochs = 10);
    bool trainOnAudioFile(const std::string& audioPath, int epochs = 10);
    bool analyzeAudioContent(const std::string& audioPath);
    std::vector<float> extractAudioFeatures(const std::string& audioPath, int temporalWindow = 5);
    std::vector<float> extractAudioFeaturesBuiltin(const std::string& audioPath);
    bool loadAudioDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim);
    bool trainOnAudioDataset(const std::string& datasetPath, int epochs = 10);

    // Database integration for file datasets
    bool saveFileDataset(int& datasetId, const std::string& datasetName, const std::string& fileType);

    bool isOfflineMode; // New member to track offline status
    int currentDatasetId; // Stores the ID of the currently loaded/saved dataset
    std::string currentDatasetName; // Stores the name of the currently loaded/saved dataset

    // Batch size optimization
    std::string batchSizeStrategy; // "fixed", "adaptive", "memory_based"
    size_t availableMemoryMB; // Available memory for batch size calculation

public:
    // Public accessors for Predict class
    bool getIsOfflineMode() const { return isOfflineMode; }
    int getCurrentDatasetId() const { return currentDatasetId; }
    std::string getCurrentDatasetName() const { return currentDatasetName; }

    bool verbose; // Control verbosity of debug output

    void printProgressBar(const std::string& prefix, long long current, long long total, int barWidth);
    int detectMaxSeqLength(const std::string& filename);
    long long countLines(const std::string& filename, bool hasHeader);

    // Progress tracking for large file conversions
    std::atomic<long long> conversionProgress;
    std::atomic<long long> conversionTotal;
    std::string currentConversionFile;
    std::mutex progressMutex;

    // Progress tracking methods
    long long getConversionProgress() const { return conversionProgress.load(); }
    long long getConversionTotal() const { return conversionTotal.load(); }
    std::string getCurrentConversionFile() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(progressMutex));
        return currentConversionFile;
    }
    double getConversionProgressPercent() const {
        long long total = conversionTotal.load();
        if (total == 0) return 0.0;
        return (static_cast<double>(conversionProgress.load()) / total) * 100.0;
    }

    std::vector<std::string> original_date_strings; // To store the original date strings


    // NEW: Database saving methods for comprehensive data persistence
    bool saveLearningSettingsDuringTraining(int datasetId, double learningRate, int batchSize, int epochs,
                                           float momentum, float weightDecay, float dropoutRate,
                                           const std::string& optimizer, const std::string& lossFunction,
                                           const std::string& activationFunction);
    bool savePredictionResultsAfterInference(int datasetId, const std::vector<std::vector<float>>& inputs,
                                            const std::vector<std::vector<float>>& actualTargets,
                                            const std::vector<std::vector<float>>& predictedTargets);
    bool saveChatSession(int& sessionId, const std::string& sessionName, const std::string& userId,
                        const std::string& modelVersion, const std::string& status);
    bool saveChatMessage(int sessionId, const std::string& role, const std::string& content,
                        const std::string& messageType, float confidence, const std::string& metadata);
    bool saveModelStateJSON(int datasetId, const nlohmann::json& modelState);
    bool saveComprehensiveTrainingData(int datasetId, const std::string& datasetName);
    int getDecimalPlacesInString(float value, int precision_for_conversion);
    // NEW: Helper method to denormalize a matrix

    // Problem complexity detection and auto-tuning methods
    enum class ProblemComplexity { SIMPLE, MODERATE, COMPLEX };

    struct ComplexityAnalysis {
        ProblemComplexity complexity = ProblemComplexity::MODERATE;
        float complexity_score = 0.0f; // 0.0 = simple, 1.0 = complex
        std::string reasoning = "";
        bool requires_3d_transforms = false;
        int recommended_layers = 3;
        int recommended_neurons = 10;
        double recommended_learning_rate = 0.01;
        int recommended_epochs = 10;
    };

    ComplexityAnalysis detectProblemComplexity();
    void applyAutoTuning(ComplexityAnalysis& analysis);
    void configure3DTransforms(bool enable_complex_transforms);

    // Hyperparameter tuning integration methods
    void setHyperparameters(const std::map<std::string, float>& float_params,
                           const std::map<std::string, int>& int_params,
                           const std::map<std::string, std::string>& categorical_params);
    void setLearningRateForTuning(double lr);
    void setNetworkArchitectureForTuning(int layers, int neurons);
    void setOptimizerForTuning(const std::string& optimizer_type);
    void setActivationForTuning(const std::string& activation_type);
    void setRegularizationForTuning(float l2_lambda, float dropout_rate);
    float evaluateForTuning(int max_epochs = 10); // Quick evaluation for hyperparameter tuning

    // Utility methods
    std::string formatValueForDisplay(float value, int customPrecision) const;
    static std::vector<std::string> parseCSVLine(const std::string& line, char delimiter);

    // Adaptive hyperparameter optimization methods
    void optimizeHyperparameters();
    int calculateOptimalBatchSize(int datasetSize);
    double calculateOptimalLearningRate(int inputSize, int outputSize, int datasetSize);
    std::string selectOptimalOptimizer(int datasetSize, int inputSize);
    int estimateOptimalEpochs(int datasetSize, int complexity);
    void applyOptimizedSettings();
    void initializeAdaptiveDefaults();

    // Convergence and adaptive training methods
    bool checkConvergence(const TrainingMetrics& metrics, const ConvergenceCriteria& criteria);
    double adaptiveLearningRate(double currentLR, const TrainingMetrics& metrics, const ConvergenceCriteria& criteria);
    void initializeTrainingMetrics(TrainingMetrics& metrics, double initialLR, int maxEpochs);
    void updateTrainingMetrics(TrainingMetrics& metrics, double currentLoss, int epoch);
    bool shouldStopTraining(const TrainingMetrics& metrics, const ConvergenceCriteria& criteria);
    void logTrainingProgress(const TrainingMetrics& metrics, bool verbose = true);
};
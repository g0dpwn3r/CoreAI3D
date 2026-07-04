#pragma once
#define EMBEDDING_DIM 50

#include "CoreAI3DCommon.hpp"

#include "Core.hpp"
#include "Language.hpp"
#include "Database.hpp"
#include "VisionModule.hpp"
#include "WebSocketServer.hpp"
#include <nlohmann/json.hpp>

class Language;
class CoreAI;
class Database;
class VisionModule;

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
    void train(double learningRate, int epochs); // Changed samples to epochs

    int numSamples; // Renamed from 'samples' for clarity
    int inputSize; // Will be set dynamically after loading data
    int outputSize; // Will be set dynamically after loading data
    int layers;
    int neurons;
    double learningRate;
    float min, max;
    
    // Setter methods
    void setLearningRate(double rate) { learningRate = rate; }
    void setEpochs(int epochs) { totalEpochs = epochs; }
    
    // Predict method
    std::vector<std::vector<float>> predict(const std::vector<std::vector<float>>& inputsToPredict);
    
    // Train method that takes inputs and targets directly
    bool train(const std::vector<std::vector<float>>& trainingInputs, const std::vector<std::vector<float>>& trainingTargets);

    std::string embedding_file;
    std::string language;
    // New evaluation methods
    float calculateRMSE();
    float calculateMSE();
    float calculateAccuracy(float threshold = 0.1f);
    const std::vector<std::vector<float>>& getTargets() const { return targets; }

    CoreAI* getCore();
    Language* getLanguage();
    nlohmann::json getNetworkTopology();
    nlohmann::json getNetworkActivity();
    
    // Getter for inputs (for testing purposes)
    const std::vector<std::vector<float>>& getInputs() const { return inputs; }
    
    // Overloaded train method with batch size (for compatibility with tests)
    void train(double learningRate, int epochs, int batchSize) {
        // Default implementation ignores batch size
        train(learningRate, epochs);
    }
    // Database interaction methods - only if USE_MYSQL is defined
#ifdef USE_MYSQL
    bool saveModel(int& datasetId);
    bool loadModel(const int& datasetId);
    bool loadModel(const std::string& datasetId);
#endif
    void printFullMatrix(std::vector<std::vector<float>>& data, int len, int precision = 6);
    void printDenormalizedAsOriginalMatrix(std::vector<std::vector<float>>& normalized_data, int len, int precision = 4);
    // Per-feature normalization bounds for independent scaling
    // These replace the single global min/max to prevent gradient collapse
    // when input features (x, y) have vastly different scales than target (z)
    std::vector<float> input_min;   // Min value for each input feature dimension (x, y)
    std::vector<float> input_max;   // Max value for each input feature dimension (x, y)
    std::vector<float> target_mins; // Min value for each target feature dimension (z)
    std::vector<float> target_maxs; // Max value for each target feature dimension (z)
    float target_min;               // Overall min for single-target convenience
    float target_max;               // Overall max for single-target convenience
    float original_data_global_min;                 // Kept for backward compatibility / fallback
    float original_data_global_max;                 // Kept for backward compatibility / fallback


    // New method to save results to CSV
    bool saveResultsToCSV(const std::string& filename, const std::string& inputFilename, bool hasHeader, const char& delimiter); // New parameter for input filename and header
    bool applySlidingWindows(int windowSize);
    std::vector<std::vector<float>> denormalizeMatrix(const std::vector<std::vector<float>>& normalized_data) const;

    // Independent feature normalization for inference on new data
    // Normalizes each column of data_to_normalize using its corresponding
    // original_mins[j] / original_maxs[j], mapping to the [target_min, target_max] range.
    std::vector<std::vector<float>> normalizeDataIndependent(const std::vector<std::vector<float>>& data_to_normalize,
        const std::vector<float>& original_mins,
        const std::vector<float>& original_maxs,
        float target_min, float target_max) const;

    // Core normalization methods with independent X,Y -> Z mapping
    // Computes per-dimension bounds for inputs and targets
    void compute_normalization_bounds();
    // Normalizes inputs (x, y) against their own per-feature bounds
    void normalize_inputs(float minRange, float maxRange);
    // Normalizes targets (z) against their own bounds, isolating the target scale
    void normalize_targets(float minRange, float maxRange);
    // Reconstructs real-world prediction from normalized z_hat using stored target bounds
    float denormalize_prediction(float z_norm) const;

    // Denormalization helpers using per-feature and per-target bounds
    void denormalizeInputs(std::vector<std::vector<float>>& input_data) const;
    void denormalizeTargets(std::vector<std::vector<float>>& target_data) const;

    // Asynchronous training methods
    bool startTraining(double learningRate, int epochs);
    bool isTraining() const;
    std::string getTrainingStatus() const;
    bool stopTraining();
    void resetTrainingState();

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
    WebSocketServer* webSocketServer;

    // Training state for async operations
    std::atomic<bool> trainingActive;
    std::atomic<bool> trainingStopRequested;
    std::future<void> trainingFuture;
    std::string trainingStatus;
    int currentEpoch;
    int totalEpochs;
    float currentLoss;
    std::chrono::steady_clock::time_point trainingStartTime;

 public:
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
     bool loadFile(const std::string& filename, int maxSeqLen, int embeddingDim); // General method that detects file type and routes to appropriate loader

    // Convergence and training metrics
    struct TrainingMetrics {
        float loss;
        int epoch;
        float accuracy;
        bool converged;
        bool earlyStopped;
    };

    struct ConvergenceCriteria {
        float lossThreshold;
        int patienceEpochs;
        float minImprovement;
    };

    // Batch processing methods for multiple files
    bool loadBatchFiles(const std::vector<std::string>& filenames, int maxSeqLen, int embeddingDim);
    bool loadDirectoryFiles(const std::string& directoryPath, const std::vector<std::string>& extensions, int maxSeqLen, int embeddingDim);

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

    // New methods for learning from 3D model content
    bool loadFrom3DModelFile(const std::string& modelPath, int maxSeqLen, int embeddingDim);
    bool trainOn3DModelContent(const std::string& modelPath, int epochs = 10);
    bool analyze3DModelContent(const std::string& modelPath);
    std::vector<float> extract3DModelFeatures(const std::string& modelPath);
    std::vector<std::vector<float>> extract3DModelFeaturesBatch(const std::vector<std::string>& modelPaths, int frameSamplingRate = 10);
    bool loadFrom3DModelDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim);
    bool trainOn3DModelDataset(const std::string& datasetPath, int epochs = 10);

    // New methods for learning from music content
    bool loadFromMusicFile(const std::string& musicPath, int maxSeqLen, int embeddingDim);
    bool trainOnMusicContent(const std::string& musicPath, int epochs = 10);
    bool analyzeMusicContent(const std::string& musicPath);
    std::vector<float> extractMusicFeatures(const std::string& musicPath, int temporalWindow = 5);
    std::vector<std::vector<float>> extractMusicFeaturesBatch(const std::vector<std::string>& musicPaths, int sampleRate = 44100);
    bool loadFromMusicDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim);
    bool trainOnMusicDataset(const std::string& datasetPath, int epochs = 10);
    std::string classifyMusicGenre(const std::string& musicPath);
    float detectMusicTempo(const std::string& musicPath);
    float detectMusicEnergy(const std::string& musicPath);

    // Database integration for file datasets
    bool saveFileDataset(int& datasetId, const std::string& datasetName, const std::string& fileType);

    bool isOfflineMode; // New member to track offline status
    int currentDatasetId; // Stores the ID of the currently loaded/saved dataset
    std::string currentDatasetName; // Stores the name of the currently loaded/saved dataset
public:
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


    int getDecimalPlacesInString(float value, int precision_for_conversion);
    std::string formatValueForDisplay(float value, int customPrecision) const;
    static std::vector<std::string> parseCSVLine(const std::string& line, char delimiter);

     // Problem complexity detection and auto-tuning
    enum class ProblemComplexity {
        SIMPLE,
        MODERATE,
        COMPLEX
    };

    struct ComplexityAnalysis {
        ProblemComplexity complexity;
        float complexity_score;
        bool requires_3d_transforms;
        int recommended_layers;
        int recommended_neurons;
        double recommended_learning_rate;
        int recommended_epochs;
    };

    ComplexityAnalysis detectProblemComplexity();
    void applyAutoTuning(const ComplexityAnalysis& analysis);

    // Load data from memory vectors
    bool loadCSVFromData(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& targets, const std::string& datasetName);

    // Additional database saving methods for comprehensive data persistence
#ifdef USE_MYSQL
    bool saveLearningSettingsDuringTraining(int datasetId, double learningRate, int batchSize, int epochs,
        float dropoutRate, float regularization, float momentum, const std::string& optimizer,
        const std::string& lossFunction, const std::string& activationFunction);
    
    bool savePredictionResultsAfterInference(int datasetId, const std::vector<std::vector<float>>& inputs,
        const std::vector<std::vector<float>>& actualOutputs, const std::vector<std::vector<float>>& predictedOutputs);
    
    bool saveChatSession(int& sessionId, const std::string& sessionName, const std::string& userId,
        const std::string& modelVersion, const std::string& status);
    
    bool saveChatMessage(int sessionId, const std::string& role, const std::string& content,
        const std::string& messageType, float confidence, const std::string& metadata);
    
    bool saveModelStateJSON(int datasetId, const nlohmann::json& modelState);
    
    bool saveComprehensiveTrainingData(int datasetId, const std::string& datasetName);
#endif
};
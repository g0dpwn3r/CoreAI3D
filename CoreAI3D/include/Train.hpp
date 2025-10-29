#pragma once
#define EMBEDDING_DIM 50

#include "CoreAI3DCommon.hpp"

#include "Core.hpp"
#include "Language.hpp"
#include "Database.hpp"
#include "VisionModule.hpp"
#include <nlohmann/json.hpp>

class Language;
class CoreAI;
class Database;
class VisionModule;

class Training {
public:
    // Constructor for online mode (with database)
    Training(const std::string& dbHost, unsigned int dbPort, const std::string& dbUser,
        const std::string dbPassword, const std::string& dbSchema, int sslDummy, bool createTables, bool verbose = true);

    // Constructor for offline mode (no database)
    Training(bool isOffline, bool verbose = true);

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
    // Database interaction methods
    bool saveModel(int& datasetId);
    bool loadModel(int& datasetId);
    void printFullMatrix(std::vector<std::vector<float>>& data, int len, int precision = 6);
    void printDenormalizedAsOriginalMatrix(std::vector<std::vector<float>>& normalized_data, int len, int precision = 4);
    float original_data_global_min;                 // Stores global min from original data for denorm
    float original_data_global_max;                 // Stores global max from original data for denorm


    // New method to save results to CSV
    bool saveResultsToCSV(const std::string& filename, const std::string& inputFilename, bool hasHeader, const char& delimiter); // New parameter for input filename and header
    std::vector<std::vector<float>> denormalizeMatrix(const std::vector<std::vector<float>>& normalized_data) const;
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
public:
    std::unique_ptr<Database> dbManager; // Changed to unique_ptr
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

    // New methods for learning from image content
    bool loadFromImageFile(const std::string& imagePath, int maxSeqLen, int embeddingDim);
    bool loadFromImageDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim);
    bool trainOnImageContent(const std::string& imagePath, int epochs = 10);
    bool trainOnImageDataset(const std::string& datasetPath, int epochs = 10);
    bool analyzeImageContent(const std::string& imagePath);
    std::vector<float> extractImageFeatures(const std::string& imagePath);

    // New methods for learning from audio content
    bool loadFromAudioFile(const std::string& audioPath, int maxSeqLen, int embeddingDim);
    bool loadFromAudioTranscription(const std::string& audioPath, int maxSeqLen, int embeddingDim);
    bool trainOnAudioContent(const std::string& audioPath, int epochs = 10);
    bool trainOnAudioFile(const std::string& audioPath, int epochs = 10);
    bool analyzeAudioContent(const std::string& audioPath);
    std::vector<float> extractAudioFeatures(const std::string& audioPath, int temporalWindow = 5);
    bool loadAudioDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim);
    bool trainOnAudioDataset(const std::string& datasetPath, int epochs = 10);

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

    std::vector<std::string> original_date_strings; // To store the original date strings


    int getDecimalPlacesInString(float value, int precision_for_conversion);
    std::string formatValueForDisplay(float value, int customPrecision) const;
    static std::vector<std::string> parseCSVLine(const std::string& line, char delimiter);

    // NEW: Helper method to denormalize a matrix

};
#pragma once
#define EMBEDDING_DIM 50

#include "CoreAI3DCommon.hpp"

#include "Core.hpp"
#include "Database.hpp"
#include "Language.hpp"
#include <nlohmann/json.hpp>

class Language;
class CoreAI;
class Database;

class Training {
public:
    // Constructor for online mode (with database)
    Training(const std::string& dbHost, unsigned int dbPort, const std::string& dbUser,
        const std::string dbPassword, const std::string& dbSchema, int sslDummy, bool createTables);

    // Constructor for offline mode (no database)
    Training(bool isOffline);

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
    std::unique_ptr<Database> dbManager; // Changed to unique_ptr
    std::vector<std::vector<float>> normalizeData(const std::vector<std::vector<float>>& data_to_normalize,
        float original_min, float original_max,
        float target_min, float target_max) const;

    float last_known_timestamp;
    bool loadTextCSV(const std::string& filename, int maxSeqLen, int embeddingDim);

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
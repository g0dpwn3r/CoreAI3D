#pragma once
#include "CoreAI3DCommon.hpp"

#include "Train.hpp"
#include "Core.hpp"
#include "Language.hpp"

// Forward declarations for nlohmann::json if needed in header for method signatures
// using json = nlohmann::json;

// Simple enum to replace mysqlx::SSLMode
enum class SSLMode {
    DISABLED,
    REQUIRED,
    VERIFY_CA,
    VERIFY_IDENTITY
};

class Database
{
public:
    // Constructor signature: Changed 'int port' to 'unsigned int port' to match implementation
    Database(const std::string& host, unsigned int port, const std::string& user, const std::string password, const std::string& schemaName, SSLMode ssl);

    void createTables();

    void saveAIModelState(
        int& datasetId, const std::vector<std::vector<float> >& inputData,
        const std::vector<std::vector<float> >& outputData,
        const std::vector<std::vector<float> >& hiddenData, // Changed to 2D
        const std::vector<float>& hiddenOutputData,         // Remains 1D
        const std::vector<float>& hiddenErrorData,          // Remains 1D
        const std::vector<std::vector<float> >& weightsHiddenInput,
        const std::vector<std::vector<float> >& weightsOutputHidden);

    // Struct to hold AI model state
    struct AIModelState
    {
        std::vector<std::vector<float> > inputData;
        std::vector<std::vector<float> > outputData; // Targets
        std::vector<std::vector<float> > hiddenData; // Changed to 2D
        std::vector<float> hiddenOutputData;         // Remains 1D
        std::vector<float> hiddenErrorData;          // Remains 1D
        std::vector<std::vector<float> > weightsHiddenInput;
        std::vector<std::vector<float> > weightsOutputHidden;
        // Add timestamp or version for loading latest state
    };

    AIModelState loadLatestAIModelState(int& datasetId);

    // Struct to hold loaded dataset data and metadata
    struct DatasetData
    {
        std::string datasetName;
        long long numRows = 0; // Changed to long long to match DB schema
        int numFeatures = 0;
        int numLabels = 0;
        std::vector<std::vector<float> > inputs;
        std::vector<std::vector<float> > targets;
    };

    // New methods for dataset management
    int addDataset(const std::string& datasetName,
        const std::string& description, int numRows, int numFeatures,
        int numLabels);
    void addDatasetRecord(int& datasetId, int rowIndex,
        const std::vector<float>& featureValues,
        const std::vector<float>& labelValues);
    DatasetData
        getDataset(int& datasetId); // New method to load dataset and its metadata
    void updateDatasetRecordLabels(
        int& datasetId, int rowIndex,
        const std::vector<float>& labelValues); // New method to update labels
    void clearDatasetRecords(
        long long& datasetId); // Changed to long long to match DB schema
    bool getDatasetRecords(long long datasetId,
        std::vector<std::vector<float>>& loadedInputs,
        std::vector<std::vector<float>>& loadedTargets,
        long long& loadedNumSamples,
        int& loadedInputSize, int& loadedOutputSize);

    // NEW: Struct to hold prediction results for a single sample
    struct PredictionResult
    {
        std::vector<float> inputFeatures;
        std::vector<float> actualTargets;
        std::vector<float> predictedTargets;
    };

    // NEW: Method to save prediction results to the database
    void savePredictionResults(int& datasetId, int sampleIndex,
        const std::vector<float>& inputFeatures,
        const std::vector<float>& actualTargets,
        const std::vector<float>& predictedTargets);

private:
    std::string dbHost;
    unsigned int dbPort;
    std::string dbUser;
    std::string dbPassword;
    std::string dbSchema;
    SSLMode sslMode;

    // Helper functions for BLOB conversion of 2D matrices
    std::vector<char> matrixToBlob(const std::vector<std::vector<float> >& matrix);
    std::vector<std::vector<float> > blobToMatrix(const std::vector<char>& blob);

    // New helper functions for 1D vector BLOB conversion
    std::vector<char> vectorToBlob(const std::vector<float>& vec);
    std::vector<float> blobToVector(const std::vector<char>& blob);

    // NEW: Helper to create the prediction results table
    void createPredictionResultsTable();
};

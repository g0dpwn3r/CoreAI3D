#include "Database.hpp"

// Constructor
Database::Database(const std::string& host, unsigned int port,
    const std::string& user, const std::string password,
    const std::string& schemaName, SSLMode ssl)
    : dbHost(host), dbPort(port), dbUser(user), dbPassword(password),
      dbSchema(schemaName), sslMode(ssl)
{
    // Stub implementation - no actual database connection
    std::cout << "Database stub initialized (no MySQL connection)" << std::endl;
}

// Helper to convert std::vector<std::vector<float>> to std::vector<char>
std::vector<char> Database::matrixToBlob(const std::vector<std::vector<float>>& matrix) {
    std::vector<char> blob_data;
    if (matrix.empty()) {
        int rows = 0;
        int cols = 0;
        blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(&rows), reinterpret_cast<const char*>(&rows) + sizeof(int));
        blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(&cols), reinterpret_cast<const char*>(&cols) + sizeof(int));
        return blob_data;
    }

    int rows = static_cast<int>(matrix.size());
    int cols = static_cast<int>(matrix[0].size());

    // Store dimensions
    blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(&rows), reinterpret_cast<const char*>(&rows) + sizeof(int));
    blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(&cols), reinterpret_cast<const char*>(&cols) + sizeof(int));

    // Store data
    for (const auto& row : matrix) {
        blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(row.data()), reinterpret_cast<const char*>(row.data()) + row.size() * sizeof(float));
    }

    return blob_data;
}

std::vector<std::vector<float>> Database::blobToMatrix(const std::vector<char>& blob) {
    std::vector<std::vector<float>> matrix;
    if (blob.empty()) {
        return matrix;
    }

    int rows, cols;
    size_t offset = 0;

    // Read dimensions
    if (blob.size() < sizeof(int) * 2) return matrix;
    std::copy(blob.data() + offset, blob.data() + offset + sizeof(int), reinterpret_cast<char*>(&rows));
    offset += sizeof(int);
    std::copy(blob.data() + offset, blob.data() + offset + sizeof(int), reinterpret_cast<char*>(&cols));
    offset += sizeof(int);

    if (rows <= 0 || cols <= 0) return matrix;

    matrix.resize(rows, std::vector<float>(cols));

    // Read data
    for (int i = 0; i < rows; ++i) {
        if (offset + cols * sizeof(float) > blob.size()) {
            std::cerr << "Error: Blob data truncated for matrix. Expected " << cols * sizeof(float) << " bytes, got " << (blob.size() - offset) << std::endl;
            return std::vector<std::vector<float>>();
        }
        std::copy(blob.data() + offset, blob.data() + offset + cols * sizeof(float), reinterpret_cast<char*>(matrix[i].data()));
        offset += cols * sizeof(float);
    }
    return matrix;
}

// Helper to convert std::vector<float> to std::vector<char>
std::vector<char> Database::vectorToBlob(const std::vector<float>& vec) {
    std::vector<char> blob_data;
    int size = static_cast<int>(vec.size());

    // Store size
    blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(&size), reinterpret_cast<const char*>(&size) + sizeof(int));

    // Store data
    blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(vec.data()), reinterpret_cast<const char*>(vec.data()) + size * sizeof(float));

    return blob_data;
}

// Helper to convert std::vector<char> to std::vector<float>
std::vector<float> Database::blobToVector(const std::vector<char>& blob) {
    std::vector<float> vec;
    if (blob.empty()) {
        return vec;
    }

    int size;
    size_t offset = 0;

    // Read size
    if (blob.size() < sizeof(int)) return vec;
    std::copy(blob.data() + offset, blob.data() + offset + sizeof(int), reinterpret_cast<char*>(&size));
    offset += sizeof(int);

    if (size <= 0) return vec;

    vec.resize(size);

    // Read data
    if (offset + size * sizeof(float) > blob.size()) {
        std::cerr << "Error: Blob data truncated for vector. Expected " << size * sizeof(float) << " bytes, got " << (blob.size() - offset) << std::endl;
        return std::vector<float>();
    }
    std::copy(blob.data() + offset, blob.data() + offset + size * sizeof(float), reinterpret_cast<char*>(vec.data()));
    offset += size * sizeof(float);

    return vec;
}

void Database::createPredictionResultsTable() {
    // Stub implementation
    std::cout << "Database stub: createPredictionResultsTable() called" << std::endl;
}

void Database::createTables() {
    // Stub implementation
    std::cout << "Database stub: createTables() called" << std::endl;
    createPredictionResultsTable();
}

int Database::addDataset(const std::string& datasetName, const std::string& description,
    int numRows, int numFeatures, int numLabels) {
    // Stub implementation - return a dummy ID
    static int nextId = 1;
    std::cout << "Database stub: addDataset('" << datasetName << "') -> ID: " << nextId << std::endl;
    return nextId++;
}

void Database::addDatasetRecord(int& datasetId, int rowIndex,
    const std::vector<float>& featureValues,
    const std::vector<float>& labelValues) {
    // Stub implementation - reduce verbosity for bulk operations
    // Only log every 100th record to avoid spam during data loading
    if (rowIndex % 100 == 0) {
        std::cout << "Database stub: addDatasetRecord(datasetId=" << datasetId << ", rowIndex=" << rowIndex << ") - processing..." << std::endl;
    }
}

Database::DatasetData Database::getDataset(int& datasetId) {
    DatasetData data;
    // Stub implementation - return empty data
    data.datasetName = "stub_dataset_" + std::to_string(datasetId);
    data.numRows = 0;
    data.numFeatures = 0;
    data.numLabels = 0;
    std::cout << "Database stub: getDataset(" << datasetId << ") -> empty data" << std::endl;
    return data;
}

void Database::updateDatasetRecordLabels(int& datasetId, int rowIndex, const std::vector<float>& labelValues) {
    // Stub implementation
    std::cout << "Database stub: updateDatasetRecordLabels(datasetId=" << datasetId << ", rowIndex=" << rowIndex << ")" << std::endl;
}

void Database::clearDatasetRecords(long long& datasetId) {
    // Stub implementation
    std::cout << "Database stub: clearDatasetRecords(datasetId=" << datasetId << ")" << std::endl;
}

void Database::saveAIModelState(int& datasetId,
    const std::vector<std::vector<float>>& inputData,
    const std::vector<std::vector<float>>& outputData,
    const std::vector<std::vector<float>>& hiddenData,
    const std::vector<float>& hiddenOutputData,
    const std::vector<float>& hiddenErrorData,
    const std::vector<std::vector<float>>& weightsHiddenInput,
    const std::vector<std::vector<float>>& weightsOutputHidden) {
    // Stub implementation
    std::cout << "Database stub: saveAIModelState(datasetId=" << datasetId << ")" << std::endl;
}

Database::AIModelState Database::loadLatestAIModelState(int& datasetId) {
    AIModelState state;
    // Stub implementation - return empty state
    std::cout << "Database stub: loadLatestAIModelState(" << datasetId << ") -> empty state" << std::endl;
    return state;
}

void Database::savePredictionResults(int& datasetId, int sampleIndex,
    const std::vector<float>& inputFeatures,
    const std::vector<float>& actualTargets,
    const std::vector<float>& predictedTargets) {
    // Stub implementation
    std::cout << "Database stub: savePredictionResults(datasetId=" << datasetId << ", sampleIndex=" << sampleIndex << ")" << std::endl;
}

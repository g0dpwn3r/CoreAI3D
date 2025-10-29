#include "Database.hpp"

// Constructor
Database::Database(const std::string& host, unsigned int port,
    const std::string& user, const std::string password,
    const std::string& schemaName, SSLMode ssl, bool createTables)
    : dbHost(host), dbPort(port), dbUser(user), dbPassword(password),
      dbSchema(schemaName), sslMode(ssl)
{
#ifdef USE_MYSQL
    try {
        // Convert SSLMode enum to mysqlx::SSLMode
        mysqlx::SSLMode mysqlSslMode;
        switch (sslMode) {
            case SSLMode::DISABLED:
                mysqlSslMode = mysqlx::SSLMode::DISABLED;
                break;
            case SSLMode::REQUIRED:
                mysqlSslMode = mysqlx::SSLMode::REQUIRED;
                break;
            case SSLMode::VERIFY_CA:
                mysqlSslMode = mysqlx::SSLMode::VERIFY_CA;
                break;
            case SSLMode::VERIFY_IDENTITY:
                mysqlSslMode = mysqlx::SSLMode::VERIFY_IDENTITY;
                break;
            default:
                mysqlSslMode = mysqlx::SSLMode::DISABLED;
                break;
        }

        // First, connect without schema to create it if it doesn't exist
        mysqlx::Session tempSession(host, port, user, password);
        tempSession.sql("CREATE DATABASE IF NOT EXISTS " + schemaName).execute();
        tempSession.close();

        // Now establish connection to the specific schema
        session = new mysqlx::Session(host, port, user, password, schemaName);
        std::cout << "Database connection established successfully to " << host << ":" << port << "/" << schemaName << std::endl;

        // Create tables if they don't exist and createTables is true
        if (createTables) {
            this->createTables();
        }
    } catch (const std::exception& ex) {
        std::cerr << "Database initialization error: " << ex.what() << std::endl;
        throw std::runtime_error("Failed to initialize database: " + std::string(ex.what()));
    }
#else
    // Stub implementation - no actual database connection
    std::cout << "Database stub initialized (no MySQL connection)" << std::endl;
    if (createTables) {
        Database::createTables();
    }
#endif
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
    ensureConnection();
#ifdef USE_MYSQL
    try {
        session->sql("CREATE TABLE IF NOT EXISTS " + dbSchema + ".prediction_results ("
                       "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                       "dataset_id BIGINT NOT NULL,"
                       "sample_index INT NOT NULL,"
                       "input_features BLOB,"
                       "actual_targets BLOB,"
                       "predicted_targets BLOB,"
                       "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                       "INDEX idx_dataset_sample (dataset_id, sample_index)"
                       ")").execute();
        std::cout << "Created prediction_results table in schema " << dbSchema << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error creating prediction_results table: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: createPredictionResultsTable() called" << std::endl;
#endif
}

void Database::createTables() {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        // Create datasets table with schema qualification
        session->sql("CREATE TABLE IF NOT EXISTS " + dbSchema + ".datasets ("
                       "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                       "name VARCHAR(255) NOT NULL,"
                       "description TEXT,"
                       "num_rows BIGINT NOT NULL,"
                       "num_features INT NOT NULL,"
                       "num_labels INT NOT NULL,"
                       "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                       "INDEX idx_name (name)"
                       ")").execute();

        // Create dataset_records table with schema qualification
        session->sql("CREATE TABLE IF NOT EXISTS " + dbSchema + ".dataset_records ("
                       "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                       "dataset_id BIGINT NOT NULL,"
                       "row_index INT NOT NULL,"
                       "feature_values BLOB,"
                       "label_values BLOB,"
                       "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                       "INDEX idx_dataset_row (dataset_id, row_index),"
                       "FOREIGN KEY (dataset_id) REFERENCES " + dbSchema + ".datasets(id) ON DELETE CASCADE"
                       ")").execute();

        // Create ai_model_states table with schema qualification
        session->sql("CREATE TABLE IF NOT EXISTS " + dbSchema + ".ai_model_states ("
                       "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                       "dataset_id BIGINT NOT NULL,"
                       "input_data BLOB,"
                       "output_data BLOB,"
                       "hidden_data BLOB,"
                       "hidden_output_data BLOB,"
                       "hidden_error_data BLOB,"
                       "weights_hidden_input BLOB,"
                       "weights_output_hidden BLOB,"
                       "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                       "INDEX idx_dataset_time (dataset_id, created_at)"
                       ")").execute();

        // Create learning_settings table for storing training parameters
        session->sql("CREATE TABLE IF NOT EXISTS " + dbSchema + ".learning_settings ("
                       "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                       "dataset_id BIGINT NOT NULL,"
                       "learning_rate DOUBLE NOT NULL DEFAULT 0.01,"
                       "batch_size INT NOT NULL DEFAULT 32,"
                       "epochs INT NOT NULL DEFAULT 100,"
                       "momentum DOUBLE DEFAULT 0.0,"
                       "weight_decay DOUBLE DEFAULT 0.0,"
                       "dropout_rate DOUBLE DEFAULT 0.0,"
                       "optimizer VARCHAR(50) DEFAULT 'adam',"
                       "loss_function VARCHAR(50) DEFAULT 'mse',"
                       "activation_function VARCHAR(50) DEFAULT 'relu',"
                       "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                       "INDEX idx_dataset (dataset_id)"
                       ")").execute();

        // Create chat_sessions table for chat history
        session->sql("CREATE TABLE IF NOT EXISTS " + dbSchema + ".chat_sessions ("
                       "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                       "session_name VARCHAR(255) DEFAULT 'default',"
                       "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                       "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,"
                       "INDEX idx_session_name (session_name)"
                       ")").execute();

        // Create chat_messages table for storing individual messages
        session->sql("CREATE TABLE IF NOT EXISTS " + dbSchema + ".chat_messages ("
                       "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                       "session_id BIGINT NOT NULL,"
                       "speaker VARCHAR(50) NOT NULL,"
                       "message TEXT NOT NULL,"
                       "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                       "INDEX idx_session_timestamp (session_id, timestamp),"
                       "FOREIGN KEY (session_id) REFERENCES " + dbSchema + ".chat_sessions(id) ON DELETE CASCADE"
                       ")").execute();

        // Create model_states table for storing AI model states
        session->sql("CREATE TABLE IF NOT EXISTS " + dbSchema + ".model_states ("
                       "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                       "session_id BIGINT NOT NULL,"
                       "model_state JSON NOT NULL,"
                       "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                       "INDEX idx_session_time (session_id, created_at),"
                       "FOREIGN KEY (session_id) REFERENCES " + dbSchema + ".chat_sessions(id) ON DELETE CASCADE"
                       ")").execute();

        std::cout << "Created main database tables in schema " << dbSchema << std::endl;
        createPredictionResultsTable();
    } catch (const std::exception& ex) {
        std::cerr << "Error creating tables: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: createTables() called" << std::endl;
    createPredictionResultsTable();
#endif
}

int Database::addDataset(const std::string& datasetName, const std::string& description,
    int numRows, int numFeatures, int numLabels) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        auto result = session->sql("INSERT INTO " + dbSchema + ".datasets (name, description, num_rows, num_features, num_labels) "
                                     "VALUES (?, ?, ?, ?, ?)")
                        .bind(datasetName)
                        .bind(description)
                        .bind(numRows)
                        .bind(numFeatures)
                        .bind(numLabels)
                        .execute();

        // Get the auto-generated ID
        auto idResult = session->sql("SELECT LAST_INSERT_ID() as id").execute();
        auto row = idResult.fetchOne();
        int datasetId = static_cast<int>(row[0].get<int64_t>());

        std::cout << "Added dataset '" << datasetName << "' with ID: " << datasetId << std::endl;
        return datasetId;
    } catch (const std::exception& ex) {
        std::cerr << "Error adding dataset: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation - return a dummy ID
    static int nextId = 1;
    std::cout << "Database stub: addDataset('" << datasetName << "') -> ID: " << nextId << std::endl;
    return nextId++;
#endif
}

void Database::addDatasetRecord(int& datasetId, int rowIndex,
    const std::vector<float>& featureValues,
    const std::vector<float>& labelValues) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        auto featureBlob = vectorToBlob(featureValues);
        auto labelBlob = vectorToBlob(labelValues);

        session->sql("INSERT INTO " + dbSchema + ".dataset_records (dataset_id, row_index, feature_values, label_values) "
                       "VALUES (?, ?, ?, ?)")
                 .bind(datasetId)
                 .bind(rowIndex)
                 .bind(featureBlob.data(), featureBlob.size())
                 .bind(labelBlob.data(), labelBlob.size())
                 .execute();

        // Only log every 100th record to avoid spam during data loading
        if (rowIndex % 100 == 0) {
            std::cout << "Added dataset record for dataset " << datasetId << ", row " << rowIndex << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error adding dataset record: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation - reduce verbosity for bulk operations
    // Only log every 100th record to avoid spam during data loading
    if (rowIndex % 100 == 0) {
        std::cout << "Database stub: addDatasetRecord(datasetId=" << datasetId << ", rowIndex=" << rowIndex << ") - processing..." << std::endl;
    }
#endif
}

Database::DatasetData Database::getDataset(int& datasetId) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        DatasetData data;

        // Get dataset metadata
        auto result = session->sql("SELECT name, num_rows, num_features, num_labels FROM " + dbSchema + ".datasets WHERE id = ?")
                       .bind(datasetId)
                       .execute();

        auto row = result.fetchOne();
        if (!row) {
            throw std::runtime_error("Dataset not found with ID: " + std::to_string(datasetId));
        }

        data.datasetName = row[0].get<std::string>();
        data.numRows = static_cast<long long>(row[1].get<int64_t>());
        data.numFeatures = row[2].get<int>();
        data.numLabels = row[3].get<int>();

        // Get all records for this dataset
        auto recordsResult = session->sql("SELECT row_index, feature_values, label_values FROM " + dbSchema + ".dataset_records "
                                          "WHERE dataset_id = ? ORDER BY row_index")
                             .bind(datasetId)
                             .execute();

        mysqlx::Row recordRow;
        while ((recordRow = recordsResult.fetchOne())) {
            int rowIndex = recordRow[0].get<int>();

            // Convert BLOBs back to vectors
            auto featureBytes = recordRow[1].getRawBytes();
            std::vector<char> featureBlob(featureBytes.begin(), featureBytes.end());
            auto labelBytes = recordRow[2].getRawBytes();
            std::vector<char> labelBlob(labelBytes.begin(), labelBytes.end());

            auto featureVec = blobToVector(featureBlob);
            auto labelVec = blobToVector(labelBlob);

            // Ensure we have enough space in the vectors
            if (static_cast<size_t>(rowIndex) >= data.inputs.size()) {
                data.inputs.resize(rowIndex + 1);
                data.targets.resize(rowIndex + 1);
            }

            data.inputs[rowIndex] = featureVec;
            data.targets[rowIndex] = labelVec;
        }

        std::cout << "Loaded dataset '" << data.datasetName << "' with " << data.numRows << " rows" << std::endl;
        return data;
    } catch (const std::exception& ex) {
        std::cerr << "Error getting dataset: " << ex.what() << std::endl;
        throw;
    }
#else
    DatasetData data;
    // Stub implementation - return empty data
    data.datasetName = "stub_dataset_" + std::to_string(datasetId);
    data.numRows = 0;
    data.numFeatures = 0;
    data.numLabels = 0;
    std::cout << "Database stub: getDataset(" << datasetId << ") -> empty data" << std::endl;
    return data;
#endif
}

void Database::updateDatasetRecordLabels(int& datasetId, int rowIndex, const std::vector<float>& labelValues) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        auto labelBlob = vectorToBlob(labelValues);

        session->sql("UPDATE " + dbSchema + ".dataset_records SET label_values = ? WHERE dataset_id = ? AND row_index = ?")
                .bind(labelBlob.data(), labelBlob.size())
                .bind(datasetId)
                .bind(rowIndex)
                .execute();

        std::cout << "Updated labels for dataset " << datasetId << ", row " << rowIndex << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error updating dataset record labels: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: updateDatasetRecordLabels(datasetId=" << datasetId << ", rowIndex=" << rowIndex << ")" << std::endl;
#endif
}

void Database::clearDatasetRecords(long long& datasetId) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        session->sql("DELETE FROM " + dbSchema + ".dataset_records WHERE dataset_id = ?")
                .bind(datasetId)
                .execute();

        std::cout << "Cleared all records for dataset " << datasetId << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error clearing dataset records: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: clearDatasetRecords(datasetId=" << datasetId << ")" << std::endl;
#endif
}

void Database::saveAIModelState(int& datasetId,
    const std::vector<std::vector<float>>& inputData,
    const std::vector<std::vector<float>>& outputData,
    const std::vector<std::vector<float>>& hiddenData,
    const std::vector<float>& hiddenOutputData,
    const std::vector<float>& hiddenErrorData,
    const std::vector<std::vector<float>>& weightsHiddenInput,
    const std::vector<std::vector<float>>& weightsOutputHidden) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        // Convert matrices and vectors to BLOBs
        auto inputBlob = matrixToBlob(inputData);
        auto outputBlob = matrixToBlob(outputData);
        auto hiddenBlob = matrixToBlob(hiddenData);
        auto hiddenOutputBlob = vectorToBlob(hiddenOutputData);
        auto hiddenErrorBlob = vectorToBlob(hiddenErrorData);
        auto weightsHiddenInputBlob = matrixToBlob(weightsHiddenInput);
        auto weightsOutputHiddenBlob = matrixToBlob(weightsOutputHidden);

        session->sql("INSERT INTO " + dbSchema + ".ai_model_states (dataset_id, input_data, output_data, hidden_data, "
                      "hidden_output_data, hidden_error_data, weights_hidden_input, weights_output_hidden) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?)")
                .bind(datasetId)
                .bind(inputBlob.data(), inputBlob.size())
                .bind(outputBlob.data(), outputBlob.size())
                .bind(hiddenBlob.data(), hiddenBlob.size())
                .bind(hiddenOutputBlob.data(), hiddenOutputBlob.size())
                .bind(hiddenErrorBlob.data(), hiddenErrorBlob.size())
                .bind(weightsHiddenInputBlob.data(), weightsHiddenInputBlob.size())
                .bind(weightsOutputHiddenBlob.data(), weightsOutputHiddenBlob.size())
                .execute();

        std::cout << "Saved AI model state for dataset " << datasetId << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error saving AI model state: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: saveAIModelState(datasetId=" << datasetId << ")" << std::endl;
#endif
}

Database::AIModelState Database::loadLatestAIModelState(int& datasetId) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        AIModelState state;

        // Get the latest model state for this dataset
        auto result = session->sql("SELECT input_data, output_data, hidden_data, hidden_output_data, "
                                    "hidden_error_data, weights_hidden_input, weights_output_hidden "
                                    "FROM " + dbSchema + ".ai_model_states WHERE dataset_id = ? "
                                    "ORDER BY created_at DESC LIMIT 1")
                       .bind(datasetId)
                       .execute();

        auto row = result.fetchOne();
        if (!row) {
            std::cout << "No AI model state found for dataset " << datasetId << std::endl;
            return state; // Return empty state
        }

        // Convert BLOBs back to matrices and vectors
        auto inputBytes = row[0].getRawBytes();
        std::vector<char> inputBlob(inputBytes.begin(), inputBytes.end());
        auto outputBytes = row[1].getRawBytes();
        std::vector<char> outputBlob(outputBytes.begin(), outputBytes.end());
        auto hiddenBytes = row[2].getRawBytes();
        std::vector<char> hiddenBlob(hiddenBytes.begin(), hiddenBytes.end());
        auto hiddenOutputBytes = row[3].getRawBytes();
        std::vector<char> hiddenOutputBlob(hiddenOutputBytes.begin(), hiddenOutputBytes.end());
        auto hiddenErrorBytes = row[4].getRawBytes();
        std::vector<char> hiddenErrorBlob(hiddenErrorBytes.begin(), hiddenErrorBytes.end());
        auto weightsHiddenInputBytes = row[5].getRawBytes();
        std::vector<char> weightsHiddenInputBlob(weightsHiddenInputBytes.begin(), weightsHiddenInputBytes.end());
        auto weightsOutputHiddenBytes = row[6].getRawBytes();
        std::vector<char> weightsOutputHiddenBlob(weightsOutputHiddenBytes.begin(), weightsOutputHiddenBytes.end());

        state.inputData = blobToMatrix(inputBlob);
        state.outputData = blobToMatrix(outputBlob);
        state.hiddenData = blobToMatrix(hiddenBlob);
        state.hiddenOutputData = blobToVector(hiddenOutputBlob);
        state.hiddenErrorData = blobToVector(hiddenErrorBlob);
        state.weightsHiddenInput = blobToMatrix(weightsHiddenInputBlob);
        state.weightsOutputHidden = blobToMatrix(weightsOutputHiddenBlob);

        std::cout << "Loaded latest AI model state for dataset " << datasetId << std::endl;
        return state;
    } catch (const std::exception& ex) {
        std::cerr << "Error loading AI model state: " << ex.what() << std::endl;
        throw;
    }
#else
    AIModelState state;
    // Stub implementation - return empty state
    std::cout << "Database stub: loadLatestAIModelState(" << datasetId << ") -> empty state" << std::endl;
    return state;
#endif
}

void Database::savePredictionResults(int& datasetId, int sampleIndex,
    const std::vector<float>& inputFeatures,
    const std::vector<float>& actualTargets,
    const std::vector<float>& predictedTargets) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        // Convert vectors to BLOBs
        auto inputBlob = vectorToBlob(inputFeatures);
        auto actualBlob = vectorToBlob(actualTargets);
        auto predictedBlob = vectorToBlob(predictedTargets);

        session->sql("INSERT INTO " + dbSchema + ".prediction_results (dataset_id, sample_index, input_features, actual_targets, predicted_targets) "
                      "VALUES (?, ?, ?, ?, ?)")
                .bind(datasetId)
                .bind(sampleIndex)
                .bind(inputBlob.data(), inputBlob.size())
                .bind(actualBlob.data(), actualBlob.size())
                .bind(predictedBlob.data(), predictedBlob.size())
                .execute();

        std::cout << "Saved prediction results for dataset " << datasetId << ", sample " << sampleIndex << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error saving prediction results: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: savePredictionResults(datasetId=" << datasetId << ", sampleIndex=" << sampleIndex << ")" << std::endl;
#endif
}

bool Database::getDatasetRecords(long long datasetId,
    std::vector<std::vector<float>>& loadedInputs,
    std::vector<std::vector<float>>& loadedTargets,
    long long& loadedNumSamples,
    int& loadedInputSize, int& loadedOutputSize) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        // Get dataset metadata first
        auto metaResult = session->sql("SELECT num_features, num_labels, num_rows FROM " + dbSchema + ".datasets WHERE id = ?")
                           .bind(datasetId)
                           .execute();

        auto metaRow = metaResult.fetchOne();
        if (!metaRow) {
            std::cerr << "Dataset not found: " << datasetId << std::endl;
            return false;
        }

        loadedInputSize = metaRow[0].get<int>();
        loadedOutputSize = metaRow[1].get<int>();
        loadedNumSamples = static_cast<long long>(metaRow[2].get<int64_t>());

        // Get all records
        auto recordsResult = session->sql("SELECT row_index, feature_values, label_values FROM " + dbSchema + ".dataset_records "
                                          "WHERE dataset_id = ? ORDER BY row_index")
                             .bind(datasetId)
                             .execute();

        loadedInputs.clear();
        loadedTargets.clear();
        loadedInputs.resize(loadedNumSamples);
        loadedTargets.resize(loadedNumSamples);

        mysqlx::Row recordRow;
        while ((recordRow = recordsResult.fetchOne())) {
            int rowIndex = recordRow[0].get<int>();

            // Convert BLOBs back to vectors
            auto featureBytes = recordRow[1].getRawBytes();
            std::vector<char> featureBlob(featureBytes.begin(), featureBytes.end());
            auto labelBytes = recordRow[2].getRawBytes();
            std::vector<char> labelBlob(labelBytes.begin(), labelBytes.end());

            loadedInputs[rowIndex] = blobToVector(featureBlob);
            loadedTargets[rowIndex] = blobToVector(labelBlob);
        }

        std::cout << "Loaded " << loadedNumSamples << " records for dataset " << datasetId << std::endl;
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "Error getting dataset records: " << ex.what() << std::endl;
        return false;
    }
#else
    // Stub implementation - return false
    std::cout << "Database stub: getDatasetRecords(datasetId=" << datasetId << ")" << std::endl;
    return false;
#endif
}

// NEW: Methods for learning settings management
void Database::saveLearningSettings(int& datasetId, double learningRate, int batchSize, int epochs,
                                    double momentum, double weightDecay, double dropoutRate,
                                    const std::string& optimizer, const std::string& lossFunction,
                                    const std::string& activationFunction) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        session->sql("INSERT INTO " + dbSchema + ".learning_settings (dataset_id, learning_rate, batch_size, epochs, "
                      "momentum, weight_decay, dropout_rate, optimizer, loss_function, activation_function) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
                .bind(datasetId)
                .bind(learningRate)
                .bind(batchSize)
                .bind(epochs)
                .bind(momentum)
                .bind(weightDecay)
                .bind(dropoutRate)
                .bind(optimizer)
                .bind(lossFunction)
                .bind(activationFunction)
                .execute();

        std::cout << "Saved learning settings for dataset " << datasetId << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error saving learning settings: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: saveLearningSettings(datasetId=" << datasetId << ")" << std::endl;
#endif
}

Database::LearningSettings Database::loadLearningSettings(int& datasetId) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        LearningSettings settings;

        // Get the latest learning settings for this dataset
        auto result = session->sql("SELECT learning_rate, batch_size, epochs, momentum, weight_decay, "
                                    "dropout_rate, optimizer, loss_function, activation_function "
                                    "FROM " + dbSchema + ".learning_settings WHERE dataset_id = ? "
                                    "ORDER BY created_at DESC LIMIT 1")
                       .bind(datasetId)
                       .execute();

        auto row = result.fetchOne();
        if (!row) {
            std::cout << "No learning settings found for dataset " << datasetId << ", using defaults" << std::endl;
            return LearningSettings{0.01, 32, 100, 0.0, 0.0, 0.0, "adam", "mse", "relu"}; // Return defaults
        }

        settings.learningRate = row[0].get<double>();
        settings.batchSize = row[1].get<int>();
        settings.epochs = row[2].get<int>();
        settings.momentum = row[3].get<double>();
        settings.weightDecay = row[4].get<double>();
        settings.dropoutRate = row[5].get<double>();
        settings.optimizer = row[6].get<std::string>();
        settings.lossFunction = row[7].get<std::string>();
        settings.activationFunction = row[8].get<std::string>();

        std::cout << "Loaded learning settings for dataset " << datasetId << std::endl;
        return settings;
    } catch (const std::exception& ex) {
        std::cerr << "Error loading learning settings: " << ex.what() << std::endl;
        throw;
    }
#else
    LearningSettings settings;
    // Stub implementation - return defaults
    settings.learningRate = 0.01;
    settings.batchSize = 32;
    settings.epochs = 100;
    settings.momentum = 0.0;
    settings.weightDecay = 0.0;
    settings.dropoutRate = 0.0;
    settings.optimizer = "adam";
    settings.lossFunction = "mse";
    settings.activationFunction = "relu";
    std::cout << "Database stub: loadLearningSettings(" << datasetId << ") -> default settings" << std::endl;
    return settings;
#endif
}

// API key management methods
void Database::saveAPIKey(const std::string& apiKey, const std::string& description, bool isActive) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        session->sql("CREATE TABLE IF NOT EXISTS " + dbSchema + ".api_keys ("
                        "id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                        "api_key VARCHAR(255) NOT NULL UNIQUE,"
                        "description TEXT,"
                        "is_active BOOLEAN DEFAULT TRUE,"
                        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                        "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,"
                        "INDEX idx_api_key (api_key),"
                        "INDEX idx_active (is_active)"
                        ")").execute();

        session->sql("INSERT INTO " + dbSchema + ".api_keys (api_key, description, is_active) VALUES (?, ?, ?) "
                        "ON DUPLICATE KEY UPDATE description = VALUES(description), is_active = VALUES(is_active), updated_at = CURRENT_TIMESTAMP")
                    .bind(apiKey)
                    .bind(description)
                    .bind(isActive)
                    .execute();

        std::cout << "Saved API key: " << apiKey.substr(0, 8) << "..." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error saving API key: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: saveAPIKey('" << apiKey.substr(0, 8) << "...')" << std::endl;
#endif
}

bool Database::validateAPIKey(const std::string& apiKey) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        auto result = session->sql("SELECT COUNT(*) FROM " + dbSchema + ".api_keys WHERE api_key = ? AND is_active = TRUE")
                        .bind(apiKey)
                        .execute();

        auto row = result.fetchOne();
        if (row) {
            return row[0].get<int64_t>() > 0;
        }
        return false;
    } catch (const std::exception& ex) {
        std::cerr << "Error validating API key: " << ex.what() << std::endl;
        return false;
    }
#else
    // Stub implementation - accept any non-empty key
    std::cout << "Database stub: validateAPIKey('" << apiKey.substr(0, 8) << "...') -> " << (!apiKey.empty()) << std::endl;
    return !apiKey.empty();
#endif
}

void Database::deactivateAPIKey(const std::string& apiKey) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        session->sql("UPDATE " + dbSchema + ".api_keys SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP WHERE api_key = ?")
                    .bind(apiKey)
                    .execute();

        std::cout << "Deactivated API key: " << apiKey.substr(0, 8) << "..." << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error deactivating API key: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: deactivateAPIKey('" << apiKey.substr(0, 8) << "...')" << std::endl;
#endif
}

std::vector<std::pair<std::string, std::string>> Database::getAPIKeys() {
    ensureConnection();
    std::vector<std::pair<std::string, std::string>> keys;
#ifdef USE_MYSQL
    try {
        auto result = session->sql("SELECT api_key, description FROM " + dbSchema + ".api_keys WHERE is_active = TRUE ORDER BY created_at DESC")
                        .execute();

        mysqlx::Row row;
        while ((row = result.fetchOne())) {
            std::string apiKey = row[0].get<std::string>();
            std::string description = row[1].get<std::string>();
            keys.emplace_back(apiKey, description);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error getting API keys: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation - return empty list
    std::cout << "Database stub: getAPIKeys() -> empty list" << std::endl;
#endif
    return keys;
}

// Chat history management methods
void Database::saveChatMessage(int sessionId, const std::string& speaker, const std::string& message) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        session->sql("INSERT INTO " + dbSchema + ".chat_messages (session_id, speaker, message) VALUES (?, ?, ?)")
                .bind(sessionId)
                .bind(speaker)
                .bind(message)
                .execute();

        // Update session timestamp
        session->sql("UPDATE " + dbSchema + ".chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?")
                .bind(sessionId)
                .execute();

        // Note: trainer is not available in Database class, removed verbose logging
    } catch (const std::exception& ex) {
        std::cerr << "Error saving chat message: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: saveChatMessage(sessionId=" << sessionId << ", speaker='" << speaker << "')" << std::endl;
#endif
}

std::vector<std::pair<std::string, std::string>> Database::loadChatHistory(int sessionId) {
    ensureConnection();
    std::vector<std::pair<std::string, std::string>> conversation;
#ifdef USE_MYSQL
    try {
        auto result = session->sql("SELECT speaker, message FROM " + dbSchema + ".chat_messages WHERE session_id = ? ORDER BY timestamp")
                       .bind(sessionId)
                       .execute();

        mysqlx::Row row;
        while ((row = result.fetchOne())) {
            std::string speaker = row[0].get<std::string>();
            std::string message = row[1].get<std::string>();
            conversation.emplace_back(speaker, message);
        }

        // Note: trainer is not available in Database class, removed verbose logging
    } catch (const std::exception& ex) {
        std::cerr << "Error loading chat history: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation - return empty conversation
    std::cout << "Database stub: loadChatHistory(sessionId=" << sessionId << ") -> empty conversation" << std::endl;
#endif
    return conversation;
}

void Database::saveModelState(int sessionId, const nlohmann::json& modelState) {
    ensureConnection();
#ifdef USE_MYSQL
    try {
        std::string jsonStr = modelState.dump();
        session->sql("INSERT INTO " + dbSchema + ".model_states (session_id, model_state) VALUES (?, ?)")
                .bind(sessionId)
                .bind(jsonStr)
                .execute();

        // Note: trainer is not available in Database class, removed verbose logging
    } catch (const std::exception& ex) {
        std::cerr << "Error saving model state: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation
    std::cout << "Database stub: saveModelState(sessionId=" << sessionId << ")" << std::endl;
#endif
}

nlohmann::json Database::loadLatestModelState(int sessionId) {
    ensureConnection();
    nlohmann::json modelState;
#ifdef USE_MYSQL
    try {
        auto result = session->sql("SELECT model_state FROM " + dbSchema + ".model_states WHERE session_id = ? ORDER BY created_at DESC LIMIT 1")
                       .bind(sessionId)
                       .execute();

        auto row = result.fetchOne();
        if (row) {
            std::string jsonStr = row[0].get<std::string>();
            modelState = nlohmann::json::parse(jsonStr);
            // Note: trainer is not available in Database class, removed verbose logging
        } else {
            // Note: trainer is not available in Database class, removed verbose logging
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error loading model state: " << ex.what() << std::endl;
        throw;
    }
#else
    // Stub implementation - return empty JSON
    std::cout << "Database stub: loadLatestModelState(sessionId=" << sessionId << ") -> empty state" << std::endl;
#endif
    return modelState;
}

// Connection management methods
bool Database::isConnectionHealthy() {
#ifdef USE_MYSQL
    if (!session) {
        return false;
    }
    try {
        // Try a simple query to check if connection is alive
        session->sql("SELECT 1").execute();
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "Connection health check failed: " << ex.what() << std::endl;
        return false;
    }
#else
    // Stub implementation - always return true
    return true;
#endif
}

void Database::reconnect() {
#ifdef USE_MYSQL
    try {
        std::cout << "Attempting to reconnect to database..." << std::endl;

        // Close existing session if it exists
        if (session) {
            session->close();
            delete session;
            session = nullptr;
        }

        // Convert SSLMode enum to mysqlx::SSLMode
        mysqlx::SSLMode mysqlSslMode;
        switch (sslMode) {
            case SSLMode::DISABLED:
                mysqlSslMode = mysqlx::SSLMode::DISABLED;
                break;
            case SSLMode::REQUIRED:
                mysqlSslMode = mysqlx::SSLMode::REQUIRED;
                break;
            case SSLMode::VERIFY_CA:
                mysqlSslMode = mysqlx::SSLMode::VERIFY_CA;
                break;
            case SSLMode::VERIFY_IDENTITY:
                mysqlSslMode = mysqlx::SSLMode::VERIFY_IDENTITY;
                break;
            default:
                mysqlSslMode = mysqlx::SSLMode::DISABLED;
                break;
        }

        // First, connect without schema to ensure database exists
        mysqlx::Session tempSession(dbHost, dbPort, dbUser, dbPassword);
        tempSession.sql("CREATE DATABASE IF NOT EXISTS " + dbSchema).execute();
        tempSession.close();

        // Now establish connection to the specific schema
        session = new mysqlx::Session(dbHost, dbPort, dbUser, dbPassword, dbSchema);
        std::cout << "Database reconnection successful to " << dbHost << ":" << dbPort << "/" << dbSchema << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Database reconnection failed: " << ex.what() << std::endl;
        throw std::runtime_error("Failed to reconnect to database: " + std::string(ex.what()));
    }
#else
    // Stub implementation
    std::cout << "Database stub: reconnect() called" << std::endl;
#endif
}

void Database::ensureConnection() {
    if (!isConnectionHealthy()) {
        reconnect();
    }
}

// Destructor for proper cleanup
Database::~Database() {
#ifdef USE_MYSQL
    if (session) {
        session->close();
        delete session;
        session = nullptr;
    }
#endif
}

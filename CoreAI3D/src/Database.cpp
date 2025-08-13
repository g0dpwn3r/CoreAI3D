#include "Database.hpp"

// Use mysqlx::expr directly
using mysqlx::expr;

// Constructor
// Changed 'int port' to 'unsigned int port' to match Database.hpp declaration
Database::Database(const std::string& host, unsigned int port,
    const std::string& user, const std::string password,
    const std::string& schemaName, mysqlx::SSLMode ssl)
    : session(mysqlx::SessionOption::HOST, host,
        mysqlx::SessionOption::PORT, port,
        mysqlx::SessionOption::USER, user,
        mysqlx::SessionOption::PWD, password,
        mysqlx::SessionOption::SSL_MODE, ssl),
    dbSchema(schemaName), // Initialize 'dbSchema' member
    schema(session.getSchema(schemaName)) // Initialize 'schema' directly in the initializer list
{
    try {
        // Test the session by attempting a simple operation
        session.sql("SELECT 1").execute();
    }
    catch (const std::exception& e) {
        throw std::runtime_error("Database connection failed: " + std::string(e.what()));
    }
}

// Helper to convert std::vector<std::vector<float>> to mysqlx::bytes
mysqlx::bytes Database::matrixToBlob(const std::vector<std::vector<float>>& matrix) {
    std::vector<char> blob_data;
    if (matrix.empty()) {
        int rows = 0;
        int cols = 0;
        blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(&rows), reinterpret_cast<const char*>(&rows) + sizeof(int));
        blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(&cols), reinterpret_cast<const char*>(&cols) + sizeof(int));
        return mysqlx::bytes(reinterpret_cast<const mysqlx::abi2::r0::common::byte*>(blob_data.data()), blob_data.size());
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

    return mysqlx::bytes(reinterpret_cast<const mysqlx::abi2::r0::common::byte*>(blob_data.data()), blob_data.size());
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

// Helper to convert std::vector<float> to mysqlx::bytes
mysqlx::bytes Database::vectorToBlob(const std::vector<float>& vec) {
    std::vector<char> blob_data;
    int size = static_cast<int>(vec.size());

    // Store size
    blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(&size), reinterpret_cast<const char*>(&size) + sizeof(int));

    // Store data
    blob_data.insert(blob_data.end(), reinterpret_cast<const char*>(vec.data()), reinterpret_cast<const char*>(vec.data()) + size * sizeof(float));

    return mysqlx::bytes(reinterpret_cast<const mysqlx::abi2::r0::common::byte*>(blob_data.data()), blob_data.size());
}


// Helper to convert mysqlx::bytes to std::vector<float>
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
    try {
        session.sql(R"(
            CREATE TABLE IF NOT EXISTS prediction_results (
                result_id INT AUTO_INCREMENT PRIMARY KEY,
                dataset_id BIGINT NOT NULL COMMENT 'Foreign Key linking to the datasets table',
                sample_index INT NOT NULL COMMENT 'The index of the sample within the dataset',
                input_features JSON COMMENT 'JSON array of input feature values',
                actual_targets JSON COMMENT 'JSON array of actual target values',
                predicted_targets JSON COMMENT 'JSON array of predicted target values',
                prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the prediction was saved',
                UNIQUE (dataset_id, sample_index),
                CONSTRAINT fk_prediction_dataset
                    FOREIGN KEY (dataset_id)
                    REFERENCES datasets(dataset_id)
                    ON DELETE CASCADE
            );
        )").execute();
        std::cout << "Table 'prediction_results' created (if it didn't exist)." << std::endl;
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error creating prediction_results table: " << err.what() << std::endl;
        throw;
    }
}


void Database::createTables() {
    try {
        // Create 'datasets' table
        session.sql(R"(
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                dataset_name VARCHAR(255) NOT NULL UNIQUE COMMENT 'Unique name for the dataset (e.g., historical_stock_prices, sensor_readings)',
                description TEXT COMMENT 'Detailed description of the dataset, e.g., source, collection method',
                num_rows BIGINT COMMENT 'Total number of rows/samples in the dataset',
                num_features INT COMMENT 'Number of input features/columns for this dataset',
                num_labels INT COMMENT 'Number of output labels/target columns for this dataset',
                uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when the dataset metadata was added'
            );
        )").execute();
        std::cout << "Table 'datasets' created (if it didn't exist)." << std::endl;

        // Create 'dataset_records' table
        session.sql(R"(
            CREATE TABLE IF NOT EXISTS dataset_records (
                record_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                dataset_id BIGINT NOT NULL COMMENT 'Foreign Key linking to the datasets table',
                row_index INT NOT NULL COMMENT 'The original row number of this record in the CSV file (0-indexed or 1-indexed)',
                feature_values JSON COMMENT 'JSON array of input feature values for this row (e.g., [val1, val2, ...])',
                label_values JSON COMMENT 'JSON array of output label/target values for this row (e.g., [label1, label2, ...])',
                UNIQUE (dataset_id, row_index),
                CONSTRAINT fk_dataset_record
                    FOREIGN KEY (dataset_id)
                    REFERENCES datasets(dataset_id)
                    ON DELETE CASCADE
            );
        )").execute();
        std::cout << "Table 'dataset_records' created (if it didn't exist)." << std::endl;

        // Create 'ai_model_states' table
        session.sql(R"(
            CREATE TABLE IF NOT EXISTS ai_model_states (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Timestamp when this model state was saved',
                dataset_id BIGINT COMMENT 'Foreign Key referencing the datasets table, indicating which dataset the model was trained on',
                input_data LONGBLOB COMMENT 'Serialized input data matrix (internal model state)',
                output_data LONGBLOB COMMENT 'Serialized output data matrix (internal model state)',
                hidden_data LONGBLOB COMMENT 'Serialized hidden layer activations (internal model state)',
                hidden_output_data LONGBLOB COMMENT 'Serialized hidden layer output (internal model state)',
                hidden_error_data LONGBLOB COMMENT 'Serialized hidden layer error (internal model state)',
                weights_hidden_input LONGBLOB COMMENT 'Serialized weights matrix connecting input to hidden layer',
                weights_output_hidden LONGBLOB COMMENT 'Serialized weights matrix connecting hidden to output layer',
                CONSTRAINT fk_model_state_dataset
                    FOREIGN KEY (dataset_id)
                    REFERENCES datasets(dataset_id)
                    ON DELETE SET NULL
            );
        )").execute();
        std::cout << "Table 'ai_model_states' created (if it didn't exist)." << std::endl;

        // NEW: Create prediction_results table
        createPredictionResultsTable();
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error creating tables: " << err.what() << std::endl;
        throw; // Re-throw the exception after logging
    }
}

int Database::addDataset(const std::string& datasetName, const std::string& description,
    int numRows, int numFeatures, int numLabels) { // numRows passed as int here, but DB is BIGINT
    try {
        mysqlx::Table datasetsTable = schema.getTable("datasets");
        // 1. Check if dataset_name already exists
        mysqlx::RowResult res = session.sql(std::string("SELECT dataset_id FROM datasets WHERE dataset_name = '") + datasetName + std::string("'")) // Fixed string concat
            .execute();
        mysqlx::Row row = res.fetchOne();

        if (row) {
            // Dataset exists, update it
            long long existingDatasetId = row[0].get<long long>(); // Changed to long long
            datasetsTable.update()
                .set("description", description)
                .set("num_rows", static_cast<long long>(numRows)) // Cast to long long
                .set("num_features", numFeatures)
                .set("num_labels", numLabels)
                .set("uploaded_at", mysqlx::expr("NOW()"))
                .where("dataset_id = :id")
                .bind("id", existingDatasetId)
                .execute();
            std::cout << "Updated existing dataset '" << datasetName << "' with ID: " << existingDatasetId << std::endl;
            return static_cast<int>(existingDatasetId); // Cast back to int for return
        }
        else {
            // Dataset does not exist, insert new
            mysqlx::Result insertRes = datasetsTable.insert("dataset_name", "description", "num_rows", "num_features", "num_labels")
                .values(datasetName, description, static_cast<long long>(numRows), numFeatures, numLabels) // Cast to long long
                .execute();
            long long newDatasetId = insertRes.getAutoIncrementValue(); // Get as long long
            std::cout << "Added new dataset '" << datasetName << "' with ID: " << newDatasetId << std::endl;
            return static_cast<int>(newDatasetId); // Cast back to int for return
        }
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error handling dataset (add/update): " << err.what() << std::endl;
        throw;
    }
}

void Database::addDatasetRecord(int& datasetId, int rowIndex,
    const std::vector<float>& featureValues,
    const std::vector<float>& labelValues) {
    try {
        nlohmann::json features_json = featureValues;
        nlohmann::json labels_json = labelValues;

        mysqlx::Table datasetRecordsTable = schema.getTable("dataset_records");
        datasetRecordsTable.insert("dataset_id", "row_index", "feature_values", "label_values")
            .values(static_cast<long long>(datasetId), rowIndex, features_json.dump(), labels_json.dump()) // Cast datasetId to long long
            .execute();
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error adding dataset record: " << err.what() << std::endl;
        throw;
    }
}

Database::DatasetData Database::getDataset(int& datasetId) {
    DatasetData data;
    try {
        // Fetch dataset metadata
        mysqlx::RowResult res_meta = session.sql(std::string("SELECT dataset_name, num_rows, num_features, num_labels FROM datasets WHERE dataset_id = ") + std::to_string(datasetId)) // Fixed string concat
            .execute();
        mysqlx::Row row_meta = res_meta.fetchOne();

        if (!row_meta) {
            std::cerr << "Error: Dataset with ID " << datasetId << " not found.\n";
            return data;
        }

        data.datasetName = row_meta[0].get<std::string>();
        data.numRows = row_meta[1].get<long long>(); // Changed to long long
        data.numFeatures = row_meta[2].get<int>();
        data.numLabels = row_meta[3].get<int>();

        // Fetch dataset records
        mysqlx::RowResult res_records = session.sql(std::string("SELECT feature_values, label_values FROM dataset_records WHERE dataset_id = ") + std::to_string(datasetId) + std::string(" ORDER BY row_index ASC")) // Fixed string concat
            .execute();

        for (mysqlx::Row row_record : res_records) {
            mysqlx::DbDoc feature_doc = row_record[0].get<mysqlx::DbDoc>();
            mysqlx::DbDoc label_doc = row_record[1].get<mysqlx::DbDoc>();

            std::ostringstream ss_features;
            ss_features << feature_doc;
            std::string feature_json_str = ss_features.str();

            std::ostringstream ss_labels;
            ss_labels << label_doc;
            std::string label_json_str = ss_labels.str();

            
            json features_json = nlohmann::json::parse(feature_json_str);
            json labels_json = nlohmann::json::parse(label_json_str);

            std::vector<float> features = features_json.get<std::vector<float>>();
            std::vector<float> labels = labels_json.get<std::vector<float>>();

            data.inputs.push_back(features);
            data.targets.push_back(labels);
        }
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error getting dataset from database: " << err.what() << std::endl;
        throw;
    }
    catch (const nlohmann::json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        throw;
    }
    return data;
}

void Database::updateDatasetRecordLabels(int& datasetId, int rowIndex, const std::vector<float>& labelValues) {
    try {
        nlohmann::json labels_json = labelValues;
        mysqlx::Table datasetRecordsTable = schema.getTable("dataset_records");
        datasetRecordsTable.update()
            .set("label_values", labels_json.dump())
            .where("dataset_id = :dataset_id AND row_index = :row_index")
            .bind("dataset_id", static_cast<long long>(datasetId)) // Cast to long long
            .bind("row_index", rowIndex)
            .execute();
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error updating dataset record labels: " << err.what() << std::endl;
        throw;
    }
}

// Corrected parameter type to long long to match declaration in Database.hpp
void Database::clearDatasetRecords(long long& datasetId) {
    try {
        session.sql(std::string("DELETE FROM dataset_records WHERE dataset_id = ") + std::to_string(datasetId)) // Fixed string concat
            .execute();
        std::cout << "Cleared existing records for dataset ID: " << datasetId << std::endl;
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error clearing dataset records: " << err.what() << std::endl;
        throw;
    }
}

void Database::saveAIModelState(int& datasetId,
    const std::vector<std::vector<float>>& inputData,
    const std::vector<std::vector<float>>& outputData,
    const std::vector<std::vector<float>>& hiddenData,
    const std::vector<float>& hiddenOutputData,
    const std::vector<float>& hiddenErrorData,
    const std::vector<std::vector<float>>& weightsHiddenInput,
    const std::vector<std::vector<float>>& weightsOutputHidden) {
    try {
        // Use raw SQL query to insert, bypassing TableInsert::values
        std::stringstream ss;
        ss << "INSERT INTO ai_model_states ("
            << "dataset_id, timestamp, input_data, output_data, "
            << "hidden_data, hidden_output_data, hidden_error_data, "
            << "weights_hidden_input, weights_output_hidden) VALUES ("
            << "?, NOW(), ?, ?, ?, ?, ?, ?, ?)"; // NOW() directly in SQL

        session.sql(ss.str())
            .bind(static_cast<long long>(datasetId)) // Cast datasetId to long long
            .bind(matrixToBlob(inputData))
            .bind(matrixToBlob(outputData))
            .bind(matrixToBlob(hiddenData))
            .bind(vectorToBlob(hiddenOutputData))
            .bind(vectorToBlob(hiddenErrorData))
            .bind(matrixToBlob(weightsHiddenInput))
            .bind(matrixToBlob(weightsOutputHidden))
            .execute();

        std::cout << "AI model state saved for dataset_id: " << datasetId << std::endl;
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error saving AI model state: " << err.what() << std::endl;
        throw;
    }
}

Database::AIModelState Database::loadLatestAIModelState(int& datasetId) {
    AIModelState state;
    try {
        // Use raw SQL query for selection to ensure correct ordering and limit
        mysqlx::RowResult res = session.sql(std::string(R"(
            SELECT input_data, output_data, hidden_data, hidden_output_data, hidden_error_data,
                   weights_hidden_input, weights_output_hidden
            FROM ai_model_states
            WHERE dataset_id = )") + std::to_string(datasetId) + std::string(R"(
            ORDER BY timestamp DESC
            LIMIT 1;
        )")) // Fixed string concat
            .execute();

        mysqlx::Row row = res.fetchOne();

        if (row) {
            // Haal blobs op als std::string en converteer naar vector<uint8_t>
            std::string inputDataBlob = row[0].get<std::string>();
            std::string outputDataBlob = row[1].get<std::string>();
            std::string hiddenDataBlob = row[2].get<std::string>();
            std::string hiddenOutputDataBlob = row[3].get<std::string>();
            std::string hiddenErrorDataBlob = row[4].get<std::string>();
            std::string weightsHiddenInputBlob = row[5].get<std::string>();
            std::string weightsOutputHiddenBlob = row[6].get<std::string>();

            // Converteer naar je datastructuren
            state.inputData = blobToMatrix(std::vector<char>(inputDataBlob.begin(), inputDataBlob.end()));
            state.outputData = blobToMatrix(std::vector<char>(outputDataBlob.begin(), outputDataBlob.end()));
            state.hiddenData = blobToMatrix(std::vector<char>(hiddenDataBlob.begin(), hiddenDataBlob.end()));
            state.hiddenOutputData = blobToVector(std::vector<char>(hiddenOutputDataBlob.begin(), hiddenOutputDataBlob.end()));
            state.hiddenErrorData = blobToVector(std::vector<char>(hiddenErrorDataBlob.begin(), hiddenErrorDataBlob.end()));
            state.weightsHiddenInput = blobToMatrix(std::vector<char>(weightsHiddenInputBlob.begin(), weightsHiddenInputBlob.end()));
            state.weightsOutputHidden = blobToMatrix(std::vector<char>(weightsOutputHiddenBlob.begin(), weightsOutputHiddenBlob.end()));
            std::cout << "Loaded latest AI model state for dataset_id: " << datasetId << std::endl;
        }
        else {
            std::cout << "No model state found for dataset_id: " << datasetId << std::endl;
        }
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error loading AI model state: " << err.what() << std::endl;
    }
    return state;
}

// NEW: Implementation for saving prediction results
void Database::savePredictionResults(int& datasetId, int sampleIndex,
    const std::vector<float>& inputFeatures,
    const std::vector<float>& actualTargets,
    const std::vector<float>& predictedTargets) {
    try {
        nlohmann::json input_features_json = inputFeatures;
        nlohmann::json actual_targets_json = actualTargets;
        nlohmann::json predicted_targets_json = predictedTargets;

        mysqlx::Table predictionResultsTable = schema.getTable("prediction_results");

        session.sql(R"(
            INSERT INTO prediction_results (dataset_id, sample_index, input_features, actual_targets, predicted_targets)
            VALUES (?, ?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                input_features = VALUES(input_features),
                actual_targets = VALUES(actual_targets),
                predicted_targets = VALUES(predicted_targets),
                prediction_timestamp = NOW();
        )")
            .bind(static_cast<long long>(datasetId)) // Cast to long long
            .bind(sampleIndex)
            .bind(input_features_json.dump())
            .bind(actual_targets_json.dump())
            .bind(predicted_targets_json.dump())
            .execute();

    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Error saving prediction results: " << err.what() << std::endl;
        throw;
    }
}

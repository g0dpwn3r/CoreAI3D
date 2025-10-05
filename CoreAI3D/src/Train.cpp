#include "Train.hpp"
// Constructor for online mode (with database)
Training::Training(const std::string& dbHost, unsigned int dbPort, const std::string& dbUser,
    const std::string dbPassword, const std::string& dbSchema, mysqlx::SSLMode ssl, bool createTables)
    : dbManager(std::make_unique<Database>(dbHost, dbPort, dbUser, dbPassword, dbSchema, ssl)),
    isOfflineMode(false), currentDatasetId(-1), numSamples(0), inputSize(0), outputSize(0),
    original_data_global_min(std::numeric_limits<float>::max()), original_data_global_max(std::numeric_limits<float>::lowest()), // Initialize
    layers(0), neurons(0), learningRate(0.0), min(0.0f), max(0.0f), last_known_timestamp(0.0f), // Initialize
    gen(std::random_device{}()) // Initialize random number generator
{
    if (dbManager && createTables) {
        dbManager->createTables(); // Ensure tables exist on startup
    }
}

// Constructor for offline mode (no database)
Training::Training(bool isOffline)
    : dbManager(nullptr), // No database manager in offline mode
    isOfflineMode(isOffline), currentDatasetId(-1), numSamples(0), inputSize(0), outputSize(0),
    original_data_global_min(std::numeric_limits<float>::max()), original_data_global_max(std::numeric_limits<float>::lowest()), // Initialize
    layers(0), neurons(0), learningRate(0.0), min(0.0f), max(0.0f), last_known_timestamp(0.0f), // Initialize
    gen(std::random_device{}()) // Initialize random number generator
{
    if (isOfflineMode) {
        std::cout << "Training initialized in OFFLINE mode. Database operations are disabled." << std::endl;
    }
}

// Helper to initialize Language processor (called from main.cpp)
void Training::initializeLanguageProcessor(std::string& embedingFile, int& embeddingDim, std::string& dbHost, int& dbPort,
    std::string& dbUser, std::string& dbPassword, std::string& dbSchema, mysqlx::SSLMode ssl, std::string& lang, int& inputSize, int& outputSize, int& layers, int& neurons) {
    this->langProc = std::make_unique<Language>(embedingFile, embeddingDim, dbHost, dbPort, dbUser, dbPassword, dbSchema, ssl, lang, inputSize, outputSize, layers, neurons);
    // After initialization, you might want to load the embeddings here.
    if (this->langProc) {
        this->langProc->loadWordEmbeddingsFromFile(embedingFile, embeddingDim);
        this->langProc->createEmbeddingsByLang(embeddingDim); // Ensure this populates embeddingsByLang_map
    }
}


long long Training::countLines(const std::string& filename, bool hasHeader) {
    std::ifstream file(filename);
    if (!file.is_open()) return 0;

    long long lines = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++lines;
    }
    if (hasHeader && lines > 0) --lines;
    return lines;
}

float Training::convertDateTimeToTimestamp(const std::string& datetime_str) {
    // Remove potential "0;UTC" at the end if present
    std::string cleaned_str = datetime_str;
    size_t found = cleaned_str.find(";UTC");
    if (found != std::string::npos) {
        cleaned_str = cleaned_str.substr(0, found);
    }
    // Replace ';' with space if it's the separator in "YYYY-MM-DD HH:MM:SS;0;UTC"
    std::replace(cleaned_str.begin(), cleaned_str.end(), ';', ' ');

    std::tm tm = {};
    std::istringstream ss(cleaned_str);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S"); // Strict format
    if (ss.fail()) {
        // Try parsing without seconds if it fails (e.g., "YYYY-MM-DD HH:MM")
        ss.clear();
        ss.str(cleaned_str);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M");
        if (ss.fail()) {
            throw std::runtime_error(std::string("Invalid datetime format: ") + cleaned_str);
        }
    }

    return static_cast<float>(std::mktime(&tm));
}

void Training::printProgressBar(const std::string& prefix, long long current, long long total, int barWidth) {
    if (total == 0) return; // Avoid division by zero

    float progress_percent = static_cast<float>(current) / total;
    int filled_width = static_cast<int>(barWidth * progress_percent);

    std::cout << "\r" << prefix << " [";
    for (int s = 0; s < barWidth; ++s) {
        if (s < filled_width) {
            std::cout << "=";
        }
        else {
            std::cout << " ";
        }
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress_percent * 100.0) << "%";
    std::cout.flush();
}

// detectMaxSeqLength is a static member of Language, not Training
// long long Training::countLines(const std::string& filename, bool hasHeader) is already defined above

bool Training::loadCSV(const std::string& filename, long long numSamplesToLoad, int outputSizeParam, bool hasHeader, bool containsText, const char& delimitor, const std::string& datasetName) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open input CSV: " << filename << std::endl;
        return false;
    }

    std::string line;
    long long lineCounter = 0;

    inputs.clear();
    targets.clear();
    raw_data.clear();
    original_date_strings.clear();

    if (hasHeader) {
        std::getline(file, line);
        lineCounter++;
    }

    long long rowIndex = 0;
    bool firstDataRow = true;

    while (std::getline(file, line) && (numSamplesToLoad == -1 || rowIndex < numSamplesToLoad)) {
        lineCounter++;
        std::stringstream ss(line);
        std::string cell_str;
        std::vector<std::string> current_raw_cells;
        std::vector<float> current_processed_values; // This will hold the converted floats/embeddings

        // 1. Read all cells as raw strings first
        while (std::getline(ss, cell_str, delimitor)) {
            // Trim whitespace from cell
            cell_str.erase(0, cell_str.find_first_not_of(" \t\r\n"));
            cell_str.erase(cell_str.find_last_not_of(" \t\r\n") + 1);
            current_raw_cells.push_back(cell_str);
        }

        if (current_raw_cells.empty()) {
            continue; // Skip empty rows
        }

        // 2. Process raw cells into float values or embeddings
        for (size_t i = 0; i < current_raw_cells.size(); ++i) {
            const std::string& cell = current_raw_cells[i];
            bool value_processed = false;

            // Try to parse as datetime first if it looks like one
            if (cell.find("-") != std::string::npos && (cell.find(":") != std::string::npos || cell.find(";") != std::string::npos) && cell.length() >= 10) {
                try {
                    current_processed_values.push_back(convertDateTimeToTimestamp(cell));
                    value_processed = true;
                    // Store original date string if this is the date column
                    // Assuming date is at a known index, e.g., 1 (second column)
                    if (i == 1) { // Adjust this index if your date column is elsewhere
                        original_date_strings.push_back(cell);
                    }
                }
                catch (const std::runtime_error& e) {
                    // Not a valid datetime, fall through to float or text processing
                    // std::cerr << "Debug: Not a datetime string or conversion error: " << cell << " - " << e.what() << std::endl;
                }
            }

            if (!value_processed) {
                // Try to parse as float
                try {
                    current_processed_values.push_back(std::stof(cell));
                    value_processed = true;
                }
                catch (const std::invalid_argument& e) {
                    // Not a valid float, fall through to text processing
                    // std::cerr << "Debug: Not a float: " << cell << " - " << e.what() << std::endl;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Warning: Number out of range in CSV at line " << lineCounter << ", cell '" << cell << "'. Defaulting to 0.0f. Error: " << e.what() << std::endl;
                    current_processed_values.push_back(0.0f);
                    value_processed = true;
                }
            }

            if (!value_processed && containsText) {
                // If it contains text, encode it. Assuming entire column or specific column is text.
                // For simplicity, let's assume the current cell is text if not numeric/date.
                if (this->langProc) {
                    std::vector<float> encoded_vec = this->langProc->encodeText(cell);
                    current_processed_values.insert(current_processed_values.end(), encoded_vec.begin(), encoded_vec.end());
                    value_processed = true;
                }
                else {
                    std::cerr << "Error: Language processor not initialized for text encoding at line " << lineCounter << ". Skipping text cell '" << cell << "'." << std::endl;
                    // Push a placeholder if language processor is not ready
                    current_processed_values.push_back(0.0f);
                    value_processed = true;
                }
            }

            if (!value_processed) {
                // If still not processed, it's an unhandled type, default to 0.0f
                current_processed_values.push_back(0.0f);
            }
        } // End of for (current_raw_cells)

        if (current_processed_values.empty()) {
            continue; // Skip if row became empty after processing
        }

        // Determine inputSize and outputSize from the first data row
        if (firstDataRow) {
            if (outputSizeParam > 0 && current_processed_values.size() < (size_t)outputSizeParam) {
                std::cerr << "ERROR: First data row has too few columns (" << current_processed_values.size() << ") for specified outputSize ("
                    << outputSizeParam << ") at file line " << lineCounter << ". Cannot proceed." << std::endl;
                return false;
            }
            this->outputSize = outputSizeParam;
            this->inputSize = current_processed_values.size() - this->outputSize;
            // If outputSizeParam was 0, calculate input/output based on heuristic (e.g., last column is target)
            if (this->outputSize == 0 && current_processed_values.size() > 1) { // Heuristic: last column is target
                this->outputSize = 1;
                this->inputSize = current_processed_values.size() - 1;
            }
            else if (this->outputSize == 0 && current_processed_values.size() == 1) { // Only one column, assume inputSize 1, outputSize 0 or 1
                this->outputSize = 0; // No target if only one column and not specified
                this->inputSize = 1;
            }
            firstDataRow = false;
            std::cout << "Determined Input Size: " << this->inputSize << ", Output Size: " << this->outputSize << std::endl;
        }
        else {
            // Validate subsequent rows against determined sizes
            if (current_processed_values.size() != (size_t)(this->inputSize + this->outputSize)) {
                std::cerr << "WARNING: Row " << rowIndex << " (file line " << lineCounter << ") has " << current_processed_values.size()
                    << " columns, but expected " << (this->inputSize + this->outputSize)
                    << " columns based on the first data row. Skipping this row." << std::endl;
                continue;
            }
        }

        // Extract inputs and targets
        std::vector<float> current_input;
        std::vector<float> current_target;

        for (size_t i = 0; i < (size_t)this->inputSize; ++i) {
            current_input.push_back(current_processed_values[i]);
        }
        for (size_t i = (size_t)this->inputSize; i < current_processed_values.size(); ++i) {
            current_target.push_back(current_processed_values[i]);
        }

        inputs.push_back(current_input);
        targets.push_back(current_target);

        rowIndex++;
    }
   

    if (inputs.empty()) {
        std::cerr << "Error: No valid data rows found in CSV or all rows were skipped due to errors." << std::endl;
        return false;
    }

    // Strict Validation
    for (const auto& row : inputs) {
        if (row.size() != (size_t)this->inputSize) {
            std::cerr << "Error: Inconsistent input row size detected after parsing. Expected "
                << this->inputSize << " columns, but found a row with " << row.size() << " columns.\n";
            inputs.clear(); targets.clear(); original_date_strings.clear(); return false;
        }
    }
    for (const auto& row : targets) {
        if (row.size() != (size_t)this->outputSize) {
            std::cerr << "Error: Inconsistent target row size detected after parsing. Expected "
                << this->outputSize << " columns, but found a row with " << row.size() << " columns.\n";
            inputs.clear(); targets.clear(); original_date_strings.clear(); return false;
        }
    }

    this->numSamples = inputs.size();
    std::cout << "Successfully loaded CSV. Samples: " << this->numSamples << std::endl;

    // Database Integration
    if (!isOfflineMode && dbManager) {
        try {
            std::string actualDatasetName = datasetName.empty() ? filename : datasetName;
            currentDatasetId = dbManager->addDataset(actualDatasetName, std::string("Data loaded from ") + filename,
                this->numSamples, this->inputSize, this->outputSize);
            currentDatasetName = actualDatasetName;
            std::cout << "Dataset metadata added/updated with ID: " << currentDatasetId << std::endl;
            long long datasetId = static_cast<long long>(currentDatasetId);
            dbManager->clearDatasetRecords(datasetId);
            std::cout << "Cleared previous records for dataset ID: " << (int)currentDatasetId << std::endl;

            for (size_t i = 0; i < inputs.size(); ++i) {
                dbManager->addDatasetRecord(currentDatasetId, i, inputs[i], targets[i]);
            }
            std::cout << "All dataset records added to database for dataset ID: " << currentDatasetId << std::endl;
        }
        catch (const mysqlx::Error& err) {
            std::cerr << "Database error during CSV loading: " << err.what() << std::endl; return false;
        }
        catch (const std::runtime_error& err) {
            std::cerr << "Runtime error during CSV loading (database related): " << err.what() << std::endl; return false;
        }
    }
    else if (!dbManager && !isOfflineMode) {
        std::cerr << "Warning: Database manager not initialized in online mode. CSV data will not be persisted." << std::endl;
    }

    std::cout << "Successfully loaded CSV: " << filename << std::endl;
    return true;
}

bool Training::loadTargetsCSV(const std::string& filename, const char& delim,
    bool hasHeader, bool containsText, int& datasetId) {
    long long lines = countLines(filename, hasHeader);
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open target file: " << filename << std::endl;
        return false;
    }

    std::vector<std::vector<float>> temp_targets_data;
    std::string line;

    if (hasHeader) {
        std::getline(file, line);
    }

    std::cout << "\nLoading Target CSV: '" << filename << "'...\n";
    const int barWidth = 70;
    long long row_idx = 0;
    bool firstTargetDataRow = true;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(ss, cell, delim)) {
            cell.erase(0, cell.find_first_not_of(" \t\n\r\f\v"));
            cell.erase(cell.find_last_not_of(" \t\n\r\f\v") + 1);

            bool value_processed = false;

            try {
                // Try as datetime
                float timestamp = convertDateTimeToTimestamp(cell);
                row.push_back(timestamp);
                value_processed = true;
            }
            catch (const std::runtime_error&) {} // Catch only if not a datetime

            if (!value_processed) {
                try {
                    // Try as float
                    row.push_back(std::stof(cell));
                    value_processed = true;
                }
                catch (const std::invalid_argument& e) {
                    // Not a float, fall through to text or default
                    // std::cerr << "Debug: Not a float in target CSV: " << cell << " - " << e.what() << std::endl;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Warning: Number out of range in target CSV: '" << cell << "'. Defaulting to 0.0f. Error: " << e.what() << std::endl;
                    row.push_back(0.0f);
                    value_processed = true;
                }
            }

            if (!value_processed && containsText) {
                // If still not processed and contains text, encode it
                if (this->langProc) {
                    std::vector<float> encoded_vec = this->langProc->encodeText(cell);
                    row.insert(row.end(), encoded_vec.begin(), encoded_vec.end());
                    value_processed = true;
                }
                else {
                    std::cerr << "Error: Language processor not initialized for target text encoding. Skipping text cell '" << cell << "'." << std::endl;
                    row.push_back(0.0f);
                    value_processed = true;
                }
            }

            if (!value_processed) {
                row.push_back(0.0f); // Default if parsing fails
            }
        }

        if (!row.empty()) {
            if (firstTargetDataRow) {
                if (this->outputSize == 0) { // If outputSize not set by loadCSV
                    this->outputSize = row.size();
                    std::cout << "Determined Output Size from target CSV: " << this->outputSize << std::endl;
                }
                firstTargetDataRow = false;
            }

            if (this->outputSize > 0 && row.size() != (size_t)this->outputSize) {
                std::cerr << "Error: Target file row " << row_idx << " has " << row.size()
                    << " columns, but expected " << this->outputSize << ". This is a critical mismatch. Aborting target loading.\n";
                temp_targets_data.clear();
                return false;
            }
            temp_targets_data.push_back(row);
            row_idx++;
            if (lines > 0) {
                printProgressBar("Loading Target CSV", row_idx, lines, barWidth);
            }
        }
    }
    file.close();

    if (temp_targets_data.empty()) {
        std::cerr << "Error: No valid target data rows found in '" << filename << "'." << std::endl;
        return false;
    }

    for (const auto& row : temp_targets_data) {
        if (row.size() != (size_t)this->outputSize) {
            std::cerr << "Error: Inconsistent target row size detected after parsing. Expected "
                << this->outputSize << " columns, but found a row with " << row.size() << " columns.\n";
            temp_targets_data.clear();
            return false;
        }
    }

    this->targets = temp_targets_data;
    if (this->outputSize == 0 && !this->targets.empty()) {
        this->outputSize = this->targets[0].size();
    }

    if (lines > 0) {
        printProgressBar("Loading Target CSV", row_idx, lines, barWidth);
    }
    std::cout << "\nTarget CSV Loaded!\n";

    if (!isOfflineMode && dbManager && datasetId != -1) {
        try {
            for (size_t i = 0; i < targets.size(); ++i) {
                dbManager->updateDatasetRecordLabels(datasetId, i, targets[i]);
            }
            std::cout << "All target records updated in database for dataset ID: " << datasetId << std::endl;
        }
        catch (const mysqlx::Error& err) {
            std::cerr << "Database error during target CSV loading/saving: " << err.what() << std::endl; return false;
        }
        catch (const std::runtime_error& err) {
            std::cerr << "Runtime error during target CSV loading/saving (database related): " << err.what() << std::endl; return false;
        }
    }
    else if (!dbManager && !isOfflineMode && datasetId != -1) {
        std::cerr << "Warning: Database manager not initialized in online mode. Target CSV data will not be persisted." << std::endl;
    }

    return true;
}

bool Training::loadDatasetFromDB(int& datasetId) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot load dataset from database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    std::cout << "Loading dataset from database (ID: " << datasetId << ")...\n";
    try {
        // Corrected call to getDataset: it now returns DatasetData struct
        Database::DatasetData loadedData = dbManager->getDataset(datasetId);

        if (loadedData.inputs.empty()) {
            std::cerr << "No data found for dataset ID: " << datasetId << std::endl;
            return false;
        }

        this->inputs = loadedData.inputs;
        this->targets = loadedData.targets;
        this->numSamples = loadedData.numRows;
        this->inputSize = loadedData.numFeatures;
        this->outputSize = loadedData.numLabels;
        this->currentDatasetId = datasetId;
        this->currentDatasetName = loadedData.datasetName;

        // Strict Validation
        if (!inputs.empty() && inputs[0].size() != (size_t)this->inputSize) {
            std::cerr << "Error: Inconsistent input row size loaded from DB. Expected " << this->inputSize << " columns, but first row has " << inputs[0].size() << " columns.\n";
            this->inputs.clear(); this->targets.clear(); return false;
        }
        for (const auto& row : inputs) {
            if (row.size() != (size_t)this->inputSize) {
                std::cerr << "Error: Inconsistent input row size loaded from DB. Expected " << this->inputSize << " columns, but found a row with " << row.size() << " columns.\n";
                this->inputs.clear(); this->targets.clear(); return false;
            }
        }
        if (!targets.empty() && targets[0].size() != (size_t)this->outputSize) {
            std::cerr << "Error: Inconsistent target row size loaded from DB. Expected " << this->outputSize << " columns, but first row has " << targets[0].size() << " columns.\n";
            this->inputs.clear(); this->targets.clear(); return false;
        }
        for (const auto& row : targets) {
            if (row.size() != (size_t)this->outputSize) {
                std::cerr << "Error: Inconsistent target row size loaded from DB. Expected " << this->outputSize << " columns, but found a row with " << row.size() << " columns.\n";
                this->inputs.clear(); this->targets.clear(); return false;
            }
        }

        std::cout << "Successfully loaded dataset '" << this->currentDatasetName
            << "' (ID: " << this->currentDatasetId << ") from database.\n";
        return true;
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Database error loading dataset: " << err.what() << std::endl; return false;
    }
    catch (const std::runtime_error& err) {
        std::cerr << "Runtime error loading dataset from database: " << err.what() << std::endl; return false;
    }
}

void Training::splitInputOutput(int outputSize) {
    inputs.clear();
    targets.clear();

    for (const auto& row : raw_data) {
        if (row.size() < (size_t)outputSize) { // Cast outputSize to size_t
            std::cerr << "Warning: Row too short for outputSize. Skipping row." << std::endl;
            continue;
        }

        std::vector<float> currentInput;
        std::vector<float> currentTarget;

        for (size_t i = 0; i < row.size() - outputSize; ++i) { // Use size_t
            currentInput.push_back(row[i]);
        }
        inputs.push_back(currentInput);

        for (size_t i = row.size() - outputSize; i < row.size(); ++i) { // Use size_t
            currentTarget.push_back(row[i]);
        }
        targets.push_back(currentTarget);
    }
}

void Training::normalize(float minRange, float maxRange) {
    if (inputs.empty()) {
        std::cerr << "Warning: No input data to normalize." << std::endl;
        return;
    }

    float tempGlobalMin = std::numeric_limits<float>::max();
    float tempGlobalMax = std::numeric_limits<float>::lowest();

    bool data_found_for_scaling = false;

    for (const auto& row : inputs) {
        for (float val : row) {
            if (!std::isnan(val) && !std::isinf(val)) {
                tempGlobalMin = std::min(tempGlobalMin, val);
                tempGlobalMax = std::max(tempGlobalMax, val);
                data_found_for_scaling = true;
            }
        }
    }

    for (const auto& row : targets) {
        for (float val : row) {
            if (!std::isnan(val) && !std::isinf(val)) {
                tempGlobalMin = std::min(tempGlobalMin, val);
                tempGlobalMax = std::max(tempGlobalMax, val);
                data_found_for_scaling = true;
            }
        }
    }

    this->original_data_global_min = tempGlobalMin;
    this->original_data_global_max = tempGlobalMax;

    if (!data_found_for_scaling || std::abs(tempGlobalMax - tempGlobalMin) < std::numeric_limits<float>::epsilon()) {
        std::cerr << "Warning: Data is constant or contains only NaN/Inf. Mapping all values to target range based on NaN/Inf rules." << std::endl;
        for (auto& row : inputs) {
            for (float& val : row) {
                if (std::isinf(val)) {
                    val = maxRange;
                }
                else if (std::isnan(val)) {
                    val = minRange;
                }
                else {
                    val = minRange;
                }
            }
        }
        for (auto& row : targets) {
            for (float& val : row) {
                if (std::isinf(val)) {
                    val = maxRange;
                }
                else if (std::isnan(val)) {
                    val = minRange;
                }
                else {
                    val = minRange;
                }
            }
        }
        return;
    }

    const int barWidth = 70;
    long long current_sample = 0;
    long long total_samples = inputs.size();

    std::cout << "\nNormalizing Data...\n";
    float dataRangeDiff = tempGlobalMax - tempGlobalMin;
    float targetRangeDiff = maxRange - minRange;

    for (size_t i = 0; i < inputs.size(); ++i) { // Use size_t
        for (size_t j = 0; j < inputs[i].size(); ++j) { // Use size_t
            float val = inputs[i][j];
            if (std::isinf(val)) {
                inputs[i][j] = maxRange;
            }
            else if (std::isnan(val)) {
                inputs[i][j] = minRange;
            }
            else {
                inputs[i][j] = minRange + ((val - tempGlobalMin) * targetRangeDiff) / dataRangeDiff;
            }
        }

        if (i < targets.size()) {
            for (size_t j = 0; j < targets[i].size(); ++j) { // Use size_t
                float val = targets[i][j];
                if (std::isinf(val)) {
                    targets[i][j] = maxRange;
                }
                else if (std::isnan(val)) {
                    targets[i][j] = minRange;
                }
                else {
                    targets[i][j] = minRange + ((val - tempGlobalMin) * targetRangeDiff) / dataRangeDiff;
                }
            }
        }
        current_sample++;
        printProgressBar("Normalizing Data", current_sample, total_samples, barWidth);
    }
    if (total_samples > 0) {
        printProgressBar("Normalizing Data", total_samples, total_samples, barWidth);
    }
    std::cout << "\nData Normalized!\n";
}

int Training::getMinInputColumns() const {
    int min_columns = std::numeric_limits<int>::max();
    if (inputs.empty()) {
        return 0;
    }

    for (const auto& row : inputs) {
        if (row.size() < (size_t)min_columns) { // Cast min_columns to size_t for comparison
            min_columns = row.size();
        }
    }
    return min_columns;
}

void Training::preprocess(float minVal, float maxVal) {
    if (inputs.empty() || targets.empty()) {
        std::cerr << "Error: No data to preprocess. Load data from file or database first." << std::endl;
        return;
    }

    normalize(minVal, maxVal);

    if (!core) { // Initialize CoreAI only if it hasn't been already
        core = std::make_unique<CoreAI>(this->inputSize, layers, neurons, this->outputSize, minVal, maxVal);
    }
    else {
        // If core is already initialized, update its parameters if necessary
        //core->updateParameters(this->inputSize, layers, neurons, this->outputSize, minVal, maxVal); // Example update method
    }

    core->setInput(inputs);
    core->setTarget(targets);
}

void Training::train(double learningRate, int epochs) {
    if (!core) {
        std::cerr << "Error: CoreAI not initialized. Cannot train model." << std::endl;
        return;
    }
    std::cout << "\nStarting training for " << epochs << " epochs...\n";
    const int barWidth = 70;
    for (int i = 0; i < epochs; ++i) {

        printProgressBar("Training Progress", i + 1, epochs, barWidth);
        core->forward(inputs);
        core->train(inputs, targets, learningRate, numSamples);
    }
    if (epochs > 0) {
        printProgressBar("Training Progress", epochs, epochs, barWidth);
    }
    std::cout << "\nTraining Complete!\n";
}

float Training::calculateRMSE() {
    if (!core || inputs.empty() || targets.empty()) {
        std::cerr << "Error: CoreAI not initialized or data not loaded for RMSE calculation." << std::endl;
        return -1.0f;
    }

    core->forward(inputs);
    // Denormalize the predictions using CoreAI's internal denormalization method.
    // CoreAI's denormalizeOutput should scale its results from its internal range ([0,1] or its minVal/maxVal)
    // back to its original data scale (which CoreAI itself doesn't explicitly store, so it relies on parameters).
    // This part relies on CoreAI's `denormalizeOutput` correctly using `minVal`/`maxVal` passed to its constructor.
    core->denormalizeOutput(); // This function should be fixed in Core.cpp to use core's minVal/maxVal

    auto predictions = core->getResults();

    // Denormalize targets using the original data's global min/max stored in Training class
    std::vector<std::vector<float>> denormalizedTargets = targets; // Create a copy to denormalize
    float target_range_min_val = this->min; // The 'min' value from `--min` argument
    float target_range_max_val = this->max; // The 'max' value from `--max` argument
    float target_range_diff = target_range_max_val - target_range_min_val;
    float original_data_range_diff = original_data_global_max - original_data_global_min;

    // Denormalize target values from the [min, max] range back to the original global min/max
    for (auto& row : denormalizedTargets) {
        for (float& val : row) {
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon() && std::abs(original_data_range_diff) > std::numeric_limits<float>::epsilon()) {
                val = ((val - target_range_min_val) / target_range_diff) * original_data_range_diff + original_data_global_min;
            }
            else {
                val = original_data_global_min; // If ranges are zero, map to original min
            }
        }
    }

    float mse = 0.0f;
    int count = 0;

    for (size_t i = 0; i < denormalizedTargets.size(); ++i) { // Use size_t
        if (i < predictions.size() && denormalizedTargets[i].size() == predictions[i].size()) {
            for (size_t j = 0; j < denormalizedTargets[i].size(); ++j) { // Use size_t
                float error = denormalizedTargets[i][j] - predictions[i][j];
                mse += error * error;
                count++;
            }
        }
        else {
            std::cerr << "Warning: Mismatched row size between targets and predictions at row " << i << ". Skipping for RMSE." << std::endl;
        }
    }

    if (count == 0) {
        std::cerr << "Error: No valid data points for RMSE calculation." << std::endl;
        return -1.0f;
    }

    return std::sqrt(mse / count);
}

CoreAI* Training::getCore() {
    return core.get();
}

Language* Training::getLanguage()
{
    return this->langProc.get();
}

bool Training::saveModel(int& datasetId) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save model to database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }
    if (!core) {
        std::cerr << "Error: CoreAI not initialized. Cannot save model." << std::endl;
        return false;
    }

    std::cout << "Saving AI model state to database (ID: " << datasetId << ")...\n";

    auto inputData = core->getInput();
    auto outputData = core->getOutput();
    auto hiddenData = core->getHiddenData();
    auto hiddenOutputData = core->getHiddenOutputData();
    auto hiddenErrorData = core->getHiddenErrorData();
    auto weightsHiddenInput = core->getWeightsHiddenInput();
    auto weightsOutputHidden = core->getWeightsOutputHidden();

    // Resize vectors and matrices before saving to ensure consistency with what CoreAI expects
    // Note: These resizes might be redundant or problematic if CoreAI's internal sizes are not consistent.
    // Ideally, CoreAI's getters return correctly sized data.
    // For now, retaining the resize calls with a warning about their necessity.
    if (!hiddenData.empty() && hiddenData[0].size() != (size_t)this->neurons) {
        std::cerr << "Warning: hiddenData size mismatch before saving model." << std::endl;
        // Consider resizing or re-populating hiddenData in CoreAI
    }
    if (hiddenOutputData.size() != (size_t)this->neurons) { // Assuming hidden_output is `neurons` size
        std::cerr << "Warning: hiddenOutputData size mismatch before saving model." << std::endl;
    }
    if (hiddenErrorData.size() != (size_t)this->outputSize) { // Assuming hidden_error is `outputSize` size
        std::cerr << "Warning: hiddenErrorData size mismatch before saving model." << std::endl;
    }
    if (!weightsHiddenInput.empty() && weightsHiddenInput[0].size() != (size_t)this->inputSize) { // Assuming input layer size
        std::cerr << "Warning: weightsHiddenInput size mismatch before saving model." << std::endl;
    }
    if (!weightsOutputHidden.empty() && weightsOutputHidden[0].size() != (size_t)this->neurons) { // Assuming hidden layer size
        std::cerr << "Warning: weightsOutputHidden size mismatch before saving model." << std::endl;
    }


    dbManager->saveAIModelState(datasetId, inputData, outputData,
        hiddenData, hiddenOutputData, hiddenErrorData,
        weightsHiddenInput, weightsOutputHidden);

    std::cout << "AI model state saved successfully." << std::endl;
    return true;
}

bool Training::loadModel(int& datasetId) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot load model from database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    std::cout << "Loading AI model state from database (ID: " << datasetId << ")...\n";

    Database::AIModelState state = dbManager->loadLatestAIModelState(datasetId);

    if (state.inputData.empty() || state.outputData.empty() || state.weightsHiddenInput.empty() || state.weightsOutputHidden.empty()) {
        std::cerr << "No valid model state found or loaded for dataset ID " << datasetId << std::endl;
        return false;
    }

    this->inputSize = state.inputData[0].size();
    this->outputSize = state.outputData[0].size();
    // Reinitialize CoreAI with loaded dimensions if not already
    if (!core) {
        core = std::make_unique<CoreAI>(this->inputSize, layers, neurons, this->outputSize, min, max);
    }
    // Set the loaded state to CoreAI
    core->setInput(state.inputData);
    core->setOutput(state.outputData); // This is likely the stored targets
    core->setHiddenData(state.hiddenData);
    core->setHiddenOutputData(state.hiddenOutputData);
    core->setHiddenErrorData(state.hiddenErrorData);
    core->setWeightsHiddenInput(state.weightsHiddenInput);
    core->setWeightsOutputHidden(state.weightsOutputHidden);

    std::cout << "Model loaded successfully for dataset ID " << datasetId << std::endl;
	return true;
}

void Training::printDenormalizedAsOriginalMatrix(std::vector<std::vector<float>>& normalized_data, int len, int precision)
{
    if (normalized_data.empty()) {
        std::cout << "  (Empty matrix to denormalize and print)\n";
        return;
    }

    std::vector<std::vector<float>> denormalized_data = denormalizeMatrix(normalized_data);
    // Note: resizing here might truncate data if len is smaller than actual row size.
    // It's better to ensure 'len' is relevant or remove it if not needed for printing.
    for (auto& row : denormalized_data) {
        row.resize((size_t)len);
    }
    std::cout << "  Denormalized Data (original scale):\n";
    printFullMatrix(denormalized_data, len, precision);
}

void Training::printFullMatrix(std::vector<std::vector<float>>& data, int len, int precision) {
    if (data.empty()) {
        std::cout << "  (Empty matrix)\n";
        return;
    }

    int maxCols = 0;
    for (auto& row : data) {
        if (row.size() > (size_t)maxCols) { // Cast maxCols for comparison
            maxCols = row.size();
        }
        row.resize((size_t)len); // Cast len to size_t
    }

    std::vector<int> columnWidths(maxCols, 0);

    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) { // Use size_t
            if (j < (size_t)maxCols) { // Cast maxCols for comparison
                std::stringstream ss;
                ss << std::fixed << std::setprecision(precision) << row[j];
                columnWidths[j] = std::max(columnWidths[j], static_cast<int>(ss.str().length()));
            }
        }
    }

    for (size_t i = 0; i < data.size(); ++i) { // Use size_t
        std::cout << "  [";
        for (size_t j = 0; j < data[i].size(); ++j) { // Use size_t
            if (j < (size_t)maxCols) { // Cast maxCols for comparison
                std::cout << std::fixed << std::setprecision(precision) << std::setw(columnWidths[j]) << data[i][j];
            }
            else {
                std::cout << std::fixed << std::setprecision(precision) << data[i][j];
            }

            if (j < data[i].size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
}

std::vector<std::vector<float>> Training::denormalizeMatrix(const std::vector<std::vector<float>>& normalized_data) const {
    if (normalized_data.empty()) {
        return {};
    }

    std::vector<std::vector<float>> denormalized_copy;
    float target_range_min = this->min;
    float target_range_max = this->max;
    float target_range_diff = target_range_max - target_range_min;
    float original_data_range_diff = original_data_global_max - original_data_global_min;

    for (const auto& row : normalized_data) {
        std::vector<float> denormalized_row;
        for (float val : row) {
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon() && std::abs(original_data_range_diff) > std::numeric_limits<float>::epsilon()) {
                denormalized_row.push_back(((val - target_range_min) / target_range_diff) * original_data_range_diff + original_data_global_min);
            }
            else {
                denormalized_row.push_back(original_data_global_min);
            }
        }
        denormalized_copy.push_back(denormalized_row);
    }
    return denormalized_copy;
}

bool Training::saveResultsToCSV(const std::string& filename, const std::string& inputFilename, bool hasHeader, const char& delimiter) {
    if (!core || inputs.empty() || targets.empty()) {
        std::cerr << "Error: No data or model initialized for saving results to CSV." << std::endl;
        return false;
    }

    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open output CSV file for saving results: " << filename << std::endl;
        return false;
    }

    core->forward(inputs);
    core->denormalizeOutput();
    auto predictions = core->getResults();

    std::vector<std::vector<float>> denormalizedTargets = targets;
    float target_range_min_val = this->min;
    float target_range_max_val = this->max;
    float target_range_diff = target_range_max_val - target_range_min_val;
    float original_data_range_diff = original_data_global_max - original_data_global_min;

    // Denormalize inputs if they were normalized and you want them in original scale in output
    std::vector<std::vector<float>> denormalizedInputs = inputs;
    for (auto& row : denormalizedInputs) {
        for (float& val : row) {
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon() && std::abs(original_data_range_diff) > std::numeric_limits<float>::epsilon()) {
                val = ((val - target_range_min_val) / target_range_diff) * original_data_range_diff + original_data_global_min;
            }
            else {
                val = original_data_global_min;
            }
        }
    }

    for (auto& row : denormalizedTargets) {
        for (float& val : row) {
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon() && std::abs(original_data_range_diff) > std::numeric_limits<float>::epsilon()) {
                val = ((val - target_range_min_val) / target_range_diff) * original_data_range_diff + original_data_global_min;
            }
            else {
                val = original_data_global_min;
            }
        }
    }

    // Ensure predictions are correctly denormalized by CoreAI already
    // The previous loop over predictions and its denormalization logic here is redundant if CoreAI::denormalizeOutput handles it.
    // So, assuming `predictions` from `core->getResults()` are already denormalized.


    // Read header from the original input file
    std::string originalInputHeaderLine = "";
    std::vector<std::string> originalInputHeaderCells;
    if (hasHeader) {
        std::ifstream inputFileStream(inputFilename); // Renamed to avoid conflict
        if (inputFileStream.is_open()) {
            std::getline(inputFileStream, originalInputHeaderLine);
            inputFileStream.close();
            std::stringstream ss_header(originalInputHeaderLine);
            std::string cell;
            while (std::getline(ss_header, cell, delimiter)) {
                originalInputHeaderCells.push_back(cell);
            }
        }
        else {
            std::cerr << "Warning: Could not open input file '" << inputFilename << "' to read header. Using default input column names.\n";
        }
    }

    // Write header to output CSV
    // Date | Other Input Features | Actual Targets | Predicted Targets
    outputFile << "Date"; // Always include Date header
    for (size_t i = 0; i < (size_t)inputSize; ++i) {
        if (i < originalInputHeaderCells.size()) {
            outputFile << delimiter << originalInputHeaderCells[i];
        }
        else {
            outputFile << delimiter << "Input_Feature_" << (i + 1);
        }
    }
    for (size_t i = 0; i < (size_t)outputSize; ++i) {
        outputFile << delimiter << "Actual_Target_" << (i + 1);
    }
    for (size_t i = 0; i < (size_t)outputSize; ++i) {
        outputFile << delimiter << "Predicted_Target_" << (i + 1);
    }
    outputFile << "\n";

    // Write data rows
    for (size_t i = 0; i < denormalizedInputs.size(); ++i) {
        // Date handling: Use original_date_strings or generate from timestamp
        if (i < original_date_strings.size() && !original_date_strings[i].empty()) {
            outputFile << original_date_strings[i];
        }
        else {
            // If no original date string, try converting timestamp to date string
            // This assumes `inputs[i][0]` or `inputs[i][1]` holds a timestamp
            // You need to adjust based on where your timestamp is.
            // For now, write empty or a placeholder if no date string
            outputFile << "";
        }
        outputFile << delimiter;

        // Write denormalized input features
        for (size_t j = 0; j < denormalizedInputs[i].size(); ++j) {
            outputFile << std::fixed << std::setprecision(4) << formatValueForDisplay(denormalizedInputs[i][j], 2);
            outputFile << delimiter;
        }

        // Write Actual Targets (denormalized)
        if (i < denormalizedTargets.size()) {
            for (size_t j = 0; j < denormalizedTargets[i].size(); ++j) {
                outputFile << std::fixed << std::setprecision(4) << formatValueForDisplay(denormalizedTargets[i][j], 2);
                outputFile << delimiter;
            }
        }
        else {
            // Handle case where targets might be missing for some inputs (shouldn't happen in training)
            for (size_t j = 0; j < (size_t)outputSize; ++j) {
                outputFile << "" << delimiter; // Empty cells for missing targets
            }
        }


        // Write Predicted Targets (denormalized)
        if (i < predictions.size()) {
            for (size_t j = 0; j < predictions[i].size(); ++j) {
                outputFile << std::fixed << std::setprecision(4) << formatValueForDisplay(predictions[i][j], 2);
                if (j < predictions[i].size() - 1) {
                    outputFile << delimiter;
                }
            }
        }
        else {
            // Handle case where predictions might be missing
            for (size_t j = 0; j < (size_t)outputSize; ++j) {
                outputFile << ""; // Empty cell for missing prediction
                if (j < (size_t)outputSize - 1) {
                    outputFile << delimiter;
                }
            }
        }
        outputFile << "\n";
    }

    outputFile.close();
    std::cout << "Results saved to: " << filename << std::endl;
    return true;
}

std::string Training::timestampToDateTimeString(float timestamp) {
    std::time_t t = static_cast<std::time_t>(timestamp);
    std::tm tm_buf; // Use a buffer for thread-safe localtime
    #ifdef _WIN32
        localtime_s(&tm_buf, &t);
    #else
        tm_buf = *std::localtime(&t);
    #endif // Use _s version for Windows, or std::localtime for POSIX
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string Training::formatValueForDisplay(float value, int customPrecision) const {
    std::stringstream ss;
    if (std::abs(value) >= 1.0e18f) {
        ss << std::fixed << std::setprecision(2) << (value / 1.0e20f) << "e+20"; // Append exponent if needed
    }
    else if (std::abs(value) >= 1000000000.0f) {
        ss << std::fixed << std::setprecision(customPrecision) << (value / 1000000000.0f) << "B"; // Billions
    }
    else if (std::abs(value) >= 1000.0f) {
        ss << std::fixed << std::setprecision(2) << value; // Keep high precision for values > 1000
    }
    else if (std::abs(value) >= 0.01f && std::abs(value) < 1.0f) {
        ss << std::fixed << std::setprecision(2) << value; // Show two decimal places for small values
    }
    else {
        ss << std::fixed << std::setprecision(customPrecision + 4) << value;
    }
    return ss.str();
}

int Training::getDecimalPlacesInString(float value, int precision_for_conversion) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision_for_conversion) << value;
    std::string s = ss.str();

    size_t decimal_pos = s.find('.');

    if (decimal_pos == std::string::npos) {
        return 0;
    }

    int count = 0;
    for (size_t i = decimal_pos + 1; i < s.length(); ++i) { // Use size_t
        if (std::isdigit(s[i])) {
            count++;
        }
        else {
            break;
        }
    }
    while (count > 0 && s[decimal_pos + count] == '0') {
        count--;
    }
    return count;
}

std::vector<std::vector<float>> Training::normalizeData(const std::vector<std::vector<float>>& data_to_normalize,
    float original_min, float original_max,
    float target_min, float target_max) const {
    if (data_to_normalize.empty()) {
        return {};
    }

    std::vector<std::vector<float>> normalized_copy;
    float original_range_diff = original_max - original_min;
    float target_range_diff = target_max - target_min;

    for (const auto& row : data_to_normalize) {
        std::vector<float> normalized_row;
        for (float val : row) {
            if (std::abs(original_range_diff) > std::numeric_limits<float>::epsilon()) {
                normalized_row.push_back(((val - original_min) / original_range_diff) * target_range_diff + target_min);
            }
            else {
                normalized_row.push_back(target_min + 0.1f); // Handle zero range
            }
        }
        normalized_copy.push_back(normalized_row);
    }
    return normalized_copy;
}

bool Training::loadTextCSV(const std::string& filename, int maxSeqLen, int embeddingDim) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1; // Default output size for text, adjust if your CSV includes labels

    if (!this->langProc) {
        std::cerr << "Error: Language processor not initialized for text CSV loading. Cannot load text data." << std::endl;
        return false;
    }

    // Assuming this->langProc->generateRandomEmbedding() uses its own rng
    // and that this->langProc->loadWordEmbeddingsFromFile is intended to load the main embedding map.

    while (std::getline(file, line)) {
        std::vector<std::string> words = this->langProc->tokenize(line); // Use tokenizer from Language class
        std::vector<std::vector<float>> sentenceEmbedding;

        for (const std::string& word : words) {
            std::string key = this->langProc->detectLanguage(word);
            // Access embeddings from the this->langProc's map
            auto it = this->langProc->embeddingsByLang.find(key);

            if (it != this->langProc->embeddingsByLang.end()) {
                sentenceEmbedding.push_back(it->second);
            }
            else {
                sentenceEmbedding.push_back(this->langProc->generateRandomEmbedding()); // Generate random if not found
            }
        }

        // Pad or truncate to maxSeqLen
        if ((int)sentenceEmbedding.size() < maxSeqLen) {
            while ((int)sentenceEmbedding.size() < maxSeqLen) {
                sentenceEmbedding.push_back(std::vector<float>(embeddingDim, 0.0f));
            }
        }
        else if ((int)sentenceEmbedding.size() > maxSeqLen) {
            sentenceEmbedding.resize(maxSeqLen);
        }

        // Flatten into a single vector
        std::vector<float> flatInput;
        for (const auto& vec : sentenceEmbedding) {
            flatInput.insert(flatInput.end(), vec.begin(), vec.end());
        }

        inputs.push_back(flatInput);

        targets.push_back({ 0.0f }); // Dummy target, adapt if CSV has labels
    }

    file.close();
    numSamples = inputs.size();

    std::cout << "Loaded " << numSamples << " text samples from " << filename << std::endl;
    return !inputs.empty();
}

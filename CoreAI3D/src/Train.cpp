#include "Train.hpp"
// Constructor for online mode (with database)
Training::Training(const std::string& dbHost, unsigned int dbPort, const std::string& dbUser,
    const std::string dbPassword, const std::string& dbSchema, mysqlx::SSLMode ssl, bool createTables)
    : dbManager(std::make_unique<Database>(dbHost, dbPort, dbUser, dbPassword, dbSchema, ssl)),
    isOfflineMode(false), currentDatasetId(-1), numSamples(0), inputSize(0), outputSize(0),
    original_data_global_min(0.0f), original_data_global_max(0.0f) // Initialize new members
{
    if (dbManager && createTables) {
        dbManager->createTables(); // Ensure tables exist on startup
    }
}

// Constructor for offline mode (no database)
Training::Training(bool isOffline)
    : dbManager(nullptr), // No database manager in offline mode
    isOfflineMode(isOffline), currentDatasetId(-1), numSamples(0), inputSize(0), outputSize(0),
    original_data_global_min(0.0f), original_data_global_max(0.0f) // Initialize new members
{
    if (isOfflineMode) {
        std::cout << "Training initialized in OFFLINE mode. Database operations are disabled.\n";
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


float Training::convertDateTimeToTimestamp(const std::string& datetime) {
    // Remove potential "0;UTC" at the end
    std::string cleaned = datetime.substr(0, datetime.find_last_of(';'));
    std::replace(cleaned.begin(), cleaned.end(), ';', ' '); // Replace ';' with space for parsing

    std::tm tm = {};
    std::istringstream ss(cleaned);
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (ss.fail()) {
        throw std::runtime_error("Invalid datetime format: " + datetime);
    }

    return static_cast<float>(std::mktime(&tm));
}

void Training::printProgressBar(const std::string& prefix, long long current, long long total, int barWidth) {
    if (total == 0) return; // Avoid division by zero

    float progress_percent = (float)current / total;
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

int Training::detectMaxSeqLength(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for max sequence length detection: " << filename << std::endl;
        return -1;
    }

    std::string line;
    int maxLen = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int count = 0;
        std::string word;

        while (iss >> word) {
            ++count;
        }

        if (count > maxLen) {
            maxLen = count;
        }
    }

    file.close();
    std::cout << "Detected maximum sequence length: " << maxLen << std::endl;
    return maxLen;
}



bool Training::loadCSV(const std::string & filename, long long numSamplesToLoad, int outputSizeParam, bool hasHeader,
    bool containsText, const char& delimitor, const std::string & datasetName) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open input CSV: " << filename << std::endl;
        return false;
    }
    

    std::string line;
    long long lineCounter = 0; // Tracks actual line number in file

    inputs.clear();
    targets.clear();
   
    raw_data.clear(); // Clear raw data too

    original_date_strings.clear(); // Clear previous date strings
    if (hasHeader) {
        std::getline(file, line); // Read and discard header
        lineCounter++;
    }

    long long rowIndex = 0; // Logical row index for parsed data
    bool firstDataRow = true; // Flag to determine input/output sizes from the first data row

    while (std::getline(file, line)) {
        lineCounter++;

        std::vector<float> row_values;
        std::vector<std::string> raw_cell_strings; // To store raw strings for later processing (e.g., date)
        std::stringstream ss(line);
        std::string cell;
        
        int cell_count = 0;
        
        cell_count++;
        // Trim whitespace from cell
        cell.erase(0, cell.find_first_not_of(" \t\r\n"));
        cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
        raw_cell_strings.push_back(cell); // Store raw string

        if (containsText) {
            inputSize = this->detectMaxSeqLength(filename) * EMBEDDING_DIM;
            // If it's a date-like string, try converting to timestamp first
            if (cell.find("-") != std::string::npos && cell.find(":") != std::string::npos && cell.length() >= 10) {
                try {
                    if (!original_date_strings.empty()) {
                        try {
                            // Get the last date string and convert it to a timestamp
                            last_known_timestamp = convertDateTimeToTimestamp(original_date_strings.back());
                        }
                        catch (const std::runtime_error& e) {
                            std::cerr << "Error converting last date string to timestamp: " << e.what() << std::endl;
                            // Handle error, perhaps set to 0 or a default
                            last_known_timestamp = 0.0f;
                        }
                    }
                    else {
                        last_known_timestamp = 0.0f; // No date strings loaded
                    }
                    row_values.push_back(convertDateTimeToTimestamp(cell));
                }
                catch (const std::runtime_error& e) {
                    // Fallback to encodeText if not a valid datetime
                    row_values = lang->encodeText(cell);
                }
            }
            else {
                row_values = lang->encodeText(cell);
            }
        }
        else {
            try {
                row_values.push_back(std::stof(cell));
            }
            catch (const std::invalid_argument& e) {
                row_values.push_back(0.0f); // Or handle error appropriately
            }
            catch (const std::out_of_range& e) {
                row_values.push_back(0.0f); // Or handle error appropriately
            }
        }

        if (row_values.empty()) {
            continue; // Skip empty rows
        }

        if (firstDataRow) {
            // For the first data row, determine inputSize and outputSize
            if (row_values.size() < (int)outputSizeParam) {
                std::cerr << "ERROR: First data row has too few columns (" << row_values.size() << ") for specified outputSize ("
                    << outputSizeParam << ") at file line " << lineCounter << ". Cannot proceed." << std::endl;
                return false; // Critical error, cannot determine sizes
            }
            this->outputSize = outputSizeParam;
            this->inputSize = row_values.size() - this->outputSize; // Calculated input size
            firstDataRow = false;
            std::cout << "Determined Input Size: " << this->inputSize << ", Output Size: " << this->outputSize << std::endl;
        }
        else {
            // For subsequent rows, validate against the determined sizes
            if (row_values.size() != (int)(this->inputSize + this->outputSize)) {
                std::cerr << "WARNING: Row " << rowIndex << " (file line " << lineCounter << ") has " << row_values.size()
                    << " columns, but expected " << (this->inputSize + this->outputSize)
                    << " columns based on the first data row. Skipping this row." << std::endl;
                continue; // Skip inconsistent rows
            }
        }

        // Extract inputs and targets based on determined inputSize and outputSize
        std::vector<float> current_input;
        std::vector<float> current_target;

        // Populate current_input and current_target
        for (int i = 0; i < (int)this->inputSize; ++i) {
            current_input.push_back(row_values[i]);
        }
        for (int i = (int)this->inputSize; i < row_values.size(); ++i) {
            current_target.push_back(row_values[i]);
        }

        inputs.push_back(current_input); // Directly add to member 'inputs'
        targets.push_back(current_target); // Directly add to member 'targets'

        // Assuming date is the second column (index 1) of the raw CSV line
        // and it corresponds to the second value in row_values / current_input
        if (raw_cell_strings.size() > 1) { // Check if there's at least a second column
            original_date_strings.push_back(raw_cell_strings[1]); // Store the raw date string
        }
        else {
            original_date_strings.push_back(""); // Store empty if date column is missing
        }


        rowIndex++; // Increment logical row index for each successfully parsed row
    }

    if (inputs.empty()) { // Check member 'inputs'
        std::cerr << "Error: No valid data rows found in CSV or all rows were skipped due to errors." << std::endl;
        return false;
    }

    // --- Strict Validation of loaded data ---
    // These checks will now run on inputs/targets directly
    for (const auto& row : inputs) {
        if (row.size() != (int)this->inputSize) {
            std::cerr << "Error: Inconsistent input row size detected after parsing. Expected "
                << this->inputSize << " columns, but found a row with " << row.size() << " columns.\n";
            inputs.clear(); // Clear potentially inconsistent data
            targets.clear();
            original_date_strings.clear();
            return false;
        }
    }
    for (const auto& row : targets) {
        if (row.size() != (int)this->outputSize) {
            std::cerr << "Error: Inconsistent target row size detected after parsing. Expected "
                << this->outputSize << " columns, but found a row with " << row.size() << " columns.\n";
            inputs.clear(); // Clear potentially inconsistent data
            targets.clear();
            original_date_strings.clear();
            return false;
        }
    }

    this->numSamples = inputs.size(); // Set numSamples based on loaded data
    std::cout << "Successfully loaded CSV. Samples: " << this->numSamples << std::endl;


    // --- Database Integration (only if not offline) ---
    if (!isOfflineMode && dbManager) {
        try {
            std::string actualDatasetName = datasetName.empty() ? filename : datasetName;
            // 1. Add/Update dataset metadata and get its ID
            currentDatasetId = dbManager->addDataset(actualDatasetName, "Data loaded from " + filename,
                this->numSamples, this->inputSize, this->outputSize);
            currentDatasetName = actualDatasetName;
            std::cout << "Dataset metadata added/updated with ID: " << currentDatasetId << std::endl;

            // 2. Clear existing records for this dataset before adding new ones
            dbManager->clearDatasetRecords(currentDatasetId);
            std::cout << "Cleared previous records for dataset ID: " << currentDatasetId << std::endl;


            // 3. Add individual dataset records
            for (int i = 0; i < inputs.size(); ++i) {
                // Ensure inputs[i] and targets[i] are already normalized/cleaned here.
                dbManager->addDatasetRecord(currentDatasetId, i, inputs[i], targets[i]);
            }
            std::cout << "All dataset records added to database for dataset ID: " << currentDatasetId << std::endl;
        }
        catch (const mysqlx::Error& err) {
            std::cerr << "Database error during CSV loading: " << err.what() << std::endl;
            return false; // Indicate failure due to database error
        }
        catch (const std::runtime_error& err) {
            std::cerr << "Runtime error during CSV loading (database related): " << err.what() << std::endl;
            return false;
        }
    }
    else if (!dbManager && !isOfflineMode) {
        std::cerr << "Warning: Database manager not initialized in online mode. CSV data will not be persisted.\n";
    }
    // --- End Database Integration ---

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
    bool firstTargetDataRow = true; // Flag to determine outputSize from the first data row if not already set

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(ss, cell, delim)) {
            cell.erase(0, cell.find_first_not_of(" \t\n\r\f\v"));
            cell.erase(cell.find_last_not_of(" \t\n\r\f\v") + 1);

            bool value_processed = false;

            try {
                if (!original_date_strings.empty()) {
                    try {
                        // Get the last date string and convert it to a timestamp
                        last_known_timestamp = convertDateTimeToTimestamp(original_date_strings.back());
                    }
                    catch (const std::runtime_error& e) {
                        std::cerr << "Error converting last date string to timestamp: " << e.what() << std::endl;
                        // Handle error, perhaps set to 0 or a default
                        last_known_timestamp = 0.0f;
                    }
                }
                else {
                    last_known_timestamp = 0.0f; // No date strings loaded
                }
                float timestamp = convertDateTimeToTimestamp(cell);
                row.push_back(timestamp);
                value_processed = true;
            }
            catch (const std::runtime_error&) {}

            if (!value_processed) {
                try {
                    bool is_numeric = true;
                    if (cell.empty()) is_numeric = false;
                    for (char c : cell) {
                        if (!(std::isdigit(c) || c == '.' || c == '-' || c == '+')) {
                            is_numeric = false;
                            break;
                        }
                    }
                    if (is_numeric) {
                        row.push_back(std::stof(cell));
                        value_processed = true;
                    }
                }
                catch (const std::exception&) {}
            }

            if (!value_processed && containsText) {
                if (!cell.empty()) {
                    row = lang->encodeText(cell);
                    value_processed = true;
                }
            }

            if (!value_processed) {
                row.push_back(0.0f);
            }
        }

        if (!row.empty()) {
            if (firstTargetDataRow) {
                // If outputSize is not yet determined (e.g., loadCSV not called), determine it from this first target row
                if (this->outputSize == 0) {
                    this->outputSize = row.size();
                    std::cout << "Determined Output Size from target CSV: " << this->outputSize << std::endl;
                }
                firstTargetDataRow = false;
            }

            // Now, validate all rows against the determined this->outputSize
            if (this->outputSize > 0 && row.size() != (int)this->outputSize) {
                std::cerr << "Error: Target file row " << row_idx << " has " << row.size()
                    << " columns, but expected " << this->outputSize << ". This is a critical mismatch. Aborting target loading.\n";
                temp_targets_data.clear(); // Clear any partially loaded data
                return false; // Critical error, stop loading targets
            }
            temp_targets_data.push_back(row);
            row_idx++;
            if (lines > 0) { // Only print if total lines is known
                printProgressBar("Loading Target CSV", row_idx, lines, barWidth);
            }
        }
    }
    file.close();

    if (temp_targets_data.empty()) {
        std::cerr << "Error: No valid target data rows found in '" << filename << "'." << std::endl;
        return false;
    }

    // --- Strict Validation of loaded data ---
    // Ensure all target rows have the same number of columns
    for (const auto& row : temp_targets_data) {
        if (row.size() != (int)this->outputSize) {
            std::cerr << "Error: Inconsistent target row size detected after parsing. Expected "
                << this->outputSize << " columns, but found a row with " << row.size() << " columns.\n";
            temp_targets_data.clear(); // Clear potentially inconsistent data
            return false; // Critical error
        }
    }
    // --- End Strict Validation ---

    // If targets are loaded, update the member variable
    this->targets = temp_targets_data;
    // Re-confirm outputSize based on loaded targets, or ensure it was already set correctly
    if (this->outputSize == 0 && !this->targets.empty()) {
        this->outputSize = this->targets[0].size();
    }


    // Ensure final state of progress bar is printed before newline
    if (lines > 0) {
        printProgressBar("Loading Target CSV", row_idx, lines, barWidth);
    }
    std::cout << "\nTarget CSV Loaded!\n";

    // --- Database Integration for Targets (only if not offline and dbManager is active) ---
    if (!isOfflineMode && dbManager && datasetId != -1) {
        try {
            // Assuming datasetId is valid and records already exist from loadCSV
            // We need to update the label_values for existing records
            for (int i = 0; i < targets.size(); ++i) {
                // This assumes a method like updateDatasetRecordLabels exists in Database.hpp/cpp
                dbManager->updateDatasetRecordLabels(datasetId, i, targets[i]);
            }
            std::cout << "All target records updated in database for dataset ID: " << datasetId << std::endl;
        }
        catch (const mysqlx::Error& err) {
            std::cerr << "Database error during target CSV loading/saving: " << err.what() << std::endl;
            return false;
        }
        catch (const std::runtime_error& err) {
            std::cerr << "Runtime error during target CSV loading/saving (database related): " << err.what() << std::endl;
            return false;
        }
    }
    else if (!dbManager && !isOfflineMode && datasetId != -1) {
        std::cerr << "Warning: Database manager not initialized in online mode. Target CSV data will not be persisted.\n";
    }
    // --- End Database Integration ---

    return true;
}

bool Training::loadDatasetFromDB(int& datasetId) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot load dataset from database in offline mode or if database manager is not initialized.\n";
        return false;
    }

    std::cout << "Loading dataset from database (ID: " << datasetId << ")...\n"; // Start message
    try {
        // This assumes Database::getDataset returns a struct or tuple
        // containing metadata and the actual data (inputs and targets)
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

        // --- Strict Validation of loaded data from DB ---
        if (!inputs.empty() && inputs[0].size() != (int)this->inputSize) {
            std::cerr << "Error: Inconsistent input row size loaded from DB. Expected "
                << this->inputSize << " columns, but first row has " << inputs[0].size() << " columns.\n";
            this->inputs.clear();
            this->targets.clear();
            return false;
        }
        for (const auto& row : inputs) {
            if (row.size() != (int)this->inputSize) {
                std::cerr << "Error: Inconsistent input row size loaded from DB. Expected "
                    << this->inputSize << " columns, but found a row with " << row.size() << " columns.\n";
                this->inputs.clear();
                this->targets.clear();
                return false;
            }
        }
        if (!targets.empty() && targets[0].size() != (int)this->outputSize) {
            std::cerr << "Error: Inconsistent target row size loaded from DB. Expected "
                << this->outputSize << " columns, but first row has " << targets[0].size() << " columns.\n";
            this->inputs.clear();
            this->targets.clear();
            return false;
        }
        for (const auto& row : targets) {
            if (row.size() != (int)this->outputSize) {
                std::cerr << "Error: Inconsistent target row size loaded from DB. Expected "
                    << this->outputSize << " columns, but found a row with " << row.size() << " columns.\n";
                this->inputs.clear();
                this->targets.clear();
                return false;
            }
        }
        // --- End Strict Validation of loaded data from DB ---

        // Note: original_date_strings and original_data_global_min/max are not loaded from DB in this path.
        // If needed, they must be part of your AIModelState or DatasetData struct.

        std::cout << "Successfully loaded dataset '" << this->currentDatasetName
            << "' (ID: " << this->currentDatasetId << ") from database.\n"; // End message
        return true;
    }
    catch (const mysqlx::Error& err) {
        std::cerr << "Database error loading dataset: " << err.what() << std::endl;
        return false;
    }
    catch (const std::runtime_error& err) {
        std::cerr << "Runtime error loading dataset from database: " << err.what() << std::endl;
        return false;
    }
}

void Training::splitInputOutput(int outputSize) {
    // This function might become less relevant if data is loaded directly into inputs/targets
    // from CSV or DB. Keep for backward compatibility or specific use cases.
    inputs.clear();
    targets.clear();

    for (const auto& row : raw_data) {
        if (row.size() < (int)outputSize) {
            std::cerr << "Warning: Row too short for outputSize. Skipping row.\n";
            continue;
        }

        std::vector<float> currentInput;
        std::vector<float> currentTarget;

        for (int i = 0; i < row.size() - outputSize; ++i) {
            currentInput.push_back(row[i]);
        }
        inputs.push_back(currentInput);

        for (int i = row.size() - outputSize; i < row.size(); ++i) {
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

    // Finding global min/max values across all inputs AND targets to normalize consistently
    float tempGlobalMin = this->getMinInputColumns();
    float tempGlobalMax = inputs.max_size();

    bool data_found_for_scaling = false; // Flag to ensure there's valid numerical data

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

    // Store the original global min/max for denormalization
    this->original_data_global_min = tempGlobalMin;
    this->original_data_global_max = tempGlobalMax;


    // Handle cases where no valid data was found or data is constant
    if (!data_found_for_scaling || std::abs(tempGlobalMax - tempGlobalMin) < std::numeric_limits<float>::epsilon()) {
        std::cerr << "Warning: Data is constant or contains only NaN/Inf. Mapping all values to target range based on NaN/Inf rules." << std::endl;
        for (auto& row : inputs) {
            for (float& val : row) {
                if (std::isinf(val)) {
                    val = maxRange; // Replace Inf with max of target range
                }
                else if (std::isnan(val)) {
                    val = minRange; // Replace NaN with min of target range
                }
                else {
                    val = minRange; // Map all other constant valid values to min of target range
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
        return; // Normalization complete for constant/NaN/Inf data
    }


    const int barWidth = 70;
    long long current_sample = 0;
    long long total_samples = inputs.size(); // Total samples for normalization

    std::cout << "\nNormalizing Data...\n";
    // Calculate the range of the data
    float dataRangeDiff = tempGlobalMax - tempGlobalMin;
    float targetRangeDiff = maxRange - minRange;


    for (int i = 0; i < inputs.size(); ++i) {
        for (int j = 0; j < inputs[i].size(); ++j) {
            float val = inputs[i][j];
            if (std::isinf(val)) {
                inputs[i][j] = maxRange; // Replace Inf with max of target range
            }
            else if (std::isnan(val)) {
                inputs[i][j] = minRange; // Replace NaN with min of target range
            }
            else {
                // Standard min-max normalization
                inputs[i][j] = minRange + ((val - tempGlobalMin) * targetRangeDiff) / dataRangeDiff;
            }
        }

        if (i < targets.size()) {
            for (int j = 0; j < targets[i].size(); ++j) {
                float val = targets[i][j];
                if (std::isinf(val)) {
                    targets[i][j] = maxRange; // Replace Inf with max of target range
                }
                else if (std::isnan(val)) {
                    targets[i][j] = minRange; // Replace NaN with min of target range
                }
                else {
                    // Standard min-max normalization
                    targets[i][j] = minRange + ((val - tempGlobalMin) * targetRangeDiff) / dataRangeDiff;
                }
            }
        }
        current_sample++;
        printProgressBar("Normalizing Data", current_sample, total_samples, barWidth);
    }
    // Ensure final state of progress bar is printed before newline
    if (total_samples > 0) {
        printProgressBar("Normalizing Data", total_samples, total_samples, barWidth);
    }
    std::cout << "\nData Normalized!\n";
}

int Training::getMinInputColumns() const {
    int min_columns = std::numeric_limits<int>::max();
    if (inputs.empty()) {
        return 0; // Or some indicator for an empty matrix
    }

    for (const auto& row : inputs) {
        if (row.size() < min_columns) {
            min_columns = row.size();
        }
    }
    return min_columns;
}

void Training::preprocess(float minVal, float maxVal) {
    // Determine the actual input and output sizes from loaded data
    if (inputs.empty() || targets.empty()) {
        std::cerr << "Error: No data to preprocess. Load data from file or database first." << std::endl;
        return;
    }

    // Call your robust normalization function
    normalize(minVal, maxVal);

    // Initialize CoreAI instance with actual data dimensions AFTER normalization
    core = std::make_unique<CoreAI>(this->inputSize, layers, neurons, this->outputSize, minVal, maxVal);
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
        core->train(inputs, targets, learningRate);
    }
    // Ensure final state of progress bar is printed before newline
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

    // Perform a forward pass to get the latest predictions
    core->forward(inputs);
    // Denormalize the predictions before calculating RMSE if they were normalized
    // This assumes core->denormalizeOutput() correctly denormalizes based on its own
    // knowledge of the normalization range (min/max passed to its constructor).
    core->denormalizeOutput(); // Denormalize predictions using training's min/max target range
    auto predictions = core->getResults(); // Get the (now denormalized) predictions

    // Denormalize targets using the original data's global min/max
    std::vector<std::vector<float>> denormalizedTargets = targets; // Create a copy to denormalize
    float target_range_min_val = this->min; // The 'min' value from `--min` argument
    float target_range_max_val = this->max; // The 'max' value from `--max` argument
    float target_range_diff = target_range_max_val - target_range_min_val;
    float original_data_range_diff = original_data_global_max - original_data_global_min;

    for (auto& row : predictions) {
        row.resize(this->outputSize);
    }

    for (auto& row : denormalizedTargets) {
        row.resize(this->outputSize);
        for (float& val : row) {
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon()) {
                val = ((val - target_range_min_val) / target_range_diff) * original_data_range_diff + original_data_global_min;
            }
            else {
                val = original_data_global_min; // If target range is zero, all values mapped to one point. Revert to original min.
            }
        }
    }



    float mse = 0.0f;
    int count = 0;

    for (int i = 0; i < denormalizedTargets.size(); ++i) {
        // Ensure rows have consistent sizes before accessing elements
        if (i < predictions.size() && denormalizedTargets[i].size() == predictions[i].size()) {
            for (int j = 0; j < denormalizedTargets[i].size(); ++j) {
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
    return lang.get();
}

void Training::saveModel(int& datasetId) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save model to database in offline mode or if database manager is not initialized.\n";
        return;
    }
    if (!core) {
        std::cerr << "Error: CoreAI not initialized. Cannot save model." << std::endl;
        return;
    }

    std::cout << "Saving AI model state to database (ID: " << datasetId << ")...\n"; // Start message

    // Retrieve the current state from CoreAI
    auto inputData = core->getInput();
    auto outputData = core->getOutput(); // This will be the target data
    auto hiddenData = core->getHiddenData(); // Now a 2D vector
    for (auto& row : hiddenData) {
        row.resize(this->inputSize);
    }
    auto hiddenOutputData = core->getHiddenOutputData();
    hiddenOutputData.resize(this->inputSize);
    
    auto hiddenErrorData = core->getHiddenErrorData();
    hiddenErrorData.resize(this->outputSize);
    auto weightsHiddenInput = core->getWeightsHiddenInput();
    weightsHiddenInput.resize(this->inputSize);
    auto weightsOutputHidden = core->getWeightsOutputHidden();
    for (auto& row : weightsOutputHidden) {
        row.resize(this->inputSize);
    }

    // Save the state to the database
    dbManager->saveAIModelState(datasetId, inputData, outputData,
        hiddenData, hiddenOutputData, hiddenErrorData,
        weightsHiddenInput, weightsOutputHidden);

    std::cout << "AI model state saved successfully.\n"; // End message
}

void Training::loadModel(int& datasetId) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot load model from database in offline mode or if database manager is not initialized.\n";
        return;
    }

    std::cout << "Loading AI model state from database (ID: " << datasetId << ")...\n"; // Start message

    // Load the latest model state from the database
    Database::AIModelState state = dbManager->loadLatestAIModelState(datasetId);

    // Check if state is valid
    if (state.inputData.empty() || state.outputData.empty() || state.weightsHiddenInput.empty() || state.weightsOutputHidden.empty()) {
        std::cerr << "No valid model state found or loaded for dataset ID " << datasetId << std::endl;
        return;
    }

    // Reinitialize CoreAI with loaded dimensions
    this->inputSize = state.inputData[0].size();
    this->outputSize = state.outputData[0].size();
    // Assuming samples, layers, neurons are implicitly known or can be derived,
    // or also saved/loaded as part of AIModelState. For now, they are passed
    // from main.cpp arguments.
    if (!core) {
        core = std::make_unique<CoreAI>(this->inputSize, layers, neurons, this->outputSize, min, max); // Removed samples from CoreAI constructor
    }

    // Set the loaded state to CoreAI
    core->setInput(state.inputData);
    core->setOutput(state.outputData);
    core->setHiddenData(state.hiddenData);
    core->setHiddenOutputData(state.hiddenOutputData);
    core->setHiddenErrorData(state.hiddenErrorData);
    core->setWeightsHiddenInput(state.weightsHiddenInput);
    core->setWeightsOutputHidden(state.weightsOutputHidden);

    std::cout << "Model loaded successfully for dataset ID " << datasetId << std::endl; // End message
}

void Training::printDenormalizedAsOriginalMatrix(std::vector<std::vector<float>>& normalized_data, int& len, int precision)
{
    if (normalized_data.empty()) {
        std::cout << "  (Empty matrix to denormalize and print)\n";
        return;
    }

    // Denormalize the data using the stored original min/max
    std::vector<std::vector<float>> denormalized_data = denormalizeMatrix(normalized_data);
    for (auto& row : denormalized_data) {
        row.resize(len);
    }
    std::cout << "  Denormalized Data (original scale):\n";
    // Use the printFullMatrix to print the denormalized data
    printFullMatrix(denormalized_data, len, precision);
}

void Training::printFullMatrix(std::vector<std::vector<float>>& data, int& len, int precision) {
    if (data.empty()) {
        std::cout << "  (Empty matrix)\n";
        return;
    }

    // Determine the maximum number of columns across all rows
    int maxCols = 0;

    for (auto& row : data) { // Add opening brace for the for loop body for clarity, though optional for single statement
        if (row.size() > maxCols) {
            maxCols = row.size();
        }
        row.resize(len);
    } // <-- This is the correct closing brace for the for loop


    // Initialize columnWidths based on the maximum number of columns
    std::vector<int> columnWidths(maxCols, 0);

    // Determine max width for each column based on all data
    for (const auto& row : data) {
        for (int j = 0; j < row.size(); ++j) {
            if (j < maxCols) { // Ensure we don't go out of bounds for columnWidths
                std::stringstream ss;
                ss << std::fixed << std::setprecision(precision) << row[j];
                columnWidths[j] = std::max(columnWidths[j], static_cast<int>(ss.str().length()));
            }
        }
    }

    // Print all rows and columns
    for (int i = 0; i < data.size(); ++i) {
        std::cout << "  [";
        for (int j = 0; j < data[i].size(); ++j) {
            if (j < maxCols) { // Ensure we use calculated width for existing columns
                std::cout << std::fixed << std::setprecision(precision) << std::setw(columnWidths[j]) << data[i][j];
            }
            else {
                // Handle cases where a row might have more columns than maxCols if maxCols was capped
                // (though in this function, maxCols is the true max).
                // Or just print without padding if beyond the common max.
                std::cout << std::fixed << std::setprecision(precision) << data[i][j];
            }

            if (j < data[i].size() - 1) { // Delimiter after element, not last column
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
}

// Function to denormalize a matrix
// Uses the stored original_data_global_min and original_data_global_max for accurate reversal.
std::vector<std::vector<float>> Training::denormalizeMatrix(const std::vector<std::vector<float>>& normalized_data) const {
    if (normalized_data.empty()) {
        return {};
    }

    std::vector<std::vector<float>> denormalized_copy;
    float target_range_min = this->min; // The 'min' value from `--min` argument
    float target_range_max = this->max; // The 'max' value from `--max` argument
    float target_range_diff = target_range_max - target_range_min;
    float original_data_range_diff = original_data_global_max - original_data_global_min;


    for (const auto& row : normalized_data) {
        std::vector<float> denormalized_row;
        for (float val : row) {
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon()) {
                // original = ((normalized - target_min) / target_range_diff) * original_data_range_diff + original_data_min
                denormalized_row.push_back(((val - target_range_min) / target_range_diff) * original_data_range_diff + original_data_global_min);
            }
            else {
                // If target range is zero, all values mapped to one point. Revert to original min.
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

    // Perform a forward pass to get the latest predictions
    core->forward(inputs);
    core->denormalizeOutput(); // Denormalize predictions using training's target min/max
    auto predictions = core->getResults(); // Get the (now denormalized) predictions

    // Denormalize the actual targets (labels) for output
    std::vector<std::vector<float>> denormalizedTargets = targets;
    float target_range_min_val = this->min;
    float target_range_max_val = this->max;
    float target_range_diff = target_range_max_val - target_range_min_val;
    float original_data_range_diff = original_data_global_max - original_data_global_min;

    for (auto& row : inputs) {
        for (float& val : row) {
            row.resize(this->inputSize);
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon()) {
                val = ((val - target_range_min_val) / target_range_diff) * original_data_range_diff + original_data_global_min;
            }
            else {
                val = original_data_global_min;
            }
        }
    }

    for (auto& row : denormalizedTargets) {
        for (float& val : row) {
            row.resize(this->outputSize);
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon()) {
                val = ((val - target_range_min_val) / target_range_diff) * original_data_range_diff + original_data_global_min;
            }
            else {
                val = original_data_global_min;
            }
        }
    }

    for (auto& row : predictions) {
        for (float& val : row) {
            row.resize(this->outputSize);
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon()) {
                val = ((val - target_range_min_val) / target_range_diff) * original_data_range_diff + original_data_global_min;
            }
            else {
                val = original_data_global_min;
            }
        }
    }




    // Read header from the original input file
    std::string originalInputHeaderLine = "";
    std::vector<std::string> originalInputHeaderCells;
    if (hasHeader) {
        std::ifstream inputFile(inputFilename);
        if (inputFile.is_open()) {
            std::getline(inputFile, originalInputHeaderLine);
            inputFile.close();
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
    // 1. Write Original Data (if includeOriginalData is true)
  
    for (size_t i = 0; i < inputs.size(); ++i) {
        // Assuming original_date_strings matches inputs size
        outputFile << original_date_strings[i]; // Write original date string
        outputFile << delimiter;

        // Write input features
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            outputFile << inputs[i][j];
            if (j < inputs[i].size() - 1) outputFile << delimiter;
        }
        // Add a placeholder or actual target if available for original data
        // outputFile << delimiter << targets[i][0]; // Example: if targets are single column
        outputFile << std::endl;
    }
    predictions = core->getResults(); // Or wherever your model's output is stored

    // Determine the time interval (e.g., daily = 24 * 60 * 60 seconds)
    // You will need to set this based on your data's frequency.
    // For example, if daily predictions, then:
    long long time_interval_seconds = 24 * 60 * 60; // Example: 1 day in seconds

    // Start projecting from the last known timestamp
    float current_projection_timestamp = last_known_timestamp;

    for (size_t i = 0; i < predictions.size(); ++i) {
        current_projection_timestamp += time_interval_seconds; // Increment for next future date
        outputFile << timestampToDateTimeString(current_projection_timestamp); // Write future date string
        outputFile << delimiter;


        // Add other input features (skipping the first two if they are ID and Date)
        // Assumes ID is column 0 and Date is column 1.
        // So, start from index 2 if originalInputHeaderCells is available and matches `inputSize`.
        // If not, use generic names.
        if (hasHeader && originalInputHeaderCells.size() > (int)inputSize && inputSize >= 2) {
            for (int i = 2; i < inputSize; ++i) { // Start from index 2 (after ID and Date)
                outputFile << delimiter << originalInputHeaderCells[i];
            }
        }
        else { // Generic headers for other inputs if no header or not enough columns in header
            for (int i = 2; i < inputSize; ++i) { // Assuming ID (0) and Date (1) handled separately
                outputFile << delimiter << "Input_Feature_" << (i + 1);
            }
        }


        // Add Actual Target Headers
        for (int i = 0; i < outputSize; ++i) {
            outputFile << delimiter << "Actual_Target_" << (i + 1);
        }

        // Add Predicted Target Headers
        for (int i = 0; i < outputSize; ++i) {
            outputFile << delimiter << "Predicted_Target_" << (i + 1);
        }

        outputFile << "\n"; // End of data row

    }


    // Write data rows
    for (int i = 0; i < inputs.size(); ++i) {
        // 1. Write Date
        if (i < original_date_strings.size()) {
            outputFile << original_date_strings[i];
        }
        else {
            outputFile << ""; // Empty if no date string
        }
        outputFile << delimiter;


        // 2. Write Other Input Features (skipping ID and Date, which are at index 0 and 1)
        // Ensure inputSize is at least 2 to skip ID and Date.
        for (int j = 2; j < inputs[i].size(); ++j) { // Start from index 2
            outputFile << std::fixed << std::setprecision(4) << formatValueForDisplay(inputs[i][j], 2);
            outputFile << delimiter;
        }

        // 3. Write Actual Targets (denormalized)
        for (int j = 0; j < denormalizedTargets[i].size(); ++j) {
            outputFile << std::fixed << std::setprecision(4) << formatValueForDisplay(denormalizedTargets[i][j], 2);
            outputFile << delimiter;
        }

        // 4. Write Predicted Targets (denormalized)
        for (int j = 0; j < predictions[i].size(); ++j) {
            outputFile << std::fixed << std::setprecision(-2) << formatValueForDisplay(predictions[i][j], 2);
            if (j < predictions[i].size() - 1) { // No delimiter after last predicted target in row
                outputFile << delimiter;
            }
        }
    }

    outputFile.close();
    std::cout << "Results saved to: " << filename << std::endl;
    return true;
}

std::string Training::timestampToDateTimeString(float timestamp) {
    std::time_t t = static_cast<std::time_t>(timestamp);
    std::tm tm = *std::localtime(&t); // Use std::gmtime for UTC, std::localtime for local time
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

std::string Training::formatValueForDisplay(float value, int customPrecision) const {
    std::stringstream ss;

    // New Rule: Handle extremely large numbers (e.g., 10^21 values)
    // Check for values roughly above 10^18 to catch these
    if (std::abs(value) >= 1.0e18f) { // Approximately 10^18
        // To get from ~10^21 to ~10, divide by 10^20
        ss << std::fixed << std::setprecision(2) << (value / 1.0e20f);
    }
    // Existing Rule: Handle very large numbers (e.g., billions)
    else if (std::abs(value) >= 1000000000.0f) { // 1 billion (10^9)
        ss << std::fixed << std::setprecision(customPrecision) << (value / 100000000.0f); // Scales down for billions
    }
    // Existing Rule: Handle numbers in the range of hundreds of thousands to millions (like 494585.750, 789781.375)
    else if (std::abs(value) >= 1000.0f && std::abs(value) < 1000000000.0f) { // Between 100K (10^5) and 1 Billion (10^9)
        ss << std::fixed << std::setprecision(2) << (value / 10000.0f); // Scales down by 10^4
    }
    // Existing Rule: Handle small numbers (e.g., 0.019 to 19)
    else if (std::abs(value) >= 0.01f && std::abs(value) < 1.0f) {
        ss << std::fixed << std::setprecision(0) << std::round(value * 1000.0f);
    }
    // Default Rule: For other magnitudes, print with a default precision
    else {
        ss << std::fixed << std::setprecision(customPrecision + 4) << value;
    }
    return ss.str();
}

int Training::getDecimalPlacesInString(float value, int precision_for_conversion = 6) {
    std::stringstream ss;
    // Convert float to string with a specific precision
    // Using std::fixed ensures a fixed number of digits after the decimal point
    ss << std::fixed << std::setprecision(precision_for_conversion) << value;
    std::string s = ss.str();

    // Find the position of the decimal point
    int decimal_pos = s.find('.');

    if (decimal_pos == std::string::npos) {
        return 0; // No decimal point found
    }

    // Count digits after the decimal point
    int count = 0;
    for (int i = decimal_pos + 1; i < s.length(); ++i) {
        if (std::isdigit(s[i])) { // Ensure it's a digit
            count++;
        }
        else {
            // Stop if we encounter a non-digit character (e.g., scientific notation suffix)
            break;
        }
    }

    // You might want to trim trailing zeros if they are not considered "significant" decimal places
    // For example, 12.3400 might be considered 2 decimal places.
    // This logic is more complex and depends on your definition.
    // For now, it counts all digits after the decimal point.
    // To trim trailing zeros:
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
                normalized_row.push_back(target_min + 0.1f);
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
    outputSize = 1; // You can change this if your CSV includes labels

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto getOrCreateEmbedding = [&](const std::string& word) -> std::vector<float> {
        auto embeddingMap = lang->loadWordEmbeddingsFromFile(filename, embeddingDim);
        auto it = embeddingMap.find(word);
        if (it != embeddingMap.end()) return it->second;

        std::vector<float> vec(embeddingDim);
        for (int i = 0; i < embeddingDim; ++i)
            vec[i] = dist(rng);
        
        lang->embeddingsByLang[lang->currentLang][word] = vec;
        return vec;
        };

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        std::vector<std::vector<float>> sentenceEmbedding;

        while (iss >> word) {
            sentenceEmbedding.push_back(getOrCreateEmbedding(word));
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

        // Dummy target for now
        targets.push_back({ 0.0f }); // You can adapt this if your CSV has labels
    }

    file.close();
    numSamples = inputs.size();

    std::cout << "Loaded " << numSamples << " text samples from " << filename << std::endl;
    return !inputs.empty();
}



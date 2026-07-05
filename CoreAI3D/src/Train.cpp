#include "Train.hpp"
#include "WebModule.hpp"
#include "VisionModule.hpp"
#include "AudioModule.hpp"
#include <filesystem>
#include <cstring>
// Constructor for online mode (with database) - only if USE_MYSQL is defined
#ifdef USE_MYSQL
Training::Training(const std::string& dbHost, unsigned int dbPort, const std::string& dbUser,
    const std::string dbPassword, const std::string& dbSchema, int sslDummy, bool createTables, bool verbose, WebSocketServer* wsServer)
    : dbManager(std::make_unique<Database>(dbHost, dbPort, dbUser, dbPassword, dbSchema, SSLMode::REQUIRED, createTables)),
    isOfflineMode(false), currentDatasetId(-1), numSamples(0), inputSize(0), outputSize(0),
    original_data_global_min(std::numeric_limits<float>::max()), original_data_global_max(std::numeric_limits<float>::lowest()), // Initialize
    input_min(), input_max(), target_min(0.0f), target_max(0.0f),
    layers(0), neurons(0), learningRate(0.0), min(0.0f), max(0.0f), last_known_timestamp(0.0f), // Initialize
    verbose(verbose), // Initialize verbosgit reset HEAD custom-ports/mysql-connector-cpp/vcpkg_installede from parameter
    webSocketServer(wsServer), // Initialize WebSocket server
    trainingActive(false), trainingStopRequested(false), currentEpoch(0), totalEpochs(0), currentLoss(0.0f), // Initialize training state
    conversionProgress(0), conversionTotal(0), // Initialize progress tracking
    gen(std::random_device{}()) // Initialize random number generator
{
    // Parameter validation
    if (dbHost.empty() || dbUser.empty() || dbSchema.empty()) {
        throw std::invalid_argument("Training: Database connection parameters cannot be empty");
    }
    if (dbPort == 0) {
        throw std::invalid_argument("Training: Database port must be positive");
    }
}
#endif

// Constructor for offline mode (no database)
Training::Training(bool isOffline, bool verbose, WebSocketServer* wsServer)
    : isOfflineMode(isOffline), currentDatasetId(-1), numSamples(0), inputSize(0), outputSize(0),
    original_data_global_min(std::numeric_limits<float>::max()), original_data_global_max(std::numeric_limits<float>::lowest()), // Initialize
    input_min(), input_max(), target_min(0.0f), target_max(0.0f),
    layers(0), neurons(0), learningRate(0.0), min(0.0f), max(1.0f), last_known_timestamp(0.0f), // Initialize min/max to [0,1] for normalization
    verbose(verbose), // Initialize verbose from parameter
    webSocketServer(wsServer), // Initialize WebSocket server
    trainingActive(false), trainingStopRequested(false), currentEpoch(0), totalEpochs(0), currentLoss(0.0f), // Initialize training state
    conversionProgress(0), conversionTotal(0), // Initialize progress tracking
    gen(std::random_device{}()) // Initialize random number generator
{
    if (isOfflineMode && verbose) {
        std::cout << "Training initialized in OFFLINE mode. Database operations are disabled." << std::endl;
    }
}

// Helper to initialize Language processor (called from main.cpp)
void Training::initializeLanguageProcessor(std::string& embedingFile, int& embeddingDim, std::string& dbHost, int& dbPort,
    std::string& dbUser, std::string& dbPassword, std::string& dbSchema, int sslDummy, std::string& lang, int& inputSize, int& outputSize, int& layers, int& neurons) {
    this->langProc = std::make_unique<Language>(embedingFile, embeddingDim, dbHost, dbPort, dbUser, dbPassword, dbSchema, 0, lang, inputSize, outputSize, layers, neurons);
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
            // Try parsing year-month only (e.g., "YYYY-MM" for monthly data)
            ss.clear();
            ss.str(cleaned_str);
            ss >> std::get_time(&tm, "%Y-%m");
            if (ss.fail()) {
                throw std::runtime_error(std::string("Invalid datetime format: ") + cleaned_str);
            }
            // For year-month only, set day to 1 so mktime doesn't use 0
            tm.tm_mday = 1;
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
    // DIAGNOSTIC LOG: Enhanced logging for debugging
    if (verbose) {
        std::cout << "[DIAGNOSTIC] Training::loadCSV START - filename='" << filename
                  << "', outputSizeParam=" << outputSizeParam << ", hasHeader=" << hasHeader
                  << ", containsText=" << containsText << ", delimiter='" << delimitor << "'" << std::endl;
    }
    
    // Check if file exists
    if (!std::filesystem::exists(filename)) {
        std::cerr << "ERROR: File does not exist: " << filename << std::endl;
        return false;
    }
    if (verbose) {
        std::cout << "[DIAGNOSTIC] File exists: " << filename << std::endl;
    }


    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open input CSV: " << filename << std::endl;
        if (verbose) {
            std::cerr << "[DIAGNOSTIC] Current working directory: " << std::filesystem::current_path() << std::endl;
        }
        return false;
    }
    if (verbose) {
        std::cout << "[DIAGNOSTIC] File opened successfully: " << filename << std::endl;
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
        if (verbose) {
            std::cout << "[DEBUG] Skipped header line: '" << line << "'" << std::endl;
        }
    }

    long long rowIndex = 0;
    bool firstDataRow = true;
    try {
        while (true) {
            if (!std::getline(file, line)) break;
            lineCounter++;
            if (verbose && lineCounter % 100 == 0) { // Only print every 100 lines to reduce verbosity
                std::cout << "[DEBUG] Processing line " << lineCounter << ": '" << line << "'" << std::endl;
            }

            // Skip empty lines
            if (line.empty()) {
                if (verbose && lineCounter % 100 == 0) {
                    std::cout << "[DEBUG] Skipping empty line " << lineCounter << std::endl;
                }
                continue; // Re-enabled continue for empty lines
            }

            // Trim whitespace from the line
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            if (line.empty()) {
                if (verbose && lineCounter % 100 == 0) {
                    std::cout << "[DEBUG] Skipping whitespace-only line " << lineCounter << std::endl;
                }
                continue; // Re-enabled continue for whitespace-only lines
            }

            if (verbose) {
                std::cout << "[DEBUG] After trimming: '" << line << "'" << std::endl;

                // Check if line starts with a number (basic validation)
                if (!line.empty() && !std::isdigit(line[0]) && line[0] != '-' && line[0] != '+') {
                    if (verbose) {
                        std::cout << "[DEBUG] Line doesn't start with digit, minus, or plus - might be invalid data" << std::endl;
                    }
                }

                if (verbose) {
                    std::cout << "[DEBUG] About to parse cells from line: '" << line << "'" << std::endl;

                    // FORCE PROCESSING - let's see what happens if we force it to process
                    std::cout << "[DEBUG] FORCE PROCESSING: Will attempt to process this line anyway" << std::endl;

                    // Check if we reached end of file
                    if (file.eof()) {
                        std::cout << "[DEBUG] Reached end of file" << std::endl;
                    }

                    // Let's try to force processing by temporarily commenting out the continue statements
                    // This will help us see what happens when we try to process lines that might be skipped
                    std::cout << "[DEBUG] Temporarily forcing processing of all lines..." << std::endl;
                }
            }

            std::vector<std::string> current_raw_cells;
            std::vector<float> current_processed_values; // This will hold the converted floats/embeddings

            // 1. Read all cells as raw strings first using proper CSV parsing
            if (verbose) {
                std::cout << "[DIAGNOSTIC] Parsing line " << lineCounter << " with delimiter '" << delimitor << "'" << std::endl;
                std::cout << "[DIAGNOSTIC] Original line: '" << line << "'" << std::endl;
            }
            
            current_raw_cells = parseCSVLine(line, delimitor);
            
            if (verbose) {
                std::cout << "[DIAGNOSTIC] Parsed " << current_raw_cells.size() << " raw cells: ";
                for (size_t i = 0; i < current_raw_cells.size(); ++i) {
                    std::cout << "'" << current_raw_cells[i] << "'";
                    if (i < current_raw_cells.size() - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            }
            
            // Trim whitespace from cells
            for (auto& cell : current_raw_cells) {
                cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                if (verbose) {
                    std::cout << "[DIAGNOSTIC] Trimmed cell: '" << cell << "'" << std::endl;
                }
            }

            if (current_raw_cells.empty()) {
                if (verbose) {
                    std::cout << "[DIAGNOSTIC] WARNING: Skipping empty row at line " << lineCounter << " (no cells after parsing)" << std::endl;
                }
                continue; // Skip empty rows
            }

            // 2. Process raw cells into float values or embeddings
            if (verbose) {
                std::cout << "[DEBUG] Processing cells for row " << rowIndex << " (line " << lineCounter << ")" << std::endl;
            }
            for (size_t i = 0; i < current_raw_cells.size(); ++i) {
                const std::string& cell = current_raw_cells[i];
                bool value_processed = false;

                if (verbose) {
                    std::cout << "[DEBUG]   Cell " << i << ": '" << cell << "' -> ";
                }

                // Try to parse as datetime first if it looks like one
                // Allow flexible date lengths (YYYY-MM is 7 chars, YYYY-MM-DD is 10)
                if ((cell.find("/") != std::string::npos || cell.find("-") != std::string::npos) && cell.length() >= 6) {
                    try {
                        float timestamp = convertDateTimeToTimestamp(cell);
                        current_processed_values.push_back(timestamp);
                        value_processed = true;
                        if (verbose) {
                            std::cout << "datetime: " << timestamp;
                        }
                        // Store original date string if this is the date column
                        // Assuming date is at a known index, e.g., 0 (first column)
                        if (i == 0) { // Adjust this index if your date column is elsewhere
                            original_date_strings.push_back(cell);
                        }
                    }
                    catch (const std::runtime_error& e) {
                        if (verbose) {
                            std::cout << "not datetime (" << e.what() << ") -> ";
                        }
                        // Not a valid datetime, fall through to float or text processing
                        // std::cerr << "Debug: Not a datetime string or conversion error: " << cell << " - " << e.what() << std::endl;
                    }
                }

                if (!value_processed) {
                    // Special parsing for Vol (column 5) and Change % (column 6)
                    if (i == 5) { // Vol column
                        std::string vol_str = cell;
                        if (vol_str.empty()) {
                            current_processed_values.push_back(0.0f);
                            value_processed = true;
                            if (verbose) {
                                std::cout << "vol empty -> 0.0f";
                            }
                        } else {
                            if (vol_str.back() == 'K') {
                                vol_str = vol_str.substr(0, vol_str.size() - 1);
                                try {
                                    float val = std::stof(vol_str) * 1000.0f;
                                    current_processed_values.push_back(val);
                                    value_processed = true;
                                    if (verbose) {
                                        std::cout << "vol: " << val;
                                    }
                                } catch (const std::exception& e) {
                                    if (verbose) {
                                        std::cout << "vol parse error -> 0.0f";
                                    }
                                    current_processed_values.push_back(0.0f);
                                    value_processed = true;
                                }
                            } else {
                                // Assume plain number, remove commas
                                std::string cleaned = vol_str;
                                cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ','), cleaned.end());
                                try {
                                    float val = std::stof(cleaned);
                                    current_processed_values.push_back(val);
                                    value_processed = true;
                                    if (verbose) {
                                        std::cout << "vol plain: " << val;
                                    }
                                } catch (const std::exception& e) {
                                    if (verbose) {
                                        std::cout << "vol parse error -> 0.0f";
                                    }
                                    current_processed_values.push_back(0.0f);
                                    value_processed = true;
                                }
                            }
                        }
                    } else if (i == 6) { // Change % column
                        std::string change_str = cell;
                        if (change_str.back() == '%') {
                            change_str = change_str.substr(0, change_str.size() - 1);
                            try {
                                float val = std::stof(change_str) / 100.0f;
                                current_processed_values.push_back(val);
                                value_processed = true;
                                if (verbose) {
                                    std::cout << "change%: " << val;
                                }
                            } catch (const std::exception& e) {
                                if (verbose) {
                                    std::cout << "change% parse error -> 0.0f";
                                }
                                current_processed_values.push_back(0.0f);
                                value_processed = true;
                            }
                        } else {
                            // Assume plain number
                            try {
                                float val = std::stof(change_str);
                                current_processed_values.push_back(val);
                                value_processed = true;
                                if (verbose) {
                                    std::cout << "change plain: " << val;
                                }
                            } catch (const std::exception& e) {
                                if (verbose) {
                                    std::cout << "change parse error -> 0.0f";
                                }
                                current_processed_values.push_back(0.0f);
                                value_processed = true;
                            }
                        }
                    } else {
                        // Try to parse as float, remove commas first
                        std::string cleaned_cell = cell;
                        cleaned_cell.erase(std::remove(cleaned_cell.begin(), cleaned_cell.end(), ','), cleaned_cell.end());
                        try {
                            float float_val = std::stof(cleaned_cell);
                            current_processed_values.push_back(float_val);
                            value_processed = true;
                            if (verbose) {
                                std::cout << "float: " << float_val;
                            }
                        }
                        catch (const std::invalid_argument& e) {
                            if (verbose) {
                                std::cout << "not float (" << e.what() << ") -> ";
                            }
                            // Not a valid float, fall through to text processing
                            // std::cerr << "Debug: Not a float: " << cell << " - " << e.what() << std::endl;
                        }
                        catch (const std::out_of_range& e) {
                            std::cerr << "Warning: Number out of range in CSV at line " << lineCounter << ", cell '" << cell << "'. Defaulting to 0.0f. Error: " << e.what() << std::endl;
                            current_processed_values.push_back(0.0f);
                            value_processed = true;
                            if (verbose) {
                                std::cout << "out_of_range -> 0.0f";
                            }
                        }
                    }
                }

                if (!value_processed && containsText) {
                    // If it contains text, encode it. For CSV parsing, we use a single hash value
                    // instead of full embeddings to maintain 1 value per cell structure
                    if (this->langProc) {
                        // Use a simple hash of the text for CSV compatibility
                        // This maintains the 1-value-per-cell structure expected by CSV parsing
                        std::hash<std::string> hasher;
                        size_t hash_value = hasher(cell);
                        float normalized_hash = static_cast<float>(hash_value % 1000000) / 1000000.0f; // Normalize to 0-1 range
                        current_processed_values.push_back(normalized_hash);
                        value_processed = true;
                        if (verbose) {
                            std::cout << "text hashed -> " << normalized_hash;
                        }
                    }
                    else {
                        std::cerr << "Error: Language processor not initialized for text encoding at line " << lineCounter << ". Skipping text cell '" << cell << "'." << std::endl;
                        // Push a placeholder if language processor is not ready
                        current_processed_values.push_back(0.0f);
                        value_processed = true;
                        if (verbose) {
                            std::cout << "text (no langProc) -> 0.0f";
                        }
                    }
                }

                if (!value_processed) {
                    // If still not processed, it's an unhandled type, default to 0.0f
                    current_processed_values.push_back(0.0f);
                    if (verbose) {
                        std::cout << "unhandled -> 0.0f";
                    }
                }
                if (verbose) {
                    std::cout << std::endl;
                }
            } // End of for (current_raw_cells)

            if (verbose) {
                std::cout << "[DEBUG] Processed values count: " << current_processed_values.size() << std::endl;
            }

            if (current_processed_values.empty()) {
                if (verbose) {
                    std::cout << "[DEBUG] Skipping row " << rowIndex << " (line " << lineCounter << ") - no processed values" << std::endl;
                    std::cout << "[DEBUG] Raw cells were: ";
                    for (size_t i = 0; i < current_raw_cells.size(); ++i) {
                        std::cout << "'" << current_raw_cells[i] << "'";
                        if (i < current_raw_cells.size() - 1) std::cout << ", ";
                    }
                    std::cout << std::endl;
                }
                continue; // Skip if row became empty after processing
            }

            // Determine inputSize and outputSize from the first data row
            if (firstDataRow) {
                if (verbose) {
                    std::cout << "[DIAGNOSTIC] First data row - determining sizes. Processed values: " << current_processed_values.size()
                            << ", outputSizeParam: " << outputSizeParam << std::endl;
                }

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
                    if (verbose) {
                        std::cout << "[DIAGNOSTIC] Applied heuristic: outputSize=1, inputSize=" << this->inputSize << std::endl;
                    }
                }
                else if (this->outputSize == 0 && current_processed_values.size() == 1) { // Only one column, assume inputSize 1, outputSize 0 or 1
                    this->outputSize = 0; // No target if only one column and not specified
                    this->inputSize = 1;
                    if (verbose) {
                        std::cout << "[DIAGNOSTIC] Single column: outputSize=0, inputSize=1" << std::endl;
                    }
                }
                firstDataRow = false;
                if (verbose) {
                    std::cout << "[DIAGNOSTIC] Determined Input Size: " << this->inputSize << ", Output Size: " << this->outputSize << std::endl;
                }
            }
            else {
                // Validate subsequent rows against determined sizes
                size_t expected_columns = (size_t)(this->inputSize + this->outputSize);
                if (current_processed_values.size() != expected_columns) {
                    if (verbose) {
                        std::cout << "[DEBUG] Adjusting row " << rowIndex << " size from " << current_processed_values.size() << " to " << expected_columns << std::endl;
                    }
                    current_processed_values.resize(expected_columns, 0.0f);
                }
            }

            // Extract inputs and targets
            std::vector<float> current_input;
            std::vector<float> current_target;

            if (verbose) {
                std::cout << "[DEBUG] Extracting inputs and targets based on outputSize=" << outputSizeParam << std::endl;
            }

            // Targets are the last outputSizeParam columns
            for (int i = 0; i < outputSizeParam; ++i) {
                size_t target_index = current_processed_values.size() - outputSizeParam + i;
                if (target_index < current_processed_values.size()) {
                    current_target.push_back(current_processed_values[target_index]);
                } else {
                    std::cerr << "[ERROR] Not enough columns for target extraction at row " << rowIndex << std::endl;
                    return false;
                }
            }

            // Inputs are all columns except the last outputSizeParam
            for (size_t i = 0; i < current_processed_values.size() - outputSizeParam; ++i) {
                current_input.push_back(current_processed_values[i]);
            }

            if (verbose) {
                std::cout << "[DEBUG] Row " << rowIndex << " - Input size: " << current_input.size()
                        << ", Target size: " << current_target.size() << std::endl;
            }

            inputs.push_back(current_input);
            targets.push_back(current_target);

            rowIndex++;
            if (numSamplesToLoad > 0 && rowIndex >= numSamplesToLoad) break;
        }
   

    if (inputs.empty()) {
        std::cerr << "ERROR: No valid data rows found in CSV or all rows were skipped due to errors." << std::endl;
        if (verbose) {
            std::cerr << "[DIAGNOSTIC] Total lines processed: " << lineCounter << std::endl;
        }
        return false;
    }

    if (verbose) {
        std::cout << "[DIAGNOSTIC] Final validation - Total rows processed: " << inputs.size() << std::endl;
    }

    // Strict Validation
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() != (size_t)this->inputSize) {
            std::cerr << "[DEBUG] Error: Inconsistent input row size detected after parsing. Row " << i << " expected "
                << this->inputSize << " columns, but found " << inputs[i].size() << " columns.\n";
            inputs.clear(); targets.clear(); original_date_strings.clear(); return false;
        }
    }
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i].size() != (size_t)this->outputSize) {
            std::cerr << "[DEBUG] Error: Inconsistent target row size detected after parsing. Row " << i << " expected "
                << this->outputSize << " columns, but found " << targets[i].size() << " columns.\n";
            inputs.clear(); targets.clear(); original_date_strings.clear(); return false;
        }
    }

    this->numSamples = inputs.size();
    if (verbose) {
        std::cout << "[DIAGNOSTIC] Training::loadCSV: Set numSamples to " << this->numSamples
                  << ", inputs.size() = " << inputs.size() << ", targets.size() = " << targets.size() << std::endl;
    }
    if (verbose) {
        std::cout << "[DIAGNOSTIC] SUCCESS: Successfully loaded CSV. Samples: " << this->numSamples
                  << ", Input Size: " << this->inputSize << ", Output Size: " << this->outputSize << std::endl;
    }

    // Database Integration - only if USE_MYSQL is defined
#ifdef USE_MYSQL
    if (!isOfflineMode && dbManager) {
        try {
            std::string actualDatasetName = datasetName.empty() ? filename : datasetName;
            currentDatasetId = dbManager->addDataset(actualDatasetName, std::string("Data loaded from ") + filename,
                this->numSamples, this->inputSize, this->outputSize);
            currentDatasetName = actualDatasetName;
            if (verbose) {
                std::cout << "Dataset metadata added/updated with ID: " << currentDatasetId << std::endl;
            }
            long long datasetId = static_cast<long long>(currentDatasetId);
            dbManager->clearDatasetRecords(datasetId);
            if (verbose) {
                std::cout << "Cleared previous records for dataset ID: " << (int)currentDatasetId << std::endl;
            }

            for (size_t i = 0; i < inputs.size(); ++i) {
                // Ensure vectors match expected sizes before database insertion
                if (inputs[i].size() != (size_t)this->inputSize) {
                    if (verbose) {
                        std::cout << "[DEBUG] Resizing input[" << i << "] from " << inputs[i].size() << " to " << this->inputSize << std::endl;
                    }
                    inputs[i].resize(this->inputSize, 0.0f);
                }
                if (targets[i].size() != (size_t)this->outputSize) {
                    if (verbose) {
                        std::cout << "[DEBUG] Resizing target[" << i << "] from " << targets[i].size() << " to " << this->outputSize << std::endl;
                    }
                    targets[i].resize(this->outputSize, 0.0f);
                }
                try {
                    dbManager->addDatasetRecord(currentDatasetId, i, inputs[i], targets[i]);
                    if (verbose && i % 100 == 0) {
                        std::cout << "Saved dataset record " << i << " to database" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error saving dataset record " << i << ": " << e.what() << std::endl;
                }
            }
            if (verbose) {
                std::cout << "All dataset records added to database for dataset ID: " << currentDatasetId << std::endl;
            }
        }
        catch (const std::runtime_error& err) {
            std::cerr << "Runtime error during CSV loading (database related): " << err.what() << std::endl; return false;
        }
        catch (const std::exception& err) {
            std::cerr << "Database error during CSV loading: " << err.what() << std::endl; return false;
        }
    }
    else if (!dbManager && !isOfflineMode) {
        std::cerr << "Warning: Database manager not initialized in online mode. CSV data will not be persisted." << std::endl;
    }
#endif

    if (verbose) {
        std::cout << "Successfully loaded CSV: " << filename << std::endl;
    }
    return true;
}
catch (const std::exception& e) {
    std::cerr << "[DEBUG] Exception during CSV loading at line " << lineCounter << ": " << e.what() << std::endl;
    file.close();
    return false;
}
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

#ifdef USE_MYSQL
    if (!isOfflineMode && dbManager && datasetId != -1) {
        try {
            for (size_t i = 0; i < targets.size(); ++i) {
                dbManager->updateDatasetRecordLabels(datasetId, i, targets[i]);
            }
            std::cout << "All target records updated in database for dataset ID: " << datasetId << std::endl;
        }
        catch (const std::runtime_error& err) {
            std::cerr << "Runtime error during target CSV loading/saving (database related): " << err.what() << std::endl; return false;
        }
        catch (const std::exception& err) {
            std::cerr << "Database error during target CSV loading/saving: " << err.what() << std::endl; return false;
        }
    }
    else if (!dbManager && !isOfflineMode && datasetId != -1) {
        std::cerr << "Warning: Database manager not initialized in online mode. Target CSV data will not be persisted." << std::endl;
    }
#endif

    return true;
}

bool Training::loadDatasetFromDB(int& datasetId) {
#ifdef USE_MYSQL
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
    catch (const std::runtime_error& err) {
        std::cerr << "Runtime error loading dataset from database: " << err.what() << std::endl; return false;
    }
    catch (const std::exception& err) {
        std::cerr << "Database error loading dataset: " << err.what() << std::endl; return false;
    }
#else
    std::cerr << "Database functionality not available (USE_MYSQL not defined)." << std::endl;
    return false;
#endif
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
    if (inputs.empty() && targets.empty()) {
        std::cerr << "Warning: No data to normalize." << std::endl;
        return;
    }

    if (minRange >= maxRange) {
        throw std::invalid_argument("Training::normalize: minRange must be less than maxRange");
    }

    // Step 1: Compute independent bounds for each feature dimension
    compute_normalization_bounds();

    // DEBUG: Log normalization parameters
    if (verbose) {
        std::cout << "[DEBUG NORMALIZE] original_data_global_min: " << this->original_data_global_min
                  << ", original_data_global_max: " << this->original_data_global_max
                  << ", range: " << (this->original_data_global_max - this->original_data_global_min) << std::endl;
        size_t num_input_features = inputs.empty() ? 0 : inputs[0].size();
        size_t num_target_features = targets.empty() ? 0 : targets[0].size();
        for (size_t j = 0; j < num_input_features; ++j) {
            std::cout << "[DEBUG NORMALIZE] Input feature " << j << " bounds: [" << input_min[j] << ", " << input_max[j] << "]" << std::endl;
        }
        for (size_t j = 0; j < num_target_features; ++j) {
            std::cout << "[DEBUG NORMALIZE] Target feature " << j << " bounds: [" << target_mins[j] << ", " << target_maxs[j] << "]" << std::endl;
        }
    }

    // Step 2: Normalize inputs and targets using their independent bounds
    normalize_inputs(minRange, maxRange);
    normalize_targets(minRange, maxRange);
}

// ============================================================================
// CORE NORMALIZATION METHODS: Independent X,Y -> Z Mapping
// ============================================================================

// Computes and stores per-feature bounds for inputs, per-target bounds for outputs,
// and global bounds for backward compatibility.
// This is the first step in the independent normalization pipeline.
// Memory-safe: Explicitly resizes input_min/input_max/target_mins/target_maxs before use.
void Training::compute_normalization_bounds() {
    size_t num_input_features = inputs.empty() ? 0 : inputs[0].size();
    size_t num_target_features = targets.empty() ? 0 : targets[0].size();

    // ========================================================================
    // PHASE 1: Compute per-feature min/max for inputs independently
    // Each input dimension (x, y) gets its own [X_min, X_max] interval.
    // This prevents features with large numerical ranges (e.g., Unix timestamps)
    // from dominating the normalization and collapsing other features.
    // ========================================================================
    input_min.assign(num_input_features, 0.0f);
    input_max.assign(num_input_features, 1.0f);

    for (size_t j = 0; j < num_input_features; ++j) {
        float featureMin = std::numeric_limits<float>::max();
        float featureMax = std::numeric_limits<float>::lowest();
        bool hasValidData = false;

        for (const auto& row : inputs) {
            if (j < row.size()) {
                float val = row[j];
                if (!std::isnan(val) && !std::isinf(val)) {
                    featureMin = std::min(featureMin, val);
                    featureMax = std::max(featureMax, val);
                    hasValidData = true;
                }
            }
        }

        if (hasValidData) {
            input_min[j] = featureMin;
            input_max[j] = featureMax;
            // Defensive: Expand zero range to avoid divide-by-zero in denorm
            float featureRangeDiff = featureMax - featureMin;
            if (std::abs(featureRangeDiff) < std::numeric_limits<float>::epsilon()) {
                input_max[j] = featureMin + 1.0f;
            }
        }
    }

    // ========================================================================
    // PHASE 2: Compute per-target min/max for outputs independently
    // Each target dimension (z) gets its own [Z_min, Z_max] interval.
    // ========================================================================
    target_mins.assign(num_target_features, 0.0f);
    target_maxs.assign(num_target_features, 1.0f);

    for (size_t j = 0; j < num_target_features; ++j) {
        float featureMin = std::numeric_limits<float>::max();
        float featureMax = std::numeric_limits<float>::lowest();
        bool hasValidData = false;

        for (const auto& row : targets) {
            if (j < row.size()) {
                float val = row[j];
                if (!std::isnan(val) && !std::isinf(val)) {
                    featureMin = std::min(featureMin, val);
                    featureMax = std::max(featureMax, val);
                    hasValidData = true;
                }
            }
        }

        if (hasValidData) {
            target_mins[j] = featureMin;
            target_maxs[j] = featureMax;
            // Defensive: Expand zero range to avoid divide-by-zero in denorm
            float featureRangeDiff = featureMax - featureMin;
            if (std::abs(featureRangeDiff) < std::numeric_limits<float>::epsilon()) {
                target_maxs[j] = featureMin + 1.0f;
            }
        }
    }

    // ========================================================================
    // PHASE 3: Also compute global min/max for backward compatibility
    // ========================================================================
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

    // Set scalar convenience bounds from first target dimension
    if (!target_mins.empty()) {
        this->target_min = target_mins[0];
        this->target_max = target_maxs[0];
    } else {
        this->target_min = tempGlobalMin;
        this->target_max = tempGlobalMax;
    }
}

// Normalize input features (x, y) independently per dimension.
// Formula: X_norm = (X - X_min) / (X_max - X_min) * (maxRange - minRange) + minRange
// Memory-safe: Explicitly resizes input_min/input_max before use, with defensive zero-variance handling.
void Training::normalize_inputs(float minRange, float maxRange) {
    if (inputs.empty()) {
        std::cerr << "Warning: No input data to normalize." << std::endl;
        return;
    }

    if (minRange >= maxRange) {
        throw std::invalid_argument("Training::normalize_inputs: minRange must be less than maxRange");
    }

    // STRICT VALIDATION: Ensure all input rows have consistent size
    size_t num_input_features = inputs[0].size();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() != num_input_features) {
            std::cerr << "[FATAL] Inconsistent input row size at index " << i
                      << ": expected " << num_input_features << ", got " << inputs[i].size() << std::endl;
            throw std::runtime_error("Training::normalize_inputs: Inconsistent input dimensions detected");
        }
    }

    float targetRangeDiff = maxRange - minRange;

    // Memory-safe: Explicitly initialize and resize per-feature bounds vectors
    // This ensures input_min and input_max are sized correctly BEFORE the normalization loop
    input_min.clear();
    input_max.clear();
    input_min.resize(num_input_features, 0.0f);
    input_max.resize(num_input_features, 1.0f);

    // Compute per-feature min/max from actual data
    for (size_t j = 0; j < num_input_features; ++j) {
        float featureMin = std::numeric_limits<float>::max();
        float featureMax = std::numeric_limits<float>::lowest();
        bool hasValidData = false;

        for (size_t i = 0; i < inputs.size(); ++i) {
            float val = inputs[i][j];
            if (!std::isnan(val) && !std::isinf(val)) {
                featureMin = std::min(featureMin, val);
                featureMax = std::max(featureMax, val);
                hasValidData = true;
            }
        }

        // Defensive: Ensure valid bounds even for constant features
        if (hasValidData) {
            input_min[j] = featureMin;
            input_max[j] = featureMax;
            float featureRangeDiff = featureMax - featureMin;
            // Defensive: If range is zero (constant feature), expand range to avoid divide-by-zero
            if (std::abs(featureRangeDiff) < std::numeric_limits<float>::epsilon()) {
                input_max[j] = featureMin + 1.0f; // Expand to [min, min+1] range
            }
        }
    }

    // VERIFICATION: Confirm vectors are fully populated before normalization loop
    if (input_min.size() != num_input_features || input_max.size() != num_input_features) {
        std::cerr << "[FATAL] Normalization bounds vectors not fully populated!" << std::endl;
        throw std::runtime_error("Training::normalize_inputs: Bounds vector size mismatch");
    }

    // Perform normalization with divide-by-zero protection
    // Zero variance features map to 0.5 (middle of target range) to preserve signal
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size() && j < num_input_features; ++j) {
            float val = inputs[i][j];
            float featureMin = input_min[j];
            float featureMax = input_max[j];
            float featureRangeDiff = featureMax - featureMin;

            if (std::isinf(val)) {
                inputs[i][j] = maxRange;
            }
            else if (std::isnan(val)) {
                inputs[i][j] = minRange;
            }
            else if (std::abs(featureRangeDiff) < std::numeric_limits<float>::epsilon()) {
                // Safe default: constant features map to middle of target range
                inputs[i][j] = minRange + targetRangeDiff * 0.5f;
            }
            else {
                inputs[i][j] = minRange + ((val - featureMin) * targetRangeDiff) / featureRangeDiff;
            }
        }
    }
}

// Normalize target features (z) independently per dimension.
// Formula: Z_norm = (Z - Z_min) / (Z_max - Z_min) * (maxRange - minRange) + minRange
// This isolates the target scale from input scales, protecting calculus gradients.
// Memory-safe: Explicitly resizes target_mins/target_maxs before use.
// Defensive: Zero variance features map to 0.5 (middle of target range).
void Training::normalize_targets(float minRange, float maxRange) {
    if (targets.empty()) {
        std::cerr << "Warning: No target data to normalize." << std::endl;
        return;
    }

    if (minRange >= maxRange) {
        throw std::invalid_argument("Training::normalize_targets: minRange must be less than maxRange");
    }

    // STRICT VALIDATION: Ensure all target rows have consistent size
    size_t num_target_features = targets[0].size();
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i].size() != num_target_features) {
            std::cerr << "[FATAL] Inconsistent target row size at index " << i
                      << ": expected " << num_target_features << ", got " << targets[i].size() << std::endl;
            throw std::runtime_error("Training::normalize_targets: Inconsistent target dimensions detected");
        }
    }

    float targetRangeDiff = maxRange - minRange;

    // Memory-safe: Explicitly initialize and resize per-feature bounds vectors
    target_mins.clear();
    target_maxs.clear();
    target_mins.resize(num_target_features, 0.0f);
    target_maxs.resize(num_target_features, 1.0f);

    // Compute per-feature min/max from actual data
    for (size_t j = 0; j < num_target_features; ++j) {
        float featureMin = std::numeric_limits<float>::max();
        float featureMax = std::numeric_limits<float>::lowest();
        bool hasValidData = false;

        for (size_t i = 0; i < targets.size(); ++i) {
            float val = targets[i][j];
            if (!std::isnan(val) && !std::isinf(val)) {
                featureMin = std::min(featureMin, val);
                featureMax = std::max(featureMax, val);
                hasValidData = true;
            }
        }

        if (hasValidData) {
            target_mins[j] = featureMin;
            target_maxs[j] = featureMax;
            float featureRangeDiff = featureMax - featureMin;
            // Defensive: Expand zero range to avoid divide-by-zero
            // STRICT CHECK: X_max == X_min (zero variance) -> safe default to 0.5
            if (featureMax == featureMin) {
                target_maxs[j] = featureMin + 1.0f;
            }
            else if (std::abs(featureRangeDiff) < std::numeric_limits<float>::epsilon()) {
                target_maxs[j] = featureMin + 1.0f;
            }
        }
    }

    // VERIFICATION: Confirm vectors are fully populated before normalization loop
    if (target_mins.size() != num_target_features || target_maxs.size() != num_target_features) {
        std::cerr << "[FATAL] Target normalization bounds vectors not fully populated!" << std::endl;
        throw std::runtime_error("Training::normalize_targets: Bounds vector size mismatch");
    }

    // Perform normalization with divide-by-zero protection
    // Zero variance features map to 0.5 (middle of target range)
    for (size_t i = 0; i < targets.size(); ++i) {
        for (size_t j = 0; j < num_target_features; ++j) {
            float val = targets[i][j];
            float featureMin = target_mins[j];
            float featureMax = target_maxs[j];
            float featureRangeDiff = featureMax - featureMin;

            if (std::isinf(val)) {
                targets[i][j] = maxRange;
            }
            else if (std::isnan(val)) {
                targets[i][j] = minRange;
            }
            else if (featureMax == featureMin) {
                // STRICT CHECK: X_max == X_min -> normalized value defaults to 0.5 (middle of target range)
                targets[i][j] = minRange + targetRangeDiff * 0.5f;
            }
            else if (std::abs(featureRangeDiff) < std::numeric_limits<float>::epsilon()) {
                // Safe default: constant features map to middle of target range
                targets[i][j] = minRange + targetRangeDiff * 0.5f;
            }
            else {
                targets[i][j] = minRange + ((val - featureMin) * targetRangeDiff) / featureRangeDiff;
            }
        }
    }
}

// Reconstructs a real-world prediction from a normalized scalar using stored target bounds.
// Formula: Z_real = Z_norm * (Z_max - Z_min) + Z_min
float Training::denormalize_prediction(float z_norm) const {
    float target_range_min = this->min;
    float target_range_max = this->max;
    float target_range_diff = target_range_max - target_range_min;

    // Use scalar convenience bounds, falling back to first vector element
    float y_min = this->target_min;
    float y_max = this->target_max;
    if (!target_mins.empty()) {
        y_min = target_mins[0];
        y_max = target_maxs[0];
    }
    float original_range_diff = y_max - y_min;

    if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon() && std::abs(original_range_diff) > std::numeric_limits<float>::epsilon()) {
        return ((z_norm - target_range_min) / target_range_diff) * original_range_diff + y_min;
    } else {
        return y_min;
    }
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

std::vector<std::vector<float>> Training::predict(const std::vector<std::vector<float>>& inputsToPredict) {
    std::vector<std::vector<float>> predictions;
    
    if (!core) {
        std::cerr << "Error: CoreAI not initialized. Cannot make predictions." << std::endl;
        return predictions;
    }
    
    // Normalize the input data using per-feature bounds computed during training.
    // This is the key fix: each feature is scaled independently using its own min/max,
    // preventing date timestamps from crushing price signal into near-constants.
    std::vector<std::vector<float>> normalizedInputs = normalizeDataIndependent(
        inputsToPredict, input_min, input_max, min, max);
    
    // Get predictions from CoreAI (forward pass with training = false)
    predictions = core->forward(normalizedInputs, false);
    
    // Denormalize predictions back to real-world values using per-target bounds
    return denormalizeMatrix(predictions);
}

bool Training::train(const std::vector<std::vector<float>>& trainingInputs, const std::vector<std::vector<float>>& trainingTargets) {
    try {
        // Set training data
        inputs = trainingInputs;
        targets = trainingTargets;
        
        if (inputs.empty() || targets.empty()) {
            std::cerr << "Error: Training data is empty." << std::endl;
            return false;
        }
        
        // Determine input and output sizes from the data
        inputSize = inputs[0].size();
        outputSize = targets[0].size();
        numSamples = inputs.size();
        
        // Initialize CoreAI if not already initialized
        if (!core) {
            if (verbose) {
                std::cout << "Initializing CoreAI with input size: " << inputSize << ", output size: " << outputSize << std::endl;
            }
            core = std::make_unique<CoreAI>(inputSize, layers, neurons, outputSize, min, max, verbose);
            core->trainer = this;
        }
        
        // Preprocess data
        preprocess(min, max);
        
        // Train the model
        train(learningRate, totalEpochs);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        return false;
    }
}

void Training::preprocess(float minVal, float maxVal) {
    if (inputs.empty() || targets.empty()) {
        std::cerr << "Error: No data to preprocess. Load data from file or database first." << std::endl;
        return;
    }

    // Parameter validation
    if (minVal >= maxVal) {
        throw std::invalid_argument("Training::preprocess: minVal must be less than maxVal");
    }
    if (layers <= 0 || neurons <= 0) {
        throw std::invalid_argument("Training::preprocess: layers and neurons must be positive");
    }
    if (inputSize <= 0 || outputSize <= 0) {
        throw std::invalid_argument("Training::preprocess: inputSize and outputSize must be positive");
    }

    // DEBUG: Print first few inputs and targets before normalization
    if (verbose) {
        std::cout << "[DEBUG PREPROCESS] First 3 inputs before normalization:" << std::endl;
        for (int i = 0; i < std::min(3, (int)inputs.size()); ++i) {
            std::cout << "  Input " << i << ": ";
            for (float val : inputs[i]) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "[DEBUG PREPROCESS] First 3 targets before normalization:" << std::endl;
        for (int i = 0; i < std::min(3, (int)targets.size()); ++i) {
            std::cout << "  Target " << i << ": ";
            for (float val : targets[i]) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    normalize(minVal, maxVal);

    // Always re-initialize CoreAI to ensure dimensions match the current dataset
    // This prevents the "input sample size must match inputSize" error when switching datasets
    if (verbose) {
        std::cout << "[DEBUG] Re-initializing CoreAI with Input Size: " << this->inputSize
                  << ", Output Size: " << this->outputSize << std::endl;
    }
    core = std::make_unique<CoreAI>(static_cast<int>(this->inputSize), layers, neurons, static_cast<int>(this->outputSize), minVal, maxVal, verbose);
    core->trainer = this; // Set the trainer pointer in CoreAI

    core->setInput(inputs);
    core->setTarget(targets);
}

void Training::train(double learningRate, int epochs) {
    if (!core) {
        std::cerr << "Error: CoreAI not initialized. Cannot train model." << std::endl;
        return;
    }

    // Parameter validation
    if (learningRate <= 0.0 || learningRate > 1.0) {
        throw std::invalid_argument("Training::train: learningRate must be between 0 and 1");
    }
    if (epochs <= 0) {
        throw std::invalid_argument("Training::train: epochs must be positive");
    }
    if (inputs.empty() || targets.empty()) {
        throw std::invalid_argument("Training::train: No training data available");
    }

    if (verbose) {
        std::cout << "\nStarting training for " << epochs << " epochs...\n";
    }
    const int barWidth = 70;

    // Early stopping variables - more lenient to allow proper learning
    float previous_loss = std::numeric_limits<float>::max();
    int patience = 100; // Increased to 100 - allow many epochs without improvement
    int patience_counter = 0;
    float min_improvement = 1e-5f; // Very small improvement threshold
    float best_loss = std::numeric_limits<float>::max();
    int epochs_since_best = 0;
    int minimum_epochs = std::max(50, epochs / 2); // Must train for at least 50 epochs or 50% of total

    for (int i = 0; i < epochs; ++i) {
        printProgressBar("Training Progress", i + 1, epochs, barWidth);
        core->forward(inputs);
        core->train(inputs, targets, learningRate, numSamples);

        // Calculate current loss for convergence monitoring
        float current_loss = calculateMSE();
        
        // DEBUG: Check for exactly 0.0 loss (likely numerical issue)
        if (current_loss == 0.0f && verbose) {
            std::cout << "\n[WARNING] Loss is exactly 0.0 at epoch " << (i + 1) << std::endl;
            std::cout << "  This may indicate:" << std::endl;
            std::cout << "    - Numerical underflow (values too small)" << std::endl;
            std::cout << "    - Predictions collapsed to constant value" << std::endl;
            std::cout << "    - Check Core.cpp transform functions for NaN/Inf" << std::endl;
        }
        
        // Track best loss
        if (current_loss < best_loss) {
            best_loss = current_loss;
            epochs_since_best = 0;
        } else {
            epochs_since_best++;
        }
        
        // Verbose logging every 10 epochs
        if (verbose && i % 10 == 0) {
            std::cout << "\nEpoch " << (i + 1) << "/" << epochs
                      << " - MSE: " << current_loss
                      << " (Best: " << best_loss << ")" << std::endl;
        }

        // Early stopping check - only after minimum epochs and if loss improvement is stagnant
        if (i + 1 >= minimum_epochs) {
            float loss_improvement = previous_loss - current_loss;
            if (loss_improvement > min_improvement) {
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience_counter >= patience) {
                    if (verbose) {
                        std::cout << "\n[EARLY STOPPING] Triggered at epoch " << (i + 1) << "/" << epochs << std::endl;
                        std::cout << "  Reason: No improvement for " << patience << " consecutive epochs" << std::endl;
                        std::cout << "  Current Loss: " << current_loss << std::endl;
                        std::cout << "  Best Loss: " << best_loss << std::endl;
                        std::cout << "  Epochs since best: " << epochs_since_best << std::endl;
                        std::cout << "  Loss improvement threshold: " << min_improvement << std::endl;
                        std::cout << "  Minimum epochs enforced: " << minimum_epochs << std::endl;
                    }
                    break;
                }
            }
        }
        previous_loss = current_loss;
    }
    // Only print 100% if we actually finished all epochs
    // if (epochs > 0) {
    //    printProgressBar("Training Progress", epochs, epochs, barWidth);
    // }
    if (verbose) {
        std::cout << "\nTraining Complete!\n";
    }
}

bool Training::startTraining(double learningRate, int epochs) {
    if (trainingActive.load()) {
        return false; // Already training
    }

    if (!core) {
        std::cerr << "Error: CoreAI not initialized. Cannot start training." << std::endl;
        return false;
    }

    // Parameter validation
    if (learningRate <= 0.0 || learningRate > 1.0) {
        throw std::invalid_argument("Training::startTraining: learningRate must be between 0 and 1");
    }
    if (epochs <= 0) {
        throw std::invalid_argument("Training::startTraining: epochs must be positive");
    }
    if (inputs.empty() || targets.empty()) {
        throw std::invalid_argument("Training::startTraining: No training data available");
    }

    // Reset training state
    resetTrainingState();
    trainingActive = true;
    trainingStopRequested = false;
    currentEpoch = 0;
    totalEpochs = epochs;
    trainingStatus = "starting";
    trainingStartTime = std::chrono::steady_clock::now();

    // Start training in a separate thread
    trainingFuture = std::async(std::launch::async, [this, learningRate, epochs]() {
        try {
            trainingStatus = "running";
            const int barWidth = 70;

            // Early stopping variables - more lenient to allow proper learning
            float previous_loss = std::numeric_limits<float>::max();
            int patience = 100; // Increased to 100 - allow many epochs without improvement
            int patience_counter = 0;
            float min_improvement = 1e-5f; // Very small improvement threshold
            float best_loss = std::numeric_limits<float>::max();
            int epochs_since_best = 0;
            int minimum_epochs = std::max(50, epochs / 2); // Must train for at least 50 epochs or 50% of total

            for (int i = 0; i < epochs; ++i) {
                if (trainingStopRequested.load()) {
                    trainingStatus = "stopped";
                    break;
                }

                currentEpoch = i + 1;
                // Update progress bar with current epoch
                printProgressBar("Training Progress", i + 1, epochs, barWidth);
                core->forward(inputs);
                core->train(inputs, targets, learningRate, numSamples);

                // Calculate current loss
                currentLoss = calculateMSE();
                
                // DEBUG: Check for exactly 0.0 loss (likely numerical issue)
                if (currentLoss == 0.0f && verbose) {
                    std::cout << "\n[WARNING] Loss is exactly 0.0 at epoch " << (i + 1) << std::endl;
                    std::cout << "  This may indicate:" << std::endl;
                    std::cout << "    - Numerical underflow (values too small)" << std::endl;
                    std::cout << "    - Predictions collapsed to constant value" << std::endl;
                    std::cout << "    - Check Core.cpp transform functions for NaN/Inf" << std::endl;
                }
                
                // Track best loss
                if (currentLoss < best_loss) {
                    best_loss = currentLoss;
                    epochs_since_best = 0;
                } else {
                    epochs_since_best++;
                }

                // Broadcast progress via WebSocket
                if (webSocketServer) {
                    nlohmann::json progressUpdate = {
                        {"type", "training_progress"},
                        {"epoch", currentEpoch},
                        {"total_epochs", totalEpochs},
                        {"loss", currentLoss},
                        {"best_loss", best_loss},
                        {"percentage", (currentEpoch * 100) / totalEpochs}
                    };
                    webSocketServer->broadcastToGroup("training", progressUpdate.dump());
                }
                
                // Verbose logging every 10 epochs
                if (verbose && i % 10 == 0) {
                    std::cout << "\nEpoch " << (i + 1) << "/" << epochs
                              << " - MSE: " << currentLoss
                              << " (Best: " << best_loss << ")" << std::endl;
                }

                // Early stopping check - only after minimum epochs and if loss improvement is stagnant
                if (i + 1 >= minimum_epochs) {
                    float loss_improvement = previous_loss - currentLoss;
                    if (loss_improvement > min_improvement) {
                        patience_counter = 0;
                    } else {
                        patience_counter++;
                        if (patience_counter >= patience) {
                            trainingStatus = "completed_early_stopping";
                            if (verbose) {
                                std::cout << "\n[EARLY STOPPING] Triggered at epoch " << (i + 1) << "/" << epochs << std::endl;
                                std::cout << "  Reason: No improvement for " << patience << " consecutive epochs" << std::endl;
                                std::cout << "  Current Loss: " << currentLoss << std::endl;
                                std::cout << "  Best Loss: " << best_loss << std::endl;
                                std::cout << "  Epochs since best: " << epochs_since_best << std::endl;
                                std::cout << "  Loss improvement threshold: " << min_improvement << std::endl;
                                std::cout << "  Minimum epochs enforced: " << minimum_epochs << std::endl;
                            }
                            break;
                        }
                    }
                }
                previous_loss = currentLoss;
            }

            if (!trainingStopRequested.load()) {
                // Only show 100% if we actually completed all epochs
                if (trainingStatus != "completed_early_stopping") {
                    trainingStatus = "completed";
                    // Show final progress bar at 100%
                    printProgressBar("Training Progress", totalEpochs, totalEpochs, barWidth);
                } else {
                    // Early stopping - show progress at the epoch where it stopped
                    printProgressBar("Training Progress", currentEpoch, totalEpochs, barWidth);
                }
            }

            trainingActive = false;
        }
        catch (const std::exception& e) {
            trainingStatus = "error: " + std::string(e.what());
            trainingActive = false;
        }
    });

    // Wait a short time to check if training started successfully
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check if the training thread is still active (indicating it started)
    if (!trainingActive.load()) {
        if (verbose) {
            std::cout << "[DEBUG] Training thread failed to start or completed immediately" << std::endl;
        }
        return false;
    }

    return true;
}

bool Training::isTraining() const {
    return trainingActive.load();
}

std::string Training::getTrainingStatus() const {
    if (!trainingActive.load()) {
        return trainingStatus;
    }

    std::stringstream ss;
    ss << "running: epoch " << currentEpoch << "/" << totalEpochs;
    if (currentLoss > 0.0f) {
        ss << ", loss: " << currentLoss;
    }

    auto elapsed = std::chrono::steady_clock::now() - trainingStartTime;
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
    ss << ", elapsed: " << elapsed_seconds << "s";

    return ss.str();
}

bool Training::stopTraining() {
    if (!trainingActive.load()) {
        return false;
    }

    trainingStopRequested = true;

    // Wait for training to stop (with timeout)
    auto start = std::chrono::steady_clock::now();
    while (trainingActive.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        auto elapsed = std::chrono::steady_clock::now() - start;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 30) {
            // Force stop after 30 seconds
            trainingActive = false;
            trainingStatus = "force_stopped";
            break;
        }
    }

    return true;
}

void Training::resetTrainingState() {
    trainingActive = false;
    trainingStopRequested = false;
    trainingStatus = "idle";
    currentEpoch = 0;
    totalEpochs = 0;
    currentLoss = 0.0f;
}

float Training::calculateRMSE() {
    if (!core || inputs.empty() || targets.empty()) {
        std::cerr << "Error: CoreAI not initialized or data not loaded for RMSE calculation." << std::endl;
        return -1.0f;
    }

    core->forward(inputs);
    // Denormalize the predictions using CoreAI's internal denormalization method.
    // CoreAI's denormalizeOutput should scale its results from its internal range ([0,1] or its minVal/maxVal)
    // back to its original data scale using per-target bounds.
    // This part relies on CoreAI's `denormalizeOutput` correctly using `minVal`/`maxVal` passed to its constructor.
    core->denormalizeOutput(); // This function should be fixed in Core.cpp to use core's minVal/maxVal

    auto predictions = core->getResults();

    // Denormalize targets back to real-world values using per-target bounds
    std::vector<std::vector<float>> denormalizedTargets = targets; // Create a copy to denormalize
    denormalizeTargets(denormalizedTargets);

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

float Training::calculateMSE() {
    if (!core || inputs.empty() || targets.empty()) {
        std::cerr << "Error: CoreAI not initialized or data not loaded for MSE calculation." << std::endl;
        return -1.0f;
    }

    core->forward(inputs);
    core->denormalizeOutput();
    auto predictions = core->getResults();

    // Denormalize targets back to real-world values using per-target bounds
    std::vector<std::vector<float>> denormalizedTargets = targets;
    denormalizeTargets(denormalizedTargets);

    float mse = 0.0f;
    int count = 0;

    for (size_t i = 0; i < denormalizedTargets.size(); ++i) {
        if (i < predictions.size() && denormalizedTargets[i].size() == predictions[i].size()) {
            for (size_t j = 0; j < denormalizedTargets[i].size(); ++j) {
                float error = denormalizedTargets[i][j] - predictions[i][j];
                mse += error * error;
                count++;
            }
        }
        else {
            std::cerr << "Warning: Mismatched row size between targets and predictions at row " << i << ". Skipping for MSE." << std::endl;
        }
    }

    if (count == 0) {
        std::cerr << "Error: No valid data points for MSE calculation." << std::endl;
        return -1.0f;
    }

    return mse / count;
}

float Training::calculateAccuracy(float threshold) {
    if (!core || inputs.empty() || targets.empty()) {
        std::cerr << "Error: CoreAI not initialized or data not loaded for accuracy calculation." << std::endl;
        return -1.0f;
    }

    core->forward(inputs);
    core->denormalizeOutput();
    auto predictions = core->getResults();

    // Denormalize targets back to real-world values using per-target bounds
    std::vector<std::vector<float>> denormalizedTargets = targets;
    denormalizeTargets(denormalizedTargets);

    int correct = 0;
    int total = 0;

    for (size_t i = 0; i < denormalizedTargets.size(); ++i) {
        if (i < predictions.size() && denormalizedTargets[i].size() == predictions[i].size()) {
            for (size_t j = 0; j < denormalizedTargets[i].size(); ++j) {
                float actual = denormalizedTargets[i][j];
                float predicted = predictions[i][j];
                if (std::abs(actual - predicted) <= threshold) {
                    correct++;
                }
                total++;
            }
        }
        else {
            std::cerr << "Warning: Mismatched row size between targets and predictions at row " << i << ". Skipping for accuracy." << std::endl;
        }
    }

    if (total == 0) {
        std::cerr << "Error: No valid data points for accuracy calculation." << std::endl;
        return -1.0f;
    }

    return static_cast<float>(correct) / total;
}

CoreAI* Training::getCore() {
    return core.get();
}

Language* Training::getLanguage()
{
    return this->langProc.get();
}

nlohmann::json Training::getNetworkTopology() {
    nlohmann::json topology;
    if (!core) {
        std::cerr << "Error: CoreAI not initialized. Cannot get network topology." << std::endl;
        topology["status"] = "error";
        topology["message"] = "CoreAI not initialized";
        return topology;
    }

    topology["inputSize"] = inputSize;
    topology["outputSize"] = outputSize;
    topology["layers"] = layers;
    topology["neurons"] = neurons;
    topology["learning_rate"] = learningRate;
    topology["num_samples"] = numSamples;
    topology["normalization_range"] = {{"min", min}, {"max", max}};
    topology["status"] = "success";

    // Add weights information
    topology["weights"] = {
        {"input_hidden", core->getWeightsHiddenInput()},
        {"hidden_output", core->getWeightsOutputHidden()}
    };

    return topology;
}

nlohmann::json Training::getNetworkActivity() {
    nlohmann::json activity;
    if (!core) {
        std::cerr << "Error: CoreAI not initialized. Cannot get network activity." << std::endl;
        activity["status"] = "error";
        activity["message"] = "CoreAI not initialized";
        return activity;
    }

    // Get current network state
    activity["input"] = core->getInput();
    activity["hidden"] = core->getHiddenData();
    activity["hidden_output"] = core->getHiddenOutputData();
    activity["output"] = core->getOutput();
    activity["results"] = core->getResults();
    activity["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
    activity["samples_processed"] = numSamples;
    activity["status"] = "active";

    // Add error information if available
    activity["hidden_error"] = core->getHiddenErrorData();

    return activity;
}

#ifdef USE_MYSQL
bool Training::saveModel(int& datasetId) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save model to database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }
    if (!core) {
        std::cerr << "Error: CoreAI not initialized. Cannot save model." << std::endl;
        return false;
    }

    std::cout << "Saving comprehensive AI model state to database (ID: " << datasetId << ")...\n";

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


    try {
        // 1. Save AI model state to ai_model_states table
        dbManager->saveAIModelState(datasetId, inputData, outputData,
            hiddenData, hiddenOutputData, hiddenErrorData,
            weightsHiddenInput, weightsOutputHidden);
        std::cout << "AI model state saved successfully for dataset ID: " << datasetId << std::endl;

        // 2. Save learning settings to learning_settings table
        dbManager->saveLearningSettings(datasetId, learningRate, 1, totalEpochs, 0.0, 0.0, 0.0, "adam", "mse", "relu");
        std::cout << "Learning settings saved successfully for dataset ID: " << datasetId << std::endl;

        // 3. Save model state in JSON format to model_states table
        nlohmann::json modelStateJson = getNetworkTopology();
        modelStateJson["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        modelStateJson["dataset_id"] = datasetId;
        modelStateJson["training_status"] = trainingStatus;
        modelStateJson["current_epoch"] = currentEpoch;
        modelStateJson["current_loss"] = currentLoss;

        // Save to model_states table (assuming sessionId = datasetId for now)
        dbManager->saveModelState(datasetId, modelStateJson);
        std::cout << "Model state (JSON) saved successfully for dataset ID: " << datasetId << std::endl;

        // 4. Save prediction results to prediction_results table (if we have predictions)
        if (!inputs.empty() && !targets.empty()) {
            core->forward(inputs);
            core->denormalizeOutput();
            auto predictions = core->getResults();

            // Denormalize targets for comparison
            std::vector<std::vector<float>> denormalizedTargets = targets;
            std::vector<std::vector<float>> denormalizedInputs = inputs;
            
            denormalizeTargets(denormalizedTargets);
            denormalizeInputs(denormalizedInputs);

            // Save prediction results for each sample
            for (size_t i = 0; i < std::min(predictions.size(), denormalizedTargets.size()); ++i) {
                if (i < inputs.size()) {
                    dbManager->savePredictionResults(datasetId, static_cast<int>(i),
                        denormalizedInputs[i],
                        denormalizedTargets[i],
                        predictions[i]);
                }
            }
            std::cout << "Prediction results saved successfully for dataset ID: " << datasetId << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error saving comprehensive model state: " << e.what() << std::endl;
        return false;
    }
    return true;
}
#endif

#ifdef USE_MYSQL
bool Training::loadModel(const int& datasetId) {
#ifdef USE_MYSQL
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot load model from database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    std::cout << "Loading AI model state from database (ID: " << datasetId << ")...\n";

    int mutableDatasetId = datasetId;
    Database::AIModelState state = dbManager->loadLatestAIModelState(mutableDatasetId);

    if (state.inputData.empty() || state.outputData.empty() || state.weightsHiddenInput.empty() || state.weightsOutputHidden.empty()) {
        std::cerr << "No valid model state found or loaded for dataset ID " << datasetId << std::endl;
        return false;
    }

    this->inputSize = state.inputData[0].size();
    this->outputSize = state.outputData[0].size();
    // Always reinitialize CoreAI with loaded dimensions to ensure consistency
    if (verbose) {
        std::cout << "[DEBUG] Re-initializing CoreAI for model load with Input Size: " << this->inputSize
                  << ", Output Size: " << this->outputSize << std::endl;
    }
    core = std::make_unique<CoreAI>(this->inputSize, layers, neurons, this->outputSize, min, max, verbose);
    core->trainer = this;

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
#else
    std::cerr << "Database functionality not available (USE_MYSQL not defined)." << std::endl;
    return false;
#endif
}

bool Training::loadModel(const std::string& datasetId) {
#ifdef USE_MYSQL
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot load model from database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    std::cout << "Loading AI model state from database (Name: " << datasetId << ")...\n";

    // First, find the dataset ID by name
    // Note: getDatasetIdByName method doesn't exist in Database class
    // We need to query the database directly or add this method to Database class
    // For now, assume datasetId is already an ID if it's numeric, otherwise error
    int actualDatasetId;
    try {
        actualDatasetId = std::stoi(datasetId);
    } catch (const std::exception&) {
        std::cerr << "Dataset name lookup not implemented. Please provide dataset ID instead of name." << std::endl;
        return false;
    }

    // Now load the model using the found ID
    return loadModel(actualDatasetId);
#else
    std::cerr << "Database functionality not available (USE_MYSQL not defined)." << std::endl;
    return false;
#endif
}
#endif

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
    size_t num_target_features = target_mins.size();

    for (const auto& row : normalized_data) {
        std::vector<float> denormalized_row;
        for (size_t j = 0; j < row.size(); ++j) {
            float val = row[j];
            
            // Use per-target bounds if available, otherwise fallback to global bounds
            float y_min, y_max;
            if (j < num_target_features) {
                y_min = target_mins[j];
                y_max = target_maxs[j];
            } else {
                y_min = original_data_global_min;
                y_max = original_data_global_max;
            }
            
            float original_range_diff = y_max - y_min;
            
            // Denormalization formula: Y_real = Y_norm * (Y_max - Y_min) + Y_min
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon() && std::abs(original_range_diff) > std::numeric_limits<float>::epsilon()) {
                denormalized_row.push_back(((val - target_range_min) / target_range_diff) * original_range_diff + y_min);
            } else {
                denormalized_row.push_back(y_min);
            }
        }
        denormalized_copy.push_back(denormalized_row);
    }
    return denormalized_copy;
}

// Helper: Denormalize input features back to original scale using per-feature bounds
void Training::denormalizeInputs(std::vector<std::vector<float>>& input_data) const {
    if (input_data.empty()) {
        return;
    }
    
    float target_range_min = this->min;
    float target_range_max = this->max;
    float target_range_diff = target_range_max - target_range_min;
    size_t num_features = input_min.size();

    for (auto& row : input_data) {
        for (size_t j = 0; j < row.size() && j < num_features; ++j) {
            float val = row[j];
            float x_min = input_min[j];
            float x_max = input_max[j];
            float original_range_diff = x_max - x_min;
            
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon() && std::abs(original_range_diff) > std::numeric_limits<float>::epsilon()) {
                row[j] = ((val - target_range_min) / target_range_diff) * original_range_diff + x_min;
            } else {
                row[j] = x_min;
            }
        }
    }
}

// Helper: Denormalize target features back to original scale using per-target bounds
void Training::denormalizeTargets(std::vector<std::vector<float>>& target_data) const {
    if (target_data.empty()) {
        return;
    }
    
    float target_range_min = this->min;
    float target_range_max = this->max;
    float target_range_diff = target_range_max - target_range_min;
    size_t num_features = target_mins.size();

    for (auto& row : target_data) {
        for (size_t j = 0; j < row.size() && j < num_features; ++j) {
            float val = row[j];
            float y_min = target_mins[j];
            float y_max = target_maxs[j];
            float original_range_diff = y_max - y_min;
            
            if (std::abs(target_range_diff) > std::numeric_limits<float>::epsilon() && std::abs(original_range_diff) > std::numeric_limits<float>::epsilon()) {
                row[j] = ((val - target_range_min) / target_range_diff) * original_range_diff + y_min;
            } else {
                row[j] = y_min;
            }
        }
    }
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

    // Create copies for denormalization using independent per-feature/per-target bounds
    std::vector<std::vector<float>> denormalizedTargets = targets;
    std::vector<std::vector<float>> denormalizedInputs = inputs;
    
    // Denormalize back to real-world values using stored per-feature bounds
    denormalizeInputs(denormalizedInputs);
    denormalizeTargets(denormalizedTargets);

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
    if (verbose) {
        std::cout << "Results saved to: " << filename << std::endl;
    }
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

// Helper method to parse a CSV line respecting quotes
std::vector<std::string> Training::parseCSVLine(const std::string& line, char delimiter) {
    std::vector<std::string> result;
    std::string current;
    bool inQuotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                // Escaped quote
                current += '"';
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c == delimiter && !inQuotes) {
            result.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    result.push_back(current);
    return result;
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
                // Safe default: constant data maps to middle of target range
                normalized_row.push_back(target_min + target_range_diff * 0.5f);
            }
        }
        normalized_copy.push_back(normalized_row);
    }
    return normalized_copy;
}

// Independent feature normalization for inference on new data.
// Each column j is independently scaled from its original bounds
// original_mins[j]..original_maxs[j] into the target range [target_min, target_max].
// This prevents a single feature's scale (e.g., timestamps) from collapsing
// the representation of other features (e.g., prices).
// Defensive: Zero variance features map to 0.5 (middle of target range).
std::vector<std::vector<float>> Training::normalizeDataIndependent(const std::vector<std::vector<float>>& data_to_normalize,
    const std::vector<float>& original_mins,
    const std::vector<float>& original_maxs,
    float target_min, float target_max) const {
    if (data_to_normalize.empty()) {
        return {};
    }

    std::vector<std::vector<float>> normalized_copy;
    float target_range_diff = target_max - target_min;
    size_t num_features = original_mins.size();

    // Defensive: Handle case where original_mins/original_maxs are empty or mismatched
    if (num_features == 0 || num_features != original_maxs.size()) {
        if (verbose) {
            std::cerr << "[NORMALIZE_INDEPENDENT] Warning: Bounds vectors empty or mismatched. Cannot normalize." << std::endl;
        }
        return normalized_copy;
    }

    for (const auto& row : data_to_normalize) {
        std::vector<float> normalized_row;
        for (size_t j = 0; j < row.size() && j < num_features; ++j) {
            float val = row[j];
            float original_range_diff = original_maxs[j] - original_mins[j];

            if (std::isinf(val)) {
                normalized_row.push_back(target_max);
            } else if (std::isnan(val)) {
                normalized_row.push_back(target_min);
            } else if (std::abs(original_range_diff) > std::numeric_limits<float>::epsilon()) {
                normalized_row.push_back(target_min + ((val - original_mins[j]) / original_range_diff) * target_range_diff);
            } else {
                // Safe default: constant features map to middle of target range
                normalized_row.push_back(target_min + target_range_diff * 0.5f);
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

    if (verbose) {
        std::cout << "Loaded " << numSamples << " text samples from " << filename << std::endl;
    }
    return !inputs.empty();
}

// New method for loading JSON files
bool Training::loadJSONFile(const std::string& filename, const std::string& dataPath, int maxSeqLen, int embeddingDim) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening JSON file: " << filename << std::endl;
        return false;
    }

    nlohmann::json jsonData;
    try {
        file >> jsonData;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
        return false;
    }
    file.close();

    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1;

    if (!this->langProc) {
        std::cerr << "Error: Language processor not initialized for JSON file loading." << std::endl;
        return false;
    }

    // Extract text data from JSON using dataPath (e.g., "articles[*].content")
    std::vector<std::string> textSamples;
    try {
        if (dataPath.empty()) {
            // If no path specified, assume the JSON is an array of strings or objects with text
            if (jsonData.is_array()) {
                for (const auto& item : jsonData) {
                    if (item.is_string()) {
                        textSamples.push_back(item);
                    } else if (item.is_object()) {
                        // Try common text fields
                        if (item.contains("text")) textSamples.push_back(item["text"]);
                        else if (item.contains("content")) textSamples.push_back(item["content"]);
                        else if (item.contains("body")) textSamples.push_back(item["body"]);
                        else textSamples.push_back(item.dump()); // Fallback to JSON string
                    }
                }
            } else if (jsonData.is_string()) {
                textSamples.push_back(jsonData);
            }
        } else {
            // Use JSON pointer or simple path parsing
            // For simplicity, assume dataPath is like "articles[*].content"
            auto pointer = nlohmann::json::json_pointer(dataPath);
            auto extracted = jsonData[pointer];
            if (extracted.is_array()) {
                for (const auto& item : extracted) {
                    if (item.is_string()) {
                        textSamples.push_back(item);
                    } else {
                        textSamples.push_back(item.dump());
                    }
                }
            } else if (extracted.is_string()) {
                textSamples.push_back(extracted);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error extracting data from JSON path '" << dataPath << "': " << e.what() << std::endl;
        return false;
    }

    for (const std::string& text : textSamples) {
        std::vector<std::string> words = this->langProc->tokenize(text);
        std::vector<std::vector<float>> sentenceEmbedding;

        for (const std::string& word : words) {
            std::string key = this->langProc->detectLanguage(word);
            auto it = this->langProc->embeddingsByLang.find(key);

            if (it != this->langProc->embeddingsByLang.end()) {
                sentenceEmbedding.push_back(it->second);
            } else {
                sentenceEmbedding.push_back(this->langProc->generateRandomEmbedding());
            }
        }

        // Pad or truncate to maxSeqLen
        if ((int)sentenceEmbedding.size() < maxSeqLen) {
            while ((int)sentenceEmbedding.size() < maxSeqLen) {
                sentenceEmbedding.push_back(std::vector<float>(embeddingDim, 0.0f));
            }
        } else if ((int)sentenceEmbedding.size() > maxSeqLen) {
            sentenceEmbedding.resize(maxSeqLen);
        }

        // Flatten into a single vector
        std::vector<float> flatInput;
        for (const auto& vec : sentenceEmbedding) {
            flatInput.insert(flatInput.end(), vec.begin(), vec.end());
        }

        inputs.push_back(flatInput);
        targets.push_back({ 0.0f }); // Dummy target
    }

    numSamples = inputs.size();

    if (verbose) {
        std::cout << "Loaded " << numSamples << " JSON text samples from " << filename << std::endl;
    }
    return !inputs.empty();
}

// New method for loading XML files
bool Training::loadXMLFile(const std::string& filename, const std::string& xpath, int maxSeqLen, int embeddingDim) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening XML file: " << filename << std::endl;
        return false;
    }

    std::string xmlContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (xmlContent.empty()) {
        std::cerr << "Error: XML file is empty: " << filename << std::endl;
        return false;
    }

    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1;

    if (!this->langProc) {
        std::cerr << "Error: Language processor not initialized for XML file loading." << std::endl;
        return false;
    }

    // Simple XML text extraction (basic implementation)
    // For production, consider using a proper XML library like pugixml
    std::vector<std::string> textSamples;

    if (xpath.empty()) {
        // Extract all text content between tags (very basic)
        std::regex textRegex(">([^<]+)<");
        std::smatch match;
        std::string::const_iterator searchStart(xmlContent.cbegin());
        while (std::regex_search(searchStart, xmlContent.cend(), match, textRegex)) {
            std::string text = match[1].str();
            // Trim whitespace
            text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), text.end());
            if (!text.empty()) {
                textSamples.push_back(text);
            }
            searchStart = match.suffix().first;
        }
    } else {
        // Basic XPath-like extraction (simplified)
        // This is a very basic implementation - for complex XPath, use a proper library
        std::regex xpathRegex("<" + xpath + "[^>]*>([^<]+)</" + xpath + ">");
        std::smatch match;
        std::string::const_iterator searchStart(xmlContent.cbegin());
        while (std::regex_search(searchStart, xmlContent.cend(), match, xpathRegex)) {
            std::string text = match[1].str();
            text.erase(text.begin(), std::find_if(text.begin(), text.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            text.erase(std::find_if(text.rbegin(), text.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), text.end());
            if (!text.empty()) {
                textSamples.push_back(text);
            }
            searchStart = match.suffix().first;
        }
    }

    for (const std::string& text : textSamples) {
        std::vector<std::string> words = this->langProc->tokenize(text);
        std::vector<std::vector<float>> sentenceEmbedding;

        for (const std::string& word : words) {
            std::string key = this->langProc->detectLanguage(word);
            auto it = this->langProc->embeddingsByLang.find(key);

            if (it != this->langProc->embeddingsByLang.end()) {
                sentenceEmbedding.push_back(it->second);
            } else {
                sentenceEmbedding.push_back(this->langProc->generateRandomEmbedding());
            }
        }

        // Pad or truncate to maxSeqLen
        if ((int)sentenceEmbedding.size() < maxSeqLen) {
            while ((int)sentenceEmbedding.size() < maxSeqLen) {
                sentenceEmbedding.push_back(std::vector<float>(embeddingDim, 0.0f));
            }
        } else if ((int)sentenceEmbedding.size() > maxSeqLen) {
            sentenceEmbedding.resize(maxSeqLen);
        }

        // Flatten into a single vector
        std::vector<float> flatInput;
        for (const auto& vec : sentenceEmbedding) {
            flatInput.insert(flatInput.end(), vec.begin(), vec.end());
        }

        inputs.push_back(flatInput);
        targets.push_back({ 0.0f }); // Dummy target
    }

    numSamples = inputs.size();

    if (verbose) {
        std::cout << "Loaded " << numSamples << " XML text samples from " << filename << std::endl;
    }
    return !inputs.empty();
}

// New method for loading document files (.docx, .pdf)
bool Training::loadDocumentFile(const std::string& filename, int maxSeqLen, int embeddingDim) {
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    std::string extractedText;

    try {
        if (extension == "pdf") {
            extractedText = extractTextFromPDF(filename);
        } else if (extension == "docx") {
            extractedText = extractTextFromDOCX(filename);
        } else if (extension == "doc") {
            extractedText = extractTextFromDOC(filename);
        } else if (extension == "txt") {
            extractedText = extractTextFromTXT(filename);
        } else {
            std::cerr << "Error: Unsupported document format: " << extension << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error extracting text from document: " << e.what() << std::endl;
        return false;
    }

    if (extractedText.empty()) {
        std::cerr << "Error: No text extracted from document: " << filename << std::endl;
        return false;
    }

    return processExtractedText(extractedText, maxSeqLen, embeddingDim, filename);
}

std::string Training::extractTextFromPDF(const std::string& filename) {
    // TODO: Replace with proper PDF library integration (e.g., Poppler)
    // For now, using system command as fallback, but this should be replaced
    // with a proper C++ PDF parsing library

    std::string extractedText;
    std::string command = "pdftotext '" + filename + "' - 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe) {
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            extractedText += buffer;
        }
        pclose(pipe);
    }

    // If system command fails, try basic text extraction from raw PDF
    if (extractedText.empty()) {
        extractedText = extractTextFromRawPDF(filename);
    }

    return extractedText;
}

std::string Training::extractTextFromRawPDF(const std::string& filename) {
    // Basic PDF text extraction by reading raw PDF content
    // This is a simplified implementation - production code should use Poppler or similar
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    std::string extractedText;

    // Look for text objects in PDF (BT ... ET blocks)
    size_t pos = 0;
    while ((pos = content.find("BT", pos)) != std::string::npos) {
        size_t endPos = content.find("ET", pos);
        if (endPos != std::string::npos) {
            std::string textBlock = content.substr(pos + 2, endPos - pos - 2);

            // Extract text between parentheses
            size_t textStart = textBlock.find("(");
            while (textStart != std::string::npos) {
                size_t textEnd = textBlock.find(")", textStart);
                if (textEnd != std::string::npos) {
                    std::string text = textBlock.substr(textStart + 1, textEnd - textStart - 1);
                    // Decode PDF text encoding (basic)
                    text = decodePDFText(text);
                    extractedText += text + " ";
                }
                textStart = textBlock.find("(", textEnd);
            }
            extractedText += "\n";
        }
        pos = endPos + 2;
    }

    return extractedText;
}

std::string Training::decodePDFText(const std::string& pdfText) {
    std::string decoded;
    for (size_t i = 0; i < pdfText.size(); ++i) {
        if (pdfText[i] == '\\') {
            if (i + 1 < pdfText.size()) {
                if (pdfText[i + 1] == 'n') {
                    decoded += '\n';
                    ++i;
                } else if (pdfText[i + 1] == 'r') {
                    decoded += '\r';
                    ++i;
                } else if (pdfText[i + 1] == 't') {
                    decoded += '\t';
                    ++i;
                } else if (pdfText[i + 1] == '(' || pdfText[i + 1] == ')') {
                    decoded += pdfText[i + 1];
                    ++i;
                } else {
                    // Octal escape sequence
                    std::string octal;
                    for (int j = 1; j <= 3 && i + j < pdfText.size(); ++j) {
                        if (isdigit(pdfText[i + j])) {
                            octal += pdfText[i + j];
                        } else {
                            break;
                        }
                    }
                    if (!octal.empty()) {
                        int code = std::stoi(octal, nullptr, 8);
                        decoded += static_cast<char>(code);
                        i += octal.size();
                    }
                }
            }
        } else {
            decoded += pdfText[i];
        }
    }
    return decoded;
}

std::string Training::extractTextFromDOCX(const std::string& filename) {
    // TODO: Replace with proper DOCX library integration (e.g., libzip + XML parsing)
    // For now, using system command as fallback

    std::string extractedText;
    std::string command = "unzip -p '" + filename + "' word/document.xml 2>/dev/null | sed 's/<[^>]*>//g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$'";
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe) {
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            extractedText += buffer;
        }
        pclose(pipe);
    }

    // If system command fails, try basic XML parsing
    if (extractedText.empty()) {
        extractedText = extractTextFromRawDOCX(filename);
    }

    return extractedText;
}

std::string Training::extractTextFromRawDOCX(const std::string& filename) {
    // Basic DOCX text extraction by parsing XML content
    // This is a simplified implementation - production code should use proper XML parsing
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    std::string extractedText;

    // Look for text content in DOCX XML
    size_t pos = 0;
    while ((pos = content.find("<w:t", pos)) != std::string::npos) {
        size_t textStart = content.find(">", pos);
        if (textStart != std::string::npos) {
            size_t textEnd = content.find("</w:t>", textStart);
            if (textEnd != std::string::npos) {
                std::string text = content.substr(textStart + 1, textEnd - textStart - 1);
                extractedText += text + " ";
            }
        }
        pos = content.find("</w:t>", pos) + 6;
        if (pos == std::string::npos) break;
    }

    return extractedText;
}

std::string Training::extractTextFromDOC(const std::string& filename) {
    // DOC files are complex binary formats
    // This is a very basic implementation - production code should use antiword or similar
    std::string extractedText;

    std::string command = "antiword '" + filename + "' 2>/dev/null || catdoc '" + filename + "' 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe) {
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            extractedText += buffer;
        }
        pclose(pipe);
    }

    return extractedText;
}

std::string Training::extractTextFromTXT(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return "";
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    return content;
}

bool Training::processExtractedText(const std::string& extractedText, int maxSeqLen, int embeddingDim, const std::string& filename) {
    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1;

    if (!this->langProc) {
        std::cerr << "Error: Language processor not initialized for document file loading." << std::endl;
        return false;
    }

    // Split extracted text into paragraphs or sentences
    std::vector<std::string> textSamples;
    std::istringstream iss(extractedText);
    std::string line;
    while (std::getline(iss, line)) {
        // Clean up the line
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), line.end());
        if (!line.empty() && line.length() > 5) { // Filter out very short lines
            textSamples.push_back(line);
        }
    }

    // Limit number of samples to prevent memory issues
    if (textSamples.size() > 1000) {
        textSamples.resize(1000);
    }

    for (const std::string& text : textSamples) {
        std::vector<std::string> words = this->langProc->tokenize(text);
        std::vector<std::vector<float>> sentenceEmbedding;

        for (const std::string& word : words) {
            std::string key = this->langProc->detectLanguage(word);
            auto it = this->langProc->embeddingsByLang.find(key);

            if (it != this->langProc->embeddingsByLang.end()) {
                sentenceEmbedding.push_back(it->second);
            } else {
                sentenceEmbedding.push_back(this->langProc->generateRandomEmbedding());
            }
        }

        // Pad or truncate to maxSeqLen
        if ((int)sentenceEmbedding.size() < maxSeqLen) {
            while ((int)sentenceEmbedding.size() < maxSeqLen) {
                sentenceEmbedding.push_back(std::vector<float>(embeddingDim, 0.0f));
            }
        } else if ((int)sentenceEmbedding.size() > maxSeqLen) {
            sentenceEmbedding.resize(maxSeqLen);
        }

        // Flatten into a single vector
        std::vector<float> flatInput;
        for (const auto& vec : sentenceEmbedding) {
            flatInput.insert(flatInput.end(), vec.begin(), vec.end());
        }

        inputs.push_back(flatInput);
        targets.push_back({ 0.0f }); // Dummy target for unsupervised learning
    }

    numSamples = inputs.size();

    if (verbose) {
        std::cout << "Loaded " << numSamples << " document text samples from " << filename << std::endl;
    }
    return !inputs.empty();
}

// General method that detects file type and routes to appropriate loader
bool Training::loadFile(const std::string& filename, int maxSeqLen, int embeddingDim) {
    // Input validation
    if (filename.empty()) {
        std::cerr << "Error: Filename cannot be empty." << std::endl;
        return false;
    }
    if (maxSeqLen <= 0) {
        std::cerr << "Error: maxSeqLen must be positive." << std::endl;
        return false;
    }
    if (embeddingDim <= 0) {
        std::cerr << "Error: embeddingDim must be positive." << std::endl;
        return false;
    }
    if (!this->langProc) {
        std::cerr << "Error: Language processor not initialized. Cannot load file." << std::endl;
        return false;
    }

    // Check if file exists
    std::ifstream testFile(filename);
    if (!testFile.is_open()) {
        std::cerr << "Error: Cannot open file: " << filename << " (Current working directory: " << std::filesystem::current_path() << ")" << std::endl;
        return false;
    }
    testFile.close();

    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    bool success = false;
    if (extension == "txt" || extension == "text") {
        success = loadTextFile(filename, maxSeqLen, embeddingDim);
    } else if (extension == "json") {
        success = loadJSONFile(filename, "", maxSeqLen, embeddingDim); // Empty dataPath for auto-detection
    } else if (extension == "xml") {
        success = loadXMLFile(filename, "", maxSeqLen, embeddingDim); // Empty xpath for auto-extraction
    } else if (extension == "pdf" || extension == "docx" || extension == "doc") {
        success = loadDocumentFile(filename, maxSeqLen, embeddingDim);
    } else if (extension == "mp4" || extension == "avi" || extension == "mov" || extension == "mkv") {
        success = loadFromVideoFile(filename, maxSeqLen, embeddingDim, 30);
    } else if (extension == "mp3" || extension == "wav" || extension == "flac" || extension == "aac") {
        success = loadFromAudioFile(filename, maxSeqLen, embeddingDim);
    } else if (extension == "jpg" || extension == "jpeg" || extension == "png" || extension == "bmp" || extension == "gif") {
        success = loadFromImageFile(filename, maxSeqLen, embeddingDim);
    } else if (extension == "csv") {
        // Use existing CSV loader with default parameters
        success = loadCSV(filename, -1, 1, true, true, ',', "file_dataset");
    } else {
        std::cerr << "Error: Unsupported file type: " << extension << ". Supported formats: txt, json, xml, pdf, docx, doc, mp4, avi, mov, mkv, mp3, wav, flac, aac, jpg, jpeg, png, bmp, gif, csv" << std::endl;
        return false;
    }

    // Additional validation after loading
    if (success) {
        if (inputs.empty()) {
            std::cerr << "Error: No data loaded from file." << std::endl;
            return false;
        }
        if (inputs.size() != targets.size()) {
            std::cerr << "Error: Input and target sizes don't match after loading." << std::endl;
            return false;
        }

        // Validate and fix input sizes - ensure all inputs have consistent size
        if (!inputs.empty()) {
            size_t expectedInputSize = inputs[0].size();
            size_t expectedOutputSize = targets[0].size();

            // Fix inconsistent input sizes by padding/truncating
            for (auto& input : inputs) {
                if (input.size() < expectedInputSize) {
                    input.resize(expectedInputSize, 0.0f);
                } else if (input.size() > expectedInputSize) {
                    input.resize(expectedInputSize);
                }
            }

            // Fix inconsistent target sizes
            for (auto& target : targets) {
                if (target.size() < expectedOutputSize) {
                    target.resize(expectedOutputSize, 0.0f);
                } else if (target.size() > expectedOutputSize) {
                    target.resize(expectedOutputSize);
                }
            }

            // Update class members with detected sizes
            this->inputSize = expectedInputSize;
            this->outputSize = expectedOutputSize;
            this->numSamples = inputs.size();

            if (verbose) {
                std::cout << "Dataset loaded successfully. Samples: " << this->numSamples
                          << ", Input Size: " << this->inputSize << ", Output Size: " << this->outputSize << std::endl;
            }
        }
    } else {
        // Enhanced error handling for file format conversion failures
        std::cerr << "Error: Failed to load file '" << filename << "' with extension '" << extension << "'." << std::endl;
        std::cerr << "Possible causes:" << std::endl;
        std::cerr << "  - File format not supported" << std::endl;
        std::cerr << "  - File is corrupted or empty" << std::endl;
        std::cerr << "  - Required external tools/libraries not available (e.g., poppler for PDF, unzip for DOCX)" << std::endl;
        std::cerr << "  - Language processor not properly initialized" << std::endl;
        std::cerr << "  - Insufficient permissions to access the file" << std::endl;
        return false;
    }

    return success;
}

// Batch processing methods for multiple files
bool Training::loadBatchFiles(const std::vector<std::string>& filenames, int maxSeqLen, int embeddingDim) {
    if (filenames.empty()) {
        std::cerr << "Error: No files provided for batch processing." << std::endl;
        return false;
    }

    // Initialize progress tracking
    conversionProgress = 0;
    conversionTotal = filenames.size();
    currentConversionFile = "Batch processing initialization";

    if (verbose) {
        std::cout << "Starting batch processing of " << filenames.size() << " files..." << std::endl;
    }

    std::vector<std::vector<float>> batchInputs;
    std::vector<std::vector<float>> batchTargets;

    int processedCount = 0;
    int failedCount = 0;

    for (size_t i = 0; i < filenames.size(); ++i) {
        const std::string& filename = filenames[i];

        // Update progress tracking
        {
            std::lock_guard<std::mutex> lock(progressMutex);
            currentConversionFile = filename;
            conversionProgress = i;
        }

        if (verbose && filenames.size() > 10) {
            printProgressBar("Processing batch files", i + 1, filenames.size(), 50);
        }

        // Clear current data before loading each file
        inputs.clear();
        targets.clear();

        if (loadFile(filename, maxSeqLen, embeddingDim)) {
            // Append loaded data to batch collections
            batchInputs.insert(batchInputs.end(), inputs.begin(), inputs.end());
            batchTargets.insert(batchTargets.end(), targets.begin(), targets.end());
            processedCount++;

            if (verbose) {
                std::cout << "Successfully processed: " << filename << " (" << inputs.size() << " samples)" << std::endl;
            }
        } else {
            failedCount++;
            if (verbose) {
                std::cerr << "Failed to process: " << filename << std::endl;
            }
        }
    }

    // Final progress update
    {
        std::lock_guard<std::mutex> lock(progressMutex);
        conversionProgress = filenames.size();
        currentConversionFile = "Batch processing completed";
    }

    if (verbose && filenames.size() > 10) {
        printProgressBar("Processing batch files", filenames.size(), filenames.size(), 50);
        std::cout << std::endl;
    }

    // Set the final batch data
    inputs = std::move(batchInputs);
    targets = std::move(batchTargets);

    // Update class members
    if (!inputs.empty()) {
        this->inputSize = inputs[0].size();
        this->outputSize = targets[0].size();
        this->numSamples = inputs.size();

        // Ensure consistency across all samples
        for (auto& input : inputs) {
            input.resize(this->inputSize, 0.0f);
        }
        for (auto& target : targets) {
            target.resize(this->outputSize, 0.0f);
        }
    }

    if (verbose) {
        std::cout << "Batch processing completed. Processed: " << processedCount
                  << ", Failed: " << failedCount << ", Total samples: " << this->numSamples << std::endl;
    }

    return processedCount > 0;
}

bool Training::loadDirectoryFiles(const std::string& directoryPath, const std::vector<std::string>& extensions, int maxSeqLen, int embeddingDim) {
    if (directoryPath.empty()) {
        std::cerr << "Error: Directory path cannot be empty." << std::endl;
        return false;
    }

    if (!std::filesystem::exists(directoryPath) || !std::filesystem::is_directory(directoryPath)) {
        std::cerr << "Error: Directory does not exist or is not a directory: " << directoryPath << std::endl;
        return false;
    }

    std::vector<std::string> matchingFiles;

    // Collect all files with matching extensions
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

            // Remove leading dot from extension
            if (!extension.empty() && extension[0] == '.') {
                extension = extension.substr(1);
            }

            // Check if extension matches any in the provided list
            for (const std::string& allowedExt : extensions) {
                std::string lowerAllowedExt = allowedExt;
                std::transform(lowerAllowedExt.begin(), lowerAllowedExt.end(), lowerAllowedExt.begin(), ::tolower);

                if (extension == lowerAllowedExt) {
                    matchingFiles.push_back(entry.path().string());
                    break;
                }
            }
        }
    }

    if (matchingFiles.empty()) {
        std::cerr << "Error: No files found with specified extensions in directory: " << directoryPath << std::endl;
        return false;
    }

    if (verbose) {
        std::cout << "Found " << matchingFiles.size() << " files to process in directory: " << directoryPath << std::endl;
        std::cout << "Extensions: ";
        for (size_t i = 0; i < extensions.size(); ++i) {
            std::cout << extensions[i];
            if (i < extensions.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }

    return loadBatchFiles(matchingFiles, maxSeqLen, embeddingDim);
}

// Database integration for new file types
#ifdef USE_MYSQL
bool Training::saveFileDataset(int& datasetId, const std::string& datasetName, const std::string& fileType) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save file dataset to database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    try {
        std::string description = "Dataset loaded from " + fileType + " file: " + datasetName;
        datasetId = dbManager->addDataset(datasetName, description, this->numSamples, this->inputSize, this->outputSize);
        currentDatasetId = datasetId;
        currentDatasetName = datasetName;

        if (verbose) {
            std::cout << "File dataset metadata added with ID: " << datasetId << std::endl;
        }

        long long datasetIdLong = static_cast<long long>(datasetId);
        dbManager->clearDatasetRecords(datasetIdLong);

        for (size_t i = 0; i < inputs.size(); ++i) {
            dbManager->addDatasetRecord(datasetId, i, inputs[i], targets[i]);
        }

        if (verbose) {
            std::cout << "All file dataset records added to database for dataset ID: " << datasetId << std::endl;
        }

        // 1. Save learning settings for file-based datasets
        dbManager->saveLearningSettings(datasetId, learningRate, 1, 100, 0.0, 0.0, 0.0, "adam", "mse", "relu");
        if (verbose) {
            std::cout << "Learning settings saved for file dataset ID: " << datasetId << std::endl;
        }

        // 2. Save model state in JSON format for file-based datasets
        nlohmann::json modelStateJson = getNetworkTopology();
        modelStateJson["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        modelStateJson["dataset_id"] = datasetId;
        modelStateJson["file_type"] = fileType;
        modelStateJson["dataset_name"] = datasetName;
        modelStateJson["training_status"] = "file_loaded";
        modelStateJson["current_epoch"] = 0;
        modelStateJson["current_loss"] = 0.0;

        // Save to model_states table
        dbManager->saveModelState(datasetId, modelStateJson);
        if (verbose) {
            std::cout << "Model state (JSON) saved for file dataset ID: " << datasetId << std::endl;
        }

        // 3. Save prediction results for file-based datasets (if we have data)
        if (!inputs.empty() && !targets.empty() && core) {
            core->forward(inputs);
            core->denormalizeOutput();
            auto predictions = core->getResults();

            // Denormalize targets for comparison
            std::vector<std::vector<float>> denormalizedTargets = targets;
            std::vector<std::vector<float>> denormalizedInputs = inputs;
            
            denormalizeTargets(denormalizedTargets);
            denormalizeInputs(denormalizedInputs);

            // Save prediction results for each sample
            for (size_t i = 0; i < std::min(predictions.size(), denormalizedTargets.size()); ++i) {
                if (i < inputs.size()) {
                    dbManager->savePredictionResults(datasetId, static_cast<int>(i),
                        denormalizedInputs[i],
                        denormalizedTargets[i],
                        predictions[i]);
                }
            }
            if (verbose) {
                std::cout << "Prediction results saved for file dataset ID: " << datasetId << std::endl;
            }
        }

        return true;
    } catch (const std::exception& err) {
        std::cerr << "Database error during file dataset saving: " << err.what() << std::endl;
        return false;
    }
}
#endif

// New method for loading plain text files
bool Training::loadTextFile(const std::string& filename, int maxSeqLen, int embeddingDim) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening text file: " << filename << std::endl;
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    if (content.empty()) {
        std::cerr << "Error: Text file is empty: " << filename << std::endl;
        return false;
    }

    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1; // Default output size for text

    if (!this->langProc) {
        std::cerr << "Error: Language processor not initialized for text file loading. Cannot load text data." << std::endl;
        return false;
    }

    // Split content into sentences or paragraphs (simple split by newlines)
    std::vector<std::string> lines;
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }

    for (const std::string& textLine : lines) {
        std::vector<std::string> words = this->langProc->tokenize(textLine);
        std::vector<std::vector<float>> sentenceEmbedding;

        for (const std::string& word : words) {
            std::string key = this->langProc->detectLanguage(word);
            auto it = this->langProc->embeddingsByLang.find(key);

            if (it != this->langProc->embeddingsByLang.end()) {
                sentenceEmbedding.push_back(it->second);
            } else {
                sentenceEmbedding.push_back(this->langProc->generateRandomEmbedding());
            }
        }

        // Pad or truncate to maxSeqLen
        if ((int)sentenceEmbedding.size() < maxSeqLen) {
            while ((int)sentenceEmbedding.size() < maxSeqLen) {
                sentenceEmbedding.push_back(std::vector<float>(embeddingDim, 0.0f));
            }
        } else if ((int)sentenceEmbedding.size() > maxSeqLen) {
            sentenceEmbedding.resize(maxSeqLen);
        }

        // Flatten into a single vector
        std::vector<float> flatInput;
        for (const auto& vec : sentenceEmbedding) {
            flatInput.insert(flatInput.end(), vec.begin(), vec.end());
        }

        inputs.push_back(flatInput);
        targets.push_back({ 0.0f }); // Dummy target
    }

    numSamples = inputs.size();

    if (verbose) {
        std::cout << "Loaded " << numSamples << " text samples from " << filename << std::endl;
    }
    return !inputs.empty();
}

// New methods for learning from web content
bool Training::loadFromWebURL(const std::string& url, int maxSeqLen, int embeddingDim) {
    try {
        // Initialize WebModule if not already done
        static std::unique_ptr<WebModule> webModule;
        if (!webModule) {
            webModule = std::make_unique<WebModule>("WebLearning");
            if (!webModule->initialize()) {
                std::cerr << "Error: Failed to initialize WebModule for web URL loading." << std::endl;
                return false;
            }
        }

        // Validate URL
        if (!webModule->isValidURL(url)) {
            std::cerr << "Error: Invalid URL: " << url << std::endl;
            return false;
        }

        // Fetch web page content
        auto page = webModule->getWebPage(url);
        if (page.content.empty()) {
            std::cerr << "Error: Failed to fetch content from URL: " << url << std::endl;
            return false;
        }

        // Check content safety
        if (!webModule->isSafeContent(page.textContent)) {
            std::cerr << "Error: Content from URL is flagged as unsafe: " << url << std::endl;
            return false;
        }

        inputs.clear();
        targets.clear();
        inputSize = maxSeqLen * embeddingDim;
        outputSize = 1;

        if (!this->langProc) {
            std::cerr << "Error: Language processor not initialized for web URL loading." << std::endl;
            return false;
        }

        // Split content into paragraphs or sentences
        std::vector<std::string> textSamples;
        std::istringstream iss(page.textContent);
        std::string line;
        while (std::getline(iss, line)) {
            // Clean up the line
            line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), line.end());
            if (!line.empty() && line.length() > 10) { // Filter out very short lines
                textSamples.push_back(line);
            }
        }

        for (const std::string& text : textSamples) {
            std::vector<std::string> words = this->langProc->tokenize(text);
            std::vector<std::vector<float>> sentenceEmbedding;

            for (const std::string& word : words) {
                std::string key = this->langProc->detectLanguage(word);
                auto it = this->langProc->embeddingsByLang.find(key);

                if (it != this->langProc->embeddingsByLang.end()) {
                    sentenceEmbedding.push_back(it->second);
                } else {
                    sentenceEmbedding.push_back(this->langProc->generateRandomEmbedding());
                }
            }

            // Pad or truncate to maxSeqLen
            if ((int)sentenceEmbedding.size() < maxSeqLen) {
                while ((int)sentenceEmbedding.size() < maxSeqLen) {
                    sentenceEmbedding.push_back(std::vector<float>(embeddingDim, 0.0f));
                }
            } else if ((int)sentenceEmbedding.size() > maxSeqLen) {
                sentenceEmbedding.resize(maxSeqLen);
            }

            // Flatten into a single vector
            std::vector<float> flatInput;
            for (const auto& vec : sentenceEmbedding) {
                flatInput.insert(flatInput.end(), vec.begin(), vec.end());
            }

            inputs.push_back(flatInput);
            targets.push_back({ 0.0f }); // Dummy target
        }

        numSamples = inputs.size();

        if (verbose) {
            std::cout << "Loaded " << numSamples << " text samples from web URL: " << url << std::endl;
        }
        return !inputs.empty();
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading from web URL: " << e.what() << std::endl;
        return false;
    }
}

bool Training::loadFromWebSearch(const std::string& query, int maxResults, int maxSeqLen, int embeddingDim) {
    try {
        // Initialize WebModule if not already done
        static std::unique_ptr<WebModule> webModule;
        if (!webModule) {
            webModule = std::make_unique<WebModule>("WebSearch");
            if (!webModule->initialize()) {
                std::cerr << "Error: Failed to initialize WebModule for web search loading." << std::endl;
                return false;
            }
        }

        // Perform web search
        auto searchResults = webModule->search(query, maxResults);
        if (searchResults.empty()) {
            std::cerr << "Error: No search results found for query: " << query << std::endl;
            return false;
        }

        inputs.clear();
        targets.clear();
        inputSize = maxSeqLen * embeddingDim;
        outputSize = 1;

        if (!this->langProc) {
            std::cerr << "Error: Language processor not initialized for web search loading." << std::endl;
            return false;
        }

        for (const auto& result : searchResults) {
            // Combine title and description for processing
            std::string combinedText = result.title + " " + result.description;

            std::vector<std::string> words = this->langProc->tokenize(combinedText);
            std::vector<std::vector<float>> sentenceEmbedding;

            for (const std::string& word : words) {
                std::string key = this->langProc->detectLanguage(word);
                auto it = this->langProc->embeddingsByLang.find(key);

                if (it != this->langProc->embeddingsByLang.end()) {
                    sentenceEmbedding.push_back(it->second);
                } else {
                    sentenceEmbedding.push_back(this->langProc->generateRandomEmbedding());
                }
            }

            // Pad or truncate to maxSeqLen
            if ((int)sentenceEmbedding.size() < maxSeqLen) {
                while ((int)sentenceEmbedding.size() < maxSeqLen) {
                    sentenceEmbedding.push_back(std::vector<float>(embeddingDim, 0.0f));
                }
            } else if ((int)sentenceEmbedding.size() > maxSeqLen) {
                sentenceEmbedding.resize(maxSeqLen);
            }

            // Flatten into a single vector
            std::vector<float> flatInput;
            for (const auto& vec : sentenceEmbedding) {
                flatInput.insert(flatInput.end(), vec.begin(), vec.end());
            }

            inputs.push_back(flatInput);
            targets.push_back(std::vector<float>{ static_cast<float>(result.relevanceScore) }); // Use relevance score as target
        }

        numSamples = inputs.size();

        if (verbose) {
            std::cout << "Loaded " << numSamples << " text samples from web search results for query: " << query << std::endl;
        }
        return !inputs.empty();
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading from web search: " << e.what() << std::endl;
        return false;
    }
}

bool Training::trainOnWebContent(const std::string& url, int epochs) {
    if (!loadFromWebURL(url, 50, 50)) { // Default parameters
        std::cerr << "Error: Failed to load web content from URL: " << url << std::endl;
        return false;
    }

    // Normalize and preprocess
    preprocess(-1.0f, 1.0f);

    // Train the model
    train(0.01, epochs); // Default learning rate

    if (verbose) {
        std::cout << "Training completed on web content from: " << url << std::endl;
    }
    return true;
}

bool Training::trainOnWebSearchResults(const std::string& query, int maxResults, int epochs) {
    if (!loadFromWebSearch(query, maxResults, 50, 50)) { // Default parameters
        std::cerr << "Error: Failed to load web search results for query: " << query << std::endl;
        return false;
    }

    // Normalize and preprocess
    preprocess(-1.0f, 1.0f);

    // Train the model
    train(0.01, epochs); // Default learning rate

    if (verbose) {
        std::cout << "Training completed on web search results for query: " << query << std::endl;
    }
    return true;
}
// Video learning methods for Training class
bool Training::loadFromVideoFile(const std::string& videoPath, int maxSeqLen, int embeddingDim, int frameSamplingRate) {
	try {
		inputs.clear();
		targets.clear();
		inputSize = maxSeqLen * embeddingDim;
		outputSize = 1;

		// Extract video features using built-in algorithms
		auto videoFeatures = extractVideoFeaturesBuiltin(videoPath, frameSamplingRate);
		if (videoFeatures.empty()) {
			std::cerr << "Error: Failed to extract features from video: " << videoPath << std::endl;
			return false;
		}

		// Process each frame's features
		for (const auto& frameFeatures : videoFeatures) {
			// Convert features to embeddings (simplified approach)
			std::vector<std::vector<float>> frameEmbedding;
			for (size_t i = 0; i < frameFeatures.size() && i < (size_t)embeddingDim; ++i) {
				frameEmbedding.push_back({frameFeatures[i]});
			}

			// Pad or truncate to embeddingDim
			if ((int)frameEmbedding.size() < embeddingDim) {
				while ((int)frameEmbedding.size() < embeddingDim) {
					frameEmbedding.push_back(std::vector<float>(1, 0.0f));
				}
			} else if ((int)frameEmbedding.size() > embeddingDim) {
				frameEmbedding.resize(embeddingDim);
			}

			// Flatten into a single vector
			std::vector<float> flatInput;
			for (const auto& vec : frameEmbedding) {
				flatInput.insert(flatInput.end(), vec.begin(), vec.end());
			}

			inputs.push_back(flatInput);
			targets.push_back({0.0f}); // Dummy target for unsupervised learning

			// Limit number of frames to prevent memory issues
			if (inputs.size() >= 1000) {
				break;
			}
		}

		numSamples = inputs.size();

		if (verbose) {
			std::cout << "Loaded " << numSamples << " video frames from " << videoPath << std::endl;
		}
		return !inputs.empty();
	}
	catch (const std::exception& e) {
		std::cerr << "Error loading video file: " << e.what() << std::endl;
		return false;
	}
}

std::vector<std::vector<float>> Training::extractVideoFeaturesBuiltin(const std::string& videoPath, int frameSamplingRate) {
    std::vector<std::vector<float>> frameFeatures;

    try {
        // Read video file as binary data
        std::ifstream file(videoPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return frameFeatures;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            return frameFeatures;
        }
        file.close();

        // Basic video feature extraction
        // Extract frame-based features by analyzing compressed video data

        // 1. Basic video statistics
        size_t fileSize = buffer.size();
        float compressionRatio = 0.0f;

        // 2. Frame analysis (simplified - treat video as sequence of compressed frames)
        const size_t frameSizeEstimate = 50 * 1024; // Estimate 50KB per frame
        size_t numFrames = std::max(size_t(1), fileSize / frameSizeEstimate);

        // Sample frames based on sampling rate
        for (size_t frameIdx = 0; frameIdx < numFrames; frameIdx += frameSamplingRate) {
            size_t frameStart = frameIdx * frameSizeEstimate;
            size_t frameEnd = std::min(frameStart + frameSizeEstimate, fileSize);

            if (frameStart >= fileSize) break;

            std::vector<float> features;

            // Extract basic frame features
            size_t frameSize = frameEnd - frameStart;
            features.push_back(static_cast<float>(frameSize)); // Frame size

            // Analyze frame data for basic patterns
            int zeroCount = 0;
            int highBitsCount = 0;
            float entropy = 0.0f;
            std::vector<int> byteHist(256, 0);

            for (size_t i = frameStart; i < frameEnd; ++i) {
                unsigned char byte = static_cast<unsigned char>(buffer[i]);
                byteHist[byte]++;

                if (byte == 0) zeroCount++;
                if (byte >= 128) highBitsCount++;
            }

            // Calculate entropy
            for (int count : byteHist) {
                if (count > 0) {
                    float p = static_cast<float>(count) / frameSize;
                    entropy -= p * std::log2(p);
                }
            }

            features.push_back(static_cast<float>(zeroCount) / frameSize); // Zero ratio
            features.push_back(static_cast<float>(highBitsCount) / frameSize); // High bit ratio
            features.push_back(entropy); // Frame entropy

            // Motion estimation (simplified - compare with previous frame)
            if (!frameFeatures.empty()) {
                const auto& prevFeatures = frameFeatures.back();
                float motionDiff = 0.0f;
                for (size_t i = 0; i < std::min(features.size(), prevFeatures.size()); ++i) {
                    motionDiff += std::abs(features[i] - prevFeatures[i]);
                }
                features.push_back(motionDiff);
            } else {
                features.push_back(0.0f); // No motion for first frame
            }

            // Color/temporal features (simplified)
            // Look for patterns that might indicate scene changes
            float patternScore = 0.0f;
            for (size_t i = frameStart + 1; i < frameEnd; ++i) {
                unsigned char curr = static_cast<unsigned char>(buffer[i]);
                unsigned char prev = static_cast<unsigned char>(buffer[i - 1]);
                patternScore += std::abs(static_cast<int>(curr) - static_cast<int>(prev));
            }
            features.push_back(patternScore / frameSize);

            // Add frame metadata
            features.push_back(static_cast<float>(frameIdx)); // Frame number
            features.push_back(static_cast<float>(frameStart)); // Frame offset in file

            frameFeatures.push_back(features);

            // Limit number of frames to prevent excessive processing
            if (frameFeatures.size() >= 100) {
                break;
            }
        }

        // Add video-level features
        if (!frameFeatures.empty()) {
            // Calculate average features across all frames
            std::vector<float> avgFeatures(frameFeatures[0].size(), 0.0f);
            for (const auto& frame : frameFeatures) {
                for (size_t i = 0; i < frame.size(); ++i) {
                    avgFeatures[i] += frame[i];
                }
            }
            for (float& val : avgFeatures) {
                val /= frameFeatures.size();
            }

            // Add video metadata
            avgFeatures.push_back(static_cast<float>(fileSize)); // Total file size
            avgFeatures.push_back(static_cast<float>(numFrames)); // Estimated frame count
            avgFeatures.push_back(static_cast<float>(frameSamplingRate)); // Sampling rate used

            frameFeatures.push_back(avgFeatures);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in built-in video feature extraction: " << e.what() << std::endl;
        return std::vector<std::vector<float>>();
    }

    return frameFeatures;
}

bool Training::trainOnVideoDataset(const std::string& videoPath, int epochs, int frameSamplingRate) {
    if (!loadFromVideoFile(videoPath, 50, 50, frameSamplingRate)) { // Default parameters
        std::cerr << "Error: Failed to load video dataset from: " << videoPath << std::endl;
        return false;
    }

    // Normalize and preprocess
    preprocess(-1.0f, 1.0f);

    // Train the model
    train(0.01, epochs); // Default learning rate

    if (verbose) {
        std::cout << "Training completed on video dataset from: " << videoPath << std::endl;
    }
    return true;
}

bool Training::analyzeVideoContent(const std::string& videoPath) {
    try {
        // Initialize VisionModule if not already done
        static std::unique_ptr<VisionModule> visionModule;
        if (!visionModule) {
            visionModule = std::make_unique<VisionModule>("VideoAnalysis");
            if (!visionModule->initialize()) {
                std::cerr << "Error: Failed to initialize VisionModule for video analysis." << std::endl;
                return false;
            }
        }

        // Analyze video content
        auto analysis = visionModule->analyzeVideo(videoPath, 30, true); // Sample every 30 frames, extract text

        if (verbose) {
            std::cout << "Video Analysis Results:" << std::endl;
            std::cout << "Video: " << analysis.videoPath << std::endl;
            std::cout << "Duration: " << analysis.duration << " seconds" << std::endl;
            std::cout << "Total Frames: " << analysis.totalFrames << std::endl;
            std::cout << "Processed Frames: " << analysis.frames.size() << std::endl;

            std::cout << "Object Counts:" << std::endl;
            for (const auto& pair : analysis.objectCounts) {
                std::cout << "  " << pair.first << ": " << pair.second << std::endl;
            }

            std::cout << "Scene Changes: " << analysis.sceneChanges.size() << std::endl;
            std::cout << "Extracted Text Segments: " << analysis.extractedText.size() << std::endl;
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error analyzing video content: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::vector<float>> Training::extractVideoFeatures(const std::string& videoPath, int temporalWindow) {
try {
    // Initialize VisionModule if not already done
    static std::unique_ptr<VisionModule> visionModule;
    if (!visionModule) {
        visionModule = std::make_unique<VisionModule>("VideoFeatureExtraction");
        if (!visionModule->initialize()) {
            std::cerr << "Error: Failed to initialize VisionModule for video feature extraction." << std::endl;
            return {};
        }
    }

    return visionModule->extractVideoFeatures(videoPath, temporalWindow);
}
catch (const std::exception& e) {
    std::cerr << "Error extracting video features: " << e.what() << std::endl;
    return {};
}
}

// Audio learning methods for Training class
bool Training::loadFromAudioFile(const std::string& audioPath, int maxSeqLen, int embeddingDim) {
	try {
		inputs.clear();
		targets.clear();
		inputSize = maxSeqLen * embeddingDim;
		outputSize = 1;

		// Extract audio features using built-in algorithms
		auto audioFeatures = extractAudioFeaturesBuiltin(audioPath);
		if (audioFeatures.empty()) {
			std::cerr << "Error: Failed to extract features from audio: " << audioPath << std::endl;
			return false;
		}

		// Split audio features into segments and create embeddings
		size_t segmentSize = audioFeatures.size() / maxSeqLen;
		for (int i = 0; i < maxSeqLen; ++i) {
			size_t startIdx = i * segmentSize;
			size_t endIdx = std::min(startIdx + segmentSize, audioFeatures.size());

			std::vector<float> segment(audioFeatures.begin() + startIdx, audioFeatures.begin() + endIdx);

			// Convert features to embeddings (simplified approach)
			std::vector<std::vector<float>> segmentEmbedding;
			for (size_t j = 0; j < segment.size() && j < (size_t)embeddingDim; ++j) {
				segmentEmbedding.push_back({segment[j]});
			}

			// Pad or truncate to embeddingDim
			if ((int)segmentEmbedding.size() < embeddingDim) {
				while ((int)segmentEmbedding.size() < embeddingDim) {
					segmentEmbedding.push_back(std::vector<float>(1, 0.0f));
				}
			} else if ((int)segmentEmbedding.size() > embeddingDim) {
				segmentEmbedding.resize(embeddingDim);
			}

			// Flatten into a single vector
			std::vector<float> flatInput;
			for (const auto& vec : segmentEmbedding) {
				flatInput.insert(flatInput.end(), vec.begin(), vec.end());
			}

			inputs.push_back(flatInput);
			targets.push_back({0.0f}); // Dummy target for unsupervised learning
		}

		numSamples = inputs.size();

		if (verbose) {
			std::cout << "Loaded " << numSamples << " audio segments from " << audioPath << std::endl;
		}
		return !inputs.empty();
	}
	catch (const std::exception& e) {
		std::cerr << "Error loading audio file: " << e.what() << std::endl;
		return false;
	}
}

std::vector<float> Training::extractAudioFeaturesBuiltin(const std::string& audioPath) {
    std::vector<float> features;

    try {
        // Read audio file as binary data
        std::ifstream file(audioPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return features;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            return features;
        }
        file.close();

        // Basic audio feature extraction
        // Extract amplitude features, frequency features, and temporal features

        // 1. Amplitude features
        float maxAmplitude = 0.0f;
        float minAmplitude = 0.0f;
        float sumAmplitude = 0.0f;
        float sumSquares = 0.0f;

        // 2. Zero crossing rate
        int zeroCrossings = 0;
        float previousSample = 0.0f;

        // 3. Simple frequency domain features (basic FFT-like)
        std::vector<float> frequencyFeatures;

        // Process audio samples (assuming 16-bit PCM)
        size_t sampleCount = buffer.size() / 2; // 16-bit samples
        std::vector<float> samples;

        for (size_t i = 0; i < buffer.size() - 1; i += 2) {
            // Convert bytes to 16-bit signed integer
            int16_t sample = static_cast<int16_t>(buffer[i]) | (static_cast<int16_t>(buffer[i + 1]) << 8);
            float normalizedSample = static_cast<float>(sample) / 32768.0f;

            samples.push_back(normalizedSample);

            // Amplitude statistics
            maxAmplitude = std::max(maxAmplitude, std::abs(normalizedSample));
            minAmplitude = std::min(minAmplitude, normalizedSample);
            sumAmplitude += std::abs(normalizedSample);
            sumSquares += normalizedSample * normalizedSample;

            // Zero crossing rate
            if (i > 0 && ((previousSample >= 0 && normalizedSample < 0) || (previousSample < 0 && normalizedSample >= 0))) {
                zeroCrossings++;
            }
            previousSample = normalizedSample;
        }

        if (samples.empty()) {
            return features;
        }

        // Calculate amplitude features
        float meanAmplitude = sumAmplitude / samples.size();
        float rms = std::sqrt(sumSquares / samples.size());
        float crestFactor = maxAmplitude / rms;

        features.push_back(maxAmplitude);
        features.push_back(minAmplitude);
        features.push_back(meanAmplitude);
        features.push_back(rms);
        features.push_back(crestFactor);

        // Zero crossing rate
        float zcr = static_cast<float>(zeroCrossings) / samples.size();
        features.push_back(zcr);

        // Simple spectral centroid (basic frequency analysis)
        float spectralCentroid = 0.0f;
        float totalEnergy = 0.0f;

        // Very basic DFT-like calculation for low frequencies
        const int fftSize = 256;
        if (samples.size() >= fftSize) {
            for (int k = 0; k < fftSize / 2; ++k) {
                float real = 0.0f, imag = 0.0f;
                for (int n = 0; n < fftSize; ++n) {
                    float angle = -2.0f * M_PI * k * n / fftSize;
                    real += samples[n] * std::cos(angle);
                    imag += samples[n] * std::sin(angle);
                }
                float magnitude = std::sqrt(real * real + imag * imag);
                float frequency = static_cast<float>(k) * 44100.0f / fftSize; // Assume 44.1kHz sample rate

                spectralCentroid += frequency * magnitude;
                totalEnergy += magnitude;

                if (k < 10) { // Store first 10 frequency bin magnitudes
                    features.push_back(magnitude);
                }
            }

            if (totalEnergy > 0) {
                spectralCentroid /= totalEnergy;
                features.push_back(spectralCentroid);
            }
        }

        // Temporal features
        float temporalCentroid = 0.0f;
        float totalTemporalEnergy = 0.0f;
        for (size_t i = 0; i < samples.size(); ++i) {
            float time = static_cast<float>(i) / samples.size();
            float energy = samples[i] * samples[i];
            temporalCentroid += time * energy;
            totalTemporalEnergy += energy;
        }
        if (totalTemporalEnergy > 0) {
            temporalCentroid /= totalTemporalEnergy;
            features.push_back(temporalCentroid);
        }

        // File metadata features
        features.push_back(static_cast<float>(buffer.size())); // File size in bytes
        features.push_back(static_cast<float>(sampleCount)); // Number of samples
        features.push_back(static_cast<float>(buffer.size()) / sampleCount); // Bytes per sample

    } catch (const std::exception& e) {
        std::cerr << "Error in built-in audio feature extraction: " << e.what() << std::endl;
        return std::vector<float>();
    }

    return features;
}

bool Training::loadFromAudioTranscription(const std::string& audioPath, int maxSeqLen, int embeddingDim) {
try {
    // Initialize AudioModule if not already done
    static std::unique_ptr<AudioModule> audioModule;
    if (!audioModule) {
        audioModule = std::make_unique<AudioModule>("AudioTranscription");
        if (!audioModule->initialize()) {
            std::cerr << "Error: Failed to initialize AudioModule for audio transcription." << std::endl;
            return false;
        }
    }

    // Transcribe audio to text
    auto speechResult = audioModule->recognizeSpeech(audioPath);
    if (speechResult.text.empty()) {
        std::cerr << "Error: Failed to transcribe audio from: " << audioPath << std::endl;
        return false;
    }

    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1;

    if (!this->langProc) {
        std::cerr << "Error: Language processor not initialized for audio transcription learning." << std::endl;
        return false;
    }

    // Process transcribed text
    std::vector<std::string> words = this->langProc->tokenize(speechResult.text);
    std::vector<std::vector<float>> sentenceEmbedding;

    for (const std::string& word : words) {
        std::string key = this->langProc->detectLanguage(word);
        auto it = this->langProc->embeddingsByLang.find(key);

        if (it != this->langProc->embeddingsByLang.end()) {
            sentenceEmbedding.push_back(it->second);
        } else {
            sentenceEmbedding.push_back(this->langProc->generateRandomEmbedding());
        }
    }

    // Pad or truncate to maxSeqLen
    if ((int)sentenceEmbedding.size() < maxSeqLen) {
        while ((int)sentenceEmbedding.size() < maxSeqLen) {
            sentenceEmbedding.push_back(std::vector<float>(embeddingDim, 0.0f));
        }
    } else if ((int)sentenceEmbedding.size() > maxSeqLen) {
        sentenceEmbedding.resize(maxSeqLen);
    }

    // Flatten into a single vector
    std::vector<float> flatInput;
    for (const auto& vec : sentenceEmbedding) {
        flatInput.insert(flatInput.end(), vec.begin(), vec.end());
    }

    inputs.push_back(flatInput);
    targets.push_back({speechResult.confidence}); // Use confidence as target

    numSamples = inputs.size();

    if (verbose) {
        std::cout << "Loaded transcription from audio: " << audioPath << std::endl;
        std::cout << "Transcribed text: " << speechResult.text << std::endl;
    }
    return !inputs.empty();
}
catch (const std::exception& e) {
    std::cerr << "Error loading audio transcription: " << e.what() << std::endl;
    return false;
}
}

bool Training::trainOnAudioContent(const std::string& audioPath, int epochs) {
if (!loadFromAudioFile(audioPath, 50, 50)) { // Default parameters
    std::cerr << "Error: Failed to load audio content from: " << audioPath << std::endl;
    return false;
}

// Normalize and preprocess
preprocess(-1.0f, 1.0f);

// Train the model
train(0.01, epochs); // Default learning rate

if (verbose) {
    std::cout << "Training completed on audio content from: " << audioPath << std::endl;
}
return true;
}

bool Training::trainOnAudioFile(const std::string& audioPath, int epochs) {
    if (!loadFromAudioFile(audioPath, 50, 50)) { // Default parameters
        std::cerr << "Error: Failed to load audio file from: " << audioPath << std::endl;
        return false;
    }

    // Normalize and preprocess
    preprocess(-1.0f, 1.0f);

    // Train the model
    train(0.01, epochs); // Default learning rate

    if (verbose) {
        std::cout << "Training completed on audio file from: " << audioPath << std::endl;
    }
    return true;
}

bool Training::analyzeAudioContent(const std::string& audioPath) {
try {
    // Initialize AudioModule if not already done
    static std::unique_ptr<AudioModule> audioModule;
    if (!audioModule) {
        audioModule = std::make_unique<AudioModule>("AudioAnalysis");
        if (!audioModule->initialize()) {
            std::cerr << "Error: Failed to initialize AudioModule for audio analysis." << std::endl;
            return false;
        }
    }

    // Analyze audio content
    auto analysis = audioModule->analyzeAudio(audioPath);

    if (verbose) {
        std::cout << "Audio Analysis Results:" << std::endl;
        std::cout << "Audio: " << audioPath << std::endl;
        std::cout << "Duration: " << analysis.duration << " seconds" << std::endl;
        std::cout << "Average Amplitude: " << analysis.averageAmplitude << std::endl;
        std::cout << "Peak Amplitude: " << analysis.peakAmplitude << std::endl;
        std::cout << "Zero Crossing Rate: " << analysis.zeroCrossingRate << std::endl;
        std::cout << "MFCC Coefficients: " << analysis.mfccCoefficients.size() << std::endl;
        std::cout << "Spectral Centroid: " << analysis.spectralCentroid.size() << std::endl;
    }

    return true;
}
catch (const std::exception& e) {
    std::cerr << "Error analyzing audio content: " << e.what() << std::endl;
    return false;
}
}

std::vector<float> Training::extractAudioFeatures(const std::string& audioPath, int temporalWindow) {
try {
    // Initialize AudioModule if not already done
    static std::unique_ptr<AudioModule> audioModule;
    if (!audioModule) {
        audioModule = std::make_unique<AudioModule>("AudioFeatureExtraction");
        if (!audioModule->initialize()) {
            std::cerr << "Error: Failed to initialize AudioModule for audio feature extraction." << std::endl;
            return {};
        }
    }

    return audioModule->extractFeatures(audioPath);
}
catch (const std::exception& e) {
    std::cerr << "Error extracting audio features: " << e.what() << std::endl;
    return {};
}
}

bool Training::loadAudioDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim) {
// Check if datasetPath is a directory containing audio files
if (std::filesystem::is_directory(datasetPath)) {
    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1;

    for (const auto& entry : std::filesystem::directory_iterator(datasetPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

            // Process audio files
            if (extension == ".wav" || extension == ".mp3" || extension == ".flac") {
                if (loadFromAudioFile(entry.path().string(), maxSeqLen, embeddingDim)) {
                    // Successfully loaded, continue to next file
                }
            }
        }
    }

    numSamples = inputs.size();

    if (verbose) {
        std::cout << "Loaded " << numSamples << " audio files from dataset: " << datasetPath << std::endl;
    }
    return !inputs.empty();
}
else {
    // Treat as single audio file
    return loadFromAudioFile(datasetPath, maxSeqLen, embeddingDim);
}
}

bool Training::trainOnAudioDataset(const std::string& datasetPath, int epochs) {
if (!loadAudioDataset(datasetPath, 50, 50)) { // Default parameters
    std::cerr << "Error: Failed to load audio dataset from: " << datasetPath << std::endl;
    return false;
}

// Normalize and preprocess
preprocess(-1.0f, 1.0f);

// Train the model
train(0.01, epochs); // Default learning rate

if (verbose) {
    std::cout << "Training completed on audio dataset from: " << datasetPath << std::endl;
}
return true;
}

// Image learning methods for Training class
bool Training::loadFromImageFile(const std::string& imagePath, int maxSeqLen, int embeddingDim) {
	try {
		inputs.clear();
		targets.clear();
		inputSize = maxSeqLen * embeddingDim;
		outputSize = 1;

		// Extract image features using built-in algorithms
		auto imageFeatures = extractImageFeaturesBuiltin(imagePath);
		if (imageFeatures.empty()) {
			std::cerr << "Error: Failed to extract features from image: " << imagePath << std::endl;
			return false;
		}

		// Convert features to embeddings (simplified approach)
		std::vector<std::vector<float>> imageEmbedding;
		for (size_t i = 0; i < imageFeatures.size() && i < (size_t)embeddingDim; ++i) {
			imageEmbedding.push_back({imageFeatures[i]});
		}

		// Pad or truncate to maxSeqLen
		if ((int)imageEmbedding.size() < maxSeqLen) {
			while ((int)imageEmbedding.size() < maxSeqLen) {
				imageEmbedding.push_back(std::vector<float>(embeddingDim, 0.0f));
			}
		} else if ((int)imageEmbedding.size() > maxSeqLen) {
			imageEmbedding.resize(maxSeqLen);
		}

		// Flatten into a single vector
		std::vector<float> flatInput;
		for (const auto& vec : imageEmbedding) {
			flatInput.insert(flatInput.end(), vec.begin(), vec.end());
		}

		inputs.push_back(flatInput);
		targets.push_back({0.0f}); // Dummy target for unsupervised learning

		numSamples = inputs.size();

		if (verbose) {
			std::cout << "Loaded image from " << imagePath << std::endl;
		}
		return !inputs.empty();
	}
	catch (const std::exception& e) {
		std::cerr << "Error loading image file: " << e.what() << std::endl;
		return false;
	}
}

std::vector<float> Training::extractImageFeaturesBuiltin(const std::string& imagePath) {
    std::vector<float> features;

    try {
        // Read image file as binary data
        std::ifstream file(imagePath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return features;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            return features;
        }
        file.close();

        // Basic image feature extraction
        // Extract color histograms, edge features, and texture features

        // 1. Color histogram features (RGB)
        std::vector<int> redHist(32, 0);
        std::vector<int> greenHist(32, 0);
        std::vector<int> blueHist(32, 0);

        // 2. Edge detection features (simple gradient)
        std::vector<float> edgeFeatures;

        // 3. Texture features (Gabor-like simple features)
        std::vector<float> textureFeatures;

        // Simple feature extraction from raw bytes
        // This is a basic implementation - production code should use OpenCV or similar
        size_t pixelCount = 0;
        for (size_t i = 0; i < buffer.size() - 2; i += 3) {
            if (i + 2 < buffer.size()) {
                unsigned char r = static_cast<unsigned char>(buffer[i]);
                unsigned char g = static_cast<unsigned char>(buffer[i + 1]);
                unsigned char b = static_cast<unsigned char>(buffer[i + 2]);

                // Color histogram
                redHist[r / 8]++;
                greenHist[g / 8]++;
                blueHist[b / 8]++;

                // Simple edge detection (gradient magnitude)
                if (pixelCount > 0) {
                    float gradR = std::abs(static_cast<float>(r) - static_cast<float>(buffer[i - 3]));
                    float gradG = std::abs(static_cast<float>(g) - static_cast<float>(buffer[i - 2]));
                    float gradB = std::abs(static_cast<float>(b) - static_cast<float>(buffer[i - 1]));
                    float gradient = (gradR + gradG + gradB) / 3.0f;
                    edgeFeatures.push_back(gradient);
                }

                pixelCount++;
            }
        }

        // Normalize histograms and add to features
        for (int val : redHist) {
            features.push_back(static_cast<float>(val) / pixelCount);
        }
        for (int val : greenHist) {
            features.push_back(static_cast<float>(val) / pixelCount);
        }
        for (int val : blueHist) {
            features.push_back(static_cast<float>(val) / pixelCount);
        }

        // Add edge features (take average and some statistics)
        if (!edgeFeatures.empty()) {
            float edgeSum = 0.0f;
            float edgeMax = 0.0f;
            for (float edge : edgeFeatures) {
                edgeSum += edge;
                edgeMax = std::max(edgeMax, edge);
            }
            features.push_back(edgeSum / edgeFeatures.size()); // Average edge strength
            features.push_back(edgeMax); // Maximum edge strength
        }

        // Add basic texture features (simple variance calculation)
        if (pixelCount > 1) {
            float meanR = 0.0f, meanG = 0.0f, meanB = 0.0f;
            for (size_t i = 0; i < buffer.size() - 2; i += 3) {
                meanR += static_cast<float>(static_cast<unsigned char>(buffer[i]));
                meanG += static_cast<float>(static_cast<unsigned char>(buffer[i + 1]));
                meanB += static_cast<float>(static_cast<unsigned char>(buffer[i + 2]));
            }
            meanR /= pixelCount;
            meanG /= pixelCount;
            meanB /= pixelCount;

            float varR = 0.0f, varG = 0.0f, varB = 0.0f;
            for (size_t i = 0; i < buffer.size() - 2; i += 3) {
                float dr = static_cast<float>(static_cast<unsigned char>(buffer[i])) - meanR;
                float dg = static_cast<float>(static_cast<unsigned char>(buffer[i + 1])) - meanG;
                float db = static_cast<float>(static_cast<unsigned char>(buffer[i + 2])) - meanB;
                varR += dr * dr;
                varG += dg * dg;
                varB += db * db;
            }
            varR /= pixelCount;
            varG /= pixelCount;
            varB /= pixelCount;

            features.push_back(std::sqrt(varR)); // Standard deviation R
            features.push_back(std::sqrt(varG)); // Standard deviation G
            features.push_back(std::sqrt(varB)); // Standard deviation B
        }

        // Add image metadata features
        features.push_back(static_cast<float>(buffer.size())); // File size
        features.push_back(static_cast<float>(pixelCount)); // Estimated pixel count

    } catch (const std::exception& e) {
        std::cerr << "Error in built-in image feature extraction: " << e.what() << std::endl;
        return std::vector<float>();
    }

    return features;
}

bool Training::loadFromImageDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim) {
// Check if datasetPath is a directory containing image files
if (std::filesystem::is_directory(datasetPath)) {
    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1;

    for (const auto& entry : std::filesystem::directory_iterator(datasetPath)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

            // Process image files
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
                if (loadFromImageFile(entry.path().string(), maxSeqLen, embeddingDim)) {
                    // Successfully loaded, continue to next file
                }
            }
        }
    }

    numSamples = inputs.size();

    if (verbose) {
        std::cout << "Loaded " << numSamples << " images from dataset: " << datasetPath << std::endl;
    }
    return !inputs.empty();
}
else {
    // Treat as single image file
    return loadFromImageFile(datasetPath, maxSeqLen, embeddingDim);
}
}

bool Training::trainOnImageContent(const std::string& imagePath, int epochs) {
if (!loadFromImageFile(imagePath, 50, 50)) { // Default parameters
    std::cerr << "Error: Failed to load image content from: " << imagePath << std::endl;
    return false;
}

// Normalize and preprocess
preprocess(-1.0f, 1.0f);

// Train the model
train(0.01, epochs); // Default learning rate

if (verbose) {
    std::cout << "Training completed on image content from: " << imagePath << std::endl;
}
return true;
}

bool Training::trainOnImageDataset(const std::string& datasetPath, int epochs) {
if (!loadFromImageDataset(datasetPath, 50, 50)) { // Default parameters
    std::cerr << "Error: Failed to load image dataset from: " << datasetPath << std::endl;
    return false;
}

// Normalize and preprocess
preprocess(-1.0f, 1.0f);

// Train the model
train(0.01, epochs); // Default learning rate

if (verbose) {
    std::cout << "Training completed on image dataset from: " << datasetPath << std::endl;
}
return true;
}

bool Training::analyzeImageContent(const std::string& imagePath) {
try {
    // Initialize VisionModule if not already done
    static std::unique_ptr<VisionModule> visionModule;
    if (!visionModule) {
        visionModule = std::make_unique<VisionModule>("ImageAnalysis");
        if (!visionModule->initialize()) {
            std::cerr << "Error: Failed to initialize VisionModule for image analysis." << std::endl;
            return false;
        }
    }

    // Analyze image content
    auto detections = visionModule->detectObjects(imagePath);
    auto classification = visionModule->classify(imagePath);

    if (verbose) {
        std::cout << "Image Analysis Results:" << std::endl;
        std::cout << "Image: " << imagePath << std::endl;
        std::cout << "Classification: " << classification.className << " (confidence: " << classification.confidence << ")" << std::endl;
        std::cout << "Detections: " << detections.size() << std::endl;
        for (const auto& detection : detections) {
            std::cout << "  - " << detection.className << " (confidence: " << detection.confidence << ")" << std::endl;
        }
    }

    return true;
}
catch (const std::exception& e) {
    std::cerr << "Error analyzing image content: " << e.what() << std::endl;
    return false;
}
}

std::vector<float> Training::extractImageFeatures(const std::string& imagePath) {
	try {
    	// Initialize VisionModule if not already done
    	static std::unique_ptr<VisionModule> visionModule;
    	if (!visionModule) {
        	visionModule = std::make_unique<VisionModule>("ImageFeatureExtraction");
        	if (!visionModule->initialize()) {
            	std::cerr << "Error: Failed to initialize VisionModule for image feature extraction." << std::endl;
            	return {};
        	}
    	}

    	return visionModule->extractFeatures(imagePath);
	}
	catch (const std::exception& e) {
    	std::cerr << "Error extracting image features: " << e.what() << std::endl;
    	return {};
	}
}

// Additional database saving methods for comprehensive data persistence

#ifdef USE_MYSQL
bool Training::saveLearningSettingsDuringTraining(int datasetId, double learningRate, int batchSize, int epochs,
    float dropoutRate, float regularization, float momentum, const std::string& optimizer,
    const std::string& lossFunction, const std::string& activationFunction) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save learning settings to database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    try {
        dbManager->saveLearningSettings(datasetId, learningRate, batchSize, epochs, dropoutRate,
            regularization, momentum, optimizer, lossFunction, activationFunction);
        if (verbose) {
            std::cout << "Learning settings saved during training for dataset ID: " << datasetId << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving learning settings during training: " << e.what() << std::endl;
        return false;
    }
}
#endif

#ifdef USE_MYSQL
bool Training::savePredictionResultsAfterInference(int datasetId, const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& actualOutputs, const std::vector<std::vector<float>>& predictedOutputs) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save prediction results to database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    try {
        for (size_t i = 0; i < std::min({inputs.size(), actualOutputs.size(), predictedOutputs.size()}); ++i) {
            dbManager->savePredictionResults(datasetId, static_cast<int>(i), inputs[i], actualOutputs[i], predictedOutputs[i]);
        }
        if (verbose) {
            std::cout << "Prediction results saved after inference for dataset ID: " << datasetId << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving prediction results after inference: " << e.what() << std::endl;
        return false;
    }
}
#endif

#ifdef USE_MYSQL
bool Training::saveChatSession(int& sessionId, const std::string& sessionName, const std::string& userId,
    const std::string& modelVersion, const std::string& status) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save chat session to database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    try {
        sessionId = dbManager->saveChatSession(sessionName, userId, modelVersion, status);
        if (verbose) {
            std::cout << "Chat session saved with ID: " << sessionId << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving chat session: " << e.what() << std::endl;
        return false;
    }
}
#endif

#ifdef USE_MYSQL
bool Training::saveChatMessage(int sessionId, const std::string& role, const std::string& content,
    const std::string& messageType, float confidence, const std::string& metadata) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save chat message to database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    try {
        dbManager->saveChatMessage(sessionId, role, content, messageType, confidence, metadata);
        if (verbose) {
            std::cout << "Chat message saved for session ID: " << sessionId << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving chat message: " << e.what() << std::endl;
        return false;
    }
}
#endif

#ifdef USE_MYSQL
bool Training::saveModelStateJSON(int datasetId, const nlohmann::json& modelState) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save model state to database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    try {
        dbManager->saveModelState(datasetId, modelState);
        if (verbose) {
            std::cout << "Model state (JSON) saved for dataset ID: " << datasetId << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving model state JSON: " << e.what() << std::endl;
        return false;
    }
}
#endif

// Load data from memory vectors
bool Training::loadCSVFromData(const std::vector<std::vector<float>>& data, const std::vector<std::vector<float>>& targets, const std::string& datasetName) {
    if (data.empty()) {
        std::cerr << "Error: No data to load" << std::endl;
        return false;
    }

    // Set input and output sizes
    inputSize = static_cast<int>(data[0].size());
    outputSize = static_cast<int>(targets[0].size());
    numSamples = static_cast<int>(data.size());

    // Copy data
    inputs = data;
    this->targets = targets;

    // Save dataset name
    currentDatasetName = datasetName;

    return true;
}

bool Training::applySlidingWindows(int windowSize) {
    if (windowSize <= 1) {
        return true;
    }

    if (numSamples < windowSize + 1) {
        std::cerr << "[TRAIN] Not enough samples (" << numSamples
                  << ") for window size " << windowSize << std::endl;
        return false;
    }

    if (targets.empty() || targets[0].empty()) {
        std::cerr << "[TRAIN] No target data available for windowing." << std::endl;
        return false;
    }

    // Step 1: Extract price series from target column (first column)
    std::vector<float> prices;
    prices.reserve(static_cast<size_t>(numSamples));
    for (const auto& t : targets) {
        if (!t.empty()) {
            prices.push_back(t[0]);
        }
    }

    if (prices.empty()) {
        std::cerr << "[TRAIN] No valid price values extracted for windowing." << std::endl;
        return false;
    }

    // Step 2: Memory-safe sliding window allocation
    // Explicitly size and initialize the output matrices BEFORE population
    const size_t newNumSamples = static_cast<size_t>(prices.size() - windowSize);
    
    // Initialize outer vectors to exact size with default-constructed inner vectors
    std::vector<std::vector<float>> newInputs;
    std::vector<std::vector<float>> newTargets;
    newInputs.resize(newNumSamples);
    newTargets.resize(newNumSamples);
    
    // Step 3: Pre-initialize ALL inner vectors to correct size BEFORE data assignment
    // This prevents any accidental out-of-bounds access during population
    for (size_t idx = 0; idx < newNumSamples; ++idx) {
        newInputs[idx].resize(static_cast<size_t>(windowSize));
        newTargets[idx].resize(1);
        // Explicitly initialize all values (defensive: zero-initialize then overwrite)
        std::fill(newInputs[idx].begin(), newInputs[idx].end(), 0.0f);
        std::fill(newTargets[idx].begin(), newTargets[idx].end(), 0.0f);
    }

    // Step 4: Populate sliding window matrices with strict bounds checking
    // Each sample: windowSize historical prices -> 1 future price
    for (size_t i = static_cast<size_t>(windowSize); i < prices.size(); ++i) {
        size_t outIdx = i - windowSize;
        
        // VERIFIED: outIdx is within bounds [0, newNumSamples)
        if (outIdx >= newNumSamples) {
            std::cerr << "[TRAIN] FATAL: outIdx out of bounds: " << outIdx << " >= " << newNumSamples << std::endl;
            return false;
        }
        
        // Fill input window with historical prices - STRICT PASS-BY-REFERENCE via index assignment
        for (int j = 0; j < windowSize; ++j) {
            size_t histIdx = i - windowSize + static_cast<size_t>(j);
            
            // VERIFIED: histIdx is within bounds [0, windowSize) relative to i
            // Therefore histIdx < prices.size() is guaranteed
            if (histIdx >= prices.size() || j >= static_cast<int>(newInputs[outIdx].size())) {
                std::cerr << "[TRAIN] FATAL: Array bounds violation at window index" << std::endl;
                return false;
            }
            
            newInputs[outIdx][static_cast<size_t>(j)] = prices[histIdx];
        }
        
        // Fill target with the next price after the window
        if (i >= prices.size()) {
            std::cerr << "[TRAIN] FATAL: Target index out of bounds" << std::endl;
            return false;
        }
        newTargets[outIdx][0] = prices[i];
    }

    // Step 5: Debug output - print first sliding window after population
    if (verbose && !newInputs.empty() && !newTargets.empty()) {
        std::cout << "[DEBUG SLIDING WINDOW] First window populated (before normalization):" << std::endl;
        std::cout << "  Input:  [";
        for (size_t j = 0; j < newInputs[0].size(); ++j) {
            std::cout << std::fixed << std::setprecision(6) << newInputs[0][j];
            if (j < newInputs[0].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Target: [" << std::fixed << std::setprecision(6) << newTargets[0][0] << "]" << std::endl;
        std::cout << "  Vector sizes - inputs[0].size(): " << newInputs[0].size()
                  << ", targets[0].size(): " << newTargets[0].size() << std::endl;
    }

// Step 6: Commit the sliding window data to member variables
     inputs.swap(newInputs);
     targets.swap(newTargets);
     inputSize = windowSize;
     outputSize = 1;
     numSamples = static_cast<int>(inputs.size());

     // Step 7: Compute normalization bounds for later use in denormalization
     // Note: We do NOT normalize here - preprocess() will handle normalization
     // to avoid double-normalization. Bounds are computed for denormalize_prediction().
     compute_normalization_bounds();

     // Step 8: Debug output - print first sliding window after sliding window creation
     if (verbose && !inputs.empty() && !targets.empty()) {
         std::cout << "[DEBUG SLIDING WINDOW] First window created (raw prices):" << std::endl;
         std::cout << "  Input:  [";
        for (size_t j = 0; j < inputs[0].size(); ++j) {
            std::cout << std::fixed << std::setprecision(6) << inputs[0][j];
            if (j < inputs[0].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Target: [" << std::fixed << std::setprecision(6) << targets[0][0] << "]" << std::endl;
    }

    if (verbose) {
        std::cout << "[TRAIN] Sliding windows created: " << numSamples
                  << " samples, inputSize=" << inputSize
                  << ", outputSize=" << outputSize << std::endl;
        std::cout << "[TRAIN] Recomputed normalization bounds for " << input_min.size() << " input features" << std::endl;
    }

    return true;
}

// Problem complexity detection
Training::ComplexityAnalysis Training::detectProblemComplexity() {
    ComplexityAnalysis analysis;
    
    // Default values
    analysis.complexity = ProblemComplexity::MODERATE;
    analysis.complexity_score = 0.5f;
    analysis.requires_3d_transforms = false;
    analysis.recommended_layers = 3;
    analysis.recommended_neurons = 10;
    analysis.recommended_learning_rate = 0.01;
    analysis.recommended_epochs = 100;

    if (numSamples > 0 && inputSize > 0) {
        // Simple complexity detection based on dataset characteristics
        if (inputSize <= 2 && numSamples <= 100) {
            analysis.complexity = ProblemComplexity::SIMPLE;
            analysis.complexity_score = 0.2f;
            analysis.recommended_layers = 2;
            analysis.recommended_neurons = 5;
            analysis.recommended_learning_rate = 0.1;
            analysis.recommended_epochs = 50;
        } else if (inputSize >= 9 && numSamples >= 500) {
            analysis.complexity = ProblemComplexity::COMPLEX;
            analysis.complexity_score = 0.8f;
            analysis.requires_3d_transforms = true;
            analysis.recommended_layers = 4;
            analysis.recommended_neurons = 20;
            analysis.recommended_learning_rate = 0.001;
            analysis.recommended_epochs = 200;
        }
    }

    return analysis;
}

// Auto-tuning implementation
void Training::applyAutoTuning(const ComplexityAnalysis& analysis) {
    layers = analysis.recommended_layers;
    neurons = analysis.recommended_neurons;
    learningRate = analysis.recommended_learning_rate;
}

#ifdef USE_MYSQL
bool Training::saveComprehensiveTrainingData(int datasetId, const std::string& datasetName) {
    if (isOfflineMode || !dbManager) {
        std::cerr << "Cannot save comprehensive training data to database in offline mode or if database manager is not initialized." << std::endl;
        return false;
    }

    try {
        // 1. Save dataset metadata
        std::string description = "Comprehensive training dataset: " + datasetName;
        int actualDatasetId = dbManager->addDataset(datasetName, description, this->numSamples, this->inputSize, this->outputSize);
        currentDatasetId = actualDatasetId;
        currentDatasetName = datasetName;

        // 2. Save dataset records
        long long datasetIdLong = static_cast<long long>(actualDatasetId);
        dbManager->clearDatasetRecords(datasetIdLong);
        for (size_t i = 0; i < inputs.size(); ++i) {
            dbManager->addDatasetRecord(actualDatasetId, i, inputs[i], targets[i]);
        }

        // 3. Save learning settings
        dbManager->saveLearningSettings(actualDatasetId, learningRate, 1, totalEpochs, 0.0, 0.0, 0.0, "adam", "mse", "relu");

        // 4. Save model state
        nlohmann::json modelStateJson = getNetworkTopology();
        modelStateJson["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        modelStateJson["dataset_id"] = actualDatasetId;
        modelStateJson["training_status"] = trainingStatus;
        modelStateJson["current_epoch"] = currentEpoch;
        modelStateJson["current_loss"] = currentLoss;
        dbManager->saveModelState(actualDatasetId, modelStateJson);

        // 5. Save prediction results if available
        if (!inputs.empty() && !targets.empty() && core) {
            core->forward(inputs);
            core->denormalizeOutput();
            auto predictions = core->getResults();

            std::vector<std::vector<float>> denormalizedTargets = targets;
            std::vector<std::vector<float>> denormalizedInputs = inputs;
            
            denormalizeTargets(denormalizedTargets);
            denormalizeInputs(denormalizedInputs);

            for (size_t i = 0; i < std::min(predictions.size(), denormalizedTargets.size()); ++i) {
                if (i < inputs.size()) {
                    dbManager->savePredictionResults(actualDatasetId, static_cast<int>(i),
                        denormalizedInputs[i], denormalizedTargets[i], predictions[i]);
                }
            }
        }

        if (verbose) {
            std::cout << "Comprehensive training data saved for dataset: " << datasetName << " (ID: " << actualDatasetId << ")" << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving comprehensive training data: " << e.what() << std::endl;
        return false;
    }
}
#endif

bool Training::loadFrom3DModelFile(const std::string& modelPath, int maxSeqLen, int embeddingDim) {
    if (!langProc) {
        std::cerr << "Error: Language processor not initialized for 3D model loading." << std::endl;
        return false;
    }
    
    std::ifstream file(modelPath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open 3D model file: " << modelPath << std::endl;
        return false;
    }
    
    std::string description;
    std::string extension = modelPath.substr(modelPath.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == "obj") {
        description = "3D Model (OBJ): ";
        std::string line;
        int vertexCount = 0, faceCount = 0, normalCount = 0, texCoordCount = 0;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line[0] == 'v' && line[1] == ' ') vertexCount++;
            else if (line[0] == 'f') faceCount++;
            else if (line[0] == 'v' && line[1] == 'n') normalCount++;
            else if (line[0] == 'v' && line[1] == 't') texCoordCount++;
            else if (line[0] == 'm' && line.substr(0, 6) == "mtllib") description += "Material library: " + line.substr(7) + ". ";
            else if (line[0] == 'o') description += "Object: " + line.substr(2) + ". ";
            else if (line[0] == 'g') description += "Group: " + line.substr(2) + ". ";
            if ((vertexCount + faceCount) > maxSeqLen) break;
        }
        description += "Vertices: " + std::to_string(vertexCount) + 
                       ", Faces: " + std::to_string(faceCount) +
                       ", Normals: " + std::to_string(normalCount) +
                       ", TexCoords: " + std::to_string(texCoordCount) + ".";
    } else if (extension == "stl") {
        description = "3D Model (STL): ";
        std::string line;
        int facetCount = 0;
        while (std::getline(file, line)) {
            if (line.find("facet") != std::string::npos) facetCount++;
            else if (line.find("solid") != std::string::npos) description += "Solid: " + line.substr(line.find("solid") + 6) + ". ";
        }
        description += "Facets: " + std::to_string(facetCount) + ".";
    } else if (extension == "gltf" || extension == "glb") {
        description = "3D Model (glTF/GLB): Binary/JSON 3D format. File: " + modelPath;
    } else if (extension == "fbx") {
        description = "3D Model (FBX): Autodesk FBX binary format. File: " + modelPath;
    } else if (extension == "ply") {
        description = "3D Model (PLY): ";
        std::string line;
        while (std::getline(file, line)) {
            if (line.find("element vertex") != std::string::npos) description += line + ". ";
            else if (line.find("element face") != std::string::npos) description += line + ". ";
            else if (line == "end_header") break;
        }
    } else if (extension == "dae") {
        description = "3D Model (COLLADA/DAE): XML-based 3D format. File: " + modelPath;
    } else {
        description = "3D Model (" + extension + "): ";
        char buffer[512];
        file.read(buffer, 512);
        int bytesRead = file.gcount();
        std::string readable;
        for (int i = 0; i < bytesRead; i++) {
            if (std::isprint(static_cast<unsigned char>(buffer[i]))) {
                readable += buffer[i];
            }
        }
        if (readable.length() > 10) {
            description += readable.substr(0, std::min((size_t)200, readable.length()));
        } else {
            description += "Binary 3D model data, " + std::to_string(bytesRead) + " bytes read.";
        }
    }
    
    file.close();
    
    return processExtractedText(description, maxSeqLen, embeddingDim, modelPath);
}

bool Training::trainOn3DModelContent(const std::string& modelPath, int epochs) {
    if (!loadFrom3DModelFile(modelPath, 512, 300)) {
        std::cerr << "Error: Failed to load 3D model: " << modelPath << std::endl;
        return false;
    }
    if (inputs.empty() || targets.empty()) {
        std::cerr << "Error: No training data extracted from 3D model." << std::endl;
        return false;
    }
    train(learningRate, epochs);
    return true;
}

bool Training::analyze3DModelContent(const std::string& modelPath) {
    auto features = extract3DModelFeatures(modelPath);
    if (features.empty()) {
        std::cout << "Could not extract features from 3D model: " << modelPath << std::endl;
        return false;
    }
    std::cout << "3D Model Analysis: " << modelPath << std::endl;
    std::cout << "  Features extracted: " << features.size() << std::endl;
    std::cout << "  Mean feature value: ";
    float sum = 0;
    for (float f : features) sum += f;
    std::cout << (features.empty() ? 0 : sum / features.size()) << std::endl;
    return true;
}

std::vector<float> Training::extract3DModelFeatures(const std::string& modelPath) {
    if (!langProc) return {};
    
    std::ifstream file(modelPath);
    if (!file.is_open()) return {};
    
    std::string text;
    char buffer[1024];
    file.read(buffer, 1024);
    int bytesRead = file.gcount();
    file.close();
    
    for (int i = 0; i < bytesRead; i++) {
        if (std::isprint(static_cast<unsigned char>(buffer[i]))) {
            text += buffer[i];
        }
    }
    
    if (text.empty()) return {};
    
    std::vector<float> embedding = langProc->encodeText(text.substr(0, 500));
    return embedding;
}

std::vector<std::vector<float>> Training::extract3DModelFeaturesBatch(const std::vector<std::string>& modelPaths, int frameSamplingRate) {
    std::vector<std::vector<float>> allFeatures;
    for (const auto& path : modelPaths) {
        auto features = extract3DModelFeatures(path);
        if (!features.empty()) {
            allFeatures.push_back(features);
        }
    }
    return allFeatures;
}

bool Training::loadFrom3DModelDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim) {
    std::vector<std::string> extensions = {"obj", "stl", "ply", "fbx", "gltf", "glb", "dae"};
    return loadDirectoryFiles(datasetPath, extensions, maxSeqLen, embeddingDim);
}

bool Training::trainOn3DModelDataset(const std::string& datasetPath, int epochs) {
    if (!loadFrom3DModelDataset(datasetPath, 512, 300)) {
        std::cerr << "Error: Failed to load 3D model dataset: " << datasetPath << std::endl;
        return false;
    }
    if (inputs.empty() || targets.empty()) {
        std::cerr << "Error: No training data extracted from 3D model dataset." << std::endl;
        return false;
    }
    train(learningRate, epochs);
    return true;
}

bool Training::loadFromMusicFile(const std::string& musicPath, int maxSeqLen, int embeddingDim) {
    if (!langProc) {
        std::cerr << "Error: Language processor not initialized for music loading." << std::endl;
        return false;
    }
    
    std::ifstream file(musicPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open music file: " << musicPath << std::endl;
        return false;
    }
    
    std::string description;
    std::string extension = musicPath.substr(musicPath.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    file.seekg(0, std::ios::end);
    long fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (extension == "mp3") {
        description = "Music (MP3): ";
        char header[10];
        file.read(header, 10);
        if (header[0] == 'I' && header[1] == 'D' && header[2] == '3') {
            int tagSize = (header[6] << 21) | (header[7] << 14) | (header[8] << 7) | header[9];
            description += "ID3v2 tag, size: " + std::to_string(tagSize) + " bytes. ";
            std::vector<char> tagData(tagSize);
            if (tagSize > 0 && tagSize < 100000) {
                file.read(tagData.data(), tagSize);
                std::string tagStr(tagData.begin(), tagData.end());
                size_t titlePos = tagStr.find("TIT2");
                size_t artistPos = tagStr.find("TPE1");
                size_t albumPos = tagStr.find("TALB");
                if (titlePos != std::string::npos) {
                    std::string title;
                    for (size_t i = titlePos + 4; i < tagStr.size() && i < titlePos + 200; i++) {
                        if (std::isprint(static_cast<unsigned char>(tagStr[i]))) title += tagStr[i];
                    }
                    if (!title.empty()) description += "Title: " + title + ". ";
                }
                if (artistPos != std::string::npos) {
                    std::string artist;
                    for (size_t i = artistPos + 4; i < tagStr.size() && i < artistPos + 200; i++) {
                        if (std::isprint(static_cast<unsigned char>(tagStr[i]))) artist += tagStr[i];
                    }
                    if (!artist.empty()) description += "Artist: " + artist + ". ";
                }
                if (albumPos != std::string::npos) {
                    std::string album;
                    for (size_t i = albumPos + 4; i < tagStr.size() && i < albumPos + 200; i++) {
                        if (std::isprint(static_cast<unsigned char>(tagStr[i]))) album += tagStr[i];
                    }
                    if (!album.empty()) description += "Album: " + album + ". ";
                }
            }
        }
        description += "File size: " + std::to_string(fileSize) + " bytes.";
    } else if (extension == "wav") {
        description = "Music (WAV): ";
        char header[44];
        file.read(header, 44);
        if (header[0] == 'R' && header[1] == 'I' && header[2] == 'F' && header[3] == 'F') {
            int sampleRate = *reinterpret_cast<int*>(&header[24]);
            int channels = *reinterpret_cast<short*>(&header[22]);
            int bitsPerSample = *reinterpret_cast<short*>(&header[34]);
            description += "Sample rate: " + std::to_string(sampleRate) + " Hz";
            description += ", Channels: " + std::to_string(channels);
            description += ", Bits/sample: " + std::to_string(bitsPerSample);
        }
        description += ". File size: " + std::to_string(fileSize) + " bytes.";
    } else if (extension == "flac") {
        description = "Music (FLAC): Lossless audio. File size: " + std::to_string(fileSize) + " bytes.";
    } else if (extension == "ogg" || extension == "oga") {
        description = "Music (OGG): Vorbis audio. File size: " + std::to_string(fileSize) + " bytes.";
    } else if (extension == "aac" || extension == "m4a") {
        description = "Music (AAC/M4A): Advanced audio coding. File size: " + std::to_string(fileSize) + " bytes.";
    } else if (extension == "midi" || extension == "mid") {
        description = "Music (MIDI): Musical Instrument Digital Interface. File size: " + std::to_string(fileSize) + " bytes.";
        char midiHeader[14];
        file.read(midiHeader, 14);
        if (midiHeader[0] == 'M' && midiHeader[1] == 'T' && midiHeader[2] == 'h' && midiHeader[3] == 'd') {
            int format = (midiHeader[8] << 8) | midiHeader[9];
            int tracks = (midiHeader[10] << 8) | midiHeader[11];
            int division = (midiHeader[12] << 8) | midiHeader[13];
            description += " Format: " + std::to_string(format);
            description += ", Tracks: " + std::to_string(tracks);
            description += ", Division: " + std::to_string(division) + " ticks/beat.";
        }
    } else {
        description = "Music (" + extension + "): Audio file. Size: " + std::to_string(fileSize) + " bytes.";
    }
    
    file.close();
    
    return processExtractedText(description, maxSeqLen, embeddingDim, musicPath);
}

bool Training::trainOnMusicContent(const std::string& musicPath, int epochs) {
    if (!loadFromMusicFile(musicPath, 512, 300)) {
        std::cerr << "Error: Failed to load music file: " << musicPath << std::endl;
        return false;
    }
    if (inputs.empty() || targets.empty()) {
        std::cerr << "Error: No training data extracted from music file." << std::endl;
        return false;
    }
    train(learningRate, epochs);
    return true;
}

bool Training::analyzeMusicContent(const std::string& musicPath) {
    auto features = extractMusicFeatures(musicPath);
    if (features.empty()) {
        std::cout << "Could not extract features from music: " << musicPath << std::endl;
        return false;
    }
    std::cout << "Music Analysis: " << musicPath << std::endl;
    std::cout << "  Features extracted: " << features.size() << std::endl;
    
    float energy = 0;
    for (float f : features) energy += f * f;
    energy = std::sqrt(energy / features.size());
    std::cout << "  RMS Energy: " << energy << std::endl;
    
    std::string genre = classifyMusicGenre(musicPath);
    if (!genre.empty()) {
        std::cout << "  Classify genre: " << genre << std::endl;
    }
    
    return true;
}

std::vector<float> Training::extractMusicFeatures(const std::string& musicPath, int temporalWindow) {
    if (!langProc) return {};
    
    std::ifstream file(musicPath, std::ios::binary);
    if (!file.is_open()) return {};
    
    std::vector<char> buffer(4096);
    file.read(buffer.data(), buffer.size());
    int bytesRead = file.gcount();
    file.close();
    
    std::string description;
    for (int i = 0; i < bytesRead; i += 64) {
        float sample = 0;
        if (i + sizeof(float) <= bytesRead) {
            std::memcpy(&sample, &buffer[i], sizeof(float));
        }
        description += std::to_string(sample).substr(0, 6) + " ";
        if (description.length() > 2000) break;
    }
    
    if (description.empty()) return {};
    
    std::vector<float> embedding = langProc->encodeText(description.substr(0, 1000));
    return embedding;
}

std::vector<std::vector<float>> Training::extractMusicFeaturesBatch(const std::vector<std::string>& musicPaths, int sampleRate) {
    std::vector<std::vector<float>> allFeatures;
    for (const auto& path : musicPaths) {
        auto features = extractMusicFeatures(path);
        if (!features.empty()) {
            allFeatures.push_back(features);
        }
    }
    return allFeatures;
}

bool Training::loadFromMusicDataset(const std::string& datasetPath, int maxSeqLen, int embeddingDim) {
    std::vector<std::string> extensions = {"mp3", "wav", "flac", "ogg", "m4a", "aac", "midi", "mid"};
    return loadDirectoryFiles(datasetPath, extensions, maxSeqLen, embeddingDim);
}

bool Training::trainOnMusicDataset(const std::string& datasetPath, int epochs) {
    if (!loadFromMusicDataset(datasetPath, 512, 300)) {
        std::cerr << "Error: Failed to load music dataset: " << datasetPath << std::endl;
        return false;
    }
    if (inputs.empty() || targets.empty()) {
        std::cerr << "Error: No training data extracted from music dataset." << std::endl;
        return false;
    }
    train(learningRate, epochs);
    return true;
}

std::string Training::classifyMusicGenre(const std::string& musicPath) {
    auto features = extractMusicFeatures(musicPath);
    if (features.empty() || !core) return "";
    
    auto reshaped = std::vector<std::vector<float>>{features};
    int expectedSize = inputSize * 300;
    if ((int)features.size() < expectedSize) {
        features.resize(expectedSize, 0.0f);
        reshaped = std::vector<std::vector<float>>{features};
    }
    
    auto output = core->forward(reshaped);
    if (output.empty() || output[0].empty()) return "";
    
    std::vector<std::string> genres = {"rock", "pop", "electronic", "classical", "jazz", "hip-hop", "metal", "ambient"};
    int bestIdx = 0;
    float bestVal = output[0][0];
    for (int i = 1; i < (int)output[0].size() && i < (int)genres.size(); i++) {
        if (output[0][i] > bestVal) {
            bestVal = output[0][i];
            bestIdx = i;
        }
    }
    return genres[bestIdx];
}

float Training::detectMusicTempo(const std::string& musicPath) {
    std::ifstream file(musicPath, std::ios::binary);
    if (!file.is_open()) return 0.0f;
    
    file.seekg(0, std::ios::end);
    long fileSize = file.tellg();
    file.close();
    
    std::string ext = musicPath.substr(musicPath.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == "midi" || ext == "mid") {
        file.open(musicPath, std::ios::binary);
        char header[14];
        file.read(header, 14);
        if (header[0] == 'M' && header[1] == 'T') {
            int division = (static_cast<unsigned char>(header[12]) << 8) | static_cast<unsigned char>(header[13]);
            if (division > 0) return 120.0f;
        }
        file.close();
    }
    
    return 120.0f;
}

float Training::detectMusicEnergy(const std::string& musicPath) {
    std::ifstream file(musicPath, std::ios::binary);
    if (!file.is_open()) return 0.0f;
    
    std::vector<short> samples(4096);
    file.read(reinterpret_cast<char*>(samples.data()), samples.size() * sizeof(short));
    int samplesRead = file.gcount() / sizeof(short);
    file.close();
    
    if (samplesRead == 0) return 0.0f;
    
    double sumSquares = 0;
    for (int i = 0; i < samplesRead; i++) {
        double s = static_cast<double>(samples[i]) / 32768.0;
        sumSquares += s * s;
    }
    
    return std::sqrt(sumSquares / samplesRead);
}

#include "Train.hpp"
#include "WebModule.hpp"
#include "VisionModule.hpp"
#include "AudioModule.hpp"
#include <filesystem>
// Constructor for online mode (with database) - only if USE_MYSQL is defined
#ifdef USE_MYSQL
Training::Training(const std::string& dbHost, unsigned int dbPort, const std::string& dbUser,
    const std::string dbPassword, const std::string& dbSchema, int sslDummy, bool createTables, bool verbose)
    : dbManager(std::make_unique<Database>(dbHost, dbPort, dbUser, dbPassword, dbSchema, SSLMode::DISABLED, createTables)),
    isOfflineMode(false), currentDatasetId(-1), numSamples(0), inputSize(0), outputSize(0),
    original_data_global_min(std::numeric_limits<float>::max()), original_data_global_max(std::numeric_limits<float>::lowest()), // Initialize
    layers(0), neurons(0), learningRate(0.0), min(0.0f), max(0.0f), last_known_timestamp(0.0f), // Initialize
    verbose(verbose), // Initialize verbose from parameter
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
Training::Training(bool isOffline, bool verbose)
    : isOfflineMode(isOffline), currentDatasetId(-1), numSamples(0), inputSize(0), outputSize(0),
    original_data_global_min(std::numeric_limits<float>::max()), original_data_global_max(std::numeric_limits<float>::lowest()), // Initialize
    layers(0), neurons(0), learningRate(0.0), min(0.0f), max(0.0f), last_known_timestamp(0.0f), // Initialize
    verbose(verbose), // Initialize verbose from parameter
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
    if (verbose) {
        std::cout << "[DEBUG] Starting CSV load: filename='" << filename << "', numSamplesToLoad=" << numSamplesToLoad
                  << ", outputSizeParam=" << outputSizeParam << ", hasHeader=" << hasHeader
                  << ", containsText=" << containsText << ", delimiter='" << delimitor << "'" << std::endl;
    }


    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[DEBUG] Failed to open input CSV: " << filename << std::endl;
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
                std::cout << "[DEBUG] Parsing cells with delimiter '" << delimitor << "'" << std::endl;
            }
            current_raw_cells = parseCSVLine(line, delimitor);
            if (verbose) {
                for (size_t cell_index = 0; cell_index < current_raw_cells.size(); ++cell_index) {
                    std::cout << "[DEBUG]   Raw cell " << cell_index << " before trim: '" << current_raw_cells[cell_index] << "'" << std::endl;
                    // Trim whitespace from cell
                    current_raw_cells[cell_index].erase(0, current_raw_cells[cell_index].find_first_not_of(" \t\r\n"));
                    current_raw_cells[cell_index].erase(current_raw_cells[cell_index].find_last_not_of(" \t\r\n") + 1);
                    std::cout << "[DEBUG]   Parsed cell " << cell_index << ": '" << current_raw_cells[cell_index] << "'" << std::endl;
                }
                std::cout << "[DEBUG] Total cells parsed: " << current_raw_cells.size() << std::endl;

                std::cout << "[DEBUG] Parsed " << current_raw_cells.size() << " raw cells: ";
                for (size_t i = 0; i < current_raw_cells.size(); ++i) {
                    std::cout << "'" << current_raw_cells[i] << "'";
                    if (i < current_raw_cells.size() - 1) std::cout << ", ";
                }
                std::cout << std::endl;
            } else {
                // Still trim even if not verbose
                for (auto& cell : current_raw_cells) {
                    cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                    cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                }
            }

            if (current_raw_cells.empty()) {
                if (verbose) {
                    std::cout << "[DEBUG] Skipping empty row at line " << lineCounter << " (no cells after parsing)" << std::endl;
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
                if ((cell.find("/") != std::string::npos || cell.find("-") != std::string::npos) && cell.length() >= 8) {
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
                    // If it contains text, encode it. Assuming entire column or specific column is text.
                    // For simplicity, let's assume the current cell is text if not numeric/date.
                    if (this->langProc) {
                        std::vector<float> encoded_vec = this->langProc->encodeText(cell);
                        current_processed_values.insert(current_processed_values.end(), encoded_vec.begin(), encoded_vec.end());
                        value_processed = true;
                        if (verbose) {
                            std::cout << "text encoded (" << encoded_vec.size() << " values)";
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
                    std::cout << "[DEBUG] First data row - determining sizes. Processed values: " << current_processed_values.size()
                            << ", outputSizeParam: " << outputSizeParam << std::endl;
                }

                if (outputSizeParam > 0 && current_processed_values.size() < (size_t)outputSizeParam) {
                    std::cerr << "[DEBUG] ERROR: First data row has too few columns (" << current_processed_values.size() << ") for specified outputSize ("
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
                        std::cout << "[DEBUG] Applied heuristic: outputSize=1, inputSize=" << this->inputSize << std::endl;
                    }
                }
                else if (this->outputSize == 0 && current_processed_values.size() == 1) { // Only one column, assume inputSize 1, outputSize 0 or 1
                    this->outputSize = 0; // No target if only one column and not specified
                    this->inputSize = 1;
                    if (verbose) {
                        std::cout << "[DEBUG] Single column: outputSize=0, inputSize=1" << std::endl;
                    }
                }
                firstDataRow = false;
                if (verbose) {
                    std::cout << "[DEBUG] Determined Input Size: " << this->inputSize << ", Output Size: " << this->outputSize << std::endl;
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
        std::cerr << "[DEBUG] Error: No valid data rows found in CSV or all rows were skipped due to errors." << std::endl;
        return false;
    }

    if (verbose) {
        std::cout << "[DEBUG] Final validation - Total rows processed: " << inputs.size() << std::endl;
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
        std::cout << "[DEBUG] Successfully loaded CSV. Samples: " << this->numSamples
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
                dbManager->addDatasetRecord(currentDatasetId, i, inputs[i], targets[i]);
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
    if (inputs.empty()) {
        std::cerr << "Warning: No input data to normalize." << std::endl;
        return;
    }

    // Parameter validation
    if (minRange >= maxRange) {
        throw std::invalid_argument("Training::normalize: minRange must be less than maxRange");
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

    // DEBUG: Log normalization parameters
    if (verbose) {
        std::cout << "[DEBUG NORMALIZE] original_data_global_min: " << this->original_data_global_min
                  << ", original_data_global_max: " << this->original_data_global_max
                  << ", range: " << (this->original_data_global_max - this->original_data_global_min)
                  << ", data_found_for_scaling: " << data_found_for_scaling << std::endl;
    }

    if (!data_found_for_scaling || std::abs(tempGlobalMax - tempGlobalMin) < std::numeric_limits<float>::epsilon()) {
        if (verbose) {
            std::cerr << "[DEBUG NORMALIZE] CRITICAL: Data range is zero or no valid data found. This will cause zero predictions!" << std::endl;
            std::cerr << "Warning: Data is constant or contains only NaN/Inf. Mapping all values to target range based on NaN/Inf rules." << std::endl;
        }
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

    if (verbose) {
        std::cout << "\nNormalizing Data...\n";
    }
    float dataRangeDiff = tempGlobalMax - tempGlobalMin;
    float targetRangeDiff = maxRange - minRange;

    // DEBUG: Log normalization formula parameters
    if (verbose) {
        std::cout << "[DEBUG NORMALIZE] dataRangeDiff: " << dataRangeDiff
                  << ", targetRangeDiff: " << targetRangeDiff
                  << ", minRange: " << minRange << ", maxRange: " << maxRange << std::endl;
    }

    for (size_t i = 0; i < inputs.size(); ++i) { // Use size_t
        for (size_t j = 0; j < inputs[i].size(); ++j) { // Use size_t
            float val = inputs[i][j];
            float original_val = val;
            if (std::isinf(val)) {
                inputs[i][j] = maxRange;
            }
            else if (std::isnan(val)) {
                inputs[i][j] = minRange;
            }
            else {
                inputs[i][j] = minRange + ((val - tempGlobalMin) * targetRangeDiff) / dataRangeDiff;
            }
            // DEBUG: Log first few normalizations
            if (verbose && i < 3 && j < 3) {
                std::cout << "[DEBUG NORMALIZE] input[" << i << "][" << j << "] "
                          << original_val << " -> " << inputs[i][j] << std::endl;
            }
        }

        if (i < targets.size()) {
            for (size_t j = 0; j < targets[i].size(); ++j) { // Use size_t
                float val = targets[i][j];
                float original_val = val;
                if (std::isinf(val)) {
                    targets[i][j] = maxRange;
                }
                else if (std::isnan(val)) {
                    targets[i][j] = minRange;
                }
                else {
                    targets[i][j] = minRange + ((val - tempGlobalMin) * targetRangeDiff) / dataRangeDiff;
                }
                // DEBUG: Log first few target normalizations
                if (verbose && i < 3 && j < 3) {
                    std::cout << "[DEBUG NORMALIZE] target[" << i << "][" << j << "] "
                              << original_val << " -> " << targets[i][j] << std::endl;
                }
            }
        }
        current_sample++;
        printProgressBar("Normalizing Data", current_sample, total_samples, barWidth);
    }
    if (total_samples > 0) {
        printProgressBar("Normalizing Data", total_samples, total_samples, barWidth);
    }
    if (verbose) {
        std::cout << "\nData Normalized!\n";
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

    normalize(minVal, maxVal);

    if (!core) { // Initialize CoreAI only if it hasn't been already
        core = std::make_unique<CoreAI>(this->inputSize, layers, neurons, this->outputSize, minVal, maxVal, verbose);
        core->trainer = this; // Set the trainer pointer in CoreAI
    }
    else {
        // If core is already initialized, update its parameters if necessary
        // core->updateParameters(this->inputSize, layers, neurons, this->outputSize, minVal, maxVal); // Example update method
    }

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

    // Early stopping variables
    float previous_loss = std::numeric_limits<float>::max();
    int patience = 10; // Stop if no improvement for 10 epochs
    int patience_counter = 0;
    float min_improvement = 1e-6f; // Minimum improvement threshold

    for (int i = 0; i < epochs; ++i) {
        printProgressBar("Training Progress", i + 1, epochs, barWidth);
        core->forward(inputs);
        core->train(inputs, targets, learningRate, numSamples);

        // Calculate current loss for convergence monitoring
        float current_loss = calculateMSE();
        if (verbose && i % 10 == 0) { // Log every 10 epochs
            std::cout << "Epoch " << i << " - MSE: " << current_loss << std::endl;
        }

        // Early stopping check
        if (previous_loss - current_loss > min_improvement) {
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= patience) {
                if (verbose) {
                    std::cout << "\nEarly stopping at epoch " << i << " due to lack of improvement." << std::endl;
                }
                break;
            }
        }
        previous_loss = current_loss;
    }
    if (epochs > 0) {
        printProgressBar("Training Progress", epochs, epochs, barWidth);
    }
    if (verbose) {
        std::cout << "\nTraining Complete!\n";
    }
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

float Training::calculateMSE() {
    if (!core || inputs.empty() || targets.empty()) {
        std::cerr << "Error: CoreAI not initialized or data not loaded for MSE calculation." << std::endl;
        return -1.0f;
    }

    core->forward(inputs);
    core->denormalizeOutput();
    auto predictions = core->getResults();

    // Denormalize targets using the original data's global min/max stored in Training class
    std::vector<std::vector<float>> denormalizedTargets = targets;
    float target_range_min_val = this->min;
    float target_range_max_val = this->max;
    float target_range_diff = target_range_max_val - target_range_min_val;
    float original_data_range_diff = original_data_global_max - original_data_global_min;

    // Denormalize target values from the [min, max] range back to the original global min/max
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

    // Denormalize targets using the original data's global min/max stored in Training class
    std::vector<std::vector<float>> denormalizedTargets = targets;
    float target_range_min_val = this->min;
    float target_range_max_val = this->max;
    float target_range_diff = target_range_max_val - target_range_min_val;
    float original_data_range_diff = original_data_global_max - original_data_global_min;

    // Denormalize target values from the [min, max] range back to the original global min/max
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
    // Reinitialize CoreAI with loaded dimensions if not already
    if (!core) {
        core = std::make_unique<CoreAI>(this->inputSize, layers, neurons, this->outputSize, min, max, verbose);
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
    // For document files, we need external tools or libraries
    // This is a placeholder implementation that attempts basic text extraction
    // For production use, integrate with libraries like poppler (PDF) or libzip (DOCX)

    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    std::string extractedText;

    if (extension == "pdf") {
        // Basic PDF text extraction using pdftotext command (requires poppler-utils)
        std::string command = "pdftotext '" + filename + "' -";
        FILE* pipe = popen(command.c_str(), "r");
        if (pipe) {
            char buffer[128];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                extractedText += buffer;
            }
            pclose(pipe);
        } else {
            std::cerr << "Error: Could not extract text from PDF. Ensure pdftotext is installed." << std::endl;
            return false;
        }
    } else if (extension == "docx") {
        // Basic DOCX text extraction using unzip and grep (very basic)
        std::string command = "unzip -p '" + filename + "' word/document.xml | grep -o '<w:t[^>]*>[^<]*</w:t>' | sed 's/<[^>]*>//g'";
        FILE* pipe = popen(command.c_str(), "r");
        if (pipe) {
            char buffer[128];
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                extractedText += buffer;
            }
            pclose(pipe);
        } else {
            std::cerr << "Error: Could not extract text from DOCX. Ensure unzip is installed." << std::endl;
            return false;
        }
    } else {
        std::cerr << "Error: Unsupported document format: " << extension << std::endl;
        return false;
    }

    if (extractedText.empty()) {
        std::cerr << "Error: No text extracted from document: " << filename << std::endl;
        return false;
    }

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
        if (!line.empty()) {
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
        std::cerr << "Error: Cannot open file: " << filename << std::endl;
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
    } else if (extension == "pdf" || extension == "docx") {
        success = loadDocumentFile(filename, maxSeqLen, embeddingDim);
    } else if (extension == "csv") {
        // Use existing CSV loader with default parameters
        success = loadCSV(filename, -1, 1, true, true, ',', "file_dataset");
    } else {
        std::cerr << "Error: Unsupported file type: " << extension << std::endl;
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
        // Validate input sizes
        for (const auto& input : inputs) {
            if (input.size() != (size_t)this->inputSize) {
                std::cerr << "Error: Inconsistent input size after loading." << std::endl;
                return false;
            }
        }
        for (const auto& target : targets) {
            if (target.size() != (size_t)this->outputSize) {
                std::cerr << "Error: Inconsistent target size after loading." << std::endl;
                return false;
            }
        }
    }

    return success;
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
        // Initialize VisionModule if not already done
        static std::unique_ptr<VisionModule> visionModule;
        if (!visionModule) {
            visionModule = std::make_unique<VisionModule>("VideoLearning");
            if (!visionModule->initialize()) {
                std::cerr << "Error: Failed to initialize VisionModule for video learning." << std::endl;
                return false;
            }
        }

        // Check if video format is supported
        if (!visionModule->isVideoFormatSupported(videoPath)) {
            std::cerr << "Error: Video format not supported: " << videoPath << std::endl;
            return false;
        }

        inputs.clear();
        targets.clear();
        inputSize = maxSeqLen * embeddingDim;
        outputSize = 1;

        if (!this->langProc) {
            std::cerr << "Error: Language processor not initialized for video learning." << std::endl;
            return false;
        }

        // Extract frames from video
        auto frames = visionModule->extractVideoFrames(videoPath, -1, frameSamplingRate);
        if (frames.empty()) {
            std::cerr << "Error: No frames extracted from video: " << videoPath << std::endl;
            return false;
        }

        for (const auto& frame : frames) {
            // Process visual features
            std::vector<float> visualFeatures = frame.features;

            // Extract and process text from frame if available
            std::vector<float> textFeatures;
            if (!frame.ocrText.empty()) {
                std::vector<std::string> words = this->langProc->tokenize(frame.ocrText);
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

                // Flatten text features
                for (const auto& vec : sentenceEmbedding) {
                    textFeatures.insert(textFeatures.end(), vec.begin(), vec.end());
                }
            } else {
                // No text, use zeros
                textFeatures.resize(maxSeqLen * embeddingDim, 0.0f);
            }

            // Combine visual and text features
            std::vector<float> combinedFeatures;
            combinedFeatures.reserve(visualFeatures.size() + textFeatures.size());
            combinedFeatures.insert(combinedFeatures.end(), visualFeatures.begin(), visualFeatures.end());
            combinedFeatures.insert(combinedFeatures.end(), textFeatures.begin(), textFeatures.end());

            // Adjust inputSize if necessary
            if (inputSize == 0 || (int)combinedFeatures.size() != inputSize) {
                inputSize = combinedFeatures.size();
            }

            inputs.push_back(combinedFeatures);
            targets.push_back({0.0f}); // Dummy target for unsupervised learning
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
    // Initialize AudioModule if not already done
    static std::unique_ptr<AudioModule> audioModule;
    if (!audioModule) {
        audioModule = std::make_unique<AudioModule>("AudioLearning");
        if (!audioModule->initialize()) {
            std::cerr << "Error: Failed to initialize AudioModule for audio learning." << std::endl;
            return false;
        }
    }

    // Load audio data
    auto audioData = audioModule->audioPathToNumericalData(audioPath);
    if (audioData.empty()) {
        std::cerr << "Error: Failed to load audio data from: " << audioPath << std::endl;
        return false;
    }

    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1;

    if (!this->langProc) {
        std::cerr << "Error: Language processor not initialized for audio learning." << std::endl;
        return false;
    }

    // Extract audio features and convert to embeddings
    auto features = audioModule->extractFeaturesFromData(audioData);

    // Split audio into segments and create embeddings
    size_t segmentSize = audioData.size() / maxSeqLen;
    for (int i = 0; i < maxSeqLen; ++i) {
        size_t startIdx = i * segmentSize;
        size_t endIdx = std::min(startIdx + segmentSize, audioData.size());

        std::vector<float> segment(audioData.begin() + startIdx, audioData.begin() + endIdx);

        // Extract features for this segment
        auto segmentFeatures = audioModule->extractFeaturesFromData(segment);

        // Convert features to embeddings (simplified approach)
        std::vector<std::vector<float>> segmentEmbedding;
        for (size_t j = 0; j < segmentFeatures.size() && j < (size_t)embeddingDim; ++j) {
            segmentEmbedding.push_back({segmentFeatures[j]});
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
    // Initialize VisionModule if not already done
    static std::unique_ptr<VisionModule> visionModule;
    if (!visionModule) {
        visionModule = std::make_unique<VisionModule>("ImageLearning");
        if (!visionModule->initialize()) {
            std::cerr << "Error: Failed to initialize VisionModule for image learning." << std::endl;
            return false;
        }
    }

    inputs.clear();
    targets.clear();
    inputSize = maxSeqLen * embeddingDim;
    outputSize = 1;

    // Extract image features
    auto imageFeatures = visionModule->extractFeatures(imagePath);
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



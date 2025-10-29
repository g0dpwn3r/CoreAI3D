// D:/code/CoreAI3D/CoreAI3D/main.cpp

// Do NOT define _WIN32_WINNT or WIN32_LEAN_AND_MEAN here.
// They are handled by CMake's target_compile_definitions and _WINSOCKAPI_ in main.hpp.

#include "main.hpp" // This MUST be the first include for your project's headers

#include "include/Core.hpp"
#include "include/Train.hpp"
#include "include/Language.hpp"
#include "include/APIServer.hpp"
#include "include/WebSocketServer.hpp"

// External library headers (Boost, cURL, MySQL, etc.)
// Include specific Boost headers as needed for their functionality.
namespace po = boost::program_options; // Alias for convenience

int main(int argc, char* argv[]) {
    // Simple argument parsing without Boost
    bool verbose = false;
    std::string inputFile;
    std::string targetFile;
    std::string delimiter = ",";
    std::string samples = "0";
    std::string language = "en";
    std::string embeddingFile = "embedding.txt";
    std::string epochs = "10";
    std::string learningRate = "0.01";
    std::string layers = "3";
    std::string neurons = "10";
    std::string minRange = "0.0";
    std::string maxRange = "1.0";
    std::string inputSize = "1";
    std::string outputSize = "1";
    std::string datasetName = "online-1a";
    std::string datasetId = "-1";
    std::string outputCsvFile;
    std::string apiPort = "8080";
    std::string dbHost = "localhost";
    std::string dbUser = "root";
    std::string dbPassword = "password";
    std::string dbPort = "3306";
    std::string dbSchema = "coreai_db";
    bool hasHeader = true;
    bool containsText = false;
    bool startChat = false;
    bool startPredict = false;
    bool isOfflineMode = false;
    bool enableWebsocket = false;
    bool createTableFlag = false;

    // Basic argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "CoreAI3D Help\n";
            std::cout << "--verbose,-v: Enable verbose output\n";
            std::cout << "--input-file,-i: Input file\n";
            std::cout << "--output-csv,-o: Output CSV file\n";
            std::cout << "--api-port: API server port (default: 8080)\n";
            std::cout << "--db-host: Database host (default: localhost)\n";
            std::cout << "--db-user: Database user (default: root)\n";
            std::cout << "--db-password: Database password (default: password)\n";
            std::cout << "--db-port: Database port (default: 3306)\n";
            std::cout << "--db-schema: Database schema (default: coreai_db)\n";
            std::cout << "--start-chat: Start chat mode\n";
            std::cout << "--start-predict: Start prediction mode\n";
            std::cout << "--offline: Run in offline mode\n";
            std::cout << "--enable-websocket: Enable WebSocket server in API mode\n";
            std::cout << "--create-table: Create database tables during initialization\n";
            return 0;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--input-file" || arg == "-i") {
            if (i + 1 < argc) inputFile = argv[++i];
        } else if (arg == "--output-csv" || arg == "-o") {
            if (i + 1 < argc) outputCsvFile = argv[++i];
        } else if (arg == "--api-port") {
            if (i + 1 < argc) apiPort = argv[++i];
        } else if (arg == "--start-predict") {
            startPredict = true;
        } else if (arg == "--start-chat") {
            startChat = true;
        } else if (arg == "--offline") {
            isOfflineMode = true;
        } else if (arg == "--enable-websocket") {
            enableWebsocket = true;
        } else if (arg == "--create-table") {
            createTableFlag = true;
        } else if (arg == "--db-host") {
            if (i + 1 < argc) dbHost = argv[++i];
        } else if (arg == "--db-user") {
            if (i + 1 < argc) dbUser = argv[++i];
        } else if (arg == "--db-password") {
            if (i + 1 < argc) dbPassword = argv[++i];
        } else if (arg == "--db-port") {
            if (i + 1 < argc) dbPort = argv[++i];
        } else if (arg == "--db-schema") {
            if (i + 1 < argc) dbSchema = argv[++i];
        }
    }

    // po::variables_map vm;
    // try {
    //     po::store(po::parse_command_line(argc, argv, desc), vm);
    //     po::notify(vm);
    // }
    // catch (const po::error& e) {
    //     std::cerr << "ERROR: Error parsing arguments: " << e.what() << std::endl;
    //     std::cerr << desc << std::endl; // Print help message from Boost.Program_options
    //     return 1;
    // }

    // Help already handled above
    // bool verbose = vm["verbose"].as<bool>();

    // --- START DEBUG PRINTS (Argc/Argv) ---
    if (verbose) {
        std::cout << "DEBUG: argc = " << argc << std::endl;
        for (int i = 0; i < argc; ++i) {
            std::cout << "DEBUG: argv[" << i << "] = \"" << argv[i] << "\"" << std::endl;
        }
        std::cout << "DEBUG: End of argv dump." << std::endl;
    }
    // --- END DEBUG PRINTS (Argc/Argv) ---

    // Help already handled above

    // Convert string arguments to appropriate types
    char delimiter_char = delimiter.empty() ? ',' : delimiter[0];

    int numSamples_val = 0;
    try { numSamples_val = std::stoi(samples); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for samples. Defaulting to 0.\n"; numSamples_val = 0; }

    int epochs_val = 10;
    try { epochs_val = std::stoi(epochs); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for epochs. Defaulting to 10.\n"; epochs_val = 10; }

    double learningRate_val = 0.01;
    try {
        std::string lr_str = learningRate;
        std::replace(lr_str.begin(), lr_str.end(), ',', '.');
        learningRate_val = std::stod(lr_str);
    }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for learning-rate. Defaulting to 0.01.\n"; learningRate_val = 0.01; }

    int layers_val = 3;
    try { layers_val = std::stoi(layers); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for layers. Defaulting to 3.\n"; layers_val = 3; }

    int neurons_val = 10;
    try { neurons_val = std::stoi(neurons); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for neurons. Defaulting to 10.\n"; neurons_val = 10; }

    float minRange_val = 0.0f;
    try {
        std::string min_str = minRange;
        std::replace(min_str.begin(), min_str.end(), ',', '.');
        minRange_val = std::stof(min_str);
    }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for min. Defaulting to 0.0.\n"; minRange_val = 0.0f; }

    float maxRange_val = 1.0f;
    try {
        std::string max_str = maxRange;
        std::replace(max_str.begin(), max_str.end(), ',', '.');
        maxRange_val = std::stof(max_str);
    }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for max. Defaulting to 1.0.\n"; maxRange_val = 1.0f; }

    int inputSize_val = 1;
    try { inputSize_val = std::stoi(inputSize); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for input-size. Defaulting to 1.\n"; inputSize_val = 1; }

    int outputSize_val = 1;
    try { outputSize_val = std::stoi(outputSize); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for output-size. Defaulting to 1.\n"; outputSize_val = 1; }

    int datasetId_val = -1;
    try { datasetId_val = std::stoi(datasetId); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for dataset-id. Defaulting to -1.\n"; datasetId_val = -1; }

    int apiPort_val = 8080;
    try { apiPort_val = std::stoi(apiPort); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for api-port. Defaulting to 8080.\n"; apiPort_val = 8080; }

    int dbPort_val = 3306;
    try { dbPort_val = std::stoi(dbPort); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for db-port. Defaulting to 3306.\n"; dbPort_val = 3306; }

    if (verbose) std::cout << "DEBUG: All arguments retrieved. Proceeding to logic.\n";


#ifdef USE_MYSQL
    // MySQL variables
    bool createTables = createTableFlag;
    std::string sslModeStr = "DISABLED";

    // Map string to SSLMode enum
    SSLMode ssl = SSLMode::DISABLED; // Default
    if (sslModeStr == "DISABLED") {
        ssl = SSLMode::DISABLED;
    }
    else if (sslModeStr == "REQUIRED") {
        ssl = SSLMode::REQUIRED;
    }
    else if (sslModeStr == "VERIFY_CA") {
        ssl = SSLMode::VERIFY_CA;
    }
    else if (sslModeStr == "VERIFY_IDENTITY") {
        ssl = SSLMode::VERIFY_IDENTITY;
    }
    else {
        std::cerr << "Warning: Unrecognized SSL mode '" << sslModeStr
            << "'. Defaulting to DISABLED." << std::endl;
        ssl = SSLMode::DISABLED; // Ensure ssl is set to a valid default
    }
#endif

    // --- Mode Check ---
    // Default to API server mode if no other mode is specified
    bool apiMode = (!startChat && !startPredict);

    // --- Main Logic Branches ---
    try
    {
        if (startChat)
        {
            if (verbose) std::cout << "Starting chat mode...\n";

            // File-based chat mode (no MySQL dependencies)
            int embeddingDimension = 300; // Default embedding dimension
            try {
                embeddingDimension = std::stoi(inputSize); // Use inputSize as embedding dimension if provided
            } catch (...) {
                embeddingDimension = 300; // Fallback
            }

            // Create Language processor with database parameters
            Language langProcessor(embeddingFile, embeddingDimension, dbHost, dbPort_val,
                                 dbUser, dbPassword, dbSchema, 0, language,
                                 inputSize_val, outputSize_val, layers_val, neurons_val, 1);

            std::string actualEmbeddingFile = embeddingFile.empty()
                ? std::string(language) + "_embeddings.txt"
                : embeddingFile;
            std::cout << "Loading embeddings from: " << actualEmbeddingFile
                << " for language: " << language << std::endl;
            langProcessor.setCurrentLanguage(language);

            // Start chat with database persistence
            langProcessor.chat();
        }
        else if (startPredict)
        {
            if (verbose) std::cout << "[PREDICT MODE] Starting prediction mode...\n";

            // Check for required arguments
            if (inputFile.empty()) {
                std::cerr << "Error: --input-file is required for predict mode." << std::endl;
                return 1;
            }
            if (outputCsvFile.empty()) {
                std::cerr << "Error: --output-csv is required for predict mode." << std::endl;
                return 1;
            }

            if (inputFile.empty() || outputCsvFile.empty())
            {
                std::cerr << "Error: Input and output files are required for "
                    "'predict' mode. Exiting.\n";
                return 1;
            }

            std::cout << "[PREDICT MODE] Initializing Training object...\n";
            Training trainer = Training(true, verbose);
            std::cout << "[PREDICT MODE] Training object initialized.\n";

            // Set training parameters (important for model structure if loading from DB)
            trainer.layers = layers_val;
            trainer.embedding_file = embeddingFile;
            trainer.language = language;
            trainer.neurons = neurons_val;
            trainer.min = minRange_val;
            trainer.max = maxRange_val;
            trainer.outputSize = outputSize_val; // Ensure this is set for backend actions
            trainer.numSamples = numSamples_val;
            // Set inputSize for trainer, as it's needed for CoreAI initialization later in preprocess
            trainer.inputSize = inputSize_val;

            // Skip language processor initialization entirely to avoid MySQL errors
            std::cout << "[PREDICT MODE] Skipping language processor initialization to avoid database connection issues.\n";
            // Proceed directly to CSV loading

            std::cout << "[PREDICT MODE] Loading data from: " << inputFile << std::endl;
            if (!trainer.loadCSV(inputFile, numSamples_val, outputSize_val, hasHeader, containsText, delimiter_char, datasetName)) {
                std::cerr << "Failed to load CSV data for prediction. Exiting." << std::endl;
                return 1;
            }
            std::cout << "[PREDICT MODE] CSV data loaded.\n";

            // If target-file is provided, load targets separately
            if (!targetFile.empty()) {
                std::cout << "[PREDICT MODE] Loading targets from: " << targetFile << std::endl;
                if (!trainer.loadTargetsCSV(targetFile, delimiter_char, hasHeader, containsText, datasetId_val)) {
                    std::cerr << "Failed to load target CSV data for prediction. Exiting." << std::endl;
                    return 1;
                }
                std::cout << "[PREDICT MODE] Target CSV data loaded.\n";
            }


            std::cout << "[PREDICT MODE] Attempting to load model from database (if datasetId is specified)....\n";
            if (datasetId_val != -1)
            {
                std::cout << "[PREDICT MODE] Using dataset ID " << datasetId_val
                    << " (likely to load a pre-trained model).\n";
                if (!trainer.loadDatasetFromDB(datasetId_val)) { // Pass datasetId by value or correct reference
                    std::cerr << "[PREDICT MODE] Failed to load dataset from DB for ID " << datasetId_val << ". Proceeding without pre-loaded data/model from DB." << std::endl;
                }
                bool model_loaded_success = trainer.loadModel(datasetId_val); // Capture the bool return value
                if (model_loaded_success) { // Use the captured bool in the conditional
                    std::cout << "[PREDICT MODE] Model loaded from database for ID " << datasetId_val << ".\n";
                }
                else {
                    std::cerr << "[PREDICT MODE] Failed to load model from DB for ID " << datasetId_val << ". Will initialize new model for training.\n";
                }
            }
            std::cout << "[PREDICT MODE] Model loading check complete.\n";


            std::cout << "[PREDICT MODE] Preprocessing data (normalization, CoreAI initialization)....\n";
            trainer.preprocess(minRange_val, maxRange_val); // This re-normalizes and initializes CoreAI
            std::cout << "[PREDICT MODE] Data preprocessing complete. CoreAI initialized.\n";

            // DEBUG: Log data statistics after preprocessing
            if (verbose) {
                std::cout << "[DEBUG PREDICT] After preprocessing - numSamples: " << trainer.numSamples
                          << ", inputSize: " << trainer.inputSize << ", outputSize: " << trainer.outputSize << std::endl;
                if (!trainer.getTargets().empty()) {
                    std::cout << "[DEBUG PREDICT] First target value: " << trainer.getTargets()[0][0] << std::endl;
                }
            }

            std::cout << "[PREDICT MODE] Starting model training...\n";
            auto start_time = std::chrono::high_resolution_clock::now();
            trainer.train(learningRate_val, epochs_val);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> training_duration = end_time - start_time;
            std::cout << "[PREDICT MODE] Model training complete.\n";

            // Calculate and display metrics
            float rmse = trainer.calculateRMSE();
            float mse = trainer.calculateMSE();
            float accuracy = trainer.calculateAccuracy();

            std::cout << "\n[PREDICT MODE] Evaluation Metrics:\n";
            std::cout << "RMSE: " << rmse << "\n";
            std::cout << "MSE: " << mse << "\n";
            std::cout << "Accuracy: " << accuracy * 100.0f << "%\n";
            std::cout << "Training Execution Time: " << training_duration.count() << " seconds\n";


            std::cout << "[PREDICT MODE] Saving prediction results to: " << outputCsvFile << std::endl;
            if (trainer.saveResultsToCSV(outputCsvFile, inputFile, hasHeader, delimiter_char)) {
                std::cout << "[PREDICT MODE] Results saved successfully.\n";
            }
            else {
                std::cerr << "[PREDICT MODE] Failed to save results to CSV.\n";
            }


            std::cout << "Prediction mode finished.\n";
        }
        else if (apiMode)
        { // API server mode

            std::cout << "Starting API server mode...\n";

            // Initialize Training object
            std::cout << "[API MODE] Initializing Training object...\n";
            std::unique_ptr<Training> trainer;
            trainer = std::make_unique<Training>(true, verbose); // Always offline for API mode to avoid MySQL issues
            std::cout << "[API MODE] Training object initialized.\n";


            // Set training parameters (now that they are members of Training
            // class)
            trainer->layers = layers_val;
            trainer->neurons = neurons_val;
            trainer->min = minRange_val;
            trainer->max = maxRange_val;
            trainer->outputSize = outputSize_val; // Ensure this is set for backend actions
            trainer->inputSize = inputSize_val; // Ensure inputSize is set for CoreAI
            trainer->numSamples = numSamples_val;

            // Initialize CoreAI via trainer's preprocess if you want it managed there
            // Or create it directly here if API server doesn't use the full training pipeline.
            // Assuming the API server will need a CoreAI instance.
            // If CoreAI is only created here, remember it won't be part of the `trainer` object.
            std::cout << "[API MODE] Creating CoreAI instance for API...\n";
            CoreAI core_api_instance(inputSize_val, layers_val, neurons_val, outputSize_val, minRange_val, maxRange_val);
            std::cout << "[API MODE] CoreAI instance created.\n";


            // Initialize API Server
            std::cout << "[API MODE] Initializing API Server...\n";
            APIServer apiServer("CoreAI3D_API", "0.0.0.0", apiPort_val);
            if (!apiServer.initialize("config.json")) {
                std::cerr << "Failed to initialize API server\n";
                return 1;
            }
            std::cout << "[API MODE] API Server initialized.\n";

            // Set training module for neural API endpoints
            apiServer.setTrainingModule(std::move(trainer));

            // Start API Server
            std::cout << "[API MODE] Starting API Server...\n";
            if (!apiServer.start()) {
                std::cerr << "Failed to start API server\n";
                return 1;
            }
            std::cout << "[API MODE] API Server started.\n";

            // Initialize WebSocket Server if enabled
            std::unique_ptr<WebSocketServer> wsServerPtr;
            if (enableWebsocket) {
                std::cout << "[API MODE] Initializing WebSocket Server...\n";
                wsServerPtr = std::make_unique<WebSocketServer>("CoreAI3D_WebSocket", "0.0.0.0");
                if (!wsServerPtr->initialize()) {
                    std::cerr << "Failed to initialize WebSocket server\n";
                    return 1;
                }
                std::cout << "[API MODE] WebSocket Server initialized.\n";

                // Start WebSocket Server
                std::cout << "[API MODE] Starting WebSocket Server...\n";
                if (!wsServerPtr->start()) {
                    std::cerr << "Failed to start WebSocket server\n";
                    return 1;
                }
                std::cout << "[API MODE] WebSocket Server started.\n";
            } else {
                std::cout << "[API MODE] WebSocket Server disabled (use --enable-websocket to enable).\n";
            }

            // Wait for servers to finish (they handle their own signals)
            bool wsRunning = wsServerPtr && wsServerPtr->isServerRunning();
            while (apiServer.isServerRunning() || wsRunning) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                wsRunning = wsServerPtr && wsServerPtr->isServerRunning();
            }

            std::cout << "Application gracefully exited.\n";
        }
    }
    catch (const std::runtime_error& err)
    {
        std::cerr << "Runtime Error caught: " << err.what() << std::endl;
        return 1;
    }
    catch (const std::exception& err)
    {
        std::cerr << "General Exception caught: " << err.what() << std::endl;
        return 1;
    }

    return 0;
}
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
    std::string loadModelId = "";
    std::string outputCsvFile;
    std::string apiPort = "8080";
    std::string dbHost = "0.0.0.0";
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

    // Helper function to check if next argument is a value (not a flag)
    auto hasValue = [&](int idx) -> bool {
        return idx + 1 < argc && std::string(argv[idx + 1]).find("--") != 0;
    };

    // Basic argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "CoreAI3D - Advanced AI Training and Prediction Tool\n";
            std::cout << "====================================================\n\n";
            std::cout << "USAGE:\n";
            std::cout << "  CoreAI3D [OPTIONS]\n\n";
            std::cout << "DESCRIPTION:\n";
            std::cout << "  CoreAI3D is a comprehensive AI tool that supports training neural networks,\n";
            std::cout << "  making predictions, running an API server, and interactive chat functionality.\n";
            std::cout << "  It can operate in online mode (with database) or offline mode.\n\n";
            std::cout << "MODES:\n";
            std::cout << "  By default, CoreAI3D runs in API server mode. Use the following flags to select other modes:\n\n";
            std::cout << "GENERAL OPTIONS:\n";
            std::cout << "  --help, -h\n";
            std::cout << "    Display this help message and exit.\n";
            std::cout << "    Usage: --help or -h\n";
            std::cout << "    Example: ./CoreAI3D --help\n\n";
            std::cout << "  --verbose, -v\n";
            std::cout << "    Enable verbose output for detailed logging and debug information.\n";
            std::cout << "    This provides additional console output during execution.\n";
            std::cout << "    Usage: --verbose or -v\n";
            std::cout << "    Example: ./CoreAI3D --verbose --start-predict\n\n";
            std::cout << "  --offline\n";
            std::cout << "    Run in offline mode without database connectivity.\n";
            std::cout << "    Useful when database is not available or for standalone operations.\n";
            std::cout << "    Usage: --offline\n";
            std::cout << "    Example: ./CoreAI3D --offline --start-predict --input-file data.csv\n\n";
            std::cout << "MODE SELECTION:\n";
            std::cout << "  --start-chat\n";
            std::cout << "    Start interactive chat mode for natural language processing.\n";
            std::cout << "    Requires database connection for embedding storage unless in offline mode.\n";
            std::cout << "    Usage: --start-chat\n";
            std::cout << "    Dependencies: Database connection (unless --offline is specified)\n";
            std::cout << "    Example: ./CoreAI3D --start-chat --db-host localhost\n\n";
            std::cout << "  --start-predict\n";
            std::cout << "    Start prediction mode for training and evaluating neural networks.\n";
            std::cout << "    Requires input and output file specifications.\n";
            std::cout << "    Usage: --start-predict --input-file <file> --output-csv <file>\n";
            std::cout << "    Dependencies: --input-file and --output-csv are required\n";
            std::cout << "    Example: ./CoreAI3D --start-predict --input-file training_data.csv --output-csv predictions.csv\n\n";
            std::cout << "FILE OPTIONS:\n";
            std::cout << "  --input-file, -i <file>\n";
            std::cout << "    Specify the input CSV file for training or prediction data.\n";
            std::cout << "    Expected value: Path to a valid CSV file\n";
            std::cout << "    Usage: --input-file <path> or -i <path>\n";
            std::cout << "    Dependencies: Required for --start-predict mode\n";
            std::cout << "    Example: --input-file ./data/input.csv\n\n";
            std::cout << "  --output-csv, -o <file>\n";
            std::cout << "    Specify the output CSV file for saving prediction results.\n";
            std::cout << "    Expected value: Path where the output CSV will be written\n";
            std::cout << "    Usage: --output-csv <path> or -o <path>\n";
            std::cout << "    Dependencies: Required for --start-predict mode\n";
            std::cout << "    Example: --output-csv ./results/predictions.csv\n\n";
            std::cout << "API SERVER OPTIONS:\n";
            std::cout << "  --api-port <port>\n";
            std::cout << "    Set the port number for the API server.\n";
            std::cout << "    Expected value: Integer between 1024-65535 (default: 8080)\n";
            std::cout << "    Usage: --api-port <number>\n";
            std::cout << "    Dependencies: Used in API server mode (default mode)\n";
            std::cout << "    Example: --api-port 3000\n\n";
            std::cout << "  --enable-websocket\n";
            std::cout << "    Enable WebSocket server alongside the API server for real-time communication.\n";
            std::cout << "    Usage: --enable-websocket\n";
            std::cout << "    Dependencies: Only effective in API server mode\n";
            std::cout << "    Example: ./CoreAI3D --enable-websocket --api-port 8080\n\n";
            std::cout << "DATABASE OPTIONS:\n";
            std::cout << "  --db-host <host>\n";
            std::cout << "    Specify the database server hostname or IP address.\n";
            std::cout << "    Expected value: Hostname or IP address (default: 0.0.0.0)\n";
            std::cout << "    Usage: --db-host <hostname>\n";
            std::cout << "    Dependencies: Required for online modes unless --offline is specified\n";
            std::cout << "    Example: --db-host mysql.example.com\n\n";
            std::cout << "  --db-user <username>\n";
            std::cout << "    Database username for authentication.\n";
            std::cout << "    Expected value: Valid database username (default: root)\n";
            std::cout << "    Usage: --db-user <username>\n";
            std::cout << "    Dependencies: Required for online modes unless --offline is specified\n";
            std::cout << "    Example: --db-user myuser\n\n";
            std::cout << "  --db-password <password>\n";
            std::cout << "    Database password for authentication.\n";
            std::cout << "    Expected value: Database password string (default: password)\n";
            std::cout << "    Usage: --db-password <password>\n";
            std::cout << "    Dependencies: Required for online modes unless --offline is specified\n";
            std::cout << "    Example: --db-password mysecretpass\n\n";
            std::cout << "  --db-port <port>\n";
            std::cout << "    Database server port number.\n";
            std::cout << "    Expected value: Integer port number (default: 3306 for MySQL)\n";
            std::cout << "    Usage: --db-port <number>\n";
            std::cout << "    Dependencies: Required for online modes unless --offline is specified\n";
            std::cout << "    Example: --db-port 3306\n\n";
            std::cout << "  --db-schema <schema>\n";
            std::cout << "    Database schema/name to connect to.\n";
            std::cout << "    Expected value: Database schema name (default: coreai_db)\n";
            std::cout << "    Usage: --db-schema <name>\n";
            std::cout << "    Dependencies: Required for online modes unless --offline is specified\n";
            std::cout << "    Example: --db-schema ai_models\n\n";
            std::cout << "  --create-table\n";
            std::cout << "    Automatically create required database tables during initialization.\n";
            std::cout << "    Use this when setting up the database for the first time.\n";
            std::cout << "    Usage: --create-table\n";
            std::cout << "    Dependencies: Requires database connection and appropriate permissions\n";
            std::cout << "    Example: ./CoreAI3D --create-table --db-host localhost\n\n";
            std::cout << "MODEL OPTIONS:\n";
            std::cout << "  --load-model <id_or_name>\n";
            std::cout << "    Load a pre-trained model from the database by ID or name.\n";
            std::cout << "    Expected value: Integer ID or string name of saved model\n";
            std::cout << "    Usage: --load-model <id> or --load-model <name>\n";
            std::cout << "    Dependencies: Requires database connection; used in prediction mode\n";
            std::cout << "    Example: --load-model 123 or --load-model my_trained_model\n\n";
            std::cout << "EXAMPLES:\n";
            std::cout << "  1. Start API server with WebSocket support:\n";
            std::cout << "     ./CoreAI3D --enable-websocket --api-port 8080\n\n";
            std::cout << "  2. Run prediction with custom database settings:\n";
            std::cout << "     ./CoreAI3D --start-predict --input-file data.csv --output-csv results.csv \\\n";
            std::cout << "                --db-host db.example.com --db-user aiuser --db-password secret\n\n";
            std::cout << "  3. Start chat mode in offline mode:\n";
            std::cout << "     ./CoreAI3D --start-chat --offline\n\n";
            std::cout << "  4. Create database tables:\n";
            std::cout << "     ./CoreAI3D --create-table --db-host localhost --db-user root --db-password mypass\n\n";
            std::cout << "NOTES:\n";
            std::cout << "  - Database options are ignored when --offline is specified\n";
            std::cout << "  - API server mode is the default when no specific mode is selected\n";
            std::cout << "  - All file paths should be accessible from the current working directory\n";
            std::cout << "  - Verbose mode (--verbose) provides additional debugging information\n\n";
            return 0;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--input-file" || arg == "-i") {
            if (hasValue(i)) inputFile = argv[++i];
            else std::cerr << "ERROR: --input-file requires a value\n";
        } else if (arg == "--output-csv" || arg == "-o") {
            if (hasValue(i)) outputCsvFile = argv[++i];
            else std::cerr << "ERROR: --output-csv requires a value\n";
        } else if (arg == "--api-port") {
            if (hasValue(i)) apiPort = argv[++i];
            else std::cerr << "ERROR: --api-port requires a value\n";
        } else if (arg == "--start-predict") {
            startPredict = true;
        } else if (arg == "--start-chat") {
            startChat = true;
        } else if (arg == "--load-model") {
            if (hasValue(i)) loadModelId = argv[++i];
            else std::cerr << "ERROR: --load-model requires a value\n";
        } else if (arg == "--offline") {
            isOfflineMode = true;
        } else if (arg == "--enable-websocket") {
            enableWebsocket = true;
        } else if (arg == "--create-table") {
            createTableFlag = true;
        } else if (arg == "--db-host") {
            if (hasValue(i)) dbHost = argv[++i];
            else std::cerr << "ERROR: --db-host requires a value\n";
        } else if (arg == "--db-user") {
            if (hasValue(i)) dbUser = argv[++i];
            else std::cerr << "ERROR: --db-user requires a value\n";
        } else if (arg == "--db-password") {
            if (hasValue(i)) dbPassword = argv[++i];
            else std::cerr << "ERROR: --db-password requires a value\n";
        } else if (arg == "--db-port") {
            if (hasValue(i)) dbPort = argv[++i];
            else std::cerr << "ERROR: --db-port requires a value\n";
        } else if (arg == "--db-schema") {
            if (hasValue(i)) dbSchema = argv[++i];
            else std::cerr << "ERROR: --db-schema requires a value\n";
        } else {
            std::cerr << "ERROR: Unknown argument: " << arg << std::endl;
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
            bool useOnlineMode = (datasetId_val != -1 || !loadModelId.empty());
            Training trainer = Training(useOnlineMode, verbose);
            std::cout << "[PREDICT MODE] Training object initialized in " << (useOnlineMode ? "online" : "offline") << " mode.\n";

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


            std::cout << "[PREDICT MODE] Attempting to load model from database (if datasetId or loadModelId is specified)....\n";
            if (datasetId_val != -1 || !loadModelId.empty())
            {
                int modelIdToLoad = datasetId_val != -1 ? datasetId_val : std::stoi(loadModelId);
                std::cout << "[PREDICT MODE] Using model ID " << modelIdToLoad
                    << " (likely to load a pre-trained model).\n";
                if (!trainer.loadDatasetFromDB(modelIdToLoad)) { // Pass modelId by value or correct reference
                    std::cerr << "[PREDICT MODE] Failed to load dataset from DB for ID " << modelIdToLoad << ". Proceeding without pre-loaded data/model from DB." << std::endl;
                }
#ifdef USE_MYSQL
                bool model_loaded_success = false;
                try {
                    if (!loadModelId.empty()) {
                        model_loaded_success = trainer.loadModel(loadModelId); // Load by name
                    } else {
                        model_loaded_success = trainer.loadModel(modelIdToLoad); // Load by ID
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[PREDICT MODE] Exception during model loading: " << e.what() << std::endl;
                    model_loaded_success = false;
                }
                if (!model_loaded_success) {
                    std::cerr << "[PREDICT MODE] Failed to load model. Proceeding with new model initialization.\n";
                }
                if (model_loaded_success) { // Use the captured bool in the conditional
                    std::cout << "[PREDICT MODE] Model loaded from database for ID " << modelIdToLoad << ".\n";
                }
                else {
                    std::cerr << "[PREDICT MODE] Failed to load model from DB for ID " << modelIdToLoad << ". Will initialize new model for training.\n";
                }
#endif
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

                // Check for connection health and attempt reconnection if needed
                if (wsServerPtr && !wsServerPtr->isHealthy()) {
                    std::cerr << "WebSocket server health check failed, attempting restart..." << std::endl;
                    wsServerPtr->restartServer();
                }
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
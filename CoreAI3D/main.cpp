// D:/code/CoreAI3D/CoreAI3D/main.cpp

// Do NOT define _WIN32_WINNT or WIN32_LEAN_AND_MEAN here.
// They are handled by CMake's target_compile_definitions and _WINSOCKAPI_ in main.hpp.

#include "main.hpp" // This MUST be the first include for your project's headers

#include "include/Core.hpp"
#include "include/Train.hpp"
#include "include/Language.hpp"
#include "include/APIServer.hpp"
#include "include/WebSocketServer.hpp"
#include "include/Predict.hpp"

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
    std::string windowSize = "1";
    std::string minRange = "0.0";
    std::string maxRange = "1.0";
    std::string inputSize = "1";
    std::string outputSize = "1";
    std::string datasetName = "online-1a";
    std::string datasetId = "-1";
    std::string loadModelId = "";
    std::string outputCsvFile;
    std::string apiPort = "8080";
    std::string dbHost = "localhost";
    std::string dbUser = "root";
    std::string dbPassword = "password";
    std::string dbPort = "33060";
    std::string dbSchema = "coreai_db";
    bool hasHeader = false;
    bool containsText = false;
    bool startChat = false;
    bool startPredict = false;
    bool isOfflineMode = false;
    bool enableWebsocket = false;
    bool createTableFlag = false;
    std::string batchSize = "32";
    bool calculusActivation = false;
    bool containsTextFlag = false;
    bool containsHeader = false;
    bool containsDate = false;
    std::string threads = "1";

    // Helper function to check if next argument is a value (not a flag)
    auto hasValue = [&](int idx) -> bool {
        return idx + 1 < argc && std::string(argv[idx + 1]).find("--") != 0;
    };

    // Helper function to extract value from argument, handling both --arg value and --arg=value formats
    auto extractValue = [&](int& idx, const std::string& argName, const std::string& arg) -> std::string {
        // Check if value is in the same argument after '='
        size_t equalsPos = arg.find('=');
        if (equalsPos != std::string::npos) {
            return arg.substr(equalsPos + 1);
        }
        // Otherwise, expect the next argument to be the value
        if (hasValue(idx)) {
            return argv[++idx];
        }
        std::cerr << "ERROR: " << argName << " requires a value\n";
        return "";
    };

    // Basic argument parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << R"(
        ,     ,
       (\____/)
        (_oo_)
          / \     ╔══════════════════════════════╗
         /| |\    ║     ARCHANGEL SERAPHIM      ║
        / \_/ \   ║   Celestial AI Guardian     ║
       /  ||  \   ╚══════════════════════════════╝
      /   ||   \
     /    ||    \
    /     ||     \
   /      ||      \
  /       ||       \
 /    /\   ||   /\   \
/   //\\  ||  //\\   \
   ///\\\ || ///\\\
  ////\\\\||////\\\\
 ||||||||| |||||||||
 ||||||||| |||||||||
 ||||||||| |||||||||
 ||||||||| |||||||||
 ||||||||| |||||||||
 ||||||||| |||||||||
)" << std::endl;
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
            std::cout << "  --target, -t <file>\n";
            std::cout << "    Specify the target CSV file for training targets.\n";
            std::cout << "    Expected value: Path to a valid CSV file containing target values\n";
            std::cout << "    Usage: --target <path> or -t <path>\n";
            std::cout << "    Dependencies: Used in prediction/training mode\n";
            std::cout << "    Example: --target ./data/targets.csv\n\n";
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
            std::cout << "NEURAL NETWORK OPTIONS:\n";
            std::cout << "  --layers, -l <number>\n";
            std::cout << "    Set the number of hidden layers in the neural network.\n";
            std::cout << "    Expected value: Integer (default: 3)\n";
            std::cout << "    Usage: --layers <number> or -l <number>\n";
            std::cout << "    Dependencies: Used in prediction/training mode\n";
            std::cout << "    Example: --layers 5 or -l 5\n\n";
            std::cout << "  --neurons, -n <number>\n";
            std::cout << "    Set the number of neurons per hidden layer.\n";
            std::cout << "    Expected value: Integer (default: 10)\n";
            std::cout << "    Usage: --neurons <number> or -n <number>\n";
            std::cout << "    Dependencies: Used in prediction/training mode\n";
            std::cout << "    Example: --neurons 20 or -n 20\n\n";
            std::cout << "  --epochs, -e <number>\n";
            std::cout << "    Set the number of training epochs.\n";
            std::cout << "    Expected value: Integer (default: 10)\n";
            std::cout << "    Usage: --epochs <number> or -e <number>\n";
            std::cout << "    Dependencies: Used in prediction/training mode\n";
            std::cout << "    Example: --epochs 100 or -e 100\n\n";
            std::cout << "  --samples, -s <number>\n";
            std::cout << "    Set the number of training samples to use.\n";
            std::cout << "    Expected value: Integer (default: 0 - use all available)\n";
            std::cout << "    Usage: --samples <number> or -s <number>\n";
            std::cout << "    Dependencies: Used in prediction/training mode\n";
            std::cout << "    Example: --samples 1000 or -s 1000\n\n";
            std::cout << "  --learning-rate, -r <rate>\n";
            std::cout << "    Set the learning rate for neural network training.\n";
            std::cout << "    Expected value: Floating point number (default: 0.01)\n";
            std::cout << "    Usage: --learning-rate <rate> or -r <rate>\n";
            std::cout << "    Dependencies: Used in prediction/training mode\n";
            std::cout << "    Example: --learning-rate 0.001 or -r 0.001\n\n";
            std::cout << "  --batch-size, -b <number>\n";
            std::cout << "    Set the batch size for training.\n";
            std::cout << "    Expected value: Integer (default: 32)\n";
            std::cout << "    Usage: --batch-size <number> or -b <number>\n";
            std::cout << "    Dependencies: Used in training mode\n";
            std::cout << "    Example: --batch-size 64 or -b 64\n\n";
            std::cout << "  --input-size, -z <number>\n";
            std::cout << "    Set the input size (dynamically determined from CSV columns if not specified).\n";
            std::cout << "    Expected value: Integer (default: dynamic from CSV)\n";
            std::cout << "    Usage: --input-size <number> or -z <number>\n";
            std::cout << "    Dependencies: Used in prediction mode\n";
            std::cout << "    Example: --input-size 5 or -z 5\n\n";
            std::cout << "  --output-size, -Z <number>\n";
            std::cout << "    Set the output size (default: 1 for regression).\n";
            std::cout << "    Expected value: Integer (default: 1)\n";
            std::cout << "    Usage: --output-size <number> or -Z <number>\n";
            std::cout << "    Dependencies: Used in prediction mode\n";
            std::cout << "    Example: --output-size 1 or -Z 1\n\n";
            std::cout << "  --calculus-activation\n";
            std::cout << "    Enable calculus-based activation functions.\n";
            std::cout << "    Usage: --calculus-activation or -C\n";
            std::cout << "    Dependencies: Used in neural network computations\n";
            std::cout << "    Example: --calculus-activation\n\n";
            std::cout << "  --contains-text\n";
            std::cout << "    Indicate that the dataset contains text data.\n";
            std::cout << "    Usage: --contains-text\n";
            std::cout << "    Dependencies: Used for data preprocessing\n";
            std::cout << "    Example: --contains-text\n\n";
            std::cout << "  --contains-header\n";
            std::cout << "    Indicate that the dataset contains header rows.\n";
            std::cout << "    Usage: --contains-header\n";
            std::cout << "    Dependencies: Used for data parsing\n";
            std::cout << "    Example: --contains-header\n\n";
            std::cout << "  --contains-date\n";
            std::cout << "    Indicate that the dataset contains date columns.\n";
            std::cout << "    Usage: --contains-date\n";
            std::cout << "    Dependencies: Used for data preprocessing\n";
            std::cout << "    Example: --contains-date\n\n";
            std::cout << "  --threads, -j <number>\n";
            std::cout << "    Set the number of threads for parallel processing.\n";
            std::cout << "    Expected value: Integer (default: 1)\n";
            std::cout << "    Usage: --threads <number> or -j <number>\n";
            std::cout << "    Dependencies: Used for multi-threading\n";
            std::cout << "    Example: --threads 4 or -j 4\n\n";
            std::cout << "MODEL OPTIONS:\n";
            std::cout << "  --load-model <id_or_name>\n";
            std::cout << "    Load a pre-trained model from the database by ID or name.\n";
            std::cout << "    Expected value: Integer ID or string name of saved model\n";
            std::cout << "    Usage: --load-model <id> or --load-model <name>\n";
            std::cout << "    Dependencies: Requires database connection; used in prediction mode\n";
            std::cout << "    Example: --load-model 123 or --load-model my_trained_model\n\n";
            std::cout << "  --embedding-file <file>\n";
            std::cout << "    Specify the path to a custom embedding file for language processing.\n";
            std::cout << "    Expected value: Path to a valid embedding file\n";
            std::cout << "    Usage: --embedding-file <path>\n";
            std::cout << "    Dependencies: Used in chat mode for custom embeddings\n";
            std::cout << "    Example: --embedding-file ./embeddings/custom_embeddings.txt\n\n";
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
        } else if (arg.find("--input-file") == 0 || arg == "-i") {
            std::string value = extractValue(i, "--input-file", arg);
            if (!value.empty()) inputFile = value;
        } else if (arg.find("--output-csv") == 0 || arg == "-o") {
            std::string value = extractValue(i, "--output-csv", arg);
            if (!value.empty()) outputCsvFile = value;
        } else if (arg.find("--api-port") == 0) {
            std::string value = extractValue(i, "--api-port", arg);
            if (!value.empty()) apiPort = value;
        } else if (arg == "--start-predict") {
            startPredict = true;
        } else if (arg == "--start-chat") {
            startChat = true;
        } else if (arg.find("--load-model") == 0) {
            std::string value = extractValue(i, "--load-model", arg);
            if (!value.empty()) loadModelId = value;
        } else if (arg.find("--embedding-file") == 0) {
            std::string value = extractValue(i, "--embedding-file", arg);
            if (!value.empty()) embeddingFile = value;
        } else if (arg == "--offline") {
            isOfflineMode = true;
        } else if (arg == "--enable-websocket") {
            enableWebsocket = true;
        } else if (arg == "--create-table" || arg == "--create-tables") {
            createTableFlag = true;
        } else if (arg.find("--db-host") == 0) {
            std::string value = extractValue(i, "--db-host", arg);
            if (!value.empty()) dbHost = value;
        } else if (arg.find("--db-user") == 0) {
            std::string value = extractValue(i, "--db-user", arg);
            if (!value.empty()) dbUser = value;
        } else if (arg.find("--db-password") == 0) {
            std::string value = extractValue(i, "--db-password", arg);
            if (!value.empty()) dbPassword = value;
        } else if (arg.find("--db-port") == 0) {
            std::string value = extractValue(i, "--db-port", arg);
            if (!value.empty()) dbPort = value;
        } else if (arg.find("--db-schema") == 0) {
            std::string value = extractValue(i, "--db-schema", arg);
            if (!value.empty()) dbSchema = value;
        } else if (arg.find("--target") == 0 || arg == "-t") {
            std::string value = extractValue(i, "--target", arg);
            if (!value.empty()) targetFile = value;
        } else if (arg.find("--layers") == 0 || arg == "-l") {
            std::string value = extractValue(i, "--layers", arg);
            if (!value.empty()) layers = value;
        } else if (arg.find("--samples") == 0 || arg == "-s") {
            std::string value = extractValue(i, "--samples", arg);
            if (!value.empty()) samples = value;
        } else if (arg.find("--epochs") == 0 || arg == "-e") {
            std::string value = extractValue(i, "--epochs", arg);
            if (!value.empty()) epochs = value;
        } else if (arg.find("--neurons") == 0 || arg == "-n") {
            std::string value = extractValue(i, "--neurons", arg);
            if (!value.empty()) neurons = value;
        } else if (arg.find("--learning-rate") == 0 || arg == "-r") {
            std::string value = extractValue(i, "--learning-rate", arg);
            if (!value.empty()) learningRate = value;
        } else if (arg.find("--min") == 0) {
            std::string value = extractValue(i, "--min", arg);
            if (!value.empty()) minRange = value;
        } else if (arg.find("--max") == 0) {
            std::string value = extractValue(i, "--max", arg);
            if (!value.empty()) maxRange = value;
        } else if (arg.find("--input-size") == 0 || arg == "--inputsize") {
            std::string value = extractValue(i, "--input-size", arg);
            if (!value.empty()) inputSize = value;
        } else if (arg.find("--output-size") == 0 || arg == "--outputsize") {
            std::string value = extractValue(i, "--output-size", arg);
            if (!value.empty()) outputSize = value;
        } else if (arg.find("--dataset-name") == 0) {
            std::string value = extractValue(i, "--dataset-name", arg);
            if (!value.empty()) datasetName = value;
        } else if (arg.find("--dataset-id") == 0) {
            std::string value = extractValue(i, "--dataset-id", arg);
            if (!value.empty()) datasetId = value;
        } else if (arg.find("--batch-size") == 0 || arg == "-b") {
            std::string value = extractValue(i, "--batch-size", arg);
            if (!value.empty()) batchSize = value;
        } else if (arg == "--calculus-activation" || arg == "-C") {
            calculusActivation = true;
        } else if (arg == "--contains-text") {
            containsTextFlag = true;
        } else if (arg == "--contains-header") {
            containsHeader = true;
        } else if (arg == "--contains-date") {
            containsDate = true;
        } else if (arg.find("--threads") == 0 || arg == "-j") {
            std::string value = extractValue(i, "--threads", arg);
            if (!value.empty()) threads = value;
        } else if (arg.find("--window-size") == 0 || arg == "-w") {
            std::string value = extractValue(i, "--window-size", arg);
            if (!value.empty()) windowSize = value;
        } else {
            std::cerr << "ERROR: Unknown argument: " << arg << std::endl;
        }
    }


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

    int dbPort_val = 33060;
    try { dbPort_val = std::stoi(dbPort); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for db-port. Defaulting to 33060.\n"; dbPort_val = 33060; }

    int windowSize_val = 1;
    try { windowSize_val = std::stoi(windowSize); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for window-size. Defaulting to 1.\n"; windowSize_val = 1; }

    if (verbose) std::cout << "DEBUG: All arguments retrieved. Proceeding to logic.\n";

    // --- Mode Check ---
    // Default to API server mode if no other mode is specified
    bool apiMode = (!startChat && !startPredict);

    // --- DIAGNOSTIC LOGGING FOR COMMAND LINE ARGUMENTS ---
    if (verbose) {
        std::cout << "DIAGNOSTIC: Command line argument analysis:\n";
        std::cout << "  dataset-name: '" << datasetName << "'\n";
        std::cout << "  dataset-id: '" << datasetId << "' (parsed as: " << datasetId_val << ")\n";
        std::cout << "  api-port: '" << apiPort << "' (parsed as: " << apiPort_val << ")\n";
        std::cout << "  enable-websocket: " << (enableWebsocket ? "true" : "false") << "\n";
        std::cout << "  start-chat: " << (startChat ? "true" : "false") << "\n";
        std::cout << "  start-predict: " << (startPredict ? "true" : "false") << "\n";
        std::cout << "  api-mode: " << (apiMode ? "true" : "false") << "\n";
        std::cout << "  offline-mode: " << (isOfflineMode ? "true" : "false") << "\n";
        std::cout << "  verbose: " << (verbose ? "true" : "false") << "\n";
        std::cout << "DIAGNOSTIC: End of command line argument analysis.\n";
    }

#ifdef USE_MYSQL
    // MySQL variables
    bool createTables = createTableFlag;
    std::string sslModeStr = "REQUIRED";

    // Map string to SSLMode enum
    SSLMode ssl = SSLMode::REQUIRED; // Default
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
            << "'. Defaulting to REQUIRED." << std::endl;
        ssl = SSLMode::REQUIRED; // Ensure ssl is set to a valid default
    }
#endif

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

            // Set dbHost to "offline" if offline mode is enabled
            std::string effectiveDbHost = isOfflineMode ? "offline" : dbHost;

            // Create Language processor with database parameters
            Language langProcessor(embeddingFile, embeddingDimension, effectiveDbHost, dbPort_val,
                                 dbUser, dbPassword, dbSchema, 0, language,
                                 inputSize_val, outputSize_val, layers_val, neurons_val, 1);

            std::string actualEmbeddingFile = embeddingFile.empty()
                ? std::string(language) + "_embeddings.txt"
                : embeddingFile;
            std::cout << "Loading embeddings from: " << actualEmbeddingFile
                << " for language: " << language << std::endl;
            langProcessor.setCurrentLanguage(language);

            // Start chat with database persistence (or offline mode)
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

            // Create Predict instance and run prediction
            Predict predictor(verbose, nullptr);
            int batchSize_val = 32;
            try { batchSize_val = std::stoi(batchSize); }
            catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for batch-size. Defaulting to 32.\n"; batchSize_val = 32; }

            bool success = predictor.runPrediction(
                inputFile, outputCsvFile, targetFile, loadModelId, datasetId_val,
                layers_val, neurons_val, minRange_val, maxRange_val,
                0, 1, numSamples_val, learningRate_val, epochs_val,
                embeddingFile, language, datasetName, containsHeader, containsTextFlag, delimiter_char,
                isOfflineMode, dbHost, dbPort_val, dbUser, dbPassword, dbSchema,
                calculusActivation, windowSize_val
            );

            if (!success) {
                std::cerr << "Prediction failed." << std::endl;
                return 1;
            }

            // Display final metrics
            std::cout << "\n[PREDICT MODE] Final Evaluation Metrics:\n";
            std::cout << "RMSE: " << predictor.getRMSE() << "\n";
            std::cout << "MSE: " << predictor.getMSE() << "\n";
            std::cout << "Accuracy: " << predictor.getAccuracy() * 100.0f << "%\n";
            std::cout << "Training Execution Time: " << predictor.getTrainingTime() << " seconds\n";

            std::cout << "Prediction mode finished successfully.\n";
        }
        else if (apiMode)
        { // API server mode

            std::cout << "Starting API server mode...\n";

            // Initialize WebSocket Server if enabled (before Training to pass it to constructor)
            std::unique_ptr<WebSocketServer> wsServerPtr;
            if (enableWebsocket) {
                std::cout << "[API MODE] Initializing WebSocket Server...\n";
                if (verbose) {
                    std::cout << "DIAGNOSTIC: Creating WebSocketServer with name='CoreAI3D_WebSocket', address='0.0.0.0', port=" << (apiPort_val + 1) << "\n";
                }
                wsServerPtr = std::make_unique<WebSocketServer>("CoreAI3D_WebSocket", "0.0.0.0", apiPort_val + 1);
                if (verbose) {
                    std::cout << "DIAGNOSTIC: WebSocketServer object created, calling initialize()...\n";
                }
                if (!wsServerPtr->initialize()) {
                    std::cerr << "Failed to initialize WebSocket server\n";
                    return 1;
                }
                std::cout << "[API MODE] WebSocket Server initialized.\n";

                // Start WebSocket Server
                std::cout << "[API MODE] Starting WebSocket Server...\n";
                if (verbose) {
                    std::cout << "DIAGNOSTIC: Calling WebSocketServer start() method...\n";
                }
                if (!wsServerPtr->start()) {
                    std::cerr << "Failed to start WebSocket server\n";
                    return 1;
                }
                std::cout << "[API MODE] WebSocket Server started on port " << (apiPort_val + 1) << ".\n";
                if (verbose) {
                    std::cout << "DIAGNOSTIC: WebSocket server health status: " << (wsServerPtr->isHealthy() ? "healthy" : "unhealthy") << "\n";
                }
            } else {
                std::cout << "[API MODE] WebSocket Server disabled (use --enable-websocket to enable).\n";
            }

            // Initialize Training object
            std::cout << "[API MODE] Initializing Training object...\n";
            std::unique_ptr<Training> trainer;
            if (verbose) {
                std::cout << "DIAGNOSTIC: Creating Training object with isOffline=true, verbose=" << (verbose ? "true" : "false")
                          << ", wsServerPtr=" << (wsServerPtr.get() ? "valid" : "null") << "\n";
            }
            trainer = std::make_unique<Training>(true, verbose, wsServerPtr.get()); // Always offline for API mode to avoid MySQL issues
            std::cout << "[API MODE] Training object initialized.\n";
            if (verbose) {
                std::cout << "DIAGNOSTIC: Training object created successfully\n";
            }


            // Set training parameters (now that they are members of Training
            // class)
            trainer->layers = layers_val;
            trainer->neurons = neurons_val;
            trainer->min = minRange_val;
            trainer->max = maxRange_val;
            trainer->outputSize = outputSize_val; // Ensure this is set for backend actions
            trainer->inputSize = inputSize_val; // Ensure inputSize is set for CoreAI
            trainer->numSamples = numSamples_val;

            // Set embedding dimension for language processor
            int embeddingDimension = 300; // Default embedding dimension for language processing
            // Note: inputSize is for neural network input size, not embedding dimension
            // Embedding dimension should remain 300 for proper text encoding

            // Initialize CoreAI via trainer's preprocess if you want it managed there
            // Or create it directly here if API server doesn't use the full training pipeline.
            // Assuming the API server will need a CoreAI instance.
            // If CoreAI is only created here, remember it won't be part of the `trainer` object.
            std::cout << "[API MODE] Creating CoreAI instance for API...\n";
            CoreAI core_api_instance(inputSize_val, layers_val, neurons_val, outputSize_val, minRange_val, maxRange_val);
            std::cout << "[API MODE] CoreAI instance created.\n";


            // Initialize API Server
            std::cout << "[API MODE] Initializing API Server...\n";
            if (verbose) {
                std::cout << "DIAGNOSTIC: Creating APIServer with name='CoreAI3D_API', address='0.0.0.0', port=" << apiPort_val << "\n";
            }
            APIServer apiServer("CoreAI3D_API", "0.0.0.0", apiPort_val);
            if (verbose) {
                std::cout << "DIAGNOSTIC: Calling APIServer initialize() with config.json...\n";
                std::cout << "DIAGNOSTIC: DB host='" << dbHost << "', port=" << dbPort_val
                          << ", user='" << dbUser << "', schema='" << dbSchema << "'\n";
            }
#ifdef USE_MYSQL
            if (!apiServer.initialize("config.json", dbHost, (unsigned int)dbPort_val,
                                       dbUser, dbPassword, dbSchema, ssl, createTables)) {
#else
            if (!apiServer.initialize("config.json", dbHost, (unsigned int)dbPort_val,
                                       dbUser, dbPassword, dbSchema)) {
#endif
                std::cerr << "Failed to initialize API server\n";
                return 1;
            }
            std::cout << "[API MODE] API Server initialized.\n";
            if (verbose) {
                std::cout << "DIAGNOSTIC: API Server initialization successful\n";
            }

            // Initialize language processor for text processing
            std::cout << "[API MODE] Initializing Language Processor...\n";
            std::cout << "[API MODE] Parameters: embeddingFile=" << embeddingFile
                      << ", embeddingDimension=" << embeddingDimension
                      << ", language=" << language << std::endl;
            
            trainer->initializeLanguageProcessor(embeddingFile, embeddingDimension, dbHost, dbPort_val,
                                               dbUser, dbPassword, dbSchema, 0, language,
                                               inputSize_val, outputSize_val, layers_val, neurons_val);
            
            // Verify language processor initialization
            if (trainer->getLanguage() != nullptr) {
                std::cout << "[API MODE] Language Processor initialized successfully.\n";
            } else {
                std::cerr << "[API MODE] ERROR: Language Processor initialization failed!\n";
                return 1;
            }

            Language* chatLanguage = trainer->getLanguage();

            // Set training module for neural API endpoints
            apiServer.setTrainingModule(std::move(trainer));
            apiServer.setLanguageProcessor(chatLanguage);

            // Initialize and set predict module for prediction API endpoints
            std::cout << "[API MODE] Initializing Predict module...\n";
            std::unique_ptr<Predict> predictor = std::make_unique<Predict>(verbose);
            apiServer.setPredictModule(std::move(predictor));
            std::cout << "[API MODE] Predict module initialized.\n";

            // Start API Server
            std::cout << "[API MODE] Starting API Server...\n";
            if (!apiServer.start()) {
                std::cerr << "Failed to start API server\n";
                return 1;
            }
            std::cout << "[API MODE] API Server started.\n";

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
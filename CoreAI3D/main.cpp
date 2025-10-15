// D:/code/CoreAI3D/CoreAI3D/main.cpp

// Do NOT define _WIN32_WINNT or WIN32_LEAN_AND_MEAN here.
// They are handled by CMake's target_compile_definitions and _WINSOCKAPI_ in main.hpp.

#include "main.hpp" // This MUST be the first include for your project's headers

#include "include/Core.hpp"
#include "include/Train.hpp"
#include "include/Database.hpp"
#include "include/Language.hpp"

// External library headers (Boost, cURL, MySQL, etc.)
// Include specific Boost headers as needed for their functionality.

namespace po = boost::program_options; // Alias for convenience

int main(int argc, char* argv[]) {
    // --- START DEBUG PRINTS (Argc/Argv) ---
    std::cout << "DEBUG: argc = " << argc << std::endl;
    for (int i = 0; i < argc; ++i) {
        std::cout << "DEBUG: argv[" << i << "] = \"" << argv[i] << "\"" << std::endl;
    }
    std::cout << "DEBUG: End of argv dump." << std::endl;
    // --- END DEBUG PRINTS (Argc/Argv) ---

    po::options_description desc("CoreAI3D Opties");
    desc.add_options()
        ("help,h", "produce help message")
        ("input-file,i", po::value<std::string>(), "Input filename...")
        ("target-file,t", po::value<std::string>()->default_value(""), "Optional: Filename containing separate target values for evaluation (only target columns)")
        ("delimiter,d", po::value<std::string>()->default_value(","), "CSV file delimiter (e.g., ',' or ';')")
        ("samples,s", po::value<std::string>()->default_value("0"), "Number of samples in the dataset (number of rows to process/train on).")
        ("language", po::value<std::string>()->default_value("en"), "A code for the language aka en or nl or ru")
        ("embedding-file", po::value<std::string>()->default_value("embedding.txt"), "the path to your embedding file for text")
        ("epochs,e", po::value<std::string>()->default_value("10"), "Number of training epochs.")
        ("learning-rate,lr", po::value<std::string>()->default_value("0.01"), "Learning rate for the neural network.")
        ("layers,l", po::value<std::string>()->default_value("3"), "Number of hidden layers in the neural network.")
        ("neurons,n", po::value<std::string>()->default_value("10"), "Number of neurons per hidden layer.")
        ("min", po::value<std::string>()->default_value("0.0"), "Minimum value for data normalization.")
        ("max", po::value<std::string>()->default_value("1.0"), "Maximum value for data normalization.")
        ("input-size,iz", po::value<std::string>()->default_value("1"), "Number of input columns (feature values).")
        ("output-size,oz", po::value<std::string>()->default_value("1"), "Number of output columns (target values).")
        ("db-host", po::value<std::string>()->default_value("localhost"), "Database host for MySQL X DevAPI.")
        ("db-port", po::value<std::string>()->default_value("33060"), "Database port for MySQL X DevAPI.")
        ("db-user", po::value<std::string>()->default_value("user"), "Database user for MySQL X DevAPI.")
        ("db-password", po::value<std::string>()->default_value("password"), "Database password for MySQL X DevAPI.")
        ("db-schema", po::value<std::string>()->default_value("coreai_db"), "Database schema name.")
        ("ssl-mode", po::value<std::string>()->default_value("DISABLED"), "SSL mode for database connection (DISABLED, REQUIRED, VERIFY_CA, VERIFY_IDENTITY).")
        ("dataset-name", po::value<std::string>()->default_value("online-1a"), "Name for the dataset")
        ("create-tables", po::bool_switch()->default_value(false), "Create database tables if they don't exist.")
        ("offline", po::bool_switch()->default_value(false), "Run in offline mode (no database connection).")
        ("dataset-id,di", po::value<std::string>()->default_value("-1"), "Specific dataset ID for database operations (load/save model/data).")
        ("output-csv,o", po::value<std::string>()->default_value(""), "Output CSV filename for results (predictions, actuals).")
        ("contains-header", po::bool_switch()->default_value(true), "Specify if the input CSV file contains a header row.")
        ("contains-text", po::bool_switch()->default_value(false), "Specify if the input CSV file contains text data that needs embedding.")
        ("start-chat", po::bool_switch()->default_value(false), "Start a chat with the AI")
        ("start-predict", po::bool_switch()->default_value(false), "Calculate and predict using a CSV file.")
        ("api-port", po::value<std::string>()->default_value("8080"), "Port for the HTTP API server to listen on.");

    po::variables_map vm;
    try {
        std::cout << "DEBUG: Parsing arguments...\n";
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        std::cout << "DEBUG: Arguments parsed successfully.\n";
    }
    catch (const po::error& e) {
        std::cerr << "ERROR: Error parsing arguments: " << e.what() << std::endl;
        std::cerr << desc << std::endl; // Print help message from Boost.Program_options
        return 1;
    }

    // Retrieve arguments and manually convert numeric types
    std::string inputFile = vm["input-file"].as<std::string>();
    std::string targetFile = vm["target-file"].as<std::string>();
    std::string datasetName = vm["dataset-name"].as<std::string>();
    std::string delimiter_str = vm["delimiter"].as<std::string>();
    char delimiter = delimiter_str.empty() ? ',' : delimiter_str[0];

    int numSamples;
    std::string numSamples_str = vm["samples"].as<std::string>();
    try { numSamples = std::stoi(numSamples_str); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --samples: \"" << numSamples_str << "\". Defaulting to 0. Error: " << e.what() << std::endl; numSamples = 0; }

    std::string language = vm["language"].as<std::string>();
    std::string embeddingFile = vm["embedding-file"].as<std::string>();

    int epochs;
    std::string epochs_str = vm["epochs"].as<std::string>();
    try { epochs = std::stoi(epochs_str); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --epochs: \"" << epochs_str << "\". Defaulting to 10. Error: " << e.what() << std::endl; epochs = 10; }

    double learningRate;
    std::string learningRate_str = vm["learning-rate"].as<std::string>();
    try {
        std::replace(learningRate_str.begin(), learningRate_str.end(), ',', '.');
        learningRate = std::stod(learningRate_str);
    }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --learning-rate: \"" << learningRate_str << "\". Defaulting to 0.01. Error: " << e.what() << std::endl; learningRate = 0.01; }

    int layers;
    std::string layers_str = vm["layers"].as<std::string>();
    try { layers = std::stoi(layers_str); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --layers: \"" << layers_str << "\". Defaulting to 3. Error: " << e.what() << std::endl; layers = 3; }

    int neurons;
    std::string neurons_str = vm["neurons"].as<std::string>();
    try { neurons = std::stoi(neurons_str); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --neurons: \"" << neurons_str << "\". Defaulting to 10. Error: " << e.what() << std::endl; neurons = 10; }

    float minRange;
    std::string minRange_str_arg = vm["min"].as<std::string>();
    try {
        std::replace(minRange_str_arg.begin(), minRange_str_arg.end(), ',', '.');
        minRange = std::stof(minRange_str_arg);
    }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --min: \"" << minRange_str_arg << "\". Defaulting to 0.0f. Error: " << e.what() << std::endl; minRange = 0.0f; }

    float maxRange;
    std::string maxRange_str_arg = vm["max"].as<std::string>();
    try {
        std::replace(maxRange_str_arg.begin(), maxRange_str_arg.end(), ',', '.');
        maxRange = std::stof(maxRange_str_arg);
    }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --max: \"" << maxRange_str_arg << "\". Defaulting to 1.0f. Error: " << e.what() << std::endl; maxRange = 1.0f; }

    int inputSize;
    std::string inputSize_str_arg = vm["input-size"].as<std::string>();
    try { inputSize = std::stoi(inputSize_str_arg); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --input-size: \"" << inputSize_str_arg << "\". Defaulting to 1. Error: " << e.what() << std::endl; inputSize = 1; }

    int outputSize;
    std::string outputSize_str_arg = vm["output-size"].as<std::string>();
    try { outputSize = std::stoi(outputSize_str_arg); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --output-size: \"" << outputSize_str_arg << "\". Defaulting to 1. Error: " << e.what() << std::endl; outputSize = 1; }

    bool hasHeader = vm["contains-header"].as<bool>();
    bool containsText = vm["contains-text"].as<bool>();
    bool startChat = vm["start-chat"].as<bool>();
    bool startPredict = vm["start-predict"].as<bool>();

    std::string dbHost = vm["db-host"].as<std::string>();
    int dbPort;
    std::string dbPort_str_arg = vm["db-port"].as<std::string>();
    try { dbPort = std::stoi(dbPort_str_arg); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --db-port: \"" << dbPort_str_arg << "\". Defaulting to 33060. Error: " << e.what() << std::endl; dbPort = 33060; }

    std::string dbUser = vm["db-user"].as<std::string>();
    std::string dbPassword = vm["db-password"].as<std::string>();
    std::string dbSchema = vm["db-schema"].as<std::string>();
    std::string sslModeStr = vm["ssl-mode"].as<std::string>();

    bool createTables = vm["create-tables"].as<bool>();
    bool isOfflineMode = vm["offline"].as<bool>();

    int datasetId;
    std::string datasetId_str_arg = vm["dataset-id"].as<std::string>();
    try { datasetId = std::stoi(datasetId_str_arg); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --dataset-id: \"" << datasetId_str_arg << "\". Defaulting to -1. Error: " << e.what() << std::endl; datasetId = -1; }

    std::string outputCsvFile = vm["output-csv"].as<std::string>();

    int apiPort;
    std::string apiPort_str_arg = vm["api-port"].as<std::string>();
    try { apiPort = std::stoi(apiPort_str_arg); }
    catch (const std::exception& e) { std::cerr << "ERROR: Invalid value for --api-port: \"" << apiPort_str_arg << "\". Defaulting to 8080. Error: " << e.what() << std::endl; apiPort = 8080; }

    std::cout << "DEBUG: All arguments retrieved. Proceeding to logic.\n";

    // Helper strings for interactive input parsing (these are no longer needed for retrieval, but kept for interactive prompts)
    std::string dbPort_str_prompt; // Re-use for interactive prompts
    std::string neuron_str;
    std::string layer_str;
    std::string inputSize_str_prompt;
    std::string outputSize_str_prompt;
    std::string learningRate_str_input; // Renamed to avoid shadowing
    std::string minRange_str_prompt; // For interactive prompt input
    std::string maxRange_str_prompt; // For interactive prompt input
    std::string sample_str;
    std::string epoch_str;
    std::string dataset_str; // For interactive prompt input
    std::string bool_str;

    // Map string to mysqlx::SSLMode enum
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

    // --- Interactive Mode Check ---
    // If neither --start-chat nor --start-predict is provided, prompt the user
    // Also, if --api-port is NOT explicitly used, prompt for interactive mode.
    // Use vm.count() to check if an argument was provided on the command line.
    bool api_port_is_used = vm.count("api-port");
    if (!startChat && !startPredict && !api_port_is_used)
    {
        std::cout << "CoreAI3D Application Menu:\n";
        std::cout << "1. Calculate CSV and Predict Future\n";
        std::cout << "2. Chat with AI\n";
        std::cout << "3. Start API Server\n";
        std::cout << "Enter your choice (1, 2, or 3): ";
        int choice;
        std::cin >> choice;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(),
            '\n'); // Consume newline

        if (choice == 1)
        {
            startPredict = true;
        }
        else if (choice == 2)
        {
            startChat = true;
        }
        else if (choice == 3)
        {
            // API server mode is implicit if no other mode is chosen and no API port is provided via CLI.
            // If the user chose option 3, then it means API server mode.
        }
        else
        {
            std::cerr << "Invalid choice. Exiting." << std::endl;
            return 1;
        }
    }

    // --- Main Logic Branches ---
    try
    {
        if (startChat)
        {
            std::cout << "Starting chat mode...\n";

            // Prompt for missing chat arguments if not provided by CLI
            // Use vm.count() for conditional prompting
            if (!vm.count("db-host")) {
                std::cout << "Enter Hostname of the database: " << std::endl;
                std::getline(std::cin, dbHost);
            }
            if (!vm.count("db-port")) {
                std::cout << "Enter the port of the database: " << std::endl;
                std::getline(std::cin, dbPort_str_prompt); // Use dbPort_str_prompt for interactive input
                try {
                    dbPort = std::stoi(dbPort_str_prompt);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the port. Defaulting to 33060. Error: " << e.what() << std::endl;
                    dbPort = 33060;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 33060. Error: " << e.what() << std::endl;
                    dbPort = 33060;
                }
            }
            if (!vm.count("db-user")) {
                std::cout << "Enter database username: " << std::endl;
                std::getline(std::cin, dbUser);
            }
            if (!vm.count("db-password")) {
                std::cout << "Enter database password: " << std::endl;
                std::getline(std::cin, dbPassword);
            }
            if (!vm.count("db-schema")) {
                std::cout << "Enter database schema: " << std::endl;
                std::getline(std::cin, dbSchema);
            }
            if (!vm.count("ssl-mode")) {
                std::cout << "Enter database SSLMode (DISABLED, REQUIRED, VERIFY_CA, VERIFY_IDENTITY): " << std::endl;
                std::getline(std::cin, sslModeStr);
                // Re-evaluate SSL mode after interactive input
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
                    ssl = SSLMode::DISABLED;
                }
            }
            if (!vm.count("layers")) {
                std::cout << "Enter amount of layers: " << std::endl;
                std::getline(std::cin, layer_str);
                try {
                    layers = std::stoi(layer_str);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the layer. Defaulting to 5. Error: " << e.what() << std::endl;
                    layers = 5;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 5. Error: " << e.what() << std::endl;
                    layers = 5;
                }
            }
            if (!vm.count("neurons")) {
                std::cout << "Enter amount of neurons: " << std::endl;
                std::getline(std::cin, neuron_str);
                try {
                    neurons = std::stoi(neuron_str);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the neuron. Defaulting to 25. Error: " << e.what() << std::endl;
                    neurons = 25;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 25. Error: " << e.what() << std::endl;
                    neurons = 25;
                }
            }
            if (!vm.count("input-size")) {
                std::cout << "Enter inputSize: " << std::endl;
                std::getline(std::cin, inputSize_str_prompt);
                try {
                    inputSize = std::stoi(inputSize_str_prompt);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the inputSize. Defaulting to 1. Error: " << e.what() << std::endl;
                    inputSize = 1;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 1. Error: " << e.what() << std::endl;
                    inputSize = 1;
                }
            }
            if (!vm.count("output-size")) {
                std::cout << "Enter outputSize: " << std::endl;
                std::getline(std::cin, outputSize_str_prompt);
                try {
                    outputSize = std::stoi(outputSize_str_prompt);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the outputSize. Defaulting to 1. Error: " << e.what() << std::endl;
                    outputSize = 1;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 1. Error: " << e.what() << std::endl;
                    outputSize = 1;
                }
            }
            if (!vm.count("learning-rate")) {
                std::cout << "Enter Learning rate: " << std::endl;
                std::getline(std::cin, learningRate_str_input); // Use unique name
                try {
                    learningRate = std::stod(learningRate_str_input); // Use stod for double
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the learningrate. Defaulting to 0.0001. Error: " << e.what() << std::endl;
                    learningRate = 0.0001;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for double type. Defaulting to 0.0001. Error: " << e.what() << std::endl;
                    learningRate = 0.0001;
                }
            }

            // Prompt for missing chat arguments if not provided by CLI
            bool embedding_file_used_flag = vm.count("embedding-file");
            if (embeddingFile == "embedding.txt" && !embedding_file_used_flag)
            {
                std::cout << "Enter path to embedding file (e.g., en_embeddings.csv): ";
                std::getline(std::cin, embeddingFile);
            }
            // Check language only if it's default AND embeddingFile doesn't imply language
            bool language_used_flag = vm.count("language");
            if (language == "en" && !language_used_flag
                && embeddingFile.find(language) == std::string::npos) // Only prompt if default AND not implicitly set by file
            {
                std::cout << "Enter language code (e.g., en, nl, ru): ";
                std::getline(std::cin, language);
            }

            // Assuming embedding dimension is fixed or passed as an argument.
            // For this example, let's assume it's 100 as per previous context.
            int embeddingDimension = 100; // This should match your embedding file's dimension.
            Language langProcessor(embeddingFile, embeddingDimension, dbHost, dbPort, dbUser, dbPassword, dbSchema, 0, language, inputSize, outputSize, layers, neurons);

            // Load embeddings for the specified language.
            std::string actualEmbeddingFile = embeddingFile.empty()
                ? std::string(language) + "_embeddings.csv" // Fixed string concat
                : embeddingFile;
            std::cout << "Loading embeddings from: " << actualEmbeddingFile
                << " for language: " << language << std::endl;
            langProcessor.setCurrentLanguage(language);

            langProcessor.chat(inputFile);
        }
        else if (startPredict)
        {
            std::cout << "[PREDICT MODE] Starting prediction mode...\n";

            // Prompt for missing predict arguments if not provided by CLI
            if (inputFile.empty() && !vm.count("input-file"))
            {
                std::cout << "Enter path to input CSV file: ";
                std::getline(std::cin, inputFile);
            }
            if (outputCsvFile.empty() && !vm.count("output-csv"))
            {
                std::cout << "Enter path to output CSV file for predictions: ";
                std::getline(std::cin, outputCsvFile);
            }
            if (!vm.count("language") && language == "en" // Check if language was default AND not CLI-provided
                && embeddingFile.find(language) == std::string::npos)
            {
                std::cout << "Enter language code (e.g., en, nl, ru): ";
                std::getline(std::cin, language);
            }
            if (!vm.count("embedding-file") && embeddingFile == "embedding.txt") // Check if embedding_file was default AND not CLI-provided
            {
                std::cout << "Enter path to embedding file (e.g., en_embeddings.csv): ";
                std::getline(std::cin, embeddingFile);
            }
            bool offline_used_flag = vm.count("offline"); // Capture result
            if (!offline_used_flag) { // Use the captured flag
                std::cout << "Do you want to run in offline mode (true/false, 1/0, y/n)? ";
                std::getline(std::cin, bool_str);
                std::transform(bool_str.begin(), bool_str.end(), bool_str.begin(), ::tolower);
                isOfflineMode = (bool_str == "true" || bool_str == "1" || bool_str == "y");
            }

            if (!isOfflineMode) {
                if (!vm.count("db-host")) {
                    std::cout << "Enter Hostname of the database: " << std::endl;
                    std::getline(std::cin, dbHost);
                }
                if (!vm.count("db-port")) {
                    std::cout << "Enter the port of the database: " << std::endl;
                    std::getline(std::cin, dbPort_str_prompt);
                    try {
                        dbPort = std::stoi(dbPort_str_prompt);
                    }
                    catch (const std::invalid_argument& e) {
                        std::cerr << "Invalid input: Please enter a valid number for the port. Defaulting to 33060. Error: " << e.what() << std::endl;
                        dbPort = 33060;
                    }
                    catch (const std::out_of_range& e) {
                        std::cerr << "Input number out of range for int type. Defaulting to 33060. Error: " << e.what() << std::endl;
                        dbPort = 33060;
                    }
                }
                if (!vm.count("db-user")) {
                    std::cout << "Enter database username: " << std::endl;
                    std::getline(std::cin, dbUser);
                }
                if (!vm.count("db-password")) {
                    std::cout << "Enter database password: " << std::endl;
                    std::getline(std::cin, dbPassword);
                }
                if (!vm.count("db-schema")) {
                    std::cout << "Enter database schema: " << std::endl;
                    std::getline(std::cin, dbSchema);
                }
                if (!vm.count("ssl-mode")) {
                    std::cout << "Enter database SSLMode (DISABLED, REQUIRED, VERIFY_CA, VERIFY_IDENTITY): " << std::endl;
                    std::getline(std::cin, sslModeStr);
                    // Re-evaluate SSL mode after interactive input
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
                        ssl = SSLMode::DISABLED;
                    }
                }
                if (!vm.count("dataset-name")) {
                    std::cout << "Enter dataset name: " << std::endl;
                    std::getline(std::cin, datasetName);
                }
                if (!vm.count("dataset-id")) {
                    std::cout << "Enter dataset ID (-1 for no specific ID): " << std::endl;
                    std::getline(std::cin, dataset_str);
                    try {
                        datasetId = std::stoi(dataset_str);
                    }
                    catch (const std::invalid_argument& e) {
                        std::cerr << "Invalid input: Please enter a valid number for datasetID. Defaulting to -1. Error: " << e.what() << std::endl;
                        datasetId = -1;
                    }
                    catch (const std::out_of_range& e) {
                        std::cerr << "Input number out of range for int type. Defaulting to -1. Error: " << e.what() << std::endl;
                        datasetId = -1;
                    }
                }
                if (!vm.count("create-tables")) {
                    std::cout << "Create database tables if they don't exist (true/false, 1/0, y/n)? ";
                    std::getline(std::cin, bool_str);
                    std::transform(bool_str.begin(), bool_str.end(), bool_str.begin(), ::tolower);
                    createTables = (bool_str == "true" || bool_str == "1" || bool_str == "y");
                }
            }

            if (!vm.count("delimiter")) {
                std::cout << "Enter delimiter: " << std::endl;
                std::string s_temp; // Use a temp string for input
                std::getline(std::cin, s_temp);
                if (!s_temp.empty()) {
                    delimiter = s_temp[0];
                }
            }
            if (!vm.count("layers")) {
                std::cout << "Enter amount of layers: " << std::endl;
                std::getline(std::cin, layer_str);
                try {
                    layers = std::stoi(layer_str);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the layer. Defaulting to 5. Error: " << e.what() << std::endl;
                    layers = 5;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 5. Error: " << e.what() << std::endl;
                    layers = 5;
                }
            }
            if (!vm.count("neurons")) {
                std::cout << "Enter amount of neurons: " << std::endl;
                std::getline(std::cin, neuron_str);
                try {
                    neurons = std::stoi(neuron_str);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the neuron. Defaulting to 25. Error: " << e.what() << std::endl;
                    neurons = 25;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 25. Error: " << e.what() << std::endl;
                    neurons = 25;
                }
            }
            if (!vm.count("input-size")) {
                std::cout << "Enter inputSize: " << std::endl;
                std::getline(std::cin, inputSize_str_prompt);
                try {
                    inputSize = std::stoi(inputSize_str_prompt);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the inputSize. Defaulting to 1. Error: " << e.what() << std::endl;
                    inputSize = 1;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 1. Error: " << e.what() << std::endl;
                    inputSize = 1;
                }
            }
            if (!vm.count("output-size")) {
                std::cout << "Enter outputSize: " << std::endl;
                std::getline(std::cin, outputSize_str_prompt);
                try {
                    outputSize = std::stoi(outputSize_str_prompt);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the outputSize. Defaulting to 1. Error: " << e.what() << std::endl;
                    outputSize = 1;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 1. Error: " << e.what() << std::endl;
                    outputSize = 1;
                }
            }
            if (!vm.count("learning-rate")) {
                std::cout << "Enter Learning rate: " << std::endl;
                std::getline(std::cin, learningRate_str_input); // Use unique name
                try {
                    learningRate = std::stod(learningRate_str_input); // Use stod for double
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the learningrate. Defaulting to 0.0001. Error: " << e.what() << std::endl;
                    learningRate = 0.0001;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for double type. Defaulting to 0.0001. Error: " << e.what() << std::endl;
                    learningRate = 0.0001;
                }
            }
            if (!vm.count("samples")) {
                std::cout << "Enter number of samples: " << std::endl;
                std::getline(std::cin, sample_str);
                try {
                    numSamples = std::stoi(sample_str);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the samples. Defaulting to 1. Error: " << e.what() << std::endl;
                    numSamples = 1;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 1. Error: " << e.what() << std::endl;
                    numSamples = 1;
                }
            }
            if (!vm.count("epochs")) {
                std::cout << "Enter number of epochs: " << std::endl;
                std::getline(std::cin, epoch_str);
                try {
                    epochs = std::stoi(epoch_str);
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the epochs. Defaulting to 1. Error: " << e.what() << std::endl;
                    epochs = 1;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for int type. Defaulting to 1. Error: " << e.what() << std::endl;
                    epochs = 1;
                }
            }
            if (!vm.count("min")) {
                std::cout << "Enter minimum value for normalization: " << std::endl;
                std::getline(std::cin, minRange_str_prompt); // Use _prompt suffix for interactive
                try {
                    minRange = std::stof(minRange_str_prompt); // Use stof for float
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the minimum value. Defaulting to 0. Error: " << e.what() << std::endl;
                    minRange = 0.0f;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for float type. Defaulting to 0. Error: " << e.what() << std::endl;
                    minRange = 0.0f;
                }
            }
            if (!vm.count("max")) {
                std::cout << "Enter maximum value for normalization: " << std::endl;
                std::getline(std::cin, maxRange_str_prompt); // Use _prompt suffix for interactive
                try {
                    maxRange = std::stof(maxRange_str_prompt); // Use stof for float
                }
                catch (const std::invalid_argument& e) {
                    std::cerr << "Invalid input: Please enter a valid number for the maximum value. Defaulting to 1. Error: " << e.what() << std::endl;
                    maxRange = 1.0f;
                }
                catch (const std::out_of_range& e) {
                    std::cerr << "Input number out of range for float type. Defaulting to 1. Error: " << e.what() << std::endl;
                    maxRange = 1.0f;
                }
            }

            if (!vm.count("contains-header")) {
                std::cout << "Does the file contain headers (true/false, 1/0, y/n)? ";
                std::getline(std::cin, bool_str);
                std::transform(bool_str.begin(), bool_str.end(), bool_str.begin(), ::tolower);
                hasHeader = (bool_str == "true" || bool_str == "1" || bool_str == "y");
            }
            if (!vm.count("contains-text")) {
                std::cout << "Does the file contain text data that needs embedding (true/false, 1/0, y/n)? ";
                std::getline(std::cin, bool_str);
                std::transform(bool_str.begin(), bool_str.end(), bool_str.begin(), ::tolower);
                containsText = (bool_str == "true" || bool_str == "1" || bool_str == "y");
            }

            if (inputFile.empty() || outputCsvFile.empty())
            {
                std::cerr << "Error: Input and output files are required for "
                    "'predict' mode. Exiting.\n";
                return 1;
            }

            std::cout << "[PREDICT MODE] Initializing Training object...\n";
            Training trainer = isOfflineMode
                ? Training(true)
                : Training(dbHost, dbPort, dbUser, dbPassword,
                    dbSchema, 0, createTables);
            std::cout << "[PREDICT MODE] Training object initialized.\n";

            // Set training parameters (important for model structure if loading from DB)
            trainer.layers = layers;
            trainer.embedding_file = embeddingFile;
            trainer.language = language;
            trainer.neurons = neurons;
            trainer.min = minRange;
            trainer.max = maxRange;
            trainer.outputSize = outputSize; // Ensure this is set for backend actions
            trainer.numSamples = numSamples;
            // Set inputSize for trainer, as it's needed for CoreAI initialization later in preprocess
            trainer.inputSize = inputSize;

            // If text is involved, initialize Language processor in trainer
            if (containsText)
            {
                int embeddingDimension = 100; // Define or derive
                std::cout << "[PREDICT MODE] Initializing Language processor...\n";
                trainer.initializeLanguageProcessor(embeddingFile, embeddingDimension, dbHost, dbPort, dbUser, dbPassword, dbSchema, 0, language, inputSize, outputSize, layers, neurons);
                std::cout << "[PREDICT MODE] Language processor initialized. Note: 'contains-text' is true.\n";
            }

            std::cout << "[PREDICT MODE] Loading data from: " << inputFile << std::endl;
            if (!trainer.loadCSV(inputFile, numSamples, outputSize, hasHeader, containsText, delimiter, datasetName)) {
                std::cerr << "Failed to load CSV data for prediction. Exiting." << std::endl;
                return 1;
            }
            std::cout << "[PREDICT MODE] CSV data loaded.\n";


            std::cout << "[PREDICT MODE] Attempting to load model from database (if datasetId is specified)....\n";
            if (datasetId != -1)
            {
                std::cout << "[PREDICT MODE] Using dataset ID " << datasetId
                    << " (likely to load a pre-trained model).\n";
                if (!trainer.loadDatasetFromDB(datasetId)) { // Pass datasetId by value or correct reference
                    std::cerr << "[PREDICT MODE] Failed to load dataset from DB for ID " << datasetId << ". Proceeding without pre-loaded data/model from DB." << std::endl;
                }
                bool model_loaded_success = trainer.loadModel(datasetId); // Capture the bool return value
                if (model_loaded_success) { // Use the captured bool in the conditional
                    std::cout << "[PREDICT MODE] Model loaded from database for ID " << datasetId << ".\n";
                }
                else {
                    std::cerr << "[PREDICT MODE] Failed to load model from DB for ID " << datasetId << ". Will initialize new model for training.\n";
                }
            }
            std::cout << "[PREDICT MODE] Model loading check complete.\n";


            std::cout << "[PREDICT MODE] Preprocessing data (normalization, CoreAI initialization)....\n";
            trainer.preprocess(minRange, maxRange); // This re-normalizes and initializes CoreAI
            std::cout << "[PREDICT MODE] Data preprocessing complete. CoreAI initialized.\n";

            std::cout << "[PREDICT MODE] Starting model training...\n";
            trainer.train(learningRate, epochs);
            std::cout << "[PREDICT MODE] Model training complete.\n";

            std::cout << "[PREDICT MODE] Printing inputs:\n";
            if (trainer.getCore()) {
                trainer.printFullMatrix(trainer.getCore()->getInput(), 25);
            }
            else {
                std::cerr << "CoreAI not initialized for printing inputs. Skipping." << std::endl;
            }
            std::cout << "[PREDICT MODE] Printing outputs:\n";
            if (trainer.getCore()) {
                trainer.printFullMatrix(trainer.getCore()->getOutput(), 5);
            }
            else {
                std::cerr << "CoreAI not initialized for printing outputs. Skipping." << std::endl;
            }
            std::cout << "[PREDICT MODE] Printing results:\n";
            if (trainer.getCore()) {
                trainer.printFullMatrix(trainer.getCore()->getResults(), 5);
            }
            else {
                std::cerr << "CoreAI not initialized for printing results. Skipping." << std::endl;
            }

            std::cout << "[PREDICT MODE] Saving prediction results to: " << outputCsvFile << std::endl;
            if (trainer.saveResultsToCSV(outputCsvFile, inputFile, hasHeader, delimiter)) {
                std::cout << "[PREDICT MODE] Results saved successfully.\n";
            }
            else {
                std::cerr << "[PREDICT MODE] Failed to save results to CSV.\n";
            }


            std::cout << "Prediction mode finished.\n";
        }
        else
        { // Default to API server mode if no other mode is selected
            std::cout << "Starting API server mode...\n";

            // Initialize Training object
            std::cout << "[API MODE] Initializing Training object...\n";
            Training trainer = isOfflineMode
                ? Training(true)
                : Training(dbHost, dbPort, dbUser, dbPassword,
                    dbSchema, 0, createTables);
            std::cout << "[API MODE] Training object initialized.\n";


            // Set training parameters (now that they are members of Training
            // class)
            trainer.layers = layers;
            trainer.neurons = neurons;
            trainer.min = minRange;
            trainer.max = maxRange;
            trainer.outputSize = outputSize; // Ensure this is set for backend actions
            trainer.inputSize = inputSize; // Ensure inputSize is set for CoreAI

            // Initialize CoreAI via trainer's preprocess if you want it managed there
            // Or create it directly here if API server doesn't use the full training pipeline.
            // Assuming the API server will need a CoreAI instance.
            // If CoreAI is only created here, remember it won't be part of the `trainer` object.
            std::cout << "[API MODE] Creating CoreAI instance for API...\n";
            CoreAI core_api_instance(inputSize, layers, neurons, outputSize, minRange, maxRange);
            std::cout << "[API MODE] CoreAI instance created.\n";


            // Initialize Boost.Asio io_context
            net::io_context ioc{ 1 }; // One thread for io_context
            std::cout << "[API MODE] Boost.Asio io_context initialized.\n";

            // Handle signals to gracefully stop the server
            net::signal_set signals(ioc, SIGINT, SIGTERM);
            signals.async_wait([&](beast::error_code const&, int) {
                std::cout << "\nShutting down server...\n";
                ioc.stop(); // Stop the io_context
                });
            std::cout << "[API MODE] Signal handler set.\n";

            // Run the io_context in a separate thread
            std::cout << "[API MODE] Starting server thread...\n";
            std::thread server_thread([&ioc]() { ioc.run(); }); // Corrected call to ioc.run()

            std::cout
                << "CoreAI3D application running and API server active on port "
                << apiPort << ".\n";
            std::cout << "Press Ctrl+C to stop the server.\n";

            server_thread.join(); // Wait for the server thread to finish (e.g., on Ctrl+C)

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
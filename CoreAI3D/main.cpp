#include "Core.hpp"
#include "Language.hpp"
#include "Train.hpp"
#include "main.hpp" // Contains Boost.Asio, mysqlx includes, etc.

int
main(int argc, char* argv[])
{
    argparse::ArgumentParser program("CoreAI3D");

    // Existing arguments - Removed .scan<'x', Type>() as p-ranav/argparse infers type
    program.add_argument("-i", "--input-file")
        .help("Input filename containing matrix data (features + targets) or text for prediction/chat.")
        .default_value(std::string("")); // Default value provides type hint

    program.add_argument("-t", "--target-file")
        .help("Optional: Filename containing separate target values for evaluation (only target columns)")
        .default_value(std::string("")); // Default value provides type hint

    program.add_argument("-d", "--delimiter")
        .help("CSV file delimiter (e.g., ',' or ';')")
        .default_value(std::string(",")); // Default value provides type hint

    program.add_argument("-s", "--num-samples")
        .help("Number of samples in the dataset (number of rows to process/train on).")
        .default_value(-1); // Default value provides type hint (int)

    program.add_argument("--language")
        .help("A code for the language aka en or nl or ru")
        .default_value(std::string("en")); // Default value provides type hint

    program
        .add_argument("--embedding-file")
        .help("the path to your embedding file for text")
        .default_value(std::string("embedding.txt")); // Default value provides type hint

    program.add_argument("-e", "--epochs")
        .help("Number of training epochs.")
        .default_value(10); // Default value provides type hint (int)

    program.add_argument("-lr", "--learning-rate")
        .help("Learning rate for the neural network.")
        .default_value(0.01); // Default value provides type hint (double)

    program.add_argument("-l", "--layers")
        .help("Number of hidden layers in the neural network.")
        .default_value(3); // Default value provides type hint (int)

    program.add_argument("-n", "--neurons")
        .help("Number of neurons per hidden layer.")
        .default_value(10); // Default value provides type hint (int)

    program.add_argument("--min")
        .help("Minimum value for data normalization.")
        .default_value(0.0f); // Default value provides type hint (float)

    program.add_argument("--max")
        .help("Maximum value for data normalization.")
        .default_value(1.0f); // Default value provides type hint (float)

    program.add_argument("-iz", "--input-size")
        .help("Number of input columns (feature values).")
        .default_value(1); // Default value provides type hint (int)

    program.add_argument("-oz", "--output-size")
        .help("Number of output columns (target values).")
        .default_value(1); // Default value provides type hint (int)

    program.add_argument("--db-host")
        .help("Database host for MySQL X DevAPI.")
        .default_value(std::string("localhost"));

    program.add_argument("--db-port")
        .help("Database port for MySQL X DevAPI.")
        .default_value(33060); // Default value provides type hint (int)

    program.add_argument("--db-user")
        .help("Database user for MySQL X DevAPI.")
        .default_value(std::string("user"));

    program.add_argument("--db-password")
        .help("Database password for MySQL X DevAPI.")
        .default_value(std::string("password"));

    program.add_argument("--db-schema")
        .help("Database schema name.")
        .default_value(std::string("coreai_db"));

    program.add_argument("--ssl-mode")
        .help("SSL mode for database connection (DISABLED, REQUIRED, VERIFY_CA, VERIFY_IDENTITY).")
        .default_value(std::string("DISABLED"));

    program.add_argument("--dataset-name")
        .help("Name for the dataset")
        .default_value(std::string("online-1a"));

    program.add_argument("--create-tables")
        .help("Create database tables if they don't exist.")
        .default_value(false)
        .implicit_value(true); // Argument presence implies true

    program.add_argument("--offline")
        .help("Run in offline mode (no database connection).")
        .default_value(false)
        .implicit_value(true); // Argument presence implies true

    program.add_argument("-di", "--dataset-id")
        .help("Specific dataset ID for database operations (load/save model/data).")
        .default_value(-1); // Default value provides type hint (int)

    program.add_argument("-o", "--output-csv")
        .help("Output CSV filename for results (predictions, actuals).")
        .default_value(std::string(""));

    program.add_argument("--contains-header")
        .help("Specify if the input CSV file contains a header row.")
        .default_value(true)
        .implicit_value(true);

    program.add_argument("--contains-text")
        .help("Specify if the input CSV file contains text data that needs embedding.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--start-chat")
        .help("Start a chat with the AI")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--start-predict")
        .help("Calculate and predict using a CSV file.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--api-port")
        .help("Port for the HTTP API server to listen on.")
        .default_value(8080); // Default value provides type hint (int)

    // Parse arguments with better error handling
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << "Error parsing arguments: " << err.what() << std::endl;
        std::cerr << program; // Print help message from argparse
        return 1;
    }

    // Retrieve arguments
    std::string inputFile = program.get<std::string>("--input-file");
    std::string targetFile = program.get<std::string>("--target-file");
    std::string datasetName = program.get<std::string>("--dataset-name");
    std::string delimiter_str = program.get<std::string>("--delimiter");
    char delimiter = delimiter_str.empty() ? ',' : delimiter_str[0];
    int numSamples = program.get<int>("--num-samples");
    std::string language = program.get<std::string>("--language");
    std::string embeddingFile = program.get<std::string>("--embedding-file");
    int epochs = program.get<int>("--epochs");
    double learningRate = program.get<double>("--learning-rate");
    int layers = program.get<int>("--layers");
    int neurons = program.get<int>("--neurons");
    float minRange = program.get<float>("--min");
    float maxRange = program.get<float>("--max");
    int inputSize = program.get<int>("--input-size");
    int outputSize = program.get<int>("--output-size");
    bool hasHeader = program.get<bool>("--contains-header");
    bool containsText = program.get<bool>("--contains-text");
    bool startChat = program.get<bool>("--start-chat");
    bool startPredict = program.get<bool>("--start-predict");

    std::string dbHost = program.get<std::string>("--db-host");
    int dbPort = program.get<int>("--db-port");
    std::string dbUser = program.get<std::string>("--db-user");
    std::string dbPassword = program.get<std::string>("--db-password");
    std::string dbSchema = program.get<std::string>("--db-schema");
    std::string sslModeStr = program.get<std::string>("--ssl-mode");
    bool createTables = program.get<bool>("--create-tables");
    bool isOfflineMode = program.get<bool>("--offline");
    int datasetId = program.get<int>("--dataset-id");
    std::string outputCsvFile = program.get<std::string>("--output-csv");
    int apiPort = program.get<int>("--api-port");

    // Helper strings for interactive input parsing
    std::string dbPort_str;
    std::string neuron_str;
    std::string layer_str;
    std::string inputSize_str;
    std::string outputSize_str;
    std::string learningRate_str_input; // Renamed to avoid shadowing
    std::string minRange_str;
    std::string maxRange_str;
    std::string sample_str;
    std::string epoch_str;
    std::string dataset_str;
    std::string bool_str;

    // Map string to mysqlx::SSLMode enum
    mysqlx::SSLMode ssl = mysqlx::SSLMode::DISABLED; // Default
    if (sslModeStr == "DISABLED") {
        ssl = mysqlx::SSLMode::DISABLED;
    }
    else if (sslModeStr == "REQUIRED") {
        ssl = mysqlx::SSLMode::REQUIRED;
    }
    else if (sslModeStr == "VERIFY_CA") {
        ssl = mysqlx::SSLMode::VERIFY_CA;
    }
    else if (sslModeStr == "VERIFY_IDENTITY") {
        ssl = mysqlx::SSLMode::VERIFY_IDENTITY;
    }
    else {
        std::cerr << "Warning: Unrecognized SSL mode '" << sslModeStr
            << "'. Defaulting to DISABLED." << std::endl;
        ssl = mysqlx::SSLMode::DISABLED; // Ensure ssl is set to a valid default
    }

    // --- Interactive Mode Check ---
    // If neither --start-chat nor --start-predict is provided, prompt the user
    // Also, if --api-port is NOT explicitly used, prompt for interactive mode.
    if (!startChat && !startPredict && !program.is_used("--api-port"))
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
            if (!program.is_used("--db-host")) {
                std::cout << "Enter Hostname of the database: " << std::endl;
                std::getline(std::cin, dbHost);
            }
            if (!program.is_used("--db-port")) {
                std::cout << "Enter the port of the database: " << std::endl;
                std::getline(std::cin, dbPort_str);
                try {
                    dbPort = std::stoi(dbPort_str);
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
            if (!program.is_used("--db-user")) {
                std::cout << "Enter database username: " << std::endl;
                std::getline(std::cin, dbUser);
            }
            if (!program.is_used("--db-password")) {
                std::cout << "Enter database password: " << std::endl;
                std::getline(std::cin, dbPassword);
            }
            if (!program.is_used("--db-schema")) {
                std::cout << "Enter database schema: " << std::endl;
                std::getline(std::cin, dbSchema);
            }
            if (!program.is_used("--ssl-mode")) {
                std::cout << "Enter database SSLMode (DISABLED, REQUIRED, VERIFY_CA, VERIFY_IDENTITY): " << std::endl;
                std::getline(std::cin, sslModeStr);
                // Re-evaluate SSL mode after interactive input
                if (sslModeStr == "DISABLED") {
                    ssl = mysqlx::SSLMode::DISABLED;
                }
                else if (sslModeStr == "REQUIRED") {
                    ssl = mysqlx::SSLMode::REQUIRED;
                }
                else if (sslModeStr == "VERIFY_CA") {
                    ssl = mysqlx::SSLMode::VERIFY_CA;
                }
                else if (sslModeStr == "VERIFY_IDENTITY") {
                    ssl = mysqlx::SSLMode::VERIFY_IDENTITY;
                }
                else {
                    std::cerr << "Warning: Unrecognized SSL mode '" << sslModeStr
                        << "'. Defaulting to DISABLED." << std::endl;
                    ssl = mysqlx::SSLMode::DISABLED;
                }
            }
            if (!program.is_used("--layers")) {
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
            if (!program.is_used("--neurons")) {
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
            if (!program.is_used("--input-size")) {
                std::cout << "Enter inputSize: " << std::endl;
                std::getline(std::cin, inputSize_str);
                try {
                    inputSize = std::stoi(inputSize_str);
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
            if (!program.is_used("--output-size")) {
                std::cout << "Enter outputSize: " << std::endl;
                std::getline(std::cin, outputSize_str);
                try {
                    outputSize = std::stoi(outputSize_str);
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
            if (!program.is_used("--learning-rate")) {
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
            if (embeddingFile == "embedding.txt" && !program.is_used("--embedding-file"))
            {
                std::cout << "Enter path to embedding file (e.g., en_embeddings.csv): ";
                std::getline(std::cin, embeddingFile);
            }
            // Check language only if it's default AND embeddingFile doesn't imply language
            if (language == "en" && !program.is_used("--language")
                && embeddingFile.find(language) == std::string::npos)
            {
                std::cout << "Enter language code (e.g., en, nl, ru): ";
                std::getline(std::cin, language);
            }

            // Assuming embedding dimension is fixed or passed as an argument.
            // For this example, let's assume it's 100 as per previous context.
            int embeddingDimension = 100; // This should match your embedding file's dimension.
            Language langProcessor(embeddingFile, embeddingDimension, dbHost, dbPort, dbUser, dbPassword, dbSchema, ssl, language, inputSize, outputSize, layers, neurons);

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
            // Prediction mode for CSV file
            std::cout << "Starting prediction mode...\n";

            // Prompt for missing predict arguments if not provided by CLI
            if (inputFile.empty() && !program.is_used("--input-file"))
            {
                std::cout << "Enter path to input CSV file: ";
                std::getline(std::cin, inputFile);
            }
            if (outputCsvFile.empty() && !program.is_used("--output-csv"))
            {
                std::cout << "Enter path to output CSV file for predictions: ";
                std::getline(std::cin, outputCsvFile);
            }
            if (language == "en" && !program.is_used("--language")
                && embeddingFile.find(language) == std::string::npos)
            {
                std::cout << "Enter language code (e.g., en, nl, ru): ";
                std::getline(std::cin, language);
            }
            if (embeddingFile == "embedding.txt" && !program.is_used("--embedding-file"))
            {
                std::cout << "Enter path to embedding file (e.g., en_embeddings.csv): ";
                std::getline(std::cin, embeddingFile);
            }
            if (!program.is_used("--offline")) {
                std::cout << "Do you want to run in offline mode (true/false, 1/0, y/n)? ";
                std::getline(std::cin, bool_str);
                std::transform(bool_str.begin(), bool_str.end(), bool_str.begin(), ::tolower);
                isOfflineMode = (bool_str == "true" || bool_str == "1" || bool_str == "y");
            }

            if (!isOfflineMode) {
                if (!program.is_used("--db-host")) {
                    std::cout << "Enter Hostname of the database: " << std::endl;
                    std::getline(std::cin, dbHost);
                }
                if (!program.is_used("--db-port")) {
                    std::cout << "Enter the port of the database: " << std::endl;
                    std::getline(std::cin, dbPort_str);
                    try {
                        dbPort = std::stoi(dbPort_str);
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
                if (!program.is_used("--db-user")) {
                    std::cout << "Enter database username: " << std::endl;
                    std::getline(std::cin, dbUser);
                }
                if (!program.is_used("--db-password")) {
                    std::cout << "Enter database password: " << std::endl;
                    std::getline(std::cin, dbPassword);
                }
                if (!program.is_used("--db-schema")) {
                    std::cout << "Enter database schema: " << std::endl;
                    std::getline(std::cin, dbSchema);
                }
                if (!program.is_used("--ssl-mode")) {
                    std::cout << "Enter database SSLMode (DISABLED, REQUIRED, VERIFY_CA, VERIFY_IDENTITY): " << std::endl;
                    std::getline(std::cin, sslModeStr);
                    // Re-evaluate SSL mode after interactive input
                    if (sslModeStr == "DISABLED") {
                        ssl = mysqlx::SSLMode::DISABLED;
                    }
                    else if (sslModeStr == "REQUIRED") {
                        ssl = mysqlx::SSLMode::REQUIRED;
                    }
                    else if (sslModeStr == "VERIFY_CA") {
                        ssl = mysqlx::SSLMode::VERIFY_CA;
                    }
                    else if (sslModeStr == "VERIFY_IDENTITY") {
                        ssl = mysqlx::SSLMode::VERIFY_IDENTITY;
                    }
                    else {
                        std::cerr << "Warning: Unrecognized SSL mode '" << sslModeStr
                            << "'. Defaulting to DISABLED." << std::endl;
                        ssl = mysqlx::SSLMode::DISABLED;
                    }
                }
                if (!program.is_used("--dataset-name")) {
                    std::cout << "Enter dataset name: " << std::endl;
                    std::getline(std::cin, datasetName);
                }
                if (!program.is_used("--dataset-id")) {
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
                if (!program.is_used("--create-tables")) {
                    std::cout << "Create database tables if they don't exist (true/false, 1/0, y/n)? ";
                    std::getline(std::cin, bool_str);
                    std::transform(bool_str.begin(), bool_str.end(), bool_str.begin(), ::tolower);
                    createTables = (bool_str == "true" || bool_str == "1" || bool_str == "y");
                }
            }

            if (!program.is_used("--delimiter")) {
                std::cout << "Enter delimiter: " << std::endl;
                std::string s_temp; // Use a temp string for input
                std::getline(std::cin, s_temp);
                if (!s_temp.empty()) {
                    delimiter = s_temp[0];
                }
            }
            if (!program.is_used("--layers")) {
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
            if (!program.is_used("--neurons")) {
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
            if (!program.is_used("--input-size")) {
                std::cout << "Enter inputSize: " << std::endl;
                std::getline(std::cin, inputSize_str);
                try {
                    inputSize = std::stoi(inputSize_str);
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
            if (!program.is_used("--output-size")) {
                std::cout << "Enter outputSize: " << std::endl;
                std::getline(std::cin, outputSize_str);
                try {
                    outputSize = std::stoi(outputSize_str);
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
            if (!program.is_used("--learning-rate")) {
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
            if (!program.is_used("--num-samples")) {
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
            if (!program.is_used("--epochs")) {
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
            if (!program.is_used("--min")) {
                std::cout << "Enter minimum value for normalization: " << std::endl;
                std::getline(std::cin, minRange_str);
                try {
                    minRange = std::stof(minRange_str); // Use stof for float
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
            if (!program.is_used("--max")) {
                std::cout << "Enter maximum value for normalization: " << std::endl;
                std::getline(std::cin, maxRange_str);
                try {
                    maxRange = std::stof(maxRange_str); // Use stof for float
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

            if (!program.is_used("--contains-header")) {
                std::cout << "Does the file contain headers (true/false, 1/0, y/n)? ";
                std::getline(std::cin, bool_str);
                std::transform(bool_str.begin(), bool_str.end(), bool_str.begin(), ::tolower);
                hasHeader = (bool_str == "true" || bool_str == "1" || bool_str == "y");
            }
            if (!program.is_used("--contains-text")) {
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

            // Initialize Training object (which likely handles model loading and
            // prediction logic)
            Training trainer = isOfflineMode
                ? Training(true)
                : Training(dbHost, dbPort, dbUser, dbPassword,
                    dbSchema, ssl, createTables);

            // Set training parameters (important for model structure if loading
            // from DB)
            trainer.layers = layers;
            trainer.embedding_file = embeddingFile;
            trainer.language = language;
            trainer.neurons = neurons;
            trainer.min = minRange;
            trainer.max = maxRange;
            trainer.outputSize = outputSize; // Ensure this is set for backend actions
            trainer.numSamples = numSamples;

            // If text is involved, initialize Language processor in trainer
            if (containsText)
            {
                int embeddingDimension = 100; // Define or derive
                trainer.initializeLanguageProcessor(embeddingFile, embeddingDimension, dbHost, dbPort, dbUser, dbPassword, dbSchema, ssl, language, inputSize, outputSize, layers, neurons);
                std::cout << "Note: 'contains-text' is true. Language processor initialized." << std::endl;
            }

            std::cout << "Loading data from: " << inputFile << std::endl;
            std::cout << "Attempting to predict and save results to: "
                << outputCsvFile << std::endl;
            if (datasetId != -1)
            {
                std::cout << "Using dataset ID " << datasetId
                    << " (likely to load a pre-trained model).\n";
                trainer.loadDatasetFromDB(datasetId); // Pass datasetId by value or correct reference
                trainer.loadModel(datasetId); // Load model after loading dataset
            }

            // After data is loaded/model is loaded, then preprocess
            if (!trainer.loadCSV(inputFile, numSamples, outputSize, hasHeader, containsText, delimiter, datasetName)) {
                std::cerr << "Failed to load CSV data for prediction. Exiting." << std::endl;
                return 1;
            }

            // Now that data is loaded and potentially normalized internally by loadCSV, preprocess
            trainer.preprocess(minRange, maxRange); // This re-normalizes and initializes CoreAI

            trainer.train(learningRate, epochs);
            std::cout << "Printing inputs" << std::endl;
            // Use trainer.getCore()->getInput() to access the CoreAI's input data
            if (trainer.getCore()) {
                trainer.printFullMatrix(trainer.getCore()->getInput(), 25);
            }
            else {
                std::cerr << "CoreAI not initialized for printing inputs." << std::endl;
            }
            std::cout << "Printing outputs" << std::endl;
            if (trainer.getCore()) {
                trainer.printFullMatrix(trainer.getCore()->getOutput(), 5);
            }
            else {
                std::cerr << "CoreAI not initialized for printing outputs." << std::endl;
            }
            std::cout << "Printing results" << std::endl;
            if (trainer.getCore()) {
                trainer.printFullMatrix(trainer.getCore()->getResults(), 5);
            }
            else {
                std::cerr << "CoreAI not initialized for printing results." << std::endl;
            }

            // Save results to CSV
            trainer.saveResultsToCSV(outputCsvFile, inputFile, hasHeader, delimiter);


            std::cout << "Prediction mode finished.\n";
        }
        else
        { // Default to API server mode if no other mode is selected
            std::cout << "Starting API server mode...\n";

            // Initialize Training object
            Training trainer = isOfflineMode
                ? Training(true)
                : Training(dbHost, dbPort, dbUser, dbPassword,
                    dbSchema, ssl, createTables);

            // Set training parameters (now that they are members of Training
            // class)
            trainer.layers = layers;
            trainer.neurons = neurons;
            trainer.min = minRange;
            trainer.max = maxRange;
            trainer.outputSize
                = outputSize; // Ensure this is set for backend actions

            // Initialize CoreAI via trainer's preprocess if you want it managed there
            // Or create it directly here if API server doesn't use the full training pipeline.
            // Assuming the API server will need a CoreAI instance.
            // If CoreAI is only created here, remember it won't be part of the `trainer` object.
            CoreAI core_api_instance(inputSize, layers, neurons, outputSize, minRange, maxRange);


            // Initialize Boost.Asio io_context
            net::io_context ioc{ 1 }; // One thread for io_context

            // Handle signals to gracefully stop the server
            net::signal_set signals(ioc, SIGINT, SIGTERM);
            signals.async_wait([&](beast::error_code const&, int) {
                std::cout << "\nShutting down server...\n";
                ioc.stop(); // Stop the io_context
                });

            // Run the io_context in a separate thread
            std::thread server_thread([&ioc]() { ioc.run(); }); // Corrected call to ioc.run()

            std::cout
                << "CoreAI3D application running and API server active on port "
                << apiPort << ".\n";
            std::cout << "Press Ctrl+C to stop the server.\n";

            server_thread.join(); // Wait for the server thread to finish (e.g., on Ctrl+C)

            std::cout << "Application gracefully exited.\n";
        }
    }
    catch (const mysqlx::Error& err)
    {
        std::cerr << "MySQL Error: " << err.what() << std::endl;
        return 1;
    }
    catch (const std::runtime_error& err)
    {
        std::cerr << "Runtime Error: " << err.what() << std::endl;
        return 1;
    }
    catch (const std::exception& err)
    {
        std::cerr << "General Error: " << err.what() << std::endl;
        return 1;
    }

    return 0;
}

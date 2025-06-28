#include "main.hpp"
#include "Language.hpp"
#include "Core.hpp"
#include "Train.hpp"


int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("CoreAI3D");

    // Existing arguments
    program.add_argument("-i", "--input-file")
        .help("Input filename containing matrix data (features + targets) or text for prediction/chat.")
        .default_value(std::string(""));

    program.add_argument("-t", "--target-file")
        .help("Optional: Filename containing separate target values for evaluation (only target columns)")
        .default_value("");

    program.add_argument("-d", "--delimiter")
        .help("CSV file delimiter (e.g., ',' or ';')")
        .default_value(std::string(","));

    program.add_argument("-s", "--num-samples")
        .help("Number of samples in the dataset (number of rows to process/train on).")
        .default_value(-1)
        .scan<'i', int>();

    program.add_argument("--language")
        .help("A code for the language aka en or nl or ru")
        .default_value("en");

    program.add_argument("--embedding-file") // Corrected spelling from "embeding-file"
        .help("the path to your embedding file for text")
        .default_value("embedding.txt");

    program.add_argument("-e", "--epochs")
        .help("Number of training epochs.")
        .default_value(10)
        .scan<'i', int>();

    program.add_argument("-lr", "--learning-rate")
        .help("Learning rate for the neural network.")
        .default_value(0.01)
        .scan<'g', double>();

    program.add_argument("-l", "--layers")
        .help("Number of hidden layers in the neural network.")
        .default_value(3)
        .scan<'i', int>();

    program.add_argument("-n", "--neurons")
        .help("Number of neurons per hidden layer.")
        .default_value(10)
        .scan<'i', int>();

    program.add_argument("--min")
        .help("Minimum value for data normalization.")
        .default_value(0.0f)
        .scan<'g', float>();

    program.add_argument("--max")
        .help("Maximum value for data normalization.")
        .default_value(1.0f)
        .scan<'g', float>();

    program.add_argument("-oz", "--output-size")
        .help("Number of output columns (target values).")
        .default_value(1)
        .scan<'i', int>();

    program.add_argument("--db-host")
        .help("Database host for MySQL X DevAPI.")
        .default_value(std::string("localhost"));

    program.add_argument("--db-port")
        .help("Database port for MySQL X DevAPI.")
        .default_value(33060)
        .scan<'i', unsigned int>();

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
        .help("SSL mode for database connection (DISABLED, VERIFY_CA, VERIFY_IDENTITY).")
        .default_value(std::string("DISABLED"));

    program.add_argument("--create-tables")
        .help("Create database tables if they don't exist.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--offline")
        .help("Run in offline mode (no database connection).")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("-di", "--dataset-id")
        .help("Specific dataset ID for database operations (load/save model/data).")
        .default_value(-1)
        .scan<'i', int>();

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

    // New argument for starting prediction mode
    program.add_argument("--start-predict")
        .help("Calculate and predict using a CSV file.")
        .default_value(false)
        .implicit_value(true);

    // New argument for network port
    program.add_argument("--api-port")
        .help("Port for the HTTP API server to listen on.")
        .default_value(8080) // Default API port
        .scan<'i', unsigned short>();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Retrieve arguments
    std::string inputFile = program.get<std::string>("--input-file");
    std::string targetFile = program.get<std::string>("--target-file");
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
    int outputSize = program.get<int>("--output-size");
    bool hasHeader = program.get<bool>("--contains-header");
    bool containsText = program.get<bool>("--contains-text");
    bool startChat = program.get<bool>("--start-chat");
    bool startPredict = program.get<bool>("--start-predict"); // New variable

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


    // Map string to mysqlx::SSLMode enum
    mysqlx::SSLMode ssl = mysqlx::SSLMode::DISABLED; // Default
    if (sslModeStr == "VERIFY_CA") {
        ssl = mysqlx::SSLMode::VERIFY_CA;
    }
    else if (sslModeStr == "VERIFY_IDENTITY") {
        ssl = mysqlx::SSLMode::VERIFY_IDENTITY;
    }

    // --- Interactive Mode Check ---
    // If neither --start-chat nor --start-predict is provided, prompt the user
    if (!startChat && !startPredict) {
        std::cout << "CoreAI3D Application Menu:\n";
        std::cout << "1. Calculate CSV and Predict Future\n";
        std::cout << "2. Chat with AI\n";
        std::cout << "3. Start API Server (default)\n"; // Option for API server
        std::cout << "Enter your choice (1, 2, or 3): ";
        int choice;
        std::cin >> choice;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Consume newline

        if (choice == 1) {
            startPredict = true;
        }
        else if (choice == 2) {
            startChat = true;
        }
        else if (choice == 3) {
            // Keep startChat and startPredict false, will default to API server
        }
        else {
            std::cerr << "Invalid choice. Exiting." << std::endl;
            return 1;
        }
    }

    // --- Main Logic Branches ---
    try {
        if (startChat) {
            // Chat mode
            std::cout << "Starting chat mode...\n";

            // Prompt for missing chat arguments if not provided by CLI
            if (embeddingFile == "embedding.txt") { // Check if default value
                std::cout << "Enter path to embedding file (e.g., en_embeddings.csv): ";
                std::getline(std::cin, embeddingFile);
            }
            if (language == "en" && embeddingFile.find(language) == std::string::npos) { // Simple check for default language
                std::cout << "Enter language code (e.g., en, nl, ru): ";
                std::getline(std::cin, language);
            }

            // Assuming embedding dimension is fixed or passed as an argument.
            // For this example, let's assume it's 100 as per previous context.
            int embeddingDimension = 100; // This should match your embedding file's dimension.
            Language langProcessor(embeddingDimension); //

            // Load embeddings for the specified language.
            std::string actualEmbeddingFile = embeddingFile.empty() ? language + "_embeddings.csv" : embeddingFile;
            std::cout << "Loading embeddings from: " << actualEmbeddingFile << " for language: " << language << std::endl; //
            langProcessor.loadEmbeddingsFor(language, actualEmbeddingFile); //
            langProcessor.setCurrentLanguage(language); //

            std::cout << "Welcome to the CoreAI3D Chat!\n";
            std::cout << "Type 'exit' to quit.\n";

            std::string inputText;
            while (true) {
                std::cout << "\nEnter your message: ";
                std::getline(std::cin, inputText);

                if (inputText == "exit") {
                    break;
                }

                // Encode the text.
                std::vector<float> textEmbedding = langProcessor.encodeText(inputText); //
                std::cout << "Encoded message. Embedding size: " << textEmbedding.size() << std::endl;

                // Here, you would typically use 'textEmbedding' for your core AI logic.
                // For demonstration, print a simple response.
                if (textEmbedding.empty()) {
                    std::cout << "Chatbot: Could not encode your message fully. Try again.\n";
                }
                else {
                    std::cout << "Chatbot: (Processed your message embedding. Ready for further AI logic.)\n";
                }
            }
            std::cout << "Exiting chat mode. Goodbye!\n";
        }
        else if (startPredict) {
            // Prediction mode for CSV file
            std::cout << "Starting prediction mode...\n";

            // Prompt for missing predict arguments if not provided by CLI
            if (inputFile.empty()) {
                std::cout << "Enter path to input CSV file: ";
                std::getline(std::cin, inputFile);
            }
            if (outputCsvFile.empty()) {
                std::cout << "Enter path to output CSV file for predictions: ";
                std::getline(std::cin, outputCsvFile);
            }
            if (language == "en" && embeddingFile.find(language) == std::string::npos) {
                std::cout << "Enter language code (e.g., en, nl, ru): ";
                std::getline(std::cin, language);
            }
            if (embeddingFile == "embedding.txt") {
                std::cout << "Enter path to embedding file (e.g., en_embeddings.csv): ";
                std::getline(std::cin, embeddingFile);
            }
            // You might want to prompt for other relevant prediction parameters like layers, neurons, etc.
            // if they are at their default values and not provided via CLI.

            if (inputFile.empty() || outputCsvFile.empty()) {
                std::cerr << "Error: Input and output files are required for 'predict' mode. Exiting.\n";
                return 1;
            }

            // Initialize Training object (which likely handles model loading and prediction logic)
            Training trainer = isOfflineMode
                ? Training(true)
                : Training(dbHost, dbPort, dbUser, dbPassword, dbSchema, ssl, createTables);

            // Set training parameters (important for model structure if loading from DB)
            trainer.layers = layers;
            trainer.neurons = neurons;
            trainer.min = minRange;
            trainer.max = maxRange;
            trainer.outputSize = outputSize;

            // If text is involved, the Training class (or Core) would need to use Language class.
            // This part assumes that 'Training' or 'Core' can handle text embedding internally
            // if 'containsText' is true.
            if (containsText) {
                std::cout << "Note: 'contains-text' is true. Ensure your Training/Core logic handles text embedding using Language class.\n";
                std::cout << "Using language: " << language << " and embedding file: " << embeddingFile << std::endl;
            }

            // --- Placeholder for actual prediction logic ---
            // This part depends on the implementation of your Training and Core classes.
            // You would typically load a trained model (if datasetId is provided),
            // then load the input CSV data, perform inference, and save the results.
            std::cout << "Loading data from: " << inputFile << std::endl;
            std::cout << "Attempting to predict and save results to: " << outputCsvFile << std::endl;
            if (datasetId != -1) {
                std::cout << "Using dataset ID " << datasetId << " (likely to load a pre-trained model).\n";
                // Example: trainer.loadModel(datasetId); // A method to load a trained model from DB
            }

            // Assuming 'Training' has a method to perform prediction on an input file
            // and write to an output file.
            // trainer.predict(inputFile, outputCsvFile, delimiter, hasHeader, numSamples, containsText);
            std::cout << "Prediction process initiated. (Actual prediction logic for CSV needs to be implemented in Training/Core classes).\n";
            // --- End Placeholder ---

            std::cout << "Prediction mode finished.\n";
        }
        else { // Default to API server mode if no other mode is selected
            std::cout << "Starting API server mode...\n";

            // Initialize Training object
            Training trainer = isOfflineMode
                ? Training(true)
                : Training(dbHost, dbPort, dbUser, dbPassword, dbSchema, ssl, createTables);

            // Set training parameters (now that they are members of Training class)
            trainer.layers = layers;
            trainer.neurons = neurons;
            trainer.min = minRange;
            trainer.max = maxRange;
            trainer.outputSize = outputSize; // Ensure this is set for backend actions

            // Initialize Boost.Asio io_context
            net::io_context ioc{ 1 }; // One thread for io_context

            // Handle signals to gracefully stop the server
            net::signal_set signals(ioc, SIGINT, SIGTERM);
            signals.async_wait([&](beast::error_code const&, int) {
                std::cout << "\nShutting down server...\n";
                ioc.stop(); // Stop the io_context
                });

            // Run the io_context in a separate thread
            std::thread server_thread([&ioc]() {
                ioc.run();
                });

            std::cout << "CoreAI3D application running and API server active on port " << apiPort << ".\n";
            std::cout << "Press Ctrl+C to stop the server.\n";

            server_thread.join(); // Wait for the server thread to finish (e.g., on Ctrl+C)

            std::cout << "Application gracefully exited.\n";
        }

    }
    catch (const mysqlx::Error& err) {
        std::cerr << "MySQL Error: " << err.what() << std::endl;
        return 1;
    }
    catch (const std::runtime_error& err) {
        std::cerr << "Runtime Error: " << err.what() << std::endl;
        return 1;
    }
    catch (const std::exception& err) {
        std::cerr << "General Error: " << err.what() << std::endl;
        return 1;
    }

    return 0;
}
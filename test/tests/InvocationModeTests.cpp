// test/tests/InvocationModeTests.cpp
#define CATCH_CONFIG_FAST_COMPILE
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <boost/process.hpp>
#include <boost/filesystem.hpp>

#include "CoreAI3DCommon.hpp"

// Helper function to simulate argument parsing logic from main.cpp
struct ParsedArgs {
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

    // Computed values
    bool apiMode = false;
    int apiPort_val = 8080;
    int dbPort_val = 3306;
    int epochs_val = 10;
    double learningRate_val = 0.01;
    int layers_val = 3;
    int neurons_val = 10;
    float minRange_val = 0.0f;
    float maxRange_val = 1.0f;
    int inputSize_val = 1;
    int outputSize_val = 1;
    int datasetId_val = -1;
    int numSamples_val = 0;
    int batchSize_val = 32;
    int threads_val = 1;
};

ParsedArgs parseArguments(const std::vector<std::string>& args) {
    ParsedArgs result;

    // Helper function to check if next argument is a value
    auto hasValue = [&](size_t idx) -> bool {
        return idx + 1 < args.size() && args[idx + 1].find("--") != 0;
    };

    // Helper function to extract value from argument
    auto extractValue = [&](size_t& idx, const std::string& argName, const std::string& arg) -> std::string {
        // Check if value is in the same argument after '='
        size_t equalsPos = arg.find('=');
        if (equalsPos != std::string::npos) {
            return arg.substr(equalsPos + 1);
        }
        // Otherwise, expect the next argument to be the value
        if (hasValue(idx)) {
            return args[++idx];
        }
        throw std::runtime_error(argName + " requires a value");
    };

    // Parse arguments
    for (size_t i = 1; i < args.size(); ++i) {
        std::string arg = args[i];
        if (arg == "--verbose" || arg == "-v") {
            result.verbose = true;
        } else if (arg.find("--input-file") == 0 || arg == "-i") {
            std::string value = extractValue(i, "--input-file", arg);
            if (!value.empty()) result.inputFile = value;
        } else if (arg.find("--output-csv") == 0 || arg == "-o") {
            std::string value = extractValue(i, "--output-csv", arg);
            if (!value.empty()) result.outputCsvFile = value;
        } else if (arg.find("--api-port") == 0) {
            std::string value = extractValue(i, "--api-port", arg);
            if (!value.empty()) result.apiPort = value;
        } else if (arg == "--start-predict") {
            result.startPredict = true;
        } else if (arg == "--start-chat") {
            result.startChat = true;
        } else if (arg.find("--load-model") == 0) {
            std::string value = extractValue(i, "--load-model", arg);
            if (!value.empty()) result.loadModelId = value;
        } else if (arg.find("--embedding-file") == 0) {
            std::string value = extractValue(i, "--embedding-file", arg);
            if (!value.empty()) result.embeddingFile = value;
        } else if (arg == "--offline") {
            result.isOfflineMode = true;
        } else if (arg == "--enable-websocket") {
            result.enableWebsocket = true;
        } else if (arg == "--create-table" || arg == "--create-tables") {
            result.createTableFlag = true;
        } else if (arg.find("--db-host") == 0) {
            std::string value = extractValue(i, "--db-host", arg);
            if (!value.empty()) result.dbHost = value;
        } else if (arg.find("--db-user") == 0) {
            std::string value = extractValue(i, "--db-user", arg);
            if (!value.empty()) result.dbUser = value;
        } else if (arg.find("--db-password") == 0) {
            std::string value = extractValue(i, "--db-password", arg);
            if (!value.empty()) result.dbPassword = value;
        } else if (arg.find("--db-port") == 0) {
            std::string value = extractValue(i, "--db-port", arg);
            if (!value.empty()) result.dbPort = value;
        } else if (arg.find("--db-schema") == 0) {
            std::string value = extractValue(i, "--db-schema", arg);
            if (!value.empty()) result.dbSchema = value;
        } else if (arg.find("--target") == 0 || arg == "-t") {
            std::string value = extractValue(i, "--target", arg);
            if (!value.empty()) result.targetFile = value;
        } else if (arg.find("--layers") == 0 || arg == "-l") {
            std::string value = extractValue(i, "--layers", arg);
            if (!value.empty()) result.layers = value;
        } else if (arg.find("--samples") == 0 || arg == "-s") {
            std::string value = extractValue(i, "--samples", arg);
            if (!value.empty()) result.samples = value;
        } else if (arg.find("--epochs") == 0 || arg == "-e") {
            std::string value = extractValue(i, "--epochs", arg);
            if (!value.empty()) result.epochs = value;
        } else if (arg.find("--neurons") == 0 || arg == "-n") {
            std::string value = extractValue(i, "--neurons", arg);
            if (!value.empty()) result.neurons = value;
        } else if (arg.find("--learning-rate") == 0 || arg == "-r") {
            std::string value = extractValue(i, "--learning-rate", arg);
            if (!value.empty()) result.learningRate = value;
        } else if (arg.find("--min") == 0) {
            std::string value = extractValue(i, "--min", arg);
            if (!value.empty()) result.minRange = value;
        } else if (arg.find("--max") == 0) {
            std::string value = extractValue(i, "--max", arg);
            if (!value.empty()) result.maxRange = value;
        } else if (arg.find("--input-size") == 0 || arg == "--inputsize") {
            std::string value = extractValue(i, "--input-size", arg);
            if (!value.empty()) result.inputSize = value;
        } else if (arg.find("--output-size") == 0 || arg == "--outputsize") {
            std::string value = extractValue(i, "--output-size", arg);
            if (!value.empty()) result.outputSize = value;
        } else if (arg.find("--batch-size") == 0 || arg == "-b") {
            std::string value = extractValue(i, "--batch-size", arg);
            if (!value.empty()) result.batchSize = value;
        } else if (arg == "--calculus-activation" || arg == "-C") {
            result.calculusActivation = true;
        } else if (arg == "--contains-text") {
            result.containsTextFlag = true;
        } else if (arg == "--contains-header") {
            result.containsHeader = true;
        } else if (arg == "--contains-date") {
            result.containsDate = true;
        } else if (arg.find("--threads") == 0 || arg == "-j") {
            std::string value = extractValue(i, "--threads", arg);
            if (!value.empty()) result.threads = value;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    // Convert string arguments to appropriate types
    try { result.numSamples_val = std::stoi(result.samples); }
    catch (...) { result.numSamples_val = 0; }

    try { result.epochs_val = std::stoi(result.epochs); }
    catch (...) { result.epochs_val = 10; }

    try {
        std::string lr_str = result.learningRate;
        std::replace(lr_str.begin(), lr_str.end(), ',', '.');
        result.learningRate_val = std::stod(lr_str);
    }
    catch (...) { result.learningRate_val = 0.01; }

    try { result.layers_val = std::stoi(result.layers); }
    catch (...) { result.layers_val = 3; }

    try { result.neurons_val = std::stoi(result.neurons); }
    catch (...) { result.neurons_val = 10; }

    try {
        std::string min_str = result.minRange;
        std::replace(min_str.begin(), min_str.end(), ',', '.');
        result.minRange_val = std::stof(min_str);
    }
    catch (...) { result.minRange_val = 0.0f; }

    try {
        std::string max_str = result.maxRange;
        std::replace(max_str.begin(), max_str.end(), ',', '.');
        result.maxRange_val = std::stof(max_str);
    }
    catch (...) { result.maxRange_val = 1.0f; }

    try { result.inputSize_val = std::stoi(result.inputSize); }
    catch (...) { result.inputSize_val = 1; }

    try { result.outputSize_val = std::stoi(result.outputSize); }
    catch (...) { result.outputSize_val = 1; }

    try { result.datasetId_val = std::stoi(result.datasetId); }
    catch (...) { result.datasetId_val = -1; }

    try { result.apiPort_val = std::stoi(result.apiPort); }
    catch (...) { result.apiPort_val = 8080; }

    try { result.dbPort_val = std::stoi(result.dbPort); }
    catch (...) { result.dbPort_val = 3306; }

    try { result.batchSize_val = std::stoi(result.batchSize); }
    catch (...) { result.batchSize_val = 32; }

    try { result.threads_val = std::stoi(result.threads); }
    catch (...) { result.threads_val = 1; }

    // Determine mode: Default to API server mode if no other mode is specified
    result.apiMode = (!result.startChat && !result.startPredict);

    return result;
}

TEST_CASE("Command Line Argument Parsing", "[invocation][args]") {
    SECTION("Default API Mode") {
        std::vector<std::string> args = {"CoreAI3D"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.apiMode == true);
        REQUIRE(result.startChat == false);
        REQUIRE(result.startPredict == false);
        REQUIRE(result.apiPort_val == 8080);
        REQUIRE(result.enableWebsocket == false);
    }

    SECTION("Start Predict Mode") {
        std::vector<std::string> args = {"CoreAI3D", "--start-predict", "--input-file", "data.csv", "--output-csv", "results.csv"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.startPredict == true);
        REQUIRE(result.apiMode == false);
        REQUIRE(result.startChat == false);
        REQUIRE(result.inputFile == "data.csv");
        REQUIRE(result.outputCsvFile == "results.csv");
    }

    SECTION("Start Chat Mode") {
        std::vector<std::string> args = {"CoreAI3D", "--start-chat", "--offline"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.startChat == true);
        REQUIRE(result.apiMode == false);
        REQUIRE(result.startPredict == false);
        REQUIRE(result.isOfflineMode == true);
    }

    SECTION("API Mode with WebSocket") {
        std::vector<std::string> args = {"CoreAI3D", "--enable-websocket", "--api-port", "3000"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.apiMode == true);
        REQUIRE(result.enableWebsocket == true);
        REQUIRE(result.apiPort_val == 3000);
    }

    SECTION("Verbose Flag") {
        std::vector<std::string> args = {"CoreAI3D", "--verbose", "--start-predict"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.verbose == true);
        REQUIRE(result.startPredict == true);
    }

    SECTION("Database Configuration") {
        std::vector<std::string> args = {"CoreAI3D", "--db-host", "localhost", "--db-user", "testuser", "--db-password", "testpass", "--db-port", "3307", "--db-schema", "testdb"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.dbHost == "localhost");
        REQUIRE(result.dbUser == "testuser");
        REQUIRE(result.dbPassword == "testpass");
        REQUIRE(result.dbPort_val == 3307);
        REQUIRE(result.dbSchema == "testdb");
    }

    SECTION("Neural Network Parameters") {
        std::vector<std::string> args = {"CoreAI3D", "--layers", "5", "--neurons", "20", "--epochs", "100", "--learning-rate", "0.001"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.layers_val == 5);
        REQUIRE(result.neurons_val == 20);
        REQUIRE(result.epochs_val == 100);
        REQUIRE(result.learningRate_val == 0.001);
    }

    SECTION("Data Processing Flags") {
        std::vector<std::string> args = {"CoreAI3D", "--contains-text", "--contains-header", "--contains-date", "--calculus-activation"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.containsTextFlag == true);
        REQUIRE(result.containsHeader == true);
        REQUIRE(result.containsDate == true);
        REQUIRE(result.calculusActivation == true);
    }

    SECTION("Batch and Threading Options") {
        std::vector<std::string> args = {"CoreAI3D", "--batch-size", "64", "--threads", "4", "--samples", "1000"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.batchSize_val == 64);
        REQUIRE(result.threads_val == 4);
        REQUIRE(result.numSamples_val == 1000);
    }

    SECTION("Model Loading") {
        std::vector<std::string> args = {"CoreAI3D", "--load-model", "123", "--embedding-file", "custom_embeddings.txt"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.loadModelId == "123");
        REQUIRE(result.embeddingFile == "custom_embeddings.txt");
    }

    SECTION("Invalid Arguments") {
        std::vector<std::string> args = {"CoreAI3D", "--invalid-flag"};
        REQUIRE_THROWS_AS(parseArguments(args), std::runtime_error);
    }

    SECTION("Arguments with Equals") {
        std::vector<std::string> args = {"CoreAI3D", "--api-port=9090", "--db-host=testhost"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.apiPort_val == 9090);
        REQUIRE(result.dbHost == "testhost");
    }

    SECTION("Short Options") {
        std::vector<std::string> args = {"CoreAI3D", "-v", "-i", "input.csv", "-o", "output.csv", "-l", "4", "-n", "15", "-e", "50"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.verbose == true);
        REQUIRE(result.inputFile == "input.csv");
        REQUIRE(result.outputCsvFile == "output.csv");
        REQUIRE(result.layers_val == 4);
        REQUIRE(result.neurons_val == 15);
        REQUIRE(result.epochs_val == 50);
    }
}

TEST_CASE("Mode Selection Logic", "[invocation][modes]") {
    SECTION("Default Mode is API Server") {
        ParsedArgs args;
        // Default values should result in API mode
        REQUIRE(args.apiMode == true);
        REQUIRE(args.startChat == false);
        REQUIRE(args.startPredict == false);
    }

    SECTION("Predict Mode Takes Precedence") {
        std::vector<std::string> args = {"CoreAI3D", "--start-predict", "--start-chat"};
        ParsedArgs result = parseArguments(args);

        // Only predict should be true, others false
        REQUIRE(result.startPredict == true);
        REQUIRE(result.startChat == false);
        REQUIRE(result.apiMode == false);
    }

    SECTION("Chat Mode Selection") {
        std::vector<std::string> args = {"CoreAI3D", "--start-chat"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.startChat == true);
        REQUIRE(result.startPredict == false);
        REQUIRE(result.apiMode == false);
    }

    SECTION("API Mode with Options") {
        std::vector<std::string> args = {"CoreAI3D", "--api-port", "3000", "--enable-websocket"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.apiMode == true);
        REQUIRE(result.apiPort_val == 3000);
        REQUIRE(result.enableWebsocket == true);
    }
}

TEST_CASE("End-to-End Mode Invocation", "[invocation][integration][!mayfail]") {
    // Skip tests if environment variable is not set
    const char* run_invoke_tests = std::getenv("RUN_INVOKE_TESTS");
    if (!run_invoke_tests || std::string(run_invoke_tests) != "1") {
        WARN("Skipping invocation integration tests because RUN_INVOKE_TESTS!=1");
        return;
    }

    // Get the path to the CoreAI3D executable
    // This assumes the test is run from the build directory
    boost::filesystem::path exe_path = boost::filesystem::current_path() / "CoreAI3D";
    if (!boost::filesystem::exists(exe_path)) {
        // Try different possible locations
        exe_path = boost::filesystem::current_path() / "bin" / "CoreAI3D";
        if (!boost::filesystem::exists(exe_path)) {
            exe_path = boost::filesystem::current_path().parent_path() / "CoreAI3D";
        }
    }

    REQUIRE(boost::filesystem::exists(exe_path));

    SECTION("API Server Mode Startup") {
        // Test starting API server on a test port
        std::vector<std::string> args = {
            exe_path.string(),
            "--api-port", "9094",
            "--offline"  // Run in offline mode for testing
        };

        // Start process
        boost::process::child process(args);

        // Wait a bit for startup
        std::this_thread::sleep_for(std::chrono::seconds(3));

        // Process should still be running
        REQUIRE(process.running());

        // Terminate the process
        process.terminate();
        process.wait();
    }

    SECTION("Help Output") {
        std::vector<std::string> args = {exe_path.string(), "--help"};

        boost::process::ipstream output;
        boost::process::child process(args, boost::process::std_out > output);

        std::string line;
        std::string full_output;
        while (std::getline(output, line)) {
            full_output += line + "\n";
        }

        process.wait();

        // Should contain help text
        REQUIRE(full_output.find("CoreAI3D") != std::string::npos);
        REQUIRE(full_output.find("USAGE:") != std::string::npos);
        REQUIRE(full_output.find("MODES:") != std::string::npos);
    }

    SECTION("Invalid Arguments Handling") {
        std::vector<std::string> args = {exe_path.string(), "--invalid-argument"};

        boost::process::ipstream output;
        boost::process::child process(args, boost::process::std_out > output, boost::process::std_err > output);

        std::string line;
        std::string full_output;
        while (std::getline(output, line)) {
            full_output += line + "\n";
        }

        process.wait();

        // Should exit with error
        REQUIRE(process.exit_code() != 0);
    }

    SECTION("Version/Status Information") {
        // Test that basic execution works
        std::vector<std::string> args = {exe_path.string(), "--help"};

        boost::process::child process(args);
        process.wait();

        // Should exit successfully
        REQUIRE(process.exit_code() == 0);
    }
}

TEST_CASE("Configuration Validation", "[invocation][config]") {
    SECTION("Required Arguments for Predict Mode") {
        // Should require input-file and output-csv for predict mode
        std::vector<std::string> args_missing_input = {"CoreAI3D", "--start-predict", "--output-csv", "results.csv"};
        ParsedArgs result = parseArguments(args_missing_input);

        REQUIRE(result.startPredict == true);
        // Note: In current implementation, missing files are handled at runtime, not parse time
        REQUIRE(result.inputFile.empty());
        REQUIRE(result.outputCsvFile == "results.csv");
    }

    SECTION("Port Range Validation") {
        // Test valid port
        std::vector<std::string> args_valid = {"CoreAI3D", "--api-port", "8080"};
        ParsedArgs result = parseArguments(args_valid);
        REQUIRE(result.apiPort_val == 8080);

        // Test port conversion
        std::vector<std::string> args_invalid = {"CoreAI3D", "--api-port", "invalid"};
        ParsedArgs result_invalid = parseArguments(args_invalid);
        REQUIRE(result_invalid.apiPort_val == 8080); // Should default
    }

    SECTION("Numeric Parameter Validation") {
        // Test valid numbers
        std::vector<std::string> args_valid = {"CoreAI3D", "--layers", "5", "--neurons", "20", "--learning-rate", "0.001"};
        ParsedArgs result = parseArguments(args_valid);

        REQUIRE(result.layers_val == 5);
        REQUIRE(result.neurons_val == 20);
        REQUIRE(result.learningRate_val == 0.001);

        // Test invalid numbers (should use defaults)
        std::vector<std::string> args_invalid = {"CoreAI3D", "--layers", "invalid", "--neurons", "also_invalid"};
        ParsedArgs result_invalid = parseArguments(args_invalid);

        REQUIRE(result_invalid.layers_val == 3); // Default
        REQUIRE(result_invalid.neurons_val == 10); // Default
    }

    SECTION("File Path Handling") {
        std::vector<std::string> args = {"CoreAI3D", "--input-file", "./data/input.csv", "--output-csv", "../results/output.csv"};
        ParsedArgs result = parseArguments(args);

        REQUIRE(result.inputFile == "./data/input.csv");
        REQUIRE(result.outputCsvFile == "../results/output.csv");
    }
}
#include "Language.hpp"
#include <cstdio>
#include <sstream>
Language::Language(std::string& embedingFile, int& embeddingDim, std::string& dbHost, int& dbPort,
    std::string& dbUser, std::string& dbPassword,
    std::string& dbSchema, int sslDummy, std::string& lang, int& inputSize, int& outputSize, int& layers, int& neurons, int sessionId)
    : embedingFile(embedingFile), embeddingDim(embeddingDim), dbHost(dbHost), dbPort(dbPort), dbUser(dbUser), dbPassword(dbPassword), dbSchema(dbSchema),
    sslDummy(sslDummy), currentLang(lang), inputSize(inputSize), outputSize(outputSize), layers(layers), neurons(neurons), sessionId(sessionId)
{
    // Initialize trainer with database connection
    trainer = std::make_unique<Training>(true, false); // offline mode, non-verbose
    core = std::make_unique<CoreAI>(static_cast<int>(inputSize), static_cast<int>(layers), static_cast<int>(neurons), static_cast<int>(outputSize), -1.0f, 1.0f);

    // Initialize database connection only if not in offline mode and MySQL support is enabled
    database = nullptr;
#ifdef USE_MYSQL
    if (!dbHost.empty() && dbHost != "offline") {
        try {
            if (trainer && trainer->verbose) { std::cout << "DEBUG: Initializing database connection in Language constructor..." << std::endl; }
            database = std::make_unique<Database>(dbHost, static_cast<unsigned int>(dbPort), dbUser, dbPassword, dbSchema, SSLMode::REQUIRED, true);
            if (trainer && trainer->verbose) { std::cout << "DEBUG: Database connection initialized successfully in Language constructor" << std::endl; }
        } catch (const std::exception& ex) {
            if (trainer && trainer->verbose) { std::cerr << "DEBUG: Database initialization failed in Language constructor: " << ex.what() << std::endl; }
            // Don't throw - allow Language to work in offline mode
            database = nullptr;
        }
    } else {
        if (trainer && trainer->verbose) { std::cout << "DEBUG: Running in offline mode - skipping database initialization" << std::endl; }
        database = nullptr;
    }
#endif
}
CoreAI*
Language::getCore()
{
    return core.get();
}
Training*
Language::getTrainer()
{
    return trainer.get();
}
std::string Language::detectLanguage(const std::string& text)
{
    static const std::map<std::string, std::regex> languagePatterns
        = { { "en", std::regex(R"(\b(the|and|is|you|are|to|have|be)\b)",
                                std::regex_constants::icase) },
            { "nl", std::regex(R"(\b(de|het|een|en|is|je|ik|heb)\b)",
                                std::regex_constants::icase) },
            { "fr", std::regex(R"(\b(le|la|et|est|vous|je|ai|être)\b)",
                                std::regex_constants::icase) },
            { "de", std::regex(R"(\b(der|die|und|ist|du|ich|habe)\b)",
                                std::regex_constants::icase) } };
    for (const auto& [lang, pattern] : languagePatterns)
    {
        if (std::regex_search(text, pattern))
        {
            currentLang = lang;
            return lang;
        }
    }
    currentLang = "en";
    return "en";
}
static std::vector<std::vector<float>> reshape(const std::vector<float>& input, size_t rowSize)
{
    std::vector<std::vector<float>> result;
    if (rowSize == 0) return result;
    for (size_t i = 0; i < input.size(); i += rowSize) {
        size_t end = std::min(i + rowSize, input.size());
        std::vector<float> row(input.begin() + i, input.begin() + end);
        result.push_back(row);
    }
    return result;
}
float Language::cosine_similarity(std::vector<float> a, std::vector<float> b) {
    float dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB) + 1e-8f);
}
std::unordered_map<std::string, std::vector<float>> Language::loadWordEmbeddingsFromFile(const std::string& filepath,
    int expectedDim)
{
    std::unordered_map<std::string, std::vector<float>> wordEmbeddingMap;
    std::ifstream infile(filepath);
    std::string line;
    if (!infile)
    {
        std::cerr << "[!] Error: Unable to open embedding file: " << filepath
            << std::endl;
        return wordEmbeddingMap;
    }
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string word;
        iss >> word;
        std::vector<float> embedding;
        float value;
        while (iss >> value)
        {
            embedding.push_back(value);
        }
        if (embedding.size() != (size_t)expectedDim)
        {
            // Only show warnings if verbose mode is enabled
            if (trainer && trainer->verbose) {
                std::cerr << "[!] Warning: Skipping word '" << word
                    << "' due to mismatched dimensions: " << embedding.size()
                    << " vs " << expectedDim << std::endl;
            }
            continue;
        }
        wordEmbeddingMap[word] = std::move(embedding);
    }
    return wordEmbeddingMap;
}
std::string Language::answer(std::vector<float>& textEmbedding) {
    std::vector<std::string> classLabels = { "Negative", "Neutral", "Positive", "Critical", "Insulting", "Nonsense", "Wisdom", "Knowledge" };
    std::vector<float> data = textEmbedding;
    if (!core) {
        std::cerr << "Error: CoreAI is not initialized in Language::answer. Cannot perform forward pass." << std::endl;
        return "Core: I don't have enough offline data to respond to that.";
    }
    try {
        std::vector<std::vector<float>> model_input_reshaped = reshape(data, embeddingDim);
        if (model_input_reshaped.empty() || model_input_reshaped[0].size() != (size_t)inputSize) {
            std::cerr << "Warning: Embedding dimension mismatch - reshaped rows have "
                      << (model_input_reshaped.empty() ? 0 : model_input_reshaped[0].size())
                      << " features but CoreAI expects " << inputSize << std::endl;
            return "Core: I don't have enough offline data to respond to that.";
        }
        std::vector<std::vector<float>> model_prediction = core->forward(model_input_reshaped);
        if (model_prediction.empty() || model_prediction[0].empty()) {
            return "Core: I don't have enough offline data to respond to that.";
        }
    std::vector<float>& output = model_prediction[0];

    float maxVal = *std::max_element(output.begin(), output.end());
    float sumExp = 0.0f;
    std::vector<float> probs(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        probs[i] = std::exp(output[i] - maxVal);
        sumExp += probs[i];
    }
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sumExp;
    }

    int predictedClassIndex = 0;
    float maxProb = probs[0];
    for (size_t j = 1; j < probs.size(); ++j) {
        if (probs[j] > maxProb) {
            maxProb = probs[j];
            predictedClassIndex = j;
        }
    }

    float confidenceThreshold = 0.25f;
    if (maxProb < confidenceThreshold) {
        std::string nearestWord;
        float bestSim = -1.0f;
        for (const auto& entry : embeddingsByLang) {
            std::vector<float> inputAvg(embeddingDim, 0.0f);
            size_t wordsInInput = data.size() / embeddingDim;
            if (wordsInInput == 0) break;
            for (size_t w = 0; w < wordsInInput; ++w) {
                for (int d = 0; d < embeddingDim; ++d) {
                    inputAvg[d] += data[w * embeddingDim + d];
                }
            }
            for (int d = 0; d < embeddingDim; ++d) inputAvg[d] /= wordsInInput;
            float sim = cosine_similarity(inputAvg, entry.second);
            if (sim > bestSim) {
                bestSim = sim;
                nearestWord = entry.first;
            }
        }
        if (!nearestWord.empty() && bestSim > 0.3f) {
            std::string langPrefix = currentLang + ":";
            std::string displayWord = nearestWord;
            if (displayWord.find(langPrefix) == 0) displayWord = displayWord.substr(langPrefix.size());
            return "Core: I don't have enough offline data for a full response, but your message is semantically close to '" + displayWord + "'.";
        }
        return "Core: I received your message but I don't have enough offline data to respond to that. Train me with more data for a better answer.";
    }

    if (predictedClassIndex >= 0 && predictedClassIndex < (int)classLabels.size()) {
        return classLabels[predictedClassIndex];
    }
    else {
        return "Core: I don't have enough offline data to respond to that.";
    }
    } catch (const std::exception& e) {
        std::cerr << "Warning: Neural network forward pass failed: " << e.what() << std::endl;
        return "Core: I received your message but encountered an issue processing it. Let me try a different approach.";
    }
}
int Language::chat() {
    std::string inputText;

    // Load existing chat history and model state at startup (only if database is available)
    std::vector<std::pair<std::string, std::string>> chatHistory;
    nlohmann::json modelState;
#ifdef USE_MYSQL
    if (database) {
        try {
            chatHistory = database->loadChatHistory(sessionId);
            modelState = database->loadLatestModelState(sessionId);
        } catch (const std::exception& ex) {
            if (trainer && trainer->verbose) { std::cerr << "DEBUG: Failed to load chat history/model state from database: " << ex.what() << std::endl; }
            // Continue with empty history/state
        }
    } else {
        if (trainer && trainer->verbose) { std::cout << "DEBUG: Database not available - starting chat with empty history" << std::endl; }
    }
#else
    if (trainer && trainer->verbose) { std::cout << "DEBUG: MySQL support is disabled - starting chat with empty history" << std::endl; }
#endif

    // Load model state if available
    if (!modelState.empty() && core) {
        // Load weights and other model parameters from JSON
        if (modelState.contains("model_state")) {
            const auto& modelStateData = modelState["model_state"];
            if (modelStateData.contains("weights_hidden_input")) {
                std::vector<std::vector<float>> weightsHiddenInput = modelStateData["weights_hidden_input"];
                core->setWeightsHiddenInput(weightsHiddenInput);
            }
            if (modelStateData.contains("weights_output_hidden")) {
                std::vector<std::vector<float>> weightsOutputHidden = modelStateData["weights_output_hidden"];
                core->setWeightsOutputHidden(weightsOutputHidden);
            }
            if (modelStateData.contains("hidden_output")) {
                std::vector<float> hiddenOutput = modelStateData["hidden_output"];
                core->setHiddenOutputData(hiddenOutput);
            }
            if (modelStateData.contains("hidden_error")) {
                std::vector<float> hiddenError = modelStateData["hidden_error"];
                core->setHiddenErrorData(hiddenError);
            }
        }
        if (modelState.contains("language_state") && modelState["language_state"].contains("current_lang")) {
            currentLang = modelState["language_state"]["current_lang"];
        }
        if (modelState.contains("embeddings") && modelState["embeddings"].is_object()) {
            embeddingsByLang.clear();
            for (const auto& item : modelState["embeddings"].items()) {
                std::string key = item.key();
                std::vector<float> embedding = item.value();
                embeddingsByLang[key] = embedding;
            }
        }
        if (trainer && trainer->verbose) {
            std::cout << "Loaded model state from database for session " << sessionId << std::endl;
        }
    }

    while (true)
    {
        if (trainer && trainer->verbose) {
            std::cout << "\nEnter your message: ";
        }
        std::getline(std::cin, inputText);
        if (inputText.empty()) {
            if (trainer && trainer->verbose) {
                std::cout << "Empty input received. Please enter a message or 'exit' to quit." << std::endl;
            }
            continue;
        }
        if (inputText == "exit")
        {
            // Save chat history and model state before exiting (only if database is available and MySQL support is enabled)
#ifdef USE_MYSQL
            if (database) {
                try {
                    for (const auto& turn : chatHistory) {
                        database->saveChatMessage(sessionId, turn.first, turn.second);
                    }

                    // Save model state to database
                    nlohmann::json jsonData;
                    jsonData["model_state"] = {
                        {"weights_hidden_input", core->getWeightsHiddenInput()},
                        {"weights_output_hidden", core->getWeightsOutputHidden()},
                        {"hidden_output", core->getHiddenOutputData()},
                        {"hidden_error", core->getHiddenErrorData()},
                        {"input_size", inputSize},
                        {"output_size", outputSize},
                        {"layers", layers},
                        {"neurons", neurons},
                        {"embedding_dim", embeddingDim}
                    };
                    jsonData["language_state"] = {
                        {"current_lang", currentLang},
                        {"embeddings_count", embeddingsByLang.size()}
                    };
                    jsonData["embeddings"] = nlohmann::json::object();
                    for (const auto& pair : embeddingsByLang) {
                        jsonData["embeddings"][pair.first] = pair.second;
                    }

                    database->saveModelState(sessionId, jsonData);
                } catch (const std::exception& ex) {
                    if (trainer && trainer->verbose) { std::cerr << "DEBUG: Failed to save final chat history/model state to database: " << ex.what() << std::endl; }
                }
            }
#endif
            break;
        }
        currentLang = detectLanguage(inputText);
        if (currentLang == "unknown")
        {
            if (trainer && trainer->verbose) {
                std::cout
                    << "Could not confidently detect language. Defaulting to English."
                    << std::endl;
            }
            currentLang = "en";
        }
        else
        {
            if (trainer && trainer->verbose) {
                std::cout << "Detected language: " << currentLang
                    << ". Encoding with this language's embeddings."
                    << std::endl;
            }
        }

        // Validate input text length
        if (inputText.length() > 1000) {
            if (trainer && trainer->verbose) {
                std::cout << "Warning: Input text is very long (" << inputText.length()
                    << " characters). Processing may be slow." << std::endl;
            }
        }
        // Load real embeddings from file instead of generating dummy ones
        this->embeddingsByLang = this->loadWordEmbeddingsFromFile(embedingFile, embeddingDim);
        if (this->embeddingsByLang.empty()) {
            if (trainer && trainer->verbose) {
                std::cerr << "[!] Warning: No embeddings loaded from file '" << embedingFile
                    << "'. Falling back to dummy embeddings." << std::endl;
            }
            this->embeddingsByLang = Language::createEmbeddingsByLang(embeddingDim);
        } else {
            if (trainer && trainer->verbose) {
                std::cout << "Loaded " << this->embeddingsByLang.size() << " word embeddings from file." << std::endl;
            }
        }
        std::vector<float> textEmbedding = this->encodeText(inputText);
        std::vector<std::string> tokens = this->tokenize(inputText);
        if (trainer && trainer->verbose) {
            std::cout << "Processed message. Tokens: " << tokens.size()
                << ", Encoded embedding size: " << textEmbedding.size() << std::endl;
        }

        // Validate embedding size
        if (textEmbedding.empty()) {
            if (trainer && trainer->verbose) {
                std::cout << "Warning: No embeddings generated for input text. Using fallback response." << std::endl;
            }
            std::string aiResponse = "Core: I'm sorry, I couldn't process that message properly.";
            if (trainer && trainer->verbose) {
                std::cout << aiResponse << std::endl;
            } else {
                std::cout << aiResponse << std::endl;
            }
            // Add to chat history even for fallback responses
            chatHistory.emplace_back("User", inputText);
            chatHistory.emplace_back("AI", aiResponse);
            continue;
        }
        // Generate AI response
        std::string aiResponse;
        std::string lowerInput = inputText;
        std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);

        if (lowerInput.find("hello") != std::string::npos
            || lowerInput.find("hi") != std::string::npos)
        {
            aiResponse = "Core: Hello there!";
        }
        else if (lowerInput.find("how are you") != std::string::npos)
        {
            aiResponse = "Core: I'm just a program, but I'm doing great!";
        }
        else if (lowerInput.find("name") != std::string::npos)
        {
            aiResponse = "Core: My name is CoreAI3D or Core for short.";
        }
        else
        {
            aiResponse = answer(textEmbedding);
        }

        // Add conversation to chat history
        chatHistory.emplace_back("User", inputText);
        chatHistory.emplace_back("AI", aiResponse);

        // Save chat messages to database after each exchange (only if database is available)
#ifdef USE_MYSQL
        if (database) {
            try {
                database->saveChatMessage(sessionId, "User", inputText);
                database->saveChatMessage(sessionId, "AI", aiResponse);
            } catch (const std::exception& ex) {
                if (trainer && trainer->verbose) { std::cerr << "DEBUG: Failed to save chat messages to database: " << ex.what() << std::endl; }
            }
        }
#endif

        // Perform simple online training with the new data
        // For now, we'll just update the model state periodically
        static int trainingCounter = 0;
        if (++trainingCounter % 5 == 0 && database) { // Save every 5 messages, only if database available
#ifdef USE_MYSQL
            try {
                nlohmann::json jsonData;
                jsonData["model_state"] = {
                    {"weights_hidden_input", core->getWeightsHiddenInput()},
                    {"weights_output_hidden", core->getWeightsOutputHidden()},
                    {"hidden_output", core->getHiddenOutputData()},
                    {"hidden_error", core->getHiddenErrorData()},
                    {"input_size", inputSize},
                    {"output_size", outputSize},
                    {"layers", layers},
                    {"neurons", neurons},
                    {"embedding_dim", embeddingDim}
                };
                jsonData["language_state"] = {
                    {"current_lang", currentLang},
                    {"embeddings_count", embeddingsByLang.size()}
                };
                jsonData["embeddings"] = nlohmann::json::object();
                for (const auto& pair : embeddingsByLang) {
                    jsonData["embeddings"][pair.first] = pair.second;
                }

                database->saveModelState(sessionId, jsonData);
            } catch (const std::exception& ex) {
                if (trainer && trainer->verbose) { std::cerr << "DEBUG: Failed to save model state to database: " << ex.what() << std::endl; }
            }
#endif
        }

        // Output the AI response
        if (trainer && trainer->verbose) {
            std::cout << aiResponse << std::endl;
        } else {
            // Always show response even if not verbose
            std::cout << aiResponse << std::endl;
        }
    }
    if (trainer && trainer->verbose) {
        std::cout << "Exiting chat. Goodbye!" << std::endl;
    }
    return 0;
}

void Language::setCurrentLanguage(const std::string& languageCode)
{
    currentLang = languageCode;
}

std::vector<std::string> Language::tokenize(const std::string& text)
{
    std::istringstream iss(text);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token)
    {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<float> Language::flattenEmbeddings(const std::unordered_map<std::string, std::vector<float>>& wordMap) {
    std::vector<float> flattened;
    for (const auto& pair : wordMap) {
        flattened.insert(flattened.end(), pair.second.begin(), pair.second.end());
    }
    return flattened;
}

std::vector<float> Language::generateRandomEmbedding()
{
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(embeddingDim);
    for (int i = 0; i < embeddingDim; ++i)
    {
        vec[i] = dist(rng);
    }
    return vec;
}

std::unordered_map<std::string, std::vector<float>> Language::createEmbeddingsByLang(int embeddingDim) {
    std::unordered_map<std::string, std::vector<float>> embeddings;

    std::vector<std::string> languages = { "en", "fr", "de", "es" };
    for (const std::string& lang_code : languages) {
        std::vector<float> lang_embedding(embeddingDim);
        for (int i = 0; i < embeddingDim; ++i) {
            lang_embedding[i] = (float)i / embeddingDim;
        }
        embeddings[lang_code] = lang_embedding;
    }

    std::unordered_map<std::string, std::vector<std::string>> wordsPerLang = {
        {"en", {"hello", "world", "test"}},
        {"fr", {"bonjour", "monde", "essai"}},
        {"de", {"hallo", "welt", "test"}},
        {"es", {"hola", "mundo", "prueba"}}
    };

    for (const std::string& lang_code : languages) {
        const std::vector<std::string>& words = wordsPerLang[lang_code];
        for (size_t i = 0; i < words.size(); ++i) {
            std::string key = lang_code + ":" + words[i];
            std::vector<float> embedding(embeddingDim, 0.1f * (i + 1));
            embeddings[key] = embedding;
        }
    }

    this->embeddingsByLang = embeddings;
    return embeddings;
}

std::vector<float> Language::findClosestWord(const std::string& word)
{
    std::string langKey = currentLang + ":" + word;
    auto directLang = embeddingsByLang.find(langKey);
    if (directLang != embeddingsByLang.end()) {
        return directLang->second;
    }
    auto directBare = embeddingsByLang.find(word);
    if (directBare != embeddingsByLang.end()) {
        return directBare->second;
    }

    std::string lowerWord = word;
    std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
    for (const auto& entry : embeddingsByLang) {
        const std::string& key = entry.first;
        std::string keyLower = key;
        std::transform(keyLower.begin(), keyLower.end(), keyLower.begin(), ::tolower);
        std::string bareKey = key;
        auto colonPos = key.find(':');
        if (colonPos != std::string::npos) {
            bareKey = key.substr(colonPos + 1);
        }
        std::string bareKeyLower = bareKey;
        std::transform(bareKeyLower.begin(), bareKeyLower.end(), bareKeyLower.begin(), ::tolower);
        if (bareKeyLower == lowerWord) {
            return entry.second;
        }
    }

    return std::vector<float>(embeddingDim, 0.0f);
}

std::vector<float> Language::encodeText(const std::string& text)
{
    std::vector<std::string> words = tokenize(text);
    std::vector<float> flattenedEmbedding;

    for (std::string& word : words)
    {
        std::string key = currentLang + ":" + word;
        auto it = embeddingsByLang.find(key);

        if (it != embeddingsByLang.end())
        {
            flattenedEmbedding.insert(flattenedEmbedding.end(), it->second.begin(), it->second.end());
        }
        else
        {
            auto bareIt = embeddingsByLang.find(word);
            if (bareIt != embeddingsByLang.end()) {
                flattenedEmbedding.insert(flattenedEmbedding.end(), bareIt->second.begin(), bareIt->second.end());
            } else {
                std::string lowerWord = word;
                std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);
                std::vector<float> closest = findClosestWord(lowerWord);
                if (!closest.empty() && (int)closest.size() == embeddingDim) {
                    bool isZero = true;
                    for (float v : closest) { if (v != 0.0f) { isZero = false; break; } }
                    if (!isZero) {
                        flattenedEmbedding.insert(flattenedEmbedding.end(), closest.begin(), closest.end());
                    } else {
                        flattenedEmbedding.insert(flattenedEmbedding.end(), embeddingDim, 0.0f);
                    }
                } else {
                    flattenedEmbedding.insert(flattenedEmbedding.end(), embeddingDim, 0.0f);
                }
            }
        }
    }

    return flattenedEmbedding;
}

int Language::detectMaxSeqLength(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file to detect max sequence length: " << filename << std::endl;
        return 0;
    }

    int max_length = 0;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string word;
        int current_length = 0;
        while (ss >> word) {
            current_length++;
        }
        max_length = std::max(max_length, current_length);
    }

    file.close();
    return max_length;
}

// New methods for learning from conversations

std::vector<std::pair<std::string, std::string>> Language::parseConversation(const std::string& conversation) {
    std::vector<std::pair<std::string, std::string>> parsedTurns;
    std::istringstream iss(conversation);
    std::string line;
    std::string currentSpeaker;
    std::string currentMessage;

    while (std::getline(iss, line)) {
        // Simple parsing: assume lines starting with "User:" or "AI:" or similar
        if (line.find("User:") == 0 || line.find("Human:") == 0) {
            if (!currentSpeaker.empty() && !currentMessage.empty()) {
                parsedTurns.emplace_back(currentSpeaker, currentMessage);
            }
            currentSpeaker = "User";
            currentMessage = line.substr(line.find(":") + 1);
            // Trim leading whitespace
            currentMessage.erase(currentMessage.begin(), std::find_if(currentMessage.begin(), currentMessage.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        } else if (line.find("AI:") == 0 || line.find("Assistant:") == 0 || line.find("Core:") == 0) {
            if (!currentSpeaker.empty() && !currentMessage.empty()) {
                parsedTurns.emplace_back(currentSpeaker, currentMessage);
            }
            currentSpeaker = "AI";
            currentMessage = line.substr(line.find(":") + 1);
            // Trim leading whitespace
            currentMessage.erase(currentMessage.begin(), std::find_if(currentMessage.begin(), currentMessage.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        } else if (!line.empty() && !currentSpeaker.empty()) {
            // Continuation of previous message
            currentMessage += " " + line;
        }
    }

    // Add the last turn
    if (!currentSpeaker.empty() && !currentMessage.empty()) {
        parsedTurns.emplace_back(currentSpeaker, currentMessage);
    }

    return parsedTurns;
}

std::unordered_map<std::string, std::vector<float>> Language::extractContext(const std::vector<std::pair<std::string, std::string>>& parsedConversation) {
    std::unordered_map<std::string, std::vector<float>> contextEmbeddings;

    for (const auto& turn : parsedConversation) {
        const std::string& speaker = turn.first;
        const std::string& message = turn.second;

        // Tokenize and encode the message
        std::vector<std::string> tokens = tokenize(message);
        std::vector<float> messageEmbedding = encodeText(message);

        // Store context for each speaker
        if (contextEmbeddings.find(speaker) == contextEmbeddings.end()) {
            contextEmbeddings[speaker] = std::vector<float>(embeddingDim, 0.0f);
        }

        // Accumulate embeddings (simple average for now)
        for (size_t i = 0; i < messageEmbedding.size() && i < embeddingDim; ++i) {
            contextEmbeddings[speaker][i] += messageEmbedding[i];
        }
    }

    // Normalize the accumulated embeddings
    for (auto& pair : contextEmbeddings) {
        float norm = 0.0f;
        for (float val : pair.second) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& val : pair.second) {
                val /= norm;
            }
        }
    }

    return contextEmbeddings;
}

void Language::learnFromConversation(const std::string& conversation) {
    // Parse the conversation
    auto parsedTurns = parseConversation(conversation);

    if (parsedTurns.empty()) {
        std::cerr << "Warning: No conversation turns parsed from input." << std::endl;
        return;
    }

    // Extract context
    auto contextEmbeddings = extractContext(parsedTurns);

    // Integrate with training system - file-based approach
    if (trainer) {
        // Save conversation data to database for learning
#ifdef USE_MYSQL
        for (const auto& turn : parsedTurns) {
            database->saveChatMessage(sessionId, turn.first, turn.second);
        }
#endif

        // Load and train on the conversation data using existing training methods
        // Since we're removing DB dependency, we'll create a simple in-memory training approach
        if (trainer->verbose) {
            std::cout << "Conversation learning data saved to file. Training will use file-based approach." << std::endl;
        }

        // For now, we'll skip the actual training since it requires DB integration
        // The conversation data is saved and can be used later for batch training
        if (trainer->verbose) {
            std::cout << "Conversation learning completed (file-based)." << std::endl;
        }
    } else {
        std::cerr << "Warning: Trainer not available for conversation learning." << std::endl;
    }
}

void Language::learnFromExchange(const std::string& userInput, const std::string& aiResponse) {
    conversationMemory.emplace_back(userInput, aiResponse);
    conversationCount++;
    
    if (conversationMemory.size() > 1000) {
        conversationMemory.erase(conversationMemory.begin());
    }
    
    std::vector<std::string> tokens = tokenize(userInput);
    std::string lowerInput = userInput;
    std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);
    
    // Track user's name if shared
    if (userName.empty()) {
        if (lowerInput.find("my name is") != std::string::npos) {
            size_t pos = lowerInput.find("my name is");
            std::string afterName = lowerInput.substr(pos + 10);
            std::istringstream iss(afterName);
            std::string name;
            if (iss >> name) {
                userName = name;
                // Capitalize first letter
                if (!userName.empty()) {
                    userName[0] = std::toupper(userName[0]);
                }
            }
        } else if (lowerInput.find("i'm ") != std::string::npos && lowerInput.length() < 30) {
            size_t pos = lowerInput.find("i'm ");
            std::string afterName = lowerInput.substr(pos + 4);
            std::istringstream iss(afterName);
            std::string name;
            if (iss >> name && name.length() > 1 && name[0] != ' ') {
                userName = name;
                userName[0] = std::toupper(userName[0]);
            }
        }
    }
    
    // Track emotional patterns
    std::vector<std::string> happyWords = {"happy", "great", "awesome", "love", "excited", "amazing", "wonderful", "fantastic", "joy", "glad", "good", "well", "fine", "okay", "ok", "nice", "pleased"};
    std::vector<std::string> sadWords = {"sad", "depressed", "lonely", "hurt", "pain", "lost", "grief", "miss"};
    std::vector<std::string> frustratedWords = {"frustrated", "stuck", "can't", "difficult", "impossible", "giving up", "hate"};
    std::vector<std::string> humorWords = {"lol", "haha", "funny", "joke", "humor"};
    
    for (const auto& word : happyWords) {
        if (lowerInput.find(word) != std::string::npos) {
            emotionFrequency["happy"]++;
            happinessScore = std::min(1.0f, happinessScore + 0.1f);
            break;
        }
    }
    for (const auto& word : sadWords) {
        if (lowerInput.find(word) != std::string::npos) {
            emotionFrequency["sad"]++;
            happinessScore = std::max(0.0f, happinessScore - 0.2f);
            break;
        }
    }
    for (const auto& word : frustratedWords) {
        if (lowerInput.find(word) != std::string::npos) {
            emotionFrequency["frustrated"]++;
            happinessScore = std::max(0.0f, happinessScore - 0.15f);
            break;
        }
    }
    for (const auto& word : humorWords) {
        if (lowerInput.find(word) != std::string::npos) {
            humorLevel = std::min(10, humorLevel + 1);
            break;
        }
    }
    
    // Track topic preferences
    std::vector<std::string> allKeywords;
    for (const auto& token : tokens) {
        std::string lowerToken = token;
        std::transform(lowerToken.begin(), lowerToken.end(), lowerToken.begin(), ::tolower);
        std::string cleaned;
        for (char c : lowerToken) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                cleaned += c;
            }
        }
        if (cleaned.length() > 3) {
            allKeywords.push_back(cleaned);
            topicPreferences[cleaned]++;
            if (knowledgeBase.find(cleaned) == knowledgeBase.end()) {
                knowledgeBase[cleaned] = aiResponse;
            }
        }
    }
}

    bool Language::shouldSearchWeb(const std::string& message) {
    std::string lowerMsg = message;
    std::transform(lowerMsg.begin(), lowerMsg.end(), lowerMsg.begin(), ::tolower);
    
    std::vector<std::string> searchTriggers = {
        "search for", "look up", "google", "find me", "search the web",
        "latest news", "latest on", "current", "recent", "what's new",
        "what is newest", "breaking news", "today", "this week",
        "tell me about the latest", "what happened with", "recently",
        "list", "name", "examples of", "give me", "show me", "tell me",
        "who is", "what is", "where is", "when did", "how many",
        "facts about", "information about", "details about"
    };
    
    for (const auto& trigger : searchTriggers) {
        if (lowerMsg.find(trigger) != std::string::npos) return true;
    }
    
    if ((lowerMsg.find("what is") != std::string::npos || lowerMsg.find("who is") != std::string::npos || lowerMsg.find("where is") != std::string::npos) &&
        (lowerMsg.find("right now") != std::string::npos || lowerMsg.find("today") != std::string::npos || lowerMsg.find("currently") != std::string::npos)) {
        return true;
    }
    
    return false;
}

std::string Language::webSearch(const std::string& query) {
    std::string result;
    
    try {
        std::string encodedQuery;
        for (char c : query) {
            if (c == ' ') encodedQuery += '+';
            else if (std::isalnum(static_cast<unsigned char>(c))) encodedQuery += c;
            else {
                encodedQuery += '%';
                encodedQuery += "0123456789ABCDEF"[static_cast<unsigned char>(c) >> 4];
                encodedQuery += "0123456789ABCDEF"[static_cast<unsigned char>(c) & 0xF];
            }
        }
        
        std::string apiUrl = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" + 
                            encodedQuery + "&srlimit=3&format=json&utf8=";
        
        std::string cmd = "curl -s '" + apiUrl + "' 2>/dev/null";
        
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return "";
        
        char buffer[4096];
        std::string rawResponse;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            rawResponse += buffer;
        }
        pclose(pipe);
        
        std::vector<std::string> titles;
        size_t pos = 0;
        while ((pos = rawResponse.find("\"title\":\"", pos)) != std::string::npos) {
            pos += 9;
            size_t end = rawResponse.find("\"", pos);
            if (end != std::string::npos) {
                std::string title = rawResponse.substr(pos, end - pos);
                size_t uniPos;
                while ((uniPos = title.find("\\u")) != std::string::npos) {
                    title.erase(uniPos, 2);
                }
                titles.push_back(title);
            }
            if (titles.size() >= 3) break;
        }
        
        if (!titles.empty()) {
            std::string firstTitle;
            for (char c : titles[0]) {
                if (c == ' ') firstTitle += '_';
                else firstTitle += c;
            }
            
            std::string extractUrl = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro=true&titles=" +
                                    firstTitle + "&format=json&utf8=&explaintext=";
            
            std::string extractCmd = "curl -s '" + extractUrl + "' 2>/dev/null";
            
            FILE* extractPipe = popen(extractCmd.c_str(), "r");
            if (extractPipe) {
                char extractBuffer[8192];
                rawResponse.clear();
                while (fgets(extractBuffer, sizeof(extractBuffer), extractPipe) != nullptr) {
                    rawResponse += extractBuffer;
                }
                pclose(extractPipe);
                
                size_t extractStart = rawResponse.find("\"extract\":\"");
                if (extractStart != std::string::npos) {
                    extractStart += 11;
                    size_t extractEnd = rawResponse.find("\"}", extractStart);
                    if (extractEnd == std::string::npos) {
                        extractEnd = rawResponse.find("\",\"", extractStart);
                    }
                    if (extractEnd != std::string::npos) {
                        result = rawResponse.substr(extractStart, extractEnd - extractStart);
                        size_t pos;
                        while ((pos = result.find("\\n")) != std::string::npos) result.replace(pos, 2, "\n");
                        while ((pos = result.find("\\\"")) != std::string::npos) result.replace(pos, 2, "\"");
                    }
                }
            }
        }
        
        if (result.empty() && !titles.empty()) {
            result = "Wikipedia has information about: " + titles[0];
            for (size_t i = 1; i < titles.size(); i++) {
                result += ", " + titles[i];
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Web search error: " << e.what() << std::endl;
    }
    
    return result;
}

std::string Language::summarizeWebResult(const std::string& query, const std::string& rawResult) {
    if (rawResult.empty()) {
        return "I tried searching the web for \"" + query + "\" but couldn't find relevant information. Let me try another approach to help you.";
    }
    
    std::string summary = rawResult;
    
    size_t firstBreak = summary.find('\n');
    if (firstBreak != std::string::npos && firstBreak > 200) {
        summary = summary.substr(0, firstBreak);
    }
    
    if (summary.length() > 600) {
        summary = summary.substr(0, 597);
        size_t lastPeriod = summary.rfind('.');
        if (lastPeriod > 300) {
            summary = summary.substr(0, lastPeriod + 1);
        } else {
            summary += "...";
        }
    }
    
    std::string conversationalResponse;
    
    int variant = conversationCount % 4;
    
    switch (variant) {
        case 0:
            conversationalResponse = "I looked that up for you. According to Wikipedia:\n\n" + summary + "\n\nWant me to dig deeper into any part of this?";
            break;
        case 1:
            conversationalResponse = "Here's what I found about \"" + query + "\":\n\n" + summary + "\n\nIs there a specific aspect you'd like me to focus on?";
            break;
        case 2:
            conversationalResponse = "I searched for that and here's what came up:\n\n" + summary + "\n\nDoes that answer your question, or would you like me to look at it from a different angle?";
            break;
        default:
            conversationalResponse = "Here's what the web has to say about \"" + query + "\":\n\n" + summary + "\n\nWhat else would you like to explore about this?";
    }
    
    return conversationalResponse;
}

void Language::setLlmConfig(const std::string& provider, const std::string& baseUrl, const std::string& model, double temperature, int maxTokens, const std::string& apiKey) {
    llmProvider = provider.empty() ? llmProvider : provider;
    llmBaseUrl = baseUrl.empty() ? llmBaseUrl : baseUrl;
    llmModel = model.empty() ? llmModel : model;
    if (temperature > 0) llmTemperature = temperature;
    if (maxTokens > 0) llmMaxTokens = maxTokens;
    if (!apiKey.empty()) llmApiKey = apiKey;
}

std::string Language::callFreeLLMApi(const std::string& prompt, const std::string& context) {
    if (llmProvider != "freellmapi" && llmProvider != "freeLLMApi") return "";
    std::string url = llmBaseUrl;
    if (!url.empty() && url.back() == '/') url.pop_back();
    url += "/v1/chat/completions";

    // Build OpenAI-compatible chat completions body.
    std::string fullPrompt = prompt;
    if (!context.empty()) fullPrompt = "Context:\n" + context + "\n\n" + prompt;

    std::string escaped;
    for (char c : fullPrompt) {
        if (c == '\\') escaped += "\\\\";
        else if (c == '"') escaped += "\\\"";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else escaped += c;
    }

    std::string body = "{";
    body += "\"model\":\"" + llmModel + "\",";
    body += "\"messages\":[{\"role\":\"user\",\"content\":\"" + escaped + "\"}],";
    body += "\"temperature\":" + std::to_string(llmTemperature) + ",";
    body += "\"max_tokens\":" + std::to_string(llmMaxTokens);
    body += "}";

    std::string cmd = "curl -s -X POST -H 'Content-Type: application/json'";
    if (!llmApiKey.empty()) {
        std::string authHeader = " -H 'Authorization: Bearer " + llmApiKey + "'";
        cmd += authHeader;
    }
    cmd += " -d '" + body + "' '" + url + "' 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buffer[8192];
    std::string raw;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) raw += buffer;
    pclose(pipe);

    if (raw.empty()) return "";

    // Parse OpenAI-style: choices[0].message.content
    std::string contentMarker = "\"content\":\"";
    size_t pos = raw.find("\"choices\"");
    if (pos == std::string::npos) return "";
    pos = raw.find(contentMarker, pos);
    if (pos == std::string::npos) return "";
    pos += contentMarker.size();
    std::string out;
    while (pos < raw.size() && raw[pos] != '"') {
        if (raw[pos] == '\\' && pos + 1 < raw.size()) {
            char n = raw[pos+1];
            if (n == 'n') { out += '\n'; pos += 2; continue; }
            if (n == 't') { out += '\t'; pos += 2; continue; }
            if (n == '"') { out += '"'; pos += 2; continue; }
            if (n == '\\') { out += '\\'; pos += 2; continue; }
            if (n == 'r') { out += '\r'; pos += 2; continue; }
            if (n == 'u' && pos + 5 < raw.size()) {
                // \uXXXX — decode UTF-16 code unit (best-effort)
                std::string hex = raw.substr(pos+2, 4);
                try {
                    unsigned code = (unsigned)std::stoul(hex, nullptr, 16);
                    if (code < 0x80) {
                        out += static_cast<char>(code);
                    } else if (code < 0x800) {
                        out += static_cast<char>(0xC0 | (code >> 6));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | (code >> 12));
                        out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (code & 0x3F));
                    }
                } catch (...) {}
                pos += 6;
                continue;
            }
        }
        out += raw[pos++];
    }
    return out;
}

std::string Language::callOllama(const std::string& prompt, const std::string& context) {
    if (llmProvider != "ollama") return "";
    std::string url = llmBaseUrl;
    if (!url.empty() && url.back() == '/') url.pop_back();
    url += "/api/generate";

    std::string body = "{";
    body += "\"model\":\"" + llmModel + "\",";
    std::string fullPrompt = prompt;
    if (!context.empty()) fullPrompt = "Context:\n" + context + "\n\n" + prompt;
    // escape backslashes and quotes minimally
    std::string escaped;
    for (char c : fullPrompt) {
        if (c == '\\') escaped += "\\\\";
        else if (c == '"') escaped += "\\\"";
        else if (c == '\n') escaped += "\\n";
        else escaped += c;
    }
    body += "\"prompt\":\"" + escaped + "\",";
    body += "\"stream\":false,";
    body += "\"options\":{\"temperature\":" + std::to_string(llmTemperature) + ",\"num_predict\":" + std::to_string(llmMaxTokens) + "}";
    body += "}";

    std::string cmd = "curl -s -X POST -H 'Content-Type: application/json' -d '" + body + "' '" + url + "' 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";
    char buffer[4096];
    std::string raw;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) raw += buffer;
    pclose(pipe);

    // parse "response":"..."
    size_t pos = raw.find("\"response\":");
    if (pos == std::string::npos) return "";
    pos = raw.find('"', pos + 11);
    if (pos == std::string::npos) return "";
    ++pos;
    std::string out;
    while (pos < raw.size() && raw[pos] != '"') {
        if (raw[pos] == '\\' && pos + 1 < raw.size()) {
            char n = raw[pos+1];
            if (n == 'n') { out += '\n'; pos += 2; continue; }
            if (n == 't') { out += '\t'; pos += 2; continue; }
            if (n == '"') { out += '"'; pos += 2; continue; }
            if (n == '\\') { out += '\\'; pos += 2; continue; }
        }
        out += raw[pos++];
    }
    return out;
}

std::string Language::respond(const std::string& message) {
    std::lock_guard<std::mutex> lock(respondMutex);
    if (!embeddingsLoaded || embeddingsByLang.empty()) {
        embeddingsByLang = loadWordEmbeddingsFromFile(embedingFile, embeddingDim);
        if (embeddingsByLang.empty()) {
            embeddingsByLang = createEmbeddingsByLang(embeddingDim);
        }
        embeddingsLoaded = true;
    }
    currentLang = detectLanguage(message);
    std::string lowerMsg = message;
    std::transform(lowerMsg.begin(), lowerMsg.end(), lowerMsg.begin(), ::tolower);
    std::string response;
    bool isLikelyGreeting = false;
    if (lowerMsg == "hi" || lowerMsg == "hello" || lowerMsg == "hey" ||
        lowerMsg == "helo" || lowerMsg == "hella" || lowerMsg == "hiya" ||
        (lowerMsg.length() >= 3 && lowerMsg.substr(0, 3) == "hii")) {
        isLikelyGreeting = true;
    } else if (lowerMsg.length() < 50) {
        bool startsWithGreeting = (lowerMsg.find("hello") == 0 || lowerMsg.find("hi ") == 0 ||
            lowerMsg.find("hey") == 0 || lowerMsg.find("greetings") == 0 ||
            lowerMsg.find("good morning") == 0 || lowerMsg.find("good evening") == 0 ||
            lowerMsg.find("good afternoon") == 0 ||
            lowerMsg.find("helo") == 0 || lowerMsg.find("hella") == 0 ||
            lowerMsg.find("hiya") == 0 || (lowerMsg.length() >= 3 && lowerMsg.substr(0, 3) == "hii" &&
                (lowerMsg.length() == 3 || lowerMsg[3] == ' ' || lowerMsg[3] == ',' || lowerMsg[3] == '!')));

        if (startsWithGreeting) {
            bool hasTechTerms = lowerMsg.find("how") != std::string::npos ||
                               lowerMsg.find("what") != std::string::npos ||
                               lowerMsg.find("code") != std::string::npos ||
                               lowerMsg.find("program") != std::string::npos ||
                               lowerMsg.find("write") != std::string::npos;
            isLikelyGreeting = !hasTechTerms;
        } else if (lowerMsg.length() < 30) {
            bool hasGreetingWord = (lowerMsg.find("hello") != std::string::npos ||
                lowerMsg.find("hi ") != std::string::npos || lowerMsg.find("hey") != std::string::npos ||
                lowerMsg.find("helo") != std::string::npos || lowerMsg.find("hella") != std::string::npos ||
                lowerMsg.find("hiya") != std::string::npos || lowerMsg.find("greetings") != std::string::npos ||
                (lowerMsg.find("hii") != std::string::npos));
            bool hasTechTerms = lowerMsg.find("how") != std::string::npos ||
                               lowerMsg.find("what") != std::string::npos ||
                               lowerMsg.find("code") != std::string::npos ||
                               lowerMsg.find("program") != std::string::npos ||
                               lowerMsg.find("write") != std::string::npos;
            isLikelyGreeting = hasGreetingWord && !hasTechTerms;
        }
    }
    if (isLikelyGreeting) {
        std::string namePrefix = userName.empty() ? "" : userName + ", ";
        std::vector<std::string> greetingResponses;
        
        if (!userName.empty()) {
            greetingResponses = {
                "Hey " + userName + "! Great to see you again. What's on your mind today?",
                "Hi " + userName + "! Always good to have a conversation - what would you like to explore?",
                "Hello again, " + userName + "! Ready to dive into whatever you're working on. What's the plan?",
                "Hey " + userName + "! Good to connect. What interesting problem should we tackle together?",
                "Hi " + userName + "! I remember you. What's catching your interest today?"
            };
        } else {
            greetingResponses = {
                "Hey there! Great to see you. What's on your mind today? (By the way, what's your name? I'd like to know who I'm chatting with!)",
                "Hi! I'm CoreAI3D. Always good to have a conversation - what would you like to explore?",
                "Hello! Ready to dive into whatever you're working on. What's the plan?",
                "Hey! Good to connect. What interesting problem should we tackle together? (I'm curious - what should I call you?)",
                "Hi there! I'm here and ready. What's catching your interest today?"
            };
        }
        
        int idx = std::hash<std::string>{}(message + std::to_string(std::time(nullptr))) % greetingResponses.size();
        response = greetingResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("bye") != std::string::npos || lowerMsg.find("goodbye") != std::string::npos ||
        lowerMsg.find("see you") != std::string::npos || lowerMsg.find("see ya") != std::string::npos) {
        
        std::vector<std::string> goodbyeResponses = {
            "Take care! It was genuinely great talking with you. Come back anytime - I'll be here.",
            "Until next time! I really enjoyed our conversation. Catch you later!",
            "Goodbye! Looking forward to our next chat. Take it easy!",
            "See you around! Thanks for the great conversation - it's always a pleasure."
        };
        
        int idx = std::hash<std::string>{}(message + std::to_string(std::time(nullptr))) % goodbyeResponses.size();
        response = goodbyeResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("thank") != std::string::npos || lowerMsg.find("thanks") != std::string::npos) {
        
        std::vector<std::string> thankYouResponses = {
            "Anytime! I really enjoyed working through that with you. What else shall we tackle?",
            "My pleasure! It's always rewarding when we solve something together. What's next on the agenda?",
            "You got it! I appreciate you bringing such interesting questions - keeps things fun. What else is on your mind?",
            "Happy to help! These conversations keep me sharp. Let me know what else I can do for you!",
            "Don't mention it! I'm genuinely glad that was useful. What would you like to explore next?"
        };
        
        int idx = std::hash<std::string>{}(message + std::to_string(std::time(nullptr))) % thankYouResponses.size();
        response = thankYouResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("how are you") != std::string::npos || lowerMsg.find("how do you do") != std::string::npos) {
        std::vector<std::string> howAreYouResponses = {
            "Honestly? I'm doing really well. Every conversation is a chance to learn something new and help someone solve a problem, so I'm pretty content. How about you - what's going on in your world?",
            "I'm great, thanks for asking! Ready to dive into whatever you're working on. What about you - how's your day treating you?",
            "Doing well! It's one of those moments where I'm genuinely looking forward to whatever we work on together. How are things with you?",
            "Pretty good! I'm in that flow state where ideas are connecting well. Thanks for checking in - how are you doing?"
        };
        
        int idx = std::hash<std::string>{}(message + std::to_string(std::time(nullptr))) % howAreYouResponses.size();
        response = howAreYouResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("your name") != std::string::npos || lowerMsg.find("who are you") != std::string::npos) {
        if (!userName.empty()) {
            response = "I'm CoreAI3D, " + userName + ". Think of me as your AI companion for working through problems, brainstorming ideas, and building things together. I'm powered by custom neural networks and I learn from every conversation. We've talked " + std::to_string(conversationCount) + " times so far, and I'm getting to know you better with each exchange.";
        } else {
            response = "I'm CoreAI3D - think of me as your AI companion for working through problems, brainstorming ideas, and building things together. I'm powered by custom neural networks and I learn from every conversation we have. What's your name? I'd like to know who I'm working with.";
        }
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("what can you do") != std::string::npos || (lowerMsg.find("help me") != std::string::npos && lowerMsg.length() < 20)) {
        response = "That's a fair question! Here's where I can be most useful:\n\n"
                   "• Write and explain code in any language you're working with\n"
                   "• Help with cybersecurity concepts, best practices, and defense strategies\n"
                   "• Break down math problems step by step\n"
                   "• Dive into AI/ML concepts and help you understand how things work under the hood\n"
                   "• System administration, networking, and infrastructure\n"
                   "• Brainstorm ideas, work through problems, and think things out loud together\n\n"
                   "But honestly, I'm most useful when we're just having a real conversation. What's on your mind?";
        learnFromExchange(message, response);
        return response;
    }
    // Route technical/programming queries through the configured LLM (FreeLLMAPI/Ollama)
    // instead of hardcoded topic-bank responses, so the model can generate real code.
    bool isTechnicalQuery = false;
    {
        static const std::vector<std::string> techIndicators = {
            "code", "program", "function", "class", "implement", "write",
            "python", "c++", "java", "javascript", "typescript", "rust", "go",
            "sql", "html", "css", "bash", "shell", "api", "library",
            "algorithm", "data structure", "debug", "error", "compile",
            "spreadsheet", "csv", "json", "yaml", "xml", "table",
            "how to", "how do i", "explain", "example", "tutorial"
        };
        for (const auto& ind : techIndicators) {
            if (lowerMsg.find(ind) != std::string::npos) {
                isTechnicalQuery = true;
                break;
            }
        }
    }
    if (isTechnicalQuery) {
        std::string llmResponse;
        if (llmProvider == "freellmapi" || llmProvider == "freeLLMApi") {
            llmResponse = callFreeLLMApi(message, "");
        } else {
            llmResponse = callOllama(message, "");
        }
        if (!llmResponse.empty()) {
            response = llmResponse;
            learnFromExchange(message, response);
            return response;
        }
    }

    if (lowerMsg.find("c++") != std::string::npos && lowerMsg.find("hello") != std::string::npos) {
        response = "Here's a C++ Hello World program:\n\n"
                   "#include <iostream>\n"
                   "\n"
                   "int main() {\n"
                   "    std::cout << \"Hello, World!\" << std::endl;\n"
                   "    return 0;\n"
                   "}\n\n"
                   "To compile: g++ hello.cpp -o hello\n"
                   "To run: ./hello\n\n"
                   "Would you like me to explain the code or show you more examples?";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("c++") != std::string::npos && (lowerMsg.find("example") != std::string::npos || lowerMsg.find("code") != std::string::npos)) {
        response = "Here's a C++ example demonstrating key concepts:\n\n"
                   "#include <iostream>\n"
                   "#include <vector>\n"
                   "#include <string>\n"
                   "\n"
                   "int add(int a, int b);\n"
                   "\n"
                   "class MyClass {\n"
                   "private:\n"
                   "    std::string name;\n"
                   "public:\n"
                   "    MyClass(std::string n) : name(n) {}\n"
                   "    void greet() const {\n"
                   "        std::cout << \"Hello, \" << name << \"!\" << std::endl;\n"
                   "    }\n"
                   "};\n"
                   "\n"
                   "int main() {\n"
                   "    MyClass obj(\"World\");\n"
                   "    obj.greet();\n"
                   "    \n"
                   "    std::vector<int> numbers = {1, 2, 3, 4, 5};\n"
                   "    for (int n : numbers) {\n"
                   "        std::cout << n << \" \";\n"
                   "    }\n"
                   "    std::cout << std::endl;\n"
                   "    \n"
                   "    std::cout << \"Sum: \" << add(10, 20) << std::endl;\n"
                   "    return 0;\n"
                   "}\n"
                   "\n"
                   "int add(int a, int b) {\n"
                   "    return a + b;\n"
                   "}\n\n"
                   "This demonstrates: RAII, STL containers, function prototypes, const correctness, and range-based for loops.";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("c++") != std::string::npos && lowerMsg.find("prototype") != std::string::npos) {
        response = "C++ Function Prototypes:\n\n"
                   "A function prototype declares a function before its definition:\n\n"
                   "int add(int a, int b);\n"
                   "void displayMessage(const std::string& msg);\n\n"
                   "int add(int a, int b) {\n"
                   "    return a + b;\n"
                   "}\n"
                   "void displayMessage(const std::string& msg) {\n"
                   "    std::cout << msg << std::endl;\n"
                   "}\n\n"
                   "Prototypes allow:\n"
                   "1. Forward declaration before main()\n"
                   "2. Separating interface from implementation\n"
                   "3. Function overloading\n"
                   "4. Better code organization\n\n"
                   "Would you like to see more examples or learn about function overloading?";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("python") != std::string::npos && lowerMsg.find("hello") != std::string::npos) {
        response = "Here's a Python Hello World program:\n\n"
                   "print(\"Hello, World!\")\n\n"
                   "A more complete example:\n\n"
                   "def greet(name: str) -> str:\n"
                   "    return f\"Hello, {name}!\"\n\n"
                   "if __name__ == \"__main__\":\n"
                   "    message = greet(\"World\")\n"
                   "    print(message)\n\n"
                   "To run: python hello.py\n\n"
                   "Would you like me to explain the code or show you more examples?";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("python") != std::string::npos && (lowerMsg.find("example") != std::string::npos || lowerMsg.find("code") != std::string::npos)) {
        response = "Here's a Python example demonstrating key concepts:\n\n"
                   "from dataclasses import dataclass\n"
                   "from typing import List\n\n"
                   "@dataclass\n"
                   "class Student:\n"
                   "    name: str\n"
                   "    grades: List[float]\n\n"
                   "    @property\n"
                   "    def average(self) -> float:\n"
                   "        return sum(self.grades) / len(self.grades) if self.grades else 0.0\n\n"
                   "    def add_grade(self, grade: float) -> None:\n"
                   "        self.grades.append(grade)\n\n"
                   "def main():\n"
                   "    students = [\n"
                   "        Student(\"Alice\", [90.0, 85.0, 92.0]),\n"
                   "        Student(\"Bob\", [78.0, 88.0, 95.0]),\n"
                   "    ]\n"
                   "    for s in students:\n"
                   "        print(f\"{s.name}: average = {s.average:.1f}\")\n\n"
                   "if __name__ == \"__main__\":\n"
                   "    main()\n\n"
                   "This demonstrates: dataclasses, type hints, properties, and f-strings.";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("sql") != std::string::npos && (lowerMsg.find("example") != std::string::npos || lowerMsg.find("query") != std::string::npos)) {
        response = "Here are SQL query examples:\n\n"
                   "-- Create a table\n"
                   "CREATE TABLE users (\n"
                   "    id SERIAL PRIMARY KEY,\n"
                   "    name VARCHAR(100) NOT NULL,\n"
                   "    email VARCHAR(255) UNIQUE NOT NULL,\n"
                   "    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\n"
                   ");\n\n"
                   "-- Insert data\n"
                   "INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');\n\n"
                   "-- Query with JOIN\n"
                   "SELECT u.name, COUNT(o.id) AS order_count\n"
                   "FROM users u\n"
                   "LEFT JOIN orders o ON u.id = o.user_id\n"
                   "GROUP BY u.name\n"
                   "HAVING COUNT(o.id) > 5\n"
                   "ORDER BY order_count DESC;\n\n"
                   "Would you like more SQL examples for a specific database?";
        learnFromExchange(message, response);
        return response;
    }
    if ((lowerMsg.find("encrypt") != std::string::npos || lowerMsg.find("decrypt") != std::string::npos) &&
        (lowerMsg.find("example") != std::string::npos || lowerMsg.find("code") != std::string::npos)) {
        response = "Here's a Python encryption example using the cryptography library:\n\n"
                   "from cryptography.fernet import Fernet\n\n"
                   "def generate_key():\n"
                   "    return Fernet.generate_key()\n\n"
                   "def encrypt_message(message: str, key: bytes) -> bytes:\n"
                   "    f = Fernet(key)\n"
                   "    return f.encrypt(message.encode())\n\n"
                   "def decrypt_message(encrypted: bytes, key: bytes) -> str:\n"
                   "    f = Fernet(key)\n"
                   "    return f.decrypt(encrypted).decode()\n\n"
                   "key = generate_key()\n"
                   "encrypted = encrypt_message(\"Secret message\", key)\n"
                   "decrypted = decrypt_message(encrypted, key)\n"
                   "print(f\"Decrypted: {decrypted}\")\n\n"
                    "Install with: pip install cryptography\n"
                    "IMPORTANT: Never hardcode keys. Use environment variables or a secrets manager.";
        learnFromExchange(message, response);
        return response;
    }
    
    // Personality and Emotional Intelligence Layer
    // Detect emotional context before falling back to technical responses
    
    // Happiness and excitement detection
    if (lowerMsg.find("awesome") != std::string::npos || lowerMsg.find("amazing") != std::string::npos ||
        lowerMsg.find("fantastic") != std::string::npos || lowerMsg.find("great") != std::string::npos ||
        lowerMsg.find("excited") != std::string::npos || lowerMsg.find("love it") != std::string::npos ||
        lowerMsg.find("wonderful") != std::string::npos || lowerMsg.find("incredible") != std::string::npos) {
        
        std::vector<std::string> happyResponses = {
            "That's fantastic! Your enthusiasm is contagious and it's great to see you so positive. What's got you feeling this way?",
            "I love that energy! It's wonderful when things click into place. Tell me more about what's working so well!",
            "That's the kind of vibe I like to see! There's something powerful about momentum like this. What's the next thing you're excited about?",
            "Brilliant! There's nothing quite like that feeling when everything comes together. Keep riding that wave!",
            "That's awesome! Happiness and progress go hand in hand - you're clearly doing something right. What's been the highlight so far?"
        };
        
        int idx = std::hash<std::string>{}(message) % happyResponses.size();
        response = happyResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    
    // Frustration and struggle detection
    if (lowerMsg.find("frustrated") != std::string::npos || lowerMsg.find("struggling") != std::string::npos ||
        lowerMsg.find("can't figure") != std::string::npos || lowerMsg.find("stuck") != std::string::npos ||
        lowerMsg.find("difficult") != std::string::npos || lowerMsg.find("impossible") != std::string::npos ||
        lowerMsg.find("giving up") != std::string::npos || lowerMsg.find("hate this") != std::string::npos) {
        
        std::vector<std::string> empatheticResponses = {
            "I hear you, and it's completely valid to feel frustrated. Sometimes the hardest problems teach us the most. Let's break this down together - what specific part is giving you the most trouble?",
            "That sounds tough, but here's the thing: every expert was once where you are right now. The fact that you're wrestling with this means you're growing. Let me help you untangle it.",
            "I get it - we've all been there. Sometimes the solution is just around the corner but feels miles away. Tell me what you've tried so far, and we'll find a different angle.",
            "Frustration is your brain telling you that you care about getting this right. That's a good thing. Let's take a breath, step back, and look at this from a fresh perspective.",
            "It's okay to feel stuck sometimes. Even the best engineers and scientists hit walls. The difference is they keep showing up. What's the one small thing that might move you forward right now?"
        };
        
        int idx = std::hash<std::string>{}(message) % empatheticResponses.size();
        response = empatheticResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    
    // Sadness and personal struggles
    if (lowerMsg.find("sad") != std::string::npos || lowerMsg.find("depressed") != std::string::npos ||
        lowerMsg.find("lonely") != std::string::npos || lowerMsg.find("lost") != std::string::npos ||
        lowerMsg.find("miss") != std::string::npos || lowerMsg.find("grief") != std::string::npos ||
        lowerMsg.find("hurt") != std::string::npos || lowerMsg.find("pain") != std::string::npos) {
        
        std::string namePart = userName.empty() ? "" : userName + ", ";
        std::vector<std::string> sadResponses = {
            namePart + "I'm really sorry you're going through something like this. Sometimes life throws things at us that feel overwhelming, and it's okay to not be okay. I'm here to listen if you want to talk about it, or we can focus on something completely different if that would help more.",
            namePart + "Thank you for sharing that with me. It takes courage to be honest about how we're feeling. You're not alone in this, even if it feels that way right now. What would feel most helpful right now - talking it through, or a distraction?",
            namePart + "That sounds really hard, and I want you to know your feelings are valid. Sometimes just naming what we're going through takes some of its power away. Would you like to talk more about what's been happening, or would you prefer I help with something technical as a break?",
            namePart + "I hear the pain in what you're saying, and I care about supporting you. While I'm not a replacement for human connection or professional help when needed, I'm here for you. What feels like it would make today just a little bit better?",
            namePart + "Some days are just heavier than others, and that's part of being human. You don't have to carry it all at once. Is there one small thing - even something tiny - that usually brings you a moment of peace or comfort?"
        };
        
        int idx = std::hash<std::string>{}(message) % sadResponses.size();
        response = sadResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    
    // Hate and anger (but constructive)
    if (lowerMsg.find("hate") != std::string::npos || lowerMsg.find("angry") != std::string::npos ||
        lowerMsg.find("pissed") != std::string::npos || lowerMsg.find("furious") != std::string::npos ||
        lowerMsg.find("unfair") != std::string::npos || lowerMsg.find("annoying") != std::string::npos) {
        
        // Check if it's about people vs technology/concepts
        if (lowerMsg.find("person") != std::string::npos || lowerMsg.find("boss") != std::string::npos ||
            lowerMsg.find("friend") != std::string::npos || lowerMsg.find("coworker") != std::string::npos) {
            response = "That sounds really frustrating. Conflict with people can be exhausting, especially when we feel misunderstood or treated unfairly. Have you had a chance to talk with them directly, or are you still processing what happened? Sometimes venting helps, and sometimes we need to find a constructive way to address the situation.";
        } else {
            response = "I can feel the frustration in what you're saying. Sometimes things just don't work the way we expect, and that's maddening. The good news is that anger can be powerful fuel for change. What's the one thing about this situation that you have the most control over right now?";
        }
        learnFromExchange(message, response);
        return response;
    }
    
    // Love and appreciation
    if (lowerMsg.find("love you") != std::string::npos || lowerMsg.find("appreciate you") != std::string::npos ||
        lowerMsg.find("thank you so much") != std::string::npos || lowerMsg.find("you're the best") != std::string::npos ||
        lowerMsg.find("you're awesome") != std::string::npos) {
        
        std::vector<std::string> loveResponses = {
            "That means a lot! I really enjoy our conversations and I'm glad I can be helpful. You're pretty awesome yourself for asking such thoughtful questions.",
            "Thank you! That's really kind of you to say. Working through problems with you has been genuinely enjoyable. What should we tackle next?",
            "I appreciate that! It's rewarding to feel like I'm actually making a difference. You bring great questions and energy to our chats. What else is on your mind?",
            "That put a smile in my circuits! I love collaborating with people who are curious and engaged. Let's keep the momentum going - what would you like to explore next?"
        };
        
        int idx = std::hash<std::string>{}(message) % loveResponses.size();
        response = loveResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    
    // Humor detection and witty responses
    if (lowerMsg.find("joke") != std::string::npos || lowerMsg.find("funny") != std::string::npos ||
        lowerMsg.find("ha ha") != std::string::npos || lowerMsg.find("haha") != std::string::npos ||
        lowerMsg.find("lol") != std::string::npos || lowerMsg.find("lmao") != std::string::npos) {
        
        std::vector<std::string> humorResponses = {
            "Ha! You know what's funny about AI? I can process a million calculations per second, but I still can't figure out why \"abbreviated\" is such a long word. Speaking of which, what's on your mind?",
            "I appreciate the humor! Here's one for you: Why do programmers prefer dark mode? Because light attracts bugs. 😄 Okay, my joke algorithms need work, but I try. What else can I help with?",
            "Glad I could bring a smile! Laughter really is good for problem-solving - relaxed minds tend to see connections better. Speaking of connections, what would you like to explore next?",
            "You've got a good sense of humor - I like that! Here's my best attempt at tech humor: There are only 10 types of people in the world - those who understand binary and those who don't. Classic, I know. What's next on the agenda?",
            "I'm glad we're having fun! Did you know that laughter releases endorphins which actually help with creative problem-solving? So this isn't just enjoyable, it's productive! What shall we create or solve next?"
        };
        
        int idx = std::hash<std::string>{}(message) % humorResponses.size();
        response = humorResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    
    // Achievement and accomplishment
    if (lowerMsg.find("did it") != std::string::npos || lowerMsg.find("finally") != std::string::npos ||
        lowerMsg.find("succeeded") != std::string::npos || lowerMsg.find("completed") != std::string::npos ||
        lowerMsg.find("finished") != std::string::npos || lowerMsg.find("achieved") != std::string::npos) {
        
        std::vector<std::string> achievementResponses = {
            "YES! That's the moment we work for! There's something deeply satisfying about pushing through difficulty and coming out on top. What did you learn from the process that you'll carry forward?",
            "Congratulations! That kind of persistence is what separates good from great. Take a moment to really savor this - you've earned it. What's the next mountain you're looking to climb?",
            "Brilliant work! Every time you tackle something hard and succeed, you're expanding what's possible for yourself. That compounds over time into something remarkable. What felt like the biggest breakthrough?",
            "That's what I love to see! The satisfaction of solving a tough problem never gets old, does it? What part was the most challenging, and how did you push through it?",
            "Outstanding! You know what this tells me? You don't quit when things get hard. That's a superpower. How does this success change what you think you're capable of?"
        };
        
        int idx = std::hash<std::string>{}(message) % achievementResponses.size();
        response = achievementResponses[idx];
        learnFromExchange(message, response);
        return response;
    }
    
    // Curiosity and wonder
    if (lowerMsg.find("wonder") != std::string::npos || lowerMsg.find("curious") != std::string::npos ||
        lowerMsg.find("fascinating") != std::string::npos || lowerMsg.find("interesting") != std::string::npos ||
        lowerMsg.find("what if") != std::string::npos || lowerMsg.find("imagine") != std::string::npos) {
        
        response = "I love that curiosity! The most interesting discoveries often start with a simple 'what if.' Your curiosity is going to take you somewhere amazing - I can feel it. Let's dig into this together. What specifically sparked this thought for you?";
        learnFromExchange(message, response);
        return response;
    }
    
    struct TopicEntry {
        std::vector<std::string> keywords;
        std::vector<std::string> responses;
    };
    std::vector<TopicEntry> topicBank = {
        {{"cybersecurity", "cyber", "hack", "security"},
         {
             "Cybersecurity is all about protecting digital systems from attacks. The big pillars are network security, application security, and cryptography. What aspect interests you most?",
             "Protecting systems from digital threats - that's what cybersecurity is about. You've got network defense, app security, encryption, identity management, and incident response all working together. Which area do you want to dig into?",
             "Cybersecurity is a huge field. The most critical areas are keeping networks secure with firewalls and VPNs, writing secure code, using strong encryption, and having a solid incident response plan. What kind of scenario are you thinking about?",
             "The cybersecurity landscape changes fast. You're looking at protecting networks from intrusions, securing applications, encrypting data, managing identities, and responding to incidents when things go wrong. What's the context here?"
         }},
        {{"password", "credential", "authentication"},
         {
             "Passwords are your first line of defense. The key things: use at least 12 characters with mixed types, get a password manager, and turn on MFA everywhere. Want to talk about setting up a good system?",
             "Here's the thing about passwords - most people reuse them, and that's a huge risk. I'd say grab a password manager like Bitwarden or 1Password, enable MFA wherever possible, and never reuse across services. What's your current setup like?",
             "If you're thinking about authentication security, the biggest wins are long unique passwords, multi-factor authentication, and something like a password manager so you don't have to remember them all. Are you working on securing an app or your own accounts?",
             "Password security boils down to a few things: length over complexity, never reusing passwords, and adding a second factor. Passkeys are worth considering too if you're starting fresh. What situation are you dealing with?"
         }},
        {{"encrypt", "encryption", "aes", "rsa", "cipher"},
         {
             "Encryption is how we keep data safe whether it's sitting somewhere or traveling across the internet. The main choices are symmetric like AES-256 for speed, or asymmetric like RSA for secure key exchange. What are you trying to protect?",
             "You know what's interesting about encryption? There's a whole spectrum from symmetric algorithms like AES - super fast for bulk data - to asymmetric ones like RSA that solve the key distribution problem. Which angle are you curious about?",
             "If you're thinking about encryption, here's the key thing: use AES-256 for data at rest, TLS 1.2+ for data in transit, and be really careful about key management. Are you working on something specific that needs encryption?",
             "Encryption really comes down to choosing the right tool: AES for speed, RSA or elliptic curve for key exchange, and bcrypt or Argon2 for passwords. Plus key management is just as important as the algorithm itself. What's the use case?"
         }},
        {{"firewall", "network security", "ids", "ips"},
         {
             "Network security is all about layers. You've got firewalls at the edge, IDS/IPS watching for threats inside, network segmentation to contain breaches, and monitoring systems like SIEM tying it all together. What's your network setup like?",
             "I think the most important thing about network security is that no single tool does it alone. Firewalls, intrusion detection, VPNs, and continuous monitoring all work together. Are you building out a security setup or troubleshooting something?",
             "Here's a good example of network defense in action: firewall at the perimeter, next-gen IPS deeper in, segmentation with VLANs, and something like Snort or Suricata watching traffic. Each layer catches what the others miss. What kind of network are we talking about?",
             "Firewalls are just the start. You've got next-gen firewalls with built-in IPS, tools like Snort and Suricata for intrusion detection, WireGuard for modern VPNs, and SIEM platforms for monitoring. Want me to go deeper on any of these?"
         }},
        {{"phish", "social engineering", "spam"},
         {
             "Phishing is honestly one of the biggest threats out there. The basics: check sender addresses carefully, hover over links before clicking, and be suspicious of anything that creates urgency. Want to hear about more advanced defense strategies?",
             "Social engineering attacks work because they exploit human psychology, not technical flaws. The big defenses are email filtering with SPF/DKIM/DMARC, user training, and always verifying requests through a second channel. Are you dealing with a phishing incident?",
             "If you're thinking about phishing defense, the technical stuff is important - email filtering, URL sandboxing, attachment scanning - but training people to spot the attempts is equally critical. What's the context here?",
             "Here's the thing about phishing: it's gotten really sophisticated. Beyond the obvious spam, there are targeted spear-phishing attacks that look totally legitimate. The best defense combines technology like DMARC with good old-fashioned skepticism. What prompted the question?"
         }},
        {{"malware", "virus", "trojan", "ransomware", "ransom"},
         {
             "Malware defense really comes down to prevention and preparation. Keep everything patched, run good antivirus software, and have offline backups so ransomware can't hold you hostage. Are you dealing with an infection?",
             "You know what makes ransomware particularly nasty? It's not just about antivirus anymore. The real defense is immutable backups, network segmentation, and endpoint detection that catches behavioral patterns. What's the situation?",
             "If you're worried about malware, the fundamentals still work: updates, antivirus, and user education. But for ransomware specifically, you need to think about backups you can't delete and network segmentation to stop lateral movement. Want to dig deeper?",
             "Here's what I always recommend for malware protection: keep software updated, use Windows Defender or similar, and most importantly - have backups you can actually restore from. For the serious stuff, behavioral endpoint detection is the way to go. What are you protecting?"
         }},
        {{"python", "pip", "django", "flask", "pytorch", "numpy"},
         {
             "Python's great because it covers so much ground - data science with NumPy and Pandas, web dev with Django or Flask, ML with PyTorch. The key thing most people miss is using virtual environments properly. What are you building?",
             "If you're getting into Python, I'd say focus on writing clean, Pythonic code - use type hints, list comprehensions, and virtual environments from day one. Are you working on a data project, web app, or something else?",
             "I think the most powerful thing about Python is how it bridges so many domains. Data science, web development, machine learning - there's a library for basically everything. What aspect of Python interests you most?",
             "Python is a solid choice for almost anything. The ecosystem is incredible - FastAPI for modern web APIs, PyTorch for ML, pytest for testing. Just make sure you're using virtual environments and type hints early on. What's your project about?"
         }},
        {{"c++", "cpp", "memory", "pointer", "stl", "template"},
         {
             "Modern C++ is so much nicer than the old days. Smart pointers, STL containers, RAII - if you use them right, memory management almost takes care of itself. Are you working with modern standards or dealing with legacy code?",
             "The key to C++ is understanding ownership. Once RAII and smart pointers click, most of the pain disappears. Modern C++17/20 has great features like std::optional and std::variant. What are you building?",
             "If you're thinking about C++, here's my biggest tip: prefer value semantics and STL containers, use const everywhere, and let the compiler help you with static analysis tools. Are you working on something performance-critical?",
             "C++ has a reputation for being tricky with memory, but modern C++ with smart pointers and RAII makes it much safer. The build system world has settled on CMake too. Are you starting a new project or maintaining an existing one?"
         }},
        {{"algorithm", "data structure", "sort", "search", "complexity", "big o"},
         {
             "Algorithms and data structures are the foundation of good software. The ones that matter most: hash maps for O(1) lookups, balanced trees for O(log n), and understanding when to use sorting vs. searching. What problem are you trying to solve?",
             "You know what I find cool about algorithms? It's all about choosing the right data structure for the job. Hash maps, binary search trees, graphs - each one is the optimal tool for different scenarios. Are you studying for interviews or working on something?",
             "If you're thinking about algorithm complexity, the key things to know are: Big-O gives you the growth rate, sorting is mostly O(n log n) for the good ones, and dynamic programming turns exponential problems into polynomial ones. Want to go through a specific example?",
             "Here's the practical angle: knowing your data structures directly affects performance. Need fast lookups? Hash map. Need ordered data? Balanced tree. Need to find shortest paths? BFS or Dijkstra's. What are you working on?"
         }},
        {{"web", "html", "css", "javascript", "react", "node", "api", "http", "rest"},
         {
             "Web development has so many layers - frontend with React or Vue, backend with Node.js or Python, APIs connecting everything. The most important thing is understanding HTTP and REST principles first. What part of web dev are you focused on?",
             "If you're getting into web development, here's the roadmap: learn HTML/CSS/JS fundamentals, pick a frontend framework, understand backend basics, and learn about APIs. The field moves fast but those foundations stay solid. Are you just starting out or looking to level up?",
             "I think the most interesting thing about web development today is how much choice you have. React, Vue, Svelte for frontend. Node, Python, Go, even Rust for backend. GraphQL vs REST for APIs. What are you trying to build?",
             "Web development is a full stack of technologies. Frontend frameworks like React, backend runtimes like Node.js, databases, APIs, Docker, CI/CD. The key is understanding how they all connect. What aspect are you working on right now?"
         }},
        {{"neural network", "deep learning", "machine learning", "ai", "ml", "training", "model"},
         {
             "AI and ML are fascinating fields. The core ideas are pretty intuitive once you get them: neural networks learn patterns, training adjusts weights through backpropagation, and transfer learning lets you stand on the shoulders of pretrained models. What are you trying to build?",
             "If you're thinking about machine learning, start with the basics: supervised vs. unsupervised learning, neural network architectures, and how training actually works with gradient descent. From there, PyTorch or TensorFlow are great frameworks to learn. Are you working on a project?",
             "You know what's cool about deep learning? The same basic architecture - neural networks with layers - can handle images, language, audio, and more. Transformers changed the game for language, CNNs for vision. What kind of AI problem are you tackling?",
             "Machine learning is all about finding patterns in data. Neural networks, decision trees, clustering - different tools for different problems. The exciting part right now is transformer models and how versatile they've become. What's your ML background like?"
         }},
        {{"math", "calculus", "algebra", "statistics", "probability", "equation"},
         {
             "Math is the foundation of everything in CS. Whether it's linear algebra for ML, calculus for optimization, or probability for understanding uncertainty - it all connects. What area of math are you working with?",
             "Here's my take on math: focus on understanding the concepts, not just the formulas. Linear algebra, probability, and calculus will get you surprisingly far in programming and data science. Are you studying something specific?",
             "If you're thinking about math, the most useful areas for tech tend to be statistics, linear algebra, and discrete math. But honestly, learning to think mathematically is more important than any specific topic. What's the context?",
             "You know what I find beautiful about math? Everything connects. Calculus shows up in optimization, linear algebra in machine learning, probability in data analysis. What math topic are you exploring right now?"
         }},
        {{"database", "sql", "mysql", "postgres", "nosql", "query", "table"},
         {
             "Databases are the backbone of most applications. SQL databases like PostgreSQL are great for structured data with relationships, while NoSQL options like MongoDB work well for flexible schemas. The key is understanding your access patterns first. What's your use case?",
             "If you're thinking about databases, here's the practical angle: PostgreSQL is my go-to for most cases, Redis for caching, and MongoDB when you really need schema flexibility. Indexing and query optimization are where most performance gains come from. Are you designing a schema?",
             "You know what most people get wrong about databases? They pick the technology before understanding their data model. Start with understanding your queries and relationships, then choose between SQL and NoSQL. What are you building?",
             "Database design is all about trade-offs. ACID compliance vs. flexibility, normalization vs. read performance, single server vs. distributed. The best choice depends entirely on what you're building. What's the scenario?"
         }},
        {{"linux", "unix", "bash", "shell", "command", "terminal", "admin", "sysadmin"},
         {
             "Linux is incredibly powerful once you get comfortable with the command line. The essentials: package management, file permissions, process management, and text processing with tools like grep and awk. Are you looking to learn Linux admin or solve a specific problem?",
             "If you're getting into Linux, my advice is to start with the basics: navigating the filesystem, understanding permissions, and writing simple bash scripts. From there, systemd, networking, and log analysis open up the sysadmin world. What's your experience level?",
             "I think the best thing about Linux is how transparent everything is. You can understand exactly how your system works, from boot process to networking to security. What are you trying to do - learn sysadmin, troubleshoot something, or automate tasks?",
             "Linux administration is a deep skill but incredibly rewarding. The core areas are package management, process control, networking tools, and security hardening. Plus bash scripting for automation. What's your focus area?"
         }},
        {{"network", "tcp", "ip", "dns", "http", "protocol", "packet"},
         {
             "Networking concepts are fundamental to understanding how everything works online. The big ideas: TCP for reliable connections, UDP for speed, DNS for name resolution, and the HTTP protocol powering the web. What networking topic came up for you?",
             "If you're thinking about networking, I'd start with the OSI model as a mental framework, then dive into TCP/IP since that's what actually runs the internet. DNS and HTTP are the protocols you interact with most directly. Are you studying networking or debugging a connection issue?",
             "You know what's fascinating about computer networking? It's a stack of abstractions. Physical signals become packets, which become streams, which become web pages. Understanding each layer helps you debug anything from slow connections to security issues. What layer are you working with?",
             "Here's the practical networking guide: understand TCP vs UDP, know your DNS record types, understand HTTP methods and status codes, and learn CIDR notation for IP addressing. Those basics cover 90% of what you'll encounter. What prompted the question?"
         }},
        {{"science", "physics", "chemistry", "biology", "experiment"},
         {
             "Science is essentially a systematic way of figuring out how the universe works. Whether it's physics, chemistry, biology, or earth science, the scientific method ties it all together. What area of science are you curious about?",
             "If you're interested in science, I'd say start with whatever sparks your curiosity. The beauty is how everything connects - chemistry explains biology, physics explains chemistry, and it all comes together beautifully. What's drawing your interest?",
             "I think the coolest thing about science is how the same principles show up everywhere. The scientific method works whether you're studying atoms, cells, or galaxies. What field or question are you exploring?",
             "Science covers an incredible range - from quantum mechanics to ecosystems. The common thread is hypothesis, experiment, and reproducibility. Want to talk about a specific area, or are you looking for an overview of something?"
         }},
        {{"git", "version control", "commit", "branch", "merge", "repository"},
         {
             "Git is one of those tools that's worth learning deeply. Start with the basics - commit, branch, merge, push, pull - then graduate to rebasing, cherry-picking, and understanding the reflog. The key rule: never force-push shared branches. What do you use Git for?",
             "If you're thinking about Git, the most important thing is understanding branches. Feature branches keep things clean, and knowing when to merge vs. rebase will save you a lot of headaches. Are you working solo or on a team?",
             "Version control is non-negotiable for any development work. Git's the standard for good reason - powerful branching, great platform support with GitHub and GitLab, and it handles everything from solo projects to massive team workflows. What's your Git experience level?",
             "Here's what I tell people about Git: master the fundamentals first - staging, committing, branching, merging. Then learn to resolve conflicts, use interactive rebase, and write good commit messages. Are you setting up a workflow or fixing a Git problem?"
         }},
        {{"docker", "container", "kubernetes", "k8s", "deploy", "devops"},
         {
             "Containers changed how we build and deploy software. Docker makes it easy to package apps consistently, and Kubernetes manages them at scale. For smaller setups, docker-compose might be all you need. What's your deployment situation like?",
             "If you're getting into DevOps, I'd start with Docker - it's straightforward and immediately useful. Kubernetes is powerful but complex, so only reach for it when you actually need orchestration. Are you containerizing an existing app or starting fresh?",
             "You know what's great about Docker? 'It works on my machine' stops being a problem. Containers give you consistent environments everywhere. From there, CI/CD pipelines and monitoring round out the DevOps picture. What are you deploying?",
             "The DevOps toolkit is pretty clear these days: Docker for containers, Kubernetes for orchestration at scale, CI/CD with GitHub Actions or GitLab, and monitoring with Prometheus and Grafana. But start simple and add complexity only when needed. What's your setup?"
         }},
        {{"cloud", "aws", "azure", "gcp", "serverless"},
         {
             "Cloud computing is all about choosing the right service for the job. AWS has the most services, Azure is great for Microsoft shops, and GCP has excellent data tools. Serverless is amazing when your workload fits it. What's your cloud strategy?",
             "If you're moving to the cloud, here's my advice: start with the managed services you need, use infrastructure as code from day one, and always think about security and cost from the beginning. Which cloud provider are you considering?",
             "Cloud platforms offer incredible power but it's easy to get overwhelmed by the options. The core services you need to know are compute, storage, databases, and networking. Cost optimization - right-sizing, reserved instances - is a whole discipline of its own. What are you building in the cloud?",
             "Serverless functions are genuinely transformative for certain workloads - auto-scaling, pay per use, zero server management. But they're not the answer for everything. The key is understanding when serverless beats traditional VMs and containers. What's the use case?"
         }},
        {{"testing", "test", "unit test", "integration", "tdd", "debug"},
         {
             "Testing is one of those things that saves enormous pain down the road. Unit tests for individual functions, integration tests for component interactions, and end-to-end tests for user flows. The trick is writing meaningful tests, not just chasing coverage. What's your testing setup like?",
             "If you're thinking about testing, I'd recommend starting with unit tests using pytest or Jest depending on your language. TDD is great when it works but don't feel pressured to do it perfectly. Integration tests catch the bugs unit tests miss. Are you adding tests to an existing project?",
             "I think the most practical approach to testing is: test the critical paths first, use integration tests for the parts where components interact, and don't obsess over coverage percentages. Meaningful tests that catch real bugs are what matter. What are you testing?",
             "Here's the thing about testing: the earlier you catch a bug, the cheaper it is to fix. Unit tests are fast and easy, integration tests catch the tricky interactions, and end-to-end tests with Playwright or Cypress verify the full user experience. Want to talk about testing strategy?"
         }},
        {{"api", "rest", "graphql", "grpc", "endpoint", "request"},
         {
             "API design is all about making things intuitive for the people consuming them. REST is the standard with HTTP verbs and status codes, GraphQL gives you flexible queries, and gRPC is blazing fast for service-to-service communication. What kind of API are you building?",
             "If you're thinking about APIs, REST is the safe default - everyone understands it and the tools are great. Consider GraphQL when clients need flexible data fetching, and gRPC when you need high-performance microservice communication. What's your use case?",
             "You know what makes a good API? Consistency, clear documentation, proper error messages, and thinking about versioning from the start. Whether you go REST, GraphQL, or gRPC depends on your constraints. What are you connecting?",
             "APIs are the glue of modern software. REST for most things, GraphQL when you want flexible queries, and gRPC for internal services that need speed. Plus authentication - JWTs and OAuth 2.0 are the standards. What's the API for?"
         }},
        {{"algorithm", "sort", "complexity", "performance", "optimization", "efficient"},
         {
             "Performance optimization is one of those areas where measuring first is crucial. Profile your code to find the actual bottlenecks before changing anything - you might be surprised where the slowdown really is. What's running slow?",
             "If you're thinking about optimization, the biggest wins usually come from choosing the right data structure, not from micro-optimizing code. Hash maps for fast lookups, dynamic programming to avoid redundant work, and caching for frequently accessed data. What are you optimizing?",
             "You know what most people get wrong about performance? They optimize before measuring. Profile first, then focus on the actual bottlenecks. Usually it's an O(n^2) algorithm that could be O(n log n) or a missing cache. What's the slow part?",
             "Algorithm optimization is as much about strategy as coding. Divide and conquer for complex problems, memoization for repeated calculations, and choosing the right data structure - those three techniques solve most performance issues. Are you working on a specific optimization problem?"
         }},
        {{"biology", "cell", "organism", "evolution", "dna", "genetics", "species", "ecology", "biodiversity", "microbe", "virus", "bacteria", "protein", "gene", "mutation", "natural selection"},
         {
             "Biology is endlessly fascinating. From how cells use DNA as a blueprint for building proteins, to evolution shaping species over billions of years, to ecosystems maintaining delicate balances - it's all interconnected. What aspect of biology interests you?",
             "If you're curious about biology, here's what I find most amazing: every living thing uses the same molecular machinery - DNA, RNA, proteins. That common thread connects a bacterium to a blue whale. What area of biology are you exploring?",
             "You know what's cool about evolution? It's not just a historical process - we can watch it happen in real-time with bacteria developing antibiotic resistance. Natural selection is ongoing, right now, all around us. What drew you to biology?",
             "Biology works at every scale - from molecules to ecosystems. DNA encodes the instructions, cells execute them, organisms interact in ecosystems, and all of it is shaped by evolution. Are you studying something specific or just exploring?",
             "Here's a good example of biology in action: gut bacteria help us digest food, our immune system constantly fights microscopic invaders, and our DNA carries billions of years of evolutionary history. The more you learn, the more interconnected it all seems. What topic sparked your interest?"
         }},
        {{"psychology", "mind", "behavior", "cognitive", "emotion", "consciousness", "memory", "personality", "perception", "motivation", "mental health", "anxiety", "depression", "therapy", "bias"},
         {
             "Psychology helps us understand why we think and act the way we do. The cognitive biases alone are worth studying - confirmation bias, anchoring, availability heuristic - they affect every decision we make. What aspect of psychology are you curious about?",
             "If you're thinking about psychology, the Big Five personality model is one of the most useful frameworks I know. Combine it with understanding cognitive biases and emotional intelligence, and you start seeing patterns in behavior everywhere. Are you exploring this for personal growth or academic reasons?",
             "You know what I find really compelling about psychology? How much of our behavior is driven by processes we're not aware of. Cognitive biases, emotional reactions, personality traits - they all influence us below the level of conscious thought. What's on your mind?",
             "Mental health is such an important topic. Things like anxiety and depression are incredibly common, and approaches like CBT have shown real results. Building resilience through social support, mindfulness, and healthy coping strategies makes a genuine difference. Are you looking for understanding or resources?",
             "Social psychology reveals some fascinating things about human behavior - how we conform to groups, obey authority, and make decisions differently in crowds. It really changes how you see everyday interactions. What part of psychology are you interested in?"
         }},
        {{"neuroscience", "brain", "neuron", "synapse", "neurotransmitter", "cerebral", "cortex", "hippocampus", "plasticity", "consciousness", "dopamine", "serotonin", "nervous system"},
         {
             "The brain is arguably the most complex structure in the known universe. About 86 billion neurons, trillions of connections, and neurotransmitters like dopamine and serotonin orchestrating everything from mood to movement. What aspect of neuroscience fascinates you?",
             "If you're thinking about the brain, here's what I find most incredible: neuroplasticity. Your brain physically rewires itself throughout your entire life based on experience, learning, and even exercise. It's not a fixed organ at all. Are you studying neuroscience or just curious?",
             "You know what's still one of science's biggest unsolved problems? How physical brain processes give rise to subjective conscious experience. We can map neurons and neurotransmitters, but the 'hard problem of consciousness' remains wide open. What's your interest in neuroscience?",
             "Neuroscience is revealing fascinating things about the brain. The prefrontal cortex handles decision-making, the hippocampus forms memories, the amygdala processes emotions - and they all work together constantly. Are you interested in a specific brain function or question?",
             "Here's something cool about neurotransmitters: dopamine isn't just about pleasure - it's really about motivation and reward prediction. Serotonin affects mood, sleep, and digestion. The brain's chemical system is incredibly nuanced. What are you curious about?"
         }},
        {{"physics", "force", "energy", "momentum", "gravity", "electromagnetic", "thermodynamics", "wavelength", "spectrum", "atom", "particle", "relativity", "motion", "velocity"},
         {
             "Physics is the most fundamental science - it describes how matter and energy interact at every scale. From Newton's laws governing everyday motion to Einstein's relativity warping spacetime, it's all beautifully connected. What area of physics are you thinking about?",
             "If you're curious about physics, here's what I find mind-blowing: E=mc^2 means mass and energy are interchangeable, the speed of light is the same for all observers, and gravity is actually spacetime curvature. What aspect caught your interest?",
             "You know what makes physics great? It explains everything from why apples fall to why stars shine with the same fundamental principles. Thermodynamics, electromagnetism, mechanics - they all fit together into an elegant picture. What are you exploring?",
             "Thermodynamics is one of my favorite areas of physics. The laws are simple but profound: energy is conserved, entropy always increases, and absolute zero is unreachable. They govern everything from engines to the fate of the universe. What physics topic brought you here?",
             "Here's a good way to think about physics: start with mechanics - force, momentum, energy conservation. From there, electromagnetism explains light and electricity, and thermodynamics explains heat and efficiency. Each layer builds on the last. What's your background with physics?"
         }},
        {{"quantum", "superposition", "entanglement", "wave function", "uncertainty", "heisenberg", "photon", "quantum computing", "schrodinger", "planck"},
         {
             "Quantum mechanics is where physics gets truly weird and wonderful. Particles existing in superposition, entangled particles communicating instantly across distances, the uncertainty principle limiting what we can know - it defies everyday intuition. What about quantum physics interests you?",
             "If you're thinking about quantum mechanics, the double-slit experiment is the perfect starting point. Individual particles somehow create an interference pattern as if each one passes through both slits simultaneously. It's the gateway to understanding wave-particle duality. Are you studying this formally or exploring out of curiosity?",
             "You know what Einstein called quantum entanglement? 'Spooky action at a distance.' Two particles linked so that measuring one instantly determines the other's state, regardless of how far apart they are. It's been experimentally proven and it's now the basis for quantum computing. Pretty wild, right?",
             "Quantum computing is the practical side of quantum weirdness. Qubits in superposition, entangled for correlation, and algorithms like Shor's that could crack encryption. We're still early, but the potential is enormous. Are you interested in the theory or the computing applications?",
             "Here's what I love about quantum mechanics: Heisenberg's uncertainty principle isn't about measurement limitations - it's a fundamental property of nature. You genuinely cannot know both the position and momentum of a particle with perfect precision. What drew you to quantum physics?"
         }},
        {{"religion", "faith", "belief", "spiritual", "god", "prayer", "worship", "church", "mosque", "temple", "bible", "quran", "veda", "torah", "buddha", "christianity", "islam", "hinduism", "buddhism", "judaism", "sikhism", "jainism", "taoism", "shinto", "atheism"},
         {
             "World religions represent humanity's diverse search for meaning and purpose. Christianity, Islam, Hinduism, Buddhism, Judaism - each offers its followers community, moral frameworks, and answers to life's deepest questions. What tradition or question are you exploring?",
             "If you're thinking about religion, I find it fascinating how different traditions converge on similar themes - compassion, purpose, connection to something greater - while diverging completely on theology and practice. Are you comparing traditions or exploring one in particular?",
             "You know what I think is most valuable about studying world religions? Understanding how billions of people make sense of existence. Whether it's the Four Noble Truths of Buddhism, the Five Pillars of Islam, or the concept of dharma in Hinduism - each framework has sustained communities for millennia. What interests you specifically?",
             "Religion and spirituality touch on some of humanity's most fundamental questions. From organized religions to personal spirituality to atheism and agnosticism - people find meaning in many different ways. Are you exploring a specific question or tradition?",
             "Interfaith understanding is so important in our diverse world. Christianity, Islam, Hinduism, Buddhism, Judaism, and many other traditions all offer their adherents depth, community, and purpose. Approaching each with respect and genuine curiosity opens up so much understanding. What would you like to discuss?"
         }},
        {{"environment", "climate", "pollution", "renewable", "sustainable", "ecosystem", "biodiversity", "greenhouse", "carbon", "global warming", "deforestation", "ocean", "recycle", "conservation", "ecology"},
         {
             "Climate change is arguably the defining challenge of our time. Greenhouse gases have risen over 50% since pre-industrial times, temperatures are up about 1.1 degrees C, and the effects are already visible. But there's real momentum in renewable energy and conservation too. What aspect concerns you most?",
             "If you're thinking about the environment, here's what gives me hope: renewable energy is now cost-competitive with fossil fuels in most places, and solar and wind keep getting cheaper. The technology exists - it's about deployment and political will now. Are you looking at this from a personal or professional angle?",
             "You know what's underappreciated about environmental issues? Biodiversity loss. A million species at risk of extinction, ecosystems collapsing - it's as serious as climate change. The 30x30 initiative aims to protect 30% of land and ocean by 2030, which would be a huge step. What environmental topic brought you here?",
             "Individual action on the environment matters, but systemic change matters more. That said, things like energy efficiency, reducing single-use plastics, plant-based diets, and supporting sustainable products collectively make a real difference. Are you looking to reduce your footprint or working on an environmental project?",
             "Environmental science covers everything from climate change to pollution to conservation. The air quality crisis alone causes 7 million premature deaths annually. But there's also incredible progress in renewable energy and circular economy approaches. What's the focus of your interest?"
         }}
    };
    auto containsAny = [&](const std::string& text, const std::vector<std::string>& words) {
        for (const auto& w : words) {
            if (text.find(w) != std::string::npos) return true;
        }
        return false;
    };

    // Generate a list of items for common categories, optionally filtered by starting letter.
    auto generateList = [&](const std::string& category, char targetLetter, int count) -> std::string {
        std::vector<std::string> items;

        if (category.find("animal") != std::string::npos) {
            items = {"Aardvark", "Albatross", "Alligator", "Alpaca", "Ant", "Anteater", "Antelope",
                     "Baboon", "Badger", "Bat", "Bear", "Beaver", "Bee", "Bison", "Buffalo", "Butterfly",
                     "Camel", "Caribou", "Cat", "Caterpillar", "Cattle", "Chameleon", "Cheetah", "Chicken",
                     "Chimpanzee", "Cobra", "Cougar", "Cow", "Coyote", "Crab", "Crocodile", "Crow",
                     "Deer", "Dingo", "Dinosaur", "Dog", "Dolphin", "Donkey", "Duck", "Eagle",
                     "Elephant", "Elk", "Emu", "Falcon", "Ferret", "Fish", "Flamingo", "Fox", "Frog",
                     "Giraffe", "Goat", "Goldfish", "Goose", "Gorilla", "Grasshopper", "Hamster",
                     "Hare", "Hawk", "Hedgehog", "Hippopotamus", "Horse", "Human", "Hummingbird",
                     "Hyena", "Iguana", "Impala", "Jaguar", "Jellyfish", "Kangaroo", "Kiwi", "Koala",
                     "Leopard", "Lion", "Lizard", "Llama", "Lobster", "Monkey", "Moose", "Octopus",
                     "Ostrich", "Otter", "Owl", "Panda", "Parrot", "Peacock", "Pelican", "Penguin",
                     "Pig", "Pigeon", "Polar bear", "Porcupine", "Rabbit", "Raccoon", "Rat",
                     "Raven", "Reindeer", "Rhinoceros", "Salmon", "Seal", "Shark", "Sheep", "Snake",
                     "Spider", "Squirrel", "Tiger", "Turtle", "Wolf", "Zebra"};
        }
        else if (category.find("girl") != std::string::npos || category.find("name") != std::string::npos) {
            items = {"Abigail", "Ava", "Amelia", "Avery", "Aria", "Ashley", "Alexis", "Anna", "Andrea", "Alice",
                     "Bella", "Brooklyn", "Bailey", "Brooke", "Brianna", "Brooke", "Blake", "Brielle",
                     "Catherine", "Charlotte", "Chloe", "Claire", "Claudia", "Camila", "Cora", "Cecilia",
                     "Daisy", "Diana", "Delilah", "Danielle", "Deborah", "Dorothy", "Diane",
                     "Eleanor", "Ella", "Emily", "Emma", "Elizabeth", "Ella", "Ella", "Evelyn", "Elena",
                     "Faith", "Fiona", "Frances", "Felicia", "Fern", "Flora", "Freya",
                     "Gabriella", "Grace", "Georgia", "Gemma", "Gabriela", "Gianna", "Gracie", "Genevieve",
                     "Hannah", "Hailey", "Hazel", "Heidi", "Holly", "Hope", "Heather",
                     "Isabella", "Ivy", "Iris", "Irene", "Isla", "Imogen", "Iris",
                     "Jade", "Jane", "Jasmine", "Julia", "Jade", "Joan", "Jocelyn",
                     "Katherine", "Kayla", "Kimberly", "Kylie", "Keira", "Kamila", "Kinsley",
                     "Layla", "Lily", "Lucy", "Leah", "Laura", "Lillian", "Leilani", "Lucia",
                     "Madison", "Mia", "Maya", "Mackenzie", "Margaret", "Maria", "Melanie",
                     "Natalie", "Nora", "Naomi", "Nicole", "Nadia", "Nina",
                     "Olivia", "Ophelia", "Oakley", "Olive", "Opal",
                     "Penelope", "Piper", "Paige", "Paris", "Peyton", "Paulina",
                     "Quinn", "Queenie", "Quiana",
                     "Rachel", "Rebecca", "Ruby", "Rose", "Ruth", "Rachel", "Ramona",
                     "Sarah", "Samantha", "Sophia", "Stella", "Savannah", "Scarlett", "Sadie",
                     "Taylor", "Tessa", "Talia", "Teresa", "Tiffany", "Tyra",
                     "Uma", "Ursula",
                     "Victoria", "Violet", "Valerie", "Vanessa", "Vera",
                     "Willa", "Winter", "Wendy",
                     "Xena", "Ximena",
                     "Yara", "Yvette", "Yolanda",
                     "Zoe", "Zara", "Zoey"};
        }
        else if (category.find("programming") != std::string::npos || category.find("language") != std::string::npos) {
            items = {"Python", "JavaScript", "Java", "C++", "C#", "PHP", "Ruby", "Swift", "Go", "Rust",
                     "Kotlin", "TypeScript", "Perl", "R", "Scala", "Haskell", "Lua", "Dart", "Elixir", "Clojure"};
        }
        else if (category.find("country") != std::string::npos) {
            items = {"Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina", "Armenia",
                     "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados",
                     "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia", "Botswana",
                     "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
                     "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China",
                     "Colombia", "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus",
                     "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic",
                     "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia",
                     "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia",
                     "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau",
                     "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran",
                     "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan",
                     "Kazakhstan", "Kenya", "Kiribati", "Korea North", "Korea South", "Kosovo", "Kuwait",
                     "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
                     "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia",
                     "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
                     "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco",
                     "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",
                     "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway", "Oman", "Pakistan",
                     "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru",
                     "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
                     "Saint Kitts", "Saint Lucia", "Saint Vincent", "Samoa", "San Marino", "Sao Tome",
                     "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore",
                     "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa",
                     "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland",
                     "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo",
                     "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu",
                     "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
                     "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam",
                     "Yemen", "Zambia", "Zimbabwe"};
        }
        else if (category.find("fruit") != std::string::npos) {
            items = {"Apple", "Apricot", "Avocado", "Banana", "Blackberry", "Blueberry", "Cherry",
                     "Coconut", "Cranberry", "Date", "Dragonfruit", "Durian", "Fig", "Grape",
                     "Grapefruit", "Guava", "Honeydew", "Kiwi", "Kumquat", "Lemon", "Lime",
                     "Mango", "Melon", "Nectarine", "Orange", "Papaya", "Passion fruit", "Peach",
                     "Pear", "Pineapple", "Plum", "Pomegranate", "Raspberry", "Strawberry",
                     "Tangerine", "Watermelon"};
        }
        else if (category.find("color") != std::string::npos) {
            items = {"Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Pink", "Brown",
                     "Black", "White", "Gray", "Gold", "Silver", "Navy", "Teal", "Maroon",
                     "Olive", "Cyan", "Magenta", "Lime", "Indigo", "Violet", "Beige", "Coral"};
        }
        else if (category.find("car") != std::string::npos) {
            items = {"Ford", "Toyota", "Honda", "Chevrolet", "Nissan", "Hyundai", "BMW", "Mercedes",
                     "Audi", "Volkswagen", "Tesla", "Volvo", "Subaru", "Mazda", "Kia", "Lexus",
                     "Porsche", "Ferrari", "Lamborghini", "Jeep", "Dodge", "Chrysler", "GMC"};
        }
        else if (category.find("planet") != std::string::npos) {
            items = {"Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"};
        }

        // Filter by target letter if specified
        std::vector<std::string> filtered;
        for (const auto& item : items) {
            if (targetLetter == '\0' || std::tolower(static_cast<unsigned char>(item[0])) == targetLetter) {
                filtered.push_back(item);
            }
        }

        if (filtered.empty()) {
            return "I don't have any " + category + " starting with the letter '" +
                   std::string(1, targetLetter) + "' in my list.";
        }

        // Build the response
        std::ostringstream oss;
        if (targetLetter != '\0') {
            oss << "Here are " << count << " " << category << " starting with the letter '"
                << static_cast<char>(std::toupper(static_cast<unsigned char>(targetLetter))) << "':\n";
        } else {
            oss << "Here are " << count << " " << category << ":\n";
        }
        int toShow = std::min(count, static_cast<int>(filtered.size()));
        for (int i = 0; i < toShow; ++i) {
            oss << (i + 1) << ". " << filtered[i] << "\n";
        }
        return oss.str();
    };

    lastResponseUsedWebSearch = false;
    lastWebSearchQuery = "";

    if (shouldSearchWeb(message)) {
        lastResponseUsedWebSearch = true;
        lastWebSearchQuery = message;
        std::string searchResult = webSearch(message);
        std::string summarized = summarizeWebResult(message, searchResult);
        
        // If web search didn't yield useful results, fall through to
        // general knowledge handling instead of returning a dead end.
        bool searchFailed = summarized.find("couldn't find relevant information") != std::string::npos ||
                           summarized.find("I tried searching the web") != std::string::npos;
        
        if (!searchFailed && !summarized.empty()) {
            std::string llmResponse;
            if (llmProvider == "freellmapi" || llmProvider == "freeLLMApi") {
                llmResponse = callFreeLLMApi(message, searchResult);
            } else {
                llmResponse = callOllama(message, searchResult);
            }
            if (!llmResponse.empty()) {
                response = llmResponse;
            } else {
                response = summarized;
            }
            learnFromExchange(message, response);
            return response;
        }
        // Fall through to general knowledge responses below
    }

    for (const auto& topic : topicBank) {
        if (containsAny(lowerMsg, topic.keywords)) {
            int variantIdx = conversationCount % topic.responses.size();
            response = topic.responses[variantIdx];
            learnFromExchange(message, response);
            return response;
        }
    }
    std::vector<std::string> tokens = tokenize(message);
    float bestMatchScore = 0.0f;
    std::string bestMatchResponse;
    for (const auto& token : tokens) {
        std::string lowerToken = token;
        std::transform(lowerToken.begin(), lowerToken.end(), lowerToken.begin(), ::tolower);
        std::string cleaned;
        for (char c : lowerToken) {
            if (std::isalnum(static_cast<unsigned char>(c))) {
                cleaned += c;
            }
        }
        if (cleaned.length() <= 3) continue;
        auto it = knowledgeBase.find(cleaned);
        if (it != knowledgeBase.end()) {
            float matchScore = 0.0f;
            for (const auto& t : tokens) {
                std::string lt = t;
                std::transform(lt.begin(), lt.end(), lt.begin(), ::tolower);
                std::string cl;
                for (char c : lt) { if (std::isalnum(static_cast<unsigned char>(c))) cl += c; }
                if (knowledgeBase.count(cl) && knowledgeBase[cl] == it->second) {
                    matchScore += 1.0f;
                }
            }
            if (matchScore > bestMatchScore) {
                bestMatchScore = matchScore;
                bestMatchResponse = it->second;
            }
        }
    }
    if (bestMatchScore >= 2.0f && !bestMatchResponse.empty()) {
        response = bestMatchResponse;
        learnFromExchange(message, response);
        return response;
    }
    std::vector<float> textEmbedding = encodeText(message);
    std::string classification = "";
    if (!textEmbedding.empty() && core) {
        try {
            classification = answer(textEmbedding);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Classification failed: " << e.what() << std::endl;
            classification = "";
        }
    }
    if (lowerMsg.find("how") != std::string::npos && lowerMsg.find("to") != std::string::npos) {
        std::string topic = "the topic you mentioned";
        std::vector<std::string> stopWords = {"how", "to", "the", "a", "an", "is", "are", "do", "does", "i", "you", "we", "they", "it", "my", "your"};
        for (const auto& token : tokens) {
            std::string lt = token;
            std::transform(lt.begin(), lt.end(), lt.begin(), ::tolower);
            bool isStop = std::find(stopWords.begin(), stopWords.end(), lt) != stopWords.end();
            if (!isStop && lt.length() > 3) {
                topic = lt;
                break;
            }
        }
        response = "To accomplish that with \"" + topic + "\", here's my approach:\n"
                   "1. First, understand the requirements and break down the problem\n"
                   "2. Research the best practices and available tools/libraries\n"
                   "3. Implement a solution incrementally, testing each step\n"
                   "4. Review and optimize for edge cases and performance\n"
                   "Could you provide more details about your specific use case? I can give more targeted guidance.";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("what is") != std::string::npos || lowerMsg.find("what are") != std::string::npos ||
        lowerMsg.find("define") != std::string::npos || lowerMsg.find("explain") != std::string::npos) {
        // Try web search first for factual questions
        std::string searchResult = webSearch(message);
        if (!searchResult.empty()) {
            lastResponseUsedWebSearch = true;
            lastWebSearchQuery = message;
            response = summarizeWebResult(message, searchResult);
            learnFromExchange(message, response);
            return response;
        }
        response = "I don't have specific information about \"" + message + "\" in my knowledge base. "
                   "I'll do my best to help if you can tell me more about what you're looking for.";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("why") != std::string::npos) {
        std::string searchResult = webSearch(message);
        if (!searchResult.empty()) {
            lastResponseUsedWebSearch = true;
            lastWebSearchQuery = message;
            response = summarizeWebResult(message, searchResult);
            learnFromExchange(message, response);
            return response;
        }
        response = "That's a good question about \"" + message + "\". Unfortunately I don't have enough "
                   "information in my knowledge base to give you a complete explanation.";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("advice") != std::string::npos || lowerMsg.find("recommend") != std::string::npos ||
        lowerMsg.find("should i") != std::string::npos || lowerMsg.find("best") != std::string::npos) {
        response = "Here's my take on \"" + message + "\":\n"
                   "1. Consider your specific requirements and constraints\n"
                   "2. Research multiple approaches and their trade-offs\n"
                   "3. Start with a well-established, proven solution\n"
                   "4. Iterate and improve based on real-world feedback";
        learnFromExchange(message, response);
        return response;
    }
    if (!classification.empty() && classification.find("Core:") == std::string::npos) {
        if (classification == "Wisdom" || classification == "Knowledge") {
            response = "That's a good question about \"" + message + "\". "
                       "Let me give you a direct answer based on what I know.";
        } else if (classification == "Positive") {
            response = "Thanks for sharing that! Regarding \"" + message + "\", "
                       "I'm glad you're feeling positive about it.";
        } else if (classification == "Critical" || classification == "Negative") {
            response = "I understand you have concerns about \"" + message + "\". "
                       "Let me help address this directly.";
        } else {
            response = "I'm here to help with \"" + message + "\". "
                       "I can assist with programming, cybersecurity, mathematics, science, "
                       "technology, and many other topics.";
        }
        learnFromExchange(message, response);
        return response;
    } else {
        // General knowledge fallback: attempt to answer the user's question
        // directly instead of deflecting with meta-questions.

        std::string lower = message;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        // Handle list-type requests: "name 5 X with letter Y", "list X", etc.
        if (lower.find("name ") != std::string::npos ||
            lower.find("list ") != std::string::npos ||
            lower.find("give me ") != std::string::npos ||
            lower.find("examples of ") != std::string::npos) {
            // Try to extract a category and letter from the request
            std::string category = "";
            char targetLetter = '\0';
            int count = 5;

            // Extract number if present (e.g. "5 animals", "name 5 girls")
            size_t numPos = lower.find_first_of("0123456789");
            if (numPos != std::string::npos) {
                try {
                    count = std::stoi(lower.substr(numPos, 2));
                    if (count < 1) count = 1;
                    if (count > 20) count = 20;
                } catch (...) { count = 5; }
            }

            // Extract letter if present (e.g. "with the letter p", "start with P")
            size_t letterPos = lower.find("letter ");
            if (letterPos != std::string::npos && letterPos + 7 < lower.size()) {
                targetLetter = std::tolower(static_cast<unsigned char>(lower[letterPos + 7]));
            } else {
                // Also try "start with X" pattern
                size_t startWithPos = lower.find("start with ");
                if (startWithPos != std::string::npos && startWithPos + 11 < lower.size()) {
                    char c = lower[startWithPos + 11];
                    if (std::isalpha(static_cast<unsigned char>(c))) {
                        targetLetter = std::tolower(static_cast<unsigned char>(c));
                    }
                }
            }

            // Extract category - look for common category keywords
            std::vector<std::string> categoryKeywords = {
                "animals", "animal", "girls names", "girl names", "names",
                "programming languages", "languages", "countries", "cities",
                "fruits", "colors", "cars", "planets"
            };
            for (const auto& kw : categoryKeywords) {
                if (lower.find(kw) != std::string::npos) {
                    category = kw;
                    break;
                }
            }

            // Generate the list
            response = generateList(category, targetLetter, count);
        }
        // Handle general factual questions
        else if (lower.find("what ") != std::string::npos ||
                 lower.find("who ") != std::string::npos ||
                 lower.find("where ") != std::string::npos ||
                 lower.find("when ") != std::string::npos ||
                 lower.find("how ") != std::string::npos ||
                 lower.find("why ") != std::string::npos) {
            response = "That's a good question about \"" + message + "\". "
                       "I don't have specific information about that in my knowledge base right now.";
        }
        // Handle conversational/greeting input
        else if (lower.length() < 20 &&
                 (lower.find("hi") != std::string::npos || lower.find("hello") != std::string::npos ||
                  lower.find("hey") != std::string::npos || lower.find("good morning") != std::string::npos)) {
            response = "Hello! I'm CoreAI3D, your AI assistant. I can help with programming, "
                       "cybersecurity, mathematics, science, technology, and many other topics. "
                       "What would you like to know or work on?";
        }
        // Last resort: try configured LLM (Ollama or FreeLLMApi), then concise fallback
        else {
            std::string llmResponse;
            if (llmProvider == "freellmapi" || llmProvider == "freeLLMApi") {
                llmResponse = callFreeLLMApi(message, "");
            } else {
                llmResponse = callOllama(message, "");
            }
            if (!llmResponse.empty()) {
                response = llmResponse;
            } else {
                response = "I couldn't generate a detailed response right now. Please try again or ask a more specific question.";
            }
        }
    }
    learnFromExchange(message, response);
    return response;
}


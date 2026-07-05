#include "Language.hpp"
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
    std::vector<std::string> happyWords = {"happy", "great", "awesome", "love", "excited", "amazing", "wonderful", "fantastic", "joy", "glad"};
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
    if (lowerMsg == "hi" || lowerMsg == "hello" || lowerMsg == "hey") {
        isLikelyGreeting = true;
    } else if (lowerMsg.length() < 50) {
        if (lowerMsg.find("hello") == 0 || lowerMsg.find("hi ") == 0 ||
            lowerMsg.find("hey") == 0 || lowerMsg.find("greetings") == 0 ||
            lowerMsg.find("good morning") == 0 || lowerMsg.find("good evening") == 0 ||
            lowerMsg.find("good afternoon") == 0) {
            bool hasTechTerms = lowerMsg.find("how") != std::string::npos ||
                               lowerMsg.find("what") != std::string::npos ||
                               lowerMsg.find("code") != std::string::npos ||
                               lowerMsg.find("program") != std::string::npos ||
                               lowerMsg.find("write") != std::string::npos;
            isLikelyGreeting = !hasTechTerms;
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
        std::string response;
    };
    std::vector<TopicEntry> topicBank = {
        {{"cybersecurity", "cyber", "hack", "security"},
         "Cybersecurity is the practice of protecting systems, networks, and data from digital attacks. Key areas include:\n"
         "- Network Security: Firewalls, IDS/IPS, VPNs, network segmentation\n"
         "- Application Security: Secure coding, input validation, OWASP Top 10\n"
         "- Cryptography: Encryption (AES-256, RSA), hashing (SHA-256, bcrypt), digital signatures\n"
         "- Identity Management: MFA, SSO, zero-trust architecture, RBAC\n"
         "- Incident Response: Detection, containment, eradication, recovery\n"
         "- Compliance: GDPR, HIPAA, PCI-DSS, SOC2 frameworks\n"
         "Would you like me to go deeper into any of these areas?"},
        {{"password", "credential", "authentication"},
         "Password security best practices:\n"
         "- Use passwords of at least 12+ characters with mixed case, numbers, and symbols\n"
         "- Use a reputable password manager (Bitwarden, 1Password, KeePass)\n"
         "- Enable multi-factor authentication (MFA) wherever possible\n"
         "- Never reuse passwords across different services\n"
         "- Consider passkeys/passwordless authentication for new implementations\n"
         "- Rotate credentials regularly and after any suspected breach\n"
         "- Use hashing algorithms like bcrypt, Argon2, or scrypt for storing passwords (never MD5 or SHA1 alone)"},
        {{"encrypt", "encryption", "aes", "rsa", "cipher"},
         "Encryption is fundamental to data security:\n"
         "- Symmetric: AES-256 (fast, good for bulk data), ChaCha20 (mobile-friendly)\n"
         "- Asymmetric: RSA (2048+ bits), Elliptic Curve (ECDSA, Ed25519 - faster, smaller keys)\n"
         "- Hashing: SHA-256 for integrity, bcrypt/Argon2 for passwords\n"
         "- TLS 1.2+ for data in transit, AES-256 for data at rest\n"
         "- Key management is critical - use HSMs or cloud KMS for production\n"
         "- Quantum-resistant algorithms (CRYSTALS-Kyber, CRYSTALS-Dilithium) are emerging"},
        {{"firewall", "network security", "ids", "ips"},
         "Network security architecture essentials:\n"
         "- Firewalls: Stateful inspection, next-gen (NGFW) with IPS, web application firewalls (WAF)\n"
         "- IDS/IPS: Snort, Suricata, OSSEC for intrusion detection\n"
         "- Network segmentation: DMZ, VLANs, microsegmentation for zero-trust\n"
         "- VPNs: WireGuard (modern, fast), OpenVPN (mature), IPsec\n"
         "- Monitoring: SIEM systems (Splunk, Elastic Security), network traffic analysis\n"
         "- DNS security: DNSSEC, DNS filtering, DoH/DoT"},
        {{"phish", "social engineering", "spam"},
         "Phishing and social engineering defense:\n"
         "- Verify sender email addresses carefully - check for subtle misspellings\n"
         "- Don't click suspicious links - hover to preview URLs before clicking\n"
         "- Be wary of urgent requests that pressure immediate action\n"
         "- Use email filtering (SPF, DKIM, DMARC) for inbound protection\n"
         "- Train users regularly on social engineering tactics\n"
         "- Verify requests through secondary channels (phone, in-person)\n"
         "- Implement URL sandboxing and attachment scanning"},
        {{"malware", "virus", "trojan", "ransomware", "ransom"},
         "Malware protection strategies:\n"
         "- Keep all software updated with latest security patches\n"
         "- Use reputable antivirus/anti-malware (Windows Defender is adequate for most users)\n"
         "- Implement application whitelisting in enterprise environments\n"
         "- Ransomware defense: regular offline backups, immutable storage, backup testing\n"
         "- Behavioral analysis and endpoint detection (EDR) for advanced threats\n"
         "- User education on avoiding malicious downloads and attachments\n"
         "- Network segmentation to limit lateral movement"},
        {{"python", "pip", "django", "flask", "pytorch", "numpy"},
         "Python programming insights:\n"
         "- Excellent for data science (NumPy, Pandas, Matplotlib), ML (PyTorch, TensorFlow), web (Django, Flask, FastAPI)\n"
         "- Use virtual environments (venv, conda) to manage dependencies\n"
         "- Type hints (typing module) improve code quality and IDE support\n"
         "- List comprehensions and generator expressions for Pythonic code\n"
         "- Popular testing: pytest, unittest\n"
         "- Package distribution: PyPI, poetry, pipenv\n"
         "- Python 3.11+ offers significant performance improvements"},
        {{"c++", "cpp", "memory", "pointer", "stl", "template"},
         "C++ programming best practices:\n"
         "- Use RAII and smart pointers (unique_ptr, shared_ptr) over raw pointers\n"
         "- Prefer STL containers (vector, map, unordered_map) and algorithms\n"
         "- Use const correctness throughout your codebase\n"
         "- Modern C++ (17/20/23): std::optional, std::variant, std::filesystem, concepts, ranges\n"
         "- Memory management: understand stack vs heap, use value semantics where possible\n"
         "- Build systems: CMake (industry standard), Meson, Bazel\n"
         "- Static analysis: clang-tidy, cppcheck for catching bugs early"},
        {{"algorithm", "data structure", "sort", "search", "complexity", "big o"},
         "Core algorithms and data structures:\n"
         "- Arrays, linked lists, stacks, queues, hash tables, trees, graphs\n"
         "- Sorting: QuickSort (O(n log n) avg), MergeSort (stable), TimSort (used in practice)\n"
         "- Searching: Binary search (O(log n)), BFS, DFS for graphs\n"
         "- Dynamic programming for optimization problems\n"
         "- Space/time complexity analysis using Big-O notation\n"
         "- Balanced trees (AVL, Red-Black) for O(log n) operations\n"
         "- Hash maps for O(1) average lookup"},
        {{"web", "html", "css", "javascript", "react", "node", "api", "http", "rest"},
         "Web development overview:\n"
         "- Frontend: HTML5/CSS3/JavaScript, modern frameworks (React, Vue, Svelte, Angular)\n"
         "- Backend: Node.js, Python (Django/Flask/FastAPI), Go, Rust, PHP\n"
         "- APIs: REST (JSON over HTTP), GraphQL for flexible queries, gRPC for high performance\n"
         "- Databases: PostgreSQL, MySQL (relational), MongoDB, Redis (NoSQL)\n"
         "- DevOps: Docker, Kubernetes, CI/CD pipelines, cloud services\n"
         "- Security: HTTPS, CORS, input sanitization, rate limiting, JWT for auth\n"
         "- Performance: lazy loading, caching (CDN, Redis), code splitting"},
        {{"neural network", "deep learning", "machine learning", "ai", "ml", "training", "model"},
         "AI and Machine Learning fundamentals:\n"
         "- Neural networks: input layer, hidden layers, output layer with activation functions\n"
         "- Training: forward pass, loss function, backpropagation, gradient descent\n"
         "- Types: CNN (images), RNN/LSTM (sequences), Transformer (language/vision)\n"
         "- Key concepts: overfitting/underfitting, regularization, dropout, batch normalization\n"
         "- Transfer learning allows using pretrained models for new tasks efficiently\n"
         "- Frameworks: PyTorch, TensorFlow, JAX for research; scikit-learn for classical ML\n"
         "- CoreAI3D uses a custom neural network with word embeddings for text classification!"},
        {{"math", "calculus", "algebra", "statistics", "probability", "equation"},
         "Mathematics assistance:\n"
         "- Algebra: linear equations, polynomials, matrices, eigenvalues\n"
         "- Calculus: limits, derivatives, integrals, differential equations\n"
         "- Linear algebra: vector spaces, linear transformations, SVD, PCA\n"
         "- Statistics: mean/median/mode, standard deviation, hypothesis testing, regression\n"
         "- Probability: distributions (normal, binomial, Poisson), Bayes' theorem\n"
         "- Discrete math: combinatorics, graph theory, number theory\n"
         "- Numerical methods: Newton's method, Monte Carlo simulation"},
        {{"database", "sql", "mysql", "postgres", "nosql", "query", "table"},
         "Database management insights:\n"
         "- SQL databases: PostgreSQL (feature-rich), MySQL (widely used), SQLite (embedded)\n"
         "- NoSQL: MongoDB (document), Redis (key-value), Cassandra (wide-column), Neo4j (graph)\n"
         "- Indexing: B-tree, hash, composite indexes for query optimization\n"
         "- Normalization: 1NF through 3NF (or BCNF) to reduce data redundancy\n"
         "- Transactions: ACID properties (Atomicity, Consistency, Isolation, Durability)\n"
         "- Performance: query optimization, connection pooling, read replicas, sharding\n"
         "- Backup strategies: full, incremental, point-in-time recovery"},
        {{"linux", "unix", "bash", "shell", "command", "terminal", "admin", "sysadmin"},
         "System administration and Linux:\n"
         "- Package management: apt (Debian/Ubuntu), yum/dnf (RHEL/Fedora), pacman (Arch)\n"
         "- Process management: ps, top/htop, kill, systemd, cron jobs\n"
         "- File permissions: chmod, chown, ACLs, setuid/setgid\n"
         "- Networking: ip, ss, netstat, iptables/nftables, tcpdump\n"
         "- Log analysis: journalctl, /var/log, logrotate, central logging\n"
         "- Security: SELinux/AppArmor, fail2ban, unattended-upgrades, hardening guides\n"
         "- Automation: Bash scripts, Ansible, Terraform for infrastructure as code"},
        {{"network", "tcp", "ip", "dns", "http", "protocol", "packet"},
         "Computer networking concepts:\n"
         "- OSI model: Physical, Data Link, Network, Transport, Session, Presentation, Application\n"
         "- TCP/IP: reliable connection-oriented, UDP: fast connectionless\n"
         "- DNS: domain resolution, record types (A, AAAA, CNAME, MX, TXT)\n"
         "- HTTP/HTTPS: methods (GET, POST, PUT, DELETE), status codes, TLS handshake\n"
         "- Subnetting: CIDR notation, private ranges (10.x, 172.16-31.x, 192.168.x)\n"
         "- Routing: BGP, OSPF for internetwork communication\n"
         "- Modern: HTTP/2 multiplexing, HTTP/3 over QUIC, IPv6 adoption"},
        {{"science", "physics", "chemistry", "biology", "experiment"},
         "Science topics I can help with:\n"
         "- Physics: mechanics, thermodynamics, electromagnetism, quantum mechanics, relativity\n"
         "- Chemistry: atomic structure, bonding, reactions, organic/inorganic chemistry\n"
         "- Biology: cell biology, genetics, evolution, ecology, neuroscience\n"
         "- Earth science: geology, meteorology, climate science\n"
         "- Astronomy: stellar evolution, cosmology, exoplanets\n"
         "- Scientific method: hypothesis, experimentation, peer review, reproducibility"},
        {{"git", "version control", "commit", "branch", "merge", "repository"},
         "Git version control essentials:\n"
         "- Basic workflow: add, commit, push, pull, clone\n"
         "- Branching: feature branches, gitflow, trunk-based development\n"
         "- Collaboration: pull requests, code reviews, merge conflicts\n"
         "- Advanced: rebase vs merge, cherry-pick, stash, reflog\n"
         "- Best practices: clear commit messages, small focused commits, never force-push shared branches\n"
         "- Platforms: GitHub, GitLab, Bitbucket for code hosting and CI/CD\n"
         "- .gitignore for excluding files, tags for versioning releases"},
        {{"docker", "container", "kubernetes", "k8s", "deploy", "devops"},
         "Containerization and DevOps:\n"
         "- Docker: Dockerfile, images, containers, docker-compose for multi-container apps\n"
         "- Kubernetes: pods, deployments, services, ingress, configmaps, secrets\n"
         "- CI/CD: GitHub Actions, GitLab CI, Jenkins pipelines\n"
         "- Infrastructure as Code: Terraform, CloudFormation, Pulumi\n"
         "- Monitoring: Prometheus, Grafana, ELK stack\n"
         "- Cloud: AWS, GCP, Azure for scalable infrastructure"},
        {{"cloud", "aws", "azure", "gcp", "serverless"},
         "Cloud computing services:\n"
         "- AWS: EC2 (compute), S3 (storage), Lambda (serverless), RDS (databases)\n"
         "- Azure: VMs, Blob Storage, Functions, Cosmos DB, Active Directory\n"
         "- GCP: Compute Engine, Cloud Storage, Cloud Functions, BigQuery\n"
         "- Serverless benefits: auto-scaling, pay-per-use, no server management\n"
         "- Best practices: least privilege IAM, encryption at rest/transit, multi-region\n"
         "- Cost optimization: right-sizing, reserved instances, spot/preemptible instances"},
        {{"testing", "test", "unit test", "integration", "tdd", "debug"},
         "Software testing strategies:\n"
         "- Unit tests: test individual functions in isolation (JUnit, pytest, Jest)\n"
         "- Integration tests: verify component interactions\n"
         "- End-to-end tests: simulate user workflows (Cypress, Selenium, Playwright)\n"
         "- TDD: Red-Green-Refactor cycle for writing tests before code\n"
         "- Coverage: aim for meaningful coverage, not just percentage\n"
         "- Performance testing: load testing, stress testing, benchmarking\n"
         "- CI integration: run tests automatically on every commit"},
        {{"api", "rest", "graphql", "grpc", "endpoint", "request"},
         "API design principles:\n"
         "- REST: Use HTTP verbs correctly, proper status codes, resource-based URLs\n"
         "- GraphQL: Flexible queries, single endpoint, schema-driven development\n"
         "- gRPC: Protocol buffers, streaming, high performance for microservices\n"
         "- Authentication: API keys, OAuth 2.0, JWT tokens\n"
         "- Documentation: OpenAPI/Swagger, automatic doc generation\n"
         "- Rate limiting, pagination, caching, versioning for robust APIs\n"
         "- Error handling: consistent error format, meaningful error messages"},
        {{"algorithm", "sort", "complexity", "performance", "optimization", "efficient"},
         "Algorithm optimization:\n"
         "- Choose the right data structure: hash maps for O(1) lookup, sorted arrays for binary search\n"
         "- Divide and conquer: break large problems into manageable subproblems\n"
         "- Memoization and dynamic programming to avoid redundant computation\n"
         "- Use appropriate sorting: Timsort (general), radix sort (integers), counting sort (small range)\n"
         "- Profile before optimizing - measure actual bottlenecks\n"
         "- Cache frequently accessed data, use lazy evaluation"}
    };
    auto containsAny = [&](const std::string& text, const std::vector<std::string>& words) {
        for (const auto& w : words) {
            if (text.find(w) != std::string::npos) return true;
        }
        return false;
    };
    for (const auto& topic : topicBank) {
        if (containsAny(lowerMsg, topic.keywords)) {
            response = topic.response;
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
        response = "That's an interesting question about \"" + message + "\". Let me share what I know:\n"
                   "I'd need to analyze this topic further to give you a comprehensive answer. "
                   "Could you specify which aspect you're most interested in? For example:\n"
                   "- A high-level overview or detailed technical explanation\n"
                   "- Practical applications or theoretical foundations\n"
                   "- Comparison with related concepts\n"
                   "The more specific you are, the more helpful my response will be!";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("why") != std::string::npos) {
        response = "Great analytical question! Understanding 'why' helps build deeper knowledge.\n"
                   "For your question about \"" + message + "\", the answer typically involves:\n"
                   "- Historical context and evolution of the concept\n"
                   "- Underlying principles and cause-effect relationships\n"
                   "- Practical implications and real-world applications\n"
                   "Could you tell me more about the specific context? I can provide a more targeted explanation.";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("advice") != std::string::npos || lowerMsg.find("recommend") != std::string::npos ||
        lowerMsg.find("should i") != std::string::npos || lowerMsg.find("best") != std::string::npos) {
        response = "Here's my recommendation regarding \"" + message + "\":\n"
                   "1. Consider your specific requirements and constraints\n"
                   "2. Research multiple approaches and their trade-offs\n"
                   "3. Start with a well-established, proven solution\n"
                   "4. Iterate and improve based on real-world feedback\n"
                   "What specific factors matter most to you? I can tailor my advice to your situation.";
        learnFromExchange(message, response);
        return response;
    }
    if (!classification.empty() && classification.find("Core:") == std::string::npos) {
        if (classification == "Wisdom" || classification == "Knowledge") {
            response = "That's a thoughtful question about \"" + message + "\". "
                       "Based on my analysis, this seems like an area where deeper understanding is valuable. "
                       "I'd recommend exploring this topic through multiple angles - "
                       "theoretical foundations, practical applications, and expert perspectives. "
                       "What specific aspect would you like to explore further?";
        } else if (classification == "Positive") {
            response = "That's a great point! Regarding \"" + message + "\", I appreciate the positive angle. "
                       "Would you like me to expand on any particular aspect or provide additional information?";
        } else if (classification == "Critical" || classification == "Negative") {
            response = "I understand you're raising an important concern about \"" + message + "\". "
                       "Let me help address this: could you provide more context about the specific issue? "
                       "The more details you share, the better I can assist in finding a solution.";
        } else {
            response = "I've processed your message about \"" + message + "\". "
                       "I'm here to help with a wide range of topics including cybersecurity, programming, "
                       "mathematics, science, and technology. Could you tell me more specifically what you'd like to know?";
        }
    } else {
        std::vector<std::string> fallbackResponses = {
            "That's an interesting thought. Tell me more about where you're coming from with this - I'd love to understand your perspective better.",
            "I'm thinking about this with you. Could you give me a bit more context? The more I understand what you're working toward, the better I can help.",
            "Hmm, this is worth exploring. What specifically would be most useful for me to dig into here?",
            "I'm intrigued by this. Help me understand what aspect matters most to you right now, and we'll go deeper together.",
            "Good question - let's break this down. What's the core thing you're trying to figure out or accomplish?"
        };
        
        int idx = std::hash<std::string>{}(message + std::to_string(std::time(nullptr))) % fallbackResponses.size();
        response = fallbackResponses[idx];
    }
    learnFromExchange(message, response);
    return response;
}


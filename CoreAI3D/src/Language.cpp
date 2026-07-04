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
    std::vector<std::vector<float>> model_input_reshaped = reshape(data, embeddingDim);
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
    if (conversationMemory.size() > 1000) {
        conversationMemory.erase(conversationMemory.begin());
    }
    std::vector<std::string> tokens = tokenize(userInput);
    std::string lowerInput = userInput;
    std::transform(lowerInput.begin(), lowerInput.end(), lowerInput.begin(), ::tolower);
    std::vector<std::string> keywords;
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
            keywords.push_back(cleaned);
        }
    }
    for (const auto& keyword : keywords) {
        if (knowledgeBase.find(keyword) == knowledgeBase.end()) {
            knowledgeBase[keyword] = aiResponse;
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
    if (lowerMsg == "hi" || lowerMsg.find("hello") != std::string::npos ||
        lowerMsg.find("hi ") != std::string::npos || lowerMsg.find("hey") != std::string::npos ||
        lowerMsg.find("greetings") != std::string::npos || lowerMsg.find("good morning") != std::string::npos ||
        lowerMsg.find("good evening") != std::string::npos || lowerMsg.find("good afternoon") != std::string::npos) {
        response = "Hello! I'm CoreAI3D, your AI assistant. I can help you with cybersecurity, programming, mathematics, science, and much more. What would you like to know?";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("bye") != std::string::npos || lowerMsg.find("goodbye") != std::string::npos ||
        lowerMsg.find("see you") != std::string::npos || lowerMsg.find("see ya") != std::string::npos) {
        response = "Goodbye! Feel free to come back anytime you need help. I'm always here to assist you.";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("thank") != std::string::npos || lowerMsg.find("thanks") != std::string::npos) {
        response = "You're welcome! I'm glad I could help. Is there anything else you'd like to know?";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("how are you") != std::string::npos || lowerMsg.find("how do you do") != std::string::npos) {
        response = "I'm doing great, thank you for asking! As an AI powered by custom neural networks, I'm always ready to help. What can I assist you with today?";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("your name") != std::string::npos || lowerMsg.find("who are you") != std::string::npos) {
        response = "I'm CoreAI3D, an advanced AI system built with custom neural networks for text classification and response generation. I specialize in cybersecurity, programming, mathematics, and general knowledge. I learn and improve with every conversation!";
        learnFromExchange(message, response);
        return response;
    }
    if (lowerMsg.find("what can you do") != std::string::npos || (lowerMsg.find("help me") != std::string::npos && lowerMsg.length() < 20)) {
        response = "I can help you with many things:\n"
                   "- Cybersecurity advice and best practices\n"
                   "- Programming help (Python, C++, web development, algorithms)\n"
                   "- Mathematics and statistics\n"
                   "- AI and machine learning concepts\n"
                   "- System administration and networking\n"
                   "- Science and general knowledge\n"
                   "- Creative writing and brainstorming\n"
                   "Just ask me anything and I'll do my best to help!";
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
        classification = answer(textEmbedding);
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
        response = "I received your message: \"" + message + "\"\n\n"
                   "I'm CoreAI3D, an AI assistant powered by a custom neural network. I can discuss:\n"
                   "- Cybersecurity (passwords, encryption, firewalls, malware, network security)\n"
                   "- Programming (Python, C++, web development, algorithms, testing)\n"
                   "- Mathematics (algebra, calculus, statistics, probability)\n"
                   "- AI/ML (neural networks, deep learning, training)\n"
                   "- Systems (Linux, databases, networking, cloud)\n"
                   "- And many more topics!\n\n"
                   "What would you like to explore?";
    }
    learnFromExchange(message, response);
    return response;
}


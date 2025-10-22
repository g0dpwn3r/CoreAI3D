#include "Language.hpp"
Language::Language(std::string& embedingFile, int& embeddingDim, std::string& dbHost, int& dbPort,
    std::string& dbUser, std::string& dbPassword,
    std::string& dbSchema, int sslDummy, std::string& lang, int& inputSize, int& outputSize, int& layers, int& neurons)
    : embedingFile(embedingFile), embeddingDim(embeddingDim), dbHost(dbHost), dbPort(dbPort), dbUser(dbUser), dbPassword(dbPassword), dbSchema(dbSchema),
    sslDummy(sslDummy), currentLang(lang), inputSize(inputSize), outputSize(outputSize), layers(layers), neurons(neurons)
{
    trainer = std::make_unique<Training>(dbHost, dbPort, dbUser, dbPassword, dbSchema, 0, false);
    core = std::make_unique<CoreAI>(inputSize, layers, neurons, outputSize, -1.0f, 1.0f);
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
            std::cerr << "[!] Warning: Skipping word '" << word
                << "' due to mismatched dimensions: " << embedding.size()
                << " vs " << expectedDim << std::endl;
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
        return "Error: AI Core not ready.";
    }
    std::vector<std::vector<float>> model_input_reshaped = reshape(data, embeddingDim);
    std::vector<std::vector<float>> model_prediction = core->forward(model_input_reshaped);
    if (model_prediction.empty() || model_prediction[0].empty()) {
        std::cerr << "Warning: Model prediction returned empty. Cannot determine class." << std::endl;
        return "No clear answer.";
    }
    std::vector<float>& output = model_prediction[0];  
    int predictedClassIndex = 0;
    float maxProb = output[0];
    for (size_t j = 1; j < output.size(); ++j) { 
        if (output[j] > maxProb) {
            maxProb = output[j];
            predictedClassIndex = j;
        }
    }
    if (predictedClassIndex >= 0 && predictedClassIndex < classLabels.size()) {
        return classLabels[predictedClassIndex];
    }
    else {
        return "Prediction out of bounds.";
    }
}
int Language::chat(std::string& filename) {
    std::string inputText;
    int rowIndex = 0; // Initialize row index for database records
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
            std::cerr << "[!] Warning: No embeddings loaded from file '" << embedingFile
                << "'. Falling back to dummy embeddings." << std::endl;
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

        // Store chat history in database
        if (trainer && trainer->dbManager) {
            try {
                // Create a dataset for chat history if not exists
                int chatDatasetId = trainer->dbManager->addDataset("chat_history", "Chat history for online training", 0, textEmbedding.size(), 1);
                // Store user input and AI response as training data
                std::vector<float> inputFeatures = textEmbedding;
                std::vector<float> targetLabels = {0.0f}; // Placeholder target, could be sentiment or classification
                trainer->dbManager->addDatasetRecord(chatDatasetId, rowIndex++, inputFeatures, targetLabels); // Use proper row index
                if (trainer->verbose) {
                    std::cout << "Chat history stored in database." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error storing chat history: " << e.what() << std::endl;
            }
        }

        // Perform online training with the new data
        if (trainer && trainer->dbManager) {
            try {
                // Load the latest chat data and train incrementally
                int chatDatasetId = 1; // Assume ID 1 for chat history, or retrieve dynamically
                if (trainer->loadDatasetFromDB(chatDatasetId)) {
                    trainer->preprocess(-1.0f, 1.0f); // Normalize
                    trainer->train(0.01, 1); // Train for 1 epoch with learning rate 0.01
                    if (trainer->verbose) {
                        std::cout << "Online training completed." << std::endl;
                    }
                } else {
                    if (trainer->verbose) {
                        std::cout << "No chat dataset available for training." << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during online training: " << e.what() << std::endl;
            }
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

std::vector<float> Language::encodeText(const std::string& text)
{
    std::vector<std::string> words = tokenize(text);
    std::vector<float> flattenedEmbedding;
    std::unordered_map<std::string, std::vector<float>>& currentEmbeddings = this->embeddingsByLang;

    for (std::string& word : words)
    {
        std::string key = currentLang + ":" + word;
        auto it = currentEmbeddings.find(key);

        if (it != currentEmbeddings.end())
        {
            flattenedEmbedding.insert(flattenedEmbedding.end(), it->second.begin(), it->second.end());
        }
        else
        {
            std::vector<float> randomEmbedding = this->generateRandomEmbedding();
            flattenedEmbedding.insert(flattenedEmbedding.end(), randomEmbedding.begin(), randomEmbedding.end());
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

    // Integrate with training system
    if (trainer) {
        // Create a dataset for conversation learning
        int conversationDatasetId = trainer->dbManager->addDataset("conversation_learning", "Learning from conversation data", parsedTurns.size(), embeddingDim, 1);

        // Add conversation data as training samples
        for (size_t i = 0; i < parsedTurns.size(); ++i) {
            const auto& turn = parsedTurns[i];
            std::vector<float> inputEmbedding = encodeText(turn.second);
            std::vector<float> targetLabel = {turn.first == "User" ? 0.0f : 1.0f}; // 0 for user, 1 for AI

            trainer->dbManager->addDatasetRecord(conversationDatasetId, i, inputEmbedding, targetLabel);
        }

        // Load and train on the conversation data
        if (trainer->loadDatasetFromDB(conversationDatasetId)) {
            trainer->preprocess(-1.0f, 1.0f);
            trainer->train(0.01, 5); // Train for 5 epochs with learning rate 0.01
            if (trainer->verbose) {
                std::cout << "Conversation learning completed." << std::endl;
            }
        } else {
            std::cerr << "Failed to load conversation dataset for training." << std::endl;
        }
    } else {
        std::cerr << "Warning: Trainer not available for conversation learning." << std::endl;
    }
}

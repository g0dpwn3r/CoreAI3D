#include "Language.hpp"

Language::Language(std::string& embedingFile, int& embeddingDim, std::string& dbHost, int& dbPort,
    std::string& dbUser, std::string& dbPassword,
    std::string& dbSchema, mysqlx::SSLMode ssl, std::string& lang, int& inputSize, int& outputSize, int& layers, int& neurons)
    : embedingFile(embedingFile), embeddingDim(embeddingDim), dbHost(dbHost), dbPort(dbPort), dbUser(dbUser), dbPassword(dbPassword), dbSchema(dbSchema),
    ssl(ssl), currentLang(lang), inputSize(inputSize), outputSize(outputSize), layers(layers), neurons(neurons)
{
    // Initialize trainer and core here, ensuring they use the class members for their construction
    // Use the actual parameters passed to Language constructor
    trainer = std::make_unique<Training>(dbHost, dbPort, dbUser, dbPassword, dbSchema, ssl, false);
    core = std::make_unique<CoreAI>(inputSize, layers, neurons, outputSize, -1.0f, 1.0f); // Ensure minVal, maxVal are floats
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

// Reshape function is a standalone helper, does not need to be part of Language class directly
// Moved outside class for better encapsulation, or can be a static member if it uses Language members
std::vector<std::vector<float>> reshape(const std::vector<float>& input, size_t rowSize)
{
    std::vector<std::vector<float>> result;

    if (rowSize == 0) return result; // Avoid division by zero

    for (size_t i = 0; i + rowSize <= input.size(); i += rowSize) {
        std::vector<float> row(input.begin() + i, input.begin() + i + rowSize);
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

        if (embedding.size() != (size_t)expectedDim) // Cast expectedDim to size_t
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

    // Check if core is initialized before using it
    if (!core) {
        std::cerr << "Error: CoreAI is not initialized in Language::answer. Cannot perform forward pass." << std::endl;
        return "Error: AI Core not ready.";
    }

    // Ensure `reshape` is callable or static/global
    // If embeddingDim is the "width" of each token's embedding, and inputSize is total,
    // then `textEmbedding.size()` should be a multiple of `embeddingDim`.
    // The `reshape` function should be called with `embeddingDim` as rowSize.
    std::vector<std::vector<float>> model_input_reshaped = reshape(data, embeddingDim);

    std::vector<std::vector<float>> model_prediction = core->forward(model_input_reshaped);

    if (model_prediction.empty() || model_prediction[0].empty()) {
        std::cerr << "Warning: Model prediction returned empty. Cannot determine class." << std::endl;
        return "No clear answer.";
    }

    std::vector<float>& output = model_prediction[0];  // prediction for the first input
    int predictedClassIndex = 0;
    float maxProb = output[0];
    for (size_t j = 1; j < output.size(); ++j) { // Use size_t
        if (output[j] > maxProb) {
            maxProb = output[j];
            predictedClassIndex = j;
        }
    }

    // Ensure classLabels has enough elements
    if (predictedClassIndex >= 0 && predictedClassIndex < classLabels.size()) {
        // Training within answer is usually not desired, as answer should be inference.
        // If it's a specific fine-tuning step, then it's ok, but typically train() is separate.
        // For demonstration, commenting out the training loop in answer.
        /*
        if (trainer) { // Check if trainer is initialized
            for (int i = 0; i < 100; i++) {
                data = generateRandomEmbedding(); // Generates random embedding of size `embeddingDim`
                // Reshape data for training if CoreAI::train expects 2D vector
                std::vector<std::vector<float>> train_input = reshape(data, embeddingDim);
                // The target for this training step is the current model_prediction.
                // You might need to adjust this target based on what you want to teach.
                trainer->train(train_input, model_prediction, trainer->learningRate, trainer->numSamples);
            }
        } else {
            std::cerr << "Warning: Trainer is not initialized in Language::answer. Skipping internal training." << std::endl;
        }
        */

        // Use the predicted class label
        return classLabels[predictedClassIndex];
    }
    else {
        return "Prediction out of bounds.";
    }
}

int Language::chat(std::string& filename) {

    std::string inputText;

    while (true)
    {
        std::cout << "\nEnter your message: ";
        // Read the entire line, including spaces
        std::getline(std::cin, inputText);

        if (inputText == "exit")
        {
            break;
        }

        // Basic language detection and setting
        currentLang = detectLanguage(inputText);
        if (currentLang == "unknown")
        {
            std::cout
                << "Could not confidently detect language. Defaulting to English."
                << std::endl;
            setCurrentLanguage("en"); // Or handle as desired
        }
        else
        {
            setCurrentLanguage(currentLang);
            std::cout << "Detected language: " << currentLang
                << ". Encoding with this language's embeddings."
                << std::endl;
        }

        // Ensure embeddingsByLang is populated based on the language
        // You might want to load the relevant embedding file here if not already loaded,
        // or ensure createEmbeddingsByLang populates it correctly based on `currentLang`.
        // The `createEmbeddingsByLang` generates dummy data in its current form.
        // It should probably load specific language embeddings from files.
        this->embeddingsByLang = createEmbeddingsByLang(embeddingDim); // Renamed to avoid shadowing member
        // After this, embeddingsByLang (member) should be properly loaded if it's meant to hold all data.

        // Encode the text
        std::vector<float> textEmbedding = this->encodeText(inputText);
        std::vector<std::string> tokens = this->tokenize(inputText);
        // int size = static_cast<int> (textEmbedding.size()); // Removed unused variable 'size'

        // Here, you would typically use 'textEmbedding' for your core logic.
        // For demonstration, we'll just print its size.
        std::cout << "Processed message. Encoded embedding size: " << textEmbedding.size() << std::endl;

        if (inputText.find("hello") != std::string::npos
            || inputText.find("hi") != std::string::npos)
        {
            std::cout << "Core: Hello there!" << std::endl;
        }
        else if (inputText.find("how are you") != std::string::npos)
        {
            std::cout << "Core: I'm just a program, but I'm doing great!"
                << std::endl;
        }
        else if (inputText.find("name") != std::string::npos)
        {
            std::cout << "Core: My name is CoreAI3D or Core for short."
                << std::endl;
        }
        else
        {
            std::cout << answer(textEmbedding) << std::endl;
        }
    }

    std::cout << "Exiting chat. Goodbye!" << std::endl;

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

// This function needs to be redesigned if it's meant to flatten specific embeddings.
// Its current usage (Language::flattenEmbeddings from outside) suggests it takes
// a map of embeddings, but the provided `embeddingsByLang` member is also a map.
// If `embeddingsByLang` holds per-language maps like `map<lang_code, map<word, embedding>>`,
// then this function should work with that.
// The previous Language.cpp snippet had a line `flattenedEmbedding.insert(flattenedEmbedding.end(), embedding);`
// which implies `embedding` is a vector, but `wordMap` is a `map<string, vector<float>>`.
// Let's assume it flattens a single language's word embeddings.
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

// Create embedding map with dummy values for each language
std::unordered_map<std::string, std::vector<float>> Language::createEmbeddingsByLang(int embeddingDim) {
    std::unordered_map<std::string, std::vector<float>> embeddings;

    // Example dummy data creation for multiple languages
    // This function is still generating dummy data. In a real scenario,
    // you would load actual embeddings for each language here.
    // For a more realistic scenario, loadWordEmbeddingsFromFile would be used.

    std::vector<std::string> languages = { "en", "fr", "de", "es" };
    for (const std::string& lang_code : languages) {
        // Create a dummy embedding for a generic "language representative"
        std::vector<float> lang_embedding(embeddingDim);
        for (int i = 0; i < embeddingDim; ++i) {
            lang_embedding[i] = (float)i / embeddingDim; // Simple dummy values
        }
        embeddings[lang_code] = lang_embedding; // Store a dummy embedding for the language key itself
    }

    // Add specific word embeddings to the top-level embeddingsByLang if it's meant to hold them all
    // This structure needs clarity: is embeddingsByLang a map of word->embedding, or lang->(word->embedding)?
    // Based on encodeText, it's treated as map<key, vector<float>> where key is "lang:word".
    // So, this function should create those specific word embeddings if needed.
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

    // Now, assign this local map to the member embeddingsByLang
    this->embeddingsByLang = embeddings; // Correctly assign to the member
    return embeddings;
}

std::vector<float> Language::encodeText(const std::string& text)
{
    std::vector<std::string> words = tokenize(text);
    std::vector<float> flattenedEmbedding;
    // Use the member embeddingsByLang_map
    std::unordered_map<std::string, std::vector<float>>& currentEmbeddings = this->embeddingsByLang;

    // The logic `if (embeddingsByLang.find(currentLang) != embeddingsByLang.end())`
    // was trying to find `currentLang` as a key, but your map now holds "lang:word" keys.
    // So, we need to iterate words and construct keys like "currentLang:word".

    for (std::string& word : words)
    {
        std::string key = currentLang + ":" + word; // Construct the key
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


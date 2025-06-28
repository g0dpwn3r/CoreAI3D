#include "Language.hpp"


Language::Language(int embeddingDim)
    : embeddingDim(embeddingDim), currentLang("en") {
}

CoreAI* Language::getCore() {
    return core.get();
}

Training* Language::getTrainer() {
    return trainer.get();
}

std::string Language::detectLanguage(const std::string& text) {
    static const std::map<std::string, std::regex> languagePatterns = {
        {"en", std::regex(R"(\b(the|and|is|you|are|to|have|be)\b)", std::regex_constants::icase)},
        {"nl", std::regex(R"(\b(de|het|een|en|is|je|ik|heb)\b)", std::regex_constants::icase)},
        {"fr", std::regex(R"(\b(le|la|et|est|vous|je|ai|être)\b)", std::regex_constants::icase)},
        {"de", std::regex(R"(\b(der|die|und|ist|du|ich|habe)\b)", std::regex_constants::icase)}
    };

    for (const auto& [lang, pattern] : languagePatterns) {
        if (std::regex_search(text, pattern)) {
            currentLang = lang;
            return lang;
        }
    }

    currentLang = "en";
    return "en";
}

std::unordered_map<std::string, std::vector<float>> Language::loadWordEmbeddingsFromFile(const std::string& filepath, int expectedDim) {
    std::unordered_map<std::string, std::vector<float>> wordEmbeddingMap;
    std::ifstream infile(filepath);
    std::string line;

    if (!infile) {
        std::cerr << "[!] Error: Unable to open embedding file: " << filepath << std::endl;
        return wordEmbeddingMap;
    }

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string word;
        iss >> word;

        std::vector<float> embedding;
        float value;
        while (iss >> value) {
            embedding.push_back(value);
        }

        if (embedding.size() != expectedDim) {
            std::cerr << "[!] Warning: Skipping word '" << word << "' due to mismatched dimensions: "
                << embedding.size() << " vs " << expectedDim << std::endl;
            continue;
        }

        wordEmbeddingMap[word] = std::move(embedding);
    }

    return wordEmbeddingMap;
}

int Language::chat(std::string& filename, int& inputSize, int& layers, int& neurons, int& outputSize, float& minVal, float& maxVal, int embeddingDimension) {
    this->loadEmbeddingsFor("en", filename);
    std::cout << "Welcome to the Language Chat!" << std::endl;
    std::cout << "Type 'exit' to quit." << std::endl;

    std::string inputText;
    while (true) {
        std::cout << "\nEnter your message: ";
        // Read the entire line, including spaces
        std::getline(std::cin, inputText);

        if (inputText == "exit") {
            break;
        }

        // Basic language detection and setting
        std::string detectedLang = this->detectLanguage(inputText);
        if (detectedLang == "unknown") {
            std::cout << "Could not confidently detect language. Defaulting to English." << std::endl;
            this->setCurrentLanguage("en"); // Or handle as desired
        }
        else {
            this->setCurrentLanguage(detectedLang);
            std::cout << "Detected language: " << detectedLang << ". Encoding with this language's embeddings." << std::endl;
        }


        // Encode the text
        std::vector<std::vector<float>> data;
        std::vector<float> textEmbedding = this->encodeText(inputText);
        data.push_back(textEmbedding);
        int size = static_cast<int>(textEmbedding.size());
        trainer->printFullMatrix(data, size, 1);
        core = std::make_unique<CoreAI>(inputSize, layers, neurons, outputSize, minVal, maxVal);

        // Here, you would typically use 'textEmbedding' for your core logic.
        // For demonstration, we'll just print its size.
        std::cout << "Processed message. Encoded embedding size: " << textEmbedding.size() << std::endl;
        // You could add logic here to "respond" based on the embedding,
        // e.g., by comparing it to other pre-encoded text embeddings.

        // Example of a simple "response" based on keywords (very basic)
        if (inputText.find("hello") != std::string::npos || inputText.find("hi") != std::string::npos) {
            std::cout << "Core: Hello there!" << std::endl;
        }
        else if (inputText.find("how are you") != std::string::npos) {
            std::cout << "Core: I'm just a program, but I'm doing great!" << std::endl;
        }
        else if (inputText.find("name") != std::string::npos) {
            std::cout << "Core: My name is CoreAI3D or Core for short." << std::endl;
        }
    }

    std::cout << "Exiting chat. Goodbye!" << std::endl;

    return 0;
}

void Language::setCurrentLanguage(const std::string& languageCode) {
    currentLang = languageCode;
}

void Language::loadEmbeddingsFor(const std::string& languageCode, const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open embedding file for language: " << languageCode << std::endl;
        return;
    }

    std::unordered_map<std::string, std::vector<float>> langEmbeddings;
    std::string word;
    while (file >> word) {
        std::vector<float> vec(embeddingDim);
        for (int i = 0; i < embeddingDim; ++i) {
            file >> vec[i];
        }
        langEmbeddings[word] = vec;
    }

    embeddingsByLang[languageCode] = langEmbeddings;
    std::cout << "Loaded " << langEmbeddings.size() << " embeddings for language: " << languageCode << std::endl;
}

std::vector<std::string> Language::tokenize(const std::string& text) {
    std::istringstream iss(text);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<float> Language::flattenEmbeddings(const std::vector<std::vector<float>>& wordVectors) {
    std::vector<float> result;
    for (const auto& vec : wordVectors) {
        result.insert(result.end(), vec.begin(), vec.end());
    }
    return result;
}

std::vector<float> Language::generateRandomEmbedding() {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(embeddingDim);
    for (int i = 0; i < embeddingDim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

std::vector<float> Language::encodeText(const std::string& text) {
    std::vector<std::string> words = tokenize(text);
    std::vector<std::vector<float>> vectors;
    const auto& langEmbeddings = embeddingsByLang[currentLang];

    for (const auto& word : words) {
        auto it = langEmbeddings.find(word);
        if (it != langEmbeddings.end()) {
            vectors.push_back(it->second);
        }
        else {
            vectors.push_back(this->generateRandomEmbedding());
        }
    }

    return flattenEmbeddings(vectors);
}



#pragma once

#include "main.hpp"
#include "Core.hpp"
#include "Train.hpp"

class Training;
class CoreAI;

class Language {
public:
    Language();
    std::string currentLang = "en";
    std::string detectLanguage(const std::string& text);
    std::vector<float> encodeText(const std::string& text);
    void loadEmbeddingsFor(const std::string& languageCode, const std::string& filepath);
    void setCurrentLanguage(const std::string& languageCode);
    Language(int embeddingDim);
    CoreAI* getCore();
    Training* getTrainer();

    std::unordered_map<std::string, std::vector<float>> loadWordEmbeddingsFromFile(const std::string& filepath, int expectedDim);

    int chat(std::string& filename, int& inputSize, int& layers, int& neurons, int& outputSize, float& minVal, float& maxVal, int embeddingDimension = 50);
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<float>>> embeddingsByLang;

private:
    
    int embeddingDim;
    
    std::unique_ptr<CoreAI> core;
    std::unique_ptr<Training> trainer;
    std::vector<std::string> tokenize(const std::string& text);
    std::vector<float> flattenEmbeddings(const std::vector<std::vector<float>>& words);
    std::vector<float> generateRandomEmbedding();
};

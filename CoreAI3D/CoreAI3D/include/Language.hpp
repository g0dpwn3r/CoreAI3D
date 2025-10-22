#pragma once

#include "CoreAI3DCommon.hpp"

#include "Train.hpp"
#include "Database.hpp"
#include "Core.hpp"

class Training;
class CoreAI;

class Language
{
public:
    Language(std::string& embedingFile, int& embeddingDim, std::string& dbHost, int& dbPort,
        std::string& dbUser, std::string& dbPassword,
        std::string& dbSchema, int sslDummy, std::string& lang, int& inputSize, int& outputSize, int& layers, int& neurons);
    std::string detectLanguage(const std::string& text);
    std::vector<float> encodeText(const std::string& text);
    void setCurrentLanguage(const std::string& languageCode);
    
    CoreAI* getCore();
    Training* getTrainer();

    std::string currentLang = "en";

    static std::vector<float> flattenEmbeddings(const std::unordered_map<std::string, std::vector<float>>& embeddingsByLang);

    static int detectMaxSeqLength(const std::string& filename);

    float cosine_similarity(std::vector<float> a, std::vector<float> b);
    std::vector<float> generateRandomEmbedding();
    std::unordered_map<std::string, std::vector<float>> loadWordEmbeddingsFromFile(const std::string& filepath, int expectedDim);
    int chat(std::string& filename);
    std::unordered_map<std::string, std::vector<float>> embeddingsByLang;

    std::unordered_map<std::string, std::vector<float>> createEmbeddingsByLang(int embeddingDim);
    std::vector<std::string> tokenize(const std::string& text);
private:

    
    int embeddingDim = 300;
    int inputSize;
    int outputSize;
    int layers;
    int neurons;
    int dbPort;
    std::string embedingFile;
    std::string dbHost;
    std::string dbUser;
    std::string dbPassword;
    std::string dbSchema;
    int sslDummy;
    bool createTables;
    std::unordered_map<std::string, std::vector<float>> db;

    std::unique_ptr<CoreAI> core;
    std::unique_ptr<Training> trainer;
    std::string answer(std::vector<float>& textEmbedding);

    // New methods for learning from conversations
    std::vector<std::pair<std::string, std::string>> parseConversation(const std::string& conversation);
    std::unordered_map<std::string, std::vector<float>> extractContext(const std::vector<std::pair<std::string, std::string>>& parsedConversation);
    void learnFromConversation(const std::string& conversation);

};

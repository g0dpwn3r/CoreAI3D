#pragma once

#include "CoreAI3DCommon.hpp"

#include "Train.hpp"
#include "Core.hpp"
#include "Database.hpp"
#include <mutex>

class Training;
class CoreAI;

class Language
{
public:
    Language(std::string& embedingFile, int& embeddingDim, std::string& dbHost, int& dbPort,
        std::string& dbUser, std::string& dbPassword,
        std::string& dbSchema, int sslDummy, std::string& lang, int& inputSize, int& outputSize, int& layers, int& neurons, int sessionId = 1);
    std::string detectLanguage(const std::string& text);
    std::vector<float> encodeText(const std::string& text);
    void setCurrentLanguage(const std::string& languageCode);

    CoreAI* getCore();
    Training* getTrainer();

    bool getLastResponseUsedWebSearch() const { return lastResponseUsedWebSearch; }
    std::string getLastWebSearchQuery() const { return lastWebSearchQuery; }

    std::string currentLang = "en";

    static std::vector<float> flattenEmbeddings(const std::unordered_map<std::string, std::vector<float>>& embeddingsByLang);

    static int detectMaxSeqLength(const std::string& filename);

    float cosine_similarity(std::vector<float> a, std::vector<float> b);
    std::vector<float> generateRandomEmbedding();
    std::unordered_map<std::string, std::vector<float>> loadWordEmbeddingsFromFile(const std::string& filepath, int expectedDim);
    int chat();
    std::string respond(const std::string& message);
    void learnFromExchange(const std::string& userInput, const std::string& aiResponse);
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
    int sessionId;
    std::string embedingFile;
    std::string dbHost;
    std::string dbUser;
    std::string dbPassword;
    std::string dbSchema;
    int sslDummy;
    bool createTables;
    std::unordered_map<std::string, std::vector<float>> db;

    std::vector<std::pair<std::string, std::string>> conversationMemory;
    std::unordered_map<std::string, std::string> knowledgeBase;
    bool embeddingsLoaded = false;
    std::mutex respondMutex;

    // Personality and emotional learning
    std::string userName = "";
    std::unordered_map<std::string, int> emotionFrequency;  // Track emotional themes user brings up
    std::unordered_map<std::string, int> topicPreferences;  // Track topics user cares about
    int conversationCount = 0;
    float happinessScore = 0.5f;  // Rolling average of user's emotional state (0.0 = sad, 1.0 = happy)
    int humorLevel = 0;  // 0 = neutral, scales with user's humor usage

    // Web search
    bool lastResponseUsedWebSearch = false;
    std::string lastWebSearchQuery = "";
    std::string webSearch(const std::string& query);
    std::string summarizeWebResult(const std::string& query, const std::string& rawResult);
    bool shouldSearchWeb(const std::string& message);

    std::unique_ptr<CoreAI> core;
    std::unique_ptr<Training> trainer;
    std::unique_ptr<class Database> database;
    std::string answer(std::vector<float>& textEmbedding);
    std::vector<float> findClosestWord(const std::string& word);

    // New methods for learning from conversations
    std::vector<std::pair<std::string, std::string>> parseConversation(const std::string& conversation);
    std::unordered_map<std::string, std::vector<float>> extractContext(const std::vector<std::pair<std::string, std::string>>& parsedConversation);
    void learnFromConversation(const std::string& conversation);


};

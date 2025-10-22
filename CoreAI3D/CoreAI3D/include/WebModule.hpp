#ifndef WEB_MODULE_HPP
#define WEB_MODULE_HPP

#include "CoreAI3DCommon.hpp"
#include "Core.hpp"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <curl/curl.h>
#include <regex>

// Forward declarations
struct SearchResult {
    std::string url;
    std::string title;
    std::string description;
    std::string content;
    float relevanceScore;
    std::string date;
    std::vector<std::string> tags;
    std::map<std::string, std::string> metadata;
};

struct WebPage {
    std::string url;
    std::string title;
    std::string content;
    std::string textContent;
    std::vector<std::string> links;
    std::vector<std::string> images;
    std::map<std::string, std::string> headers;
    int statusCode;
    std::string contentType;
    size_t contentLength;
};

struct DocumentInfo {
    std::string path;
    std::string type;
    std::string title;
    std::string content;
    std::vector<std::string> sections;
    std::map<std::string, std::string> metadata;
    std::vector<float> numericalFeatures;
};

class WebModule {
private:
    std::unique_ptr<CoreAI> webCore;
    std::string moduleName;
    bool isInitialized;
    CURL* curlHandle;

    // Web processing parameters
    int timeoutSeconds;
    int maxRetries;
    int maxRedirects;
    size_t maxContentSize;
    bool followRedirects;
    bool enableJavaScript;
    bool respectRobotsTxt;

    // Search and indexing
    std::map<std::string, std::vector<float>> searchIndex;
    std::map<std::string, WebPage> cachedPages;
    std::map<std::string, DocumentInfo> indexedDocuments;

    // Content processing
    std::vector<std::string> userAgents;
    std::map<std::string, std::string> defaultHeaders;
    std::vector<std::regex> contentFilters;

protected:
    // Core web processing functions
    virtual std::vector<float> processWebContent(const std::string& content);
    virtual std::vector<float> extractWebFeatures(const WebPage& page);
    virtual float calculateRelevanceScore(const std::string& query, const WebPage& page);
    virtual std::vector<std::string> extractKeywords(const std::string& content);

    // HTTP operations
    virtual std::string performHttpRequest(const std::string& url, const std::string& method = "GET");
    virtual WebPage fetchWebPage(const std::string& url);
    virtual std::vector<std::string> extractLinks(const std::string& html);
    virtual std::vector<std::string> extractImages(const std::string& html);
    virtual std::string extractTextContent(const std::string& html);

    // Search engine integration
    virtual std::vector<SearchResult> searchGoogle(const std::string& query, int maxResults = 10);
    virtual std::vector<SearchResult> searchBing(const std::string& query, int maxResults = 10);
    virtual std::vector<SearchResult> searchDuckDuckGo(const std::string& query, int maxResults = 10);

    // Content analysis
    virtual std::string analyzeSentiment(const std::string& text);
    virtual std::vector<std::string> extractEntities(const std::string& text);
    virtual std::string summarizeContent(const std::string& content, int maxSentences = 5);
    virtual std::string translateContent(const std::string& content, const std::string& targetLanguage);

    // Document processing
    virtual DocumentInfo processPDF(const std::string& filePath);
    virtual DocumentInfo processDOCX(const std::string& filePath);
    virtual DocumentInfo processTXT(const std::string& filePath);
    virtual DocumentInfo processHTML(const std::string& filePath);

    // Numerical content extraction
    virtual std::vector<float> extractNumbersFromText(const std::string& text);
    virtual std::vector<float> extractNumbersFromWebPage(const WebPage& page);
    virtual std::vector<float> extractNumbersFromDocument(const DocumentInfo& doc);

public:
    // Constructor
    WebModule(const std::string& name);
    virtual ~WebModule();

    // Initialization
    bool initialize(const std::string& configPath = "");
    void setTimeout(int seconds);
    void setMaxRetries(int retries);
    void setMaxRedirects(int redirects);
    void setMaxContentSize(size_t maxSize);
    void setFollowRedirects(bool follow);
    void setUserAgent(const std::string& userAgent);
    void setRespectRobotsTxt(bool respect);

    // Core web search interface
    virtual std::vector<SearchResult> search(const std::string& query, int maxResults = 10);
    virtual std::vector<SearchResult> searchAdvanced(const std::map<std::string, std::string>& searchParams);
    virtual std::vector<float> searchAsNumbers(const std::string& query);

    // Web page processing
    WebPage getWebPage(const std::string& url);
    std::string getWebPageContent(const std::string& url);
    std::vector<std::string> getWebPageLinks(const std::string& url);
    std::vector<std::string> getWebPageImages(const std::string& url);

    // Content analysis
    std::string analyzeWebContent(const std::string& url);
    std::string analyzeTextContent(const std::string& content);
    std::vector<std::string> extractKeywordsFromWebPage(const std::string& url);
    std::vector<std::string> extractEntitiesFromWebPage(const std::string& url);

    // Document indexing and retrieval
    bool indexDocument(const std::string& filePath);
    bool indexWebPage(const std::string& url);
    std::vector<DocumentInfo> searchDocuments(const std::string& query);
    std::vector<float> searchDocumentsAsNumbers(const std::string& query);

    // Multi-format content extraction
    std::vector<float> extractNumbersFromFile(const std::string& filePath);
    std::vector<float> extractNumbersFromURL(const std::string& url);
    std::vector<float> extractNumbersFromContent(const std::string& content);

    // Web scraping
    struct ScrapedData {
        std::string title;
        std::vector<std::string> headings;
        std::vector<std::string> paragraphs;
        std::vector<std::string> lists;
        std::vector<std::string> tables;
        std::vector<float> numericalData;
    };

    virtual ScrapedData scrapeWebPage(const std::string& url);
    virtual std::vector<ScrapedData> scrapeMultiplePages(const std::vector<std::string>& urls);

    // News and article processing
    struct NewsArticle {
        std::string title;
        std::string content;
        std::string author;
        std::string date;
        std::string source;
        std::vector<std::string> tags;
        float sentimentScore;
        std::vector<float> keyNumbers;
    };

    virtual std::vector<NewsArticle> getLatestNews(const std::string& topic, int maxArticles = 10);
    virtual std::vector<NewsArticle> searchNews(const std::string& query, int maxArticles = 10);
    virtual NewsArticle analyzeArticle(const std::string& url);

    // Social media integration
    struct SocialMediaPost {
        std::string platform;
        std::string author;
        std::string content;
        std::string timestamp;
        int likes;
        int shares;
        int comments;
        std::vector<std::string> hashtags;
        float engagementScore;
    };

    virtual std::vector<SocialMediaPost> searchSocialMedia(const std::string& query, const std::string& platform = "");
    virtual std::vector<float> analyzeSocialMediaTrends(const std::string& topic);

    // API integration
    virtual std::string callWebAPI(const std::string& apiUrl, const std::string& method = "GET",
                                   const std::map<std::string, std::string>& headers = {},
                                   const std::string& postData = "");
    virtual std::vector<float> getAPIDataAsNumbers(const std::string& apiUrl,
                                                   const std::map<std::string, std::string>& params = {});

    // Content filtering and validation
    virtual bool isValidURL(const std::string& url);
    virtual bool isSafeContent(const std::string& content);
    virtual std::string filterSensitiveContent(const std::string& content);
    virtual std::vector<std::string> validateLinks(const std::vector<std::string>& links);

    // Knowledge base management
    virtual bool addToKnowledgeBase(const std::string& key, const std::string& content);
    virtual std::string queryKnowledgeBase(const std::string& query);
    virtual std::vector<float> queryKnowledgeBaseAsNumbers(const std::string& query);
    virtual bool saveKnowledgeBase(const std::string& filePath);
    virtual bool loadKnowledgeBase(const std::string& filePath);

    // Real-time web monitoring
    virtual bool startMonitoring(const std::string& url, int intervalSeconds = 60);
    virtual void stopMonitoring(const std::string& url);
    virtual std::vector<std::string> getMonitoringUpdates();
    virtual bool isMonitoringActive(const std::string& url);

private:
    // Monitoring data structures
    std::map<std::string, std::thread> monitoringThreads;
    std::map<std::string, bool> monitoringActive;
    std::map<std::string, std::vector<std::string>> monitoringUpdates;
    std::mutex monitoringMutex;

    // Web automation
    virtual bool automateWebInteraction(const std::string& url, const std::vector<std::string>& actions);
    virtual std::vector<float> extractDataFromWebForm(const std::string& url, const std::map<std::string, std::string>& formData);

    // Status and information
    std::string getModuleName() const { return moduleName; }
    int getTimeout() const { return timeoutSeconds; }
    size_t getCacheSize() const { return cachedPages.size(); }

    // Memory management
    void clearIndex();

public:
    // Public status and memory methods
    bool isReady() const { return isInitialized; }
    void clearCache();
    size_t getMemoryUsage() const;

    // Training interface for web-specific learning
    virtual bool trainOnWebData(const std::string& dataPath, int epochs = 10);
    virtual bool learnSearchPatterns(const std::vector<std::string>& queryResults);

    // Utility functions
    std::string sanitizeURL(const std::string& url);
    std::string extractDomain(const std::string& url);
    bool isValidEmail(const std::string& email);
    std::string formatDateForWeb(const std::string& date);
};

#endif // WEB_MODULE_HPP
#include "WebModule.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <regex>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

// WebModule constants
const int DEFAULT_TIMEOUT = 30;
const int DEFAULT_MAX_RETRIES = 3;
const int DEFAULT_MAX_REDIRECTS = 5;
const size_t DEFAULT_MAX_CONTENT_SIZE = 10 * 1024 * 1024; // 10MB

// Callback function for libcurl
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Constructor
WebModule::WebModule(const std::string& name)
    : moduleName(name), isInitialized(false), curlHandle(nullptr),
      timeoutSeconds(DEFAULT_TIMEOUT), maxRetries(DEFAULT_MAX_RETRIES),
      maxRedirects(DEFAULT_MAX_REDIRECTS), maxContentSize(DEFAULT_MAX_CONTENT_SIZE),
      followRedirects(true), enableJavaScript(false), respectRobotsTxt(true) {
}

// Destructor
WebModule::~WebModule() {
    if (curlHandle) {
        curl_easy_cleanup(curlHandle);
    }
    curl_global_cleanup();
}

// Initialization
bool WebModule::initialize(const std::string& configPath) {
    try {
        if (isInitialized) {
            return true;
        }

        // Initialize libcurl
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw std::runtime_error("Failed to initialize libcurl");
        }

        curlHandle = curl_easy_init();
        if (!curlHandle) {
            throw std::runtime_error("Failed to create curl handle");
        }

        // Initialize CoreAI for web processing
        webCore = std::make_unique<CoreAI>(256, 5, 128, 1, -1.0f, 1.0f);

        // Load configuration if provided
        if (!configPath.empty()) {
            // TODO: Load configuration from file
        }

        // Set default user agents
        userAgents = {
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        };

        // Set default headers
        defaultHeaders = {
            {"User-Agent", userAgents[0]},
            {"Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"},
            {"Accept-Language", "en-US,en;q=0.5"},
            {"Accept-Encoding", "gzip, deflate"},
            {"Connection", "keep-alive"}
        };

        isInitialized = true;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing WebModule: " << e.what() << std::endl;
        return false;
    }
}

void WebModule::setTimeout(int seconds) {
    timeoutSeconds = std::max(1, std::min(300, seconds));
}

void WebModule::setMaxRetries(int retries) {
    maxRetries = std::max(0, std::min(10, retries));
}

void WebModule::setMaxRedirects(int redirects) {
    maxRedirects = std::max(0, std::min(20, redirects));
}

void WebModule::setMaxContentSize(size_t maxSize) {
    maxContentSize = maxSize;
}

void WebModule::setFollowRedirects(bool follow) {
    followRedirects = follow;
}

void WebModule::setUserAgent(const std::string& userAgent) {
    defaultHeaders["User-Agent"] = userAgent;
}

void WebModule::setRespectRobotsTxt(bool respect) {
    respectRobotsTxt = respect;
}

// Core web search interface
std::vector<SearchResult> WebModule::search(const std::string& query, int maxResults) {
    try {
        std::vector<SearchResult> results;

        // Search multiple engines
        auto googleResults = searchGoogle(query, maxResults / 3);
        auto bingResults = searchBing(query, maxResults / 3);
        auto duckduckgoResults = searchDuckDuckGo(query, maxResults / 3);

        results.insert(results.end(), googleResults.begin(), googleResults.end());
        results.insert(results.end(), bingResults.begin(), bingResults.end());
        results.insert(results.end(), duckduckgoResults.begin(), duckduckgoResults.end());

        // Sort by relevance score
        std::sort(results.begin(), results.end(),
                 [](const SearchResult& a, const SearchResult& b) {
                     return a.relevanceScore > b.relevanceScore;
                 });

        // Limit results
        if (results.size() > static_cast<size_t>(maxResults)) {
            results.resize(maxResults);
        }

        return results;
    }
    catch (const std::exception& e) {
        std::cerr << "Error performing web search: " << e.what() << std::endl;
        return {};
    }
}

std::vector<SearchResult> WebModule::searchAdvanced(const std::map<std::string, std::string>& searchParams) {
    // TODO: Implement advanced search
    return {};
}

std::vector<float> WebModule::searchAsNumbers(const std::string& query) {
    try {
        auto results = search(query, 10);
        std::vector<float> numbers;

        for (const auto& result : results) {
            auto extractedNumbers = extractNumbersFromText(result.description + " " + result.content);
            numbers.insert(numbers.end(), extractedNumbers.begin(), extractedNumbers.end());
        }

        return numbers;
    }
    catch (const std::exception& e) {
        std::cerr << "Error performing numerical search: " << e.what() << std::endl;
        return {};
    }
}

// Web page processing
WebPage WebModule::getWebPage(const std::string& url) {
    try {
        return fetchWebPage(url);
    }
    catch (const std::exception& e) {
        std::cerr << "Error getting web page: " << e.what() << std::endl;
        return WebPage{};
    }
}

std::string WebModule::getWebPageContent(const std::string& url) {
    try {
        auto page = fetchWebPage(url);
        return page.content;
    }
    catch (const std::exception& e) {
        std::cerr << "Error getting web page content: " << e.what() << std::endl;
        return "";
    }
}

std::vector<std::string> WebModule::getWebPageLinks(const std::string& url) {
    try {
        auto page = fetchWebPage(url);
        return page.links;
    }
    catch (const std::exception& e) {
        std::cerr << "Error getting web page links: " << e.what() << std::endl;
        return {};
    }
}

std::vector<std::string> WebModule::getWebPageImages(const std::string& url) {
    try {
        auto page = fetchWebPage(url);
        return page.images;
    }
    catch (const std::exception& e) {
        std::cerr << "Error getting web page images: " << e.what() << std::endl;
        return {};
    }
}

// Content analysis
std::string WebModule::analyzeWebContent(const std::string& url) {
    try {
        auto page = fetchWebPage(url);
        return analyzeSentiment(page.textContent);
    }
    catch (const std::exception& e) {
        std::cerr << "Error analyzing web content: " << e.what() << std::endl;
        return "";
    }
}

std::string WebModule::analyzeTextContent(const std::string& content) {
    try {
        return analyzeSentiment(content);
    }
    catch (const std::exception& e) {
        std::cerr << "Error analyzing text content: " << e.what() << std::endl;
        return "";
    }
}

std::vector<std::string> WebModule::extractKeywordsFromWebPage(const std::string& url) {
    try {
        auto page = fetchWebPage(url);
        return extractKeywords(page.textContent);
    }
    catch (const std::exception& e) {
        std::cerr << "Error extracting keywords: " << e.what() << std::endl;
        return {};
    }
}

std::vector<std::string> WebModule::extractEntitiesFromWebPage(const std::string& url) {
    try {
        auto page = fetchWebPage(url);
        return extractEntities(page.textContent);
    }
    catch (const std::exception& e) {
        std::cerr << "Error extracting entities: " << e.what() << std::endl;
        return {};
    }
}

// Document indexing and retrieval
bool WebModule::indexDocument(const std::string& filePath) {
    try {
        DocumentInfo doc = processTXT(filePath);
        indexedDocuments[doc.path] = doc;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error indexing document: " << e.what() << std::endl;
        return false;
    }
}

bool WebModule::indexWebPage(const std::string& url) {
    try {
        WebPage page = fetchWebPage(url);
        DocumentInfo doc;
        doc.path = url;
        doc.type = "web";
        doc.title = page.title;
        doc.content = page.textContent;
        doc.numericalFeatures = extractNumbersFromWebPage(page);

        indexedDocuments[url] = doc;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error indexing web page: " << e.what() << std::endl;
        return false;
    }
}

std::vector<DocumentInfo> WebModule::searchDocuments(const std::string& query) {
    // TODO: Implement document search
    return {};
}

std::vector<float> WebModule::searchDocumentsAsNumbers(const std::string& query) {
    // TODO: Implement numerical document search
    return {};
}

// Multi-format content extraction
std::vector<float> WebModule::extractNumbersFromFile(const std::string& filePath) {
    try {
        DocumentInfo doc = processTXT(filePath);
        return extractNumbersFromDocument(doc);
    }
    catch (const std::exception& e) {
        std::cerr << "Error extracting numbers from file: " << e.what() << std::endl;
        return {};
    }
}

std::vector<float> WebModule::extractNumbersFromURL(const std::string& url) {
    try {
        WebPage page = fetchWebPage(url);
        return extractNumbersFromWebPage(page);
    }
    catch (const std::exception& e) {
        std::cerr << "Error extracting numbers from URL: " << e.what() << std::endl;
        return {};
    }
}

std::vector<float> WebModule::extractNumbersFromContent(const std::string& content) {
    try {
        return extractNumbersFromText(content);
    }
    catch (const std::exception& e) {
        std::cerr << "Error extracting numbers from content: " << e.what() << std::endl;
        return {};
    }
}

// Web scraping
WebModule::ScrapedData WebModule::scrapeWebPage(const std::string& url) {
    try {
        WebPage page = fetchWebPage(url);
        ScrapedData data;

        data.title = page.title;

        // Extract headings
        std::regex hRegex(R"(<h[1-6][^>]*>([^<]+)</h[1-6]>)");
        std::sregex_iterator hBegin(page.content.begin(), page.content.end(), hRegex);
        std::sregex_iterator hEnd;
        for (std::sregex_iterator i = hBegin; i != hEnd; ++i) {
            data.headings.push_back((*i)[1].str());
        }

        // Extract paragraphs
        std::regex pRegex(R"(<p[^>]*>([^<]+)</p>)");
        std::sregex_iterator pBegin(page.content.begin(), page.content.end(), pRegex);
        std::sregex_iterator pEnd;
        for (std::sregex_iterator i = pBegin; i != pEnd; ++i) {
            data.paragraphs.push_back((*i)[1].str());
        }

        // Extract lists
        std::regex liRegex(R"(<li[^>]*>([^<]+)</li>)");
        std::sregex_iterator liBegin(page.content.begin(), page.content.end(), liRegex);
        std::sregex_iterator liEnd;
        for (std::sregex_iterator i = liBegin; i != liEnd; ++i) {
            data.lists.push_back((*i)[1].str());
        }

        // Extract numerical data
        data.numericalData = extractNumbersFromText(page.textContent);

        return data;
    }
    catch (const std::exception& e) {
        std::cerr << "Error scraping web page: " << e.what() << std::endl;
        return ScrapedData{};
    }
}

std::vector<WebModule::ScrapedData> WebModule::scrapeMultiplePages(const std::vector<std::string>& urls) {
    std::vector<WebModule::ScrapedData> results;
    for (const auto& url : urls) {
        auto data = scrapeWebPage(url);
        if (!data.title.empty()) {
            results.push_back(data);
        }
    }
    return results;
}

// News and article processing
std::vector<WebModule::NewsArticle> WebModule::getLatestNews(const std::string& topic, int maxArticles) {
    // TODO: Implement news retrieval
    return {};
}

std::vector<WebModule::NewsArticle> WebModule::searchNews(const std::string& query, int maxArticles) {
    // TODO: Implement news search
    return {};
}

WebModule::NewsArticle WebModule::analyzeArticle(const std::string& url) {
    // TODO: Implement article analysis
    return NewsArticle{};
}

// Social media integration
std::vector<WebModule::SocialMediaPost> WebModule::searchSocialMedia(const std::string& query, const std::string& platform) {
    // TODO: Implement social media search
    return {};
}

std::vector<float> WebModule::analyzeSocialMediaTrends(const std::string& topic) {
    // TODO: Implement social media trend analysis
    return {};
}

// API integration
std::string WebModule::callWebAPI(const std::string& apiUrl, const std::string& method,
                                 const std::map<std::string, std::string>& headers,
                                 const std::string& postData) {
    try {
        return performHttpRequest(apiUrl, method);
    }
    catch (const std::exception& e) {
        std::cerr << "Error calling web API: " << e.what() << std::endl;
        return "";
    }
}

std::vector<float> WebModule::getAPIDataAsNumbers(const std::string& apiUrl,
                                                 const std::map<std::string, std::string>& params) {
    try {
        std::string url = apiUrl;
        if (!params.empty()) {
            url += "?";
            for (auto it = params.begin(); it != params.end(); ++it) {
                if (it != params.begin()) url += "&";
                url += it->first + "=" + it->second;
            }
        }

        std::string response = performHttpRequest(url, "GET");
        return extractNumbersFromText(response);
    }
    catch (const std::exception& e) {
        std::cerr << "Error getting API data as numbers: " << e.what() << std::endl;
        return {};
    }
}

// Content filtering and validation
bool WebModule::isValidURL(const std::string& url) {
    std::regex urlRegex(R"(^https?://)");
    return std::regex_search(url, urlRegex);
}

bool WebModule::isSafeContent(const std::string& content) {
    // TODO: Implement content safety check
    return true;
}

std::string WebModule::filterSensitiveContent(const std::string& content) {
    // TODO: Implement sensitive content filtering
    return content;
}

std::vector<std::string> WebModule::validateLinks(const std::vector<std::string>& links) {
    std::vector<std::string> validLinks;
    for (const auto& link : links) {
        if (isValidURL(link)) {
            validLinks.push_back(link);
        }
    }
    return validLinks;
}

// Knowledge base management
bool WebModule::addToKnowledgeBase(const std::string& key, const std::string& content) {
    // TODO: Implement knowledge base addition
    return false;
}

std::string WebModule::queryKnowledgeBase(const std::string& query) {
    // TODO: Implement knowledge base query
    return "";
}

std::vector<float> WebModule::queryKnowledgeBaseAsNumbers(const std::string& query) {
    // TODO: Implement numerical knowledge base query
    return {};
}

bool WebModule::saveKnowledgeBase(const std::string& filePath) {
    // TODO: Implement knowledge base saving
    return false;
}

bool WebModule::loadKnowledgeBase(const std::string& filePath) {
    // TODO: Implement knowledge base loading
    return false;
}

// Real-time web monitoring
bool WebModule::startMonitoring(const std::string& url, int intervalSeconds) {
    // TODO: Implement web monitoring
    return false;
}

void WebModule::stopMonitoring(const std::string& url) {
    // TODO: Implement monitoring stop
}

std::vector<std::string> WebModule::getMonitoringUpdates() {
    // TODO: Implement monitoring updates retrieval
    return {};
}

bool WebModule::isMonitoringActive(const std::string& url) {
    // TODO: Implement monitoring status check
    return false;
}

// Web automation
bool WebModule::automateWebInteraction(const std::string& url, const std::vector<std::string>& actions) {
    // TODO: Implement web automation
    return false;
}

std::vector<float> WebModule::extractDataFromWebForm(const std::string& url, const std::map<std::string, std::string>& formData) {
    // TODO: Implement web form data extraction
    return {};
}

// Memory management
void WebModule::clearCache() {
    cachedPages.clear();
}

void WebModule::clearIndex() {
    indexedDocuments.clear();
}

size_t WebModule::getMemoryUsage() const {
    size_t usage = cachedPages.size() * sizeof(WebPage);
    usage += indexedDocuments.size() * sizeof(DocumentInfo);
    return usage;
}

// Training interface
bool WebModule::trainOnWebData(const std::string& dataPath, int epochs) {
    // TODO: Implement web data training
    return false;
}

bool WebModule::learnSearchPatterns(const std::vector<std::string>& queryResults) {
    // TODO: Implement search pattern learning
    return false;
}

// Utility functions
std::string WebModule::sanitizeURL(const std::string& url) {
    // TODO: Implement URL sanitization
    return url;
}

std::string WebModule::extractDomain(const std::string& url) {
    try {
        std::regex domainRegex(R"(https?://([^/]+))");
        std::smatch match;
        if (std::regex_search(url, match, domainRegex)) {
            return match[1].str();
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error extracting domain: " << e.what() << std::endl;
    }
    return "";
}

bool WebModule::isValidEmail(const std::string& email) {
    std::regex emailRegex(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");
    return std::regex_match(email, emailRegex);
}

std::string WebModule::formatDateForWeb(const std::string& date) {
    // TODO: Implement date formatting for web
    return date;
}

// Protected methods implementation
std::vector<float> WebModule::processWebContent(const std::string& content) {
    // Basic web content processing - extract numerical features
    return extractNumbersFromText(content);
}

std::vector<float> WebModule::extractWebFeatures(const WebPage& page) {
    std::vector<float> features;

    // Content length
    features.push_back(static_cast<float>(page.content.length()));

    // Text content length
    features.push_back(static_cast<float>(page.textContent.length()));

    // Number of links
    features.push_back(static_cast<float>(page.links.size()));

    // Number of images
    features.push_back(static_cast<float>(page.images.size()));

    // Title length
    features.push_back(static_cast<float>(page.title.length()));

    return features;
}

float WebModule::calculateRelevanceScore(const std::string& query, const WebPage& page) {
    // Simple relevance scoring based on keyword matches
    std::string queryLower = query;
    std::string contentLower = page.textContent;
    std::string titleLower = page.title;

    std::transform(queryLower.begin(), queryLower.end(), queryLower.begin(), ::tolower);
    std::transform(contentLower.begin(), contentLower.end(), contentLower.begin(), ::tolower);
    std::transform(titleLower.begin(), titleLower.end(), titleLower.begin(), ::tolower);

    float score = 0.0f;

    // Title matches are more important
    size_t titleMatches = 0;
    size_t pos = 0;
    while ((pos = titleLower.find(queryLower, pos)) != std::string::npos) {
        titleMatches++;
        pos += queryLower.length();
    }
    score += titleMatches * 3.0f;

    // Content matches
    size_t contentMatches = 0;
    pos = 0;
    while ((pos = contentLower.find(queryLower, pos)) != std::string::npos) {
        contentMatches++;
        pos += queryLower.length();
    }
    score += contentMatches * 1.0f;

    return score;
}

std::vector<std::string> WebModule::extractKeywords(const std::string& content) {
    // Simple keyword extraction - split by spaces and filter
    std::vector<std::string> keywords;
    std::istringstream iss(content);
    std::string word;

    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());

        // Convert to lowercase
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        // Filter out short words and common stop words
        if (word.length() > 3) {
            keywords.push_back(word);
        }
    }

    return keywords;
}

// HTTP operations
std::string WebModule::performHttpRequest(const std::string& url, const std::string& method) {
    if (!curlHandle) {
        throw std::runtime_error("Curl handle not initialized");
    }

    std::string response;
    struct curl_slist* headers = nullptr;

    try {
        // Set URL
        curl_easy_setopt(curlHandle, CURLOPT_URL, url.c_str());

        // Set method
        if (method == "POST") {
            curl_easy_setopt(curlHandle, CURLOPT_POST, 1L);
        } else {
            curl_easy_setopt(curlHandle, CURLOPT_HTTPGET, 1L);
        }

        // Set headers
        for (const auto& header : defaultHeaders) {
            std::string headerStr = header.first + ": " + header.second;
            headers = curl_slist_append(headers, headerStr.c_str());
        }
        curl_easy_setopt(curlHandle, CURLOPT_HTTPHEADER, headers);

        // Set response callback
        curl_easy_setopt(curlHandle, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curlHandle, CURLOPT_WRITEDATA, &response);

        // Set timeouts
        curl_easy_setopt(curlHandle, CURLOPT_TIMEOUT, timeoutSeconds);
        curl_easy_setopt(curlHandle, CURLOPT_FOLLOWLOCATION, followRedirects ? 1L : 0L);
        curl_easy_setopt(curlHandle, CURLOPT_MAXREDIRS, maxRedirects);

        // Perform request
        CURLcode res = curl_easy_perform(curlHandle);
        if (res != CURLE_OK) {
            throw std::runtime_error(curl_easy_strerror(res));
        }

        // Cleanup headers
        if (headers) {
            curl_slist_free_all(headers);
        }

        return response;
    }
    catch (const std::exception& e) {
        if (headers) {
            curl_slist_free_all(headers);
        }
        throw;
    }
}

WebPage WebModule::fetchWebPage(const std::string& url) {
    WebPage page;
    page.url = url;

    try {
        std::string html = performHttpRequest(url, "GET");

        page.content = html;
        page.textContent = extractTextContent(html);
        page.links = extractLinks(html);
        page.images = extractImages(html);

        // Extract title
        std::regex titleRegex(R"(<title[^>]*>([^<]+)</title>)");
        std::smatch titleMatch;
        if (std::regex_search(html, titleMatch, titleRegex)) {
            page.title = titleMatch[1].str();
        }

        return page;
    }
    catch (const std::exception& e) {
        std::cerr << "Error fetching web page: " << e.what() << std::endl;
        return page;
    }
}

std::vector<std::string> WebModule::extractLinks(const std::string& html) {
    std::vector<std::string> links;
    std::regex linkRegex(R"(<a[^>]+href=["']([^"']+)["'][^>]*>)");
    std::sregex_iterator linkBegin(html.begin(), html.end(), linkRegex);
    std::sregex_iterator linkEnd;

    for (std::sregex_iterator i = linkBegin; i != linkEnd; ++i) {
        std::string link = (*i)[1].str();
        if (!link.empty() && link[0] != '#') {
            links.push_back(link);
        }
    }

    return links;
}

std::vector<std::string> WebModule::extractImages(const std::string& html) {
    std::vector<std::string> images;
    std::regex imgRegex(R"(<img[^>]+src=["']([^"']+)["'][^>]*>)");
    std::sregex_iterator imgBegin(html.begin(), html.end(), imgRegex);
    std::sregex_iterator imgEnd;

    for (std::sregex_iterator i = imgBegin; i != imgEnd; ++i) {
        images.push_back((*i)[1].str());
    }

    return images;
}

std::string WebModule::extractTextContent(const std::string& html) {
    std::string text = html;

    // Remove script and style elements
    std::regex scriptRegex(R"(<script[^>]*>.*?</script>)");
    std::regex styleRegex(R"(<style[^>]*>.*?</style>)");
    text = std::regex_replace(text, scriptRegex, "");
    text = std::regex_replace(text, styleRegex, "");

    // Remove HTML tags
    std::regex tagRegex(R"(<[^>]+>)");
    text = std::regex_replace(text, tagRegex, " ");

    // Decode HTML entities
    std::regex ampRegex(R"(&)");
    std::regex ltRegex(R"(<)");
    std::regex gtRegex(R"(>)");
    std::regex quotRegex(R"(")");
    text = std::regex_replace(text, ampRegex, "&");
    text = std::regex_replace(text, ltRegex, "<");
    text = std::regex_replace(text, gtRegex, ">");
    text = std::regex_replace(text, quotRegex, "\"");

    // Clean up whitespace
    std::regex whitespaceRegex(R"(\s+)");
    text = std::regex_replace(text, whitespaceRegex, " ");

    return text;
}

// Search engine integration
std::vector<SearchResult> WebModule::searchGoogle(const std::string& query, int maxResults) {
    // TODO: Implement Google search
    return {};
}

std::vector<SearchResult> WebModule::searchBing(const std::string& query, int maxResults) {
    // TODO: Implement Bing search
    return {};
}

std::vector<SearchResult> WebModule::searchDuckDuckGo(const std::string& query, int maxResults) {
    // TODO: Implement DuckDuckGo search
    return {};
}

// Content analysis
std::string WebModule::analyzeSentiment(const std::string& text) {
    // TODO: Implement sentiment analysis
    return "neutral";
}

std::vector<std::string> WebModule::extractEntities(const std::string& text) {
    // TODO: Implement entity extraction
    return {};
}

std::string WebModule::summarizeContent(const std::string& content, int maxSentences) {
    // TODO: Implement content summarization
    return content.substr(0, 200) + "...";
}

std::string WebModule::translateContent(const std::string& content, const std::string& targetLanguage) {
    // TODO: Implement content translation
    return content;
}

// Document processing
DocumentInfo WebModule::processPDF(const std::string& filePath) {
    // TODO: Implement PDF processing
    return DocumentInfo{};
}

DocumentInfo WebModule::processDOCX(const std::string& filePath) {
    // TODO: Implement DOCX processing
    return DocumentInfo{};
}

DocumentInfo WebModule::processTXT(const std::string& filePath) {
    DocumentInfo doc;
    doc.path = filePath;
    doc.type = "txt";

    try {
        std::ifstream file(filePath);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                doc.content += line + "\n";
            }
            file.close();

            // Extract title from first line
            if (!doc.content.empty()) {
                std::istringstream iss(doc.content);
                std::getline(iss, doc.title);
            }

            doc.numericalFeatures = extractNumbersFromText(doc.content);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing TXT file: " << e.what() << std::endl;
    }

    return doc;
}

DocumentInfo WebModule::processHTML(const std::string& filePath) {
    // TODO: Implement HTML processing
    return DocumentInfo{};
}

// Numerical content extraction
std::vector<float> WebModule::extractNumbersFromText(const std::string& text) {
    std::vector<float> numbers;
    std::regex numberRegex(R"(-?\d+\.?\d*)");
    std::sregex_iterator numBegin(text.begin(), text.end(), numberRegex);
    std::sregex_iterator numEnd;

    for (std::sregex_iterator i = numBegin; i != numEnd; ++i) {
        try {
            numbers.push_back(std::stof((*i)[0].str()));
        }
        catch (const std::exception&) {
            // Skip invalid numbers
        }
    }

    return numbers;
}

std::vector<float> WebModule::extractNumbersFromWebPage(const WebPage& page) {
    std::vector<float> numbers;

    // Extract from title
    auto titleNumbers = extractNumbersFromText(page.title);
    numbers.insert(numbers.end(), titleNumbers.begin(), titleNumbers.end());

    // Extract from content
    auto contentNumbers = extractNumbersFromText(page.content);
    numbers.insert(numbers.end(), contentNumbers.begin(), contentNumbers.end());

    // Extract from text content
    auto textNumbers = extractNumbersFromText(page.textContent);
    numbers.insert(numbers.end(), textNumbers.begin(), textNumbers.end());

    return numbers;
}

std::vector<float> WebModule::extractNumbersFromDocument(const DocumentInfo& doc) {
    return extractNumbersFromText(doc.content);
}
#ifndef GEMINI_H
#define GEMINI_H

#include "main.hpp"

using json = nlohmann::json;

class Gemini
{
public:
    Gemini(const std::string& apiKey);
    ~Gemini();

    std::future<std::string> sendRequestAsync(const std::string& endpoint,
        const std::string& payload);

    static size_t writeCallback(char* contents, size_t size, size_t nmemb,
        std::string* output);

private:
    std::string apiKey;
    CURL* curl;
};

#endif
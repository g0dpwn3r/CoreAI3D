#include "Gemini.hpp"

Gemini::Gemini(const std::string& apiKey) : apiKey(apiKey), curl(curl_easy_init()) {
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl");
    }
}

Gemini::~Gemini() {
    if (curl) {
        curl_easy_cleanup(curl);
    }
}

size_t Gemini::writeCallback(char* contents, size_t size, size_t nmemb, std::string* output) {
    size_t totalSize = size * nmemb;
    output->append(contents, totalSize);
    return totalSize;
}

std::future<std::string> Gemini::sendRequestAsync(const std::string& endpoint, const std::string& payload) {
    std::packaged_task<std::string()> task([this, endpoint, payload]() {
        std::string responseData;
        CURLcode res;

        curl_easy_setopt(curl, CURLOPT_URL, endpoint.c_str());
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());

        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, "Content-Type: application/json");
        headers = curl_slist_append(headers, ("x-goog-api-key: " + this->apiKey).c_str()); // Include API key
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, Gemini::writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseData);

        res = curl_easy_perform(curl);

        curl_slist_free_all(headers);

        if (res != CURLE_OK) {
            throw std::runtime_error("curl_easy_perform() failed: " + std::string(curl_easy_strerror(res)));
        }
        return responseData;
        });

    std::future<std::string> future = task.get_future();
    std::thread(std::move(task)).detach();

    return future;
}
#ifndef AUTH_TABLES_HPP
#define AUTH_TABLES_HPP

#include "CoreAI3DCommon.hpp"
#include "Database.hpp"
#include <string>
#include <optional>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <memory>

namespace AuthConstants {
    constexpr int JWT_EXPIRY_HOURS = 1;
    constexpr int JWT_EXPIRY_MINUTES = 15;
    constexpr int REFRESH_TOKEN_EXPIRY_DAYS = 7;
    constexpr size_t API_KEY_LENGTH = 32;
    constexpr size_t SESSION_ID_LENGTH = 64;
}

struct User {
    int64_t id;
    std::string username;
    std::string email;
    std::string passwordHash;
    std::string salt;
};

struct Session {
    std::string id;
    int64_t userId;
    std::string jwtToken;
};

struct APIKey {
    std::string key;
    int64_t userId;
    std::string description;
    bool isActive;
};

class AuthTables {
public:
    explicit AuthTables(Database* db);
    
    bool createUser(const std::string& username, const std::string& password, 
                    const std::string& email = "");
    std::optional<User> getUserByUsername(const std::string& username);
    std::optional<User> getUserById(int64_t userId);
    bool deleteUser(int64_t userId);
    
    std::string createSession(int64_t userId);
    bool validateSession(const std::string& sessionId, std::string& outJwtToken);
    void invalidateSession(const std::string& sessionId);
    void cleanupExpiredSessions();
    
    std::string generateAPIKey();
    bool saveAPIKey(const std::string& apiKey, int64_t userId, 
                    const std::string& description = "");
    bool validateAPIKey(const std::string& apiKey, int64_t& outUserId);
    void deactivateAPIKey(const std::string& apiKey);
    
    std::string hashPassword(const std::string& password, std::string& salt);
    bool verifyPassword(const std::string& password, const std::string& passwordHash, 
                        const std::string& salt);
    
    std::string generateJWT(int64_t userId, const std::string& username);
    bool validateJWT(const std::string& token, int64_t& outUserId, std::string& outUsername);

private:
    Database* database;
    
    std::string generateSecureRandomString(size_t length);
    std::string base64Encode(const std::string& input);
    std::string base64Decode(const std::string& input);
    std::string hmacSha256(const std::string& data, const std::string& key);
};

#endif // AUTH_TABLES_HPP
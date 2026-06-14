#include "AuthTables.hpp"
#include <openssl/rand.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/bn.h>
#include <iomanip>
#include <sstream>
#include <random>
#include <memory>

using namespace std::chrono;

AuthTables::AuthTables(Database* db) : database(db) {}

std::string AuthTables::generateSecureRandomString(size_t length) {
    static const char chars[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string result;
    result.reserve(length);
    
    unsigned char randomBytes[32];
    for (size_t i = 0; i < length; ++i) {
        if (length - i >= sizeof(randomBytes)) {
            RAND_bytes(randomBytes, sizeof(randomBytes));
            for (size_t j = 0; j < sizeof(randomBytes) && result.size() < length; ++j) {
                result += chars[randomBytes[j] % (sizeof(chars) - 1)];
            }
        } else {
            RAND_bytes(randomBytes, 1);
            result += chars[randomBytes[0] % (sizeof(chars) - 1)];
        }
    }
    return result;
}

std::string AuthTables::base64Encode(const std::string& input) {
    BIO* bio = BIO_new(BIO_f_base64());
    BIO* bmem = BIO_new(BIO_s_mem());
    bio = BIO_push(bio, bmem);
    
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(bio, input.c_str(), static_cast<int>(input.length()));
    BIO_flush(bio);
    
    char* ptr;
    long len = BIO_get_mem_data(bmem, &ptr);
    std::string result(ptr, static_cast<size_t>(len));
    
    BIO_free_all(bio);
    return result;
}

std::string AuthTables::base64Decode(const std::string& input) {
    BIO* bio = BIO_new(BIO_f_base64());
    BIO* bmem = BIO_new(BIO_s_mem());
    bio = BIO_push(bio, bmem);
    
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL);
    BIO_write(bio, input.c_str(), static_cast<int>(input.length()));
    BIO_flush(bio);
    
    char ptr[4096];
    long len = BIO_read(bio, ptr, static_cast<int>(input.length()));
    std::string result(ptr, static_cast<size_t>(len));
    
    BIO_free_all(bio);
    return result;
}

std::string AuthTables::hmacSha256(const std::string& data, const std::string& key) {
    unsigned int len = 0;
    unsigned char* digest = new unsigned char[EVP_MAX_MD_SIZE];
    
    HMAC_CTX* ctx = HMAC_CTX_new();
    if (!ctx) {
        delete[] digest;
        return "";
    }
    
    HMAC_Init_ex(ctx, key.c_str(), static_cast<int>(key.length()), EVP_sha256(), nullptr);
    HMAC_Update(ctx, reinterpret_cast<const unsigned char*>(data.c_str()), data.length());
    HMAC_Final(ctx, digest, &len);
    
    std::stringstream ss;
    for (unsigned int i = 0; i < len; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(digest[i]);
    }
    
    HMAC_CTX_free(ctx);
    delete[] digest;
    
    return ss.str();
}

std::string AuthTables::hashPassword(const std::string& password, std::string& salt) {
    salt = generateSecureRandomString(16);
    
    if (password.empty() || salt.empty()) {
        return "";
    }
    
    auto ctx = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(
        EVP_MD_CTX_new(), EVP_MD_CTX_free);
    
    if (!ctx) {
        return "";
    }
    
    unsigned char hash[64];
    
    if (EVP_DigestInit_ex(ctx.get(), EVP_sha512(), nullptr) != 1) {
        return "";
    }
    
    std::string saltPassword = salt + password;
    
    if (EVP_DigestUpdate(ctx.get(), saltPassword.c_str(), saltPassword.length()) != 1) {
        return "";
    }
    
    unsigned int hashLen = 0;
    if (EVP_DigestFinal_ex(ctx.get(), hash, &hashLen) != 1) {
        return "";
    }
    
    std::stringstream ss;
    for (unsigned int i = 0; i < hashLen; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    
    return ss.str();
}

bool AuthTables::verifyPassword(const std::string& password, const std::string& passwordHash,
                               const std::string& salt) {
    if (passwordHash.empty() || salt.empty()) {
        return false;
    }
    
    std::string computedHash = hashPassword(password, const_cast<std::string&>(salt));
    
    if (computedHash.empty()) {
        return false;
    }
    
    return passwordHash == computedHash;
}

bool AuthTables::createUser(const std::string& username, const std::string& password,
                           const std::string& email) {
    if (username.empty() || password.empty()) {
        return false;
    }
    
    try {
        std::string salt;
        std::string passwordHash = hashPassword(password, salt);
        
        if (passwordHash.empty()) {
            return false;
        }
        
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        database->session->sql(
            "INSERT INTO " + database->dbSchema + ".users (username, password_hash, email, salt) "
            "VALUES (?, ?, ?, ?)"
        ).bind(username)
         .bind(passwordHash)
         .bind(email)
         .bind(salt)
         .execute();
        return true;
        #else
        std::cout << "Database stub: createUser(username='" << username << "')" << std::endl;
        return true;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error creating user: " << e.what() << std::endl;
        return false;
    }
}

std::optional<User> AuthTables::getUserByUsername(const std::string& username) {
    try {
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        auto result = database->session->sql(
            "SELECT id, username, email, password_hash, salt FROM "
            + database->dbSchema + ".users WHERE username = ?"
        ).bind(username)
         .execute();
        
        auto row = result.fetchOne();
        if (!row) {
            return std::nullopt;
        }
        
        User user;
        user.id = row[0].get<int64_t>();
        user.username = row[1].get<std::string>();
        user.email = row[2].get<std::string>();
        user.passwordHash = row[3].get<std::string>();
        user.salt = row[4].get<std::string>();
        
        return user;
        #else
        std::cout << "Database stub: getUserByUsername(username='" << username << "')" << std::endl;
        return std::nullopt;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error getting user by username: " << e.what() << std::endl;
        return std::nullopt;
    }
}

std::optional<User> AuthTables::getUserById(int64_t userId) {
    try {
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        auto result = database->session->sql(
            "SELECT id, username, email, password_hash, salt FROM "
            + database->dbSchema + ".users WHERE id = ?"
        ).bind(userId)
         .execute();
        
        auto row = result.fetchOne();
        if (!row) {
            return std::nullopt;
        }
        
        User user;
        user.id = row[0].get<int64_t>();
        user.username = row[1].get<std::string>();
        user.email = row[2].get<std::string>();
        user.passwordHash = row[3].get<std::string>();
        user.salt = row[4].get<std::string>();
        
        return user;
        #else
        std::cout << "Database stub: getUserById(userId=" << userId << ")" << std::endl;
        return std::nullopt;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error getting user by ID: " << e.what() << std::endl;
        return std::nullopt;
    }
}

bool AuthTables::deleteUser(int64_t userId) {
    try {
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        database->session->sql(
            "DELETE FROM " + database->dbSchema + ".users WHERE id = ?"
        ).bind(userId)
         .execute();
        
        return true;
        #else
        std::cout << "Database stub: deleteUser(userId=" << userId << ")" << std::endl;
        return true;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error deleting user: " << e.what() << std::endl;
        return false;
    }
}

std::string AuthTables::generateJWT(int64_t userId, const std::string& username) {
    if (userId <= 0 || username.empty()) {
        return "";
    }
    
    auto now = std::chrono::system_clock::now();
    auto expiry = now + std::chrono::minutes(AuthConstants::JWT_EXPIRY_MINUTES);
    
    auto nowTime_t = std::chrono::system_clock::to_time_t(now);
    auto expiryTime_t = std::chrono::system_clock::to_time_t(expiry);
    
    json header = {{"alg", "HS256"}, {"typ", "JWT"}};
    json payload = {
        {"sub", userId},
        {"username", username},
        {"iat", static_cast<int64_t>(nowTime_t)},
        {"exp", static_cast<int64_t>(expiryTime_t)},
        {"jti", generateSecureRandomString(16)}
    };
    
    std::string headerB64 = base64Encode(header.dump());
    std::string payloadB64 = base64Encode(payload.dump());
    
    std::string signingInput = headerB64 + "." + payloadB64;
    
    static const std::string jwtSecret = []() -> std::string {
        const char* envSecret = std::getenv("JWT_SECRET");
        if (envSecret && !std::string(envSecret).empty()) {
            return std::string(envSecret);
        }
        return "CoreAI3D_JWT_SuperSecretKey_ChangeMeInProduction!";
    }();
    
    std::string signature = hmacSha256(signingInput, jwtSecret);
    std::string signatureB64 = base64Encode(signature);
    
    std::replace(signatureB64.begin(), signatureB64.end(), '+', '-');
    std::replace(signatureB64.begin(), signatureB64.end(), '/', '_');
    
    std::string token = headerB64 + "." + payloadB64 + "." + signatureB64;
    
    std::replace(token.begin(), token.end(), '+', '-');
    std::replace(token.begin(), token.end(), '/', '_');
    
    return token;
}

bool AuthTables::validateJWT(const std::string& token, int64_t& outUserId, std::string& outUsername) {
    if (token.empty() || token.find('.') == std::string::npos) {
        return false;
    }
    
    try {
        std::vector<std::string> parts;
        std::stringstream ss(token);
        std::string part;
        while (std::getline(ss, part, '.')) {
            if (!part.empty()) {
                std::replace(part.begin(), part.end(), '-', '+');
                std::replace(part.begin(), part.end(), '_', '/');
                parts.push_back(part);
            }
        }
        
        if (parts.size() != 3) {
            return false;
        }
        
        std::string headerB64 = parts[0];
        std::string payloadB64 = parts[1];
        std::string signatureB64 = parts[2];
        
        std::string signingInput = headerB64 + "." + payloadB64;
        
        static const std::string jwtSecret = []() -> std::string {
            const char* envSecret = std::getenv("JWT_SECRET");
            if (envSecret && !std::string(envSecret).empty()) {
                return std::string(envSecret);
            }
            return "CoreAI3D_JWT_SuperSecretKey_ChangeMeInProduction!";
        }();
        
        std::string expectedSignature = hmacSha256(signingInput, jwtSecret);
        
        std::replace(expectedSignature.begin(), expectedSignature.end(), '+', '-');
        std::replace(expectedSignature.begin(), expectedSignature.end(), '/', '_');
        
        if (expectedSignature != signatureB64) {
            return false;
        }
        
        std::string payloadJson = base64Decode(payloadB64);
        json payload = json::parse(payloadJson);
        
        auto now = std::chrono::system_clock::now();
        auto nowTime_t = std::chrono::system_clock::to_time_t(now);
        
        int64_t exp = payload.value("exp", 0);
        if (exp < static_cast<int64_t>(nowTime_t)) {
            return false;
        }
        
        outUserId = payload.value("sub", 0);
        outUsername = payload.value("username", "");
        
        if (outUserId <= 0 || outUsername.empty()) {
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "JWT validation error: " << e.what() << std::endl;
        return false;
    }
}

std::string AuthTables::createSession(int64_t userId) {
    try {
        std::string sessionId = generateSecureRandomString(AuthConstants::SESSION_ID_LENGTH);
        std::string jwtToken = generateJWT(userId, "");
        
        if (userId <= 0 || jwtToken.empty()) {
            return "";
        }
        
        auto now = std::chrono::system_clock::now();
        auto expiry = now + std::chrono::hours(AuthConstants::JWT_EXPIRY_HOURS) + 
                      std::chrono::minutes(AuthConstants::JWT_EXPIRY_MINUTES);
        
        auto expiryTime_t = std::chrono::system_clock::to_time_t(expiry);
        
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        database->session->sql(
            "INSERT INTO " + database->dbSchema + ".sessions (id, user_id, jwt_token, expires_at) "
            "VALUES (?, ?, ?, FROM_UNIXTIME(?))"
        ).bind(sessionId)
         .bind(userId)
         .bind(jwtToken)
         .bind(static_cast<int64_t>(expiryTime_t))
         .execute();
        #else
        std::cout << "Database stub: createSession(userId=" << userId << ", sessionId=" << sessionId.substr(0, 16) << "...)" << std::endl;
        #endif
        
        return sessionId;
    } catch (const std::exception& e) {
        std::cerr << "Error creating session: " << e.what() << std::endl;
        return "";
    }
}

bool AuthTables::validateSession(const std::string& sessionId, std::string& outJwtToken) {
    if (sessionId.empty()) {
        return false;
    }
    
    try {
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        auto result = database->session->sql(
            "SELECT jwt_token, expires_at FROM " + database->dbSchema + ".sessions "
            "WHERE id = ? AND expires_at > NOW()"
        ).bind(sessionId)
         .execute();
        
        auto row = result.fetchOne();
        if (!row) {
            return false;
        }
        
        outJwtToken = row[0].get<std::string>();
        return true;
        #else
        std::cout << "Database stub: validateSession(sessionId=" << sessionId.substr(0, 16) << "...)" << std::endl;
        outJwtToken = "stub_jwt_token";
        return true;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error validating session: " << e.what() << std::endl;
        return false;
    }
}

void AuthTables::invalidateSession(const std::string& sessionId) {
    if (sessionId.empty()) {
        return;
    }
    
    try {
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        database->session->sql(
            "DELETE FROM " + database->dbSchema + ".sessions WHERE id = ?"
        ).bind(sessionId)
         .execute();
        #else
        std::cout << "Database stub: invalidateSession(sessionId=" << sessionId.substr(0, 16) << "...)" << std::endl;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error invalidating session: " << e.what() << std::endl;
    }
}

void AuthTables::cleanupExpiredSessions() {
    try {
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        database->session->sql(
            "DELETE FROM " + database->dbSchema + ".sessions WHERE expires_at < NOW()"
        ).execute();
        #else
        std::cout << "Database stub: cleanupExpiredSessions()" << std::endl;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning up expired sessions: " << e.what() << std::endl;
    }
}

std::string AuthTables::generateAPIKey() {
    return "ck_" + generateSecureRandomString(AuthConstants::API_KEY_LENGTH);
}

bool AuthTables::saveAPIKey(const std::string& apiKey, int64_t userId, 
                            const std::string& description) {
    if (apiKey.empty() || userId <= 0) {
        return false;
    }
    
    try {
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        database->session->sql(
            "INSERT INTO " + database->dbSchema + ".api_keys (key, user_id, description, is_active) "
            "VALUES (?, ?, ?, TRUE) "
            "ON DUPLICATE KEY UPDATE description = VALUES(description), is_active = TRUE"
        ).bind(apiKey)
         .bind(userId)
         .bind(description)
         .execute();
        #else
        std::cout << "Database stub: saveAPIKey(apiKey=" << apiKey.substr(0, 12) << "..., userId=" << userId << ")" << std::endl;
        #endif
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving API key: " << e.what() << std::endl;
        return false;
    }
}

bool AuthTables::validateAPIKey(const std::string& apiKey, int64_t& outUserId) {
    if (apiKey.empty()) {
        return false;
    }
    
    try {
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        auto result = database->session->sql(
            "SELECT user_id FROM " + database->dbSchema + ".api_keys "
            "WHERE key = ? AND is_active = TRUE"
        ).bind(apiKey)
         .execute();
        
        auto row = result.fetchOne();
        if (!row) {
            return false;
        }
        
        outUserId = row[0].get<int64_t>();
        return true;
        #else
        std::cout << "Database stub: validateAPIKey(apiKey=" << apiKey.substr(0, 12) << "...)" << std::endl;
        outUserId = 1;
        return true;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error validating API key: " << e.what() << std::endl;
        return false;
    }
}

void AuthTables::deactivateAPIKey(const std::string& apiKey) {
    if (apiKey.empty()) {
        return;
    }
    
    try {
        #ifdef USE_MYSQL
        database->ensureConnection();
        
        database->session->sql(
            "UPDATE " + database->dbSchema + ".api_keys SET is_active = FALSE "
            "WHERE key = ?"
        ).bind(apiKey)
         .execute();
        #else
        std::cout << "Database stub: deactivateAPIKey(apiKey=" << apiKey.substr(0, 12) << "...)" << std::endl;
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Error deactivating API key: " << e.what() << std::endl;
    }
}
#ifndef TEST_HELPERS_HPP
#define TEST_HELPERS_HPP

#include <vector>
#include <string>
#include <functional>
#include <memory>

class TestFixture {
public:
    virtual ~TestFixture() = default;
    virtual void setUp() {}
    virtual void tearDown() {}
};

// RAII helper for test setup/teardown
class TestEnvironment {
public:
    TestEnvironment(std::function<void()> setup, std::function<void()> teardown);
    ~TestEnvironment();

private:
    std::function<void()> teardown_;
};

class TestHelpers {
public:
    // CSV file I/O
    static bool loadCSV(const std::string& filename,
                       std::vector<std::vector<float>>& data,
                       bool hasHeader = false);

    static bool saveCSV(const std::string& filename,
                       const std::vector<std::vector<float>>& data,
                       const std::vector<std::string>& headers = {});

    static bool loadCSVLabels(const std::string& filename,
                             std::vector<std::vector<float>>& features,
                             std::vector<int>& labels,
                             bool hasHeader = false);

    // File existence and operations
    static bool fileExists(const std::string& filename);
    static bool createDirectory(const std::string& path);
    static bool removeFile(const std::string& filename);
    static std::string getTempFilePath(const std::string& prefix = "test_");

    // Assertions (beyond GTest)
    static void assertVectorsEqual(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  float tolerance = 1e-6f,
                                  const std::string& message = "");

    static void assertMatricesEqual(const std::vector<std::vector<float>>& a,
                                   const std::vector<std::vector<float>>& b,
                                   float tolerance = 1e-6f,
                                   const std::string& message = "");

    static void assertVectorsNear(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 float tolerance = 1e-6f,
                                 const std::string& message = "");

    static void assertMatricesNear(const std::vector<std::vector<float>>& a,
                                  const std::vector<std::vector<float>>& b,
                                  float tolerance = 1e-6f,
                                  const std::string& message = "");

    // Test utilities
    static void randomizeVector(std::vector<float>& vec, float minVal = -1.0f, float maxVal = 1.0f);
    static void randomizeMatrix(std::vector<std::vector<float>>& matrix, float minVal = -1.0f, float maxVal = 1.0f);

    // Memory and performance checks
    static size_t getCurrentMemoryUsage();
    static void logMemoryUsage(const std::string& label = "");

    // Test fixture helpers
    template<typename FixtureType>
    static std::unique_ptr<FixtureType> createFixture() {
        auto fixture = std::make_unique<FixtureType>();
        fixture->setUp();
        return fixture;
    }

    static void destroyFixture(TestFixture* fixture) {
        if (fixture) {
            fixture->tearDown();
        }
    }

private:
    static std::vector<std::string> splitCSVLine(const std::string& line, char delimiter = ',');
    static std::string joinCSVLine(const std::vector<std::string>& fields, char delimiter = ',');
};

#endif // TEST_HELPERS_HPP
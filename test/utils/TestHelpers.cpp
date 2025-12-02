#include "TestHelpers.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

TestEnvironment::TestEnvironment(std::function<void()> setup, std::function<void()> teardown)
    : teardown_(teardown) {
    if (setup) {
        setup();
    }
}

TestEnvironment::~TestEnvironment() {
    if (teardown_) {
        teardown_();
    }
}

bool TestHelpers::loadCSV(const std::string& filename,
                         std::vector<std::vector<float>>& data,
                         bool hasHeader) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    data.clear();
    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        if (firstLine && hasHeader) {
            firstLine = false;
            continue;
        }

        auto fields = splitCSVLine(line);
        std::vector<float> row;

        for (const auto& field : fields) {
            try {
                row.push_back(std::stof(field));
            } catch (const std::exception&) {
                // Skip invalid numbers
                continue;
            }
        }

        if (!row.empty()) {
            data.push_back(std::move(row));
        }
    }

    return true;
}

bool TestHelpers::saveCSV(const std::string& filename,
                         const std::vector<std::vector<float>>& data,
                         const std::vector<std::string>& headers) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    // Write headers
    if (!headers.empty()) {
        file << joinCSVLine(headers) << "\n";
    }

    // Write data
    for (const auto& row : data) {
        std::vector<std::string> fields;
        for (float val : row) {
            fields.push_back(std::to_string(val));
        }
        file << joinCSVLine(fields) << "\n";
    }

    return true;
}

bool TestHelpers::loadCSVLabels(const std::string& filename,
                               std::vector<std::vector<float>>& features,
                               std::vector<int>& labels,
                               bool hasHeader) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    features.clear();
    labels.clear();
    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        if (firstLine && hasHeader) {
            firstLine = false;
            continue;
        }

        auto fields = splitCSVLine(line);
        if (fields.size() < 2) continue;

        std::vector<float> featureRow;
        for (size_t i = 0; i < fields.size() - 1; ++i) {
            try {
                featureRow.push_back(std::stof(fields[i]));
            } catch (const std::exception&) {
                continue;
            }
        }

        try {
            int label = std::stoi(fields.back());
            features.push_back(std::move(featureRow));
            labels.push_back(label);
        } catch (const std::exception&) {
            continue;
        }
    }

    return true;
}

bool TestHelpers::fileExists(const std::string& filename) {
    return std::filesystem::exists(filename);
}

bool TestHelpers::createDirectory(const std::string& path) {
    try {
        return std::filesystem::create_directories(path);
    } catch (const std::exception&) {
        return false;
    }
}

bool TestHelpers::removeFile(const std::string& filename) {
    try {
        return std::filesystem::remove(filename);
    } catch (const std::exception&) {
        return false;
    }
}

std::string TestHelpers::getTempFilePath(const std::string& prefix) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 999999);

    std::stringstream ss;
    ss << prefix << dis(gen) << ".tmp";
    return ss.str();
}

void TestHelpers::assertVectorsEqual(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    float tolerance,
                                    const std::string& message) {
    if (a.size() != b.size()) {
        throw std::runtime_error(message + ": Vector sizes differ (" +
                                std::to_string(a.size()) + " vs " +
                                std::to_string(b.size()) + ")");
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            throw std::runtime_error(message + ": Vectors differ at index " +
                                    std::to_string(i) + " (" +
                                    std::to_string(a[i]) + " vs " +
                                    std::to_string(b[i]) + ")");
        }
    }
}

void TestHelpers::assertMatricesEqual(const std::vector<std::vector<float>>& a,
                                     const std::vector<std::vector<float>>& b,
                                     float tolerance,
                                     const std::string& message) {
    if (a.size() != b.size()) {
        throw std::runtime_error(message + ": Matrix row counts differ (" +
                                std::to_string(a.size()) + " vs " +
                                std::to_string(b.size()) + ")");
    }

    for (size_t i = 0; i < a.size(); ++i) {
        assertVectorsEqual(a[i], b[i], tolerance, message + " at row " + std::to_string(i));
    }
}

void TestHelpers::assertVectorsNear(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   float tolerance,
                                   const std::string& message) {
    assertVectorsEqual(a, b, tolerance, message);
}

void TestHelpers::assertMatricesNear(const std::vector<std::vector<float>>& a,
                                    const std::vector<std::vector<float>>& b,
                                    float tolerance,
                                    const std::string& message) {
    assertMatricesEqual(a, b, tolerance, message);
}

void TestHelpers::randomizeVector(std::vector<float>& vec, float minVal, float maxVal) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(minVal, maxVal);

    for (auto& val : vec) {
        val = static_cast<float>(dis(gen));
    }
}

void TestHelpers::randomizeMatrix(std::vector<std::vector<float>>& matrix, float minVal, float maxVal) {
    for (auto& row : matrix) {
        randomizeVector(row, minVal, maxVal);
    }
}

size_t TestHelpers::getCurrentMemoryUsage() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
#else
    // Simple implementation for Unix-like systems
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        size_t size;
        statm >> size;
        return size * getpagesize();
    }
    return 0;
#endif
}

void TestHelpers::logMemoryUsage(const std::string& label) {
    size_t memory = getCurrentMemoryUsage();
    std::cout << "Memory usage";
    if (!label.empty()) {
        std::cout << " (" << label << ")";
    }
    std::cout << ": " << (memory / 1024.0 / 1024.0) << " MB" << std::endl;
}

std::vector<std::string> TestHelpers::splitCSVLine(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;

    while (std::getline(ss, field, delimiter)) {
        // Remove quotes if present
        if (!field.empty() && field.front() == '"' && field.back() == '"') {
            field = field.substr(1, field.size() - 2);
        }
        fields.push_back(field);
    }

    return fields;
}

std::string TestHelpers::joinCSVLine(const std::vector<std::string>& fields, char delimiter) {
    std::stringstream ss;
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i > 0) ss << delimiter;
        ss << fields[i];
    }
    return ss.str();
}
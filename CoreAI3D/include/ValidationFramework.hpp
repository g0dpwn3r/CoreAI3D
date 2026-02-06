#ifndef VALIDATION_FRAMEWORK_HPP
#define VALIDATION_FRAMEWORK_HPP

#include "CoreAI3DCommon.hpp"
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <functional>
#include <chrono>
#include <thread>
#include <future>
#include "Core.hpp"
#include "Train.hpp"

// Validation result structure
struct ValidationResult {
    bool isValid;
    std::string errorMessage;
    std::string warningMessage;
    std::map<std::string, float> statistics;
    std::vector<std::string> validationChecks;
    float qualityScore; // 0.0 to 1.0
};

// Dataset validation configuration
struct ValidationConfig {
    bool checkDataTypes = true;
    bool checkMissingValues = true;
    bool checkOutliers = true;
    bool checkCorrelation = true;
    bool checkDistribution = true;
    bool checkFeatureScaling = true;
    bool checkSampleSize = true;
    bool checkFeatureVariance = true;
    bool checkDataIntegrity = true;
    
    float outlierThreshold = 3.0f; // Standard deviations
    float minVarianceThreshold = 1e-6f;
    int minSampleSize = 10;
    float maxMissingValuePercentage = 0.1f; // 10%
    float minCorrelationThreshold = 0.01f;
    int maxFeatures = 1000;
};

// Performance metrics structure
struct PerformanceMetrics {
    float accuracy;
    float precision;
    float recall;
    float f1_score;
    float rmse;
    float mse; // Added missing field
    float mae;
    float r2_score;
    float auc_roc;
    double trainingTime;
    double inferenceTime;
    size_t memoryUsage;
    int parameters;
    bool isClassification; // Flag to indicate if these are classification metrics
};

// Benchmark result comparing with baseline models
struct BenchmarkResult {
    std::string modelName;
    PerformanceMetrics metrics;
    std::vector<float> predictions;
    std::vector<float> targets;
    std::map<std::string, float> additionalMetrics;
    std::chrono::microseconds totalTime;
    std::chrono::microseconds trainingTime;
    std::chrono::microseconds inferenceTime;
};

class ValidationFramework {
private:
    std::vector<std::vector<float>> validationData;
    std::vector<std::vector<float>> validationTargets;
    ValidationConfig config;
    std::mt19937 randomGen;
    bool verbose;

public:
    ValidationFramework(bool verbose_ = false);
    
    // Dataset validation methods
    ValidationResult validateDataset(const std::string& filename, bool hasHeader = true);
    ValidationResult validateDataMatrix(const std::vector<std::vector<float>>& data);
    ValidationResult validateDatasetSplit(const std::vector<std::vector<float>>& trainData,
                                         const std::vector<std::vector<float>>& testData,
                                         const std::vector<std::vector<float>>& trainTargets,
                                         const std::vector<std::vector<float>>& testTargets);
    
    // Data quality checks
    bool checkDataTypes(const std::vector<std::vector<float>>& data);
    bool checkMissingValues(const std::vector<std::vector<float>>& data, float threshold = 0.1f);
    bool checkOutliers(const std::vector<std::vector<float>>& data, float threshold = 3.0f);
    bool checkCorrelation(const std::vector<std::vector<float>>& data, float minCorrelation = 0.01f);
    bool checkDistribution(const std::vector<std::vector<float>>& data);
    bool checkFeatureScaling(const std::vector<std::vector<float>>& data);
    bool checkSampleSize(const std::vector<std::vector<float>>& data, int minSize = 10);
    bool checkFeatureVariance(const std::vector<std::vector<float>>& data, float minVariance = 1e-6f);
    bool checkDataIntegrity(const std::vector<std::vector<float>>& data);
    
    // Statistical analysis methods
    std::map<std::string, float> computeDatasetStatistics(const std::vector<std::vector<float>>& data);
    std::vector<float> computeFeatureStatistics(const std::vector<float>& feature);
    std::vector<std::vector<float>> computeCorrelationMatrix(const std::vector<std::vector<float>>& data);
    std::pair<std::vector<float>, std::vector<float>> detectOutliers(const std::vector<float>& data, float threshold = 3.0f);
    
    // Data preprocessing for validation
    std::vector<std::vector<float>> normalizeData(const std::vector<std::vector<float>>& data, 
                                                 std::vector<float>* means = nullptr, 
                                                 std::vector<float>* stds = nullptr);
    std::vector<std::vector<float>> standardizeData(const std::vector<std::vector<float>>& data,
                                                   std::vector<float>* means = nullptr,
                                                   std::vector<float>* stds = nullptr);
    
    // Performance metrics calculation
    PerformanceMetrics calculateMetrics(const std::vector<std::vector<float>>& predictions, 
                                       const std::vector<std::vector<float>>& targets);
    void splitData(const std::vector<std::vector<float>>& data,
                    const std::vector<std::vector<float>>& targets,
                    float trainRatio,
                    std::vector<std::vector<float>>& trainData,
                    std::vector<std::vector<float>>& testData,
                    std::vector<std::vector<float>>& trainTargets,
                    std::vector<std::vector<float>>& testTargets);
    
    // Standard dataset loaders
    bool loadIrisDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets);
    bool loadBostonHousingDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets);
    bool loadWineDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets);
    bool loadDiabetesDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets);
    bool loadDigitsDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets);
    
    // Cross-validation methods
    std::vector<PerformanceMetrics> kFoldCrossValidation(std::unique_ptr<Training> trainer,
                                                        const std::vector<std::vector<float>>& data,
                                                        const std::vector<std::vector<float>>& targets,
                                                        int k = 5, double learningRate = 0.01,
                                                        int epochs = 100);
    
    // Performance evaluation methods
    PerformanceMetrics evaluateModel(std::unique_ptr<Training> trainer,
                                    const std::vector<std::vector<float>>& testData,
                                    const std::vector<std::vector<float>>& testTargets);

    // Problem type detection
    bool isClassificationProblem(const std::vector<std::vector<float>>& targets);
    
    // Memory usage monitoring
    size_t getCurrentMemoryUsage() const;
    int getNumClasses(const std::vector<std::vector<float>>& targets);
    
    // Benchmarking methods
    std::vector<BenchmarkResult> benchmarkAgainstBaselines(const std::vector<std::vector<float>>& data,
                                                          const std::vector<std::vector<float>>& targets,
                                                          float testSize = 0.2f);

    // Scikit-learn benchmarking integration
    std::vector<BenchmarkResult> benchmarkAgainstScikitLearn(const std::vector<std::vector<float>>& data,
                                                           const std::vector<std::vector<float>>& targets,
                                                           float testSize = 0.2f,
                                                           const std::string& pythonPath = "python");
    
    // CoreAI3D benchmark implementation
    BenchmarkResult runCoreAI3D(const std::vector<std::vector<float>>& trainData,
                               const std::vector<std::vector<float>>& trainTargets,
                               const std::vector<std::vector<float>>& testData,
                               const std::vector<std::vector<float>>& testTargets);

    // Baseline model implementations (simplified for C++)
    BenchmarkResult runLinearRegression(const std::vector<std::vector<float>>& trainData,
                                        const std::vector<std::vector<float>>& trainTargets,
                                        const std::vector<std::vector<float>>& testData,
                                        const std::vector<std::vector<float>>& testTargets);
    
    BenchmarkResult runRandomForest(const std::vector<std::vector<float>>& trainData,
                                   const std::vector<std::vector<float>>& trainTargets,
                                   const std::vector<std::vector<float>>& testData,
                                   const std::vector<std::vector<float>>& testTargets,
                                   int numTrees = 10, int maxDepth = 5);
    
    BenchmarkResult runKNN(const std::vector<std::vector<float>>& trainData,
                          const std::vector<std::vector<float>>& trainTargets,
                          const std::vector<std::vector<float>>& testData,
                          const std::vector<std::vector<float>>& testTargets,
                          int k = 5);
    
    BenchmarkResult runLogisticRegression(const std::vector<std::vector<float>>& trainData,
                                         const std::vector<std::vector<float>>& trainTargets,
                                         const std::vector<std::vector<float>>& testData,
                                         const std::vector<std::vector<float>>& testTargets);
    
    // Utility methods
    void setValidationConfig(const ValidationConfig& config);
    ValidationConfig getValidationConfig() const { return config; }
    
    // Report generation
    std::string generateValidationReport(const ValidationResult& result);
    std::string generateBenchmarkReport(const std::vector<BenchmarkResult>& results);
    void saveReportToFile(const std::string& filename, const std::string& report);
    
    // Performance metrics calculation
    static float calculateAccuracy(const std::vector<float>& predictions, const std::vector<float>& targets);
    static float calculatePrecision(const std::vector<float>& predictions, const std::vector<float>& targets, float threshold = 0.5f);
    static float calculateRecall(const std::vector<float>& predictions, const std::vector<float>& targets, float threshold = 0.5f);
    static float calculateF1Score(const std::vector<float>& predictions, const std::vector<float>& targets, float threshold = 0.5f);
    static float calculateRMSE(const std::vector<float>& predictions, const std::vector<float>& targets);
    static float calculateMAE(const std::vector<float>& predictions, const std::vector<float>& targets);
    static float calculateR2Score(const std::vector<float>& predictions, const std::vector<float>& targets);
    
private:
    // Internal helper methods
    std::vector<float> linspace(float start, float end, int num);
    std::vector<float> computeMean(const std::vector<std::vector<float>>& data);
    std::vector<float> computeStd(const std::vector<std::vector<float>>& data);
    float computePearsonCorrelation(const std::vector<float>& x, const std::vector<float>& y);

    // Simple matrix operations
    std::vector<std::vector<float>> transposeMatrix(const std::vector<std::vector<float>>& matrix);
    std::vector<std::vector<float>> multiplyMatrices(const std::vector<std::vector<float>>& A,
                                                    const std::vector<std::vector<float>>& B);
    std::vector<std::vector<float>> invertMatrix(const std::vector<std::vector<float>>& matrix);

    // JSON parsing helpers for scikit-learn integration
    std::string extractJsonString(const std::string& json, const std::string& key);
    float extractJsonFloat(const std::string& json, const std::string& key);
    int extractJsonInt(const std::string& json, const std::string& key);

    // Random utility
    float randomNormal(float mean = 0.0f, float std = 1.0f);
};

#endif // VALIDATION_FRAMEWORK_HPP
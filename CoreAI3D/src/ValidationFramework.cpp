#include "ValidationFramework.hpp"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <random>

ValidationFramework::ValidationFramework(bool verbose_) : verbose(verbose_) {
    std::random_device rd;
    randomGen.seed(rd());
}

ValidationResult ValidationFramework::validateDataset(const std::string& filename, bool hasHeader) {
    ValidationResult result;
    result.isValid = true;
    result.qualityScore = 1.0f;
    
    if (verbose) {
        std::cout << "Validating dataset: " << filename << std::endl;
    }
    
    // Load and validate the dataset
    std::ifstream file(filename);
    if (!file.is_open()) {
        result.isValid = false;
        result.errorMessage = "Cannot open file: " + filename;
        result.qualityScore = 0.0f;
        return result;
    }
    
    std::vector<std::vector<float>> data;
    std::string line;
    int lineCount = 0;
    
    // Skip header if present
    if (hasHeader) {
        std::getline(file, line);
        lineCount++;
    }
    
    // Parse CSV data
    while (std::getline(file, line)) {
        lineCount++;
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            try {
                float value = std::stof(cell);
                row.push_back(value);
            } catch (const std::exception& e) {
                result.isValid = false;
                result.errorMessage = "Invalid data format at line " + std::to_string(lineCount);
                result.qualityScore = 0.0f;
                return result;
            }
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    file.close();
    
    if (data.empty()) {
        result.isValid = false;
        result.errorMessage = "Dataset is empty";
        result.qualityScore = 0.0f;
        return result;
    }
    
    // Validate the parsed data
    return validateDataMatrix(data);
}

ValidationResult ValidationFramework::validateDataMatrix(const std::vector<std::vector<float>>& data) {
    ValidationResult result;
    result.isValid = true;
    result.qualityScore = 1.0f;
    result.validationChecks.clear();
    
    if (verbose) {
        std::cout << "Validating data matrix with " << data.size() << " samples and " 
                  << (data.empty() ? 0 : data[0].size()) << " features" << std::endl;
    }
    
    // Check sample size
    if (checkSampleSize(data, config.minSampleSize)) {
        result.validationChecks.push_back("Sample size: PASS");
    } else {
        result.validationChecks.push_back("Sample size: FAIL");
        result.isValid = false;
        result.errorMessage = "Insufficient sample size";
        result.qualityScore = 0.0f;
    }
    
    // Check data integrity
    if (checkDataIntegrity(data)) {
        result.validationChecks.push_back("Data integrity: PASS");
    } else {
        result.validationChecks.push_back("Data integrity: FAIL");
        result.isValid = false;
        result.errorMessage = "Data integrity check failed";
        result.qualityScore *= 0.5f;
    }
    
    // Check missing values
    if (checkMissingValues(data, config.maxMissingValuePercentage)) {
        result.validationChecks.push_back("Missing values: PASS");
    } else {
        result.validationChecks.push_back("Missing values: FAIL");
        result.warningMessage = "Missing values detected";
        result.qualityScore *= 0.9f;
    }
    
    // Check outliers
    if (checkOutliers(data, config.outlierThreshold)) {
        result.validationChecks.push_back("Outliers: PASS");
    } else {
        result.validationChecks.push_back("Outliers: WARNING");
        result.warningMessage = "Outliers detected";
        result.qualityScore *= 0.95f;
    }
    
    // Check correlation
    if (checkCorrelation(data, config.minCorrelationThreshold)) {
        result.validationChecks.push_back("Feature correlation: PASS");
    } else {
        result.validationChecks.push_back("Feature correlation: WARNING");
        result.warningMessage = "Low feature correlation detected";
        result.qualityScore *= 0.98f;
    }
    
    // Check feature variance
    if (checkFeatureVariance(data, config.minVarianceThreshold)) {
        result.validationChecks.push_back("Feature variance: PASS");
    } else {
        result.validationChecks.push_back("Feature variance: WARNING");
        result.warningMessage = "Low variance features detected";
        result.qualityScore *= 0.97f;
    }
    
    // Check feature scaling
    if (checkFeatureScaling(data)) {
        result.validationChecks.push_back("Feature scaling: PASS");
    } else {
        result.validationChecks.push_back("Feature scaling: WARNING");
        result.warningMessage = "Features may need scaling";
        result.qualityScore *= 0.95f;
    }
    
    // Compute dataset statistics
    result.statistics = computeDatasetStatistics(data);
    
    if (verbose) {
        std::cout << "Dataset validation complete. Quality score: " << result.qualityScore << std::endl;
    }
    
    return result;
}

ValidationResult ValidationFramework::validateDatasetSplit(const std::vector<std::vector<float>>& trainData,
                                                         const std::vector<std::vector<float>>& testData,
                                                         const std::vector<std::vector<float>>& trainTargets,
                                                         const std::vector<std::vector<float>>& testTargets) {
    ValidationResult result;
    result.isValid = true;
    result.qualityScore = 1.0f;
    result.validationChecks.clear();
    
    // Check if train and test data sizes are reasonable
    if (trainData.size() < config.minSampleSize) {
        result.isValid = false;
        result.errorMessage = "Training set too small";
        result.qualityScore = 0.0f;
        return result;
    }
    
    if (testData.size() < config.minSampleSize / 10) { // At least 10% of minimum
        result.warningMessage = "Test set may be too small for reliable evaluation";
        result.qualityScore *= 0.9f;
    }
    
    // Check if splits have consistent feature dimensions
    if (!trainData.empty() && !testData.empty() && 
        trainData[0].size() != testData[0].size()) {
        result.isValid = false;
        result.errorMessage = "Inconsistent feature dimensions between train and test sets";
        result.qualityScore = 0.0f;
        return result;
    }
    
    // Validate both datasets
    ValidationResult trainResult = validateDataMatrix(trainData);
    ValidationResult testResult = validateDataMatrix(testData);
    
    result.validationChecks.insert(result.validationChecks.end(), 
                                  trainResult.validationChecks.begin(),
                                  trainResult.validationChecks.end());
    result.validationChecks.insert(result.validationChecks.end(),
                                  testResult.validationChecks.begin(),
                                  testResult.validationChecks.end());
    
    // Check target consistency
    if (trainTargets.size() != trainData.size()) {
        result.isValid = false;
        result.errorMessage = "Training targets size mismatch";
        result.qualityScore = 0.0f;
    }
    
    if (testTargets.size() != testData.size()) {
        result.isValid = false;
        result.errorMessage = "Test targets size mismatch";
        result.qualityScore = 0.0f;
    }
    
    return result;
}

bool ValidationFramework::checkSampleSize(const std::vector<std::vector<float>>& data, int minSize) {
    return data.size() >= minSize;
}

bool ValidationFramework::checkDataIntegrity(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return false;
    
    size_t expectedSize = data[0].size();
    
    for (const auto& row : data) {
        if (row.size() != expectedSize) {
            return false;
        }
        
        // Check for NaN or infinite values
        for (float value : row) {
            if (std::isnan(value) || std::isinf(value)) {
                return false;
            }
        }
    }
    
    return true;
}

bool ValidationFramework::checkMissingValues(const std::vector<std::vector<float>>& data, float threshold) {
    size_t totalValues = 0;
    size_t missingValues = 0;
    
    for (const auto& row : data) {
        for (float value : row) {
            totalValues++;
            if (std::isnan(value)) {
                missingValues++;
            }
        }
    }
    
    if (totalValues == 0) return false;
    
    float missingPercentage = static_cast<float>(missingValues) / totalValues;
    return missingPercentage <= threshold;
}

bool ValidationFramework::checkOutliers(const std::vector<std::vector<float>>& data, float threshold) {
    size_t outliers = 0;
    size_t totalValues = 0;
    
    for (const auto& row : data) {
        for (float value : row) {
            if (std::abs(value) > threshold * 10) { // Simple outlier check
                outliers++;
            }
            totalValues++;
        }
    }
    
    if (totalValues == 0) return true;
    
    float outlierPercentage = static_cast<float>(outliers) / totalValues;
    return outlierPercentage <= 0.05f; // Allow up to 5% outliers
}

bool ValidationFramework::checkCorrelation(const std::vector<std::vector<float>>& data, float minCorrelation) {
    if (data.empty() || data[0].size() < 2) return true;
    
    auto correlationMatrix = computeCorrelationMatrix(data);
    
    for (size_t i = 0; i < correlationMatrix.size(); i++) {
        for (size_t j = i + 1; j < correlationMatrix[i].size(); j++) {
            if (std::abs(correlationMatrix[i][j]) < minCorrelation) {
                return false;
            }
        }
    }
    
    return true;
}

bool ValidationFramework::checkDistribution(const std::vector<std::vector<float>>& data) {
    // Simple check for reasonable distribution
    // Could be enhanced with statistical tests
    return true;
}

bool ValidationFramework::checkFeatureScaling(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return true;
    
    auto means = computeMean(data);
    auto stds = computeStd(data);
    
    for (size_t i = 0; i < means.size(); i++) {
        // Check if features are roughly normalized (mean ~ 0, std ~ 1)
        if (std::abs(means[i]) > 2.0f || stds[i] < 0.1f || stds[i] > 10.0f) {
            return false;
        }
    }
    
    return true;
}

bool ValidationFramework::checkFeatureVariance(const std::vector<std::vector<float>>& data, float minVariance) {
    if (data.empty()) return true;
    
    auto means = computeMean(data);
    
    for (size_t j = 0; j < means.size(); j++) {
        float variance = 0.0f;
        for (const auto& row : data) {
            variance += std::pow(row[j] - means[j], 2);
        }
        variance /= data.size();
        
        if (variance < minVariance) {
            return false;
        }
    }
    
    return true;
}

std::map<std::string, float> ValidationFramework::computeDatasetStatistics(const std::vector<std::vector<float>>& data) {
    std::map<std::string, float> stats;
    
    if (data.empty()) return stats;
    
    size_t numSamples = data.size();
    size_t numFeatures = data[0].size();
    
    stats["num_samples"] = static_cast<float>(numSamples);
    stats["num_features"] = static_cast<float>(numFeatures);
    
    // Compute global statistics
    float globalMin = std::numeric_limits<float>::max();
    float globalMax = std::numeric_limits<float>::lowest();
    float globalSum = 0.0f;
    size_t totalValues = 0;
    
    for (const auto& row : data) {
        for (float value : row) {
            globalMin = std::min(globalMin, value);
            globalMax = std::max(globalMax, value);
            globalSum += value;
            totalValues++;
        }
    }
    
    float globalMean = globalSum / totalValues;
    
    stats["global_min"] = globalMin;
    stats["global_max"] = globalMax;
    stats["global_mean"] = globalMean;
    stats["global_range"] = globalMax - globalMin;
    
    // Compute feature-level statistics
    auto featureMeans = computeMean(data);
    auto featureStds = computeStd(data);
    
    float avgMean = 0.0f, avgStd = 0.0f;
    for (float mean : featureMeans) avgMean += mean;
    for (float std : featureStds) avgStd += std;
    avgMean /= featureMeans.size();
    avgStd /= featureStds.size();
    
    stats["avg_feature_mean"] = avgMean;
    stats["avg_feature_std"] = avgStd;
    
    return stats;
}

std::vector<float> ValidationFramework::computeFeatureStatistics(const std::vector<float>& feature) {
    if (feature.empty()) return {};
    
    float mean = std::accumulate(feature.begin(), feature.end(), 0.0f) / feature.size();
    
    float variance = 0.0f;
    for (float value : feature) {
        variance += std::pow(value - mean, 2);
    }
    variance /= feature.size();
    
    float std = std::sqrt(variance);
    
    float minVal = *std::min_element(feature.begin(), feature.end());
    float maxVal = *std::max_element(feature.begin(), feature.end());
    
    return {mean, std, minVal, maxVal};
}

std::vector<std::vector<float>> ValidationFramework::computeCorrelationMatrix(const std::vector<std::vector<float>>& data) {
    if (data.empty() || data[0].empty()) return {};
    
    size_t numFeatures = data[0].size();
    std::vector<std::vector<float>> correlationMatrix(numFeatures, std::vector<float>(numFeatures, 0.0f));
    
    // Extract features as separate vectors
    std::vector<std::vector<float>> features(numFeatures);
    for (size_t j = 0; j < numFeatures; j++) {
        features[j].resize(data.size());
        for (size_t i = 0; i < data.size(); i++) {
            features[j][i] = data[i][j];
        }
    }
    
    // Compute pairwise correlations
    for (size_t i = 0; i < numFeatures; i++) {
        correlationMatrix[i][i] = 1.0f;
        for (size_t j = i + 1; j < numFeatures; j++) {
            float corr = computePearsonCorrelation(features[i], features[j]);
            correlationMatrix[i][j] = corr;
            correlationMatrix[j][i] = corr;
        }
    }
    
    return correlationMatrix;
}

float ValidationFramework::computePearsonCorrelation(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) return 0.0f;
    
    float meanX = std::accumulate(x.begin(), x.end(), 0.0f) / x.size();
    float meanY = std::accumulate(y.begin(), y.end(), 0.0f) / y.size();
    
    float numerator = 0.0f;
    float sumXSquared = 0.0f;
    float sumYSquared = 0.0f;
    
    for (size_t i = 0; i < x.size(); i++) {
        float diffX = x[i] - meanX;
        float diffY = y[i] - meanY;
        numerator += diffX * diffY;
        sumXSquared += diffX * diffX;
        sumYSquared += diffY * diffY;
    }
    
    float denominator = std::sqrt(sumXSquared * sumYSquared);
    return denominator > 0 ? numerator / denominator : 0.0f;
}

std::vector<std::vector<float>> ValidationFramework::normalizeData(const std::vector<std::vector<float>>& data,
                                                                  std::vector<float>* means,
                                                                  std::vector<float>* stds) {
    auto computedMeans = computeMean(data);
    auto computedStds = computeStd(data);
    
    if (means) {
        *means = computedMeans;
    }
    if (stds) {
        *stds = computedStds;
    }
    
    std::vector<std::vector<float>> normalizedData = data;
    
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[i].size(); j++) {
            if (computedStds[j] > 1e-6f) { // Avoid division by zero
                normalizedData[i][j] = (data[i][j] - computedMeans[j]) / computedStds[j];
            } else {
                normalizedData[i][j] = 0.0f;
            }
        }
    }
    
    return normalizedData;
}

std::vector<std::vector<float>> ValidationFramework::standardizeData(const std::vector<std::vector<float>>& data,
                                                                    std::vector<float>* means,
                                                                    std::vector<float>* stds) {
    // For standardization, we use min-max scaling
    std::vector<float> mins, maxs;
    mins.resize(data[0].size());
    maxs.resize(data[0].size());
    
    // Find min and max for each feature
    for (size_t j = 0; j < data[0].size(); j++) {
        mins[j] = std::numeric_limits<float>::max();
        maxs[j] = std::numeric_limits<float>::lowest();
        
        for (const auto& row : data) {
            mins[j] = std::min(mins[j], row[j]);
            maxs[j] = std::max(maxs[j], row[j]);
        }
    }
    
    std::vector<float> computedMeans;
    std::vector<float> computedStds;
    for (size_t j = 0; j < data[0].size(); j++) {
        computedMeans.push_back((mins[j] + maxs[j]) / 2.0f);
        computedStds.push_back((maxs[j] - mins[j]) / 2.0f);
    }
    
    if (means) {
        *means = computedMeans;
    }
    if (stds) {
        *stds = computedStds;
    }
    
    // Apply standardization
    std::vector<std::vector<float>> standardizedData = data;
    
    for (size_t i = 0; i < data.size(); i++) {
        for (size_t j = 0; j < data[i].size(); j++) {
            if (computedStds[j] > 1e-6f) { // Avoid division by zero
                standardizedData[i][j] = (data[i][j] - mins[j]) / (maxs[j] - mins[j]);
            } else {
                standardizedData[i][j] = 0.5f; // Middle value for zero variance features
            }
        }
    }
    
    return standardizedData;
}

void ValidationFramework::splitData(const std::vector<std::vector<float>>& data,
                                    const std::vector<std::vector<float>>& targets,
                                    float trainRatio,
                                    std::vector<std::vector<float>>& trainData,
                                    std::vector<std::vector<float>>& testData,
                                    std::vector<std::vector<float>>& trainTargets,
                                    std::vector<std::vector<float>>& testTargets) {
    size_t totalSize = data.size();
    size_t trainSize = static_cast<size_t>(totalSize * trainRatio);

    // Shuffle indices
    std::vector<size_t> indices(totalSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), randomGen);

    // Split data
    for (size_t i = 0; i < totalSize; i++) {
        if (i < trainSize) {
            trainData.push_back(data[indices[i]]);
            trainTargets.push_back(targets[indices[i]]);
        } else {
            testData.push_back(data[indices[i]]);
            testTargets.push_back(targets[indices[i]]);
        }
    }
}

bool ValidationFramework::loadIrisDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets) {
    // Iris dataset implementation
    // For now, return false as we need to implement actual dataset loading
    return false;
}

bool ValidationFramework::loadBostonHousingDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets) {
    // Boston Housing dataset implementation
    return false;
}

bool ValidationFramework::loadWineDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets) {
    // Wine dataset implementation
    return false;
}

bool ValidationFramework::loadDiabetesDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets) {
    // Diabetes dataset implementation
    return false;
}

bool ValidationFramework::loadDigitsDataset(std::vector<std::vector<float>>& data, std::vector<float>& targets) {
    // Digits dataset implementation
    return false;
}

std::vector<float> ValidationFramework::linspace(float start, float end, int num) {
    std::vector<float> result(num);
    if (num == 1) {
        result[0] = start;
        return result;
    }
    
    float step = (end - start) / (num - 1);
    for (int i = 0; i < num; i++) {
        result[i] = start + i * step;
    }
    return result;
}

std::vector<float> ValidationFramework::computeMean(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return {};
    
    size_t numFeatures = data[0].size();
    std::vector<float> means(numFeatures, 0.0f);
    
    for (const auto& row : data) {
        for (size_t j = 0; j < numFeatures; j++) {
            means[j] += row[j];
        }
    }
    
    for (auto& mean : means) {
        mean /= data.size();
    }
    
    return means;
}

std::vector<float> ValidationFramework::computeStd(const std::vector<std::vector<float>>& data) {
    if (data.empty()) return {};
    
    auto means = computeMean(data);
    size_t numFeatures = means.size();
    std::vector<float> stds(numFeatures, 0.0f);
    
    for (const auto& row : data) {
        for (size_t j = 0; j < numFeatures; j++) {
            stds[j] += std::pow(row[j] - means[j], 2);
        }
    }
    
    for (auto& std : stds) {
        std /= data.size();
        std = std::sqrt(std);
    }
    
    return stds;
}

float ValidationFramework::randomNormal(float mean, float std) {
    std::normal_distribution<float> distribution(mean, std);
    return distribution(randomGen);
}

void ValidationFramework::setValidationConfig(const ValidationConfig& config) {
    this->config = config;
}

std::string ValidationFramework::generateValidationReport(const ValidationResult& result) {
    std::stringstream report;
    
    report << "=== DATASET VALIDATION REPORT ===" << std::endl;
    report << "Timestamp: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
    report << "Validation Status: " << (result.isValid ? "PASSED" : "FAILED") << std::endl;
    report << "Quality Score: " << std::fixed << std::setprecision(3) << result.qualityScore << std::endl;
    report << std::endl;
    
    if (!result.errorMessage.empty()) {
        report << "ERROR: " << result.errorMessage << std::endl;
        report << std::endl;
    }
    
    if (!result.warningMessage.empty()) {
        report << "WARNING: " << result.warningMessage << std::endl;
        report << std::endl;
    }
    
    report << "VALIDATION CHECKS:" << std::endl;
    for (const auto& check : result.validationChecks) {
        report << "  - " << check << std::endl;
    }
    report << std::endl;
    
    if (!result.statistics.empty()) {
        report << "DATASET STATISTICS:" << std::endl;
        for (const auto& stat : result.statistics) {
            report << "  " << stat.first << ": " << std::fixed << std::setprecision(4) << stat.second << std::endl;
        }
    }
    
    return report.str();
}

std::string ValidationFramework::generateBenchmarkReport(const std::vector<BenchmarkResult>& results) {
    std::stringstream report;
    
    report << "=== PERFORMANCE BENCHMARK REPORT ===" << std::endl;
    report << "Timestamp: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
    report << "Number of Models: " << results.size() << std::endl;
    report << std::endl;
    
    for (const auto& result : results) {
        report << "MODEL: " << result.modelName << std::endl;
        report << "===============================" << std::endl;
        
        const auto& metrics = result.metrics;
        report << "Accuracy: " << std::fixed << std::setprecision(4) << metrics.accuracy << std::endl;
        report << "Precision: " << std::fixed << std::setprecision(4) << metrics.precision << std::endl;
        report << "Recall: " << std::fixed << std::setprecision(4) << metrics.recall << std::endl;
        report << "F1-Score: " << std::fixed << std::setprecision(4) << metrics.f1_score << std::endl;
        report << "RMSE: " << std::fixed << std::setprecision(4) << metrics.rmse << std::endl;
        report << "MAE: " << std::fixed << std::setprecision(4) << metrics.mae << std::endl;
        report << "R² Score: " << std::fixed << std::setprecision(4) << metrics.r2_score << std::endl;
        report << "Training Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(result.trainingTime).count() << " ms" << std::endl;
        report << "Inference Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(result.inferenceTime).count() << " ms" << std::endl;
        report << "Memory Usage: " << metrics.memoryUsage << " bytes" << std::endl;
        report << "Parameters: " << metrics.parameters << std::endl;
        report << std::endl;
    }
    
    return report.str();
}

void ValidationFramework::saveReportToFile(const std::string& filename, const std::string& report) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << report;
        file.close();
        if (verbose) {
            std::cout << "Report saved to: " << filename << std::endl;
        }
    } else {
        std::cerr << "Error: Cannot save report to " << filename << std::endl;
    }
}

// Problem type detection
bool ValidationFramework::isClassificationProblem(const std::vector<std::vector<float>>& targets) {
    if (targets.empty()) return false;

    // Check if targets are discrete (integer values)
    std::set<float> uniqueValues;
    for (const auto& target_row : targets) {
        if (!target_row.empty()) {
            uniqueValues.insert(target_row[0]);
        }
    }

    // If we have few unique values (≤ 10) and they are integers, it's likely classification
    return uniqueValues.size() <= 10 && uniqueValues.size() > 1 &&
           std::all_of(uniqueValues.begin(), uniqueValues.end(),
                      [](float val) { return val == std::floor(val); });
}

int ValidationFramework::getNumClasses(const std::vector<std::vector<float>>& targets) {
    std::set<float> uniqueValues;
    for (const auto& target_row : targets) {
        if (!target_row.empty()) {
            uniqueValues.insert(target_row[0]);
        }
    }
    return uniqueValues.size();
}

// Performance metrics calculation functions
float ValidationFramework::calculateAccuracy(const std::vector<float>& predictions, const std::vector<float>& targets) {
    if (predictions.size() != targets.size() || predictions.empty()) return 0.0f;

    size_t correct = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] == targets[i]) {
            correct++;
        }
    }

    return static_cast<float>(correct) / predictions.size();
}

float ValidationFramework::calculatePrecision(const std::vector<float>& predictions, const std::vector<float>& targets, float threshold) {
    // Simplified precision calculation for binary classification
    if (predictions.size() != targets.size() || predictions.empty()) return 0.0f;
    
    size_t truePositives = 0;
    size_t falsePositives = 0;
    
    for (size_t i = 0; i < predictions.size(); i++) {
        bool pred = predictions[i] > threshold;
        bool actual = targets[i] > threshold;
        
        if (pred && actual) truePositives++;
        if (pred && !actual) falsePositives++;
    }
    
    return (truePositives + falsePositives) > 0 ? 
           static_cast<float>(truePositives) / (truePositives + falsePositives) : 0.0f;
}

float ValidationFramework::calculateRecall(const std::vector<float>& predictions, const std::vector<float>& targets, float threshold) {
    // Simplified recall calculation for binary classification
    if (predictions.size() != targets.size() || predictions.empty()) return 0.0f;
    
    size_t truePositives = 0;
    size_t falseNegatives = 0;
    
    for (size_t i = 0; i < predictions.size(); i++) {
        bool pred = predictions[i] > threshold;
        bool actual = targets[i] > threshold;
        
        if (pred && actual) truePositives++;
        if (!pred && actual) falseNegatives++;
    }
    
    return (truePositives + falseNegatives) > 0 ? 
           static_cast<float>(truePositives) / (truePositives + falseNegatives) : 0.0f;
}

float ValidationFramework::calculateF1Score(const std::vector<float>& predictions, const std::vector<float>& targets, float threshold) {
    float precision = calculatePrecision(predictions, targets, threshold);
    float recall = calculateRecall(predictions, targets, threshold);
    
    return (precision + recall) > 0 ? 
           2.0f * precision * recall / (precision + recall) : 0.0f;
}

float ValidationFramework::calculateRMSE(const std::vector<float>& predictions, const std::vector<float>& targets) {
    if (predictions.size() != targets.size() || predictions.empty()) return 0.0f;
    
    float sumSquaredError = 0.0f;
    for (size_t i = 0; i < predictions.size(); i++) {
        float error = predictions[i] - targets[i];
        sumSquaredError += error * error;
    }
    
    return std::sqrt(sumSquaredError / predictions.size());
}

float ValidationFramework::calculateMAE(const std::vector<float>& predictions, const std::vector<float>& targets) {
    if (predictions.size() != targets.size() || predictions.empty()) return 0.0f;
    
    float sumAbsoluteError = 0.0f;
    for (size_t i = 0; i < predictions.size(); i++) {
        sumAbsoluteError += std::abs(predictions[i] - targets[i]);
    }
    
    return sumAbsoluteError / predictions.size();
}

float ValidationFramework::calculateR2Score(const std::vector<float>& predictions, const std::vector<float>& targets) {
    if (predictions.size() != targets.size() || predictions.empty()) return 0.0f;

    // Calculate mean of targets
    float targetMean = std::accumulate(targets.begin(), targets.end(), 0.0f) / targets.size();

    // Calculate total sum of squares and residual sum of squares
    float totalSumSquares = 0.0f;
    float residualSumSquares = 0.0f;

    for (size_t i = 0; i < predictions.size(); i++) {
        float targetDiff = targets[i] - targetMean;
        float residual = targets[i] - predictions[i];

        totalSumSquares += targetDiff * targetDiff;
        residualSumSquares += residual * residual;
    }

    return totalSumSquares > 0 ? 1.0f - (residualSumSquares / totalSumSquares) : 0.0f;
}



// Scikit-learn benchmarking integration
std::vector<BenchmarkResult> ValidationFramework::benchmarkAgainstScikitLearn(
    const std::vector<std::vector<float>>& data,
    const std::vector<std::vector<float>>& targets,
    float testSize,
    const std::string& pythonPath) {

    std::vector<BenchmarkResult> results;

    if (verbose) {
        std::cout << "Running scikit-learn benchmark..." << std::endl;
    }

    try {
        // Create temporary CSV file for Python script
        std::string tempCsvFile = "temp_benchmark_data.csv";
        std::ofstream csvFile(tempCsvFile);

        // Write header
        csvFile << "feature_";
        for (size_t j = 0; j < data[0].size(); j++) {
            csvFile << j;
            if (j < data[0].size() - 1) csvFile << ",";
        }
        csvFile << ",target\n";

        // Write data
        for (size_t i = 0; i < data.size(); i++) {
            for (size_t j = 0; j < data[i].size(); j++) {
                csvFile << data[i][j];
                if (j < data[i].size() - 1) csvFile << ",";
            }
            csvFile << "," << targets[i][0] << "\n";
        }
        csvFile.close();

        // Create Python script to run benchmarking
        std::string pythonScript = R"(
import sys
import os
sys.path.append('python')
from benchmarking_scikit import benchmark_from_csv

try:
    results = benchmark_from_csv('temp_benchmark_data.csv', 'target', )" + std::to_string(testSize) + R"()
    print("BENCHMARK_RESULTS_START")
    import json
    for result in results:
        print(json.dumps({
            'model_name': result.model_name,
            'accuracy': result.metrics.accuracy,
            'precision': result.metrics.precision,
            'recall': result.metrics.recall,
            'f1_score': result.metrics.f1_score,
            'rmse': result.metrics.rmse,
            'mae': result.metrics.mae,
            'r2_score': result.metrics.r2_score,
            'auc_roc': result.metrics.auc_roc,
            'training_time': result.training_time,
            'inference_time': result.inference_time,
            'memory_usage': result.metrics.memory_usage,
            'parameters': result.metrics.parameters,
            'total_time': result.total_time,
            'predictions': result.predictions[:10],  # Limit for output
            'targets': result.targets[:10]
        }))
    print("BENCHMARK_RESULTS_END")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
)";

        // Write Python script to temporary file
        std::string scriptFile = "temp_benchmark_script.py";
        std::ofstream script(scriptFile);
        script << pythonScript;
        script.close();

        // Execute Python script
        std::string command = pythonPath + " " + scriptFile + " 2>&1";
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("Failed to execute Python script");
        }

        // Read output
        char buffer[4096];
        std::string output;
        bool inResults = false;

        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            std::string line = buffer;
            output += line;

            if (line.find("BENCHMARK_RESULTS_START") != std::string::npos) {
                inResults = true;
                continue;
            }
            if (line.find("BENCHMARK_RESULTS_END") != std::string::npos) {
                inResults = false;
                break;
            }

            if (inResults && !line.empty() && line.find("{") != std::string::npos) {
                try {
                    // Parse JSON result
                    auto jsonStart = line.find("{");
                    auto jsonEnd = line.rfind("}") + 1;
                    if (jsonStart != std::string::npos && jsonEnd != std::string::npos) {
                        std::string jsonStr = line.substr(jsonStart, jsonEnd - jsonStart);

                        // Simple JSON parsing (basic implementation)
                        BenchmarkResult result;
                        result.modelName = extractJsonString(jsonStr, "model_name");
                        result.metrics.accuracy = extractJsonFloat(jsonStr, "accuracy");
                        result.metrics.precision = extractJsonFloat(jsonStr, "precision");
                        result.metrics.recall = extractJsonFloat(jsonStr, "recall");
                        result.metrics.f1_score = extractJsonFloat(jsonStr, "f1_score");
                        result.metrics.rmse = extractJsonFloat(jsonStr, "rmse");
                        result.metrics.mae = extractJsonFloat(jsonStr, "mae");
                        result.metrics.r2_score = extractJsonFloat(jsonStr, "r2_score");
                        result.metrics.auc_roc = extractJsonFloat(jsonStr, "auc_roc");
                        result.metrics.trainingTime = extractJsonFloat(jsonStr, "training_time");
                        result.metrics.inferenceTime = extractJsonFloat(jsonStr, "inference_time");
                        result.metrics.memoryUsage = extractJsonInt(jsonStr, "memory_usage");
                        result.metrics.parameters = extractJsonInt(jsonStr, "parameters");
                        float totalTimeSec = extractJsonFloat(jsonStr, "total_time");
                        result.totalTime = std::chrono::microseconds(static_cast<long long>(totalTimeSec * 1000000));

                        results.push_back(result);
                    }
                } catch (const std::exception& e) {
                    if (verbose) {
                        std::cout << "Error parsing result: " << e.what() << std::endl;
                    }
                }
            }
        }

        int returnCode = pclose(pipe);

        // Clean up temporary files
        std::remove(tempCsvFile.c_str());
        std::remove(scriptFile.c_str());

        if (returnCode != 0) {
            if (verbose) {
                std::cout << "Python script execution failed with code: " << returnCode << std::endl;
                std::cout << "Output: " << output << std::endl;
            }
        }

    } catch (const std::exception& e) {
        if (verbose) {
            std::cout << "Scikit-learn benchmarking failed: " << e.what() << std::endl;
        }
    }

    if (verbose) {
        std::cout << "Scikit-learn benchmark completed. " << results.size() << " models evaluated." << std::endl;
    }

    return results;
}

// Helper functions for JSON parsing
std::string ValidationFramework::extractJsonString(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\": \"";
    size_t start = json.find(searchKey);
    if (start == std::string::npos) return "";

    start += searchKey.length();
    size_t end = json.find("\"", start);
    if (end == std::string::npos) return "";

    return json.substr(start, end - start);
}

float ValidationFramework::extractJsonFloat(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\": ";
    size_t start = json.find(searchKey);
    if (start == std::string::npos) return 0.0f;

    start += searchKey.length();
    size_t end = json.find(",", start);
    if (end == std::string::npos) end = json.find("}", start);
    if (end == std::string::npos) return 0.0f;

    std::string valueStr = json.substr(start, end - start);
    try {
        return std::stof(valueStr);
    } catch (...) {
        return 0.0f;
    }
}

int ValidationFramework::extractJsonInt(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\": ";
    size_t start = json.find(searchKey);
    if (start == std::string::npos) return 0;

    start += searchKey.length();
    size_t end = json.find(",", start);
    if (end == std::string::npos) end = json.find("}", start);
    if (end == std::string::npos) return 0;

    std::string valueStr = json.substr(start, end - start);
    try {
        return std::stoi(valueStr);
    } catch (...) {
        return 0;
    }
}

size_t ValidationFramework::getCurrentMemoryUsage() const {
    // This is a placeholder implementation
    // On Linux, you could read from /proc/self/status
    // For cross-platform compatibility, we return a dummy value
    #ifdef __linux__
        std::ifstream status("/proc/self/status");
        if (status.is_open()) {
            std::string line;
            while (std::getline(status, line)) {
                if (line.substr(0, 6) == "VmRSS:") {
                    std::istringstream iss(line.substr(6));
                    size_t memory;
                    iss >> memory;
                    return memory * 1024; // Convert KB to bytes
                }
            }
        }
    #endif
    
    // Fallback: return dummy value
    return 0;
}

PerformanceMetrics ValidationFramework::calculateMetrics(const std::vector<std::vector<float>>& predictions, 
                                                         const std::vector<std::vector<float>>& targets) {
    PerformanceMetrics metrics;
    
    // Initialize all metrics to zero
    metrics.accuracy = 0.0f;
    metrics.precision = 0.0f;
    metrics.recall = 0.0f;
    metrics.f1_score = 0.0f;
    metrics.rmse = 0.0f;
    metrics.mae = 0.0f;
    metrics.r2_score = 0.0f;
    metrics.auc_roc = 0.0f;
    metrics.trainingTime = 0.0;
    metrics.inferenceTime = 0.0;
    metrics.memoryUsage = 0;
    metrics.parameters = 0;
    metrics.isClassification = false;
    
    if (predictions.empty() || targets.empty() || predictions.size() != targets.size()) {
        return metrics;
    }
    
    float mse = 0.0f;
    float mae = 0.0f;
    float sumTargets = 0.0f;
    float sumPredictions = 0.0f;
    float sumSquaredTargets = 0.0f;
    
    size_t numSamples = predictions.size();
    size_t outputSize = targets[0].size();
    
    for (size_t i = 0; i < numSamples; ++i) {
        for (size_t j = 0; j < outputSize; ++j) {
            float prediction = predictions[i][j];
            float target = targets[i][j];
            
            float error = prediction - target;
            mse += error * error;
            mae += std::abs(error);
            
            sumTargets += target;
            sumPredictions += prediction;
            sumSquaredTargets += target * target;
        }
    }
    
    mse /= (numSamples * outputSize);
    mae /= (numSamples * outputSize);
    float rmse = std::sqrt(mse);
    
    float meanTarget = sumTargets / (numSamples * outputSize);
    float meanPrediction = sumPredictions / (numSamples * outputSize);
    
    float sst = sumSquaredTargets / (numSamples * outputSize) - meanTarget * meanTarget;
    float sse = mse;
    
    float r2 = 0.0f;
    if (sst > 1e-6f) {
        r2 = 1.0f - (sse / sst);
    }
    
    metrics.rmse = rmse;
    metrics.mse = mse;
    metrics.mae = mae;
    metrics.r2_score = r2;
    
    return metrics;
}
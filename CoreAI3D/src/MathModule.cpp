#include "MathModule.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <iomanip>

// Mathematical constants
const double PI = 3.14159265358979323846;
const double E = 2.71828182845904523536;
const double GOLDEN_RATIO = 1.618033988749895;

// Constructor
MathModule::MathModule(const std::string& name)
    : moduleName(name), isInitialized(false), precision(1e-10), maxIterations(1000), convergenceThreshold(1e-8) {
    // Initialize random number generators
    std::random_device rd;
    randomGenerator = std::mt19937(rd());
    uniformDistribution = std::uniform_real_distribution<float>(0.0f, 1.0f);
    normalDistribution = std::normal_distribution<float>(0.0f, 1.0f);
}

// Destructor
MathModule::~MathModule() {
    clearAllData();
}

// Initialization
bool MathModule::initialize(const std::string& configPath) {
    try {
        if (isInitialized) {
            return true;
        }

        // Initialize CoreAI for mathematical processing
        mathCore = std::make_unique<CoreAI>(256, 5, 128, 1, -1.0f, 1.0f);

        // Load configuration if provided
        if (!configPath.empty()) {
            // TODO: Load configuration from file
        }

        isInitialized = true;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing MathModule: " << e.what() << std::endl;
        return false;
    }
}

void MathModule::setPrecision(float precision) {
    this->precision = std::max(1e-15f, std::min(1e-3f, precision));
}

void MathModule::setMaxIterations(int iterations) {
    this->maxIterations = std::max(1, std::min(10000, iterations));
}

void MathModule::setConvergenceThreshold(float threshold) {
    this->convergenceThreshold = std::max(1e-15f, std::min(1e-3f, threshold));
}

// Core mathematical processing interface
std::vector<float> MathModule::processNumbers(const std::vector<float>& numbers) {
    try {
        if (!isInitialized) {
            throw std::runtime_error("MathModule not initialized");
        }

        return processNumericalData(numbers);
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing numbers: " << e.what() << std::endl;
        return numbers;
    }
}

std::vector<float> MathModule::transformData(const std::vector<float>& data, const std::string& transform) {
    try {
        return applyMathematicalTransform(data, transform);
    }
    catch (const std::exception& e) {
        std::cerr << "Error transforming data: " << e.what() << std::endl;
        return data;
    }
}

float MathModule::evaluateExpression(const std::string& expression) {
    try {
        return evaluateMathematicalExpression(expression);
    }
    catch (const std::exception& e) {
        std::cerr << "Error evaluating expression: " << e.what() << std::endl;
        return 0.0f;
    }
}

// Matrix operations
bool MathModule::createMatrix(const std::string& name, int rows, int cols, const std::vector<float>& data) {
    try {
        MatrixInfo matrix;
        matrix.name = name;
        matrix.rows = rows;
        matrix.cols = cols;
        matrix.description = "User-created matrix";

        if (data.size() >= static_cast<size_t>(rows * cols)) {
            matrix.data.resize(rows, std::vector<float>(cols));
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    matrix.data[i][j] = data[i * cols + j];
                }
            }
        } else {
            matrix.data.resize(rows, std::vector<float>(cols, 0.0f));
        }

        matrices[name] = matrix;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error creating matrix: " << e.what() << std::endl;
        return false;
    }
}

bool MathModule::deleteMatrix(const std::string& name) {
    return matrices.erase(name) > 0;
}

std::vector<std::vector<float>> MathModule::getMatrix(const std::string& name) {
    auto it = matrices.find(name);
    if (it != matrices.end()) {
        return it->second.data;
    }
    return {};
}

bool MathModule::setMatrix(const std::string& name, const std::vector<std::vector<float>>& data) {
    auto it = matrices.find(name);
    if (it != matrices.end()) {
        it->second.data = data;
        return true;
    }
    return false;
}

std::vector<std::string> MathModule::listMatrices() {
    std::vector<std::string> names;
    names.reserve(matrices.size());
    for (const auto& pair : matrices) {
        names.push_back(pair.first);
    }
    return names;
}

// Vector operations
bool MathModule::createVector(const std::string& name, const std::vector<float>& data) {
    vectors[name] = data;
    return true;
}

bool MathModule::deleteVector(const std::string& name) {
    return vectors.erase(name) > 0;
}

std::vector<float> MathModule::getVector(const std::string& name) {
    auto it = vectors.find(name);
    if (it != vectors.end()) {
        return it->second;
    }
    return {};
}

bool MathModule::setVector(const std::string& name, const std::vector<float>& data) {
    auto it = vectors.find(name);
    if (it != vectors.end()) {
        it->second = data;
        return true;
    }
    return false;
}

std::vector<std::string> MathModule::listVectors() {
    std::vector<std::string> names;
    names.reserve(vectors.size());
    for (const auto& pair : vectors) {
        names.push_back(pair.first);
    }
    return names;
}

// Linear algebra
std::vector<std::vector<float>> MathModule::addMatrices(const std::string& matrixA, const std::string& matrixB) {
    auto itA = matrices.find(matrixA);
    auto itB = matrices.find(matrixB);

    if (itA != matrices.end() && itB != matrices.end()) {
        return matrixAddition(itA->second.data, itB->second.data);
    }
    return {};
}

std::vector<std::vector<float>> MathModule::multiplyMatrices(const std::string& matrixA, const std::string& matrixB) {
    auto itA = matrices.find(matrixA);
    auto itB = matrices.find(matrixB);

    if (itA != matrices.end() && itB != matrices.end()) {
        return matrixMultiplication(itA->second.data, itB->second.data);
    }
    return {};
}

std::vector<std::vector<float>> MathModule::transposeMatrix(const std::string& matrixName) {
    auto it = matrices.find(matrixName);
    if (it != matrices.end()) {
        return matrixTranspose(it->second.data);
    }
    return {};
}

float MathModule::determinant(const std::string& matrixName) {
    auto it = matrices.find(matrixName);
    if (it != matrices.end()) {
        return matrixDeterminant(it->second.data);
    }
    return 0.0f;
}

std::vector<std::vector<float>> MathModule::inverseMatrix(const std::string& matrixName) {
    auto it = matrices.find(matrixName);
    if (it != matrices.end()) {
        return matrixInverse(it->second.data);
    }
    return {};
}

std::vector<float> MathModule::solveLinearSystem(const std::string& matrixName, const std::vector<float>& b) {
    auto it = matrices.find(matrixName);
    if (it != matrices.end()) {
        // TODO: Implement linear system solver
        return b;
    }
    return {};
}

// Vector calculations
float MathModule::dotProduct(const std::string& vectorA, const std::string& vectorB) {
    auto itA = vectors.find(vectorA);
    auto itB = vectors.find(vectorB);

    if (itA != vectors.end() && itB != vectors.end()) {
        return dotProduct(itA->second, itB->second);
    }
    return 0.0f;
}

std::vector<float> MathModule::crossProduct(const std::string& vectorA, const std::string& vectorB) {
    auto itA = vectors.find(vectorA);
    auto itB = vectors.find(vectorB);

    if (itA != vectors.end() && itB != vectors.end()) {
        return crossProduct(itA->second, itB->second);
    }
    return {};
}

float MathModule::vectorMagnitude(const std::string& vectorName) {
    auto it = vectors.find(vectorName);
    if (it != vectors.end()) {
        return vectorMagnitude(it->second);
    }
    return 0.0f;
}

std::vector<float> MathModule::normalizeVector(const std::string& vectorName) {
    auto it = vectors.find(vectorName);
    if (it != vectors.end()) {
        return vectorNormalize(it->second);
    }
    return {};
}

std::vector<float> MathModule::projectVector(const std::string& vectorA, const std::string& vectorB) {
    auto itA = vectors.find(vectorA);
    auto itB = vectors.find(vectorB);

    if (itA != vectors.end() && itB != vectors.end()) {
        float dot = dotProduct(itA->second, itB->second);
        float magB = vectorMagnitude(itB->second);
        if (magB > 0) {
            std::vector<float> result = itB->second;
            float scale = dot / (magB * magB);
            for (float& val : result) {
                val *= scale;
            }
            return result;
        }
    }
    return {};
}

// Calculus operations
std::vector<float> MathModule::derivative(const std::vector<float>& data, float stepSize) {
    return numericalDerivative(data, stepSize);
}

std::vector<float> MathModule::integral(const std::vector<float>& data) {
    return numericalIntegral(data);
}

float MathModule::definiteIntegral(const std::string& expression, float lowerLimit, float upperLimit) {
    // TODO: Implement symbolic integration
    return 0.0f;
}

std::vector<float> MathModule::solveODE(const std::string& equation, float initialValue, float start, float end, int steps) {
    // TODO: Implement ODE solver
    return {};
}

// Optimization
OptimizationResult MathModule::optimize(const std::string& objectiveFunction, const std::vector<float>& initialGuess,
                                                   const std::string& method) {
    // TODO: Implement optimization
    return OptimizationResult{initialGuess, 0.0f, 0, "not_implemented", {}};
}

OptimizationResult MathModule::minimize(const std::function<float(const std::vector<float>&)>& objective,
                                                   const std::vector<float>& initialGuess, const std::string& method) {
    return gradientDescent(objective, initialGuess, 0.01f);
}

OptimizationResult MathModule::maximize(const std::function<float(const std::vector<float>&)>& objective,
                                                   const std::vector<float>& initialGuess, const std::string& method) {
    auto negativeObjective = [&](const std::vector<float>& x) -> float {
        return -objective(x);
    };
    return gradientDescent(negativeObjective, initialGuess, 0.01f);
}

// Statistics
StatisticalSummary MathModule::getStatistics(const std::vector<float>& data) {
    return calculateStatistics(data);
}

StatisticalSummary MathModule::getStatistics(const std::string& vectorName) {
    auto it = vectors.find(vectorName);
    if (it != vectors.end()) {
        return calculateStatistics(it->second);
    }
    return StatisticalSummary{};
}

std::vector<float> MathModule::generateRandomData(int count, const std::string& distribution, const std::vector<float>& parameters) {
    return generateRandomNumbers(count, distribution);
}

float MathModule::probability(const std::string& distribution, float x, const std::vector<float>& parameters) {
    return probabilityDensity(distribution, x, parameters);
}

std::vector<float> MathModule::fitDistribution(const std::vector<float>& data, const std::string& distribution) {
    // TODO: Implement distribution fitting
    return {};
}


// Special functions
float MathModule::gamma(float x) {
    return gammaFunction(x);
}

float MathModule::beta(float a, float b) {
    return betaFunction(a, b);
}

float MathModule::bessel(int n, float x) {
    return besselFunction(n, x);
}

std::complex<float> MathModule::complexGamma(const std::complex<float>& z) {
    return complexGammaFunction(z);
}

float MathModule::errorFunction(float x) {
    // TODO: Implement error function
    return 0.0f;
}

float MathModule::complementaryErrorFunction(float x) {
    // TODO: Implement complementary error function
    return 0.0f;
}

// Number theory
bool MathModule::isPrimeNumber(int n) {
    return isPrime(n);
}

std::vector<int> MathModule::factorize(int n) {
    return primeFactors(n);
}

int MathModule::gcd(int a, int b) {
    return greatestCommonDivisor(a, b);
}

int MathModule::lcm(int a, int b) {
    return leastCommonMultiple(a, b);
}

std::vector<int> MathModule::generatePrimesUpTo(int limit) {
    return generatePrimes(limit);
}

std::vector<int> MathModule::generatePrimesInRange(int start, int end) {
    std::vector<int> primes;
    for (int i = std::max(2, start); i <= end; ++i) {
        if (isPrime(i)) {
            primes.push_back(i);
        }
    }
    return primes;
}

// Symbolic mathematics
std::string MathModule::simplify(const std::string& expression) {
    return simplifyExpression(expression);
}

std::string MathModule::expand(const std::string& expression) {
    return expandExpression(expression);
}

std::string MathModule::factor(const std::string& expression) {
    return factorExpression(expression);
}

std::vector<std::string> MathModule::solve(const std::string& equation, const std::string& variable) {
    return solveEquation(equation);
}

std::string MathModule::differentiate(const std::string& expression, const std::string& variable) {
    // TODO: Implement symbolic differentiation
    return expression;
}

std::string MathModule::integrate(const std::string& expression, const std::string& variable) {
    // TODO: Implement symbolic integration
    return expression;
}

// Data visualization
DataVisualization MathModule::createPlot(const std::string& type, const std::vector<float>& x, const std::vector<float>& y,
                                                    const std::string& title, const std::string& xLabel, const std::string& yLabel) {
    DataVisualization plot;
    plot.type = type;
    plot.xData = x;
    plot.yData = y;
    plot.title = title;
    plot.xLabel = xLabel;
    plot.yLabel = yLabel;
    return plot;
}

DataVisualization MathModule::create3DPlot(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& z,
                                                     const std::string& title, const std::string& xLabel, const std::string& yLabel, const std::string& zLabel) {
    DataVisualization plot;
    plot.type = "3d";
    plot.xData = x;
    plot.yData = y;
    plot.zData = z;
    plot.title = title;
    plot.xLabel = xLabel;
    plot.yLabel = yLabel;
    plot.zLabel = zLabel;
    return plot;
}

bool MathModule::savePlotToFile(const DataVisualization& plot, const std::string& filePath) {
    // TODO: Implement plot saving
    return false;
}

// Mathematical modeling
std::vector<float> MathModule::modelData(const std::string& modelType, const std::vector<float>& parameters, const std::vector<float>& x) {
    // TODO: Implement mathematical modeling
    return x;
}

std::vector<float> MathModule::fitModel(const std::vector<float>& x, const std::vector<float>& y, const std::string& modelType) {
    // TODO: Implement model fitting
    return {};
}

float MathModule::evaluateModel(const std::string& modelType, const std::vector<float>& parameters, float x) {
    // TODO: Implement model evaluation
    return x;
}

// Financial mathematics
float MathModule::compoundInterest(float principal, float rate, float time, int compoundsPerYear) {
    float r = rate / compoundsPerYear;
    float n = compoundsPerYear * time;
    return principal * std::pow(1 + r, n);
}

float MathModule::presentValue(float futureValue, float rate, float time) {
    return futureValue / std::pow(1 + rate, time);
}

float MathModule::futureValue(float presentValue, float rate, float time) {
    return presentValue * std::pow(1 + rate, time);
}

std::vector<float> MathModule::calculateAmortization(float principal, float annualRate, int years, int paymentsPerYear) {
    // TODO: Implement amortization calculation
    return {};
}

// Cryptography (basic)
std::string MathModule::hashData(const std::vector<float>& data, const std::string& algorithm) {
    // TODO: Implement hashing
    return std::to_string(data.size());
}

std::vector<float> MathModule::encryptData(const std::vector<float>& data, const std::string& key) {
    // TODO: Implement encryption
    return data;
}

std::vector<float> MathModule::decryptData(const std::vector<float>& data, const std::string& key) {
    // TODO: Implement decryption
    return data;
}

// Protected methods implementation
std::vector<float> MathModule::processNumericalData(const std::vector<float>& data) {
    // Basic numerical processing - normalize and apply simple transform
    std::vector<float> processed = data;

    // Normalize
    float minVal = *std::min_element(processed.begin(), processed.end());
    float maxVal = *std::max_element(processed.begin(), processed.end());
    if (maxVal > minVal) {
        for (float& val : processed) {
            val = (val - minVal) / (maxVal - minVal);
        }
    }

    return processed;
}

std::vector<float> MathModule::applyMathematicalTransform(const std::vector<float>& data, const std::string& transformType) {
    std::vector<float> result = data;

    if (transformType == "log") {
        for (float& val : result) {
            val = std::log(std::abs(val) + 1.0f);
        }
    } else if (transformType == "sqrt") {
        for (float& val : result) {
            val = std::sqrt(std::abs(val));
        }
    } else if (transformType == "square") {
        for (float& val : result) {
            val = val * val;
        }
    }

    return result;
}

float MathModule::evaluateMathematicalExpression(const std::string& expression) {
    // Simple expression evaluator for basic arithmetic
    std::regex numberRegex(R"(\d+\.?\d*)");
    std::regex operatorRegex(R"([+\-*/])");

    // This is a simplified implementation
    // In a real system, you would use a proper expression parser
    std::istringstream iss(expression);
    float result = 0.0f;
    char op = '+';

    float num;
    char nextOp;
    while (iss >> num) {
        if (op == '+') result += num;
        else if (op == '-') result -= num;
        else if (op == '*') result *= num;
        else if (op == '/') result /= num;

        if (iss >> nextOp) {
            op = nextOp;
        }
    }

    return result;
}

// Linear algebra operations
std::vector<std::vector<float>> MathModule::matrixAddition(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        return {};
    }

    std::vector<std::vector<float>> result = A;
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] += B[i][j];
        }
    }
    return result;
}

std::vector<std::vector<float>> MathModule::matrixMultiplication(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    if (A[0].size() != B.size()) {
        return {};
    }

    std::vector<std::vector<float>> result(A.size(), std::vector<float>(B[0].size(), 0.0f));

    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < B[0].size(); ++j) {
            for (size_t k = 0; k < A[0].size(); ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

std::vector<std::vector<float>> MathModule::matrixTranspose(const std::vector<std::vector<float>>& A) {
    std::vector<std::vector<float>> result(A[0].size(), std::vector<float>(A.size()));

    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

float MathModule::matrixDeterminant(const std::vector<std::vector<float>>& A) {
    if (A.size() != A[0].size()) {
        return 0.0f;
    }

    int n = A.size();
    if (n == 1) return A[0][0];
    if (n == 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];

    // TODO: Implement general determinant calculation
    return 0.0f;
}

std::vector<std::vector<float>> MathModule::matrixInverse(const std::vector<std::vector<float>>& A) {
    // TODO: Implement matrix inversion
    return A;
}

// Vector operations
float MathModule::dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 0.0f;

    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

std::vector<float> MathModule::crossProduct(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != 3 || b.size() != 3) {
        return {};
    }

    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

float MathModule::vectorMagnitude(const std::vector<float>& v) {
    float sum = 0.0f;
    for (float val : v) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

std::vector<float> MathModule::vectorNormalize(const std::vector<float>& v) {
    float mag = vectorMagnitude(v);
    if (mag == 0.0f) return v;

    std::vector<float> result = v;
    for (float& val : result) {
        val /= mag;
    }
    return result;
}

// Calculus operations
std::vector<float> MathModule::numericalDerivative(const std::vector<float>& data, float h) {
    std::vector<float> result(data.size() - 1);

    for (size_t i = 0; i < data.size() - 1; ++i) {
        result[i] = (data[i + 1] - data[i]) / h;
    }

    return result;
}

std::vector<float> MathModule::numericalIntegral(const std::vector<float>& data) {
    std::vector<float> result(data.size() + 1, 0.0f);

    for (size_t i = 1; i <= data.size(); ++i) {
        result[i] = result[i - 1] + data[i - 1];
    }

    return result;
}

float MathModule::definiteIntegral(const std::function<float(float)>& func, float a, float b, int steps) {
    float h = (b - a) / steps;
    float sum = 0.5f * (func(a) + func(b));

    for (int i = 1; i < steps; ++i) {
        sum += func(a + i * h);
    }

    return sum * h;
}

std::vector<float> MathModule::solveDifferentialEquation(const std::string& equation, float initialCondition, float start, float end) {
    // TODO: Implement ODE solver
    return {};
}


std::vector<float> MathModule::generateRandomNumbers(int count, const std::string& distribution) {
    std::vector<float> numbers;
    numbers.reserve(count);

    if (distribution == "uniform") {
        for (int i = 0; i < count; ++i) {
            numbers.push_back(uniformDistribution(randomGenerator));
        }
    } else if (distribution == "normal") {
        for (int i = 0; i < count; ++i) {
            numbers.push_back(normalDistribution(randomGenerator));
        }
    } else {
        // Default to uniform
        for (int i = 0; i < count; ++i) {
            numbers.push_back(uniformDistribution(randomGenerator));
        }
    }

    return numbers;
}

float MathModule::probabilityDensity(const std::string& distribution, float x, const std::vector<float>& parameters) {
    if (distribution == "normal") {
        if (parameters.size() >= 2) {
            float mean = parameters[0];
            float stddev = parameters[1];
            return (1.0f / (stddev * std::sqrt(2.0f * PI))) * std::exp(-0.5f * ((x - mean) / stddev) * ((x - mean) / stddev));
        }
    } else if (distribution == "uniform") {
        if (parameters.size() >= 2) {
            float a = parameters[0];
            float b = parameters[1];
            if (x >= a && x <= b) {
                return 1.0f / (b - a);
            }
        }
    }

    return 0.0f;
}

std::vector<float> MathModule::linearRegressionFit(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) {
        return {};
    }

    size_t n = x.size();
    float sumX = 0.0f, sumY = 0.0f, sumXY = 0.0f, sumXX = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumXX += x[i] * x[i];
    }

    float slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    float intercept = (sumY - slope * sumX) / n;

    return {slope, intercept};
}

std::vector<float> MathModule::polynomialRegressionFit(const std::vector<float>& x, const std::vector<float>& y, int degree) {
    // TODO: Implement polynomial regression
    return linearRegressionFit(x, y);
}

// Special functions
float MathModule::gammaFunction(float x) {
    // Lanczos approximation for gamma function
    if (x <= 0) return 0.0f; // Simplified

    // TODO: Implement proper gamma function
    return std::tgamma(x);
}

float MathModule::betaFunction(float a, float b) {
    return gammaFunction(a) * gammaFunction(b) / gammaFunction(a + b);
}

float MathModule::besselFunction(int n, float x) {
    // TODO: Implement Bessel function
    return 0.0f;
}

std::complex<float> MathModule::complexGammaFunction(const std::complex<float>& z) {
    // TODO: Implement complex gamma function
    return std::complex<float>(1.0f, 0.0f);
}

// Number theory
bool MathModule::isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

std::vector<int> MathModule::primeFactors(int n) {
    std::vector<int> factors;

    while (n % 2 == 0) {
        factors.push_back(2);
        n /= 2;
    }

    for (int i = 3; i * i <= n; i += 2) {
        while (n % i == 0) {
            factors.push_back(i);
            n /= i;
        }
    }

    if (n > 1) {
        factors.push_back(n);
    }

    return factors;
}

int MathModule::greatestCommonDivisor(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

int MathModule::leastCommonMultiple(int a, int b) {
    return (a * b) / greatestCommonDivisor(a, b);
}

std::vector<int> MathModule::generatePrimes(int limit) {
    std::vector<bool> isPrime(limit + 1, true);
    std::vector<int> primes;

    isPrime[0] = isPrime[1] = false;

    for (int i = 2; i <= limit; ++i) {
        if (isPrime[i]) {
            primes.push_back(i);
            for (int j = i * 2; j <= limit; j += i) {
                isPrime[j] = false;
            }
        }
    }

    return primes;
}

// Symbolic mathematics
std::string MathModule::simplifyExpression(const std::string& expression) {
    // TODO: Implement expression simplification
    return expression;
}

std::string MathModule::expandExpression(const std::string& expression) {
    // TODO: Implement expression expansion
    return expression;
}

std::string MathModule::factorExpression(const std::string& expression) {
    // TODO: Implement expression factoring
    return expression;
}

std::vector<std::string> MathModule::solveEquation(const std::string& equation) {
    // TODO: Implement equation solving
    return {equation};
}

// Memory management
void MathModule::clearAllData() {
    matrices.clear();
    vectors.clear();
}

size_t MathModule::getMemoryUsage() const {
    size_t usage = 0;
    for (const auto& matrix : matrices) {
        usage += matrix.second.data.size() * sizeof(float);
    }
    for (const auto& vector : vectors) {
        usage += vector.second.size() * sizeof(float);
    }
    return usage;
}

// Training interface
bool MathModule::trainOnMathematicalData(const std::string& dataPath, int epochs) {
    // TODO: Implement mathematical data training
    return false;
}

bool MathModule::learnMathematicalPatterns(const std::vector<std::vector<float>>& datasets) {
    // TODO: Implement pattern learning
    return false;
}

// Utility functions
std::string MathModule::formatNumber(float number, int precision) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << number;
    return ss.str();
}

bool MathModule::isValidExpression(const std::string& expression) {
    // TODO: Implement expression validation
    return !expression.empty();
}

std::vector<std::string> MathModule::getSupportedFunctions() {
    return {
        "sin", "cos", "tan", "exp", "log", "sqrt", "abs",
        "min", "max", "sum", "mean", "std", "var"
    };
}

std::vector<std::string> MathModule::getSupportedConstants() {
    return {
        "pi", "e", "phi", "gamma"
    };
}
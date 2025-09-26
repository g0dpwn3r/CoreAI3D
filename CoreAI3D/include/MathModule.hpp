#ifndef MATH_MODULE_HPP
#define MATH_MODULE_HPP

#include "main.hpp"
#include "Core.hpp"
#include <vector>
#include <string>
#include <memory>
#include <complex>
#include <functional>

// Forward declarations for mathematical structures
struct MatrixInfo {
    std::string name;
    int rows;
    int cols;
    std::vector<std::vector<float>> data;
    std::string description;
};

struct OptimizationResult {
    std::vector<float> solution;
    float optimalValue;
    int iterations;
    std::string convergenceStatus;
    std::vector<float> convergenceHistory;
};

struct StatisticalSummary {
    float mean;
    float median;
    float mode;
    float standardDeviation;
    float variance;
    float skewness;
    float kurtosis;
    std::vector<float> percentiles;
    std::vector<float> quartiles;
};

struct DataVisualization {
    std::string type;
    std::vector<float> xData;
    std::vector<float> yData;
    std::vector<float> zData;
    std::string title;
    std::string xLabel;
    std::string yLabel;
    std::string zLabel;
};

class MathModule {
private:
    std::unique_ptr<CoreAI> mathCore;
    std::string moduleName;
    bool isInitialized;

    // Mathematical constants and parameters
    float precision;
    int maxIterations;
    float convergenceThreshold;
    std::map<std::string, MatrixInfo> matrices;
    std::map<std::string, std::vector<float>> vectors;

    // Random number generation
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<float> uniformDistribution;
    std::normal_distribution<float> normalDistribution;

protected:
    // Core mathematical functions
    virtual std::vector<float> processNumericalData(const std::vector<float>& data);
    virtual std::vector<float> applyMathematicalTransform(const std::vector<float>& data, const std::string& transformType);
    virtual float evaluateMathematicalExpression(const std::string& expression);

    // Linear algebra operations
    virtual std::vector<std::vector<float>> matrixAddition(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B);
    virtual std::vector<std::vector<float>> matrixMultiplication(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B);
    virtual std::vector<std::vector<float>> matrixTranspose(const std::vector<std::vector<float>>& A);
    virtual float matrixDeterminant(const std::vector<std::vector<float>>& A);
    virtual std::vector<std::vector<float>> matrixInverse(const std::vector<std::vector<float>>& A);

    // Vector operations
    virtual float dotProduct(const std::vector<float>& a, const std::vector<float>& b);
    virtual std::vector<float> crossProduct(const std::vector<float>& a, const std::vector<float>& b);
    virtual float vectorMagnitude(const std::vector<float>& v);
    virtual std::vector<float> vectorNormalize(const std::vector<float>& v);

    // Calculus operations
    virtual std::vector<float> numericalDerivative(const std::vector<float>& data, float h = 1e-5);
    virtual std::vector<float> numericalIntegral(const std::vector<float>& data);
    virtual float definiteIntegral(const std::function<float(float)>& func, float a, float b, int steps = 1000);
    virtual std::vector<float> solveDifferentialEquation(const std::string& equation, float initialCondition, float start, float end);

    // Optimization algorithms
    virtual OptimizationResult gradientDescent(const std::function<float(const std::vector<float>&)>& objective,
                                               const std::vector<float>& initialGuess, float learningRate = 0.01);
    virtual OptimizationResult newtonMethod(const std::function<float(const std::vector<float>&)>& objective,
                                            const std::function<std::vector<float>(const std::vector<float>&)>& gradient,
                                            const std::vector<float>& initialGuess);
    virtual OptimizationResult conjugateGradient(const std::function<float(const std::vector<float>&)>& objective,
                                                 const std::vector<float>& initialGuess);

    // Statistical functions
    virtual StatisticalSummary calculateStatistics(const std::vector<float>& data);
    virtual std::vector<float> generateRandomNumbers(int count, const std::string& distribution = "uniform");
    virtual float probabilityDensity(const std::string& distribution, float x, const std::vector<float>& parameters);
    virtual std::vector<float> linearRegressionFit(const std::vector<float>& x, const std::vector<float>& y);
    virtual std::vector<float> polynomialRegressionFit(const std::vector<float>& x, const std::vector<float>& y, int degree);

    // Special functions
    virtual float gammaFunction(float x);
    virtual float betaFunction(float a, float b);
    virtual float besselFunction(int n, float x);
    virtual std::complex<float> complexGammaFunction(const std::complex<float>& z);

    // Number theory
    virtual bool isPrime(int n);
    virtual std::vector<int> primeFactors(int n);
    virtual int greatestCommonDivisor(int a, int b);
    virtual int leastCommonMultiple(int a, int b);
    virtual std::vector<int> generatePrimes(int limit);

    // Symbolic mathematics
    virtual std::string simplifyExpression(const std::string& expression);
    virtual std::string expandExpression(const std::string& expression);
    virtual std::string factorExpression(const std::string& expression);
    virtual std::vector<std::string> solveEquation(const std::string& equation);

public:
    // Constructor
    MathModule(const std::string& name);
    virtual ~MathModule();

    // Initialization
    bool initialize(const std::string& configPath = "");
    void setPrecision(float precision);
    void setMaxIterations(int iterations);
    void setConvergenceThreshold(float threshold);

    // Core mathematical processing interface
    virtual std::vector<float> processNumbers(const std::vector<float>& numbers);
    virtual std::vector<float> transformData(const std::vector<float>& data, const std::string& transform);
    virtual float evaluateExpression(const std::string& expression);

    // Matrix operations
    bool createMatrix(const std::string& name, int rows, int cols, const std::vector<float>& data = {});
    bool deleteMatrix(const std::string& name);
    std::vector<std::vector<float>> getMatrix(const std::string& name);
    bool setMatrix(const std::string& name, const std::vector<std::vector<float>>& data);
    std::vector<std::string> listMatrices();

    // Vector operations
    bool createVector(const std::string& name, const std::vector<float>& data);
    bool deleteVector(const std::string& name);
    std::vector<float> getVector(const std::string& name);
    bool setVector(const std::string& name, const std::vector<float>& data);
    std::vector<std::string> listVectors();

    // Linear algebra
    std::vector<std::vector<float>> addMatrices(const std::string& matrixA, const std::string& matrixB);
    std::vector<std::vector<float>> multiplyMatrices(const std::string& matrixA, const std::string& matrixB);
    std::vector<std::vector<float>> transposeMatrix(const std::string& matrixName);
    float determinant(const std::string& matrixName);
    std::vector<std::vector<float>> inverseMatrix(const std::string& matrixName);
    std::vector<float> solveLinearSystem(const std::string& matrixName, const std::vector<float>& b);

    // Vector calculations
    float dotProduct(const std::string& vectorA, const std::string& vectorB);
    std::vector<float> crossProduct(const std::string& vectorA, const std::string& vectorB);
    float vectorMagnitude(const std::string& vectorName);
    std::vector<float> normalizeVector(const std::string& vectorName);
    std::vector<float> projectVector(const std::string& vectorA, const std::string& vectorB);

    // Calculus operations
    std::vector<float> derivative(const std::vector<float>& data, float stepSize = 1e-5);
    std::vector<float> integral(const std::vector<float>& data);
    float definiteIntegral(const std::string& expression, float lowerLimit, float upperLimit);
    std::vector<float> solveODE(const std::string& equation, float initialValue, float start, float end, int steps = 100);

    // Optimization
    OptimizationResult optimize(const std::string& objectiveFunction, const std::vector<float>& initialGuess,
                               const std::string& method = "gradient_descent");
    OptimizationResult minimize(const std::function<float(const std::vector<float>&)>& objective,
                                const std::vector<float>& initialGuess, const std::string& method = "gradient_descent");
    OptimizationResult maximize(const std::function<float(const std::vector<float>&)>& objective,
                                const std::vector<float>& initialGuess, const std::string& method = "gradient_descent");

    // Statistics
    StatisticalSummary getStatistics(const std::vector<float>& data);
    StatisticalSummary getStatistics(const std::string& vectorName);
    std::vector<float> generateRandomData(int count, const std::string& distribution, const std::vector<float>& parameters = {});
    float probability(const std::string& distribution, float x, const std::vector<float>& parameters = {});
    std::vector<float> fitDistribution(const std::vector<float>& data, const std::string& distribution);

    // Regression analysis
    struct RegressionResult {
        std::vector<float> coefficients;
        float rSquared;
        float adjustedRSquared;
        std::vector<float> residuals;
        std::vector<float> predictions;
    };

    RegressionResult linearRegression(const std::vector<float>& x, const std::vector<float>& y);
    RegressionResult polynomialRegression(const std::vector<float>& x, const std::vector<float>& y, int degree);
    RegressionResult exponentialRegression(const std::vector<float>& x, const std::vector<float>& y);
    RegressionResult logarithmicRegression(const std::vector<float>& x, const std::vector<float>& y);

    // Special functions
    float gamma(float x);
    float beta(float a, float b);
    float bessel(int n, float x);
    std::complex<float> complexGamma(const std::complex<float>& z);
    float errorFunction(float x);
    float complementaryErrorFunction(float x);

    // Number theory
    bool isPrimeNumber(int n);
    std::vector<int> factorize(int n);
    int gcd(int a, int b);
    int lcm(int a, int b);
    std::vector<int> generatePrimesUpTo(int limit);
    std::vector<int> generatePrimesInRange(int start, int end);

    // Symbolic mathematics
    std::string simplify(const std::string& expression);
    std::string expand(const std::string& expression);
    std::string factor(const std::string& expression);
    std::vector<std::string> solve(const std::string& equation, const std::string& variable = "x");
    std::string differentiate(const std::string& expression, const std::string& variable = "x");
    std::string integrate(const std::string& expression, const std::string& variable = "x");

    // Data visualization (returns data for plotting)
    DataVisualization createPlot(const std::string& type, const std::vector<float>& x, const std::vector<float>& y,
                                const std::string& title = "", const std::string& xLabel = "", const std::string& yLabel = "");
    DataVisualization create3DPlot(const std::vector<float>& x, const std::vector<float>& y, const std::vector<float>& z,
                                  const std::string& title = "", const std::string& xLabel = "", const std::string& yLabel = "", const std::string& zLabel = "");
    bool savePlotToFile(const DataVisualization& plot, const std::string& filePath);

    // Mathematical modeling
    std::vector<float> modelData(const std::string& modelType, const std::vector<float>& parameters, const std::vector<float>& x);
    std::vector<float> fitModel(const std::vector<float>& x, const std::vector<float>& y, const std::string& modelType);
    float evaluateModel(const std::string& modelType, const std::vector<float>& parameters, float x);

    // Financial mathematics
    float compoundInterest(float principal, float rate, float time, int compoundsPerYear = 1);
    float presentValue(float futureValue, float rate, float time);
    float futureValue(float presentValue, float rate, float time);
    std::vector<float> calculateAmortization(float principal, float annualRate, int years, int paymentsPerYear = 12);

    // Cryptography (basic)
    std::string hashData(const std::vector<float>& data, const std::string& algorithm = "sha256");
    std::vector<float> encryptData(const std::vector<float>& data, const std::string& key);
    std::vector<float> decryptData(const std::vector<float>& data, const std::string& key);

    // Status and information
    bool isReady() const { return isInitialized; }
    std::string getModuleName() const { return moduleName; }
    float getPrecision() const { return precision; }
    int getMaxIterations() const { return maxIterations; }

    // Memory management
    void clearAllData();
    size_t getMemoryUsage() const;

    // Training interface for math-specific learning
    virtual bool trainOnMathematicalData(const std::string& dataPath, int epochs = 10);
    virtual bool learnMathematicalPatterns(const std::vector<std::vector<float>>& datasets);

    // Utility functions
    std::string formatNumber(float number, int precision = 6);
    bool isValidExpression(const std::string& expression);
    std::vector<std::string> getSupportedFunctions();
    std::vector<std::string> getSupportedConstants();
};

#endif // MATH_MODULE_HPP
#include "Core.hpp"

CoreAI::CoreAI(int inputSize_, int layers_, int neurons_, int outputSize_,
    float min_, float max_, bool verbose_)
    : inputSize(inputSize_), layers(layers_), neurons(neurons_),
    outputSize(outputSize_), minVal(min_), maxVal(max_), verbose(verbose_),
    x(1.0f), y(1.0f), z(1.0f)  // Initialize transform variables to meaningful values
{
    if (verbose) {
        std::cout << "[DEBUG CONSTRUCTOR] Starting CoreAI constructor with params: "
                  << "inputSize=" << inputSize_ << ", layers=" << layers_
                  << ", neurons=" << neurons_ << ", outputSize=" << outputSize_
                  << ", min=" << min_ << ", max=" << max_ << std::endl;
        
        std::cout << "[DEBUG CONSTRUCTOR] Transform variables initialized - x: " << x
                  << ", y: " << y << ", z: " << z << ", epsilon: " << epsilon << std::endl;
    }

    // Parameter validation
    if (inputSize_ <= 0 || layers_ <= 0 || neurons_ <= 0 || outputSize_ <= 0) {
        if (verbose) {
            std::cout << "[DEBUG CONSTRUCTOR] Parameter validation failed: sizes must be positive" << std::endl;
        }
        throw std::invalid_argument("CoreAI: All size parameters must be positive integers");
    }
    if (min_ >= max_) {
        if (verbose) {
            std::cout << "[DEBUG CONSTRUCTOR] Parameter validation failed: min >= max" << std::endl;
        }
        throw std::invalid_argument("CoreAI: minVal must be less than maxVal");
    }

    if (verbose) {
        std::cout << "[DEBUG CONSTRUCTOR] Parameter validation passed" << std::endl;
    }

    // Initialize random number generator with time-based seed for better randomness
    std::random_device rd;
    std::mt19937 gen(rd());  // Use random seed instead of fixed seed for true randomness
    
    if (verbose) {
        std::cout << "[DEBUG CONSTRUCTOR] Initializing weights with random values" << std::endl;
    }

    // Initialize vectors (these should be zero, not random)
    initializeVector(this->hidden_error, neurons);
    std::fill(this->hidden_error.begin(), this->hidden_error.end(), 0.0f);

    initializeVector(this->hidden_output, neurons);
    std::fill(this->hidden_output.begin(), this->hidden_output.end(), 0.0f);

    // CRITICAL: Initialize weight matrices with random values
    // Use Xavier/Glorot initialization for better gradient flow
    float scale = std::sqrt(2.0f / (inputSize + neurons));
    std::uniform_real_distribution<float> weight_dist(-scale, scale);

    // Initialize hidden-to-output weights with explicit random values
    initializeMatrix(this->weigth_output_hidden, outputSize, neurons);
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < neurons; ++j) {
            this->weigth_output_hidden[i][j] = weight_dist(gen);
        }
    }

    // Initialize input-to-hidden weights with explicit random values
    initializeMatrix(this->weigth_input_hidden, neurons, inputSize);
    for (int i = 0; i < neurons; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            this->weigth_input_hidden[i][j] = weight_dist(gen);
        }
    }

    // DEBUG: Verify weight initialization
    if (verbose) {
        std::cout << "[DEBUG CONSTRUCTOR] Weight initialization complete:" << std::endl;
        std::cout << "[DEBUG CONSTRUCTOR] Input-Hidden weights sample (first 5x5):" << std::endl;
        for (int i = 0; i < std::min(5, neurons); ++i) {
            std::cout << "  Row " << i << ": ";
            for (int j = 0; j < std::min(5, inputSize); ++j) {
                std::cout << this->weigth_input_hidden[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "[DEBUG CONSTRUCTOR] Hidden-Output weights sample (first 5x5):" << std::endl;
        for (int i = 0; i < std::min(5, outputSize); ++i) {
            std::cout << "  Row " << i << ": ";
            for (int j = 0; j < std::min(5, neurons); ++j) {
                std::cout << this->weigth_output_hidden[i][j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "[DEBUG CONSTRUCTOR] Weight range: [" << -scale << ", " << scale << "]" << std::endl;
    }

    // Initialize trainer pointer to nullptr
    this->trainer = nullptr;
}
std::vector<std::vector<float>>& CoreAI::getInput()
{
    return this->input;
}

std::vector<std::vector<float>>& CoreAI::getOutput()
{
    return this->output;
}

std::vector<std::vector<float>>& CoreAI::getResults() 
{
    return this->results;
}

// Implementations for getters (example for one, others similar)
const std::vector<std::vector<float>>& CoreAI::getHiddenData() const
{
    return this->hidden; // Assuming 'hiddenData' is a member
}
const std::vector<float>& CoreAI::getHiddenOutputData() const
{
    return this->hidden_output;
}
const std::vector<float>& CoreAI::getHiddenErrorData() const
{
    return this->hidden_error;
}
const std::vector<std::vector<float>>& CoreAI::getWeightsHiddenInput() const
{
    return this->weigth_input_hidden;
}
const std::vector<std::vector<float>>& CoreAI::getWeightsOutputHidden() const
{
    return this->weigth_output_hidden;
}

// Implementations for setters (example for one, others similar)
void CoreAI::setInput(const std::vector<std::vector<float> >& data)
{
    this->input = data;
}
void CoreAI::setOutput(const std::vector<std::vector<float> >& data)
{
    this->output = data; // Assuming this is for targets, or actual model output
}

float CoreAI::sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

void CoreAI::setTarget(const std::vector<std::vector<float> >& data)
{
    // If output is used for targets, then assign:
    this->output = data;
    // Or if you have a separate 'targets' member:
    // this->targets = data;
}
void CoreAI::setHiddenData(const std::vector<std::vector<float> >& data)
{
    this->hidden = data;
}
void CoreAI::setHiddenOutputData(const std::vector<float>& data)
{
    this->hidden_output = data;
}
void CoreAI::setHiddenErrorData(const std::vector<float>& data)
{
    this->hidden_error = data;
}
void CoreAI::setWeightsHiddenInput(const std::vector<std::vector<float> >& data)
{
    this->weigth_input_hidden = data;
}
void CoreAI::setWeightsOutputHidden(const std::vector<std::vector<float> >& data)
{
    this->weigth_output_hidden = data;
}


// Helper functions to make the code cleaner:
void CoreAI::initializeMatrix(std::vector<std::vector<float>>& matrix, int rows, int cols) {
    matrix.resize(rows);
    for (auto& row : matrix) {
        row.resize(cols);
    }
}

void CoreAI::initializeVector(std::vector<float>& vec, int size) {
    vec.resize(size);
}

template<typename Func>
void CoreAI::fillMatrixWithTransform(std::vector<std::vector<float>>& matrix, Func transform) {
    for (auto& row : matrix) {
        for (auto& val : row) {
            val = transform();
        }
    }
}

template<typename Func>
void CoreAI::fillVectorWithTransform(std::vector<float>& vec, Func transform) {
    for (auto& val : vec) {
        val = transform();
    }
}

void CoreAI::computeMinMax(const std::vector<std::vector<float>>& data, float& min_val, float& max_val) {
    if (data.empty() || data[0].empty()) {
        min_val = 0.0f;
        max_val = 1.0f;
        return;
    }
    min_val = std::numeric_limits<float>::max();
    max_val = std::numeric_limits<float>::lowest();
    for (const auto& row : data) {
        for (float val : row) {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }
    if (min_val == max_val) {
        max_val = min_val + 1.0f; // Avoid division by zero
    }
}

float CoreAI::beautifulRandom()
{
    static std::mt19937 generator(42);
    static std::uniform_real_distribution<float> distribution(this->minVal,
        this->maxVal);
    return distribution(generator);
}

bool CoreAI::safe1DIndex(const std::vector<float>& vec, int index)
{
    return index >= 0 && index < (int)vec.size();
}

bool CoreAI::safe2DIndex(const std::vector<std::vector<float> >& matrix, int i,
    int j)
{
    return i >= 0 && i < matrix.size() && j >= 0 && j < matrix[i].size();
}

std::vector<std::vector<float>> CoreAI::forward(const std::vector<std::vector<float> >& inputvalue)
{
    // Parameter validation
    if (inputvalue.empty()) {
        throw std::invalid_argument("CoreAI::forward: input cannot be empty");
    }
    for (const auto& input : inputvalue) {
        if (input.size() != (size_t)inputSize) {
            throw std::invalid_argument("CoreAI::forward: input sample size must match inputSize");
        }
    }

    this->input = inputvalue;
    this->results.clear();
    this->hidden.clear();
    this->output.clear();

    // DEBUG: Log input statistics
    if (verbose && !inputvalue.empty() && !inputvalue[0].empty()) {
        std::cout << "[DEBUG FORWARD] Input size: " << inputvalue.size() << " samples, "
                  << inputvalue[0].size() << " features per sample" << std::endl;
        std::cout << "[DEBUG FORWARD] First input sample: ";
        for (size_t k = 0; k < std::min(size_t(5), inputvalue[0].size()); ++k) {
            std::cout << inputvalue[0][k] << " ";
        }
        std::cout << "..." << std::endl;
    }

    for (size_t i = 0; i < this->input.size(); ++i)
    {
        std::vector<float> hiddenLayer(this->neurons, 0.0f);
        std::vector<float> outputLayer(this->outputSize, 0.0f);

        // Input to hidden layer - 3D neural network computation with transforms
        for (int j = 0; j < this->neurons; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < this->input[i].size(); ++k)
            {
                if (safe2DIndex(this->weigth_input_hidden, j, k))
                {
                    // Apply 3D transforms to create spatial features
                    // Use different transforms for different dimensions
                    float input_val = this->input[i][k];
                    float weight_val = this->weigth_input_hidden[j][k];
                    
                    // Apply 3D transformation based on position in feature space
                    float transformed;
                    int transform_type = k % 3; // Cycle through x, y, z transforms
                    
                    if (transform_type == 0) {
                        // X-dimension: tanh-based transform
                        transformed = xTransform2(input_val, epsilon, 0.0f, weight_val);
                    } else if (transform_type == 1) {
                        // Y-dimension: sine-based transform
                        transformed = yTransform2(input_val, epsilon, 0.0f, weight_val);
                    } else {
                        // Z-dimension: sigmoid-based transform
                        transformed = zTransform2(input_val, epsilon, 0.0f, weight_val);
                    }
                    
                    sum += transformed;
                }
            }
            hiddenLayer[j] = this->sigmoid(sum);
        }

        // DEBUG: Log hidden layer activation for first sample
        if (verbose && i == 0) {
            std::cout << "[DEBUG FORWARD] Hidden layer activations (first 5): ";
            for (int j = 0; j < std::min(5, this->neurons); ++j) {
                std::cout << hiddenLayer[j] << " ";
            }
            std::cout << "..." << std::endl;
        }

        // Hidden to output layer - standard neural network computation
        for (int j = 0; j < this->outputSize; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < this->neurons; ++k)
            {
                if (safe2DIndex(this->weigth_output_hidden, j, k))
                {
                    // Standard weighted sum: sum += hidden * weight
                    sum += hiddenLayer[k] * this->weigth_output_hidden[j][k];
                }
            }
            outputLayer[j] = this->sigmoid(sum);
        }

        // DEBUG: Log output layer activation for first sample
        if (verbose && i == 0) {
            std::cout << "[DEBUG FORWARD] Output layer activations: ";
            for (int j = 0; j < this->outputSize; ++j) {
                std::cout << outputLayer[j] << " ";
            }
            std::cout << std::endl;
        }

        this->hidden.push_back(hiddenLayer);
        this->output.push_back(outputLayer);
        this->results.push_back(outputLayer);
    }

    return this->results;
}

void CoreAI::train(const std::vector<std::vector<float> >& inputs,
    const std::vector<std::vector<float> >& targets,
    double learningRate, int& numSamples)
{
    // Parameter validation
    if (learningRate <= 0.0 || learningRate > 1.0) {
        throw std::invalid_argument("CoreAI::train: learningRate must be between 0 and 1");
    }
    if (inputs.empty() || targets.empty()) {
        throw std::invalid_argument("CoreAI::train: inputs and targets cannot be empty");
    }
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("CoreAI::train: inputs and targets must have the same number of samples");
    }
    for (const auto& input : inputs) {
        if (input.size() != (size_t)inputSize) {
            throw std::invalid_argument("CoreAI::train: input sample size must match inputSize");
        }
    }
    for (const auto& target : targets) {
        if (target.size() != (size_t)outputSize) {
            throw std::invalid_argument("CoreAI::train: target sample size must match outputSize");
        }
    }
    if (weigth_input_hidden.size() != this->neurons) {
        throw std::runtime_error("Invalid weight_input_hidden row count!");
    }

    for (int j = 0; j < this->neurons; j++) {
        if (weigth_input_hidden[j].size() != inputSize) {
            throw std::runtime_error("Invalid weight_input_hidden column count!");
        }
    }

    // Normalize inputs and targets
    float input_min, input_max, target_min, target_max;
    computeMinMax(inputs, input_min, input_max);
    computeMinMax(targets, target_min, target_max);

    std::vector<std::vector<float>> normalized_inputs = inputs;
    std::vector<std::vector<float>> normalized_targets = targets;

    // Normalize inputs to [0,1]
    for (auto& row : normalized_inputs) {
        for (float& val : row) {
            val = (val - input_min) / (input_max - input_min);
        }
    }

    // Normalize targets to [0,1]
    for (auto& row : normalized_targets) {
        for (float& val : row) {
            val = (val - target_min) / (target_max - target_min);
        }
    }

    // Store normalization parameters in trainer if available
    if (trainer) {
        trainer->original_data_global_min = target_min;
        trainer->original_data_global_max = target_max;
    }

    // Ensure forward pass has been run
    if (this->results.empty()) {
        this->forward(normalized_inputs);
    }

    // Backpropagation
    for (size_t sample = 0; sample < normalized_inputs.size(); ++sample)
    {
        
        // DEBUG: Log transform function inputs
        if (verbose && sample == 0) {
            std::cout << "[DEBUG TRAIN] Transform function inputs - x: " << this->x
                      << ", y: " << this->y << ", z: " << this->z
                      << ", epsilon: " << this->epsilon << std::endl;
        }
        
        // Output layer error
        std::vector<float> output_errors(this->outputSize, 0.0f);
        for (int j = 0; j < this->outputSize; ++j)
        {
            if (!safe2DIndex(this->results, sample, j) || !safe2DIndex(normalized_targets, sample, j)) {
                throw std::out_of_range("CoreAI::train: Index out of range during backpropagation");
            }

            float output = this->results[sample][j];
            float target = normalized_targets[sample][j];
            
            // DEBUG: Log error calculation components
            if (verbose && sample == 0 && j == 0) {
                std::cout << "[DEBUG TRAIN] Sample 0, Output 0 - output: " << output
                          << ", target: " << target << std::endl;
            }
            
            // Standard backpropagation: error = (target - output) * derivative
            output_errors[j] = (target - output) * output * (1.0f - output); // Derivative of sigmoid
        
        }

        // Hidden layer error
        std::vector<float> hidden_errors(this->neurons, 0.0f);
        for (int j = 0; j < this->neurons; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < this->outputSize; ++k)
            {
                if (!safe2DIndex(this->weigth_output_hidden, k, j)) {
                    throw std::out_of_range("CoreAI::train: Index out of range during hidden error calculation");
                }
                // Standard backpropagation: sum += output_errors[k] * weight
                sum += output_errors[k] * this->weigth_output_hidden[k][j];
            }
            float hidden = this->hidden[sample][j];
            hidden_errors[j] = sum * hidden * (1.0f - hidden); // Derivative of sigmoid
            
            // DEBUG: Log hidden error calculation
            if (verbose && sample == 0 && j == 0) {
                std::cout << "[DEBUG TRAIN] Sample 0, Hidden 0 - sum: " << sum
                          << ", hidden activation: " << hidden
                          << ", hidden error: " << hidden_errors[j] << std::endl;
            }
        }

        // Update weights output to hidden
        for (int j = 0; j < this->outputSize; ++j)
        {
            for (int k = 0; k < this->neurons; ++k)
            {
                if(!safe2DIndex(this->weigth_output_hidden, j, k)) {
                    throw std::out_of_range("CoreAI::train: Index out of range during weight update");
                }
                // Standard backpropagation: gradient = error * activation
                float gradient = output_errors[j] * hidden[sample][k];
                
                // Gradient clipping to prevent exploding gradients
                if (gradient > 1.0f) gradient = 1.0f;
                if (gradient < -1.0f) gradient = -1.0f;

                this->weigth_output_hidden[j][k] += learningRate * gradient;
            }
        }

        // Update weights input to hidden with 3D transforms
        for (int j = 0; j < this->neurons; ++j)
        {
            for (size_t k = 0; k < normalized_inputs[sample].size(); ++k)
            {
                if(!safe2DIndex(this->weigth_input_hidden, j, k)) {
                    throw std::out_of_range("CoreAI::train: Index out of range during weight update");
                }
                
                // Apply 3D transform derivatives based on position in feature space
                float input_val = normalized_inputs[sample][k];
                float weight_val = this->weigth_input_hidden[j][k];
                
                int transform_type = k % 3; // Match the transform used in forward pass
                float transform_derivative;
                
                if (transform_type == 0) {
                    // X-dimension: tanh derivative
                    transform_derivative = xTransform2_derivative_m(input_val, epsilon, 0.0f, weight_val);
                } else if (transform_type == 1) {
                    // Y-dimension: sine derivative
                    transform_derivative = yTransform2_derivative_m(input_val, epsilon, 0.0f, weight_val);
                } else {
                    // Z-dimension: sigmoid derivative
                    transform_derivative = zTransform2_derivative_m(input_val, epsilon, 0.0f, weight_val);
                }
                
                // Backpropagation with 3D transform: gradient = error * transform_derivative
                float gradient = hidden_errors[j] * transform_derivative;
                
                // Gradient clipping to prevent exploding gradients
                if (gradient > 1.0f) gradient = 1.0f;
                if (gradient < -1.0f) gradient = -1.0f;

                this->weigth_input_hidden[j][k] += learningRate * gradient;
            }
        }
    }
}

void CoreAI::normalizeInput(float min_val, float max_val)
{
    for (auto& row : input)
    {
        for (float& val : row)
        {
            val = (val - min_val) / (max_val - min_val);
        }
    }
}

void CoreAI::normalizeOutput(float min_val, float max_val)
{
    for (auto& row : output)
    {
        for (float& val : row)
        {
            val = (val - min_val) / (max_val - min_val);
        }
    }
}

void CoreAI::denormalizeOutput()
{
    if (!trainer) {
        std::cerr << "Error: Trainer not set in CoreAI for denormalization." << std::endl;
        return;
    }

    // Validation
    if (results.empty()) {
        std::cerr << "Warning: No results to denormalize." << std::endl;
        return;
    }

    // DEBUG: Log denormalization parameters
    if (verbose) {
        std::cout << "[DEBUG DENORMALIZE] original_data_global_min: " << trainer->original_data_global_min
                  << ", original_data_global_max: " << trainer->original_data_global_max
                  << ", range: " << (trainer->original_data_global_max - trainer->original_data_global_min)
                  << ", minVal: " << this->minVal << ", maxVal: " << this->maxVal << std::endl;
    }

    float normalized_range_diff = this->maxVal - this->minVal;
    float original_range_diff = trainer->original_data_global_max - trainer->original_data_global_min;

    // Check for valid ranges
    if (std::abs(normalized_range_diff) < std::numeric_limits<float>::epsilon()) {
        std::cerr << "Warning: Normalized range is zero, cannot denormalize properly." << std::endl;
        return;
    }
    if (std::abs(original_range_diff) < std::numeric_limits<float>::epsilon()) {
        std::cerr << "Warning: Original data range is zero, denormalizing to constant value." << std::endl;
    }

    for (auto& row : results)
    { // Denormalize the results/predictions
        for (float& val : row)
        {
            float original_val = val;
            if (std::abs(original_range_diff) > std::numeric_limits<float>::epsilon()) {
                val = ((val - this->minVal) / normalized_range_diff) * original_range_diff + trainer->original_data_global_min;
            } else {
                val = trainer->original_data_global_min; // If original range is zero, map to original min
            }

            // DEBUG: Log first few denormalizations (per call, not static)
            if (verbose) {
                static int call_count = 0;
                if (call_count++ < 5) {
                    std::cout << "[DEBUG DENORMALIZE] prediction " << original_val << " -> " << val << std::endl;
                }
            }
        }
    }
}

float CoreAI::xTransform(float x, float y, float z, float n, float m)
{
    // Remove restrictive conditions and simplify for meaningful transformations
    if (y == 0.0f || m == 0.0f) {
        return std::abs(x) + this->epsilon; // Return meaningful value instead of just epsilon
    }

    double denom = static_cast<double>(y) * std::pow(static_cast<double>(m), 2.0) + this->epsilon;
    double num = std::pow(static_cast<double>(x), 2.0) * static_cast<double>(n) + static_cast<double>(z);
    
    float result = static_cast<float>(num / denom);

    return (std::isnan(result) || std::isinf(result)) ? this->epsilon : result;
}

float CoreAI::yTransform(float x, float y, float z, float n, float m)
{
    float denom = std::pow(x, 2.0f) * n + this->epsilon;
    if (std::abs(denom) < this->epsilon) {
        denom = this->epsilon;
    }
    float result = (std::pow(y, 3.0f) + std::pow(x, n) + std::pow(z, 2.0f)) / denom;
    return (std::isnan(result) || std::isinf(result)) ? this->epsilon : result;
}

float CoreAI::zTransform(float x, float y, float z, float n, float m)
{
   if (std::abs(y) < this->epsilon) {
        y = this->epsilon;
    }
    float result = std::pow(z, 2.0f) * std::pow(std::abs(y), n) + m;
    return (std::isnan(result) || std::isinf(result)) ? this->epsilon : result;
}

float CoreAI::xTransform2(float x, float e, float n, float m)
{
    // User's custom X-axis transform: f(x) = Σ(x³ⁿ + z)/(3^(n²))
    // where z is represented by parameter 'n' from training data
    // Summation from n=1 to 5 terms
    float result = 0.0f;
    const int max_terms = 5;
    
    for (int i = 1; i <= max_terms; ++i) {
        float power_term = std::pow(x, 3.0f * i);  // x^(3n)
        float denominator = std::pow(3.0f, i * i);  // 3^(n²)
        
        // Add epsilon to prevent division by zero
        if (std::abs(denominator) < this->epsilon) {
            denominator = this->epsilon;
        }
        
        result += (power_term + n) / denominator;
    }
    
    // Additional term: x^m/Ly * e^(e^m)
    // Using m as the exponent and e as the base
    float ly = std::abs(m) + 1.0f;  // Ly factor (avoid zero)
    float exp_term = std::pow(x, m) / ly * std::exp(std::pow(e, m));
    result += exp_term;
    
    // Clamp result to prevent numerical instability
    if (std::isnan(result) || std::isinf(result)) {
        return this->epsilon;
    }
    return result;
}

float CoreAI::yTransform2(float y, float e, float n, float m)
{
    // User's custom Y-axis transform: f(y) = Σ(y^(z+x^n))/(x³ N)
    // where z is parameter 'n', x is parameter 'm', and N is the iteration count
    // Summation from n=1 to 5 terms
    float result = 0.0f;
    const int max_terms = 5;
    
    for (int i = 1; i <= max_terms; ++i) {
        float exponent = n + std::pow(m, static_cast<float>(i));  // z + x^n
        float numerator = std::pow(std::abs(y), exponent);  // y^(z+x^n)
        float denominator = std::pow(std::abs(m), 3.0f) * i;  // x³ * N
        
        // Add epsilon to prevent division by zero
        if (std::abs(denominator) < this->epsilon) {
            denominator = this->epsilon;
        }
        
        result += numerator / denominator;
    }
    
    // Additional term: y^n/Ly * e^(e^(S/N))
    // Using n as the exponent, e as base, and m as S/N ratio
    float ly = std::abs(n) + 1.0f;  // Ly factor (avoid zero)
    float s_over_n = m / (std::abs(n) + this->epsilon);
    float exp_term = std::pow(std::abs(y), n) / ly * std::exp(std::pow(e, s_over_n));
    result += exp_term;
    
    // Clamp result to prevent numerical instability
    if (std::isnan(result) || std::isinf(result)) {
        return this->epsilon;
    }
    return result;
}

float CoreAI::zTransform2(float z, float e, float n, float m)
{
    // User's custom Z-axis transform: f(z) = Σ z³ * y^n
    // where y is parameter 'm' and n is the iteration count
    // Summation from n=1 to 5 terms
    float result = 0.0f;
    const int max_terms = 5;
    
    for (int i = 1; i <= max_terms; ++i) {
        float z_cubed = std::pow(z, 3.0f);  // z³
        float y_power = std::pow(std::abs(m), static_cast<float>(i));  // y^n
        result += z_cubed * y_power;
    }
    
    // Additional term: z^n * a/(3*b³*n)
    // Using n as exponent, m as 'a', and e as 'b'
    float a = m;
    float b = std::abs(e) + this->epsilon;  // Avoid zero
    float denominator = 3.0f * std::pow(b, 3.0f) * std::abs(n + this->epsilon);
    
    if (std::abs(denominator) < this->epsilon) {
        denominator = this->epsilon;
    }
    
    float extra_term = std::pow(z, n) * a / denominator;
    result += extra_term;
    
    // Clamp result to prevent numerical instability
    if (std::isnan(result) || std::isinf(result)) {
        return this->epsilon;
    }
    return result;
}

float CoreAI::xTransform2_derivative_m(float x, float e_arg, float n, float m)
{
    // Derivative of custom X-axis transform with respect to m
    // d/dm[Σ(x³ⁿ + z)/(3^(n²)) + x^m/Ly * e^(e^m)]
    
    float derivative = 0.0f;
    
    // Derivative of the exponential term: x^m/Ly * e^(e^m)
    // d/dm[x^m * e^(e^m)] = x^m * ln(x) * e^(e^m) + x^m * e^(e^m) * e^m * ln(e)
    float ly = std::abs(m) + 1.0f;
    if (x > 0) {
        float x_m = std::pow(x, m);
        float exp_em = std::exp(std::pow(e_arg, m));
        float ln_x = std::log(x);
        
        // Simplified derivative approximation for stability
        derivative = x_m * ln_x * exp_em / ly;
    }
    
    // Clamp to prevent numerical instability
    if (std::isnan(derivative) || std::isinf(derivative)) {
        return this->epsilon;
    }
    
    // Limit gradient magnitude
    if (derivative > 10.0f) derivative = 10.0f;
    if (derivative < -10.0f) derivative = -10.0f;
    
    return derivative;
}

float CoreAI::yTransform2_derivative_m(float y, float e_arg, float n, float m)
{
    // Derivative of custom Y-axis transform with respect to m
    // d/dm[Σ(y^(z+x^n))/(x³ N)]
    
    float derivative = 0.0f;
    const int max_terms = 5;
    
    for (int i = 1; i <= max_terms; ++i) {
        float exponent = n + std::pow(m, static_cast<float>(i));
        float denominator = std::pow(std::abs(m), 3.0f) * i;
        
        if (std::abs(denominator) > this->epsilon && std::abs(y) > this->epsilon) {
            // Derivative of y^(n+m^i) / (m³*i) with respect to m
            float y_power = std::pow(std::abs(y), exponent);
            float ln_y = std::log(std::abs(y) + this->epsilon);
            
            // Chain rule: derivative of exponent * y^exponent * ln(y) / denominator
            derivative += i * std::pow(m, i - 1.0f) * y_power * ln_y / denominator;
            
            // Also account for derivative of denominator
            derivative -= 3.0f * y_power / (std::pow(m, 4.0f) * i + this->epsilon);
        }
    }
    
    // Clamp to prevent numerical instability
    if (std::isnan(derivative) || std::isinf(derivative)) {
        return this->epsilon;
    }
    
    // Limit gradient magnitude
    if (derivative > 10.0f) derivative = 10.0f;
    if (derivative < -10.0f) derivative = -10.0f;
    
    return derivative;
}

float CoreAI::zTransform2_derivative_m(float z, float e_arg, float n, float m)
{
    // Derivative of custom Z-axis transform with respect to m
    // d/dm[Σ z³ * y^n + z^n * a/(3*b³*n)]
    
    float derivative = 0.0f;
    const int max_terms = 5;
    
    // Derivative of Σ z³ * m^n with respect to m
    for (int i = 1; i <= max_terms; ++i) {
        float z_cubed = std::pow(z, 3.0f);
        // d/dm[m^n] = n * m^(n-1)
        derivative += z_cubed * i * std::pow(std::abs(m), i - 1.0f);
    }
    
    // Derivative of z^n * m/(3*e³*n) with respect to m is: z^n/(3*e³*n)
    float b = std::abs(e_arg) + this->epsilon;
    float denominator = 3.0f * std::pow(b, 3.0f) * std::abs(n + this->epsilon);
    
    if (std::abs(denominator) > this->epsilon) {
        derivative += std::pow(z, n) / denominator;
    }
    
    // Clamp to prevent numerical instability
    if (std::isnan(derivative) || std::isinf(derivative)) {
        return this->epsilon;
    }
    
    // Limit gradient magnitude
    if (derivative > 10.0f) derivative = 10.0f;
    if (derivative < -10.0f) derivative = -10.0f;
    
    return derivative;
}

void CoreAI::setInputSize(int size)
{
    this->inputSize = size;
    this->resizeWeights();
}

void CoreAI::setOutputSize(int size)
{
    this->outputSize = size;
    this->resizeWeights();
}

void CoreAI::resizeWeights()
{
    // Use random seed for weight re-initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if (verbose) {
        std::cout << "[DEBUG RESIZE] Resizing and re-initializing weights" << std::endl;
    }
    
    // Resize weight matrices with Xavier/Glorot initialization
    float scale = std::sqrt(2.0f / (inputSize + neurons));
    std::uniform_real_distribution<float> weight_dist(-scale, scale);

    // Resize and initialize hidden-to-output weights with explicit random values
    initializeMatrix(this->weigth_output_hidden, outputSize, neurons);
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < neurons; ++j) {
            this->weigth_output_hidden[i][j] = weight_dist(gen);
        }
    }

    // Resize and initialize input-to-hidden weights with explicit random values
    initializeMatrix(this->weigth_input_hidden, neurons, inputSize);
    for (int i = 0; i < neurons; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            this->weigth_input_hidden[i][j] = weight_dist(gen);
        }
    }
    
    if (verbose) {
        std::cout << "[DEBUG RESIZE] Weight re-initialization complete" << std::endl;
    }
}

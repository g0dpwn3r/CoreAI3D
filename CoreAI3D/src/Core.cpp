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

    // Initialize weights and biases with better distribution for learning
    std::mt19937 gen(42);
    // Use Xavier/Glorot initialization for better gradient flow
    float scale = std::sqrt(2.0f / (inputSize + neurons));
    std::uniform_real_distribution<float> weight_dist(-scale, scale);

    // Initialize vectors with small random values
    initializeVector(this->hidden_error, neurons);
    fillVectorWithTransform(this->hidden_error, [&]() { return weight_dist(gen); });

    initializeVector(this->hidden_output, neurons);
    fillVectorWithTransform(this->hidden_output, [&]() { return weight_dist(gen); });

    // Initialize weight matrices with uniform random distribution between -0.1 and 0.1
    initializeMatrix(this->weigth_output_hidden, outputSize, neurons);
    fillMatrixWithTransform(this->weigth_output_hidden, [&]() { return weight_dist(gen); });

    initializeMatrix(this->weigth_input_hidden, neurons, inputSize);
    fillMatrixWithTransform(this->weigth_input_hidden, [&]() { return weight_dist(gen); });

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

        // Input to hidden layer - use transform functions with proper weight multiplication
        for (int j = 0; j < this->neurons; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < this->input[i].size(); ++k)
            {
                if (safe2DIndex(this->weigth_input_hidden, j, k))
                {
                    // Apply transform functions to input values and multiply by weights
                    float input_val = this->input[i][k];
                    float weight_val = this->weigth_input_hidden[j][k];
                    
                    // Transform the input value and multiply by weight
                    float transformed_val = 0.0f;
                    if (this->input[i].size() == 1) {
                        transformed_val = this->xTransform2(input_val, this->epsilon, this->neurons, 1.0f);
                    } else if (this->input[i].size() >= 3) {
                        if (k % 3 == 0) {
                            transformed_val = this->xTransform2(input_val, this->epsilon, this->neurons, 1.0f);
                        } else if (k % 3 == 1) {
                            transformed_val = this->yTransform2(input_val, this->epsilon, this->neurons, 1.0f);
                        } else {
                            transformed_val = this->zTransform2(input_val, this->epsilon, this->neurons, 1.0f);
                        }
                    } else {
                        transformed_val = this->xTransform2(input_val, this->epsilon, this->neurons, 1.0f);
                    }
                    
                    // Multiply transformed value by weight and add to sum
                    sum += transformed_val * weight_val;
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

        // Hidden to output layer - use transform functions with proper weight multiplication
        for (int j = 0; j < this->outputSize; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < this->neurons; ++k)
            {
                if (safe2DIndex(this->weigth_output_hidden, j, k))
                {
                    // Apply transform functions to hidden activations and multiply by weights
                    float hidden_val = hiddenLayer[k];
                    float weight_val = this->weigth_output_hidden[j][k];
                    
                    // Transform the hidden activation and multiply by weight
                    float transformed_val = 0.0f;
                    if (k % 3 == 0) {
                        transformed_val = this->xTransform2(hidden_val, this->epsilon, this->neurons, 1.0f);
                    } else if (k % 3 == 1) {
                        transformed_val = this->yTransform2(hidden_val, this->epsilon, this->neurons, 1.0f);
                    } else {
                        transformed_val = this->zTransform2(hidden_val, this->epsilon, this->neurons, 1.0f);
                    }
                    
                    // Multiply transformed value by weight and add to sum
                    sum += transformed_val * weight_val;
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
                // Backpropagation with transform derivatives
                float deriv = 0.0f;
                if (k % 3 == 0) {
                    deriv = xTransform2_derivative_m(hidden[sample][k], epsilon, neurons, 1.0f);
                } else if (k % 3 == 1) {
                    deriv = yTransform2_derivative_m(hidden[sample][k], epsilon, neurons, 1.0f);
                } else {
                    deriv = zTransform2_derivative_m(hidden[sample][k], epsilon, neurons, 1.0f);
                }
                // Multiply by the transformed value for proper derivative calculation
                deriv *= hidden[sample][k]; // Using the hidden activation as the transformed value
                // Add a minimum derivative value to prevent vanishing gradients
                if (std::abs(deriv) < 1e-6f) {
                    deriv = (deriv >= 0) ? 1e-6f : -1e-6f;
                }
                
                // Gradient clipping
                float gradient = output_errors[j] * deriv;
                if (gradient > 1.0f) gradient = 1.0f;
                if (gradient < -1.0f) gradient = -1.0f;

                this->weigth_output_hidden[j][k] += learningRate * gradient;
            }
        }

        // Update weights input to hidden
        for (int j = 0; j < this->neurons; ++j)
        {
            for (size_t k = 0; k < normalized_inputs[sample].size(); ++k)
            {
                if(!safe2DIndex(this->weigth_input_hidden, j, k)) {
                    throw std::out_of_range("CoreAI::train: Index out of range during weight update");
                }
                // Backpropagation with transform derivatives
                float deriv = 0.0f;
                // Use transform logic based on input dimensionality
                if (normalized_inputs[sample].size() == 1) {
                    // 1D input - use xTransform2 derivative for all
                    deriv = xTransform2_derivative_m(normalized_inputs[sample][k], epsilon, neurons, 1.0f);
                } else if (normalized_inputs[sample].size() >= 3) {
                    // 3D input - use appropriate transform derivative based on axis
                    if (k % 3 == 0) {
                        deriv = xTransform2_derivative_m(normalized_inputs[sample][k], epsilon, neurons, 1.0f);
                    } else if (k % 3 == 1) {
                        deriv = yTransform2_derivative_m(normalized_inputs[sample][k], epsilon, neurons, 1.0f);
                    } else {
                        deriv = zTransform2_derivative_m(normalized_inputs[sample][k], epsilon, neurons, 1.0f);
                    }
                } else {
                    // Other dimensionalities - use xTransform2 derivative as default
                    deriv = xTransform2_derivative_m(normalized_inputs[sample][k], epsilon, neurons, 1.0f);
                }
                // Multiply by the transformed value for proper derivative calculation
                deriv *= normalized_inputs[sample][k]; // Using the input value as the transformed value
                // Add a minimum derivative value to prevent vanishing gradients
                if (std::abs(deriv) < 1e-6f) {
                    deriv = (deriv >= 0) ? 1e-6f : -1e-6f;
                }

                // Gradient clipping
                float gradient = hidden_errors[j] * deriv;
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
    // Improved transform function that produces meaningful values
    // Use a combination of linear and non-linear terms for better learning
    // Added scaling factor 0.1 to prevent saturation
    float linear_term = x * m;
    float non_linear_term = std::pow(x, 2.0f) * n / 10.0f;
    float result = (linear_term + non_linear_term + e) * 0.1f;
    return (std::isnan(result) || std::isinf(result)) ? 0.1f : result;
}

float CoreAI::yTransform2(float y, float e, float n, float m)
{
    // Improved transform function that produces meaningful values
    // Added scaling factor 0.1 to prevent saturation
    float linear_term = y * m;
    float non_linear_term = std::pow(y, 2.0f) * n / 10.0f;
    float result = (linear_term + non_linear_term + e) * 0.1f;
    return (std::isnan(result) || std::isinf(result)) ? 0.1f : result;
}

float CoreAI::zTransform2(float z, float e, float n, float m)
{
    // Improved transform function that produces meaningful values
    // Added scaling factor 0.1 to prevent saturation
    float linear_term = z * m;
    float non_linear_term = std::pow(z, 2.0f) * n / 10.0f;
    float result = (linear_term + non_linear_term + e) * 0.1f;
    return (std::isnan(result) || std::isinf(result)) ? 0.1f : result;
}

float CoreAI::xTransform2_derivative_m(float x, float e_arg, float n, float m)
{
    // Simple derivative for the improved transform function
    // Derivative of (x*m + x^2*n/10 + e) * 0.1 with respect to m is x * 0.1
    float result = x * 0.1f;
    return (std::isnan(result) || std::isinf(result)) ? 0.1f : result;
}

float CoreAI::yTransform2_derivative_m(float y, float e_arg, float n, float m)
{
    // Simple derivative for the improved transform function
    float result = y * 0.1f;
    return (std::isnan(result) || std::isinf(result)) ? 0.1f : result;
}

float CoreAI::zTransform2_derivative_m(float z, float e_arg, float n, float m)
{
    // Simple derivative for the improved transform function
    float result = z * 0.1f;
    return (std::isnan(result) || std::isinf(result)) ? 0.1f : result;
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
    std::mt19937 gen(42);
    
    // Resize weight matrices with better initialization
    float scale = std::sqrt(2.0f / (inputSize + neurons));
    std::uniform_real_distribution<float> weight_dist(-scale, scale);

    initializeMatrix(this->weigth_output_hidden, outputSize, neurons);
    fillMatrixWithTransform(this->weigth_output_hidden, [&]() { return weight_dist(gen); });

    initializeMatrix(this->weigth_input_hidden, neurons, inputSize);
    fillMatrixWithTransform(this->weigth_input_hidden, [&]() { return weight_dist(gen); });
}

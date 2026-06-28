#include "Core.hpp"

CoreAI::CoreAI()
    : inputSize(0), layers(0), neurons(0), outputSize(0),
    minVal(0.0f), maxVal(1.0f), verbose(false),
    x(1.0f), y(1.0f), z(1.0f)  // Initialize transform variables to meaningful values
{
    // Default initialization of advanced 3D transform parameters
    this->enable_polynomial = false;
    this->enable_trigonometric = false;
    this->enable_log_exp = false;
    this->enable_rbf = false;
    this->polynomial_weight = 0.0f;
    this->trig_weight = 0.0f;
    this->log_exp_weight = 0.0f;
    this->rbf_weight = 0.0f;
    this->rbf_center = {0.0f, 0.0f, 0.0f};
    this->rbf_sigma = 1.0f;

    // Initialize new 3D feature engineering parameters
    this->enable_angular = false;
    this->enable_projections = false;
    this->enable_ratios = false;
    this->enable_transformations = false;
    this->angular_weight = 0.0f;
    this->projections_weight = 0.0f;
    this->ratios_weight = 0.0f;
    this->transformations_weight = 0.0f;

    // Initialize transformation parameters
    this->rotation_angle_x = 0.0f;
    this->rotation_angle_y = 0.0f;
    this->rotation_angle_z = 0.0f;
    this->scale_x = 1.0f;
    this->scale_y = 1.0f;
    this->scale_z = 1.0f;
    this->translation_x = 0.0f;
    this->translation_y = 0.0f;
    this->translation_z = 0.0f;

    if (verbose) {
        std::cout << "[DEBUG CONSTRUCTOR] Starting CoreAI default constructor" << std::endl;
    }
}

CoreAI::CoreAI(int inputSize_, int layers_, int neurons_, int outputSize_,
    float min_, float max_, bool verbose_)
    : inputSize(inputSize_), layers(layers_), neurons(neurons_),
    outputSize(outputSize_), minVal(min_), maxVal(max_), verbose(verbose_),
    x(1.0f), y(1.0f), z(1.0f)  // Initialize transform variables to meaningful values
{
    // Initialize advanced 3D transform parameters - DISABLE COMPLEX TRANSFORMS FOR DEBUGGING
    this->enable_polynomial = false;
    this->enable_trigonometric = false;
    this->enable_log_exp = false;
    this->enable_rbf = false;
    this->polynomial_weight = 0.0f;
    this->trig_weight = 0.0f;
    this->log_exp_weight = 0.0f;
    this->rbf_weight = 0.0f;
    this->rbf_center = {0.0f, 0.0f, 0.0f};
    this->rbf_sigma = 1.0f;

    // Initialize new 3D feature engineering parameters - DISABLE COMPLEX TRANSFORMS FOR DEBUGGING
    this->enable_angular = false;
    this->enable_projections = false;
    this->enable_ratios = false;
    this->enable_transformations = false;
    this->angular_weight = 0.0f;
    this->projections_weight = 0.0f;
    this->ratios_weight = 0.0f;
    this->transformations_weight = 0.0f;

    // Initialize transformation parameters
    this->rotation_angle_x = 0.0f;
    this->rotation_angle_y = 0.0f;
    this->rotation_angle_z = 0.0f;
    this->scale_x = 1.0f;
    this->scale_y = 1.0f;
    this->scale_z = 1.0f;
    this->translation_x = 0.0f;
    this->translation_y = 0.0f;
    this->translation_z = 0.0f;

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

    // Initialize random number generator with fixed seed for reproducible results
    std::mt19937 gen(42);  // Use fixed seed for reproducible results
    
    if (verbose) {
        std::cout << "[DEBUG CONSTRUCTOR] Initializing weights with random values" << std::endl;
    }

    // Initialize vectors (these should be zero, not random)
    initializeVector(this->hidden_error, neurons);
    std::fill(this->hidden_error.begin(), this->hidden_error.end(), 0.0f);

    initializeVector(this->hidden_output, neurons);
    std::fill(this->hidden_output.begin(), this->hidden_output.end(), 0.0f);

    // Initialize weight matrices with adaptive initialization
    // Use Xavier/Glorot initialization by default for backward compatibility
    initializeMatrix(this->weigth_output_hidden, outputSize, neurons);
    initializeWeightsAdaptive(neurons, outputSize, this->weigth_output_hidden, ActivationType::SIGMOID);

    initializeMatrix(this->weigth_input_hidden, neurons, inputSize);
    initializeWeightsAdaptive(inputSize, neurons, this->weigth_input_hidden, ActivationType::SIGMOID);

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
        std::cout << "[DEBUG CONSTRUCTOR] Weight range: [" << -this->weight_init_scale << ", " << this->weight_init_scale << "]" << std::endl;
    }

    // Initialize trainer pointer to nullptr
    this->trainer = nullptr;

    // Initialize optimizer and scheduler to nullptr (will be set via configuration)
    this->optimizer = nullptr;
    this->lrScheduler = nullptr;
    this->currentEpoch = 0;

    // Initialize regularization and stability parameters with defaults
    this->l2_lambda = 0.01f;
    this->dropout_rate = 0.2f;
    this->bn_momentum = 0.1f;
    this->grad_clip_threshold = 1.0f;

    this->use_dropout = true;
    this->use_batch_norm = true;
    this->use_l2_reg = true;
    this->use_grad_clip = true;

    // Initialize batch normalization parameters
    initializeVector(this->bn_running_mean, neurons);
    std::fill(this->bn_running_mean.begin(), this->bn_running_mean.end(), 0.0f);
    initializeVector(this->bn_running_var, neurons);
    std::fill(this->bn_running_var.begin(), this->bn_running_var.end(), 1.0f);
    initializeVector(this->bn_gamma, neurons);
    std::fill(this->bn_gamma.begin(), this->bn_gamma.end(), 1.0f);
    initializeVector(this->bn_beta, neurons);
    // Initialize multi-layer architecture (default to single hidden layer for backward compatibility)
    this->layer_sizes = {inputSize_, neurons_, outputSize_}; // input, hidden, output
    this->enable_residual = false;
    this->enable_attention = false;
    this->attention_temperature = 1.0f;
    this->attention_weight = 1.0f;

    // Initialize layer weights for multi-layer network
    initializeMultiLayerWeights();

    // Initialize attention weights if enabled
    if (enable_attention) {
        initializeAttentionWeights();
    }
    std::fill(this->bn_beta.begin(), this->bn_beta.end(), 0.0f);

    // Initialize activation configurations (default to sigmoid for backward compatibility)
    ActivationConfig default_config;
    default_config.type = ActivationType::SIGMOID;
    default_config.alpha = 1.0f;
    default_config.beta = 0.1f;
    default_config.spatial_weight = 1.0f;
    default_config.calculus_config = {10, 1.0f, 1.0f};
    layer_activation_configs.resize(layer_sizes.size(), default_config);
    std::fill(this->bn_beta.begin(), this->bn_beta.end(), 0.0f);

    // Initialize weight initialization parameters
    weight_init_type = WeightInitialization::XAVIER_UNIFORM; // Default to Xavier uniform
    weight_init_scale = 1.0f;
    weight_init_seed = 42;
    std::fill(this->bn_beta.begin(), this->bn_beta.end(), 0.0f);
    
    // DEBUG: Verify weight initialization parameters
    if (verbose) {
        std::cout << "[DEBUG CONSTRUCTOR] Weight initialization: type=" << static_cast<int>(weight_init_type)
                  << ", scale=" << weight_init_scale << ", seed=" << weight_init_seed << std::endl;
    }

    // Initialize thread pool with number of hardware threads
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // fallback
    threadPool = std::make_unique<ThreadPool>(num_threads);

    // Initialize gradient accumulation parameters
    this->enable_gradient_accumulation = false;
    this->gradient_accumulation_steps = 4;
    this->max_historical_gradients = 100;
    this->current_accumulation_step = 0;

    // Initialize accumulated gradients matrices (will be resized when needed)
    initializeMatrix(this->accumulated_output_hidden_gradients, outputSize, neurons);
    std::fill(accumulated_output_hidden_gradients.begin(), accumulated_output_hidden_gradients.end(),
              std::vector<float>(neurons, 0.0f));
    initializeMatrix(this->accumulated_input_hidden_gradients, neurons, inputSize);
    std::fill(accumulated_input_hidden_gradients.begin(), accumulated_input_hidden_gradients.end(),
              std::vector<float>(inputSize, 0.0f));
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

// Modern activation functions

float CoreAI::elu(float x, float alpha)
{
    return x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f);
}

float CoreAI::elu_derivative(float x, float alpha)
{
    return x >= 0.0f ? 1.0f : alpha * std::exp(x);
}

float CoreAI::leaky_relu(float x, float alpha)
{
    return x >= 0.0f ? x : alpha * x;
}

float CoreAI::leaky_relu_derivative(float x, float alpha)
{
    return x >= 0.0f ? 1.0f : alpha;
}

float CoreAI::parametric_relu(float x, float beta)
{
    return x >= 0.0f ? x : beta * x;
}

float CoreAI::parametric_relu_derivative(float x, float beta)
{
    return x >= 0.0f ? 1.0f : beta;
}

float CoreAI::swish(float x)
{
    return x * sigmoid(x);
}

float CoreAI::swish_derivative(float x)
{
    float s = sigmoid(x);
    return s + x * s * (1.0f - s);
}

float CoreAI::gelu(float x)
{
    // GELU approximation: x * Φ(x) where Φ(x) is the CDF of standard normal
    // Using tanh approximation for numerical stability
    float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    float x_cubed = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x_cubed)));
}

float CoreAI::gelu_derivative(float x)
{
    // Derivative of GELU approximation
    float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    float sech_sq = 1.0f / (std::cosh(tanh_arg) * std::cosh(tanh_arg));

    float phi_prime = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x) * sech_sq;
    float phi = 0.5f * (1.0f + std::tanh(tanh_arg));

    return 0.5f * (phi + x * phi_prime);
}

float CoreAI::spatial_elu(float x, float distance, float alpha, float spatial_weight)
{
    // Spatial-aware ELU: combines standard ELU with distance-based modulation
    float base_activation = elu(x, alpha);
    float spatial_factor = 1.0f + spatial_weight * std::exp(-distance * distance);
    return base_activation * spatial_factor;
}

float CoreAI::spatial_elu_derivative(float x, float distance, float alpha, float spatial_weight)
{
    // Derivative of spatial ELU
    float base_derivative = elu_derivative(x, alpha);
    float spatial_factor = 1.0f + spatial_weight * std::exp(-distance * distance);
    float spatial_factor_derivative = -2.0f * distance * spatial_weight * std::exp(-distance * distance);

    // Chain rule: d/dx[elu(x) * f(distance)] = elu'(x) * f(distance) + elu(x) * f'(distance)
    // But f'(distance) depends on distance, not x, so second term is 0 for derivative w.r.t. x
    return base_derivative * spatial_factor;
}

float CoreAI::spatial_leaky_relu(float x, float distance, float alpha, float spatial_weight)
{
    // Spatial-aware Leaky ReLU: combines standard Leaky ReLU with distance-based modulation
    float base_activation = leaky_relu(x, alpha);
    float spatial_factor = 1.0f + spatial_weight * (1.0f / (1.0f + distance));
    return base_activation * spatial_factor;
}

float CoreAI::spatial_leaky_relu_derivative(float x, float distance, float alpha, float spatial_weight)
{
    // Derivative of spatial Leaky ReLU
    float base_derivative = leaky_relu_derivative(x, alpha);
    float spatial_factor = 1.0f + spatial_weight * (1.0f / (1.0f + distance));
    return base_derivative * spatial_factor;
}

// Calculus-based activation functions using numerical integration

float CoreAI::simpsons_rule(float a, float b, int n, std::function<float(float)> f)
{
    // Simpson's rule for numerical integration with numerical stability checks
    if (n <= 0) n = 2; // Minimum 2 intervals
    if (n > 1000) n = 1000; // Maximum to prevent excessive computation

    // Ensures n is even
    if (n % 2 != 0) n++;

    // Handle edge case where a == b
    if (std::abs(b - a) < std::numeric_limits<float>::epsilon()) {
        return 0.0f;
    }

    float h = (b - a) / n;
    float sum = f(a) + f(b);

    // Sum for odd indices (4x multiplier)
    for (int i = 1; i < n; i += 2) {
        float x = a + i * h;
        // Clamp x to prevent overflow in function evaluation
        x = std::max(-100.0f, std::min(100.0f, x));
        sum += 4.0f * f(x);
    }

    // Sum for even indices (2x multiplier)
    for (int i = 2; i < n; i += 2) {
        float x = a + i * h;
        // Clamp x to prevent overflow in function evaluation
        x = std::max(-100.0f, std::min(100.0f, x));
        sum += 2.0f * f(x);
    }

    float result = (h / 3.0f) * sum;
    // Clamp result to prevent extreme values
    return std::max(-1e6f, std::min(1e6f, result));
}

float CoreAI::integral_sigmoid(float x, const CalculusConfig& config)
{
    // Integral of sigmoid from 0 to x using Simpson's rule
    // ∫ sigmoid(t) dt from 0 to x
    auto sigmoid_func = [this](float t) { return sigmoid(t); };

    // Clamp x to reasonable range for numerical stability
    float clamped_x = std::max(-10.0f, std::min(10.0f, x));

    float integral = simpsons_rule(0.0f, clamped_x, config.N, sigmoid_func);
    float result = config.a * config.SM * integral;

    // Additional clamping for final result
    return std::max(-1e3f, std::min(1e3f, result));
}

float CoreAI::integral_sigmoid_derivative(float x, const CalculusConfig& config)
{
    // Derivative of integral_sigmoid is just sigmoid(x) scaled
    return config.a * config.SM * sigmoid(x);
}

float CoreAI::integral_relu(float x, const CalculusConfig& config)
{
    // Integral of ReLU from 0 to x using Simpson's rule
    // ∫ max(0, t) dt from 0 to x
    auto relu_func = [](float t) { return std::max(0.0f, t); };

    // Clamp x to reasonable range for numerical stability
    float clamped_x = std::max(-10.0f, std::min(10.0f, x));

    float integral = simpsons_rule(0.0f, clamped_x, config.N, relu_func);
    float result = config.a * config.SM * integral;

    // Additional clamping for final result
    return std::max(-1e3f, std::min(1e3f, result));
}

float CoreAI::integral_relu_derivative(float x, const CalculusConfig& config)
{
    // Derivative of integral_relu is just ReLU(x) scaled
    return config.a * config.SM * std::max(0.0f, x);
}

float CoreAI::integral_tanh(float x, const CalculusConfig& config)
{
    // Integral of tanh from 0 to x using Simpson's rule
    // ∫ tanh(t) dt from 0 to x
    auto tanh_func = [](float t) { return std::tanh(t); };

    // Clamp x to reasonable range for numerical stability
    float clamped_x = std::max(-10.0f, std::min(10.0f, x));

    float integral = simpsons_rule(0.0f, clamped_x, config.N, tanh_func);
    float result = config.a * config.SM * integral;

    // Additional clamping for final result
    return std::max(-1e3f, std::min(1e3f, result));
}

float CoreAI::integral_tanh_derivative(float x, const CalculusConfig& config)
{
    // Derivative of integral_tanh is just tanh(x) scaled
    return config.a * config.SM * std::tanh(x);
}

// Unified activation function interface
float CoreAI::activate(float x, const ActivationConfig& config, float distance)
{
    switch (config.type) {
        case ActivationType::SIGMOID:
            return sigmoid(x);
        case ActivationType::ELU:
            return elu(x, config.alpha);
        case ActivationType::LEAKY_RELU:
            return leaky_relu(x, config.alpha);
        case ActivationType::PARAMETRIC_RELU:
            return parametric_relu(x, config.beta);
        case ActivationType::SWISH:
            return swish(x);
        case ActivationType::GELU:
            return gelu(x);
        case ActivationType::SPATIAL_ELU:
            return spatial_elu(x, distance, config.alpha, config.spatial_weight);
        case ActivationType::SPATIAL_LEAKY_RELU:
            return spatial_leaky_relu(x, distance, config.alpha, config.spatial_weight);
        case ActivationType::INTEGRAL_SIGMOID:
            return integral_sigmoid(x, config.calculus_config);
        case ActivationType::INTEGRAL_RELU:
            return integral_relu(x, config.calculus_config);
        case ActivationType::INTEGRAL_TANH:
            return integral_tanh(x, config.calculus_config);
        default:
            return sigmoid(x); // fallback
    }
}

float CoreAI::activate_derivative(float x, const ActivationConfig& config, float distance)
{
    switch (config.type) {
        case ActivationType::SIGMOID:
            return x * (1.0f - x); // sigmoid derivative
        case ActivationType::ELU:
            return elu_derivative(x, config.alpha);
        case ActivationType::LEAKY_RELU:
            return leaky_relu_derivative(x, config.alpha);
        case ActivationType::PARAMETRIC_RELU:
            return parametric_relu_derivative(x, config.beta);
        case ActivationType::SWISH:
            return swish_derivative(x);
        case ActivationType::GELU:
            return gelu_derivative(x);
        case ActivationType::SPATIAL_ELU:
            return spatial_elu_derivative(x, distance, config.alpha, config.spatial_weight);
        case ActivationType::SPATIAL_LEAKY_RELU:
            return spatial_leaky_relu_derivative(x, distance, config.alpha, config.spatial_weight);
        case ActivationType::INTEGRAL_SIGMOID:
            return integral_sigmoid_derivative(x, config.calculus_config);
        case ActivationType::INTEGRAL_RELU:
            return integral_relu_derivative(x, config.calculus_config);
        case ActivationType::INTEGRAL_TANH:
            return integral_tanh_derivative(x, config.calculus_config);
        default:
            return x * (1.0f - x); // sigmoid derivative fallback
    }
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

std::vector<std::vector<float>> CoreAI::forward(const std::vector<std::vector<float> >& inputvalue, bool training)
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
    this->dropout_masks.clear();

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

    // DEBUG: Check RBF parameters
    if (verbose) {
        std::cout << "[DEBUG FORWARD] RBF enabled: " << (enable_rbf ? "true" : "false")
                  << ", rbf_center size: " << rbf_center.size()
                  << ", rbf_weight: " << rbf_weight
                  << ", rbf_sigma: " << rbf_sigma << std::endl;
    }

    // First pass: compute all hidden layer activations before batch normalization
    std::vector<std::vector<float>> raw_hidden_activations;
    for (size_t i = 0; i < this->input.size(); ++i)
    {
        std::vector<float> hiddenLayer(this->neurons, 0.0f);

// Input to hidden layer computation
         for (int j = 0; j < this->neurons; ++j)
         {
             float sum = 0.0f;
             float last_distance = 0.0f;
             for (size_t k = 0; k < this->input[i].size(); ++k)
             {
                 if (safe2DIndex(this->weigth_input_hidden, j, k))
                 {
                     float input_val = this->input[i][k];
                     float weight_val = this->weigth_input_hidden[j][k];

                     // Only use 3D triplet coordinates when at least one 3D transform is enabled.
                     // For sliding window time series (inputSize % 3 == 0 but no 3D intent),
                     // use the direct input value without coordinate transformations.
                     if (enable_transformations || enable_polynomial || enable_trigonometric ||
                         enable_log_exp || enable_rbf || enable_angular || 
                         enable_projections || enable_ratios) {
                         // Get triplet coordinates for spatial context
                         size_t triplet_base = (k / 3) * 3;
                         float x = (triplet_base < this->input[i].size()) ? this->input[i][triplet_base] : 0.0f;
                         float y = (triplet_base + 1 < this->input[i].size()) ? this->input[i][triplet_base + 1] : 0.0f;
                         float z = (triplet_base + 2 < this->input[i].size()) ? this->input[i][triplet_base + 2] : 0.0f;

                         // Apply coordinate transformations if enabled
                         if (enable_transformations) {
                             // Apply scaling
                             x = x * scale_x;
                             y = y * scale_y;
                             z = z * scale_z;

                             // Apply translation
                             x += translation_x;
                             y += translation_y;
                             z += translation_z;

                             // Apply rotations (order: Z, Y, X)
                             // Rotation around Z-axis
                             float cos_z = std::cos(rotation_angle_z);
                             float sin_z = std::sin(rotation_angle_z);
                             float temp_x = x * cos_z - y * sin_z;
                             float temp_y = x * sin_z + y * cos_z;
                             x = temp_x;
                             y = temp_y;

                             // Rotation around Y-axis
                             float cos_y = std::cos(rotation_angle_y);
                             float sin_y = std::sin(rotation_angle_y);
                             float temp_x2 = x * cos_y + z * sin_y;
                             float temp_z = -x * sin_y + z * cos_y;
                             x = temp_x2;
                             z = temp_z;

                             // Rotation around X-axis
                             float cos_x = std::cos(rotation_angle_x);
                             float sin_x = std::sin(rotation_angle_x);
                             float temp_y2 = y * cos_x - z * sin_x;
                             float temp_z2 = y * sin_x + z * cos_x;
                             y = temp_y2;
                             z = temp_z2;
                         }

                         // Compute spatial features
                         last_distance = std::sqrt(x*x + y*y + z*z + this->epsilon);

                         // Apply advanced 3D transformations
                         float transformed = 0.0f;
                         int coord_type = k % 3;

                         // Original linear transforms
                         if (coord_type == 0) {
                             transformed += x * weight_val;
                         } else if (coord_type == 1) {
                             transformed += y * weight_val;
                         } else {
                             transformed += z * weight_val;
                         }

                         // Distance weighting
                         transformed += last_distance * 0.1f;

                         // Polynomial transforms
                         if (enable_polynomial) {
                             if (coord_type == 0) {
                                 transformed += polynomial_weight * (x*x + x*y + x*z);
                             } else if (coord_type == 1) {
                                 transformed += polynomial_weight * (y*y + x*y + y*z);
                             } else {
                                 transformed += polynomial_weight * (z*z + x*z + y*z);
                             }
                         }

                         // Trigonometric transforms
                         if (enable_trigonometric) {
                             if (coord_type == 0) {
                                 transformed += trig_weight * std::sin(x);
                             } else if (coord_type == 1) {
                                 transformed += trig_weight * std::cos(y);
                             } else {
                                 float z_clamped = std::max(-std::acos(-1.0f)/2 + 0.01f, std::min(std::acos(-1.0f)/2 - 0.01f, z));
                                 transformed += trig_weight * std::tan(z_clamped);
                             }
                         }

                         // Logarithmic/exponential transforms
                         if (enable_log_exp) {
                             if (coord_type == 0) {
                                 transformed += log_exp_weight * std::log(1.0f + std::abs(x));
                             } else if (coord_type == 1) {
                                 transformed += log_exp_weight * std::exp(-last_distance);
                             } else {
                                 transformed += log_exp_weight * std::log(1.0f + std::abs(z));
                             }
                         }

                         // Radial basis function
                         if (enable_rbf) {
                             float dx = x - rbf_center[0];
                             float dy = y - rbf_center[1];
                             float dz = z - rbf_center[2];
                             float dist_sq = dx*dx + dy*dy + dz*dz;
                             transformed += rbf_weight * std::exp(-dist_sq / (rbf_sigma * rbf_sigma));
                         }

                         // Angular features (spherical coordinates θ, φ)
                         if (enable_angular) {
                             float r = last_distance;
                             float theta = (r > this->epsilon) ? std::acos(std::max(-1.0f, std::min(1.0f, z / r))) : 0.0f;
                             float phi = std::atan2(y, x);
                             transformed += angular_weight * theta;
                             transformed += angular_weight * phi;
                         }

                         // Projection features (plane distances)
                         if (enable_projections) {
                             float proj_xy = std::sqrt(x*x + y*y + this->epsilon);
                             float proj_xz = std::sqrt(x*x + z*z + this->epsilon);
                             float proj_yz = std::sqrt(y*y + z*z + this->epsilon);
                             transformed += projections_weight * proj_xy;
                             transformed += projections_weight * proj_xz;
                             transformed += projections_weight * proj_yz;
                         }

                         // Distance ratios
                         if (enable_ratios) {
                             float ratio_xy = (std::abs(y) > this->epsilon) ? x / y : 0.0f;
                             float ratio_yz = (std::abs(z) > this->epsilon) ? y / z : 0.0f;
                             float ratio_xz = (std::abs(z) > this->epsilon) ? x / z : 0.0f;
                             transformed += ratios_weight * ratio_xy;
                             transformed += ratios_weight * ratio_yz;
                             transformed += ratios_weight * ratio_xz;
                         }

                         sum += transformed;
                     } else {
                         // For non-3D data (sliding windows, standard regression),
                         // use the direct input value without coordinate transformations
                         sum += input_val * weight_val;
                     }
                 }
             }
             hiddenLayer[j] = this->activate(sum, layer_activation_configs[1], 0.0f);
         }
        raw_hidden_activations.push_back(hiddenLayer);
    }

    // Apply batch normalization if enabled
    if (use_batch_norm) {
        size_t batch_size = raw_hidden_activations.size();
        for (int j = 0; j < this->neurons; ++j) {
            // Compute batch mean and variance
            float batch_mean = 0.0f;
            float batch_var = 0.0f;

            for (size_t i = 0; i < batch_size; ++i) {
                batch_mean += raw_hidden_activations[i][j];
            }
            batch_mean /= batch_size;

            for (size_t i = 0; i < batch_size; ++i) {
                float diff = raw_hidden_activations[i][j] - batch_mean;
                batch_var += diff * diff;
            }
            batch_var /= batch_size;

            // Update running statistics
            if (training) {
                bn_running_mean[j] = bn_momentum * batch_mean + (1 - bn_momentum) * bn_running_mean[j];
                bn_running_var[j] = bn_momentum * batch_var + (1 - bn_momentum) * bn_running_var[j];
            }

            // Use batch stats during training, running stats during inference
            float mean = training ? batch_mean : bn_running_mean[j];
            float var = training ? batch_var : bn_running_var[j];

            // Normalize and scale
            for (size_t i = 0; i < batch_size; ++i) {
                float normalized = (raw_hidden_activations[i][j] - mean) / std::sqrt(var + epsilon);
                raw_hidden_activations[i][j] = bn_gamma[j] * normalized + bn_beta[j];
            }
        }
    }

    // Now process each sample with dropout and output computation
    for (size_t i = 0; i < this->input.size(); ++i)
    {
        std::vector<float> hiddenLayer = raw_hidden_activations[i];
        std::vector<float> outputLayer(this->outputSize, 0.0f);

        // Apply dropout to hidden layer if enabled and training
        std::vector<float> dropout_mask(neurons, 1.0f);
        if (use_dropout && training) {
            std::mt19937 gen(42 + i); // Different seed per sample
            std::bernoulli_distribution dist(1.0f - dropout_rate);
            float scale = 1.0f / (1.0f - dropout_rate);

            for (int j = 0; j < neurons; ++j) {
                dropout_mask[j] = dist(gen) ? scale : 0.0f;
                hiddenLayer[j] *= dropout_mask[j];
            }
        }

        // Store dropout mask for backpropagation
        if (training) {
            dropout_masks.push_back(dropout_mask);
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
            outputLayer[j] = this->activate(sum, layer_activation_configs.back(), 0.0f);
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
    double learningRate, int& numSamples, int batchSize)
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

// Normalize inputs and targets for 3D spatial coordinates
     std::vector<std::vector<float>> normalized_inputs = inputs;
    std::vector<std::vector<float>> normalized_targets = targets;

    // For 3D coordinates, normalize each dimension separately ONLY when at least one 3D transform is enabled.
    // The check `inputs[0].size() % 3 == 0` is insufficient to detect 3D intent - sliding windows also
    // produce divisible-by-3 sizes. We use the enable flags to determine the correct path.
    bool use_3d_transforms = enable_transformations || enable_polynomial || enable_trigonometric ||
                             enable_log_exp || enable_rbf || enable_angular || 
                             enable_projections || enable_ratios;

    if (!inputs.empty() && inputs[0].size() % 3 == 0 && use_3d_transforms) {
        // Per-dimension normalization for spatial coordinates
        for (size_t dim = 0; dim < 3; ++dim) {
            float dim_min = std::numeric_limits<float>::max();
            float dim_max = std::numeric_limits<float>::lowest();

            // Find min/max for this dimension across all samples
            for (const auto& sample : inputs) {
                for (size_t k = dim; k < sample.size(); k += 3) {
                    if (sample[k] < dim_min) dim_min = sample[k];
                    if (sample[k] > dim_max) dim_max = sample[k];
                }
            }

            // Avoid division by zero
            if (dim_min == dim_max) {
                dim_max = dim_min + 1.0f;
            }

            // Normalize this dimension in all samples
            for (auto& sample : normalized_inputs) {
                for (size_t k = dim; k < sample.size(); k += 3) {
                    sample[k] = (sample[k] - dim_min) / (dim_max - dim_min);
                }
            }
        }
    } else {
        // Fallback to global normalization if not 3D or no 3D transforms enabled
        // Check if data is already normalized to [0, 1] range to avoid double-normalization
        float input_min, input_max;
        computeMinMax(inputs, input_min, input_max);
        
        // Only normalize if data is NOT already in [0, 1] range
        // Data is considered normalized if min >= 0 and max <= 1 (with small tolerance for numerical precision)
        bool already_normalized = (input_min >= 0.0f - 1e-5f) && (input_max <= 1.0f + 1e-5f) && 
                                   (input_min >= 0.0f) && (input_max <= 1.0f);
        
        if (!already_normalized) {
            for (auto& row : normalized_inputs) {
                for (float& val : row) {
                    val = (val - input_min) / (input_max - input_min);
                }
            }
        }
    }

    // Normalize targets globally to [0,1]
    // Only normalize if data is NOT already in [0, 1] range
    float target_min, target_max;
    computeMinMax(targets, target_min, target_max);
    bool targets_already_normalized = (target_min >= 0.0f - 1e-5f) && (target_max <= 1.0f + 1e-5f) &&
                                       (target_min >= 0.0f) && (target_max <= 1.0f);
    
    if (!targets_already_normalized) {
        for (auto& row : normalized_targets) {
            for (float& val : row) {
                val = (val - target_min) / (target_max - target_min);
            }
        }
    }

    // Do NOT overwrite the trainer's original data global min/max with normalized target min/max
    // The trainer already has the correct original data range from the preprocess step
    if (verbose && trainer) {
        std::cout << "[DEBUG TRAIN] Target normalization - min: " << target_min
                  << ", max: " << target_max
                  << ", trainer original min: " << trainer->original_data_global_min
                  << ", trainer original max: " << trainer->original_data_global_max << std::endl;
    }

    // Ensure forward pass has been run
    if (this->results.empty()) {
        this->forward(normalized_inputs, true); // training = true
    }

    // Initialize gradient accumulators for batch update
    std::vector<std::vector<float>> output_hidden_gradients(this->outputSize, std::vector<float>(this->neurons, 0.0f));
    std::vector<std::vector<float>> input_hidden_gradients(this->neurons, std::vector<float>(this->inputSize, 0.0f));

    // Get current learning rate from scheduler if available
    float currentLearningRate = learningRate;
    if (lrScheduler) {
        currentLearningRate = lrScheduler->getLearningRate(currentEpoch);
        if (verbose) {
            std::cout << "[DEBUG TRAIN] Epoch " << currentEpoch << ", Learning rate: " << currentLearningRate << std::endl;
        }
    }

    // Batched training with multi-threading
    size_t total_samples = normalized_inputs.size();
    size_t num_batches = (total_samples + batchSize - 1) / batchSize; // Ceiling division

    if (verbose) {
        std::cout << "[DEBUG TRAIN] Training with batch size: " << batchSize
                  << ", total samples: " << total_samples
                  << ", number of batches: " << num_batches << std::endl;
    }

    // Process samples in batches
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        size_t batch_start = batch_idx * batchSize;
        size_t batch_end = std::min(batch_start + batchSize, total_samples);
        size_t current_batch_size = batch_end - batch_start;

        if (verbose) {
            std::cout << "[DEBUG TRAIN] Processing batch " << (batch_idx + 1) << "/" << num_batches
                      << " (samples " << batch_start << " to " << (batch_end - 1) << ")" << std::endl;
        }

        // Reset gradients for this batch
        std::fill(output_hidden_gradients.begin(), output_hidden_gradients.end(),
                  std::vector<float>(this->neurons, 0.0f));
        std::fill(input_hidden_gradients.begin(), input_hidden_gradients.end(),
                  std::vector<float>(this->inputSize, 0.0f));

        // Forward pass for the current batch
        std::vector<std::vector<float>> batch_inputs(normalized_inputs.begin() + batch_start,
                                                    normalized_inputs.begin() + batch_end);
        std::vector<std::vector<float>> batch_targets(normalized_targets.begin() + batch_start,
                                                     normalized_targets.begin() + batch_end);

        if (verbose) {
            std::cout << "[DEBUG TRAIN] Batch inputs size: " << batch_inputs.size()
                      << ", batch targets size: " << batch_targets.size() << std::endl;
        }

        auto batch_results = this->forward(batch_inputs, true); // training = true

        if (verbose) {
            std::cout << "[DEBUG TRAIN] After forward pass - hidden size: " << this->hidden.size()
                      << ", results size: " << batch_results.size() << std::endl;
        }

        // Backpropagation for the current batch
        for (size_t batch_sample = 0; batch_sample < current_batch_size; ++batch_sample) {
            size_t global_sample = batch_start + batch_sample;

            // DEBUG: Log transform function inputs
            if (verbose && global_sample == 0) {
                std::cout << "[DEBUG TRAIN] Transform function inputs - x: " << this->x
                          << ", y: " << this->y << ", z: " << this->z
                          << ", epsilon: " << this->epsilon << std::endl;
            }

            // Output layer error
            std::vector<float> output_errors(this->outputSize, 0.0f);
            for (int j = 0; j < this->outputSize; ++j) {
                if (!safe2DIndex(batch_results, batch_sample, j) || !safe2DIndex(batch_targets, batch_sample, j)) {
                    throw std::out_of_range("CoreAI::train: Index out of range during backpropagation");
                }

                float output = batch_results[batch_sample][j];
                float target = batch_targets[batch_sample][j];

                // DEBUG: Log error calculation components
                if (verbose && global_sample == 0 && j == 0) {
                    std::cout << "[DEBUG TRAIN] Sample 0, Output 0 - output: " << output
                              << ", target: " << target << std::endl;
                }

                // Standard backpropagation: error = (target - output) * derivative
                output_errors[j] = (target - output) * output * (1.0f - output); // Derivative of sigmoid
            }

            // Hidden layer error
            std::vector<float> hidden_errors(this->neurons, 0.0f);
            for (int j = 0; j < this->neurons; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < this->outputSize; ++k) {
                    if (!safe2DIndex(this->weigth_output_hidden, k, j)) {
                        throw std::out_of_range("CoreAI::train: Index out of range during hidden error calculation");
                    }
                    // Standard backpropagation: sum += output_errors[k] * weight
                    sum += output_errors[k] * this->weigth_output_hidden[k][j];
                }
                // Use batch_results to get hidden activations instead of this->hidden
                if (!safe2DIndex(this->hidden, batch_sample, j)) {
                    throw std::out_of_range("CoreAI::train: Hidden layer index out of range");
                }
                float hidden_activation = this->hidden[batch_sample][j];
                hidden_errors[j] = sum * hidden_activation * (1.0f - hidden_activation); // Derivative of sigmoid

                // DEBUG: Log hidden error calculation
                if (verbose && global_sample == 0 && j == 0) {
                    std::cout << "[DEBUG TRAIN] Sample 0, Hidden 0 - sum: " << sum
                              << ", hidden activation: " << hidden_activation
                              << ", hidden error: " << hidden_errors[j] << std::endl;
                }
            }

            // Accumulate gradients for output to hidden weights
            for (int j = 0; j < this->outputSize; ++j) {
                for (int k = 0; k < this->neurons; ++k) {
                    if(!safe2DIndex(this->weigth_output_hidden, j, k)) {
                        throw std::out_of_range("CoreAI::train: Index out of range during weight update");
                    }
                    if (!safe2DIndex(this->hidden, batch_sample, k)) {
                        throw std::out_of_range("CoreAI::train: Hidden layer index out of range in output gradients");
                    }
                    // Standard backpropagation: gradient = error * activation
                    float gradient = output_errors[j] * this->hidden[batch_sample][k];

                    // Add L2 regularization gradient: 2 * lambda * weight
                    if (use_l2_reg) {
                        gradient += 2.0f * l2_lambda * this->weigth_output_hidden[j][k];
                    }

                    // Gradient clipping to prevent exploding gradients
                    if (use_grad_clip && std::abs(gradient) > grad_clip_threshold) {
                        gradient = (gradient > 0 ? grad_clip_threshold : -grad_clip_threshold);
                    }

                    output_hidden_gradients[j][k] += gradient;
                }
            }

// Accumulate gradients for input to hidden weights with CORRECT 3D transform derivatives
             for (int j = 0; j < this->neurons; ++j) {
                 for (size_t k = 0; k < batch_inputs[batch_sample].size(); ++k) {
                     if(!safe2DIndex(this->weigth_input_hidden, j, k)) {
                         throw std::out_of_range("CoreAI::train: Index out of range during weight update");
                     }

                     bool use_3d = enable_transformations || enable_polynomial || enable_trigonometric ||
                                 enable_log_exp || enable_rbf || enable_angular || 
                                 enable_projections || enable_ratios;

                     float transform_derivative = 0.0f;

                     if (use_3d) {
                         // Get triplet coordinates for spatial context (same as forward pass)
                         size_t triplet_base = (k / 3) * 3;
                         float x = (triplet_base < batch_inputs[batch_sample].size()) ? batch_inputs[batch_sample][triplet_base] : 0.0f;
                         float y = (triplet_base + 1 < batch_inputs[batch_sample].size()) ? batch_inputs[batch_sample][triplet_base + 1] : 0.0f;
                         float z = (triplet_base + 2 < batch_inputs[batch_sample].size()) ? batch_inputs[batch_sample][triplet_base + 2] : 0.0f;

                         // Apply coordinate transformations if enabled (same as forward pass)
                         if (enable_transformations) {
                             // Apply scaling
                             x = x * scale_x;
                             y = y * scale_y;
                             z = z * scale_z;

                             // Apply translation
                             x += translation_x;
                             y += translation_y;
                             z += translation_z;

                             // Apply rotations (order: Z, Y, X)
                             // Rotation around Z-axis
                             float cos_z = std::cos(rotation_angle_z);
                             float sin_z = std::sin(rotation_angle_z);
                             float temp_x = x * cos_z - y * sin_z;
                             float temp_y = x * sin_z + y * cos_z;
                             x = temp_x;
                             y = temp_y;

                             // Rotation around Y-axis
                             float cos_y = std::cos(rotation_angle_y);
                             float sin_y = std::sin(rotation_angle_y);
                             float temp_x2 = x * cos_y + z * sin_y;
                             float temp_z = -x * sin_y + z * cos_y;
                             x = temp_x2;
                             z = temp_z;

                             // Rotation around X-axis
                             float cos_x = std::cos(rotation_angle_x);
                             float sin_x = std::sin(rotation_angle_x);
                             float temp_y2 = y * cos_x - z * sin_x;
                             float temp_z2 = y * sin_x + z * cos_x;
                             y = temp_y2;
                             z = temp_z2;
                         }

                         // Compute spatial features for consistency with forward pass
                         float distance = std::sqrt(x*x + y*y + z*z + this->epsilon);

                         // Compute CORRECT derivative of the transform w.r.t weight
                         // When complex transforms are disabled, use simple linear derivatives
                         int coord_type = k % 3;

if (!enable_transformations && !enable_polynomial && !enable_trigonometric &&
                              !enable_log_exp && !enable_rbf && !enable_angular && !enable_projections && !enable_ratios) {
                              // SIMPLE LINEAR CASE: When all complex transforms are disabled
                              // d/dw(x*w) = x, d/dw(y*w) = y, d/dw(z*w) = z
                              if (coord_type == 0) {
                                  transform_derivative = x;
                              } else if (coord_type == 1) {
                                  transform_derivative = y;
                              } else {
                                  transform_derivative = z;
                              }
                              // No distance weighting for simple linear case
} else {
                              // COMPLEX TRANSFORMS ENABLED: Use full derivative calculations
                              // Original linear transforms: d/dw(x*w) = x, d/dw(y*w) = y, d/dw(z*w) = z
                              if (coord_type == 0) {
                                  transform_derivative += x;
                              } else if (coord_type == 1) {
                                  transform_derivative += y;
                              } else {
                                  transform_derivative += z;
                              }

                              // Polynomial transforms: d/dw(polynomial_weight * (x*x + x*y + x*z)) etc.
                              if (enable_polynomial) {
                                 if (coord_type == 0) {
                                     transform_derivative += polynomial_weight * (x*x + x*y + x*z);
                                 } else if (coord_type == 1) {
                                     transform_derivative += polynomial_weight * (y*y + x*y + y*z);
                                 } else {
                                     transform_derivative += polynomial_weight * (z*z + x*z + y*z);
                                 }
                             }

                             // Trigonometric transforms: d/dw(trig_weight * sin(x)) = trig_weight * cos(x), etc.
                             if (enable_trigonometric) {
                                 if (coord_type == 0) {
                                     transform_derivative += trig_weight * std::cos(x);
                                 } else if (coord_type == 1) {
                                     transform_derivative += trig_weight * std::cos(y);
                                 } else {
                                     float z_clamped = std::max(-std::acos(-1.0f)/2 + 0.01f, std::min(std::acos(-1.0f)/2 - 0.01f, z));
                                     transform_derivative += trig_weight * (1.0f / std::cos(z_clamped)); // derivative of tan
                                 }
                             }

                             // Logarithmic/exponential transforms
                             if (enable_log_exp) {
                                 if (coord_type == 0) {
                                     transform_derivative += log_exp_weight * (1.0f / (1.0f + std::abs(x))); // derivative of log(1+|x|)
                                 } else if (coord_type == 1) {
                                     transform_derivative += log_exp_weight * (-distance); // derivative of exp(-distance)
                                 } else {
                                     transform_derivative += log_exp_weight * (1.0f / (1.0f + std::abs(z))); // derivative of log(1+|z|)
                                 }
                             }

                             // Radial basis function: d/dw(rbf_weight * exp(-dist_sq/sigma^2))
                             if (enable_rbf) {
                                 float dx = x - rbf_center[0];
                                 float dy = y - rbf_center[1];
                                 float dz = z - rbf_center[2];
                                 float dist_sq = dx*dx + dy*dy + dz*dz;
                                 float rbf_derivative = rbf_weight * std::exp(-dist_sq / (rbf_sigma * rbf_sigma));
                                 rbf_derivative *= (-2.0f / (rbf_sigma * rbf_sigma)); // derivative of exp(-dist_sq/sigma^2)

                                 if (coord_type == 0) {
                                     transform_derivative += rbf_derivative * dx;
                                 } else if (coord_type == 1) {
                                     transform_derivative += rbf_derivative * dy;
                                 } else {
                                     transform_derivative += rbf_derivative * dz;
                                 }
                             }

                             // Angular features: d/dw(theta) and d/dw(phi)
                             if (enable_angular) {
                                 float r = distance;
                                 if (r > this->epsilon) {
                                     float theta = std::acos(std::max(-1.0f, std::min(1.0f, z / r)));
                                     // derivative of theta w.r.t coordinates is complex, approximate with finite difference
                                     transform_derivative += angular_weight * 0.1f; // simplified approximation
                                 }
                                 // phi derivative also complex, use approximation
                                 transform_derivative += angular_weight * 0.1f;
                             }

                             // Projection features: derivatives are straightforward
                             if (enable_projections) {
                                 float proj_xy = std::sqrt(x*x + y*y + this->epsilon);
                                 float proj_xz = std::sqrt(x*x + z*z + this->epsilon);
                                 float proj_yz = std::sqrt(y*y + z*z + this->epsilon);

                                 if (coord_type == 0) {
                                     transform_derivative += projections_weight * (x / proj_xy + x / proj_xz);
                                 } else if (coord_type == 1) {
                                     transform_derivative += projections_weight * (y / proj_xy + y / proj_yz);
                                 } else {
                                     transform_derivative += projections_weight * (z / proj_xz + z / proj_yz);
                                 }
                             }

                             // Distance ratios: derivatives are complex, use approximation
                             if (enable_ratios) {
                                 transform_derivative += ratios_weight * 0.1f; // simplified approximation
                             }
                         }
} else {
                          // For non-3D data (sliding windows, standard regression),
                          // use the direct input value derivative
                          transform_derivative = batch_inputs[batch_sample][k];
                      }

                     // Backpropagation with CORRECT 3D transform derivative: gradient = error * transform_derivative
                     float gradient = hidden_errors[j] * transform_derivative;

                     // DEBUG: Log gradient calculation for first sample, first neuron, first few inputs
                     if (verbose && global_sample == 0 && j == 0 && k < 5) {
                         std::cout << "[DEBUG GRADIENT] Input-Hidden gradient calc - sample: " << global_sample
                                   << ", neuron: " << j << ", input: " << k
                                   << ", hidden_error: " << hidden_errors[j]
                                   << ", transform_derivative: " << transform_derivative
                                   << ", gradient: " << gradient << std::endl;
                     }

                     // Add L2 regularization gradient: 2 * lambda * weight
                     if (use_l2_reg) {
                         gradient += 2.0f * l2_lambda * this->weigth_input_hidden[j][k];
                     }

                     // Gradient clipping to prevent exploding gradients
                     if (use_grad_clip && std::abs(gradient) > grad_clip_threshold) {
                         gradient = (gradient > 0 ? grad_clip_threshold : -grad_clip_threshold);
                     }

                     input_hidden_gradients[j][k] += gradient;
                 }
             }
         }

        // Average gradients over the current batch
        for (auto& row : output_hidden_gradients) {
            for (float& grad : row) {
                grad /= current_batch_size;
            }
        }
        for (auto& row : input_hidden_gradients) {
            for (float& grad : row) {
                grad /= current_batch_size;
            }
        }

        // Gradient accumulation with historical storage
        if (enable_gradient_accumulation) {
            // Accumulate gradients across batches
            for (size_t i = 0; i < output_hidden_gradients.size(); ++i) {
                for (size_t j = 0; j < output_hidden_gradients[i].size(); ++j) {
                    accumulated_output_hidden_gradients[i][j] += output_hidden_gradients[i][j];
                }
            }
            for (size_t i = 0; i < input_hidden_gradients.size(); ++i) {
                for (size_t j = 0; j < input_hidden_gradients[i].size(); ++j) {
                    accumulated_input_hidden_gradients[i][j] += input_hidden_gradients[i][j];
                }
            }

            current_accumulation_step++;

            // Check if we should update weights (every accumulation_steps batches)
            if (current_accumulation_step >= gradient_accumulation_steps) {
                // Store current accumulated gradients in history before scaling
                if (historical_output_hidden_gradients.size() >= static_cast<size_t>(max_historical_gradients)) {
                    historical_output_hidden_gradients.pop_front();
                    historical_input_hidden_gradients.pop_front();
                }
                historical_output_hidden_gradients.push_back(accumulated_output_hidden_gradients);
                historical_input_hidden_gradients.push_back(accumulated_input_hidden_gradients);

                // Average accumulated gradients over accumulation steps
                for (auto& row : accumulated_output_hidden_gradients) {
                    for (float& grad : row) {
                        grad /= gradient_accumulation_steps;
                    }
                }
                for (auto& row : accumulated_input_hidden_gradients) {
                    for (float& grad : row) {
                        grad /= gradient_accumulation_steps;
                    }
                }

                // Memory-intensive gradient computations
                // Add historical gradient influence to current gradients
                if (!historical_output_hidden_gradients.empty()) {
                    float historical_weight = 0.1f; // Weight for historical gradients
                    size_t history_size = historical_output_hidden_gradients.size();

                    // Create additional memory-intensive computations by processing historical gradients
                    std::vector<std::vector<std::vector<float>>> expanded_history;
                    for (size_t h = 0; h < history_size; ++h) {
                        // Duplicate and transform historical gradients to increase memory usage
                        auto hist_out = historical_output_hidden_gradients[h];
                        auto hist_in = historical_input_hidden_gradients[h];

                        // Create expanded versions with additional computations
                        std::vector<std::vector<float>> expanded_out(outputSize, std::vector<float>(neurons, 0.0f));
                        std::vector<std::vector<float>> expanded_in(neurons, std::vector<float>(inputSize, 0.0f));

                        // Memory-intensive operations: compute gradient statistics and correlations
                        for (int i = 0; i < outputSize; ++i) {
                            for (int j = 0; j < neurons; ++j) {
                                float grad_val = hist_out[i][j];
                                // Compute running statistics (mean, variance) across historical gradients
                                float running_mean = 0.0f;
                                float running_var = 0.0f;
                                for (size_t k = 0; k < history_size; ++k) {
                                    float hist_grad = historical_output_hidden_gradients[k][i][j];
                                    running_mean += hist_grad;
                                    running_var += hist_grad * hist_grad;
                                }
                                running_mean /= history_size;
                                running_var = running_var / history_size - running_mean * running_mean;

                                // Use statistics to modulate current gradients
                                expanded_out[i][j] = grad_val * (1.0f + historical_weight * running_mean / (std::sqrt(running_var + epsilon)));
                            }
                        }

                        for (int i = 0; i < neurons; ++i) {
                            for (size_t j = 0; j < inputSize; ++j) {
                                float grad_val = hist_in[i][j];
                                // Similar statistical computations for input-hidden gradients
                                float running_mean = 0.0f;
                                float running_var = 0.0f;
                                for (size_t k = 0; k < history_size; ++k) {
                                    float hist_grad = historical_input_hidden_gradients[k][i][j];
                                    running_mean += hist_grad;
                                    running_var += hist_grad * hist_grad;
                                }
                                running_mean /= history_size;
                                running_var = running_var / history_size - running_mean * running_mean;

                                expanded_in[i][j] = grad_val * (1.0f + historical_weight * running_mean / (std::sqrt(running_var + epsilon)));
                            }
                        }

                        expanded_history.push_back(expanded_out);
                        expanded_history.push_back(expanded_in);
                    }

                    // Add influence from processed historical gradients to current accumulated gradients
                    for (size_t h = 0; h < expanded_history.size(); h += 2) {
                        auto& hist_out = expanded_history[h];
                        auto& hist_in = expanded_history[h + 1];

                        for (int i = 0; i < outputSize; ++i) {
                            for (int j = 0; j < neurons; ++j) {
                                accumulated_output_hidden_gradients[i][j] += historical_weight * hist_out[i][j];
                            }
                        }
                        for (int i = 0; i < neurons; ++i) {
                            for (size_t j = 0; j < inputSize; ++j) {
                                accumulated_input_hidden_gradients[i][j] += historical_weight * hist_in[i][j];
                            }
                        }
                    }
                }

                // Now update weights using accumulated gradients
                if (optimizer) {
                    optimizer->update(this->weigth_output_hidden, accumulated_output_hidden_gradients);
                    optimizer->update(this->weigth_input_hidden, accumulated_input_hidden_gradients);

                    if (verbose) {
                        std::cout << "[DEBUG GRADIENT ACCUMULATION] Updated weights using accumulated gradients after "
                                  << gradient_accumulation_steps << " batches" << std::endl;
                    }
                } else {
                    // Fallback to basic gradient descent
                    for (int j = 0; j < this->outputSize; ++j) {
                        for (int k = 0; k < this->neurons; ++k) {
                            this->weigth_output_hidden[j][k] += currentLearningRate * accumulated_output_hidden_gradients[j][k];
                        }
                    }
                    for (int j = 0; j < this->neurons; ++j) {
                        for (size_t k = 0; k < this->inputSize; ++k) {
                            this->weigth_input_hidden[j][k] += currentLearningRate * accumulated_input_hidden_gradients[j][k];
                        }
                    }

                    if (verbose) {
                        std::cout << "[DEBUG GRADIENT ACCUMULATION] Updated weights using basic GD with accumulated gradients" << std::endl;
                    }
                }

                // Reset accumulation for next cycle
                resetGradientAccumulation();
            } else {
                // Don't update weights yet, just continue accumulating
                if (verbose) {
                    std::cout << "[DEBUG GRADIENT ACCUMULATION] Accumulated gradients for batch " << (batch_idx + 1)
                              << " (" << current_accumulation_step << "/" << gradient_accumulation_steps << ")" << std::endl;
                }
            }
        } else {
            // Original behavior: update weights after each batch
            if (optimizer) {
                optimizer->update(this->weigth_output_hidden, output_hidden_gradients);
                optimizer->update(this->weigth_input_hidden, input_hidden_gradients);

                if (verbose) {
                    std::cout << "[DEBUG TRAIN] Weights updated using " << optimizer->getName() << " optimizer for batch " << (batch_idx + 1) << std::endl;
                }
            } else {
                // Fallback to basic gradient descent if no optimizer set
                for (int j = 0; j < this->outputSize; ++j) {
                    for (int k = 0; k < this->neurons; ++k) {
                        this->weigth_output_hidden[j][k] += currentLearningRate * output_hidden_gradients[j][k];
                    }
                }
                for (int j = 0; j < this->neurons; ++j) {
                    for (size_t k = 0; k < this->inputSize; ++k) {
                        this->weigth_input_hidden[j][k] += currentLearningRate * input_hidden_gradients[j][k];
                    }
                }

                if (verbose) {
                    std::cout << "[DEBUG TRAIN] Weights updated using basic gradient descent for batch " << (batch_idx + 1) << std::endl;
                }
            }
        }

        // Update weights after each batch
        if (optimizer) {
            optimizer->update(this->weigth_output_hidden, output_hidden_gradients);
            optimizer->update(this->weigth_input_hidden, input_hidden_gradients);

            if (verbose) {
                std::cout << "[DEBUG TRAIN] Weights updated using " << optimizer->getName() << " optimizer for batch " << (batch_idx + 1) << std::endl;
            }
        } else {
            // Fallback to basic gradient descent if no optimizer set
            // DEBUG: Log weight updates for first few weights
            if (verbose && batch_idx == 0) {
                std::cout << "[DEBUG WEIGHT UPDATE] Before update - Input-Hidden[0][0]: " << this->weigth_input_hidden[0][0]
                          << ", Hidden-Output[0][0]: " << this->weigth_output_hidden[0][0] << std::endl;
                std::cout << "[DEBUG WEIGHT UPDATE] Gradients - Input-Hidden[0][0]: " << input_hidden_gradients[0][0]
                          << ", Hidden-Output[0][0]: " << output_hidden_gradients[0][0] << std::endl;
                std::cout << "[DEBUG WEIGHT UPDATE] Learning rate: " << currentLearningRate << std::endl;

                // Check if gradients are zero (potential issue)
                bool all_gradients_zero = true;
                for (int j = 0; j < this->outputSize; ++j) {
                    for (int k = 0; k < this->neurons; ++k) {
                        if (std::abs(output_hidden_gradients[j][k]) > 1e-10f) {
                            all_gradients_zero = false;
                            break;
                        }
                    }
                    if (!all_gradients_zero) break;
                }
                if (all_gradients_zero) {
                    std::cout << "[WARNING] All output-hidden gradients are zero or near-zero!" << std::endl;
                }

                all_gradients_zero = true;
                for (int j = 0; j < this->neurons; ++j) {
                    for (size_t k = 0; k < this->inputSize; ++k) {
                        if (std::abs(input_hidden_gradients[j][k]) > 1e-10f) {
                            all_gradients_zero = false;
                            break;
                        }
                    }
                    if (!all_gradients_zero) break;
                }
                if (all_gradients_zero) {
                    std::cout << "[WARNING] All input-hidden gradients are zero or near-zero!" << std::endl;
                }
            }

            for (int j = 0; j < this->outputSize; ++j) {
                for (int k = 0; k < this->neurons; ++k) {
                    this->weigth_output_hidden[j][k] += currentLearningRate * output_hidden_gradients[j][k];
                }
            }
            for (int j = 0; j < this->neurons; ++j) {
                for (size_t k = 0; k < this->inputSize; ++k) {
                    this->weigth_input_hidden[j][k] += currentLearningRate * input_hidden_gradients[j][k];
                }
            }

            if (verbose && batch_idx == 0) {
                std::cout << "[DEBUG WEIGHT UPDATE] After update - Input-Hidden[0][0]: " << this->weigth_input_hidden[0][0]
                          << ", Hidden-Output[0][0]: " << this->weigth_output_hidden[0][0] << std::endl;

                // Check if weights actually changed
                static float prev_ih_weight = this->weigth_input_hidden[0][0];
                static float prev_ho_weight = this->weigth_output_hidden[0][0];
                float ih_change = this->weigth_input_hidden[0][0] - prev_ih_weight;
                float ho_change = this->weigth_output_hidden[0][0] - prev_ho_weight;
                std::cout << "[DEBUG WEIGHT UPDATE] Weight changes - Input-Hidden[0][0]: " << ih_change
                          << ", Hidden-Output[0][0]: " << ho_change << std::endl;
                if (std::abs(ih_change) < 1e-10f && std::abs(ho_change) < 1e-10f) {
                    std::cout << "[WARNING] Weights are not changing! Check gradients and learning rate." << std::endl;
                }
                prev_ih_weight = this->weigth_input_hidden[0][0];
                prev_ho_weight = this->weigth_output_hidden[0][0];
            }

            if (verbose) {
                std::cout << "[DEBUG TRAIN] Weights updated using basic gradient descent for batch " << (batch_idx + 1) << std::endl;
            }
        }
    }


    // Increment epoch counter
    currentEpoch++;
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
        std::cout << "[DEBUG DENORMALIZE] First few results before denormalization: ";
        if (!results.empty() && !results[0].empty()) {
            for (size_t i = 0; i < std::min(size_t(5), results[0].size()); ++i) {
                std::cout << results[0][i] << " ";
            }
        }
        std::cout << std::endl;
        // Log all results if there are few samples
        if (results.size() <= 10) {
            std::cout << "[DEBUG DENORMALIZE] All results before denormalization:" << std::endl;
            for (size_t i = 0; i < results.size(); ++i) {
                std::cout << "  Sample " << i << ": ";
                for (size_t j = 0; j < results[i].size(); ++j) {
                    std::cout << results[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }
    }

    float normalized_range_diff = this->maxVal - this->minVal;
    size_t num_target_features = trainer->target_mins.size();

    // Check for valid ranges
    if (std::abs(normalized_range_diff) < std::numeric_limits<float>::epsilon()) {
        std::cerr << "Warning: Normalized range is zero, cannot denormalize properly." << std::endl;
        return;
    }

    for (auto& row : results)
    { // Denormalize the results/predictions
        for (size_t j = 0; j < row.size(); ++j)
        {
            float original_val = row[j];
            
            // Use per-target bounds if available, otherwise fallback to global bounds
            float y_min, y_max, original_range_diff;
            if (j < num_target_features && !trainer->target_mins.empty()) {
                y_min = trainer->target_mins[j];
                y_max = trainer->target_maxs[j];
                original_range_diff = y_max - y_min;
            } else {
                y_min = trainer->original_data_global_min;
                y_max = trainer->original_data_global_max;
                original_range_diff = y_max - y_min;
            }
            
            if (std::abs(original_range_diff) > std::numeric_limits<float>::epsilon()) {
                row[j] = ((original_val - this->minVal) / normalized_range_diff) * original_range_diff + y_min;
            } else {
                row[j] = y_min; // If original range is zero, map to original min
            }

            // DEBUG: Log first few denormalizations (per call, not static)
            if (verbose) {
                static int call_count = 0;
                if (call_count++ < 5) {
                    std::cout << "[DEBUG DENORMALIZE] prediction " << original_val << " -> " << row[j] << std::endl;
                }
            }
        }
    }
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
    if (verbose) {
        std::cout << "[DEBUG RESIZE] Resizing and re-initializing weights" << std::endl;
    }

    // Resize and initialize weight matrices with adaptive initialization
    initializeMatrix(this->weigth_output_hidden, outputSize, neurons);
    initializeWeightsAdaptive(neurons, outputSize, this->weigth_output_hidden, ActivationType::SIGMOID);

    initializeMatrix(this->weigth_input_hidden, neurons, inputSize);
    initializeWeightsAdaptive(inputSize, neurons, this->weigth_input_hidden, ActivationType::SIGMOID);

    if (verbose) {
        std::cout << "[DEBUG RESIZE] Weight re-initialization complete" << std::endl;
    }
}

void CoreAI::setOptimizer(const std::string& optimizerType, float learningRate,
                         float beta1, float beta2, float epsilon,
                         float momentum, float rho) {
    if (optimizerType == "SGD") {
        optimizer = std::make_unique<SGD>(learningRate, momentum);
    } else if (optimizerType == "Adam") {
        optimizer = std::make_unique<Adam>(learningRate, beta1, beta2, epsilon);
    } else if (optimizerType == "RMSprop") {
        optimizer = std::make_unique<RMSprop>(learningRate, rho, epsilon);
    } else {
        throw std::invalid_argument("Unknown optimizer type: " + optimizerType);
    }

    if (verbose) {
        std::cout << "[DEBUG OPTIMIZER] Set optimizer to: " << optimizerType
                  << " with learning rate: " << learningRate << std::endl;
    }
}

void CoreAI::setLearningRateScheduler(const std::string& schedulerType, float initialLearningRate,
                                      float decayRate, int decaySteps,
                                      float decayFactor, int totalEpochs,
                                      float minLearningRate, int warmupEpochs) {
    std::unique_ptr<LearningRateScheduler> baseScheduler;

    if (schedulerType == "ExponentialDecay") {
        baseScheduler = std::make_unique<ExponentialDecay>(initialLearningRate, decayRate, decaySteps);
    } else if (schedulerType == "StepDecay") {
        baseScheduler = std::make_unique<StepDecay>(initialLearningRate, decayFactor, decaySteps);
    } else if (schedulerType == "CosineAnnealing") {
        baseScheduler = std::make_unique<CosineAnnealing>(initialLearningRate, totalEpochs, minLearningRate);
    } else {
        throw std::invalid_argument("Unknown scheduler type: " + schedulerType);
    }

    // Apply warmup if specified
    if (warmupEpochs > 0) {
        lrScheduler = std::make_unique<WarmupScheduler>(initialLearningRate, warmupEpochs, std::move(baseScheduler));
    } else {
        lrScheduler = std::move(baseScheduler);
    }

    if (verbose) {
        std::cout << "[DEBUG SCHEDULER] Set learning rate scheduler to: " << schedulerType
                  << " with initial learning rate: " << initialLearningRate;
        if (warmupEpochs > 0) {
            std::cout << " (with " << warmupEpochs << " warmup epochs)";
        }
        std::cout << std::endl;
    }
}

void CoreAI::resetOptimizerState() {
    if (optimizer) {
        optimizer->reset();
    }
    if (lrScheduler) {
        lrScheduler->reset();
    }
    currentEpoch = 0;

    if (verbose) {
        std::cout << "[DEBUG RESET] Optimizer and scheduler state reset" << std::endl;
    }
}

void CoreAI::setRegularizationParams(float l2_lambda, float dropout_rate,
                                    float bn_momentum, float grad_clip_threshold) {
    this->l2_lambda = l2_lambda;
    this->dropout_rate = dropout_rate;
    this->bn_momentum = bn_momentum;
    this->grad_clip_threshold = grad_clip_threshold;

    if (verbose) {
        std::cout << "[DEBUG REGULARIZATION] Set regularization parameters:" << std::endl;
        std::cout << "  L2 lambda: " << l2_lambda << std::endl;
        std::cout << "  Dropout rate: " << dropout_rate << std::endl;
        std::cout << "  BN momentum: " << bn_momentum << std::endl;
        std::cout << "  Grad clip threshold: " << grad_clip_threshold << std::endl;
    }
}

void CoreAI::enableRegularization(bool dropout, bool batch_norm,
                                  bool l2_reg, bool grad_clip) {
    this->use_dropout = dropout;
    this->use_batch_norm = batch_norm;
    this->use_l2_reg = l2_reg;
    this->use_grad_clip = grad_clip;

    if (verbose) {
        std::cout << "[DEBUG REGULARIZATION] Enabled features:" << std::endl;
        std::cout << "  Dropout: " << (dropout ? "yes" : "no") << std::endl;
        std::cout << "  Batch norm: " << (batch_norm ? "yes" : "no") << std::endl;
        std::cout << "  L2 regularization: " << (l2_reg ? "yes" : "no") << std::endl;
        std::cout << "  Gradient clipping: " << (grad_clip ? "yes" : "no") << std::endl;
    }
}

// Weight initialization configuration methods

void CoreAI::setWeightInitialization(WeightInitialization type, float scale, unsigned int seed) {
    weight_init_type = type;
    weight_init_scale = scale;
    weight_init_seed = seed;

    if (verbose) {
        std::cout << "[DEBUG WEIGHT INIT] Set weight initialization to type " << static_cast<int>(type)
                  << " with scale " << scale << " and seed " << seed << std::endl;
    }
}

void CoreAI::initializeWeightsAdaptive(int from_size, int to_size, std::vector<std::vector<float>>& weights, ActivationType activation_type) {
    std::mt19937 gen(weight_init_seed);
    float scale = weight_init_scale;
    
    // Ensure scale is not zero
    if (scale <= 0.0f) {
        scale = 1.0f;
        if (verbose) {
            std::cout << "[WARNING] Weight initialization scale was <= 0.0, using default 1.0" << std::endl;
        }
    }

    // Adaptive initialization based on activation function
    switch (activation_type) {
        case ActivationType::SIGMOID:
        case ActivationType::INTEGRAL_SIGMOID:
        case ActivationType::INTEGRAL_TANH:
            // Xavier/Glorot initialization for sigmoid/tanh
            scale *= std::sqrt(6.0f / (from_size + to_size));
            break;
        case ActivationType::ELU:
        case ActivationType::LEAKY_RELU:
        case ActivationType::PARAMETRIC_RELU:
        case ActivationType::SWISH:
        case ActivationType::GELU:
        case ActivationType::SPATIAL_ELU:
        case ActivationType::SPATIAL_LEAKY_RELU:
        case ActivationType::INTEGRAL_RELU:
            // He initialization for ReLU variants
            scale *= std::sqrt(2.0f / from_size);
            break;
        default:
            // Default Xavier initialization
            scale *= std::sqrt(6.0f / (from_size + to_size));
            break;
    }

    // Apply the specific initialization method
    switch (weight_init_type) {
        case WeightInitialization::XAVIER_UNIFORM: {
            std::uniform_real_distribution<float> dist(-scale, scale);
            for (auto& row : weights) {
                for (float& val : row) {
                    val = dist(gen);
                }
            }
            break;
        }
        case WeightInitialization::XAVIER_NORMAL: {
            std::normal_distribution<float> dist(0.0f, scale);
            for (auto& row : weights) {
                for (float& val : row) {
                    val = dist(gen);
                }
            }
            break;
        }
        case WeightInitialization::HE_UNIFORM: {
            float limit = scale * std::sqrt(3.0f);
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (auto& row : weights) {
                for (float& val : row) {
                    val = dist(gen);
                }
            }
            break;
        }
        case WeightInitialization::HE_NORMAL: {
            std::normal_distribution<float> dist(0.0f, scale);
            for (auto& row : weights) {
                for (float& val : row) {
                    val = dist(gen);
                }
            }
            break;
        }
        case WeightInitialization::LECUN_UNIFORM: {
            float limit = scale * std::sqrt(3.0f);
            std::uniform_real_distribution<float> dist(-limit, limit);
            for (auto& row : weights) {
                for (float& val : row) {
                    val = dist(gen);
                }
            }
            break;
        }
        case WeightInitialization::LECUN_NORMAL: {
            std::normal_distribution<float> dist(0.0f, scale);
            for (auto& row : weights) {
                for (float& val : row) {
                    val = dist(gen);
                }
            }
            break;
        }
        case WeightInitialization::RANDOM_UNIFORM: {
            std::uniform_real_distribution<float> dist(-scale, scale);
            for (auto& row : weights) {
                for (float& val : row) {
                    val = dist(gen);
                }
            }
            break;
        }
        case WeightInitialization::RANDOM_NORMAL: {
            std::normal_distribution<float> dist(0.0f, scale);
            for (auto& row : weights) {
                for (float& val : row) {
                    val = dist(gen);
                }
            }
            break;
        }
    }

    if (verbose) {
        std::cout << "[DEBUG WEIGHT INIT] Initialized " << weights.size() << "x" << weights[0].size()
                  << " weights with " << static_cast<int>(weight_init_type) << " method, scale: " << scale << std::endl;
    }
}

// Multi-layer architecture implementation
void CoreAI::initializeMultiLayerWeights() {
    layer_weights.clear();
    layer_weights.resize(layer_sizes.size() - 1); // weights between layers

    for (size_t i = 0; i < layer_weights.size(); ++i) {
        int from_size = layer_sizes[i];
        int to_size = layer_sizes[i + 1];

        // Determine activation type for this layer (use default if not set)
        ActivationType activation_type = ActivationType::SIGMOID;
        if (i < layer_activation_configs.size()) {
            activation_type = layer_activation_configs[i].type;
        }

        // Initialize weights using adaptive initialization
        layer_weights[i].resize(to_size);
        for (int j = 0; j < to_size; ++j) {
            layer_weights[i][j].resize(from_size);
            // Create a temporary 2D vector for this weight matrix
            std::vector<std::vector<float>> temp_weights = {layer_weights[i][j]};
            initializeWeightsAdaptive(from_size, to_size, temp_weights, activation_type);
            layer_weights[i][j] = temp_weights[0];
        }
    }

    if (verbose) {
        std::cout << "[DEBUG MULTI-LAYER] Initialized " << layer_sizes.size() << " layers with sizes: ";
        for (int size : layer_sizes) std::cout << size << " ";
        std::cout << std::endl;
    }
}


void CoreAI::initializeAttentionWeights() {
    if (inputSize % 3 != 0) {
        if (verbose) {
            std::cout << "[WARNING ATTENTION] Input size not divisible by 3, attention may not work optimally" << std::endl;
        }
    }

    int num_triplets = inputSize / 3;
    attention_weights.resize(num_triplets);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> att_dist(0.0f, 1.0f);

    for (int i = 0; i < num_triplets; ++i) {
        attention_weights[i].resize(num_triplets);
        for (int j = 0; j < num_triplets; ++j) {
            attention_weights[i][j] = att_dist(gen);
        }
        // Normalize attention weights for each triplet
        float sum = 0.0f;
        for (float w : attention_weights[i]) sum += w;
        if (sum > 0) {
            for (float& w : attention_weights[i]) w /= sum;
        }
    }

    if (verbose) {
        std::cout << "[DEBUG ATTENTION] Initialized attention weights for " << num_triplets << " triplets" << std::endl;
    }
}

void CoreAI::setLayerArchitecture(const std::vector<int>& layer_sizes) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("CoreAI::setLayerArchitecture: Must have at least input and output layers");
    }

    this->layer_sizes = layer_sizes;
    this->inputSize = layer_sizes[0];
    this->outputSize = layer_sizes.back();
    this->layers = layer_sizes.size() - 2; // hidden layers only

    // Reinitialize weights for new architecture
    initializeMultiLayerWeights();

    if (enable_attention) {
        initializeAttentionWeights();
    }

    if (verbose) {
        std::cout << "[DEBUG ARCHITECTURE] Set layer architecture with " << layer_sizes.size() << " layers" << std::endl;
    }
}

void CoreAI::enableResidualConnections(bool enable, const std::vector<std::vector<int>>& connections) {
    this->enable_residual = enable;
    if (enable && !connections.empty()) {
        this->residual_connections = connections;
    } else if (enable && connections.empty()) {
        // Default: connect every other layer (1->3, 2->4, etc.)
        residual_connections.clear();
        for (size_t i = 0; i < layer_sizes.size() - 2; ++i) {
            residual_connections.push_back({static_cast<int>(i), static_cast<int>(i + 2)});
        }
    } else {
        residual_connections.clear();
    }

    if (verbose) {
        std::cout << "[DEBUG RESIDUAL] " << (enable ? "Enabled" : "Disabled") << " residual connections" << std::endl;
        if (enable && !residual_connections.empty()) {
            std::cout << "[DEBUG RESIDUAL] Connections: ";
            for (const auto& conn : residual_connections) {
                std::cout << "(" << conn[0] << "->" << conn[1] << ") ";
            }
            std::cout << std::endl;
        }
    }
}

void CoreAI::configureAttention(bool enable, float temperature, float weight) {
    this->enable_attention = enable;
    this->attention_temperature = temperature;
    this->attention_weight = weight;

    if (enable) {
        initializeAttentionWeights();
    } else {
        attention_weights.clear();
    }

    if (verbose) {
        std::cout << "[DEBUG ATTENTION] " << (enable ? "Enabled" : "Disabled") << " spatial attention" << std::endl;
        if (enable) {
            std::cout << "[DEBUG ATTENTION] Temperature: " << temperature << ", Weight: " << weight << std::endl;
        }
    }
}

void CoreAI::setLayerNeurons(int layer_index, int neurons) {
    if (layer_index < 0 || layer_index >= static_cast<int>(layer_sizes.size())) {
        throw std::invalid_argument("CoreAI::setLayerNeurons: Invalid layer index");
    }
    if (neurons <= 0) {
        throw std::invalid_argument("CoreAI::setLayerNeurons: Neurons must be positive");
    }

    layer_sizes[layer_index] = neurons;

    // Reinitialize weights
    initializeMultiLayerWeights();

    if (verbose) {
        std::cout << "[DEBUG LAYER] Set layer " << layer_index << " to " << neurons << " neurons" << std::endl;
    }
}

// Activation function configuration methods
void CoreAI::setLayerActivation(int layer_index, ActivationType type, float alpha, float beta, float spatial_weight,
                               int calculus_N, float calculus_SM, float calculus_a) {
    if (layer_index < 0 || layer_index >= static_cast<int>(layer_sizes.size())) {
        throw std::invalid_argument("CoreAI::setLayerActivation: Invalid layer index");
    }

    if (layer_activation_configs.size() != layer_sizes.size()) {
        layer_activation_configs.resize(layer_sizes.size());
    }

    layer_activation_configs[layer_index].type = type;
    layer_activation_configs[layer_index].alpha = alpha;
    layer_activation_configs[layer_index].beta = beta;
    layer_activation_configs[layer_index].spatial_weight = spatial_weight;
    layer_activation_configs[layer_index].calculus_config = {calculus_N, calculus_SM, calculus_a};

    if (verbose) {
        std::cout << "[DEBUG ACTIVATION] Set layer " << layer_index << " activation to type " << static_cast<int>(type)
                  << " with alpha=" << alpha << ", beta=" << beta << ", spatial_weight=" << spatial_weight
                  << ", calculus_N=" << calculus_N << ", calculus_SM=" << calculus_SM << ", calculus_a=" << calculus_a << std::endl;
    }
}

void CoreAI::setAllLayerActivations(ActivationType type, float alpha, float beta, float spatial_weight,
                                   int calculus_N, float calculus_SM, float calculus_a) {
    if (layer_activation_configs.size() != layer_sizes.size()) {
        layer_activation_configs.resize(layer_sizes.size());
    }

    for (size_t i = 0; i < layer_activation_configs.size(); ++i) {
        layer_activation_configs[i].type = type;
        layer_activation_configs[i].alpha = alpha;
        layer_activation_configs[i].beta = beta;
        layer_activation_configs[i].spatial_weight = spatial_weight;
        layer_activation_configs[i].calculus_config = {calculus_N, calculus_SM, calculus_a};
    }

    if (verbose) {
        std::cout << "[DEBUG ACTIVATION] Set all layers to activation type " << static_cast<int>(type)
                  << " with alpha=" << alpha << ", beta=" << beta << ", spatial_weight=" << spatial_weight
                  << ", calculus_N=" << calculus_N << ", calculus_SM=" << calculus_SM << ", calculus_a=" << calculus_a << std::endl;
    }
}

// 3D transform configuration methods (for auto-tuning)
void CoreAI::set3DTransformConfig(bool enable_polynomial, bool enable_trigonometric, bool enable_log_exp,
                                 bool enable_rbf, bool enable_angular, bool enable_projections,
                                 bool enable_ratios, bool enable_transformations) {
    this->enable_polynomial = enable_polynomial;
    this->enable_trigonometric = enable_trigonometric;
    this->enable_log_exp = enable_log_exp;
    this->enable_rbf = enable_rbf;
    this->enable_angular = enable_angular;
    this->enable_projections = enable_projections;
    this->enable_ratios = enable_ratios;
    this->enable_transformations = enable_transformations;

    if (verbose) {
        std::cout << "[DEBUG 3D CONFIG] Set 3D transform configuration:" << std::endl;
        std::cout << "  Polynomial: " << (enable_polynomial ? "enabled" : "disabled") << std::endl;
        std::cout << "  Trigonometric: " << (enable_trigonometric ? "enabled" : "disabled") << std::endl;
        std::cout << "  Log/Exp: " << (enable_log_exp ? "enabled" : "disabled") << std::endl;
        std::cout << "  RBF: " << (enable_rbf ? "enabled" : "disabled") << std::endl;
        std::cout << "  Angular: " << (enable_angular ? "enabled" : "disabled") << std::endl;
        std::cout << "  Projections: " << (enable_projections ? "enabled" : "disabled") << std::endl;
        std::cout << "  Ratios: " << (enable_ratios ? "enabled" : "disabled") << std::endl;
        std::cout << "  Transformations: " << (enable_transformations ? "enabled" : "disabled") << std::endl;
    }
}

void CoreAI::set3DTransformWeights(float polynomial_weight, float trig_weight, float log_exp_weight,
                                  float rbf_weight, float angular_weight, float projections_weight,
                                  float ratios_weight) {
    this->polynomial_weight = polynomial_weight;
    this->trig_weight = trig_weight;
    this->log_exp_weight = log_exp_weight;
    this->rbf_weight = rbf_weight;
    this->angular_weight = angular_weight;
    this->projections_weight = projections_weight;
    this->ratios_weight = ratios_weight;

    if (verbose) {
        std::cout << "[DEBUG 3D WEIGHTS] Set 3D transform weights:" << std::endl;
        std::cout << "  Polynomial: " << polynomial_weight << std::endl;
        std::cout << "  Trigonometric: " << trig_weight << std::endl;
        std::cout << "  Log/Exp: " << log_exp_weight << std::endl;
        std::cout << "  RBF: " << rbf_weight << std::endl;
        std::cout << "  Angular: " << angular_weight << std::endl;
        std::cout << "  Projections: " << projections_weight << std::endl;
        std::cout << "  Ratios: " << ratios_weight << std::endl;
    }
}

// Gradient accumulation configuration methods
void CoreAI::enableGradientAccumulation(bool enable, int accumulation_steps, int max_history) {
    this->enable_gradient_accumulation = enable;
    this->gradient_accumulation_steps = accumulation_steps;
    this->max_historical_gradients = max_history;

    if (enable) {
        // Resize accumulated gradients matrices
        initializeMatrix(this->accumulated_output_hidden_gradients, outputSize, neurons);
        std::fill(accumulated_output_hidden_gradients.begin(), accumulated_output_hidden_gradients.end(),
                  std::vector<float>(neurons, 0.0f));
        initializeMatrix(this->accumulated_input_hidden_gradients, neurons, inputSize);
        std::fill(accumulated_input_hidden_gradients.begin(), accumulated_input_hidden_gradients.end(),
                  std::vector<float>(inputSize, 0.0f));
    } else {
        // Clear accumulated gradients when disabled
        accumulated_output_hidden_gradients.clear();
        accumulated_input_hidden_gradients.clear();
        historical_output_hidden_gradients.clear();
        historical_input_hidden_gradients.clear();
    }

    this->current_accumulation_step = 0;

    if (verbose) {
        std::cout << "[DEBUG GRADIENT ACCUMULATION] " << (enable ? "Enabled" : "Disabled")
                  << " gradient accumulation with " << accumulation_steps << " steps, max history: " << max_history << std::endl;
    }
}

void CoreAI::resetGradientAccumulation() {
    this->current_accumulation_step = 0;
    if (!accumulated_output_hidden_gradients.empty()) {
        std::fill(accumulated_output_hidden_gradients.begin(), accumulated_output_hidden_gradients.end(),
                  std::vector<float>(neurons, 0.0f));
    }
    if (!accumulated_input_hidden_gradients.empty()) {
        std::fill(accumulated_input_hidden_gradients.begin(), accumulated_input_hidden_gradients.end(),
                  std::vector<float>(inputSize, 0.0f));
    }

    if (verbose) {
        std::cout << "[DEBUG GRADIENT ACCUMULATION] Reset gradient accumulation state" << std::endl;
    }
}

size_t CoreAI::getGradientHistorySize() const {
    return historical_output_hidden_gradients.size();
}

void CoreAI::clearGradientHistory() {
    historical_output_hidden_gradients.clear();
    historical_input_hidden_gradients.clear();

    if (verbose) {
        std::cout << "[DEBUG GRADIENT ACCUMULATION] Cleared gradient history" << std::endl;
    }
}

std::vector<std::vector<float>> CoreAI::getAccumulatedGradientsOutputHidden() const {
    return accumulated_output_hidden_gradients;
}

std::vector<std::vector<float>> CoreAI::getAccumulatedGradientsInputHidden() const {
    return accumulated_input_hidden_gradients;
}

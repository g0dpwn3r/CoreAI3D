#ifndef CORE_HPP
#define CORE_HPP

#include "CoreAI3DCommon.hpp"
#include <memory>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>

#include "Train.hpp"
#include "Language.hpp"
#include "Database.hpp"
#include "Optimizer.hpp"
#include "LearningRateScheduler.hpp"

// Calculus-based activation configuration
struct CalculusConfig {
    int N = 10;      // Number of integration points
    float SM = 1.0f; // Simpson's method multiplier
    float a = 1.0f;  // Alpha scaling parameter
};

// Activation function types
enum class ActivationType {
    SIGMOID,    // Current implementation (for backward compatibility)
    ELU,        // Exponential Linear Unit
    LEAKY_RELU, // Leaky Rectified Linear Unit
    PARAMETRIC_RELU, // Parametric ReLU
    SWISH,      // Swish activation
    GELU,       // Gaussian Error Linear Unit
    SPATIAL_ELU, // Spatial-aware ELU for coordinate data
    SPATIAL_LEAKY_RELU, // Spatial-aware Leaky ReLU for coordinate data
    INTEGRAL_SIGMOID,   // Calculus-based: integral of sigmoid
    INTEGRAL_RELU,      // Calculus-based: integral of ReLU
    INTEGRAL_TANH       // Calculus-based: integral of tanh
};

// Weight initialization types
enum class WeightInitialization {
    XAVIER_UNIFORM,    // Xavier/Glorot uniform initialization
    XAVIER_NORMAL,     // Xavier/Glorot normal initialization
    HE_UNIFORM,        // He uniform initialization (for ReLU)
    HE_NORMAL,         // He normal initialization (for ReLU)
    LECUN_UNIFORM,     // LeCun uniform initialization
    LECUN_NORMAL,      // LeCun normal initialization
    RANDOM_UNIFORM,    // Random uniform [-limit, limit]
    RANDOM_NORMAL      // Random normal with mean 0, std dev
};

// Activation configuration structure
struct ActivationConfig {
    ActivationType type = ActivationType::SIGMOID;
    float alpha = 1.0f;  // For ELU, Leaky ReLU
    float beta = 0.1f;   // For Parametric ReLU
    float spatial_weight = 1.0f; // For spatial activations
    CalculusConfig calculus_config; // For calculus-based activations
};


class Training;

class CoreAI
{
private:
    int inputSize;
    int outputSize;
    int layers;
    int neurons;
    float epsilon = std::numeric_limits<float>::epsilon();
    double e = std::exp(1.0);
    double defaultLearningRate = 0.001; // Default learning rate for stable convergence
   
    // Advanced 3D transform parameters
    bool enable_polynomial;
    bool enable_trigonometric;
    bool enable_log_exp;
    bool enable_rbf;
    float polynomial_weight;
    float trig_weight;
    float log_exp_weight;
    float rbf_weight;
    std::vector<float> rbf_center;
    float rbf_sigma;

    // New 3D feature engineering parameters
    bool enable_angular;
    bool enable_projections;
    bool enable_ratios;
    bool enable_transformations;
    float angular_weight;
    float projections_weight;
    float ratios_weight;
    float transformations_weight;

    // Transformation parameters
    float rotation_angle_x, rotation_angle_y, rotation_angle_z;
    float scale_x, scale_y, scale_z;
    float translation_x, translation_y, translation_z;

    bool verbose; // Control verbosity of debug output

    Training* getTrainer();

    std::vector<std::vector<float> > input;
    std::vector<std::vector<float> > output; // This could be the targets or actual output after forward pass
    std::vector<std::vector<float> > results; // This will store the final output of the forward pass

    // Multi-layer network architecture
    std::vector<int> layer_sizes; // Number of neurons per layer (including input and output)
    std::vector<std::vector<std::vector<float>>> layer_weights; // Weights between layers [layer][neuron_from][neuron_to]
    std::vector<std::vector<std::vector<float>>> layer_activations; // Activations per layer per sample
    std::vector<std::vector<std::vector<float>>> layer_errors; // Errors per layer per sample

    // Residual connections
    bool enable_residual;
    std::vector<std::vector<int>> residual_connections; // Pairs of layers that have residual connections

    // Spatial attention mechanism
    bool enable_attention;
    std::vector<std::vector<float>> attention_weights; // Attention weights for coordinate triplets
    float attention_temperature; // Temperature parameter for attention softmax
    float attention_weight; // Overall weight for attention mechanism

    // Legacy single-layer support (for backward compatibility)
    // Internal state of the AI model to be saved/loaded
    std::vector<std::vector<float>> hidden; // Example: activations of hidden layer
    std::vector<std::vector<float>> weigth_input_hidden; // Weights from input to hidden layer
    std::vector<std::vector<float>> weigth_output_hidden; // Weights from hidden to output layer

    std::vector<float> hidden_output; // Example: output after activation function
    // of hidden layer
    std::vector<float> hidden_error; // Example: error from backpropagation
    // Private helper for random number generation
    float beautifulRandom();

    bool safe1DIndex(const std::vector<float>& vec, int index);

    bool safe2DIndex(const std::vector<std::vector<float> >& matrix, int i,
        int j);

    // Placeholder for transformation functions, assume they exist
    float xTransform(float x, float y, float z, float n, float m);
    float yTransform(float x, float y, float z, float n, float m);
    float zTransform(float x, float y, float z, float n, float m);
    float xTransform2(float x, float e, float n, float m);
    float yTransform2(float y, float e, float n, float m);
    float zTransform2(float z, float e, float n, float m);
    float xTransform2_derivative_m(float x, float e, float n, float m);
    float yTransform2_derivative_m(float y, float e, float n, float m);
    float zTransform2_derivative_m(float z, float e, float n, float m);
    void computeMinMax(const std::vector<std::vector<float>>& data, float& min_val, float& max_val);

public:
    CoreAI(int inputSize_, int layers_, int neurons_, int outputSize_,
        float min_, float max_, bool verbose_ = false);


    float minVal;
    float maxVal;
    
     // Member variables for the transform functions (added for compilation)
    float x;
    float y;
    float z;
    // Methods to get internal state for saving

    std::vector<std::vector<float>>& getInput();
    std::vector<std::vector<float>>& getOutput();
    std::vector<std::vector<float>>& getResults();
    const std::vector<std::vector<float>>& getHiddenData() const;
    const std::vector<std::vector<float>>& getWeightsHiddenInput() const;
    const std::vector<std::vector<float>>& getWeightsOutputHidden() const;
    const std::vector<float>& getHiddenOutputData() const;
    const std::vector<float>& getHiddenErrorData() const;

    Training* trainer;

    // Optimizer and learning rate scheduler
    std::unique_ptr<Optimizer> optimizer;
    std::unique_ptr<LearningRateScheduler> lrScheduler;
    int currentEpoch; // Track current training epoch for scheduler

    // Regularization and stability parameters
    float l2_lambda; // L2 regularization strength (weight decay)
    float dropout_rate; // Dropout probability (0.0 to 1.0)
    float bn_momentum; // Batch normalization momentum
    float grad_clip_threshold; // Gradient clipping threshold

    // Feature flags
    bool use_dropout;
    bool use_batch_norm;
    bool use_l2_reg;
    bool use_grad_clip;
    // Weight initialization configuration
    WeightInitialization weight_init_type; // Type of weight initialization to use
    float weight_init_scale; // Scale factor for initialization (default 1.0)
    unsigned int weight_init_seed; // Random seed for reproducible initialization

    // Batch normalization parameters
    std::vector<float> bn_running_mean;
    std::vector<float> bn_running_var;
    std::vector<float> bn_gamma; // scale parameters
    std::vector<float> bn_beta; // shift parameters

    // Dropout masks (per sample during training)
    std::vector<std::vector<float>> dropout_masks;

    // Activation function configurations per layer
    std::vector<ActivationConfig> layer_activation_configs; // One config per layer (including output)

    // Gradient accumulation with historical storage
    bool enable_gradient_accumulation;
    int gradient_accumulation_steps; // Number of batches to accumulate before update
    int max_historical_gradients; // Maximum number of historical gradient sets to keep
    std::deque<std::vector<std::vector<float>>> historical_output_hidden_gradients; // Historical gradients for output-hidden weights
    std::deque<std::vector<std::vector<float>>> historical_input_hidden_gradients; // Historical gradients for input-hidden weights
    std::vector<std::vector<float>> accumulated_output_hidden_gradients; // Current accumulated gradients
    std::vector<std::vector<float>> accumulated_input_hidden_gradients; // Current accumulated gradients
    int current_accumulation_step; // Current step in accumulation cycle

    // Thread pool for parallel batch processing
    class ThreadPool {
    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        bool stop;

    public:
        ThreadPool(size_t threads) : stop(false) {
            for(size_t i = 0; i < threads; ++i) {
                workers.emplace_back([this] {
                    for(;;) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                            if(this->stop && this->tasks.empty()) return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }
                        task();
                    }
                });
            }
        }

        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
            using return_type = typename std::result_of<F(Args...)>::type;
            auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
            std::future<return_type> res = task->get_future();
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
                tasks.emplace([task](){ (*task)(); });
            }
            condition.notify_one();
            return res;
        }

        ~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
            }
            condition.notify_all();
            for(std::thread &worker: workers) {
                worker.join();
            }
        }
    };

    std::unique_ptr<ThreadPool> threadPool;

    // Methods to set internal state for loading
    void setInput(const std::vector<std::vector<float> >& data);
    void setOutput(const std::vector<std::vector<float> >
        & data); // If this is for original targets
    void setTarget(const std::vector<std::vector<float> >
        & data); // Clarifies setting target data
    void setHiddenData(const std::vector<std::vector<float> >& data);
    void setHiddenOutputData(const std::vector<float>& data);
    void setHiddenErrorData(const std::vector<float>& data);
    void setWeightsHiddenInput(const std::vector<std::vector<float> >& data);
    void setWeightsOutputHidden(const std::vector<std::vector<float> >& data);
    void initializeMatrix(std::vector<std::vector<float>>& matrix, int rows, int cols);
    void initializeVector(std::vector<float>& vec, int size);
    template<typename Func>
    void fillMatrixWithTransform(std::vector<std::vector<float>>& matrix, Func transform);
    template<typename Func>
    void fillVectorWithTransform(std::vector<float>& vec, Func transform);
    // Core AI functionalities
    void setInputSize(int size);
    void setOutputSize(int size);
    void resizeWeights();
    std::vector<std::vector<float> >
        forward(const std::vector<std::vector<float> >& inputData, bool training = false);
    void train(const std::vector<std::vector<float> >& inputData,
        const std::vector<std::vector<float> >& targetData,
        double learningRate, int& numSamples, int batchSize = 32);
    void normalizeInput(float min_val, float max_val);
    void normalizeOutput(float min_val,
        float max_val); // This is likely normalizing targets
    void denormalizeOutput();
    // Activation functions
    float sigmoid(float x);
    float elu(float x, float alpha);
    float elu_derivative(float x, float alpha);
    float leaky_relu(float x, float alpha);
    float leaky_relu_derivative(float x, float alpha);
    float parametric_relu(float x, float beta);
    float parametric_relu_derivative(float x, float beta);
    float swish(float x);
    float swish_derivative(float x);
    float gelu(float x);
    float gelu_derivative(float x);
    float spatial_elu(float x, float distance, float alpha, float spatial_weight);
    float spatial_elu_derivative(float x, float distance, float alpha, float spatial_weight);
    float spatial_leaky_relu(float x, float distance, float alpha, float spatial_weight);
    float spatial_leaky_relu_derivative(float x, float distance, float alpha, float spatial_weight);
    // Calculus-based activation functions
    float integral_sigmoid(float x, const CalculusConfig& config);
    float integral_sigmoid_derivative(float x, const CalculusConfig& config);
    float integral_relu(float x, const CalculusConfig& config);
    float integral_relu_derivative(float x, const CalculusConfig& config);
    float integral_tanh(float x, const CalculusConfig& config);
    float integral_tanh_derivative(float x, const CalculusConfig& config);
    // Numerical integration helper
    float simpsons_rule(float a, float b, int n, std::function<float(float)> f);
    float activate(float x, const ActivationConfig& config, float distance);
    float activate_derivative(float x, const ActivationConfig& config, float distance);

    // Optimizer and scheduler configuration
    void setOptimizer(const std::string& optimizerType, float learningRate,
                     float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f,
                     float momentum = 0.0f, float rho = 0.9f);
    void setLearningRateScheduler(const std::string& schedulerType, float initialLearningRate,
                                  float decayRate = 0.96f, int decaySteps = 1000,
                                  float decayFactor = 0.1f, int totalEpochs = 100,
                                  float minLearningRate = 0.0f, int warmupEpochs = 0);
    void resetOptimizerState();

    // Regularization and stability configuration
    void setRegularizationParams(float l2_lambda = 0.01f, float dropout_rate = 0.2f,
                                float bn_momentum = 0.1f, float grad_clip_threshold = 1.0f);
    void enableRegularization(bool dropout = true, bool batch_norm = true,
                             bool l2_reg = true, bool grad_clip = true);
    // Multi-layer architecture configuration
    void setLayerArchitecture(const std::vector<int>& layer_sizes);
    void enableResidualConnections(bool enable, const std::vector<std::vector<int>>& connections = {});
    void configureAttention(bool enable, float temperature = 1.0f, float weight = 1.0f);
    void setLayerNeurons(int layer_index, int neurons);

    // 3D transform configuration methods (for auto-tuning)
    void set3DTransformConfig(bool enable_polynomial, bool enable_trigonometric, bool enable_log_exp,
                             bool enable_rbf, bool enable_angular, bool enable_projections,
                             bool enable_ratios, bool enable_transformations);
    void set3DTransformWeights(float polynomial_weight, float trig_weight, float log_exp_weight,
                              float rbf_weight, float angular_weight, float projections_weight,
                              float ratios_weight);

    // Gradient accumulation configuration methods
    void enableGradientAccumulation(bool enable, int accumulation_steps = 4, int max_history = 100);
    void resetGradientAccumulation();
    size_t getGradientHistorySize() const;
    void clearGradientHistory();
    std::vector<std::vector<float>> getAccumulatedGradientsOutputHidden() const;
    std::vector<std::vector<float>> getAccumulatedGradientsInputHidden() const;
};

#endif // CORE_HPP
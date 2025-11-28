#ifndef CORE_HPP
#define CORE_HPP

#include "CoreAI3DCommon.hpp"
#include <memory>

#include "Train.hpp"
#include "Language.hpp"
#include "Database.hpp"
#include "Optimizer.hpp"
#include "LearningRateScheduler.hpp"

// Activation function types
enum class ActivationType {
    SIGMOID,    // Current implementation (for backward compatibility)
    ELU,        // Exponential Linear Unit
    LEAKY_RELU, // Leaky Rectified Linear Unit
    PARAMETRIC_RELU, // Parametric ReLU
    SWISH,      // Swish activation
    GELU,       // Gaussian Error Linear Unit
    SPATIAL_ELU, // Spatial-aware ELU for coordinate data
    SPATIAL_LEAKY_RELU // Spatial-aware Leaky ReLU for coordinate data
};

// Activation configuration structure
struct ActivationConfig {
    ActivationType type = ActivationType::SIGMOID;
    float alpha = 1.0f;  // For ELU, Leaky ReLU
    float beta = 0.1f;   // For Parametric ReLU
    float spatial_weight = 1.0f; // For spatial activations
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
    // Member variables for the transform functions (added for compilation)
    float x;
    float y;
    float z;
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

    // Batch normalization parameters
    std::vector<float> bn_running_mean;
    std::vector<float> bn_running_var;
    std::vector<float> bn_gamma; // scale parameters
    std::vector<float> bn_beta; // shift parameters

    // Dropout masks (per sample during training)
    std::vector<std::vector<float>> dropout_masks;

    // Activation function configurations per layer
    std::vector<ActivationConfig> layer_activations; // One config per layer (including output)

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
        double learningRate, int& numSamples);
    void normalizeInput(float min_val, float max_val);
    void normalizeOutput(float min_val,
        float max_val); // This is likely normalizing targets
    void denormalizeOutput();
    float sigmoid(float x);

    // Optimizer and scheduler configuration
    void setOptimizer(const std::string& optimizerType, float learningRate,
                     float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f,
                     float momentum = 0.0f, float rho = 0.9f);
    void setLearningRateScheduler(const std::string& schedulerType, float initialLearningRate,
                                 float decayRate = 0.96f, int decaySteps = 1000,
                                 float decayFactor = 0.1f, int totalEpochs = 100,
                                 float minLearningRate = 0.0f);
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
    // Activation function configuration
    void setLayerActivation(int layer_index, ActivationType type, float alpha = 1.0f, float beta = 0.1f, float spatial_weight = 1.0f);
    void setAllLayerActivations(ActivationType type, float alpha = 1.0f, float beta = 0.1f, float spatial_weight = 1.0f);
};

#endif // CORE_HPP
#include "Core.hpp"

CoreAI::CoreAI(int inputSize_, int layers_, int neurons_, int outputSize_,
    float min_, float max_)
    : inputSize(inputSize_), layers(layers_), neurons(neurons_),
    outputSize(outputSize_), minVal(min_), maxVal(max_)
{
    this->populateFields(inputSize_, outputSize_);
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

void CoreAI::populateFields(int numInput, int numOutput) {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(minVal, maxVal);

    // Initialize transformation parameters
    this->x = this->beautifulRandom();
    this->y = this->beautifulRandom();
    this->z = this->beautifulRandom();
    this->e = this->epsilon;

    // Initialize vectors
    initializeVector(this->hidden_error, neurons);
    fillVectorWithTransform(this->hidden_error, [this]() { return this->zTransform(this->x, this->y, this->z, 12, 2); });

    initializeVector(this->hidden_output, neurons);
    fillVectorWithTransform(this->hidden_output, [this]() { return this->zTransform2(this->z, this->e, this->z, 2); });

    // Initialize weight matrices
    initializeMatrix(this->weigth_output_hidden, numOutput, neurons);
    fillMatrixWithTransform(this->weigth_output_hidden, [this]() { return this->xTransform2(this->x, this->e, this->y, 2); });

    initializeMatrix(this->weigth_input_hidden, neurons, numInput);
    fillMatrixWithTransform(this->weigth_input_hidden, [this]() { return this->xTransform2(this->x, this->e, this->y, 2); });
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

float CoreAI::beautifulRandom()
{
    static std::mt19937 generator(std::random_device{}());
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
    this->input = inputvalue;
    this->results.clear();
    this->hidden.clear();
    this->output.clear();

    for (size_t i = 0; i < this->input.size(); ++i)
    {
        std::vector<float> hiddenLayer(this->neurons, 0.0f);
        std::vector<float> outputLayer(this->outputSize, 0.0f);

        // Input to hidden layer
        for (int j = 0; j < this->neurons; ++j)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < this->input[i].size(); ++k)
            {
                if (safe2DIndex(this->weigth_input_hidden, j, k))
                {
                    sum += this->input[i][k] * this->weigth_input_hidden[j][k];
                }
            }
            hiddenLayer[j] = this->sigmoid(sum);
        }

        // Hidden to output layer
        for (int j = 0; j < this->outputSize; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < this->neurons; ++k)
            {
                if (safe2DIndex(this->weigth_output_hidden, j, k))
                {
                    sum += hiddenLayer[k] * this->weigth_output_hidden[j][k];
                }
            }
            outputLayer[j] = this->sigmoid(sum);
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
    // Ensure forward pass has been run
    if (this->results.empty()) {
        this->forward(inputs);
    }

    // Backpropagation
    for (size_t sample = 0; sample < inputs.size(); ++sample)
    {
        // Output layer error
        std::vector<float> output_errors(this->outputSize, 0.0f);
        for (int j = 0; j < this->outputSize; ++j)
        {
            float output = this->results[sample][j];
            float target = targets[sample][j];
            output_errors[j] = (target - output) * output * (1.0f - output); // Derivative of sigmoid
        }

        // Hidden layer error
        std::vector<float> hidden_errors(this->neurons, 0.0f);
        for (int j = 0; j < this->neurons; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < this->outputSize; ++k)
            {
                sum += output_errors[k] * this->weigth_output_hidden[k][j];
            }
            float hidden = this->hidden[sample][j];
            hidden_errors[j] = sum * hidden * (1.0f - hidden); // Derivative of sigmoid
        }

        // Update weights output to hidden
        for (int j = 0; j < this->outputSize; ++j)
        {
            for (int k = 0; k < this->neurons; ++k)
            {
                this->weigth_output_hidden[j][k] += learningRate * output_errors[j] * this->hidden[sample][k];
            }
        }

        // Update weights input to hidden
        for (int j = 0; j < this->neurons; ++j)
        {
            for (size_t k = 0; k < inputs[sample].size(); ++k)
            {
                this->weigth_input_hidden[j][k] += learningRate * hidden_errors[j] * inputs[sample][k];
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
    for (auto& row : results)
    { // Denormalize the results/predictions
        for (float& val : row)
        {
            val = val
                * (trainer->original_data_global_max
                    - trainer->original_data_global_min)
                + trainer->original_data_global_min;
        }
    }
}

float CoreAI::xTransform(float x, float y, float z, float n, float m)
{
    if (n <= 1)
    {
        if (y == 0 || m == 0)
            return this->epsilon;

        double denom = y * std::pow(m, 3);
        double num = std::pow(x, 3) * n + z;
        float result = this->epsilon * (num / denom);

        // FIX: Return the result, not whether it's NaN!
        return std::isnan(result) ? this->epsilon : result;
    }
    return this->epsilon;
}

float CoreAI::yTransform(float x, float y, float z, float n, float m)
{
    if (m <= 1)
    {
        float denom = std::pow(x, 3) * n;
        if (std::abs(denom) < this->epsilon)
            denom = this->epsilon;

        float result = this->epsilon * ((std::pow(y, 5) * 2 + std::pow(x, n)) / denom);

        // FIX: Return the result, not whether it's NaN!
        return std::isnan(result) ? this->epsilon : result;
    }
    else
    {
        return this->epsilon;
    }
}

float CoreAI::zTransform(float x, float y, float z, float n, float m)
{
    if (std::abs(y) < this->epsilon)
        y = this->epsilon;

    float result = this->epsilon * (std::pow(z, 3.0f) * std::pow(std::abs(y), n));

    // FIX: Return the result, not whether it's NaN!
    return std::isnan(result) ? this->epsilon : result;
}

float CoreAI::xTransform2(float x, float e, float n, float m)
{
    float result = ((std::pow(x, n) / 5.0f) * std::pow(e, m));
    return result; // Return the calculated result if it's finite and within
    // bounds
}

float CoreAI::yTransform2(float y, float e, float n, float m)
{
    float result = ((std::pow(y, n) / 5) * std::pow(e, m));
    return (std::isnan(result) || std::isinf(result)) ? this->e : result;
}

float CoreAI::zTransform2(float z, float e, float n, float m)
{
    float result = ((std::pow(z, n) * e) / 3.69);
    return (std::isnan(result) || std::isinf(result)) ? this->z : result;
}

#include "Core.hpp"



CoreAI::CoreAI(int inputSize_, int layers_, int neurons_, int outputSize_, float min_, float max_)
    : inputSize(inputSize_), layers(layers_), neurons(neurons_),
    outputSize(outputSize_), minVal(min_), maxVal(max_)
{
    //populateFields(inputSize, outputSize);
}
std::vector<std::vector<float>>& CoreAI::getInput() {
    return this->input;
}

std::vector<std::vector<float>>& CoreAI::getOutput() {
    return this->output;
}

std::vector<std::vector<float>>& CoreAI::getResults() {
    return this->results;
}

// Implementations for getters (example for one, others similar)
const std::vector<std::vector<float>>& CoreAI::getHiddenData() const {
    return this->hidden; // Assuming 'hiddenData' is a member
}
const std::vector<float>& CoreAI::getHiddenOutputData() const {
    return this->hidden_output;
}
const std::vector<float>& CoreAI::getHiddenErrorData() const {
    return this->hidden_error;
}
const std::vector<std::vector<float>>& CoreAI::getWeightsHiddenInput() const {
    return this->weigth_input_hidden;
}
const std::vector<std::vector<float>>& CoreAI::getWeightsOutputHidden() const {
    return this->weigth_output_hidden;
}

// Implementations for setters (example for one, others similar)
void CoreAI::setInput(const std::vector<std::vector<float>>& data) {
    this->input = data;
}
void CoreAI::setOutput(const std::vector<std::vector<float>>& data) {
    this->output = data; // Assuming this is for targets, or actual model output
}

float CoreAI::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void CoreAI::setTarget(const std::vector<std::vector<float>>& data) {
    // If output is used for targets, then assign:
    this->output = data;
    // Or if you have a separate 'targets' member:
    // this->targets = data;
}
void CoreAI::setHiddenData(const std::vector<std::vector<float>>& data) {
    this->hidden = data;
}
void CoreAI::setHiddenOutputData(const std::vector<float>& data) {
    this->hidden_output = data;
}
void CoreAI::setHiddenErrorData(const std::vector<float>& data) {
    this->hidden_error = data;
}
void CoreAI::setWeightsHiddenInput(const std::vector<std::vector<float>>& data) {
    this->weigth_input_hidden = data;
}
void CoreAI::setWeightsOutputHidden(const std::vector<std::vector<float>>& data) {
    this->weigth_output_hidden = data;
}

//void CoreAI::populateFields(int numInput, int numOutput) {
//    std::mt19937 gen(std::random_device{}());
//    std::uniform_real_distribution<float> dist(0, 10);
//
//    this->x = this->beautifulRandom();
//    this->y = this->beautifulRandom();
//    this->z = this->beautifulRandom();
//    this->e = this->epsilon;
//    this->hidden = this->getHiddenData();
//
//    this->input.resize(numInput);
//    for (auto& row : this->input) {
//        row.resize(numInput);
//    }
//
//    for (auto& row : this->input)
//        for (auto& val : row)
//            val = this->xTransform2(this->beautifulRandom(), this->e, this->x, 2); // or any transform
//
//
//    this->hidden.resize(numInput);
//    for (auto& row : this->hidden) {
//        row.resize(numInput);
//    }
//
//    for (auto& row : this->hidden)
//        for (auto& val : row)
//            val = this->yTransform2(this->beautifulRandom(), this->e, this->y, 2);
//
//    this->output.resize(numOutput);
//    for (auto& row : this->output) {
//        row.resize(numOutput);
//    }
//
//    for (auto& row : this->output)
//        for (auto& val : row)
//            val = this->zTransform2(this->z, this->e, this->z, 2);
//    
//    this->hidden_error.resize(numOutput);
//
//    for (auto& row : this->hidden_error)
//        row = this->zTransform(this->x, this->y, this->z, 12, 2);
//
//    this->output.resize(numOutput);
//    for (auto& row : this->output) {
//        row.resize(numOutput);
//    }
//
//    this->hidden_output.resize(numOutput);
//    for (auto& row : this->hidden_output)
//        row = this->zTransform2(this->z, this->e, this->z, 2);
//
//    this->weigth_output_hidden.resize(numInput);
//    for (auto& row : this->weigth_output_hidden) {
//        row.resize(numInput);
//    }
//
//    for (auto& row : this->weigth_output_hidden)
//        for (auto& val : row)
//            val = this->xTransform2(this->x, this->e, this->y, 2);
//
//    this->weigth_input_hidden.resize(numInput);
//    for (auto& row : this->weigth_input_hidden) {
//        row.resize(numInput);
//    }
//
//    for (auto& row : this->weigth_input_hidden)
//        for (auto& val : row)
//            val = this->xTransform2(this->x, this->e, this->y, 2);
//
//}

float CoreAI::beautifulRandom() {
    static std::mt19937 generator(std::random_device{}());
    static std::uniform_real_distribution<float> distribution(this->minVal, this->maxVal);
    return distribution(generator);
}

bool CoreAI::safe1DIndex(const std::vector<float>& vec, int index) {
    return index >= 0 && index < (int)vec.size();
}

bool CoreAI::safe2DIndex(const std::vector<std::vector<float>>& matrix, int i, int j) {
    return i >= 0 && i < matrix.size() &&
        j >= 0 && j < matrix[i].size();
}

std::vector<std::vector<float>> CoreAI::forward(const std::vector<std::vector<float>>& inputvalue) {
    this->input = inputvalue;

    for (int i = 0; i < this->input.size(); ++i) {
        std::vector hiddenLayer(this->hidden.size(), 0.0f);
        std::vector outputLayer(this->output.size(), 0.0f);

        for (int j = 0; j < this->hidden.size(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < this->input.size(); ++k) {
                if (k < this->input[i].size()) {
                    if (safe2DIndex(this->input, i, k) && safe2DIndex(this->weigth_input_hidden, j, k)) {
                        sum += this->input[i][k] * this->weigth_input_hidden[j][k];
                    }
                    else
                    {
                        sum = this->beautifulRandom();
                    }
                }
            }
            hiddenLayer[j] = sum;
        }

        for (int j = 0; j < this->output.size(); ++j) {
            float sum = 0.0f;
            for (int k = 0; k < this->hidden.size(); ++k) {
                if (safe2DIndex(this->weigth_output_hidden, j, k)) {
                    sum += hiddenLayer[k] * this->sigmoid(this->weigth_output_hidden[j][k]);
                }
                else
                {
                    sum = this->beautifulRandom();
                }
            }
            outputLayer[j] = sum;
        }

        this->hidden.push_back(hiddenLayer);
        this->output.push_back(outputLayer);
        this->results.push_back(outputLayer);
    }

    return this->output;
}


void CoreAI::train(const std::vector<std::vector<float>>& inputs,
    const std::vector<std::vector<float>>& targets,
    double learningRate) {
    float targetVal;
    for (int i = 0; i < this->output.size(); ++i) {
        bool skip = false;
        for (int j = 0; j < this->output[i].size(); ++j) {
            skip = false;
            if (safe2DIndex(this->output, i, j)) {
                float outputVal = this->output[i][j];
                if (safe2DIndex(targets, i, j)) {
                    targetVal = targets[i][j];
                }
                else
                {
                    skip = true;
                    break;
                }
                this->hidden_output.push_back((targetVal - this->sigmoid(outputVal)) * this->sigmoid(outputVal));
            }
        }
    }

   ;
    for (int n = 0; n < hidden.size(); ++n) {
        bool skip = false;
        for (int m = 0; m < hidden[n].size(); ++m) {
            float sum = 0.0f;
            skip = false;
            for (int o = 0; o < hidden_output.size(); ++o) {
                if (safe2DIndex(weigth_output_hidden, n, o)) {
                    sum += this->sigmoid(hidden_output[o] * weigth_output_hidden[n][o]);
                }
                else
                {
                    skip = true;
                    break;
                }
            }

            if (safe2DIndex(hidden, n, m) && safe1DIndex(hidden_error, n)) {
                hidden_error[n] = this->sigmoid(sum * hidden[n][m]);
            }
            else
            {
                skip = true;
                break;
            }
        }
    }


    for (int n = 0; n < this->hidden.size(); ++n) {
        bool skip = false;
        for (int m = 0; m < this->output.size(); ++m) {
            skip = false;
            if (safe2DIndex(this->input, n, m) && safe2DIndex(this->output, n, m) && safe2DIndex(this->weigth_input_hidden, n, m) && safe2DIndex(this->weigth_output_hidden, n, m) && safe1DIndex(this->hidden_error, n) && safe1DIndex(this->hidden_output, n)) {
                this->x = this->xTransform2(this->x, this->e, this->input.max_size(), trainer->numSamples);
                this->y = this->yTransform2(this->y, this->e, this->input.max_size(), trainer->numSamples);
                this->z = this->zTransform2(this->z, this->e, this->input.max_size(), trainer->numSamples);
                float delta = this->sigmoid(learningRate * this->hidden_output[m] * this->output[n][m] * this->hidden[n][m]);
                this->weigth_output_hidden[n][m] += delta;
            }
            else
            {
                skip = true;
                break;
            }
        }
    }

    for (int n = 0; n < this->input.size(); ++n) {
        bool skip = false;
        for (int m = 0; m < this->hidden[n].size(); ++m) {
            skip = false;
            if (safe2DIndex(this->hidden, n, m) && safe2DIndex(this->weigth_output_hidden, n, m) && safe2DIndex(this->output, n, m) && safe2DIndex(this->input, n, m) && safe2DIndex(this->weigth_input_hidden, n, m)) {
                skip = false;
                this->x = this->xTransform(this->x, this->y, this-> z, this->input.max_size(), trainer->numSamples);
                this->y = this->yTransform(this->x, this->y, this->z, this->input.max_size(), trainer->numSamples);
                this->z = this->zTransform(this->x, this->y, this->z, this->input.max_size(), trainer->numSamples);
                if (safe1DIndex(this->hidden_error, n) && safe2DIndex(this->hidden, n, m) && safe2DIndex(this->input, n, m) && safe2DIndex(this->weigth_input_hidden, n, m)) {
                    float delta = this->sigmoid(learningRate * this->hidden_error[m] * this->hidden[n][m] * this->input[n][m]);
                    this->weigth_input_hidden[n][m] += delta;
                }
                else
                {
                    skip = true;
                    break;
                }
            }
            else
            {
                break;
            }
        }
    }
}

void CoreAI::normalizeInput(float min_val, float max_val) {
    for (auto& row : input) {
        for (float& val : row) {
            val = (val - min_val) / (max_val - min_val);
        }
    }
}

void CoreAI::normalizeOutput(float min_val, float max_val) {
    for (auto& row : output) {
        for (float& val : row) {
            val = (val - min_val) / (max_val - min_val);
        }
    }
}

void CoreAI::denormalizeOutput() {
    for (auto& row : results) { // Denormalize the results/predictions
        for (float& val : row) {
            val = val * (trainer->original_data_global_max - trainer->original_data_global_min) + trainer->original_data_global_min;
        }
    }
}


float CoreAI::xTransform(float x, float y, float z, float n, float m) {
    if (n <= 1) {
        if (y == 0 || m == 0) return this->epsilon;

        double denom = y * std::pow(m, 3);
        double num = std::pow(x, 3) * n + z;
        float result = this->epsilon * (num / denom);
        return std::isnan(result);
    }
    return this->epsilon;
}

float CoreAI::yTransform(float x, float y, float z, float n, float m) {
    if (m <= 1) {
        float denom = std::pow(x, 3) * n;
        if (std::abs(denom) < this->epsilon) denom = this->epsilon;
        return std::isnan(this->epsilon * ((std::pow(y, 5) * 2 + std::pow(x, n)) / denom));
    }
    else {
        return this->epsilon;
    }
}

float CoreAI::zTransform(float x, float y, float z, float n, float m) {
    if (std::abs(y) < this->epsilon) y = this->epsilon;
    return std::isnan(this->epsilon * (std::pow(z, 3.0f) * std::pow(std::abs(y), n)));
}

float CoreAI::xTransform2(float x, float e, float n, float m) {
    float result = ((std::pow(x, n) / 5.0f) * std::pow(e, m));
    return result; // Return the calculated result if it's finite and within bounds
}

float CoreAI::yTransform2(float y, float e, float n, float m) {
    float result = ((std::pow(y, n) / 5) * std::pow(e, m));
    return (std::isnan(result) || std::isinf(result)) ? this->e : result;
}

float CoreAI::zTransform2(float z, float e, float n, float m) {
    float result = ((std::pow(z, n) * e) / 3.69);
    return (std::isnan(result) || std::isinf(result)) ? this->z : result;
}

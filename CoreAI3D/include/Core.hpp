#ifndef CORE_HPP
#define CORE_HPP

#include "main.hpp"

#include "Train.hpp"
#include "Database.hpp"
#include "Language.hpp"

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
    // Member variables for the transform functions (added for compilation)
    float x;
    float y;
    float z;

    Training* getTrainer();

    std::vector<std::vector<float> > input;
    std::vector<std::vector<float> > output; // This could be the targets or actual output after forward pass
    std::vector<std::vector<float> > results; // This will store the final output of the forward pass

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

public:
    CoreAI(int inputSize_, int layers_, int neurons_, int outputSize_,
        float min_, float max_);


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

    std::unique_ptr<Training> trainer;

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
    void populateFields(int numInput,
        int numOutput); // Initializes weights, etc.
    std::vector<std::vector<float> >
        forward(const std::vector<std::vector<float> >& inputData);
    void train(const std::vector<std::vector<float> >& inputData,
        const std::vector<std::vector<float> >& targetData,
        double learningRate, int& numSamples);
    void normalizeInput(float min_val, float max_val);
    void normalizeOutput(float min_val,
        float max_val); // This is likely normalizing targets
    void denormalizeOutput();
    float sigmoid(float x);
};

#endif // CORE_HPP
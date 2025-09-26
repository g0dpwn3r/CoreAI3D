#ifndef VISION_MODULE_HPP
#define VISION_MODULE_HPP

#include "main.hpp"
#include "Core.hpp"
#include <vector>
#include <string>
#include <memory>

// Forward declarations for OpenCV types (to avoid including OpenCV headers)
struct cv_Mat;
struct cv_VideoCapture;
struct cv_VideoWriter;

class VisionModule {
private:
    std::unique_ptr<CoreAI> visionCore;
    std::string moduleName;
    int inputWidth, inputHeight;
    int channels;
    bool isInitialized;

    // Numerical processing buffers
    std::vector<std::vector<float>> numericalBuffer;
    std::vector<std::vector<float>> featureVector;

    // Image processing parameters
    float confidenceThreshold;
    int maxDetections;
    bool enableGPU;

protected:
    // Core vision processing functions
    virtual std::vector<float> processImageFeatures(const std::vector<float>& imageData);
    virtual std::vector<float> extractNumericalFeatures(cv_Mat* image);
    virtual std::vector<std::vector<float>> detectObjects(const std::vector<float>& features);
    virtual std::string classifyImage(const std::vector<float>& features);

    // Utility functions for numerical processing
    std::vector<float> flattenImageMatrix(const std::vector<std::vector<float>>& matrix);
    std::vector<std::vector<float>> normalizeImageData(const std::vector<std::vector<float>>& data);
    float calculateConfidenceScore(const std::vector<float>& prediction);

public:
    // Constructor
    VisionModule(const std::string& name, int width = 224, int height = 224, int ch = 3);
    virtual ~VisionModule();

    // Initialization
    bool initialize(const std::string& modelPath = "");
    void setInputDimensions(int width, int height, int channels = 3);
    void setConfidenceThreshold(float threshold);
    void setMaxDetections(int maxDet);
    void enableGPUAcceleration(bool enable);

    // Core processing interface
    virtual std::vector<std::vector<float>> processImage(const std::string& imagePath);
    virtual std::vector<std::vector<float>> processImageData(const std::vector<float>& imageData);
    virtual std::vector<std::vector<float>> processVideoStream(const std::string& videoPath, int frameSkip = 1);

    // Feature extraction
    std::vector<float> extractFeatures(const std::string& imagePath);
    std::vector<std::vector<float>> batchExtractFeatures(const std::vector<std::string>& imagePaths);

    // Object detection interface
    struct DetectionResult {
        float confidence;
        int classId;
        std::vector<float> boundingBox; // x, y, width, height
        std::string className;
    };

    virtual std::vector<DetectionResult> detectObjects(const std::string& imagePath);
    virtual std::vector<DetectionResult> detectObjectsFromData(const std::vector<float>& imageData);

    // Classification interface
    struct ClassificationResult {
        float confidence;
        int classId;
        std::string className;
        std::vector<float> probabilities;
    };

    virtual ClassificationResult classify(const std::string& imagePath);
    virtual ClassificationResult classifyFromData(const std::vector<float>& imageData);

    // Image manipulation
    virtual std::vector<float> resizeImage(const std::vector<float>& imageData, int newWidth, int newHeight);
    virtual std::vector<float> cropImage(const std::vector<float>& imageData, int x, int y, int width, int height);
    virtual std::vector<float> enhanceImage(const std::vector<float>& imageData);

    // Video processing
    virtual bool processVideoFile(const std::string& inputPath, const std::string& outputPath, int frameInterval = 1);
    virtual std::vector<std::vector<float>> processVideoFrames(const std::string& videoPath, int maxFrames = -1);

    // OCR interface
    virtual std::string performOCR(const std::string& imagePath);
    virtual std::string performOCROnData(const std::vector<float>& imageData);

    // Facial recognition interface
    struct FaceResult {
        std::vector<float> faceEmbedding;
        std::vector<float> boundingBox;
        float confidence;
        std::string faceId;
    };

    virtual std::vector<FaceResult> detectFaces(const std::string& imagePath);
    virtual std::vector<FaceResult> detectFacesFromData(const std::vector<float>& imageData);
    virtual float compareFaces(const std::vector<float>& face1, const std::vector<float>& face2);

    // Medical imaging interface
    virtual std::vector<float> analyzeMedicalImage(const std::string& imagePath, const std::string& modality);
    virtual std::vector<float> segmentMedicalImage(const std::string& imagePath);

    // Utility functions
    std::vector<float> imagePathToNumericalData(const std::string& imagePath);
    std::string numericalDataToImagePath(const std::vector<float>& data, const std::string& outputPath);

    // Status and information
    bool isReady() const { return isInitialized; }
    std::string getModuleName() const { return moduleName; }
    int getInputWidth() const { return inputWidth; }
    int getInputHeight() const { return inputHeight; }
    int getChannels() const { return channels; }

    // Memory management
    void clearBuffers();
    size_t getMemoryUsage() const;

    // Training interface for vision-specific learning
    virtual bool trainOnImageDataset(const std::string& datasetPath, int epochs = 10);
    virtual bool fineTuneModel(const std::vector<std::string>& imagePaths, const std::vector<int>& labels);

    // Real-time processing
    virtual std::vector<float> processRealTimeFrame(const std::vector<float>& frameData);
    virtual bool initializeRealTimeProcessing(int cameraId = 0);
    virtual void stopRealTimeProcessing();
};

#endif // VISION_MODULE_HPP
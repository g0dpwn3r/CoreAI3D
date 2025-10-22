#include "VisionModule.hpp"
#include "Core.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// OpenCV forward declarations (actual OpenCV includes would be added when available)
struct cv_Mat {
    int rows, cols, channels;
    std::vector<float> data;
    cv_Mat(int r, int c, int ch) : rows(r), cols(c), channels(ch), data(r*c*ch, 0.0f) {}
    cv_Mat(const cv::Mat& mat) : rows(mat.rows), cols(mat.cols), channels(mat.channels()) {
        data.resize(rows * cols * channels);
        if (mat.isContinuous()) {
            std::memcpy(data.data(), mat.data, data.size() * sizeof(float));
        } else {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    for (int k = 0; k < channels; ++k) {
                        data[i * cols * channels + j * channels + k] = mat.at<cv::Vec3b>(i, j)[k] / 255.0f;
                    }
                }
            }
        }
    }
};

struct cv_VideoCapture {
    cv::VideoCapture cap;
    cv_VideoCapture(const std::string& path) : cap(path) {}
    cv_VideoCapture(int device) : cap(device) {}
    bool read(cv_Mat& frame) {
        cv::Mat cvFrame;
        bool success = cap.read(cvFrame);
        if (success) {
            frame = cv_Mat(cvFrame);
        }
        return success;
    }
    bool isOpened() const { return cap.isOpened(); }
};

struct cv_VideoWriter {
    cv::VideoWriter writer;
    cv_VideoWriter(const std::string& filename, int fourcc, double fps, cv_Mat frame) {
        cv::Mat cvFrame(frame.rows, frame.cols, CV_8UC3);
        writer.open(filename, fourcc, fps, cvFrame.size());
    }
    void write(const cv_Mat& frame) {
        cv::Mat cvFrame(frame.rows, frame.cols, CV_8UC3);
        // Convert float data back to uint8
        for (int i = 0; i < frame.rows; ++i) {
            for (int j = 0; j < frame.cols; ++j) {
                for (int k = 0; k < frame.channels; ++k) {
                    cvFrame.at<cv::Vec3b>(i, j)[k] = static_cast<unsigned char>(
                        frame.data[i * frame.cols * frame.channels + j * frame.channels + k] * 255.0f);
                }
            }
        }
        writer.write(cvFrame);
    }
    bool isOpened() const { return writer.isOpened(); }
};

// VisionModule implementation
VisionModule::VisionModule(const std::string& name, int width, int height, int ch)
    : moduleName(name), inputWidth(width), inputHeight(height), channels(ch), isInitialized(false),
      confidenceThreshold(0.5f), maxDetections(100), enableGPU(false) {

    // Initialize vision-specific CoreAI instance
    visionCore = std::make_unique<CoreAI>(width * height * ch, 3, 64, 10, 0.0f, 1.0f);
}

VisionModule::~VisionModule() {
    clearBuffers();
}

bool VisionModule::initialize(const std::string& modelPath) {
    if (isInitialized) return true;

    try {
        // Initialize numerical buffers
        numericalBuffer.resize(inputHeight, std::vector<float>(inputWidth * channels, 0.0f));
        featureVector.resize(1, std::vector<float>(1024, 0.0f)); // Default feature size

        // Initialize CoreAI for vision processing
        // populateFields() removed - initialization handled in CoreAI constructor

        isInitialized = true;
        std::cout << "VisionModule '" << moduleName << "' initialized successfully" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to initialize VisionModule: " << e.what() << std::endl;
        return false;
    }
}

void VisionModule::setInputDimensions(int width, int height, int ch) {
    inputWidth = width;
    inputHeight = height;
    channels = ch;

    // Resize buffers
    numericalBuffer.resize(height, std::vector<float>(width * ch, 0.0f));

    // Recreate CoreAI with new dimensions
    visionCore = std::make_unique<CoreAI>(width * height * ch, 3, 64, 10, 0.0f, 1.0f);
}

void VisionModule::setConfidenceThreshold(float threshold) {
    confidenceThreshold = std::max(0.0f, std::min(1.0f, threshold));
}

void VisionModule::setMaxDetections(int maxDet) {
    maxDetections = std::max(1, maxDet);
}

void VisionModule::enableGPUAcceleration(bool enable) {
    enableGPU = enable;
}

std::vector<float> VisionModule::processImageFeatures(const std::vector<float>& imageData) {
    if (!visionCore) return {};

    // Convert flat image data to matrix format
    std::vector<std::vector<float>> inputMatrix(1, imageData);

    // Forward pass through vision core
    auto results = visionCore->forward(inputMatrix);

    // Extract features (flatten the results)
    std::vector<float> features;
    for (const auto& row : results) {
        features.insert(features.end(), row.begin(), row.end());
    }

    return features;
}

std::vector<float> VisionModule::extractNumericalFeatures(cv_Mat* image) {
    std::vector<float> features;

    if (!image || image->data.empty()) return features;

    // Basic feature extraction - convert image data to numerical features
    size_t totalPixels = image->rows * image->cols * image->channels;
    features.reserve(totalPixels);

    // Normalize and flatten image data
    for (float pixel : image->data) {
        features.push_back(pixel / 255.0f); // Normalize to [0,1]
    }

    return features;
}

std::vector<std::vector<float>> VisionModule::detectObjects(const std::vector<float>& features) {
    // Base implementation - override in derived classes
    return {features}; // Return input as single detection
}

std::string VisionModule::classifyImage(const std::vector<float>& features) {
    // Base implementation - override in derived classes
    return "unknown";
}

std::vector<std::vector<float>> VisionModule::processImage(const std::string& imagePath) {
    std::vector<float> imageData = imagePathToNumericalData(imagePath);
    return processImageData(imageData);
}

std::vector<std::vector<float>> VisionModule::processImageData(const std::vector<float>& imageData) {
    if (!isInitialized || imageData.empty()) return {};

    try {
        // Process features through vision core
        std::vector<float> features = processImageFeatures(imageData);

        // Apply object detection
        auto detections = detectObjects(features);

        return detections;
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing image data: " << e.what() << std::endl;
        return {};
    }
}

std::vector<std::vector<float>> VisionModule::processVideoStream(const std::string& videoPath, int frameSkip) {
    std::vector<std::vector<float>> allFrames;

    try {
        cv_VideoCapture cap(videoPath);
        if (!cap.isOpened()) return allFrames;

        cv_Mat frame(inputHeight, inputWidth, channels);
        int frameCount = 0;

        while (cap.read(frame)) {
            if (frameCount++ % (frameSkip + 1) != 0) continue;

            // Extract features from frame
            std::vector<float> features = extractNumericalFeatures(&frame);
            if (!features.empty()) {
                allFrames.push_back(features);
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing video stream: " << e.what() << std::endl;
    }

    return allFrames;
}

std::vector<float> VisionModule::extractFeatures(const std::string& imagePath) {
    std::vector<float> imageData = imagePathToNumericalData(imagePath);
    return processImageFeatures(imageData);
}

std::vector<std::vector<float>> VisionModule::batchExtractFeatures(const std::vector<std::string>& imagePaths) {
    std::vector<std::vector<float>> allFeatures;

    for (const auto& path : imagePaths) {
        auto features = extractFeatures(path);
        if (!features.empty()) {
            allFeatures.push_back(features);
        }
    }

    return allFeatures;
}

std::vector<VisionModule::DetectionResult> VisionModule::detectObjects(const std::string& imagePath) {
    std::vector<float> imageData = imagePathToNumericalData(imagePath);
    return detectObjectsFromData(imageData);
}

std::vector<VisionModule::DetectionResult> VisionModule::detectObjectsFromData(const std::vector<float>& imageData) {
    std::vector<DetectionResult> results;

    try {
        auto detections = processImageData(imageData);

        // Convert numerical detections to DetectionResult objects
        for (size_t i = 0; i < detections.size() && i < maxDetections; ++i) {
            if (detections[i].size() >= 5) { // Need at least confidence + bbox
                DetectionResult result;
                result.confidence = detections[i][0];
                result.classId = static_cast<int>(detections[i][1]);
                result.boundingBox = {detections[i][2], detections[i][3], detections[i][4], detections[i][5]};
                result.className = "object_" + std::to_string(result.classId);

                if (result.confidence >= confidenceThreshold) {
                    results.push_back(result);
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error detecting objects: " << e.what() << std::endl;
    }

    return results;
}

VisionModule::ClassificationResult VisionModule::classify(const std::string& imagePath) {
    std::vector<float> imageData = imagePathToNumericalData(imagePath);
    return classifyFromData(imageData);
}

VisionModule::ClassificationResult VisionModule::classifyFromData(const std::vector<float>& imageData) {
    ClassificationResult result;

    try {
        std::vector<float> features = processImageFeatures(imageData);
        std::string className = classifyImage(features);

        result.className = className;
        result.confidence = calculateConfidenceScore(features);
        result.classId = 0; // Base implementation
        result.probabilities = features; // Use features as probabilities

    }
    catch (const std::exception& e) {
        std::cerr << "Error classifying image: " << e.what() << std::endl;
        result.className = "error";
        result.confidence = 0.0f;
    }

    return result;
}

std::vector<float> VisionModule::flattenImageMatrix(const std::vector<std::vector<float>>& matrix) {
    std::vector<float> flattened;
    for (const auto& row : matrix) {
        flattened.insert(flattened.end(), row.begin(), row.end());
    }
    return flattened;
}

std::vector<std::vector<float>> VisionModule::normalizeImageData(const std::vector<std::vector<float>>& data) {
    std::vector<std::vector<float>> normalized = data;

    // Find min and max values
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::min();

    for (const auto& row : data) {
        for (float val : row) {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }

    // Normalize to [0, 1]
    if (maxVal > minVal) {
        for (auto& row : normalized) {
            for (float& val : row) {
                val = (val - minVal) / (maxVal - minVal);
            }
        }
    }

    return normalized;
}

float VisionModule::calculateConfidenceScore(const std::vector<float>& prediction) {
    if (prediction.empty()) return 0.0f;

    // Simple confidence calculation - average of absolute values
    float sum = 0.0f;
    for (float val : prediction) {
        sum += std::abs(val);
    }
    return sum / prediction.size();
}

// Placeholder implementations for other methods
std::vector<float> VisionModule::resizeImage(const std::vector<float>& imageData, int newWidth, int newHeight) {
    try {
        // Reconstruct image from numerical data
        cv::Mat image(inputHeight, inputWidth, CV_32FC3);

        for (int i = 0; i < inputHeight; ++i) {
            for (int j = 0; j < inputWidth; ++j) {
                for (int k = 0; k < channels; ++k) {
                    size_t idx = i * inputWidth * channels + j * channels + k;
                    if (idx < imageData.size()) {
                        image.at<cv::Vec3f>(i, j)[k] = imageData[idx];
                    }
                }
            }
        }

        // Resize the image
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(newWidth, newHeight));

        // Flatten back to vector
        std::vector<float> resizedData(newWidth * newHeight * channels);
        if (resizedImage.isContinuous()) {
            std::memcpy(resizedData.data(), resizedImage.data, resizedData.size() * sizeof(float));
        } else {
            for (int i = 0; i < newHeight; ++i) {
                for (int j = 0; j < newWidth; ++j) {
                    for (int k = 0; k < channels; ++k) {
                        resizedData[i * newWidth * channels + j * channels + k] =
                            resizedImage.at<cv::Vec3f>(i, j)[k];
                    }
                }
            }
        }

        return resizedData;
    }
    catch (const std::exception& e) {
        std::cerr << "Error resizing image: " << e.what() << std::endl;
        return imageData; // Return original on error
    }
}

std::vector<float> VisionModule::cropImage(const std::vector<float>& imageData, int x, int y, int width, int height) {
    try {
        // Reconstruct image from numerical data
        cv::Mat image(inputHeight, inputWidth, CV_32FC3);

        for (int i = 0; i < inputHeight; ++i) {
            for (int j = 0; j < inputWidth; ++j) {
                for (int k = 0; k < channels; ++k) {
                    size_t idx = i * inputWidth * channels + j * channels + k;
                    if (idx < imageData.size()) {
                        image.at<cv::Vec3f>(i, j)[k] = imageData[idx];
                    }
                }
            }
        }

        // Define crop rectangle
        cv::Rect cropRect(x, y, std::min(width, inputWidth - x), std::min(height, inputHeight - y));
        cv::Mat croppedImage = image(cropRect);

        // Flatten back to vector
        std::vector<float> croppedData(cropRect.width * cropRect.height * channels);
        if (croppedImage.isContinuous()) {
            std::memcpy(croppedData.data(), croppedImage.data, croppedData.size() * sizeof(float));
        } else {
            for (int i = 0; i < cropRect.height; ++i) {
                for (int j = 0; j < cropRect.width; ++j) {
                    for (int k = 0; k < channels; ++k) {
                        croppedData[i * cropRect.width * channels + j * channels + k] =
                            croppedImage.at<cv::Vec3f>(i, j)[k];
                    }
                }
            }
        }

        return croppedData;
    }
    catch (const std::exception& e) {
        std::cerr << "Error cropping image: " << e.what() << std::endl;
        return imageData; // Return original on error
    }
}

std::vector<float> VisionModule::enhanceImage(const std::vector<float>& imageData) {
    try {
        // Reconstruct image from numerical data
        cv::Mat image(inputHeight, inputWidth, CV_32FC3);

        for (int i = 0; i < inputHeight; ++i) {
            for (int j = 0; j < inputWidth; ++j) {
                for (int k = 0; k < channels; ++k) {
                    size_t idx = i * inputWidth * channels + j * channels + k;
                    if (idx < imageData.size()) {
                        image.at<cv::Vec3f>(i, j)[k] = imageData[idx];
                    }
                }
            }
        }

        // Convert to 8-bit for OpenCV operations
        cv::Mat uint8Image;
        image.convertTo(uint8Image, CV_8UC3, 255.0);

        // Apply histogram equalization for enhancement
        cv::Mat enhancedImage;
        if (channels == 3) {
            cv::cvtColor(uint8Image, uint8Image, cv::COLOR_RGB2YCrCb);
            std::vector<cv::Mat> channels_vec;
            cv::split(uint8Image, channels_vec);
            cv::equalizeHist(channels_vec[0], channels_vec[0]);
            cv::merge(channels_vec, uint8Image);
            cv::cvtColor(uint8Image, enhancedImage, cv::COLOR_YCrCb2RGB);
        } else {
            cv::equalizeHist(uint8Image, enhancedImage);
        }

        // Convert back to float
        enhancedImage.convertTo(enhancedImage, CV_32F, 1.0 / 255.0);

        // Flatten back to vector
        std::vector<float> enhancedData(inputWidth * inputHeight * channels);
        if (enhancedImage.isContinuous()) {
            std::memcpy(enhancedData.data(), enhancedImage.data, enhancedData.size() * sizeof(float));
        } else {
            for (int i = 0; i < inputHeight; ++i) {
                for (int j = 0; j < inputWidth; ++j) {
                    for (int k = 0; k < channels; ++k) {
                        enhancedData[i * inputWidth * channels + j * channels + k] =
                            enhancedImage.at<cv::Vec3f>(i, j)[k];
                    }
                }
            }
        }

        return enhancedData;
    }
    catch (const std::exception& e) {
        std::cerr << "Error enhancing image: " << e.what() << std::endl;
        return imageData; // Return original on error
    }
}

bool VisionModule::processVideoFile(const std::string& inputPath, const std::string& outputPath, int frameInterval) {
    try {
        cv_VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << inputPath << std::endl;
            return false;
        }

        cv_VideoWriter writer(outputPath, -1, 30.0, cv_Mat(inputHeight, inputWidth, channels)); // FourCC -1 for default
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not create output video file: " << outputPath << std::endl;
            return false;
        }

        cv_Mat frame(inputHeight, inputWidth, channels);
        int frameCount = 0;

        while (cap.read(frame)) {
            if (frameCount++ % frameInterval == 0) {
                // Process frame if needed
                std::vector<float> processedData = processImageFeatures(extractNumericalFeatures(&frame));
                // Convert back to cv_Mat format if needed for writing
                writer.write(frame);
            }
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing video file: " << e.what() << std::endl;
        return false;
    }
}

std::vector<std::vector<float>> VisionModule::processVideoFrames(const std::string& videoPath, int maxFrames) {
    std::vector<std::vector<float>> allFrames;

    try {
        cv_VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
            return allFrames;
        }

        cv_Mat frame(inputHeight, inputWidth, channels);
        int frameCount = 0;

        while (cap.read(frame) && (maxFrames == -1 || frameCount < maxFrames)) {
            std::vector<float> features = extractNumericalFeatures(&frame);
            if (!features.empty()) {
                allFrames.push_back(features);
            }
            frameCount++;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing video frames: " << e.what() << std::endl;
    }

    return allFrames;
}

std::string VisionModule::performOCR(const std::string& imagePath) {
    return "OCR not implemented in base class"; // Base implementation
}

std::string VisionModule::performOCROnData(const std::vector<float>& imageData) {
    return "OCR not implemented in base class"; // Base implementation
}

std::vector<VisionModule::FaceResult> VisionModule::detectFaces(const std::string& imagePath) {
    return {}; // Base implementation - no faces detected
}

std::vector<VisionModule::FaceResult> VisionModule::detectFacesFromData(const std::vector<float>& imageData) {
    return {}; // Base implementation - no faces detected
}

float VisionModule::compareFaces(const std::vector<float>& face1, const std::vector<float>& face2) {
    return 0.0f; // Base implementation - no similarity
}

std::vector<float> VisionModule::analyzeMedicalImage(const std::string& imagePath, const std::string& modality) {
    return {}; // Base implementation - no analysis
}

std::vector<float> VisionModule::segmentMedicalImage(const std::string& imagePath) {
    return {}; // Base implementation - no segmentation
}

std::vector<float> VisionModule::imagePathToNumericalData(const std::string& imagePath) {
    std::vector<float> data;

    try {
        // Load image using OpenCV
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Could not load image: " << imagePath << std::endl;
            return data;
        }

        // Convert to RGB if necessary
        if (image.channels() == 3) {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        }

        // Resize to input dimensions
        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(inputWidth, inputHeight));

        // Convert to float and normalize to [0,1]
        resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

        // Flatten the image data
        size_t expectedSize = inputWidth * inputHeight * channels;
        data.resize(expectedSize);

        if (resizedImage.isContinuous()) {
            std::memcpy(data.data(), resizedImage.data, expectedSize * sizeof(float));
        } else {
            for (int i = 0; i < inputHeight; ++i) {
                for (int j = 0; j < inputWidth; ++j) {
                    for (int k = 0; k < channels; ++k) {
                        data[i * inputWidth * channels + j * channels + k] =
                            resizedImage.at<cv::Vec3f>(i, j)[k];
                    }
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading image file: " << e.what() << std::endl;
    }

    return data;
}

std::string VisionModule::numericalDataToImagePath(const std::vector<float>& data, const std::string& outputPath) {
    try {
        // Reconstruct image from numerical data
        cv::Mat image(inputHeight, inputWidth, CV_32FC3);

        for (int i = 0; i < inputHeight; ++i) {
            for (int j = 0; j < inputWidth; ++j) {
                for (int k = 0; k < channels; ++k) {
                    size_t idx = i * inputWidth * channels + j * channels + k;
                    if (idx < data.size()) {
                        image.at<cv::Vec3f>(i, j)[k] = data[idx];
                    }
                }
            }
        }

        // Convert back to 8-bit and BGR for saving
        cv::Mat outputImage;
        image.convertTo(outputImage, CV_8UC3, 255.0);
        cv::cvtColor(outputImage, outputImage, cv::COLOR_RGB2BGR);

        // Save the image
        if (cv::imwrite(outputPath, outputImage)) {
            return ""; // Success
        } else {
            return "Failed to save image to: " + outputPath;
        }
    }
    catch (const std::exception& e) {
        return "Error saving image: " + std::string(e.what());
    }
}

// Video learning interface implementations
bool VisionModule::isVideoFormatSupported(const std::string& videoPath) {
    // Check file extension for basic format support
    std::string extension = videoPath.substr(videoPath.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    std::vector<std::string> supportedFormats = {"mp4", "avi", "mov", "mkv", "wmv", "flv", "webm"};
    return std::find(supportedFormats.begin(), supportedFormats.end(), extension) != supportedFormats.end();
}

VisionModule::VideoAnalysis VisionModule::analyzeVideo(const std::string& videoPath, int frameSamplingRate, bool extractText) {
    VideoAnalysis analysis;
    analysis.videoPath = videoPath;

    if (!isVideoFormatSupported(videoPath)) {
        std::cerr << "Error: Video format not supported: " << videoPath << std::endl;
        return analysis;
    }

    try {
        cv_VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
            return analysis;
        }

        // Get video properties
        analysis.fps = cap.cap.get(cv::CAP_PROP_FPS);
        if (analysis.fps <= 0) analysis.fps = 30; // Default fallback
        analysis.totalFrames = static_cast<int>(cap.cap.get(cv::CAP_PROP_FRAME_COUNT));
        if (analysis.totalFrames <= 0) analysis.totalFrames = 0;

        cv_Mat frame(inputHeight, inputWidth, channels);
        int frameCount = 0;
        double timestamp = 0.0;

        while (cap.read(frame)) {
            if (frameCount % frameSamplingRate == 0) {
                VideoFrame videoFrame;
                videoFrame.frameNumber = frameCount;
                videoFrame.timestamp = timestamp;

                // Extract image data
                videoFrame.imageData = extractNumericalFeatures(&frame);

                // Extract features
                videoFrame.features = processImageFeatures(videoFrame.imageData);

                // Perform OCR if requested
                if (extractText) {
                    videoFrame.ocrText = performOCROnData(videoFrame.imageData);
                }

                // Detect objects
                videoFrame.detections = detectObjectsFromData(videoFrame.imageData);

                analysis.frames.push_back(videoFrame);
            }

            frameCount++;
            timestamp += 1.0 / analysis.fps;
        }

        analysis.duration = timestamp;
        analysis.totalFrames = frameCount;

        // Analyze content - count objects and extract text
        for (const auto& frame : analysis.frames) {
            for (const auto& detection : frame.detections) {
                if (detection.confidence >= confidenceThreshold) {
                    analysis.objectCounts[detection.className]++;
                }
            }
            if (!frame.ocrText.empty()) {
                analysis.extractedText.push_back(frame.ocrText);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error analyzing video: " << e.what() << std::endl;
    }

    return analysis;
}

std::vector<VisionModule::VideoFrame> VisionModule::extractVideoFrames(const std::string& videoPath, int maxFrames, int samplingRate) {
    std::vector<VideoFrame> frames;

    try {
        cv_VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file: " << videoPath << std::endl;
            return frames;
        }

        cv_Mat frame(inputHeight, inputWidth, channels);
        int frameCount = 0;
        double timestamp = 0.0;
        int extractedCount = 0;

        while (cap.read(frame) && (maxFrames == -1 || extractedCount < maxFrames)) {
            if (frameCount % samplingRate == 0) {
                VideoFrame videoFrame;
                videoFrame.frameNumber = frameCount;
                videoFrame.timestamp = timestamp;
                videoFrame.imageData = extractNumericalFeatures(&frame);
                videoFrame.features = processImageFeatures(videoFrame.imageData);
                videoFrame.ocrText = performOCROnData(videoFrame.imageData);
                videoFrame.detections = detectObjectsFromData(videoFrame.imageData);

                frames.push_back(videoFrame);
                extractedCount++;
            }

            frameCount++;
            timestamp += 1.0 / 30.0; // Assume 30 fps - could be improved to use actual fps
        }

    } catch (const std::exception& e) {
        std::cerr << "Error extracting video frames: " << e.what() << std::endl;
    }

    return frames;
}

std::vector<std::string> VisionModule::detectSceneChanges(const std::string& videoPath) {
    std::vector<std::string> sceneChanges;

    try {
        auto frames = extractVideoFrames(videoPath, -1, 30); // Sample every 30 frames

        std::vector<float> prevFeatures;
        for (size_t i = 0; i < frames.size(); ++i) {
            if (!prevFeatures.empty()) {
                // Calculate feature difference (simple Euclidean distance)
                float diff = 0.0f;
                for (size_t j = 0; j < prevFeatures.size() && j < frames[i].features.size(); ++j) {
                    float delta = prevFeatures[j] - frames[i].features[j];
                    diff += delta * delta;
                }
                diff = std::sqrt(diff);

                // Threshold for scene change detection
                if (diff > 1.0f) { // Adjustable threshold
                    sceneChanges.push_back("Scene change at frame " + std::to_string(frames[i].frameNumber) +
                                         " (timestamp: " + std::to_string(frames[i].timestamp) + "s)");
                }
            }
            prevFeatures = frames[i].features;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error detecting scene changes: " << e.what() << std::endl;
    }

    return sceneChanges;
}

std::map<std::string, int> VisionModule::countObjectsInVideo(const std::string& videoPath) {
    std::map<std::string, int> objectCounts;

    try {
        auto frames = extractVideoFrames(videoPath, -1, 60); // Sample every 60 frames for efficiency

        for (const auto& frame : frames) {
            for (const auto& detection : frame.detections) {
                if (detection.confidence >= confidenceThreshold) {
                    objectCounts[detection.className]++;
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error counting objects in video: " << e.what() << std::endl;
    }

    return objectCounts;
}

std::vector<std::string> VisionModule::extractTextFromVideo(const std::string& videoPath, int frameInterval) {
    std::vector<std::string> extractedText;

    try {
        auto frames = extractVideoFrames(videoPath, -1, frameInterval);

        for (const auto& frame : frames) {
            if (!frame.ocrText.empty()) {
                extractedText.push_back(frame.ocrText);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error extracting text from video: " << e.what() << std::endl;
    }

    return extractedText;
}

std::vector<float> VisionModule::analyzeVideoContent(const std::string& videoPath) {
    std::vector<float> contentFeatures;

    try {
        auto analysis = analyzeVideo(videoPath, 30, false); // Sample every 30 frames, no OCR

        if (!analysis.frames.empty()) {
            // Aggregate features across frames
            size_t featureSize = analysis.frames[0].features.size();
            contentFeatures.resize(featureSize, 0.0f);

            for (const auto& frame : analysis.frames) {
                for (size_t i = 0; i < featureSize && i < frame.features.size(); ++i) {
                    contentFeatures[i] += frame.features[i];
                }
            }

            // Average the features
            for (float& feature : contentFeatures) {
                feature /= analysis.frames.size();
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error analyzing video content: " << e.what() << std::endl;
    }

    return contentFeatures;
}

std::vector<std::vector<float>> VisionModule::extractVideoFeatures(const std::string& videoPath, int temporalWindow) {
    std::vector<std::vector<float>> temporalFeatures;

    try {
        auto frames = extractVideoFrames(videoPath, -1, 15); // Sample every 15 frames

        for (size_t i = 0; i < frames.size(); ++i) {
            std::vector<float> windowFeatures;

            // Collect features from temporal window
            for (int j = -temporalWindow/2; j <= temporalWindow/2; ++j) {
                int frameIdx = static_cast<int>(i) + j;
                if (frameIdx >= 0 && frameIdx < static_cast<int>(frames.size())) {
                    windowFeatures.insert(windowFeatures.end(),
                                        frames[frameIdx].features.begin(),
                                        frames[frameIdx].features.end());
                }
            }

            if (!windowFeatures.empty()) {
                temporalFeatures.push_back(windowFeatures);
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error extracting video features: " << e.what() << std::endl;
    }

    return temporalFeatures;
}

// Memory management
void VisionModule::clearBuffers() {
    numericalBuffer.clear();
    featureVector.clear();
    // Clear any other buffers if added in the future
}

size_t VisionModule::getMemoryUsage() const {
    size_t totalSize = 0;

    // Calculate size of numericalBuffer
    for (const auto& row : numericalBuffer) {
        totalSize += row.size() * sizeof(float);
    }

    // Calculate size of featureVector
    for (const auto& row : featureVector) {
        totalSize += row.size() * sizeof(float);
    }

    // Add size of other members (approximate)
    totalSize += sizeof(*this);

    return totalSize;
}

// Training interface implementations
bool VisionModule::trainOnImageDataset(const std::string& datasetPath, int epochs) {
    // Base implementation - override in derived classes for actual training
    std::cout << "VisionModule::trainOnImageDataset - Base implementation called for dataset: "
              << datasetPath << " with " << epochs << " epochs" << std::endl;
    std::cout << "Override this method in derived classes for actual training functionality." << std::endl;
    return false; // Base implementation doesn't perform actual training
}

bool VisionModule::fineTuneModel(const std::vector<std::string>& imagePaths, const std::vector<int>& labels) {
    // Base implementation - override in derived classes for actual fine-tuning
    std::cout << "VisionModule::fineTuneModel - Base implementation called with "
              << imagePaths.size() << " images and " << labels.size() << " labels" << std::endl;
    std::cout << "Override this method in derived classes for actual fine-tuning functionality." << std::endl;
    return false; // Base implementation doesn't perform actual fine-tuning
}

// Real-time processing implementations
std::vector<float> VisionModule::processRealTimeFrame(const std::vector<float>& frameData) {
    // Base implementation - process frame data through vision core
    if (!isInitialized || frameData.empty()) {
        return {};
    }

    try {
        return processImageFeatures(frameData);
    } catch (const std::exception& e) {
        std::cerr << "Error processing real-time frame: " << e.what() << std::endl;
        return {};
    }
}

bool VisionModule::initializeRealTimeProcessing(int cameraId) {
    // Base implementation - override in derived classes for camera initialization
    std::cout << "VisionModule::initializeRealTimeProcessing - Base implementation called for camera ID: "
              << cameraId << std::endl;
    std::cout << "Override this method in derived classes for actual camera initialization." << std::endl;
    return false; // Base implementation doesn't initialize camera
}

void VisionModule::stopRealTimeProcessing() {
    // Base implementation - override in derived classes for camera cleanup
    std::cout << "VisionModule::stopRealTimeProcessing - Base implementation called" << std::endl;
    std::cout << "Override this method in derived classes for actual camera cleanup." << std::endl;
    clearBuffers(); // Clear buffers as part of stopping
}



#ifndef AUDIO_MODULE_HPP
#define AUDIO_MODULE_HPP

#include "CoreAI3DCommon.hpp"
#include "Core.hpp"
#include <vector>
#include <string>
#include <memory>
#include <complex>

// Forward declarations for audio libraries
struct AudioBuffer {
    std::vector<float> data;
    int sampleRate;
    int channels;
    int bitsPerSample;
};

struct FFTResult {
    std::vector<float> magnitudes;
    std::vector<float> phases;
    int fftSize;
};

class AudioModule {
private:
    std::unique_ptr<CoreAI> audioCore;
    std::string moduleName;
    int sampleRate;
    int channels;
    int bitsPerSample;
    int bufferSize;
    bool isInitialized;

    // Audio processing buffers
    std::vector<float> audioBuffer;
    std::vector<std::complex<float>> frequencyBuffer;
    std::vector<float> featureVector;

    // Audio processing parameters
    float gain;
    float noiseGateThreshold;
    bool noiseReductionEnabled;

protected:
    // Core audio processing functions
    virtual std::vector<float> processAudioFeatures(const std::vector<float>& audioData);
    virtual std::vector<float> extractMFCCFeatures(const std::vector<float>& audioData);
    virtual std::vector<float> extractSpectralFeatures(const std::vector<float>& audioData);
    virtual std::vector<float> extractTemporalFeatures(const std::vector<float>& audioData);

    // FFT and frequency domain processing
    virtual FFTResult performFFT(const std::vector<float>& audioData);
    virtual std::vector<float> performIFFT(const FFTResult& fftData);

    // Audio filtering
    virtual std::vector<float> applyLowPassFilter(const std::vector<float>& audioData, float cutoffFreq);
    virtual std::vector<float> applyHighPassFilter(const std::vector<float>& audioData, float cutoffFreq);
    virtual std::vector<float> applyBandPassFilter(const std::vector<float>& audioData, float lowFreq, float highFreq);

    // Noise processing
    virtual std::vector<float> applyNoiseReduction(const std::vector<float>& audioData);
    virtual std::vector<float> applyNoiseGate(const std::vector<float>& audioData, float threshold);

public:
    // Constructor
    AudioModule(const std::string& name, int sr = 44100, int ch = 1, int bufSize = 1024);
    virtual ~AudioModule();

    // Initialization
    bool initialize(const std::string& modelPath = "");
    void setAudioParameters(int sampleRate, int channels, int bufferSize);
    void setGain(float gainValue);
    void setNoiseGateThreshold(float threshold);
    void enableNoiseReduction(bool enable);

    // Core audio processing interface
    virtual std::vector<float> processAudio(const std::string& audioPath);
    virtual std::vector<float> processAudioData(const std::vector<float>& audioData);
    virtual std::vector<float> processRealTimeAudio(const std::vector<float>& audioChunk);

    // Feature extraction
    std::vector<float> extractFeatures(const std::string& audioPath);
    std::vector<float> extractFeaturesFromData(const std::vector<float>& audioData);
    std::vector<std::vector<float>> batchExtractFeatures(const std::vector<std::string>& audioPaths);

    // Speech recognition interface
    struct SpeechResult {
        std::string text;
        float confidence;
        std::vector<float> timestamps;
        std::vector<std::string> alternatives;
    };

    virtual SpeechResult recognizeSpeech(const std::string& audioPath);
    virtual SpeechResult recognizeSpeechFromData(const std::vector<float>& audioData);
    virtual std::vector<SpeechResult> recognizeContinuousSpeech(const std::string& audioPath);

    // Audio analysis
    struct AudioAnalysis {
        float duration;
        float averageAmplitude;
        float peakAmplitude;
        float zeroCrossingRate;
        std::vector<float> spectralCentroid;
        std::vector<float> spectralRolloff;
        std::vector<float> mfccCoefficients;
        std::vector<float> chromaFeatures;
    };

    virtual AudioAnalysis analyzeAudio(const std::string& audioPath);
    virtual AudioAnalysis analyzeAudioData(const std::vector<float>& audioData);

    // Audio manipulation
    virtual std::vector<float> normalizeAudio(const std::vector<float>& audioData);
    virtual std::vector<float> amplifyAudio(const std::vector<float>& audioData, float factor);
    virtual std::vector<float> fadeAudio(const std::vector<float>& audioData, float fadeInTime, float fadeOutTime);
    virtual std::vector<float> reverseAudio(const std::vector<float>& audioData);
    virtual std::vector<float> pitchShift(const std::vector<float>& audioData, float semitones);
    virtual std::vector<float> timeStretch(const std::vector<float>& audioData, float ratio);

    // Audio effects
    virtual std::vector<float> applyReverb(const std::vector<float>& audioData, float roomSize, float damping);
    virtual std::vector<float> applyEcho(const std::vector<float>& audioData, float delay, float decay);
    virtual std::vector<float> applyChorus(const std::vector<float>& audioData, float rate, float depth);
    virtual std::vector<float> applyEqualizer(const std::vector<float>& audioData, const std::vector<float>& bandGains);

    // Audio synthesis
    virtual std::vector<float> generateTone(float frequency, float duration, float amplitude = 1.0f);
    virtual std::vector<float> generateNoise(float duration, const std::string& noiseType = "white");
    virtual std::vector<float> generateSilence(float duration);

    // Text-to-Speech interface
    virtual std::vector<float> textToSpeech(const std::string& text, const std::string& voice = "default");
    virtual bool saveSpeechToFile(const std::string& text, const std::string& outputPath, const std::string& voice = "default");

    // Audio format conversion
    virtual std::vector<float> convertSampleRate(const std::vector<float>& audioData, int targetSampleRate);
    virtual std::vector<float> convertChannels(const std::vector<float>& audioData, int targetChannels);
    virtual std::vector<float> convertBitDepth(const std::vector<float>& audioData, int targetBits);

    // Voice activity detection
    struct VADResult {
        std::vector<float> speechSegments;
        std::vector<float> silenceSegments;
        float speechRatio;
        std::vector<std::pair<float, float>> speechIntervals;
    };

    virtual VADResult detectVoiceActivity(const std::string& audioPath);
    virtual VADResult detectVoiceActivityFromData(const std::vector<float>& audioData);

    // Speaker recognition interface
    struct SpeakerResult {
        std::string speakerId;
        float confidence;
        std::vector<float> speakerEmbedding;
    };

    virtual SpeakerResult identifySpeaker(const std::string& audioPath);
    virtual SpeakerResult identifySpeakerFromData(const std::vector<float>& audioData);
    virtual float compareSpeakers(const std::vector<float>& speaker1, const std::vector<float>& speaker2);

    // Music analysis
    struct MusicAnalysis {
        std::string genre;
        float tempo;
        std::string key;
        std::vector<float> chordProgression;
        std::vector<float> beatPositions;
        std::vector<float> noteOnsets;
    };

    virtual MusicAnalysis analyzeMusic(const std::string& audioPath);
    virtual MusicAnalysis analyzeMusicFromData(const std::vector<float>& audioData);

    // Real-time processing
    virtual bool initializeRealTimeProcessing(int inputDevice = -1, int outputDevice = -1);
    virtual std::vector<float> processRealTimeInput();
    virtual void stopRealTimeProcessing();

    // Utility functions
    std::vector<float> audioPathToNumericalData(const std::string& audioPath);
    std::string numericalDataToAudioPath(const std::vector<float>& data, const std::string& outputPath, int sampleRate = 44100);

    // Audio format loading functions
    std::vector<float> loadWAVFile(const std::string& audioPath);
    std::vector<float> loadMP3File(const std::string& audioPath);
    std::vector<float> loadFLACFile(const std::string& audioPath);
    std::vector<float> loadRawPCMFile(const std::string& audioPath);

    // Audio format saving functions
    std::string saveWAVFile(const std::vector<float>& data, const std::string& outputPath, int sampleRate);
    std::string saveMP3File(const std::vector<float>& data, const std::string& outputPath, int sampleRate);
    std::string saveFLACFile(const std::vector<float>& data, const std::string& outputPath, int sampleRate);

    // Status and information
    bool isReady() const { return isInitialized; }
    std::string getModuleName() const { return moduleName; }
    int getSampleRate() const { return sampleRate; }
    int getChannels() const { return channels; }
    int getBufferSize() const { return bufferSize; }

    // Memory management
    void clearBuffers();
    size_t getMemoryUsage() const;

    // Training interface for audio-specific learning
    virtual bool trainOnAudioDataset(const std::string& datasetPath, int epochs = 10);
    virtual bool fineTuneModel(const std::vector<std::string>& audioPaths, const std::vector<std::string>& labels);

    // Audio streaming
    virtual bool startAudioStream(const std::string& outputPath);
    virtual void stopAudioStream();
    virtual bool isStreaming() const;
};

#endif // AUDIO_MODULE_HPP
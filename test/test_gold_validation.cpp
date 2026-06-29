#include "../CoreAI3D/include/Train.hpp"
#include "../CoreAI3D/include/Core.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <limits>
#include <cstdio>

// ── Gold Price Validation Test ──────────────────────────────────────────────

struct ValidationMetrics {
    float rmse          = 0.0f;
    float mae           = 0.0f;
    float mape          = 0.0f;
    float r2            = 0.0f;
    float dir_accuracy  = 0.0f;
    float naive_rmse    = 0.0f;
    float naive_mae     = 0.0f;
    int   test_samples  = 0;
};

bool loadGoldCSV(const std::string& filepath,
                 std::vector<std::string>& dates,
                 std::vector<float>& prices) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open " << filepath << "\n";
        return false;
    }
    std::string line;
    std::getline(file, line); // skip header
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t comma = line.find(',');
        if (comma == std::string::npos) continue;
        try {
            dates.push_back(line.substr(0, comma));
            prices.push_back(std::stof(line.substr(comma + 1)));
        } catch (...) { continue; }
    }
    return !prices.empty();
}

bool createSlidingWindows(const std::vector<float>& prices, int windowSize,
                          std::vector<std::vector<float>>& inputs,
                          std::vector<std::vector<float>>& targets) {
    if ((int)prices.size() <= windowSize) return false;
    inputs.clear(); targets.clear();
    for (int i = windowSize; i < (int)prices.size(); i++) {
        std::vector<float> w(windowSize);
        for (int j = 0; j < windowSize; j++) w[j] = prices[i - windowSize + j];
        inputs.push_back(w);
        targets.push_back({prices[i]});
    }
    return true;
}

float denorm(float val, float mn, float mx, float lo, float hi) {
    return (val - lo) / (hi - lo) * (mx - mn) + mn;
}

ValidationMetrics evaluate(const std::vector<float>& actual,
                           const std::vector<float>& predicted,
                           const std::vector<float>& naive) {
    int n = (int)actual.size();
    if (n == 0) return {};
    float sum_sq = 0, abs_err = 0, pct_err = 0;
    float mean = std::accumulate(actual.begin(), actual.end(), 0.0f) / n;
    float ss_tot = 0, ss_res = 0;
    int dir_ok = 0, dir_n = 0;
    for (int i = 0; i < n; i++) {
        float d = actual[i] - predicted[i];
        sum_sq += d * d; abs_err += std::abs(d);
        if (std::abs(actual[i]) > 1e-9f) pct_err += std::abs(d / actual[i]) * 100.0f;
        ss_tot += (actual[i] - mean) * (actual[i] - mean);
        ss_res += d * d;
        if (i > 0) { if ((actual[i] > actual[i-1]) == (predicted[i] > predicted[i-1])) dir_ok++; dir_n++; }
    }
    ValidationMetrics m;
    m.rmse = std::sqrt(sum_sq / n);
    m.mae = abs_err / n;
    m.mape = pct_err / n;
    m.r2 = (ss_tot > 1e-9f) ? 1.0f - ss_res / ss_tot : 0.0f;
    m.dir_accuracy = (dir_n > 0) ? (float)dir_ok / dir_n * 100.0f : 0.0f;
    m.test_samples = n;
    float nsq = 0, na = 0;
    for (int i = 0; i < n; i++) { float d = actual[i] - naive[i]; nsq += d*d; na += std::abs(d); }
    m.naive_rmse = std::sqrt(nsq / n);
    m.naive_mae = na / n;
    return m;
}

int main() {
    std::cout << "================================================================\n"
              << "  GOLD PRICE PREDICTION - REAL-WORLD VALIDATION TEST\n"
              << "================================================================\n\n";

    const int   WINDOW_SIZE    = 12;
    const float TRAIN_RATIO    = 0.80f;
    const int   LAYERS         = 3;
    const int   NEURONS        = 24;
    const double LEARNING_RATE = 0.01;
    const int   EPOCHS         = 500;
    const float NLO = 0.0f, NHI = 1.0f;

    std::cout << "  Window size   : " << WINDOW_SIZE << " months\n"
              << "  Train ratio   : " << (int)(TRAIN_RATIO*100) << "%\n"
              << "  Neurons       : " << NEURONS << "\n"
              << "  Learning rate : " << LEARNING_RATE << "\n"
              << "  Epochs        : " << EPOCHS << "\n\n";

    // ── Load ────────────────────────────────────────────────────────────
    std::vector<std::string> allDates;
    std::vector<float> allPrices;
    if (!loadGoldCSV("/CoreAI3D/gold.csv", allDates, allPrices)) { return 1; }
    int totalRows = (int)allPrices.size();
    std::cout << "Loaded " << totalRows << " records\n"
              << "  Range: " << allDates.front() << " to " << allDates.back() << "\n"
              << "  Prices: $" << *std::min_element(allPrices.begin(), allPrices.end())
              << " to $" << *std::max_element(allPrices.begin(), allPrices.end()) << "\n\n";

    // ── Split ───────────────────────────────────────────────────────────
    int trainEnd = (int)(totalRows * TRAIN_RATIO);
    std::vector<float> trainP(allPrices.begin(), allPrices.begin() + trainEnd);
    std::vector<float> testP(allPrices.begin() + trainEnd, allPrices.end());
    int testStartFull = trainEnd - WINDOW_SIZE;
    std::vector<float> fullTestP(allPrices.begin() + testStartFull, allPrices.end());

    std::cout << "  Train: " << trainP.size() << " months\n"
              << "  Test : " << testP.size() << " months\n\n";

    // ── Windows ─────────────────────────────────────────────────────────
    std::vector<std::vector<float>> trIn, trTg, teIn, teTg;
    createSlidingWindows(trainP, WINDOW_SIZE, trIn, trTg);
    createSlidingWindows(fullTestP, WINDOW_SIZE, teIn, teTg);
    std::cout << "  Train windows: " << trIn.size() << "\n"
              << "  Test windows : " << teIn.size() << "\n\n";

    // ── Normalize (training bounds only) ────────────────────────────────
    std::vector<float> inMin(WINDOW_SIZE, 1e30f), inMax(WINDOW_SIZE, -1e30f);
    for (auto& r : trIn) for (int j = 0; j < WINDOW_SIZE; j++) {
        inMin[j] = std::min(inMin[j], r[j]); inMax[j] = std::max(inMax[j], r[j]);
    }
    float tgMin = trTg[0][0], tgMax = trTg[0][0];
    for (auto& r : trTg) { tgMin = std::min(tgMin, r[0]); tgMax = std::max(tgMax, r[0]); }

    auto doNorm = [](std::vector<std::vector<float>>& data,
                     const std::vector<float>& mn, const std::vector<float>& mx, float lo, float hi) {
        for (auto& r : data) for (int j = 0; j < (int)r.size(); j++) {
            float rng = mx[j] - mn[j];
            r[j] = (rng > 1e-9f) ? (r[j] - mn[j]) / rng * (hi - lo) + lo : 0.0f;
        }
    };
    auto doNormScalar = [](std::vector<std::vector<float>>& data, float mn, float mx, float lo, float hi) {
        for (auto& r : data) for (int j = 0; j < (int)r.size(); j++) {
            float rng = mx - mn;
            r[j] = (rng > 1e-9f) ? (r[j] - mn) / rng * (hi - lo) + lo : 0.0f;
        }
    };

    // Write combined CSV: [window_12_cols, target_1_col] — RAW, no normalization
    const std::string combCSV = "/tmp/gold_combined.csv";
    {
        std::ofstream f(combCSV);
        for (size_t i = 0; i < trIn.size(); i++) {
            for (int j = 0; j < WINDOW_SIZE; j++) f << std::setprecision(10) << trIn[i][j] << ",";
            f << std::setprecision(10) << trTg[i][0] << "\n";
        }
        f.close();
    }

    // ── Train via Training class ────────────────────────────────────────
    std::cout << "Training...\n" << std::flush;

    try {
        Training trainer(true, false, nullptr);
        trainer.layers = LAYERS;
        trainer.neurons = NEURONS;
        trainer.min = NLO;
        trainer.max = NHI;
        trainer.inputSize = WINDOW_SIZE;
        trainer.outputSize = 1;

        // Load combined CSV (12 input cols + 1 target col)
        if (!trainer.loadCSV(combCSV, 0, 1, false, false, ',', "gold_val")) {
            std::cerr << "FATAL: loadCSV failed\n"; return 1;
        }
        std::cout << "  Loaded " << trainer.numSamples << " samples\n" << std::flush;

        // Preprocess creates CoreAI with normalized data
        // Since our data is already in [0,1], this re-normalization should be near no-op
        trainer.preprocess(NLO, NHI);

        CoreAI* core = trainer.getCore();
        if (!core) { std::cerr << "FATAL: CoreAI null after preprocess\n"; return 1; }
        std::cout << "  CoreAI created.\n" << std::flush;

        // Extract normalization bounds from trainer (set by preprocess)
        std::vector<float> tInMin = trainer.input_min;
        std::vector<float> tInMax = trainer.input_max;
        float tTgMin = trainer.target_mins.empty() ? trainer.target_min : trainer.target_mins[0];
        float tTgMax = trainer.target_maxs.empty() ? trainer.target_max : trainer.target_maxs[0];
        std::cout << "  Norm bounds - Target: [" << tTgMin << ", " << tTgMax << "]\n" << std::flush;

        // Train with our data
        auto t0 = std::chrono::high_resolution_clock::now();
        trainer.train(LEARNING_RATE, EPOCHS);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "  Done in " << std::setprecision(1)
                  << std::chrono::duration<double>(t1-t0).count() << " s\n\n";

        // ── Evaluate ────────────────────────────────────────────────────
        std::cout << "Evaluating...\n" << std::flush;

        // Normalize test inputs using trainer's bounds
        std::vector<std::vector<float>> nTeIn = teIn;
        for (auto& r : nTeIn) for (int j = 0; j < WINDOW_SIZE; j++) {
            float rng = tInMax[j] - tInMin[j];
            r[j] = (rng > 1e-9f) ? (r[j] - tInMin[j]) / rng * (NHI - NLO) + NLO : 0.0f;
        }

        auto testOut = core->forward(nTeIn, false);
        int nTest = (int)testOut.size();

        std::vector<float> actual(nTest), predicted(nTest), naivePred(nTest);
        for (int i = 0; i < nTest; i++) {
            // Predictions are in normalized [0,1] space — denormalize back to $
            predicted[i] = denorm(testOut[i][0], tTgMin, tTgMax, NLO, NHI);
            // Actual and naive prices are RAW from gold.csv — use directly
            actual[i]    = teTg[i][0];
            naivePred[i] = teIn[i][WINDOW_SIZE - 1]; // yesterday's raw price
        }

        ValidationMetrics m = evaluate(actual, predicted, naivePred);

        // ── Report ──────────────────────────────────────────────────────
        std::cout << "\n  ┌──────────────────────────────────────────────────┐\n";
        std::cout << "  │           COREAI3D MODEL METRICS                 │\n";
        std::cout << "  ├─────────────────────┬────────────────────────────┤\n";
        std::cout << std::fixed << std::setprecision(2);
        auto row = [&](const char* label, const std::string& val) {
            std::cout << "  │ " << std::left << std::setw(21) << label << " │ "
                      << std::right << std::setw(26) << val << " │\n";
        };
        row("RMSE",              "$" + std::to_string(m.rmse));
        row("MAE",               "$" + std::to_string(m.mae));
        row("MAPE",              std::to_string(m.mape).substr(0,8) + " %");
        row("R-squared",         std::to_string(m.r2).substr(0,8));
        row("Directional Acc",   std::to_string(m.dir_accuracy).substr(0,8) + " %");
        row("Test Samples",      std::to_string(m.test_samples));
        std::cout << "  ├─────────────────────┼────────────────────────────┤\n";
        row("Naive RMSE",        "$" + std::to_string(m.naive_rmse));
        row("Naive MAE",         "$" + std::to_string(m.naive_mae));
        std::cout << "  ├─────────────────────┴────────────────────────────┤\n";
        bool beat = m.rmse < m.naive_rmse;
        std::cout << "  │ Beats naive?  " << (beat ? "YES" : "NO")
                  << "                                      │\n";
        std::cout << "  └──────────────────────────────────────────────────┘\n\n";

        // ── Sample predictions ──────────────────────────────────────────
        std::cout << "  Last 15 test predictions:\n";
        std::cout << "  " << std::setw(10) << "Date"
                  << " | " << std::setw(12) << "Actual"
                  << " | " << std::setw(12) << "Predicted"
                  << " | " << std::setw(10) << "Error"
                  << " | " << std::setw(8) << "Err%\n";
        std::cout << "  " << std::string(58, '-') << "\n";

        int dateOff = trainEnd;
        for (int i = std::max(0, nTest - 15); i < nTest; i++) {
            std::string dt = (dateOff + i < (int)allDates.size()) ? allDates[dateOff + i] : "?";
            float err = actual[i] - predicted[i];
            float ep = (std::abs(actual[i]) > 1e-9f) ? err / actual[i] * 100 : 0;
            std::cout << "  " << std::setw(10) << dt
                      << " | $" << std::setw(10) << actual[i]
                      << " | $" << std::setw(10) << predicted[i]
                      << " | $" << std::setw(8) << err
                      << " | " << std::setw(6) << std::setprecision(1) << ep << "%\n";
        }

        // ── Save CSV ────────────────────────────────────────────────────
        std::ofstream csv("/CoreAI3D/gold_validation_results.csv");
        csv << "Date,Actual,Predicted,Error,Error_Pct,Naive\n";
        for (int i = 0; i < nTest; i++) {
            std::string dt = (dateOff + i < (int)allDates.size()) ? allDates[dateOff + i] : "";
            float err = actual[i] - predicted[i];
            float ep = (std::abs(actual[i]) > 1e-9f) ? err / actual[i] * 100 : 0;
            csv << dt << "," << std::fixed << std::setprecision(2)
                << actual[i] << "," << predicted[i] << "," << err << "," << ep << "," << naivePred[i] << "\n";
        }
        csv.close();
        std::cout << "\n  Results saved to /CoreAI3D/gold_validation_results.csv\n";

        // ── Verdict ─────────────────────────────────────────────────────
        std::cout << "\n  VERDICT: ";
        if (m.r2 > 0.95f && beat)
            std::cout << "STRONG - R2>0.95 and beats naive\n";
        else if (m.r2 > 0.80f || beat)
            std::cout << "MODERATE - acceptable but not production-ready\n";
        else
            std::cout << "WEAK - does not beat naive baseline\n";

    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }

    std::remove(combCSV.c_str());
    std::cout << "\n================================================================\n";

    return 0;
}

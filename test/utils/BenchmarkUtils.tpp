#ifndef BENCHMARK_UTILS_TPP
#define BENCHMARK_UTILS_TPP

#include "BenchmarkUtils.hpp"

template<typename Func, typename... Args>
BenchmarkResult BenchmarkUtils::benchmarkFunction(
    const std::string& name,
    Func&& func,
    Args&&... args,
    size_t iterations,
    size_t warmupIterations
) {
    // Warmup
    for (size_t i = 0; i < warmupIterations; ++i) {
        func(std::forward<Args>(args)...);
    }

    // Benchmark
    auto times = runBenchmarkIterations(
        [&]() { func(std::forward<Args>(args)...); },
        iterations
    );

    // Calculate statistics
    double totalTime = 0.0;
    double minTime = *std::min_element(times.begin(), times.end());
    double maxTime = *std::max_element(times.begin(), times.end());

    for (double t : times) {
        totalTime += t;
    }

    double avgTime = totalTime / iterations;
    double variance = 0.0;
    for (double t : times) {
        double diff = t - avgTime;
        variance += diff * diff;
    }
    double stdDev = std::sqrt(variance / iterations);

    return BenchmarkResult{
        name,
        totalTime,
        iterations,
        avgTime,
        minTime,
        maxTime,
        stdDev
    };
}

#endif // BENCHMARK_UTILS_TPP
#pragma once
#include <chrono>
#include <cstdio>

// tiny stopwatch for the per-stage timings. make one at the top of a block and
// it prints how long the block took when it goes out of scope. printing only
// happens when the global flag is on which --timers flips. you can also pass an
// accumulator so a loop can add all its times together
struct ScopedTimer {
    static inline bool enabled = false;

    const char* name;
    double*     accumMs;
    std::chrono::steady_clock::time_point t0;

    explicit ScopedTimer(const char* n, double* accum = nullptr)
        : name(n), accumMs(accum), t0(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (accumMs) *accumMs += ms;
        if (enabled) std::printf("[t] %-20s %7.1f ms\n", name, ms);
    }
};

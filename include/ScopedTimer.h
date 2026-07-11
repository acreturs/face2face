#pragma once
#include <chrono>
#include <cstdio>

// Minimal RAII wall-clock timer for per-stage profiling (realtime plan,
// Phase 0). Printing is gated on the global flag (set by --timers); when off,
// a timer costs two clock reads. Optionally accumulates into *accumMs so a
// loop can report totals (e.g. the live-mode FPS HUD).
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

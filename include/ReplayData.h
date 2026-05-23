#pragma once
#include <vector>

struct ReplayFrame {
    std::vector<float> mathematicalState;
};

struct ReplayHistory {
    std::vector<ReplayFrame> frames;
    int totalIterations = 0;
    double executionTimeMs = 0.0;
};
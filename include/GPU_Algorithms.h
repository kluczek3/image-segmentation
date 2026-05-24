#pragma once
#include "SegmentationAlgorithm.h"

class OtsuGPU : public SegmentationAlgorithm {
public:
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
};

class KMeansGPU : public SegmentationAlgorithm {
public:
    KMeansGPU(int clusters = 3, int maxIter = 100);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    int k;
    int maxIterations;
};

class FCMGPU : public SegmentationAlgorithm {
public:
    FCMGPU(int clusters = 3, int maxIter = 100, float fuzziness = 2.0f, float epsilon = 0.01f);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    int k;
    int maxIterations;
    float m;
    float eps;
};

class MeanShiftGPU : public SegmentationAlgorithm {
public:
    MeanShiftGPU(float spatialBandwidth = 8.0f, float colorBandwidth = 16.0f, int maxIter = 10);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    float hs;
    float hr;
    int maxIterations;
};
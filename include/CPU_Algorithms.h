#pragma once
#include "SegmentationAlgorithm.h"

class OtsuCPU : public SegmentationAlgorithm {
public:
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
};

class KMeansCPU : public SegmentationAlgorithm {
public:
    KMeansCPU(int clusters = 3, int maxIter = 100);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    int k;
    int maxIterations;
};

class FCMCPU : public SegmentationAlgorithm {
public:
    FCMCPU(int clusters = 3, int maxIter = 100, float fuzziness = 2.0f, float epsilon = 0.01f);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    int k;
    int maxIterations;
    float m;
    float eps;
};

class MeanShiftCPU : public SegmentationAlgorithm {
public:
    MeanShiftCPU(float spatialBandwidth = 8.0f, float colorBandwidth = 16.0f, int maxIter = 10);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    float hs;
    float hr;
    int maxIterations;
};
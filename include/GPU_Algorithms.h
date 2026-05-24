#pragma once
#include "SegmentationAlgorithm.h"

class OtsuGPU : public SegmentationAlgorithm {
public:
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
};

class KMeansGPU : public SegmentationAlgorithm {
public:
    KMeansGPU(int clusters, int maxIter);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    int k;
    int maxIterations;
};

class FCMGPU : public SegmentationAlgorithm {
public:
    FCMGPU(int clusters, int maxIter, float fuzziness, float epsilon);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    int k;
    int maxIterations;
    float m;
    float eps;
};

class MeanShiftGPU : public SegmentationAlgorithm {
public:
    MeanShiftGPU(float spatialBandwidth, float colorBandwidth, int maxIter);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    float hs;
    float hr;
    int maxIterations;
};
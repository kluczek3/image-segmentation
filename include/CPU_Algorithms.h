#pragma once
#include "SegmentationAlgorithm.h"

class OtsuCPU : public SegmentationAlgorithm {
public:
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
};

class KMeansCPU : public SegmentationAlgorithm {
public:
    KMeansCPU(int clusters, int maxIter);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    int k;
    int maxIterations;
};

class FCMCPU : public SegmentationAlgorithm {
public:
    FCMCPU(int clusters, int maxIter, float fuzziness, float epsilon);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    int k;
    int maxIterations;
    float m;
    float eps;
};

class MeanShiftCPU : public SegmentationAlgorithm {
public:
    MeanShiftCPU(float spatialBandwidth, float colorBandwidth, int maxIter);
    std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) override;
private:
    float hs;
    float hr;
    int maxIterations;
};
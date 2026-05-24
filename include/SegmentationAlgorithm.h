#pragma once
#include <opencv2/opencv.hpp>
#include <utility>

struct AlgoParameters {
    int clusters = 3;
    int maxIter = 100;
    float fuzziness = 2.0f;
    float epsilon = 0.01f;
    float spatialBandwidth = 8.0f;
    float colorBandwidth = 16.0f;
};

class SegmentationAlgorithm {
public:
    virtual ~SegmentationAlgorithm() = default;
    virtual std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) = 0;
};
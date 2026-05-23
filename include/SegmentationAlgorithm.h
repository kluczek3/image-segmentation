#pragma once
#include <opencv2/opencv.hpp>
#include <utility>

class SegmentationAlgorithm {
public:
    virtual ~SegmentationAlgorithm() = default;
    virtual std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) = 0;

};
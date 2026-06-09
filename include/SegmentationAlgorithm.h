#pragma once
#include <opencv2/opencv.hpp>
#include <utility>

enum AlgorithmIndex {
    AlgorithmKMeans = 0,
    AlgorithmFCM = 1,
    AlgorithmOtsu = 2,
    AlgorithmMeanShift = 3
};

struct AlgoParameters {
    int kmeansClusters = 3;
    int kmeansMaxIter = 100;

    int fcmClusters = 3;
    int fcmMaxIter = 100;
    float fcmFuzziness = 2.0f;
    float fcmEpsilon = 0.01f;

    float meanShiftSpatialBandwidth = 8.0f;
    float meanShiftColorBandwidth = 16.0f;
    int meanShiftMaxIter = 10;
};

class SegmentationAlgorithm {
public:
    virtual ~SegmentationAlgorithm() = default;
    virtual std::pair<cv::Mat, double> execute(const cv::Mat& inputImage) = 0;
};

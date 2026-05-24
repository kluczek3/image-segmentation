#pragma once
#include <QObject>
#include <QImage>
#include <QFutureWatcher>
#include <memory>
#include <opencv2/opencv.hpp>
#include "SegmentationAlgorithm.h"

class CoreManager : public QObject {
    Q_OBJECT
public:
    explicit CoreManager(QObject* parent = nullptr);
    ~CoreManager() override;
    void startProcessing(const cv::Mat& inputImage, int algoIndex, const AlgoParameters& params);
    void cancelProcessing();

signals:
    void processingStarted();
    void cpuFinished(const QImage& cpuImage, double cpuTime);
    void gpuFinished(const QImage& gpuImage, double gpuTime);
    void processingFailed(const QString& errorMessage);

private:
    std::unique_ptr<SegmentationAlgorithm> createAlgorithmCPU(int type, const AlgoParameters& params);
    std::unique_ptr<SegmentationAlgorithm> createAlgorithmGPU(int type, const AlgoParameters& params);

    QFutureWatcher<std::pair<cv::Mat, double>> cpuWatcher;
    QFutureWatcher<std::pair<cv::Mat, double>> gpuWatcher;
};
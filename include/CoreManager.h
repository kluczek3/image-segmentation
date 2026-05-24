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
    void startProcessing(const QImage& inputImage, int algoIndex);
    void cancelProcessing();

signals:
    void cpuFinished(const QImage& cpuImage, double cpuTime);
    void gpuFinished(const QImage& gpuImage, double gpuTime);

private slots:
    void checkCompletion();

private:
    std::unique_ptr<SegmentationAlgorithm> createAlgorithmCPU(int type);
    std::unique_ptr<SegmentationAlgorithm> createAlgorithmGPU(int type);

    QFutureWatcher<std::pair<cv::Mat, double>> cpuWatcher;
    QFutureWatcher<std::pair<cv::Mat, double>> gpuWatcher;
};
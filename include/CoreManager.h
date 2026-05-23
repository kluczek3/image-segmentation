#pragma once
#include <QObject>
#include <QImage>
#include <QFutureWatcher>
#include <memory>
#include "SegmentationAlgorithm.h"

enum class AlgorithmType {
    KMeans_CPU, KMeans_GPU,
    FCM_CPU, FCM_GPU,
    Otsu_CPU, Otsu_GPU,
    MeanShift_CPU, MeanShift_GPU
};

class CoreManager : public QObject {
    Q_OBJECT
public:
    explicit CoreManager(QObject* parent = nullptr);
    ~CoreManager() override;
    void startProcessing(const QImage& inputImage, AlgorithmType type);
    void cancelProcessing();

signals:
    void processingStarted();
    void processingFinished(const QImage& resultImage, double executionTimeMs);
    void processingFailed(const QString& errorMessage);

private:
    std::unique_ptr<SegmentationAlgorithm> createAlgorithm(AlgorithmType type);
    QFutureWatcher<std::pair<cv::Mat, double>> watcher;
};
#include "CoreManager.h"
#include "CPU_Algorithms.h"
#include "ImageConverter.h"
#include <QtConcurrent/QtConcurrent>

CoreManager::CoreManager(QObject* parent) : QObject(parent) {
    connect(&watcher, &QFutureWatcher<std::pair<cv::Mat, double>>::finished, this, [this]() {
        if (watcher.isCanceled()) return;
        try {
            auto result = watcher.result();
            QImage finalImage = ImageConverter::CvMatToQImage(result.first);
            emit processingFinished(finalImage, result.second);
        }
        catch (const std::exception& e) {
            emit processingFailed(QString::fromStdString(e.what()));
        }
        });
}

CoreManager::~CoreManager() {
    cancelProcessing();
}

void CoreManager::cancelProcessing() {
    if (watcher.isRunning()) {
        watcher.cancel();
        watcher.waitForFinished();
    }
}

std::unique_ptr<SegmentationAlgorithm> CoreManager::createAlgorithm(AlgorithmType type) {
    switch (type) {
    case AlgorithmType::Otsu_CPU: return std::make_unique<OtsuCPU>();
    case AlgorithmType::KMeans_CPU: return std::make_unique<KMeansCPU>();
    case AlgorithmType::FCM_CPU: return std::make_unique<FCMCPU>();
    case AlgorithmType::MeanShift_CPU: return std::make_unique<MeanShiftCPU>();
    default: return std::make_unique<KMeansCPU>();
    }
}

void CoreManager::startProcessing(const QImage& inputImage, AlgorithmType type) {
    cancelProcessing();
    emit processingStarted();

    cv::Mat cvInput = ImageConverter::QImageToCvMat(inputImage);
    std::shared_ptr<SegmentationAlgorithm> algo = createAlgorithm(type);

    QFuture<std::pair<cv::Mat, double>> future = QtConcurrent::run(
        [algo = std::move(algo), cvInput]() -> std::pair<cv::Mat, double> {
            return algo->execute(cvInput);
        }
    );

    watcher.setFuture(future);
}
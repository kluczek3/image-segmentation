#include "CoreManager.h"
#include "CPU_Algorithms.h"
#include "GPU_Algorithms.h"
#include "ImageConverter.h"
#include <QtConcurrent/QtConcurrent>

CoreManager::CoreManager(QObject* parent) : QObject(parent) {
    connect(&cpuWatcher, &QFutureWatcher<std::pair<cv::Mat, double>>::finished, this, &CoreManager::checkCompletion);
    connect(&gpuWatcher, &QFutureWatcher<std::pair<cv::Mat, double>>::finished, this, &CoreManager::checkCompletion);
}

CoreManager::~CoreManager() {
    cancelProcessing();
}

void CoreManager::cancelProcessing() {
    if (cpuWatcher.isRunning()) { cpuWatcher.cancel(); cpuWatcher.waitForFinished(); }
    if (gpuWatcher.isRunning()) { gpuWatcher.cancel(); gpuWatcher.waitForFinished(); }
}

std::unique_ptr<SegmentationAlgorithm> CoreManager::createAlgorithmCPU(int type) {
    switch (type) {
    case 0: return std::make_unique<KMeansCPU>();
    case 1: return std::make_unique<FCMCPU>();
    case 2: return std::make_unique<OtsuCPU>();
    case 3: return std::make_unique<MeanShiftCPU>();
    default: return std::make_unique<KMeansCPU>();
    }
}

std::unique_ptr<SegmentationAlgorithm> CoreManager::createAlgorithmGPU(int type) {
    switch (type) {
    case 0: return std::make_unique<KMeansGPU>();
    case 1: return std::make_unique<FCMGPU>();
    case 2: return std::make_unique<OtsuGPU>();
    case 3: return std::make_unique<MeanShiftGPU>();
    default: return std::make_unique<KMeansGPU>();
    }
}

void CoreManager::startProcessing(const QImage& inputImage, int algoIndex) {
    cancelProcessing();
    emit processingStarted();

    QImage copyCPU = inputImage.copy();
    QImage copyGPU = inputImage.copy();
    std::shared_ptr<SegmentationAlgorithm> algoCPU = createAlgorithmCPU(algoIndex);
    std::shared_ptr<SegmentationAlgorithm> algoGPU = createAlgorithmGPU(algoIndex);

    QFuture<std::pair<cv::Mat, double>> futureCPU = QtConcurrent::run(
        [algoCPU = std::move(algoCPU), copyCPU]() -> std::pair<cv::Mat, double> {
            cv::Mat cvInput = ImageConverter::QImageToCvMat(copyCPU);
            return algoCPU->execute(cvInput);
        }
    );

    QFuture<std::pair<cv::Mat, double>> futureGPU = QtConcurrent::run(
        [algoGPU = std::move(algoGPU), copyGPU]() -> std::pair<cv::Mat, double> {
            cv::Mat cvInput = ImageConverter::QImageToCvMat(copyGPU);
            return algoGPU->execute(cvInput);
        }
    );

    cpuWatcher.setFuture(futureCPU);
    gpuWatcher.setFuture(futureGPU);
}

void CoreManager::checkCompletion() {
    if (cpuWatcher.isCanceled() || gpuWatcher.isCanceled()) return;
    if (cpuWatcher.isFinished() && gpuWatcher.isFinished()) {
        try {
            auto resCPU = cpuWatcher.result();
            auto resGPU = gpuWatcher.result();
            QImage imgCPU = ImageConverter::CvMatToQImage(resCPU.first);
            QImage imgGPU = ImageConverter::CvMatToQImage(resGPU.first);
            emit processingFinished(imgCPU, resCPU.second, imgGPU, resGPU.second);
        }
        catch (const std::exception& e) {
            emit processingFailed(QString::fromStdString(e.what()));
        }
    }
}
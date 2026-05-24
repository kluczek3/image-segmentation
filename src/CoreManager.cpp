#include "CoreManager.h"
#include "CPU_Algorithms.h"
#include "GPU_Algorithms.h"
#include "ImageConverter.h"
#include <QtConcurrent/QtConcurrent>

CoreManager::CoreManager(QObject* parent) : QObject(parent) {
    connect(&cpuWatcher, &QFutureWatcher<std::pair<cv::Mat, double>>::finished, this, [this]() {
        if (cpuWatcher.isCanceled()) return;
        try {
            auto res = cpuWatcher.result();
            emit cpuFinished(ImageConverter::CvMatToQImage(res.first), res.second);
        }
        catch (const std::exception& e) { emit processingFailed(QString::fromStdString(e.what())); }
        });

    connect(&gpuWatcher, &QFutureWatcher<std::pair<cv::Mat, double>>::finished, this, [this]() {
        if (gpuWatcher.isCanceled()) return;
        try {
            auto res = gpuWatcher.result();
            emit gpuFinished(ImageConverter::CvMatToQImage(res.first), res.second);
        }
        catch (const std::exception& e) { emit processingFailed(QString::fromStdString(e.what())); }
        });
}

CoreManager::~CoreManager() {
    cancelProcessing();
}

void CoreManager::cancelProcessing() {
    if (cpuWatcher.isRunning()) { cpuWatcher.cancel(); cpuWatcher.waitForFinished(); }
    if (gpuWatcher.isRunning()) { gpuWatcher.cancel(); gpuWatcher.waitForFinished(); }
}

std::unique_ptr<SegmentationAlgorithm> CoreManager::createAlgorithmCPU(int type, const AlgoParameters& params) {
    switch (type) {
    case 0: return std::make_unique<KMeansCPU>(params.clusters, params.maxIter);
    case 1: return std::make_unique<FCMCPU>(params.clusters, params.maxIter, params.fuzziness, params.epsilon);
    case 2: return std::make_unique<OtsuCPU>();
    case 3: return std::make_unique<MeanShiftCPU>(params.spatialBandwidth, params.colorBandwidth, params.maxIter);
    default: return std::make_unique<KMeansCPU>(params.clusters, params.maxIter);
    }
}

std::unique_ptr<SegmentationAlgorithm> CoreManager::createAlgorithmGPU(int type, const AlgoParameters& params) {
    switch (type) {
    case 0: return std::make_unique<KMeansGPU>(params.clusters, params.maxIter);
    case 1: return std::make_unique<FCMGPU>(params.clusters, params.maxIter, params.fuzziness, params.epsilon);
    case 2: return std::make_unique<OtsuGPU>();
    case 3: return std::make_unique<MeanShiftGPU>(params.spatialBandwidth, params.colorBandwidth, params.maxIter);
    default: return std::make_unique<KMeansGPU>(params.clusters, params.maxIter);
    }
}

void CoreManager::startProcessing(const cv::Mat& inputImage, int algoIndex, const AlgoParameters& params) {
    cancelProcessing();
    emit processingStarted();

    cv::Mat cvInputCPU = inputImage;
    cv::Mat cvInputGPU = inputImage;

    std::shared_ptr<SegmentationAlgorithm> algoCPU = createAlgorithmCPU(algoIndex, params);
    std::shared_ptr<SegmentationAlgorithm> algoGPU = createAlgorithmGPU(algoIndex, params);

    QFuture<std::pair<cv::Mat, double>> futureCPU = QtConcurrent::run(
        [algoCPU, cvInputCPU]() -> std::pair<cv::Mat, double> {
            return algoCPU->execute(cvInputCPU);
        }
    );

    QFuture<std::pair<cv::Mat, double>> futureGPU = QtConcurrent::run(
        [algoGPU, cvInputGPU]() -> std::pair<cv::Mat, double> {
            return algoGPU->execute(cvInputGPU);
        }
    );

    cpuWatcher.setFuture(futureCPU);
    gpuWatcher.setFuture(futureGPU);
}
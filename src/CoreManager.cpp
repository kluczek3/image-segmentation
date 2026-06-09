#include "CoreManager.h"
#include "CPU_Algorithms.h"
#include "GPU_Algorithms.h"
#include "ImageConverter.h"
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrent>

CoreManager::CoreManager(QObject* parent)
    : QObject(parent), activeRequestId(0) {
}

CoreManager::~CoreManager() {
    cancelProcessing();
}

void CoreManager::cancelProcessing() {
    ++activeRequestId;
}

std::unique_ptr<SegmentationAlgorithm> CoreManager::createAlgorithmCPU(int type, const AlgoParameters& params) {
    switch (type) {
    case AlgorithmKMeans:
        return std::make_unique<KMeansCPU>(params.kmeansClusters, params.kmeansMaxIter);
    case AlgorithmFCM:
        return std::make_unique<FCMCPU>(params.fcmClusters, params.fcmMaxIter, params.fcmFuzziness, params.fcmEpsilon);
    case AlgorithmOtsu:
        return std::make_unique<OtsuCPU>();
    case AlgorithmMeanShift:
        return std::make_unique<MeanShiftCPU>(params.meanShiftSpatialBandwidth, params.meanShiftColorBandwidth, params.meanShiftMaxIter);
    default:
        return std::make_unique<KMeansCPU>(params.kmeansClusters, params.kmeansMaxIter);
    }
}

std::unique_ptr<SegmentationAlgorithm> CoreManager::createAlgorithmGPU(int type, const AlgoParameters& params) {
    switch (type) {
    case AlgorithmKMeans:
        return std::make_unique<KMeansGPU>(params.kmeansClusters, params.kmeansMaxIter);
    case AlgorithmFCM:
        return std::make_unique<FCMGPU>(params.fcmClusters, params.fcmMaxIter, params.fcmFuzziness, params.fcmEpsilon);
    case AlgorithmOtsu:
        return std::make_unique<OtsuGPU>();
    case AlgorithmMeanShift:
        return std::make_unique<MeanShiftGPU>(params.meanShiftSpatialBandwidth, params.meanShiftColorBandwidth, params.meanShiftMaxIter);
    default:
        return std::make_unique<KMeansGPU>(params.kmeansClusters, params.kmeansMaxIter);
    }
}

void CoreManager::startProcessing(const cv::Mat& inputImage, int algoIndex, const AlgoParameters& params) {
    const quint64 requestId = ++activeRequestId;
    emit processingStarted();

    cv::Mat cvInputCPU = inputImage.clone();
    cv::Mat cvInputGPU = inputImage.clone();

    std::shared_ptr<SegmentationAlgorithm> algoCPU(createAlgorithmCPU(algoIndex, params).release());
    std::shared_ptr<SegmentationAlgorithm> algoGPU(createAlgorithmGPU(algoIndex, params).release());

    auto* cpuWatcher = new QFutureWatcher<std::pair<cv::Mat, double>>(this);
    connect(cpuWatcher, &QFutureWatcher<std::pair<cv::Mat, double>>::finished, this, [this, cpuWatcher, requestId]() {
        const bool isCurrent = (requestId == activeRequestId.load());
        try {
            if (isCurrent) {
                const auto result = cpuWatcher->result();
                emit cpuFinished(ImageConverter::CvMatToQImage(result.first), result.second);
            }
        }
        catch (const std::exception& e) {
            if (isCurrent) {
                emit processingFailed(QString::fromStdString(e.what()));
            }
        }
        cpuWatcher->deleteLater();
    });

    auto* gpuWatcher = new QFutureWatcher<std::pair<cv::Mat, double>>(this);
    connect(gpuWatcher, &QFutureWatcher<std::pair<cv::Mat, double>>::finished, this, [this, gpuWatcher, requestId]() {
        const bool isCurrent = (requestId == activeRequestId.load());
        try {
            if (isCurrent) {
                const auto result = gpuWatcher->result();
                emit gpuFinished(ImageConverter::CvMatToQImage(result.first), result.second);
            }
        }
        catch (const std::exception& e) {
            if (isCurrent) {
                emit processingFailed(QString::fromStdString(e.what()));
            }
        }
        gpuWatcher->deleteLater();
    });

    const QFuture<std::pair<cv::Mat, double>> futureCPU = QtConcurrent::run(
        [algoCPU, cvInputCPU]() -> std::pair<cv::Mat, double> {
            return algoCPU->execute(cvInputCPU);
        }
    );

    const QFuture<std::pair<cv::Mat, double>> futureGPU = QtConcurrent::run(
        [algoGPU, cvInputGPU]() -> std::pair<cv::Mat, double> {
            return algoGPU->execute(cvInputGPU);
        }
    );

    cpuWatcher->setFuture(futureCPU);
    gpuWatcher->setFuture(futureGPU);
}

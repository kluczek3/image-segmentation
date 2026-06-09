#pragma once

#include <QImage>
#include <QWidget>
#include <QStringList>
#include <opencv2/opencv.hpp>
#include <vector>
#include "SegmentationAlgorithm.h"

class CoreManager;
class QButtonGroup;
class QLabel;
class QPushButton;
class QResizeEvent;
class QStackedWidget;
class QSpinBox;
class QDoubleSpinBox;

namespace Ui { class StartScreen; }

class StartScreen : public QWidget {
    Q_OBJECT
public:
    explicit StartScreen(QWidget* parent = nullptr);
    ~StartScreen() override;

protected:
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void onRunClicked();
    void onBundledSampleToggled(bool checked);
    void onChooseImageClicked();
    void onProcessingStarted();
    void onCpuFinished(const QImage& cpuImage, double cpuTimeMs);
    void onGpuFinished(const QImage& gpuImage, double gpuTimeMs);
    void onProcessingFailed(const QString& errorMessage);
    void showOriginalWhilePressed();
    void restoreProcessedView();

private:
    struct SegmentationMetrics {
        int approxSegments = 0;
        double intraVariance = 0.0;
        double separationRatio = 0.0;
        double edgeAgreement = 0.0;
    };

    void setupAlgorithmSelector();
    void setupParameterPages();
    void setupConnections();
    void setCurrentAlgorithm(int index, bool rerunIfPossible);
    void resetPanelsForNextRun(bool keepPreviousImages);
    void updateDisplayedImages();
    void setLabelImage(QLabel* label, const QImage& image);
    AlgoParameters collectParameters() const;
    QString currentAlgorithmName() const;
    QString parameterSummaryForAlgorithm(int algorithmIndex, const AlgoParameters& params) const;
    QString summaryTextForResult(const QString& backend,
                                 int algorithmIndex,
                                 const AlgoParameters& params,
                                 const QImage& resultImage,
                                 double runtimeMs,
                                 bool finished) const;
    QString panelTitleForBackend(const QString& backend) const;
    bool loadSelectedImage(QString* errorMessage = nullptr);
    void updateSelectedImageText();
    bool hasAnyResult() const;

    static cv::Mat ensureBgr(const cv::Mat& image);
    static cv::Mat buildApproximateLabelMap(const cv::Mat& segmentedBgr, int& approxSegments);
    static cv::Mat buildBoundaryMapFromLabels(const cv::Mat& labels);
    static SegmentationMetrics computeMetrics(const cv::Mat& originalImage, const cv::Mat& segmentedImage);

    Ui::StartScreen* ui;
    CoreManager* coreManager;

    QStringList algorithms;
    int currentIndex;
    QString selectedImagePath;

    QButtonGroup* algorithmButtonGroup;
    std::vector<QPushButton*> algorithmButtons;
    QStackedWidget* paramStack;

    QSpinBox* spinKMeansClusters;
    QSpinBox* spinKMeansIter;
    QSpinBox* spinFCMClusters;
    QSpinBox* spinFCMIter;
    QDoubleSpinBox* spinFCMFuzz;
    QDoubleSpinBox* spinFCMEps;
    QDoubleSpinBox* spinMSHS;
    QDoubleSpinBox* spinMSHR;
    QSpinBox* spinMSIter;

    cv::Mat sourceImage;
    QImage originalImage;
    QImage latestCpuImage;
    QImage latestGpuImage;

    AlgoParameters lastRunParameters;
    bool cpuFinishedForCurrentRun;
    bool gpuFinishedForCurrentRun;
    bool showingOriginal;
    double latestCpuTimeMs;
    double latestGpuTimeMs;
};

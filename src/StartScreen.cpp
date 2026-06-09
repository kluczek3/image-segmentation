#include "StartScreen.h"
#include "ui_StartScreen.h"

#include "CoreManager.h"
#include "ImageConverter.h"

#include <QButtonGroup>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QResizeEvent>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStackedWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace {
QString bundledSamplePath() {
    return QString::fromStdString(std::string(PROJECT_SOURCE_DIR) + "/assets/peppers_color.tif");
}

QString formatMetric(double value, int decimals = 3) {
    return QString::number(value, 'f', decimals);
}

QString fileNameOnly(const QString& path) {
    return QFileInfo(path).fileName();
}
constexpr double kIntraVarianceMin = 0.0;
constexpr double kIntraVarianceMax = 0.25;
constexpr double kSeparationRatioMin = 0.0;
constexpr double kSeparationRatioMax = 10000.0;
constexpr double kEdgeAgreementMin = 0.0;
constexpr double kEdgeAgreementMax = 1.0;


}

StartScreen::StartScreen(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::StartScreen)
    , coreManager(new CoreManager(this))
    , currentIndex(AlgorithmKMeans)
    , selectedImagePath(bundledSamplePath())
    , algorithmButtonGroup(nullptr)
    , paramStack(nullptr)
    , spinKMeansClusters(nullptr)
    , spinKMeansIter(nullptr)
    , spinFCMClusters(nullptr)
    , spinFCMIter(nullptr)
    , spinFCMFuzz(nullptr)
    , spinFCMEps(nullptr)
    , spinMSHS(nullptr)
    , spinMSHR(nullptr)
    , spinMSIter(nullptr)
    , cpuFinishedForCurrentRun(false)
    , gpuFinishedForCurrentRun(false)
    , showingOriginal(false)
    , latestCpuTimeMs(0.0)
    , latestGpuTimeMs(0.0) {
    ui->setupUi(this);
    ui->subtitleLabel->setText("Select an algorithm, change parameters, run again in place, and hold the button to inspect the original image.");
    algorithms = { "K-Means", "Fuzzy C-Means", "Otsu Thresholding", "Mean Shift" };

    setupAlgorithmSelector();
    setupParameterPages();
    setupConnections();

    setCurrentAlgorithm(AlgorithmKMeans, false);
    loadSelectedImage();
    updateSelectedImageText();
    updateDisplayedImages();

    ui->statusLabel->setText("Ready. Choose parameters and click Run / Re-run.");
}


StartScreen::~StartScreen() {
    delete ui;
}

void StartScreen::resizeEvent(QResizeEvent* event) {
    QWidget::resizeEvent(event);
    updateDisplayedImages();
}

void StartScreen::setupAlgorithmSelector() {
    auto* layout = new QHBoxLayout(ui->algorithmSelectorContainer);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(10);

    algorithmButtonGroup = new QButtonGroup(this);
    algorithmButtonGroup->setExclusive(true);

    for (int i = 0; i < algorithms.size(); ++i) {
        auto* button = new QPushButton(algorithms[i], ui->algorithmSelectorContainer);
        button->setObjectName("algorithmButton");
        button->setCheckable(true);
        button->setCursor(Qt::PointingHandCursor);
        button->setMinimumHeight(44);
        button->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        algorithmButtonGroup->addButton(button, i);
        algorithmButtons.push_back(button);
        layout->addWidget(button);
    }
}

void StartScreen::setupParameterPages() {
    auto* hostLayout = new QVBoxLayout(ui->paramStackHost);
    hostLayout->setContentsMargins(0, 0, 0, 0);

    paramStack = new QStackedWidget(ui->paramStackHost);
    paramStack->setObjectName("paramStack");
    hostLayout->addWidget(paramStack);

    auto* pageKMeans = new QWidget(paramStack);
    auto* layoutKMeans = new QFormLayout(pageKMeans);
    layoutKMeans->setContentsMargins(0, 0, 0, 0);
    layoutKMeans->setSpacing(10);
    spinKMeansClusters = new QSpinBox(pageKMeans);
    spinKMeansClusters->setRange(2, 20);
    spinKMeansClusters->setValue(3);
    spinKMeansIter = new QSpinBox(pageKMeans);
    spinKMeansIter->setRange(1, 1000);
    spinKMeansIter->setValue(100);
    layoutKMeans->addRow("Clusters", spinKMeansClusters);
    layoutKMeans->addRow("Max iterations", spinKMeansIter);
    paramStack->addWidget(pageKMeans);

    auto* pageFCM = new QWidget(paramStack);
    auto* layoutFCM = new QFormLayout(pageFCM);
    layoutFCM->setContentsMargins(0, 0, 0, 0);
    layoutFCM->setSpacing(10);
    spinFCMClusters = new QSpinBox(pageFCM);
    spinFCMClusters->setRange(2, 20);
    spinFCMClusters->setValue(3);
    spinFCMIter = new QSpinBox(pageFCM);
    spinFCMIter->setRange(1, 1000);
    spinFCMIter->setValue(100);
    spinFCMFuzz = new QDoubleSpinBox(pageFCM);
    spinFCMFuzz->setRange(1.1, 5.0);
    spinFCMFuzz->setValue(2.0);
    spinFCMFuzz->setSingleStep(0.1);
    spinFCMFuzz->setDecimals(2);
    spinFCMEps = new QDoubleSpinBox(pageFCM);
    spinFCMEps->setRange(0.0001, 0.1);
    spinFCMEps->setValue(0.01);
    spinFCMEps->setSingleStep(0.001);
    spinFCMEps->setDecimals(4);
    layoutFCM->addRow("Clusters", spinFCMClusters);
    layoutFCM->addRow("Max iterations", spinFCMIter);
    layoutFCM->addRow("Fuzziness (m)", spinFCMFuzz);
    layoutFCM->addRow("Epsilon", spinFCMEps);
    paramStack->addWidget(pageFCM);

    auto* pageOtsu = new QWidget(paramStack);
    auto* layoutOtsu = new QVBoxLayout(pageOtsu);
    layoutOtsu->setContentsMargins(0, 0, 0, 0);
    layoutOtsu->setSpacing(10);
    layoutOtsu->addStretch();
    paramStack->addWidget(pageOtsu);

    auto* pageMeanShift = new QWidget(paramStack);
    auto* layoutMeanShift = new QFormLayout(pageMeanShift);
    layoutMeanShift->setContentsMargins(0, 0, 0, 0);
    layoutMeanShift->setSpacing(10);
    spinMSHS = new QDoubleSpinBox(pageMeanShift);
    spinMSHS->setRange(1.0, 50.0);
    spinMSHS->setValue(8.0);
    spinMSHS->setSingleStep(0.5);
    spinMSHS->setDecimals(1);
    spinMSHR = new QDoubleSpinBox(pageMeanShift);
    spinMSHR->setRange(1.0, 80.0);
    spinMSHR->setValue(16.0);
    spinMSHR->setSingleStep(0.5);
    spinMSHR->setDecimals(1);
    spinMSIter = new QSpinBox(pageMeanShift);
    spinMSIter->setRange(1, 100);
    spinMSIter->setValue(10);
    layoutMeanShift->addRow("Spatial bandwidth", spinMSHS);
    layoutMeanShift->addRow("Color bandwidth", spinMSHR);
    layoutMeanShift->addRow("Max iterations", spinMSIter);
    paramStack->addWidget(pageMeanShift);
}

void StartScreen::setupConnections() {
    connect(algorithmButtonGroup, &QButtonGroup::idClicked, this, [this](int id) {
        setCurrentAlgorithm(id, true);
    });

    connect(ui->defaultCheckbox, &QCheckBox::toggled, this, &StartScreen::onBundledSampleToggled);
    connect(ui->uploadButton, &QPushButton::clicked, this, &StartScreen::onChooseImageClicked);
    connect(ui->segmentButton, &QPushButton::clicked, this, &StartScreen::onRunClicked);

    connect(ui->showOriginalButton, &QPushButton::pressed, this, &StartScreen::showOriginalWhilePressed);
    connect(ui->showOriginalButton, &QPushButton::released, this, &StartScreen::restoreProcessedView);

    connect(coreManager, &CoreManager::processingStarted, this, &StartScreen::onProcessingStarted);
    connect(coreManager, &CoreManager::cpuFinished, this, &StartScreen::onCpuFinished);
    connect(coreManager, &CoreManager::gpuFinished, this, &StartScreen::onGpuFinished);
    connect(coreManager, &CoreManager::processingFailed, this, &StartScreen::onProcessingFailed);
}

void StartScreen::setCurrentAlgorithm(int index, bool rerunIfPossible) {
    if (index < 0 || index >= algorithms.size()) {
        return;
    }

    currentIndex = index;
    ui->currentAlgorithmLabel->setText(QString("Selected algorithm: %1").arg(currentAlgorithmName()));
    ui->algorithmHintLabel->setText("Click another algorithm above to switch in place. Change parameters on the left, then click Run / Re-run.");

    if (paramStack) {
        paramStack->setCurrentIndex(index);
    }

    if (algorithmButtonGroup && algorithmButtonGroup->button(index) && !algorithmButtonGroup->button(index)->isChecked()) {
        const QSignalBlocker blocker(algorithmButtonGroup);
        algorithmButtonGroup->button(index)->setChecked(true);
    }

    if (rerunIfPossible && !sourceImage.empty()) {
        onRunClicked();
    } else {
        const AlgoParameters params = collectParameters();
        ui->cpuSummaryLabel->setText(summaryTextForResult("CPU", currentIndex, params, latestCpuImage, latestCpuTimeMs, cpuFinishedForCurrentRun));
        ui->gpuSummaryLabel->setText(summaryTextForResult("GPU", currentIndex, params, latestGpuImage, latestGpuTimeMs, gpuFinishedForCurrentRun));
    }
}

void StartScreen::onBundledSampleToggled(bool checked) {
    ui->uploadButton->setEnabled(!checked);
    if (checked) {
        selectedImagePath = bundledSamplePath();
        loadSelectedImage();
        latestCpuImage = originalImage;
        latestGpuImage = originalImage;
        cpuFinishedForCurrentRun = false;
        gpuFinishedForCurrentRun = false;
        ui->statusLabel->setText("Using bundled sample image.");
    } else {
        sourceImage.release();
        originalImage = QImage();
        latestCpuImage = QImage();
        latestGpuImage = QImage();
        if (selectedImagePath == bundledSamplePath()) {
            selectedImagePath.clear();
        }
        ui->statusLabel->setText("Choose an image and click Run / Re-run.");
    }

    const AlgoParameters params = collectParameters();
    ui->cpuSummaryLabel->setText(summaryTextForResult("CPU", currentIndex, params, latestCpuImage, latestCpuTimeMs, cpuFinishedForCurrentRun));
    ui->gpuSummaryLabel->setText(summaryTextForResult("GPU", currentIndex, params, latestGpuImage, latestGpuTimeMs, gpuFinishedForCurrentRun));
    updateSelectedImageText();
    updateDisplayedImages();
}

void StartScreen::onChooseImageClicked() {
    const QString fileName = QFileDialog::getOpenFileName(
        this,
        "Choose image",
        QString(),
        "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
    );

    if (fileName.isEmpty()) {
        return;
    }

    if (ui->defaultCheckbox->isChecked()) {
        ui->defaultCheckbox->setChecked(false);
    }

    selectedImagePath = fileName;
    QString errorMessage;
    if (!loadSelectedImage(&errorMessage)) {
        QMessageBox::critical(this, "Unable to open image", errorMessage);
        return;
    }

    latestCpuImage = originalImage;
    latestGpuImage = originalImage;
    cpuFinishedForCurrentRun = false;
    gpuFinishedForCurrentRun = false;
    ui->statusLabel->setText(QString("Loaded %1. Click Run / Re-run to segment it.").arg(fileNameOnly(selectedImagePath)));
    const AlgoParameters params = collectParameters();
    ui->cpuSummaryLabel->setText(summaryTextForResult("CPU", currentIndex, params, latestCpuImage, latestCpuTimeMs, cpuFinishedForCurrentRun));
    ui->gpuSummaryLabel->setText(summaryTextForResult("GPU", currentIndex, params, latestGpuImage, latestGpuTimeMs, gpuFinishedForCurrentRun));
    updateSelectedImageText();
    updateDisplayedImages();
}

bool StartScreen::loadSelectedImage(QString* errorMessage) {
    QString pathToLoad = selectedImagePath;
    if (ui->defaultCheckbox->isChecked()) {
        pathToLoad = bundledSamplePath();
        selectedImagePath = pathToLoad;
    }

    if (pathToLoad.isEmpty()) {
        if (errorMessage) {
            *errorMessage = "No image selected.";
        }
        return false;
    }

    cv::Mat loaded = cv::imread(pathToLoad.toStdString(), cv::IMREAD_COLOR);
    if (loaded.empty()) {
        if (errorMessage) {
            *errorMessage = QString("OpenCV could not load '%1'.").arg(pathToLoad);
        }
        return false;
    }

    sourceImage = loaded;
    originalImage = ImageConverter::CvMatToQImage(sourceImage);
    updateSelectedImageText();

    if (latestCpuImage.isNull()) {
        latestCpuImage = originalImage;
    }
    if (latestGpuImage.isNull()) {
        latestGpuImage = originalImage;
    }
    return true;
}

void StartScreen::updateSelectedImageText() {
    if (selectedImagePath.isEmpty()) {
        ui->selectedImageLabel->setText("Image: none selected");
        return;
    }

    QString details = QString("Image: %1").arg(fileNameOnly(selectedImagePath));
    if (!sourceImage.empty()) {
        details += QString("  |  %1 x %2").arg(sourceImage.cols).arg(sourceImage.rows);
    }
    ui->selectedImageLabel->setText(details);
}

AlgoParameters StartScreen::collectParameters() const {
    AlgoParameters params;
    params.kmeansClusters = spinKMeansClusters->value();
    params.kmeansMaxIter = spinKMeansIter->value();

    params.fcmClusters = spinFCMClusters->value();
    params.fcmMaxIter = spinFCMIter->value();
    params.fcmFuzziness = static_cast<float>(spinFCMFuzz->value());
    params.fcmEpsilon = static_cast<float>(spinFCMEps->value());

    params.meanShiftSpatialBandwidth = static_cast<float>(spinMSHS->value());
    params.meanShiftColorBandwidth = static_cast<float>(spinMSHR->value());
    params.meanShiftMaxIter = spinMSIter->value();
    return params;
}

QString StartScreen::currentAlgorithmName() const {
    return algorithms.value(currentIndex, "K-Means");
}

void StartScreen::onRunClicked() {
    QString errorMessage;
    if (!loadSelectedImage(&errorMessage)) {
        QMessageBox::critical(this, "Unable to open image", errorMessage);
        return;
    }

    lastRunParameters = collectParameters();
    resetPanelsForNextRun(hasAnyResult());
    coreManager->startProcessing(sourceImage, currentIndex, lastRunParameters);
}

void StartScreen::resetPanelsForNextRun(bool keepPreviousImages) {
    cpuFinishedForCurrentRun = false;
    gpuFinishedForCurrentRun = false;
    latestCpuTimeMs = 0.0;
    latestGpuTimeMs = 0.0;

    if (!keepPreviousImages) {
        latestCpuImage = originalImage;
        latestGpuImage = originalImage;
    }

    ui->cpuTimeLabel->setText("CPU: running...");
    ui->gpuTimeLabel->setText("GPU: running...");
    ui->statusLabel->setText(QString("Running %1 on CPU and GPU in parallel...").arg(currentAlgorithmName()));

    const AlgoParameters params = collectParameters();
    ui->cpuSummaryLabel->setText(summaryTextForResult("CPU", currentIndex, params, latestCpuImage, latestCpuTimeMs, false));
    ui->gpuSummaryLabel->setText(summaryTextForResult("GPU", currentIndex, params, latestGpuImage, latestGpuTimeMs, false));
    updateDisplayedImages();
}

void StartScreen::onProcessingStarted() {
    ui->statusLabel->setText(QString("Processing %1...").arg(currentAlgorithmName()));
}

void StartScreen::onCpuFinished(const QImage& cpuImage, double cpuTimeMs) {
    latestCpuImage = cpuImage;
    latestCpuTimeMs = cpuTimeMs;
    cpuFinishedForCurrentRun = true;
    ui->cpuTimeLabel->setText(QString("CPU: %1 ms").arg(QString::number(cpuTimeMs, 'f', 2)));
    ui->cpuSummaryLabel->setText(summaryTextForResult("CPU", currentIndex, lastRunParameters, latestCpuImage, latestCpuTimeMs, true));
    updateDisplayedImages();

    if (gpuFinishedForCurrentRun) {
        ui->statusLabel->setText(QString("Done. %1 finished on CPU and GPU.").arg(currentAlgorithmName()));
    }
}

void StartScreen::onGpuFinished(const QImage& gpuImage, double gpuTimeMs) {
    latestGpuImage = gpuImage;
    latestGpuTimeMs = gpuTimeMs;
    gpuFinishedForCurrentRun = true;
    ui->gpuTimeLabel->setText(QString("GPU: %1 ms").arg(QString::number(gpuTimeMs, 'f', 2)));
    ui->gpuSummaryLabel->setText(summaryTextForResult("GPU", currentIndex, lastRunParameters, latestGpuImage, latestGpuTimeMs, true));
    updateDisplayedImages();

    if (cpuFinishedForCurrentRun) {
        ui->statusLabel->setText(QString("Done. %1 finished on CPU and GPU.").arg(currentAlgorithmName()));
    }
}

void StartScreen::onProcessingFailed(const QString& errorMessage) {
    ui->statusLabel->setText("Failed.");
    QMessageBox::critical(this, "Segmentation failed", errorMessage);
}

void StartScreen::showOriginalWhilePressed() {
    showingOriginal = true;
    updateDisplayedImages();
}

void StartScreen::restoreProcessedView() {
    showingOriginal = false;
    updateDisplayedImages();
}

void StartScreen::updateDisplayedImages() {
    const QImage cpuToShow = showingOriginal ? originalImage : (!latestCpuImage.isNull() ? latestCpuImage : originalImage);
    const QImage gpuToShow = showingOriginal ? originalImage : (!latestGpuImage.isNull() ? latestGpuImage : originalImage);

    setLabelImage(ui->cpuImageLabel, cpuToShow);
    setLabelImage(ui->gpuImageLabel, gpuToShow);
}

void StartScreen::setLabelImage(QLabel* label, const QImage& image) {
    if (!label) {
        return;
    }

    if (image.isNull()) {
        label->clear();
        label->setText("No image");
        return;
    }

    const QSize targetSize = label->size() - QSize(12, 12);
    QPixmap pixmap = QPixmap::fromImage(image).scaled(
        targetSize.width(),
        targetSize.height(),
        Qt::KeepAspectRatio,
        Qt::SmoothTransformation
    );
    label->setPixmap(pixmap);
}

QString StartScreen::parameterSummaryForAlgorithm(int algorithmIndex, const AlgoParameters& params) const {
    switch (algorithmIndex) {
    case AlgorithmKMeans:
        return QString("k=%1, maxIter=%2")
            .arg(params.kmeansClusters)
            .arg(params.kmeansMaxIter);
    case AlgorithmFCM:
        return QString("k=%1, maxIter=%2, m=%3, eps=%4")
            .arg(params.fcmClusters)
            .arg(params.fcmMaxIter)
            .arg(params.fcmFuzziness, 0, 'f', 2)
            .arg(params.fcmEpsilon, 0, 'f', 4);
    case AlgorithmOtsu:
        return QString("global grayscale threshold, no user parameters");
    case AlgorithmMeanShift:
        return QString("hs=%1, hr=%2, maxIter=%3")
            .arg(params.meanShiftSpatialBandwidth, 0, 'f', 1)
            .arg(params.meanShiftColorBandwidth, 0, 'f', 1)
            .arg(params.meanShiftMaxIter);
    default:
        return QString("parameters unavailable");
    }
}

QString StartScreen::panelTitleForBackend(const QString& backend) const {
    if (backend == "CPU") {
        return "CPU result";
    }
    return "GPU result";
}

QString StartScreen::summaryTextForResult(const QString& backend,
                                          int algorithmIndex,
                                          const AlgoParameters& params,
                                          const QImage& resultImage,
                                          double runtimeMs,
                                          bool finished) const {
    QString text;
    text += QString("%1\n").arg(panelTitleForBackend(backend));
    text += QString("Algorithm: %1\n").arg(algorithms.value(algorithmIndex));
    text += QString("Parameters: %1\n").arg(parameterSummaryForAlgorithm(algorithmIndex, params));

    if (!finished || resultImage.isNull() || originalImage.isNull()) {
        text += "Status: waiting for current result...\n";
        text += "Metrics appear after the current run finishes.";
        return text;
    }

    const cv::Mat originalCv = ImageConverter::QImageToCvMat(originalImage);
    const cv::Mat resultCv = ImageConverter::QImageToCvMat(resultImage);
    const SegmentationMetrics metrics = computeMetrics(originalCv, resultCv);

    text += QString("Runtime: %1 ms\n").arg(QString::number(runtimeMs, 'f', 2));
    text += QString("Approx. segments: %1\n").arg(metrics.approxSegments);
    text += QString("Intra-segment variance: %1 (range %2-%3; lower = more homogeneous regions)\n")
        .arg(formatMetric(metrics.intraVariance, 4))
        .arg(formatMetric(kIntraVarianceMin, 4))
        .arg(formatMetric(kIntraVarianceMax, 4));
    text += QString("Separation ratio: %1 (range %2-%3 in this implementation; higher = more distinct segment colors)\n")
        .arg(formatMetric(metrics.separationRatio, 3))
        .arg(formatMetric(kSeparationRatioMin, 3))
        .arg(formatMetric(kSeparationRatioMax, 3));
    text += QString("Edge agreement: %1 (range %2-%3; higher = boundaries follow image edges better)")
        .arg(formatMetric(metrics.edgeAgreement, 3))
        .arg(formatMetric(kEdgeAgreementMin, 3))
        .arg(formatMetric(kEdgeAgreementMax, 3));


    return text;
}

bool StartScreen::hasAnyResult() const {
    return cpuFinishedForCurrentRun || gpuFinishedForCurrentRun;
}

cv::Mat StartScreen::ensureBgr(const cv::Mat& image) {
    if (image.empty()) {
        return cv::Mat();
    }

    if (image.channels() == 3) {
        if (image.type() == CV_8UC3) {
            return image.clone();
        }
        cv::Mat converted;
        image.convertTo(converted, CV_8UC3);
        return converted;
    }

    if (image.channels() == 1) {
        cv::Mat gray;
        if (image.type() == CV_8UC1) {
            gray = image;
        } else {
            image.convertTo(gray, CV_8UC1);
        }
        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }

    cv::Mat converted;
    image.convertTo(converted, CV_8UC3);
    return converted;
}

cv::Mat StartScreen::buildApproximateLabelMap(const cv::Mat& segmentedBgr, int& approxSegments) {
    constexpr int quantStep = 16;

    cv::Mat bgr = ensureBgr(segmentedBgr);
    cv::Mat labels(bgr.rows, bgr.cols, CV_32S);

    std::unordered_map<int, int> codeToLabel;
    codeToLabel.reserve(static_cast<size_t>(bgr.rows * bgr.cols / 8 + 16));

    int nextLabel = 0;
    for (int y = 0; y < bgr.rows; ++y) {
        const auto* row = bgr.ptr<cv::Vec3b>(y);
        auto* labelRow = labels.ptr<int>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b pixel = row[x];
            const int code = (pixel[0] / quantStep)
                + ((pixel[1] / quantStep) << 4)
                + ((pixel[2] / quantStep) << 8);

            auto it = codeToLabel.find(code);
            if (it == codeToLabel.end()) {
                it = codeToLabel.emplace(code, nextLabel++).first;
            }
            labelRow[x] = it->second;
        }
    }

    approxSegments = nextLabel;
    return labels;
}

cv::Mat StartScreen::buildBoundaryMapFromLabels(const cv::Mat& labels) {
    cv::Mat boundary = cv::Mat::zeros(labels.size(), CV_8UC1);

    for (int y = 0; y < labels.rows; ++y) {
        const auto* row = labels.ptr<int>(y);
        auto* out = boundary.ptr<uchar>(y);
        for (int x = 0; x < labels.cols; ++x) {
            const int label = row[x];
            bool isBoundary = false;
            if (x + 1 < labels.cols && label != row[x + 1]) {
                isBoundary = true;
            }
            if (!isBoundary && y + 1 < labels.rows && label != labels.at<int>(y + 1, x)) {
                isBoundary = true;
            }
            if (isBoundary) {
                out[x] = 255;
            }
        }
    }

    return boundary;
}

StartScreen::SegmentationMetrics StartScreen::computeMetrics(const cv::Mat& originalImage, const cv::Mat& segmentedImage) {
    SegmentationMetrics metrics;

    if (originalImage.empty() || segmentedImage.empty()) {
        return metrics;
    }

    const cv::Mat originalBgr = ensureBgr(originalImage);
    const cv::Mat segmentedBgr = ensureBgr(segmentedImage);

    int approxSegments = 0;
    const cv::Mat labels = buildApproximateLabelMap(segmentedBgr, approxSegments);
    metrics.approxSegments = std::max(1, approxSegments);

    std::vector<cv::Vec3d> sums(metrics.approxSegments, cv::Vec3d(0.0, 0.0, 0.0));
    std::vector<int> counts(metrics.approxSegments, 0);

    for (int y = 0; y < segmentedBgr.rows; ++y) {
        const auto* segRow = segmentedBgr.ptr<cv::Vec3b>(y);
        const auto* labelRow = labels.ptr<int>(y);
        for (int x = 0; x < segmentedBgr.cols; ++x) {
            const int label = labelRow[x];
            const cv::Vec3b pixel = segRow[x];
            sums[label][0] += pixel[0];
            sums[label][1] += pixel[1];
            sums[label][2] += pixel[2];
            counts[label] += 1;
        }
    }

    std::vector<cv::Vec3d> means(metrics.approxSegments, cv::Vec3d(0.0, 0.0, 0.0));
    for (int label = 0; label < metrics.approxSegments; ++label) {
        if (counts[label] > 0) {
            means[label] = sums[label] / static_cast<double>(counts[label]);
        }
    }

    double totalSquaredError = 0.0;
    for (int y = 0; y < segmentedBgr.rows; ++y) {
        const auto* segRow = segmentedBgr.ptr<cv::Vec3b>(y);
        const auto* labelRow = labels.ptr<int>(y);
        for (int x = 0; x < segmentedBgr.cols; ++x) {
            const int label = labelRow[x];
            const cv::Vec3d pixel(segRow[x][0], segRow[x][1], segRow[x][2]);
            const cv::Vec3d diff = pixel - means[label];
            totalSquaredError += diff.dot(diff);
        }
    }

    const double normDenominator = static_cast<double>(segmentedBgr.total()) * 3.0 * 255.0 * 255.0;
    metrics.intraVariance = normDenominator > 0.0 ? totalSquaredError / normDenominator : 0.0;

    double pairwiseDistance = 0.0;
    int pairCount = 0;
    for (int i = 0; i < metrics.approxSegments; ++i) {
        if (counts[i] == 0) {
            continue;
        }
        for (int j = i + 1; j < metrics.approxSegments; ++j) {
            if (counts[j] == 0) {
                continue;
            }
            pairwiseDistance += cv::norm(means[i] - means[j]) / (std::sqrt(3.0) * 255.0);
            pairCount += 1;
        }
    }

    const double avgSeparation = pairCount > 0 ? pairwiseDistance / pairCount : 0.0;
    metrics.separationRatio = avgSeparation / std::sqrt(std::max(metrics.intraVariance, 1e-8));

    cv::Mat originalGray;
    cv::cvtColor(originalBgr, originalGray, cv::COLOR_BGR2GRAY);
    cv::Mat originalEdges;
    cv::Canny(originalGray, originalEdges, 60.0, 140.0);
    cv::Mat dilatedOriginalEdges;
    cv::dilate(originalEdges, dilatedOriginalEdges, cv::Mat::ones(3, 3, CV_8UC1));

    const cv::Mat boundary = buildBoundaryMapFromLabels(labels);
    const int boundaryPixels = cv::countNonZero(boundary);
    if (boundaryPixels > 0) {
        cv::Mat overlap;
        cv::bitwise_and(boundary, dilatedOriginalEdges, overlap);
        metrics.edgeAgreement = static_cast<double>(cv::countNonZero(overlap)) / static_cast<double>(boundaryPixels);
    } else {
        metrics.edgeAgreement = 0.0;
    }

    return metrics;
}

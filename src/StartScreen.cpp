#include "StartScreen.h"
#include "ui_StartScreen.h"
#include <QPropertyAnimation>
#include <QParallelAnimationGroup>
#include <QGraphicsOpacityEffect>
#include <QLabel>
#include <QFileDialog>
#include <QFileInfo>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFormLayout>
#include <QCheckBox>
#include <QPushButton>
#include <QStackedWidget>
#include <QSpinBox>

StartScreen::StartScreen(QWidget* parent) :
    QWidget(parent), ui(new Ui::StartScreen), currentIndex(0), isAnimating(false) {
    ui->setupUi(this);
    algorithms = { "K-Means", "Fuzzy C-Means", "Otsu Thresholding", "Mean Shift" };
    ui->currentAlgoLabel->setText(algorithms[currentIndex]);

    selectedImagePath = QString::fromStdString(std::string(PROJECT_SOURCE_DIR) + "/assets/peppers_color.tif");

    setupDynamicUI();

    connect(ui->segmentButton, &QPushButton::clicked, this, &StartScreen::on_segmentButton_clicked);
    connect(ui->prevAlgoButton, &QPushButton::clicked, this, &StartScreen::on_prevAlgoButton_clicked);
    connect(ui->nextAlgoButton, &QPushButton::clicked, this, &StartScreen::on_nextAlgoButton_clicked);
}

StartScreen::~StartScreen() { delete ui; }

void StartScreen::setupDynamicUI() {
    QWidget* container = findChild<QWidget*>("dynamicUiContainer");
    if (!container) return;

    QVBoxLayout* layout = new QVBoxLayout(container);

    defaultCheckbox = new QCheckBox("use default file?", this);
    defaultCheckbox->setObjectName("defaultCheckbox");
    defaultCheckbox->setChecked(true);
    layout->addWidget(defaultCheckbox);

    uploadButton = new QPushButton("peppers_color.tiff", this);
    uploadButton->setObjectName("uploadButton");
    uploadButton->setEnabled(false);
    layout->addWidget(uploadButton);

    paramStack = new QStackedWidget(this);
    layout->addWidget(paramStack);

    QWidget* pageKMeans = new QWidget();
    QFormLayout* layoutKMeans = new QFormLayout(pageKMeans);
    spinKMeansClusters = new QSpinBox(); spinKMeansClusters->setRange(2, 20); spinKMeansClusters->setValue(3);
    spinKMeansIter = new QSpinBox(); spinKMeansIter->setRange(1, 1000); spinKMeansIter->setValue(100);
    layoutKMeans->addRow("clusters:", spinKMeansClusters);
    layoutKMeans->addRow("max_iter:", spinKMeansIter);
    paramStack->addWidget(pageKMeans);

    QWidget* pageFCM = new QWidget();
    QFormLayout* layoutFCM = new QFormLayout(pageFCM);
    spinFCMClusters = new QSpinBox(); spinFCMClusters->setRange(2, 20); spinFCMClusters->setValue(3);
    spinFCMIter = new QSpinBox(); spinFCMIter->setRange(1, 1000); spinFCMIter->setValue(100);
    spinFCMFuzz = new QDoubleSpinBox(); spinFCMFuzz->setRange(1.1, 5.0); spinFCMFuzz->setValue(2.0); spinFCMFuzz->setSingleStep(0.1);
    spinFCMEps = new QDoubleSpinBox(); spinFCMEps->setRange(0.0001, 0.1); spinFCMEps->setValue(0.01); spinFCMEps->setDecimals(4);
    layoutFCM->addRow("clusters:", spinFCMClusters);
    layoutFCM->addRow("max_iter:", spinFCMIter);
    layoutFCM->addRow("fuzziness:", spinFCMFuzz);
    layoutFCM->addRow("epsilon:", spinFCMEps);
    paramStack->addWidget(pageFCM);

    QWidget* pageOtsu = new QWidget();
    QVBoxLayout* layoutOtsu = new QVBoxLayout(pageOtsu);
    paramStack->addWidget(pageOtsu);

    QWidget* pageMS = new QWidget();
    QFormLayout* layoutMS = new QFormLayout(pageMS);
    spinMSHS = new QDoubleSpinBox(); spinMSHS->setRange(1.0, 50.0); spinMSHS->setValue(8.0);
    spinMSHR = new QDoubleSpinBox(); spinMSHR->setRange(1.0, 50.0); spinMSHR->setValue(16.0);
    spinMSIter = new QSpinBox(); spinMSIter->setRange(1, 100); spinMSIter->setValue(10);
    layoutMS->addRow("spatial bandwidth:", spinMSHS);
    layoutMS->addRow("color bandwidth:", spinMSHR);
    layoutMS->addRow("max_iter:", spinMSIter);
    paramStack->addWidget(pageMS);

    paramStack->setCurrentIndex(0);

    connect(defaultCheckbox, &QCheckBox::toggled, [this](bool checked) {
        uploadButton->setEnabled(!checked);
        if (checked) {
            selectedImagePath = QString::fromStdString(std::string(PROJECT_SOURCE_DIR) + "/assets/peppers_color.tif");
            uploadButton->setText("peppers_color.tif");
        }
        else {
            uploadButton->setText("choose file..");
        }
        });

    connect(uploadButton, &QPushButton::clicked, [this]() {
        QString fileName = QFileDialog::getOpenFileName(this, "Choose file", "", "files (*.png *.jpg *.jpeg *.tif *.bmp)");
        if (!fileName.isEmpty()) {
            selectedImagePath = fileName;
            uploadButton->setText( QFileInfo(fileName).fileName());
        }
        });
}

AlgoParameters StartScreen::collectParameters() {
    AlgoParameters p;
    p.clusters = (currentIndex == 0) ? spinKMeansClusters->value() : spinFCMClusters->value();
    p.maxIter = (currentIndex == 0) ? spinKMeansIter->value() : (currentIndex == 1) ? spinFCMIter->value() : spinMSIter->value();
    p.fuzziness = spinFCMFuzz->value();
    p.epsilon = spinFCMEps->value();
    p.spatialBandwidth = spinMSHS->value();
    p.colorBandwidth = spinMSHR->value();
    return p;
}

void StartScreen::on_segmentButton_clicked() {
    emit segmentRequested(currentIndex, algorithms[currentIndex], selectedImagePath, collectParameters());
}

void StartScreen::on_prevAlgoButton_clicked() { switchAlgorithm(-1); }
void StartScreen::on_nextAlgoButton_clicked() { switchAlgorithm(1); }

void StartScreen::switchAlgorithm(int direction) {
    if (isAnimating) return;
    isAnimating = true;

    currentIndex = (currentIndex + direction + (int)algorithms.size()) % algorithms.size();
    QString newText = algorithms[currentIndex];
    if (paramStack) paramStack->setCurrentIndex(currentIndex);

    QLabel* tempLabel = new QLabel(ui->currentAlgoLabel->text(), this);
    tempLabel->setFont(ui->currentAlgoLabel->font());
    tempLabel->setAlignment(ui->currentAlgoLabel->alignment());
    tempLabel->setStyleSheet(ui->currentAlgoLabel->styleSheet());
    tempLabel->setGeometry(ui->currentAlgoLabel->geometry());
    tempLabel->show();

    ui->currentAlgoLabel->setText(newText);
    QRect finalGeo = ui->currentAlgoLabel->geometry();
    QRect startGeo = finalGeo;
    int offset = 60 * direction;
    startGeo.translate(offset, 0);
    ui->currentAlgoLabel->setGeometry(startGeo);

    QGraphicsOpacityEffect* effNew = new QGraphicsOpacityEffect(this);
    ui->currentAlgoLabel->setGraphicsEffect(effNew);
    QGraphicsOpacityEffect* effOld = new QGraphicsOpacityEffect(this);
    tempLabel->setGraphicsEffect(effOld);

    QParallelAnimationGroup* group = new QParallelAnimationGroup(this);

    QPropertyAnimation* animOldPos = new QPropertyAnimation(tempLabel, "geometry");
    animOldPos->setDuration(350); animOldPos->setEndValue(finalGeo.translated(-offset, 0)); animOldPos->setEasingCurve(QEasingCurve::OutCubic);
    QPropertyAnimation* animOldOp = new QPropertyAnimation(effOld, "opacity");
    animOldOp->setDuration(350); animOldOp->setStartValue(1.0); animOldOp->setEndValue(0.0);
    QPropertyAnimation* animNewPos = new QPropertyAnimation(ui->currentAlgoLabel, "geometry");
    animNewPos->setDuration(350); animNewPos->setStartValue(startGeo); animNewPos->setEndValue(finalGeo); animNewPos->setEasingCurve(QEasingCurve::OutCubic);
    QPropertyAnimation* animNewOp = new QPropertyAnimation(effNew, "opacity");
    animNewOp->setDuration(350); animNewOp->setStartValue(0.0); animNewOp->setEndValue(1.0);

    group->addAnimation(animOldPos); group->addAnimation(animOldOp); group->addAnimation(animNewPos); group->addAnimation(animNewOp);

    connect(group, &QParallelAnimationGroup::finished, [this, tempLabel]() {
        tempLabel->deleteLater();
        ui->currentAlgoLabel->setGraphicsEffect(nullptr);
        isAnimating = false;
        });

    group->start(QAbstractAnimation::DeleteWhenStopped);
}
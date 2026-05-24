#pragma once
#include <QWidget>
#include <vector>
#include "SegmentationAlgorithm.h"

class QCheckBox;
class QPushButton;
class QStackedWidget;
class QSpinBox;
class QDoubleSpinBox;

namespace Ui { class StartScreen; }

class StartScreen : public QWidget {
    Q_OBJECT
public:
    explicit StartScreen(QWidget* parent = nullptr);
    ~StartScreen();

signals:
    void segmentRequested(int algorithmIndex, const QString& algorithmName, const QString& imagePath, const AlgoParameters& params);

private slots:
    void on_segmentButton_clicked();
    void on_prevAlgoButton_clicked();
    void on_nextAlgoButton_clicked();

private:
    Ui::StartScreen* ui;
    std::vector<QString> algorithms;
    int currentIndex;
    bool isAnimating;
    void switchAlgorithm(int direction);
    void setupDynamicUI();
    AlgoParameters collectParameters();

    QString selectedImagePath;
    QCheckBox* defaultCheckbox;
    QPushButton* uploadButton;
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
};
#pragma once
#include <QWidget>
#include <QTimer>
#include <QElapsedTimer>

namespace Ui { class ResultScreen; }

class ResultScreen : public QWidget {
    Q_OBJECT
public:
    explicit ResultScreen(QWidget* parent = nullptr);
    ~ResultScreen();
    void prepareProcessing(const QString& algoName);
    void showCpuResult(const QImage& image, double time);
    void showGpuResult(const QImage& image, double time);
    void cleanScreen();

signals:
    void backRequested();

private slots:
    void on_backButton_clicked();
    void updateLiveTimer();
    void updateSpinner();

private:
    Ui::ResultScreen* ui;
    QTimer* liveTimer;
    QTimer* spinnerTimer;
    QElapsedTimer elapsedTimer;
    int spinnerAngle;
    bool cpuDone;
    bool gpuDone;
};
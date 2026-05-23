#ifndef RESULTSCREEN_H
#define RESULTSCREEN_H

#include <QWidget>
#include <QString>
#include <QImage>

namespace Ui {
    class ResultScreen;
}

class ResultScreen : public QWidget {
    Q_OBJECT

public:
    explicit ResultScreen(QWidget* parent = nullptr);
    ~ResultScreen();

    void setResults(const QImage& cpuImage, const QImage& gpuImage,
        double cpuTimeMs, double gpuTimeMs, const QString& algoName);

signals:
    void backRequested();

private slots:
    void on_backButton_clicked();

private:
    Ui::ResultScreen* ui;
};

#endif
#include "ResultScreen.h"
#include "ui_ResultScreen.h"

ResultScreen::ResultScreen(QWidget* parent) :
    QWidget(parent),
    ui(new Ui::ResultScreen) {

    ui->setupUi(this);

    connect(ui->backButton, &QPushButton::clicked, this, &ResultScreen::on_backButton_clicked);
}

ResultScreen::~ResultScreen() {
    delete ui;
}

void ResultScreen::setResults(const QImage& cpuImage, const QImage& gpuImage,
    double cpuTimeMs, double gpuTimeMs, const QString& algoName) {
    ui->algoNameLabel->setText(algoName);

    ui->cpuImageLabel->setPixmap(QPixmap::fromImage(cpuImage).scaled(
        ui->cpuImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    ui->gpuImageLabel->setPixmap(QPixmap::fromImage(gpuImage).scaled(
        ui->gpuImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));

    ui->cpuTimeLabel->setText(QString::number(cpuTimeMs, 'f', 2) + " ms");
    ui->gpuTimeLabel->setText(QString::number(gpuTimeMs, 'f', 2) + " ms");

}

void ResultScreen::on_backButton_clicked() {
    emit backRequested();
}
#include "ResultScreen.h"
#include "ui_ResultScreen.h"
#include <QPainter>
#include <QPen>

ResultScreen::ResultScreen(QWidget* parent) : QWidget(parent), ui(new Ui::ResultScreen), spinnerAngle(0) {
    ui->setupUi(this);
    liveTimer = new QTimer(this);
    spinnerTimer = new QTimer(this);

    connect(liveTimer, &QTimer::timeout, this, &ResultScreen::updateLiveTimer);
    connect(spinnerTimer, &QTimer::timeout, this, &ResultScreen::updateSpinner);
    connect(ui->backButton, &QPushButton::clicked, this, &ResultScreen::on_backButton_clicked);
}

ResultScreen::~ResultScreen() {
    delete ui;
}

void ResultScreen::prepareProcessing(const QString& algoName) {
    cleanScreen();
    ui->algoNameLabel->setText(algoName);
    ui->backButton->setEnabled(false);

    elapsedTimer.start();
    liveTimer->start(10);
    spinnerTimer->start(30);
}

void ResultScreen::showResults(const QImage& cpuImage, double cpuTime, const QImage& gpuImage, double gpuTime) {
    liveTimer->stop();
    spinnerTimer->stop();

    ui->cpuImageLabel->setPixmap(QPixmap::fromImage(cpuImage).scaled(ui->cpuImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    ui->gpuImageLabel->setPixmap(QPixmap::fromImage(gpuImage).scaled(ui->gpuImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    ui->cpuTimeLabel->setText(QString::number(cpuTime, 'f', 2) + " ms");
    ui->gpuTimeLabel->setText(QString::number(gpuTime, 'f', 2) + " ms");
    ui->backButton->setEnabled(true);
}

void ResultScreen::cleanScreen() {
    liveTimer->stop();
    spinnerTimer->stop();
    ui->cpuImageLabel->clear();
    ui->gpuImageLabel->clear();
    ui->cpuTimeLabel->setText("0.00 ms");
    ui->gpuTimeLabel->setText("0.00 ms");
    spinnerAngle = 0;
}

void ResultScreen::updateLiveTimer() {
    QString timeStr = QString::number(elapsedTimer.elapsed()) + ".00 ms";
    ui->cpuTimeLabel->setText(timeStr);
    ui->gpuTimeLabel->setText(timeStr);
}

void ResultScreen::updateSpinner() {
    auto drawSpinner = [this](QLabel* label) {
        QPixmap pix(label->size());
        pix.fill(Qt::transparent);
        QPainter p(&pix);
        p.setRenderHint(QPainter::Antialiasing);
        int size = qMin(pix.width(), pix.height()) / 4;
        QRect rect((pix.width() - size) / 2, (pix.height() - size) / 2, size, size);
        QPen pen(QColor("#1E1E1E"), 8);
        p.setPen(pen);
        p.drawArc(rect, spinnerAngle * 16, 120 * 16);
        label->setPixmap(pix);
        };

    drawSpinner(ui->cpuImageLabel);
    drawSpinner(ui->gpuImageLabel);
    spinnerAngle = (spinnerAngle + 10) % 360;
}

void ResultScreen::on_backButton_clicked() {
    cleanScreen();
    emit backRequested();
}
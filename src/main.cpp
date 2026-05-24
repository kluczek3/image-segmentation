#include <QApplication>
#include <QStackedWidget>
#include <QIcon>
#include <QImage>
#include "StartScreen.h"
#include "ResultScreen.h"
#include "CoreManager.h"
#include "ImageConverter.h"
#include <QLabel>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    QString style = R"(
        QWidget { background-color: #FAFAFA; }
        QPushButton#segmentButton, QPushButton#backButton {
            background-color: #1E1E1E; color: #FFFFFF; border: none;
            border-radius: 8px; padding: 12px 32px;
            font-family: 'Consolas'; font-size: 16px; font-weight: bold;
        }
        QPushButton#segmentButton:hover, QPushButton#backButton:hover { background-color: #3A3A3A; }
        QPushButton#segmentButton:pressed, QPushButton#backButton:pressed { background-color: #000000; }
        QPushButton#backButton:disabled {
            background-color: #CCCCCC; color: #888888;
        }
        QPushButton#prevAlgoButton, QPushButton#nextAlgoButton {
            background-color: transparent; color: #1E1E1E; border: 2px solid #1E1E1E;
            border-radius: 8px; padding: 8px 16px;
            font-family: 'Consolas'; font-size: 16px; font-weight: bold;
        }
        QPushButton#prevAlgoButton:hover, QPushButton#nextAlgoButton:hover { background-color: #E0E0E0; }
        QPushButton#prevAlgoButton:pressed, QPushButton#nextAlgoButton:pressed { background-color: #CCCCCC; }
        QLabel { font-family: 'Consolas'; color: #1E1E1E; }
        QLabel#titleLabel { font-size: 36px; font-weight: bold; }
        QLabel#currentAlgoLabel { font-size: 22px; }
    )";

    QStackedWidget mainWindow;
    mainWindow.setWindowTitle("Segmentation CUDA vs CPU");
    mainWindow.resize(800, 800);
    mainWindow.setStyleSheet(style);

    StartScreen* startScreen = new StartScreen();
    ResultScreen* resultScreen = new ResultScreen();
    CoreManager* coreManager = new CoreManager(&mainWindow);

    mainWindow.addWidget(startScreen);
    mainWindow.addWidget(resultScreen);

    std::string path = std::string(PROJECT_SOURCE_DIR) + "/assets/peppers_color.tif";
    cv::Mat cvInput = cv::imread(path);
    QImage inputImage = ImageConverter::CvMatToQImage(cvInput);

    QObject::connect(startScreen, &StartScreen::segmentRequested, [&](int algoIndex, const QString& algoName) {
        if (inputImage.isNull()) return;
        mainWindow.setCurrentIndex(1);
        coreManager->startProcessing(inputImage, algoIndex);
        });

    QObject::connect(coreManager, &CoreManager::processingStarted, [&]() {
        resultScreen->prepareProcessing(startScreen->findChild<QLabel*>("currentAlgoLabel")->text());
        });

    QObject::connect(coreManager, &CoreManager::processingFinished, [&](const QImage& cpuImage, double cpuTime, const QImage& gpuImage, double gpuTime) {
        resultScreen->showResults(cpuImage, cpuTime, gpuImage, gpuTime);
        });

    QObject::connect(resultScreen, &ResultScreen::backRequested, [&]() {
        coreManager->cancelProcessing();
        mainWindow.setCurrentIndex(0);
        });

    mainWindow.show();
    return app.exec();
}
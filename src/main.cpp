#include <QApplication>
#include <QStackedWidget>
#include <QIcon>
#include <QImage>
#include "StartScreen.h"
#include "ResultScreen.h"
#include "CoreManager.h"
#include "ImageConverter.h"
#include <qlabel.h>

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
        QPushButton#backButton:disabled { background-color: #CCCCCC; color: #888888; }
        
        QPushButton#prevAlgoButton, QPushButton#nextAlgoButton {
            background-color: transparent; color: #1E1E1E; border: 2px solid #1E1E1E;
            border-radius: 8px; padding: 8px 16px; font-family: 'Consolas'; font-size: 16px; font-weight: bold;
        }
        QPushButton#prevAlgoButton:hover, QPushButton#nextAlgoButton:hover { background-color: #E0E0E0; }
        QPushButton#prevAlgoButton:pressed, QPushButton#nextAlgoButton:pressed { background-color: #CCCCCC; }
        
        QPushButton#uploadButton {
            background-color: #F0F0F0; color: #1E1E1E; border: 2px dashed #1E1E1E;
            border-radius: 8px; padding: 12px 16px; font-family: 'Consolas'; font-size: 14px;
        }
        QPushButton#uploadButton:hover:enabled { background-color: #E0E0E0; border-style: solid; }
        QPushButton#uploadButton:disabled { border-color: #CCCCCC; color: #AAAAAA; background-color: transparent; }
        
        QCheckBox#defaultCheckbox { font-family: 'Consolas'; font-size: 15px; color: #1E1E1E; padding: 8px 0; }
        QCheckBox#defaultCheckbox::indicator { width: 20px; height: 20px; border: 2px solid #1E1E1E; border-radius: 4px; background: #FFFFFF; }
        QCheckBox#defaultCheckbox::indicator:checked { background-color: #1E1E1E; image: url(checked.png); }
        QCheckBox#defaultCheckbox::indicator:disabled { border-color: #CCCCCC; }
        
        QStackedWidget#paramStack { background: transparent; border: 1px solid #E0E0E0; border-radius: 8px; }
        
        QSpinBox, QDoubleSpinBox {
            padding: 6px 10px; border: 2px solid #E0E0E0; border-radius: 6px;
            font-family: 'Consolas'; font-size: 15px; background: #FFFFFF; color: #1E1E1E;
            selection-background-color: #1E1E1E; selection-color: #FFFFFF;
        }
        QSpinBox:focus, QDoubleSpinBox:focus { border-color: #1E1E1E; }
        QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {
            width: 24px; border: none; background: #F0F0F0; border-radius: 2px; margin: 1px;
        }
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #E0E0E0; }
        
        QLabel { color: #1E1E1E; }
        QLabel#titleLabel { font-size: 36px; font-weight: bold; }
        QLabel#currentAlgoLabel { font-size: 24px; font-weight: bold; }
    )";

    QStackedWidget mainWindow;
    mainWindow.setWindowTitle("Segmentation CUDA vs CPU");
    mainWindow.resize(800, 600);
    mainWindow.setStyleSheet(style);

    StartScreen* startScreen = new StartScreen();
    ResultScreen* resultScreen = new ResultScreen();
    CoreManager* coreManager = new CoreManager(&mainWindow);

    mainWindow.addWidget(startScreen);
    mainWindow.addWidget(resultScreen);

    QObject::connect(startScreen, &StartScreen::segmentRequested, [&](int algoIndex, const QString& algoName, const QString& imagePath, const AlgoParameters& params) {
        cv::Mat cvInput = cv::imread(imagePath.toStdString());
        if (cvInput.empty()) 
            return;

        mainWindow.setCurrentIndex(1);
        coreManager->startProcessing(cvInput, algoIndex, params);
        });

    QObject::connect(coreManager, &CoreManager::processingStarted, [&]() {
        resultScreen->prepareProcessing(startScreen->findChild<QLabel*>("currentAlgoLabel")->text());
        });

    QObject::connect(coreManager, &CoreManager::cpuFinished, [&](const QImage& img, double time) {
        resultScreen->showCpuResult(img, time);
        });

    QObject::connect(coreManager, &CoreManager::gpuFinished, [&](const QImage& img, double time) {
        resultScreen->showGpuResult(img, time);
        });

    QObject::connect(resultScreen, &ResultScreen::backRequested, [&]() {
        coreManager->cancelProcessing();
        mainWindow.setCurrentIndex(0);
        });

    mainWindow.show();
    return app.exec();
}
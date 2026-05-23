#include <QApplication>
#include <QStackedWidget>
#include <QIcon>
#include "StartScreen.h"
#include "ResultScreen.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    QString style = R"(
        QWidget {
            background-color: #FAFAFA;
        }
        QPushButton#segmentButton, QPushButton#backButton {
            background-color: #1E1E1E;
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 12px 32px;
            font-family: 'Consolas';
            font-size: 16px;
            font-weight: bold;
        }
        QPushButton#segmentButton:hover, QPushButton#backButton:hover {
            background-color: #3A3A3A;
        }
        QPushButton#segmentButton:pressed, QPushButton#backButton:pressed {
            background-color: #000000;
        }
        QPushButton#prevAlgoButton, QPushButton#nextAlgoButton {
            background-color: transparent;
            color: #1E1E1E;
            border: 2px solid #1E1E1E;
            border-radius: 8px;
            padding: 8px 16px;
            font-family: 'Consolas';
            font-size: 16px;
            font-weight: bold;
        }
        QPushButton#prevAlgoButton:hover, QPushButton#nextAlgoButton:hover {
            background-color: #E0E0E0;
        }
        QPushButton#prevAlgoButton:pressed, QPushButton#nextAlgoButton:pressed {
            background-color: #CCCCCC;
        }
        QLabel {
            font-family: 'Consolas';
            color: #1E1E1E;
        }
        QLabel#titleLabel {
            font-size: 36px;
            font-weight: bold;
        }
        QLabel#currentAlgoLabel {
            font-size: 22px;
        }
    )";

    QStackedWidget mainWindow;
    mainWindow.setWindowTitle("Segmentation CUDA vs CPU");
    mainWindow.setWindowIcon(QIcon());
    mainWindow.resize(800, 800);
    mainWindow.setStyleSheet(style);

    StartScreen* startScreen = new StartScreen();
    ResultScreen* resultScreen = new ResultScreen();

    mainWindow.addWidget(startScreen);
    mainWindow.addWidget(resultScreen);

    QObject::connect(startScreen, &StartScreen::segmentRequested, [&]() {
        mainWindow.setCurrentIndex(1);
        });

    QObject::connect(resultScreen, &ResultScreen::backRequested, [&]() {
        mainWindow.setCurrentIndex(0);
        });

    mainWindow.show();

    return app.exec();
}
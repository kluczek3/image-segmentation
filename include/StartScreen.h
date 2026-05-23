#ifndef STARTSCREEN_H
#define STARTSCREEN_H

#include <QWidget>
#include <QStringList>

namespace Ui {
    class StartScreen;
}

class StartScreen : public QWidget {
    Q_OBJECT

public:
    explicit StartScreen(QWidget* parent = nullptr);
    ~StartScreen();

signals:
    void segmentRequested();

private slots:
    void on_segmentButton_clicked();
    void on_prevAlgoButton_clicked();
    void on_nextAlgoButton_clicked();

private:
    Ui::StartScreen* ui;

    QStringList algorithms;
    int currentIndex;
    bool isAnimating;

    void switchAlgorithm(int direction);
};

#endif
#pragma once
#include <QWidget>
#include <vector>

namespace Ui { class StartScreen; }

class StartScreen : public QWidget {
    Q_OBJECT
public:
    explicit StartScreen(QWidget* parent = nullptr);
    ~StartScreen();

signals:
    void segmentRequested(int algorithmIndex, const QString& algorithmName);

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
};
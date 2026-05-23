#include "StartScreen.h"
#include "ui/ui_StartScreen.h"
#include <QPropertyAnimation>
#include <QParallelAnimationGroup>
#include <QGraphicsOpacityEffect>
#include <QLabel>

StartScreen::StartScreen(QWidget* parent) :
    QWidget(parent),
    ui(new Ui::StartScreen),
    currentIndex(0),
    isAnimating(false) {

    ui->setupUi(this);

    algorithms = { "K-Means", "Fuzzy C-Means", "Otsu Thresholding", "Mean Shift" };
    ui->currentAlgoLabel->setText(algorithms[currentIndex]);

    connect(ui->segmentButton, &QPushButton::clicked, this, &StartScreen::on_segmentButton_clicked);
    connect(ui->prevAlgoButton, &QPushButton::clicked, this, &StartScreen::on_prevAlgoButton_clicked);
    connect(ui->nextAlgoButton, &QPushButton::clicked, this, &StartScreen::on_nextAlgoButton_clicked);
}

StartScreen::~StartScreen() {
    delete ui;
}

void StartScreen::on_segmentButton_clicked() {
    emit segmentRequested();
}

void StartScreen::on_prevAlgoButton_clicked() {
    switchAlgorithm(-1);
}

void StartScreen::on_nextAlgoButton_clicked() {
    switchAlgorithm(1);
}

void StartScreen::switchAlgorithm(int direction) {
    if (isAnimating) return;
    isAnimating = true;

    currentIndex = (currentIndex + direction + algorithms.size()) % algorithms.size();
    QString newText = algorithms[currentIndex];

    QLabel* tempLabel = new QLabel(ui->currentAlgoLabel->text(), this);
    tempLabel->setFont(ui->currentAlgoLabel->font());
    tempLabel->setAlignment(ui->currentAlgoLabel->alignment());
    tempLabel->setStyleSheet(ui->currentAlgoLabel->styleSheet());
    tempLabel->setGeometry(ui->currentAlgoLabel->geometry());
    tempLabel->show();

    ui->currentAlgoLabel->setText(newText);

    QRect finalGeo = ui->currentAlgoLabel->geometry();
    QRect startGeo = finalGeo;
    int offset = 60 * direction;
    startGeo.translate(offset, 0);
    ui->currentAlgoLabel->setGeometry(startGeo);

    QGraphicsOpacityEffect* effNew = new QGraphicsOpacityEffect(this);
    ui->currentAlgoLabel->setGraphicsEffect(effNew);

    QGraphicsOpacityEffect* effOld = new QGraphicsOpacityEffect(this);
    tempLabel->setGraphicsEffect(effOld);

    QParallelAnimationGroup* group = new QParallelAnimationGroup(this);

    QPropertyAnimation* animOldPos = new QPropertyAnimation(tempLabel, "geometry");
    animOldPos->setDuration(350);
    animOldPos->setEndValue(finalGeo.translated(-offset, 0));
    animOldPos->setEasingCurve(QEasingCurve::OutCubic);

    QPropertyAnimation* animOldOp = new QPropertyAnimation(effOld, "opacity");
    animOldOp->setDuration(350);
    animOldOp->setStartValue(1.0);
    animOldOp->setEndValue(0.0);

    QPropertyAnimation* animNewPos = new QPropertyAnimation(ui->currentAlgoLabel, "geometry");
    animNewPos->setDuration(350);
    animNewPos->setStartValue(startGeo);
    animNewPos->setEndValue(finalGeo);
    animNewPos->setEasingCurve(QEasingCurve::OutCubic);

    QPropertyAnimation* animNewOp = new QPropertyAnimation(effNew, "opacity");
    animNewOp->setDuration(350);
    animNewOp->setStartValue(0.0);
    animNewOp->setEndValue(1.0);

    group->addAnimation(animOldPos);
    group->addAnimation(animOldOp);
    group->addAnimation(animNewPos);
    group->addAnimation(animNewOp);

    connect(group, &QParallelAnimationGroup::finished, [this, tempLabel]() {
        tempLabel->deleteLater();
        ui->currentAlgoLabel->setGraphicsEffect(nullptr);
        isAnimating = false;
        });

    group->start(QAbstractAnimation::DeleteWhenStopped);
}
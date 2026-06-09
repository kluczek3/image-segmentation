#include <QApplication>
#include "StartScreen.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    const QString style = R"(
        QWidget {
            background-color: #F6F7FB;
            color: #171A21;
            font-family: 'Consolas';
        }

        QLabel#titleLabel {
            font-size: 30px;
            font-weight: 800;
            padding: 4px 0;
        }

        QLabel#subtitleLabel {
            color: #5F6776;
            font-size: 14px;
            padding-bottom: 4px;
        }

        QLabel#currentAlgorithmLabel,
        QLabel#resultsTitleLabel,
        QLabel#sourceTitleLabel,
        QLabel#paramsTitleLabel,
        QLabel#cpuTypeLabel,
        QLabel#gpuTypeLabel {
            font-size: 16px;
            font-weight: 700;
        }

        QLabel#algorithmHintLabel,
        QLabel#selectedImageLabel,
        QLabel#statusLabel,
        QLabel#helpLabel,
        QLabel#cpuTimeLabel,
        QLabel#gpuTimeLabel,
        QLabel#cpuSummaryLabel,
        QLabel#gpuSummaryLabel,
        QLabel#infoTextLabel {
            font-size: 13px;
        }

        QFrame#controlsFrame,
        QFrame#cpuPanel,
        QFrame#gpuPanel,
        QFrame#sourceCard,
        QFrame#paramsCard,
        QFrame#actionsCard {
            background-color: #FFFFFF;
            border: 1px solid #E3E6EE;
            border-radius: 16px;
        }

        QPushButton#segmentButton,
        QPushButton#showOriginalButton {
            background-color: #171A21;
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            padding: 12px 16px;
            font-size: 14px;
            font-weight: 700;
        }

        QPushButton#segmentButton:hover,
        QPushButton#showOriginalButton:hover {
            background-color: #2A2F3A;
        }

        QPushButton#segmentButton:pressed,
        QPushButton#showOriginalButton:pressed {
            background-color: #0D1016;
        }

        QPushButton#uploadButton {
            background-color: #F7F8FB;
            color: #171A21;
            border: 1px dashed #A8B0C0;
            border-radius: 10px;
            padding: 10px 12px;
            font-size: 13px;
        }

        QPushButton#uploadButton:hover:enabled {
            background-color: #EEF1F7;
            border-style: solid;
        }

        QPushButton#uploadButton:disabled {
            color: #9CA3AF;
            border-color: #D5DAE5;
            background-color: #FBFBFD;
        }

        QPushButton#algorithmButton {
            background-color: #EDF0F5;
            color: #778194;
            border: 1px solid #D7DDE9;
            border-radius: 18px;
            padding: 10px 16px;
            font-size: 14px;
            font-weight: 700;
        }

        QPushButton#algorithmButton:hover:!checked {
            background-color: #E6EAF3;
            color: #4F596B;
        }

        QPushButton#algorithmButton:checked {
            background-color: #171A21;
            color: #FFFFFF;
            border: 1px solid #171A21;
        }

        QLabel#cpuImageLabel,
        QLabel#gpuImageLabel {
            background-color: #F8FAFD;
            border: 1px dashed #D5DAE5;
            border-radius: 12px;
            padding: 4px;
        }

        QLabel#cpuSummaryLabel,
        QLabel#gpuSummaryLabel,
        QLabel#helpLabel,
        QLabel#statusLabel,
        QLabel#algorithmHintLabel,
        QLabel#selectedImageLabel,
        QLabel#infoTextLabel {
            background-color: #F8FAFD;
            border: 1px solid #E8ECF3;
            border-radius: 10px;
            padding: 10px;
        }

        QCheckBox#defaultCheckbox {
            font-size: 13px;
            spacing: 8px;
        }

        QCheckBox#defaultCheckbox::indicator {
            width: 16px;
            height: 16px;
        }

        QSpinBox,
        QDoubleSpinBox {
            background-color: #FFFFFF;
            border: 1px solid #D7DDE9;
            border-radius: 8px;
            padding: 6px 10px;
            font-size: 13px;
            min-height: 20px;
        }

        QSpinBox:focus,
        QDoubleSpinBox:focus {
            border-color: #171A21;
        }
    )";

    app.setStyleSheet(style);

    StartScreen window;
    window.resize(1440, 860);
    window.show();

    return app.exec();
}

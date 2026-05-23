#pragma once
#include <QImage>
#include <opencv2/opencv.hpp>

class ImageConverter {
public:
    static cv::Mat QImageToCvMat(const QImage& inImage) {
        QImage swapped = inImage.convertToFormat(QImage::Format_RGB888).rgbSwapped();
        return cv::Mat(swapped.height(), swapped.width(),
            CV_8UC3,
            const_cast<uchar*>(swapped.bits()),
            static_cast<size_t>(swapped.bytesPerLine())).clone();
    }

    static QImage CvMatToQImage(const cv::Mat& inMat) {
        if (inMat.type() == CV_8UC3) {
            cv::Mat rgb;
            cv::cvtColor(inMat, rgb, cv::COLOR_BGR2RGB);
            return QImage(rgb.data, rgb.cols, rgb.rows,
                static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
        }
        else if (inMat.type() == CV_8UC1) {
            return QImage(inMat.data, inMat.cols, inMat.rows,
                static_cast<int>(inMat.step), QImage::Format_Grayscale8).copy();
        }
        return QImage();
    }
};
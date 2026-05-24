#include "CPU_Algorithms.h"
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>

std::pair<cv::Mat, double> OtsuCPU::execute(const cv::Mat& inputImage) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat gray;

    if (inputImage.channels() == 3) cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    else gray = inputImage.clone();

    std::vector<int> histogram(256, 0);
    int totalPixels = gray.rows * gray.cols;
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) 
            histogram[gray.at<uchar>(i, j)]++;
    }

    float sum = 0;
    for (int i = 0; i < 256; ++i) sum += i * histogram[i];

    float sumB = 0, varMax = 0;
    int wB = 0, wF = 0, threshold = 0;
    for (int i = 0; i < 256; ++i) {
        wB += histogram[i];
        if (wB == 0) continue;
        
        wF = totalPixels - wB;
        if (wF == 0) break;
        
        sumB += static_cast<float>(i * histogram[i]);
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        float varBetween = static_cast<float>(wB) * static_cast<float>(wF) * (mB - mF) * (mB - mF);
        
        if (varBetween > varMax) { varMax = varBetween; threshold = i; }
    }

    cv::Mat result = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (int i = 0; i < gray.rows; ++i) {
        for (int j = 0; j < gray.cols; ++j) {
            if (gray.at<uchar>(i, j) >= threshold) result.at<uchar>(i, j) = 255;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    return std::make_pair(result, time);
}

KMeansCPU::KMeansCPU(int clusters, int maxIter) : k(clusters), maxIterations(maxIter) {}

std::pair<cv::Mat, double> KMeansCPU::execute(const cv::Mat& inputImage) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat data;
    inputImage.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());

    std::vector<cv::Vec3f> centroids(k);
    for (int i = 0; i < k; ++i) centroids[i] = data.at<cv::Vec3f>(std::rand() % data.rows, 0);

    std::vector<int> labels(data.rows, -1);
    bool changed = true;
    int iter = 0;
   
    while (changed && iter < maxIterations) {
        changed = false;
        std::vector<cv::Vec3f> newCentroids(k, cv::Vec3f(0, 0, 0));
        std::vector<int> counts(k, 0);
       
        for (int i = 0; i < data.rows; ++i) {
            cv::Vec3f pixel = data.at<cv::Vec3f>(i, 0);
            int best = 0; float minDist = std::numeric_limits<float>::max();
            
            for (int c = 0; c < k; ++c) {
                float dist = cv::norm(pixel - centroids[c]);
                if (dist < minDist) { minDist = dist; best = c; }
            }
            if (labels[i] != best) { labels[i] = best; changed = true; }
            newCentroids[best] += pixel;
            counts[best]++;
        }
        for (int c = 0; c < k; ++c) if (counts[c] > 0) centroids[c] = newCentroids[c] / (float)counts[c];
        iter++;
    }
    cv::Mat result(data.size(), data.type());
    
    for (int i = 0; i < data.rows; ++i) 
        result.at<cv::Vec3f>(i, 0) = centroids[labels[i]];
    
    result = result.reshape(3, inputImage.rows);
    result.convertTo(result, CV_8UC3);
    
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    return std::make_pair(result, time);
}

FCMCPU::FCMCPU(int clusters, int maxIter, float fuzziness, float epsilon) : k(clusters), maxIterations(maxIter), m(fuzziness), eps(epsilon) {}

std::pair<cv::Mat, double> FCMCPU::execute(const cv::Mat& inputImage) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat data;
    inputImage.convertTo(data, CV_32F);
    data = data.reshape(1, data.total());
    int numPoints = data.rows;
   
    std::vector<cv::Vec3f> centroids(k);
    std::vector<std::vector<float>> U(numPoints, std::vector<float>(k));
    
    for (int i = 0; i < numPoints; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) { U[i][j] = (float)std::rand() / RAND_MAX; sum += U[i][j]; }
        for (int j = 0; j < k; ++j) U[i][j] /= sum;
    }
    int iter = 0;
    
    while (iter < maxIterations) {
        std::vector<cv::Vec3f> newCentroids(k, cv::Vec3f(0, 0, 0));
        for (int j = 0; j < k; ++j) {
            cv::Vec3f num(0, 0, 0); float den = 0.0f;
            for (int i = 0; i < numPoints; ++i) { 
                float u_m = std::pow(U[i][j], m); 
                num += data.at<cv::Vec3f>(i, 0) * u_m; 
                den += u_m; 
            }
            if (den > 0) 
                newCentroids[j] = num / den;
        }
        float maxDiff = 0.0f;
        for (int i = 0; i < numPoints; ++i) {
            for (int j = 0; j < k; ++j) {
                float dist = cv::norm(data.at<cv::Vec3f>(i, 0) - newCentroids[j]);
                float sumD = 0.0f;
                
                for (int c = 0; c < k; ++c) { 
                    float dC = cv::norm(data.at<cv::Vec3f>(i, 0) - newCentroids[c]); 
                    if (dC > 0) 
                        sumD += std::pow(dist / dC, 2.0f / (m - 1.0f)); 
                }
                float nU = (sumD > 0) ? (1.0f / sumD) : 1.0f;
                maxDiff = std::max(maxDiff, std::abs(U[i][j] - nU));
                U[i][j] = nU;
            }
        }
        centroids = newCentroids;
        
        if (maxDiff < eps) 
            break;
        iter++;
    }
    cv::Mat result(data.size(), data.type());
    
    for (int i = 0; i < numPoints; ++i) {
        int best = 0; float maxU = 0.0f;
        
        for (int j = 0; j < k; ++j) if (U[i][j] > maxU) {
            maxU = U[i][j]; 
            best = j; 
        }
        result.at<cv::Vec3f>(i, 0) = centroids[best];
    }
    result = result.reshape(3, inputImage.rows);
    result.convertTo(result, CV_8UC3);
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    return std::make_pair(result, time);
}

MeanShiftCPU::MeanShiftCPU(float spatialBandwidth, float colorBandwidth, int maxIter) : hs(spatialBandwidth), hr(colorBandwidth), maxIterations(maxIter) {}

std::pair<cv::Mat, double> MeanShiftCPU::execute(const cv::Mat& inputImage) {
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat result;
    inputImage.convertTo(result, CV_32F);
    int rows = result.rows, cols = result.cols;
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        cv::Mat next = result.clone();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cv::Vec3f curr = result.at<cv::Vec3f>(i, j), sumC(0, 0, 0);
                float weight = 0.0f;
                for (int r = std::max(0, i - (int)hs); r <= std::min(rows - 1, i + (int)hs); ++r) {
                    for (int c = std::max(0, j - (int)hs); c <= std::min(cols - 1, j + (int)hs); ++c) {
                        cv::Vec3f neighbor = result.at<cv::Vec3f>(r, c);
                        if (std::pow(i - r, 2) + std::pow(j - c, 2) <= hs * hs && cv::norm(curr - neighbor) <= hr) {
                            sumC += neighbor; weight += 1.0f;
                        }
                    }
                }
                if (weight > 0) 
                    next.at<cv::Vec3f>(i, j) = sumC / weight;
            }
        }
        result = next;
    }
    result.convertTo(result, CV_8UC3);
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    
    return std::make_pair(result, time);
}
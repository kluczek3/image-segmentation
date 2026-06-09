#include "CPU_Algorithms.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
constexpr float kDistanceEpsilon = 1e-12f;

cv::Mat ensureBgr8(const cv::Mat& inputImage) {
    if (inputImage.empty()) {
        return cv::Mat();
    }

    if (inputImage.type() == CV_8UC3) {
        return inputImage.clone();
    }

    if (inputImage.channels() == 1) {
        cv::Mat gray;
        if (inputImage.type() == CV_8UC1) {
            gray = inputImage;
        } else {
            inputImage.convertTo(gray, CV_8UC1);
        }

        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }

    if (inputImage.channels() == 4) {
        cv::Mat bgra8;
        if (inputImage.type() == CV_8UC4) {
            bgra8 = inputImage;
        } else {
            inputImage.convertTo(bgra8, CV_8UC4);
        }

        cv::Mat bgr;
        cv::cvtColor(bgra8, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }

    cv::Mat converted;
    inputImage.convertTo(converted, CV_8UC3);
    return converted;
}

cv::Mat makeGray8(const cv::Mat& inputImage) {
    if (inputImage.empty()) {
        return cv::Mat();
    }

    if (inputImage.channels() == 1) {
        if (inputImage.type() == CV_8UC1) {
            return inputImage.clone();
        }

        cv::Mat gray;
        inputImage.convertTo(gray, CV_8UC1);
        return gray;
    }

    const cv::Mat bgr = ensureBgr8(inputImage);
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

float squaredColorDistance(const cv::Vec3f& a, const cv::Vec3f& b) {
    const cv::Vec3f diff = a - b;
    return diff.dot(diff);
}

int computeOtsuThreshold(const cv::Mat& gray) {
    std::vector<int> histogram(256, 0);
    const int totalPixels = gray.rows * gray.cols;

    for (int y = 0; y < gray.rows; ++y) {
        const uchar* row = gray.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            histogram[row[x]]++;
        }
    }

    double totalSum = 0.0;
    for (int i = 0; i < 256; ++i) {
        totalSum += static_cast<double>(i) * static_cast<double>(histogram[i]);
    }

    double backgroundSum = 0.0;
    double maxBetweenClassVariance = -1.0;
    int backgroundWeight = 0;
    int threshold = 0;

    for (int i = 0; i < 256; ++i) {
        backgroundWeight += histogram[i];
        if (backgroundWeight == 0) {
            continue;
        }

        const int foregroundWeight = totalPixels - backgroundWeight;
        if (foregroundWeight == 0) {
            break;
        }

        backgroundSum += static_cast<double>(i) * static_cast<double>(histogram[i]);
        const double meanBackground = backgroundSum / static_cast<double>(backgroundWeight);
        const double meanForeground = (totalSum - backgroundSum) / static_cast<double>(foregroundWeight);
        const double betweenClassVariance =
            static_cast<double>(backgroundWeight) *
            static_cast<double>(foregroundWeight) *
            (meanBackground - meanForeground) *
            (meanBackground - meanForeground);

        if (betweenClassVariance > maxBetweenClassVariance) {
            maxBetweenClassVariance = betweenClassVariance;
            threshold = i;
        }
    }

    return threshold;
}

cv::Mat buildBinaryMask(const cv::Mat& gray, int threshold) {
    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < gray.rows; ++y) {
        const uchar* srcRow = gray.ptr<uchar>(y);
        uchar* dstRow = mask.ptr<uchar>(y);
        for (int x = 0; x < gray.cols; ++x) {
            dstRow[x] = (srcRow[x] >= threshold) ? 255 : 0;
        }
    }

    return mask;
}

cv::Vec3b meanColorOrFallback(const cv::Vec3d& sum, int count, const cv::Vec3b& fallback) {
    if (count <= 0) {
        return fallback;
    }

    return cv::Vec3b(
        cv::saturate_cast<uchar>(sum[0] / static_cast<double>(count)),
        cv::saturate_cast<uchar>(sum[1] / static_cast<double>(count)),
        cv::saturate_cast<uchar>(sum[2] / static_cast<double>(count))
    );
}

uchar quantizedChannel(float value, float step) {
    const float safeStep = std::max(1.0f, step);
    const float quantized = std::round(value / safeStep) * safeStep;
    return cv::saturate_cast<uchar>(quantized);
}

cv::Mat quantizeMeanShiftModes(const cv::Mat& filtered32, float colorBandwidth) {
    const float binSize = std::max(1.0f, colorBandwidth);
    cv::Mat segmented(filtered32.rows, filtered32.cols, CV_8UC3);

    for (int y = 0; y < filtered32.rows; ++y) {
        const cv::Vec3f* srcRow = filtered32.ptr<cv::Vec3f>(y);
        cv::Vec3b* dstRow = segmented.ptr<cv::Vec3b>(y);

        for (int x = 0; x < filtered32.cols; ++x) {
            const cv::Vec3f& p = srcRow[x];

            dstRow[x] = cv::Vec3b(
                quantizedChannel(p[0], binSize),
                quantizedChannel(p[1], binSize),
                quantizedChannel(p[2], binSize)
            );
        }
    }

    return segmented;
}

cv::Mat colorizeBinaryMaskWithClassMeans(const cv::Mat& originalBgr, const cv::Mat& binaryMask) {
    cv::Vec3d sums[2] = {
        cv::Vec3d(0.0, 0.0, 0.0),
        cv::Vec3d(0.0, 0.0, 0.0)
    };
    int counts[2] = { 0, 0 };

    for (int y = 0; y < originalBgr.rows; ++y) {
        const cv::Vec3b* colorRow = originalBgr.ptr<cv::Vec3b>(y);
        const uchar* maskRow = binaryMask.ptr<uchar>(y);
        for (int x = 0; x < originalBgr.cols; ++x) {
            const int bucket = (maskRow[x] > 0) ? 1 : 0;
            sums[bucket][0] += static_cast<double>(colorRow[x][0]);
            sums[bucket][1] += static_cast<double>(colorRow[x][1]);
            sums[bucket][2] += static_cast<double>(colorRow[x][2]);
            counts[bucket] += 1;
        }
    }

    const cv::Scalar overallMean = cv::mean(originalBgr);
    const cv::Vec3b fallback(
        cv::saturate_cast<uchar>(overallMean[0]),
        cv::saturate_cast<uchar>(overallMean[1]),
        cv::saturate_cast<uchar>(overallMean[2])
    );

    const cv::Vec3b classColors[2] = {
        meanColorOrFallback(sums[0], counts[0], fallback),
        meanColorOrFallback(sums[1], counts[1], fallback)
    };

    cv::Mat colorized(originalBgr.size(), CV_8UC3);
    for (int y = 0; y < colorized.rows; ++y) {
        const uchar* maskRow = binaryMask.ptr<uchar>(y);
        cv::Vec3b* dstRow = colorized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < colorized.cols; ++x) {
            dstRow[x] = classColors[(maskRow[x] > 0) ? 1 : 0];
        }
    }

    return colorized;
}
} // namespace

std::pair<cv::Mat, double> OtsuCPU::execute(const cv::Mat& inputImage) {
    auto start = std::chrono::high_resolution_clock::now();

    const cv::Mat originalBgr = ensureBgr8(inputImage);
    const cv::Mat gray = makeGray8(inputImage);

    const int threshold = computeOtsuThreshold(gray);
    const cv::Mat binaryMask = buildBinaryMask(gray, threshold);

    cv::Mat result = colorizeBinaryMaskWithClassMeans(originalBgr, binaryMask);

    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }

    if (result.type() != CV_8UC3) {
        result.convertTo(result, CV_8UC3);
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double time = std::chrono::duration<double, std::milli>(end - start).count();

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

FCMCPU::FCMCPU(int clusters, int maxIter, float fuzziness, float epsilon)
    : k(clusters), maxIterations(maxIter), m(fuzziness), eps(epsilon) {}

std::pair<cv::Mat, double> FCMCPU::execute(const cv::Mat& inputImage) {
    if (k < 2) {
        throw std::invalid_argument("FCM requires at least 2 clusters.");
    }
    if (m <= 1.0f) {
        throw std::invalid_argument("FCM fuzziness parameter m must be greater than 1.");
    }

    auto start = std::chrono::high_resolution_clock::now();

    const cv::Mat bgrInput = ensureBgr8(inputImage);
    cv::Mat data32;
    bgrInput.convertTo(data32, CV_32FC3);
    if (!data32.isContinuous()) {
        data32 = data32.clone();
    }

    const int rows = data32.rows;
    const int cols = data32.cols;
    const int numPoints = rows * cols;
    const cv::Vec3f* samples = data32.ptr<cv::Vec3f>(0);

    std::vector<float> U(static_cast<size_t>(numPoints) * static_cast<size_t>(k), 0.0f);
    std::vector<float> nextU(static_cast<size_t>(numPoints) * static_cast<size_t>(k), 0.0f);
    std::vector<cv::Vec3f> centroids(k, cv::Vec3f(0.0f, 0.0f, 0.0f));

    std::mt19937 rng(1337u);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (int i = 0; i < numPoints; ++i) {
        float rowSum = 0.0f;
        for (int j = 0; j < k; ++j) {
            const float value = distribution(rng);
            U[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)] = value;
            rowSum += value;
        }
        for (int j = 0; j < k; ++j) {
            U[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(j)] /= rowSum;
        }
    }

    const float membershipExponent = 1.0f / (m - 1.0f);

    for (int iter = 0; iter < maxIterations; ++iter) {
        std::vector<cv::Vec3f> newCentroids(k, cv::Vec3f(0.0f, 0.0f, 0.0f));
        std::vector<float> weights(k, 0.0f);

        for (int i = 0; i < numPoints; ++i) {
            const cv::Vec3f sample = samples[i];
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k);
            for (int j = 0; j < k; ++j) {
                const float w = std::pow(U[base + static_cast<size_t>(j)], m);
                newCentroids[j] += sample * w;
                weights[j] += w;
            }
        }

        for (int j = 0; j < k; ++j) {
            if (weights[j] > 0.0f) {
                centroids[j] = newCentroids[j] / weights[j];
            }
        }

        float maxDiff = 0.0f;
        std::vector<float> distances(static_cast<size_t>(k), 0.0f);

        for (int i = 0; i < numPoints; ++i) {
            const cv::Vec3f sample = samples[i];
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k);
            int zeroDistanceCount = 0;

            for (int j = 0; j < k; ++j) {
                distances[static_cast<size_t>(j)] = squaredColorDistance(sample, centroids[j]);
                if (distances[static_cast<size_t>(j)] <= kDistanceEpsilon) {
                    zeroDistanceCount++;
                }
            }

            if (zeroDistanceCount > 0) {
                const float sharedMembership = 1.0f / static_cast<float>(zeroDistanceCount);
                for (int j = 0; j < k; ++j) {
                    const float updatedMembership = (distances[static_cast<size_t>(j)] <= kDistanceEpsilon) ? sharedMembership : 0.0f;
                    maxDiff = std::max(maxDiff, std::abs(updatedMembership - U[base + static_cast<size_t>(j)]));
                    nextU[base + static_cast<size_t>(j)] = updatedMembership;
                }
                continue;
            }

            for (int j = 0; j < k; ++j) {
                double denominator = 0.0;
                const float distanceJ = distances[static_cast<size_t>(j)];
                for (int c = 0; c < k; ++c) {
                    denominator += std::pow(distanceJ / distances[static_cast<size_t>(c)], membershipExponent);
                }

                const float updatedMembership = static_cast<float>(1.0 / denominator);
                maxDiff = std::max(maxDiff, std::abs(updatedMembership - U[base + static_cast<size_t>(j)]));
                nextU[base + static_cast<size_t>(j)] = updatedMembership;
            }
        }

        U.swap(nextU);
        if (maxDiff < eps) {
            break;
        }
    }

    cv::Mat result(rows, cols, CV_8UC3);
    for (int i = 0; i < numPoints; ++i) {
        const size_t base = static_cast<size_t>(i) * static_cast<size_t>(k);
        int bestCluster = 0;
        float bestMembership = U[base];
        for (int j = 1; j < k; ++j) {
            const float membership = U[base + static_cast<size_t>(j)];
            if (membership > bestMembership) {
                bestMembership = membership;
                bestCluster = j;
            }
        }

        const cv::Vec3f& centroid = centroids[bestCluster];
        result.ptr<cv::Vec3b>(0)[i] = cv::Vec3b(
            cv::saturate_cast<uchar>(centroid[0]),
            cv::saturate_cast<uchar>(centroid[1]),
            cv::saturate_cast<uchar>(centroid[2])
        );
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double time = std::chrono::duration<double, std::milli>(end - start).count();

    return std::make_pair(result, time);
}

MeanShiftCPU::MeanShiftCPU(float spatialBandwidth, float colorBandwidth, int maxIter)
    : hs(spatialBandwidth), hr(colorBandwidth), maxIterations(maxIter) {}

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
    result = quantizeMeanShiftModes(result, hr);
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();

    return std::make_pair(result, time);
}
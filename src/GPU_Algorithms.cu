#include "GPU_Algorithms.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr float kDistanceEpsilon = 1e-12f;
constexpr int kMaxFcmClustersForCuda = 32;

void throwCudaError(cudaError_t status, const char* call, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string("CUDA error at ") + file + ":" + std::to_string(line) +
        " in " + call + ": " + cudaGetErrorString(status)
    );
}

#define CUDA_CHECK(call) throwCudaError((call), #call, __FILE__, __LINE__)

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

int computeOtsuThreshold(const int histogram[256], int totalPixels) {
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

__device__ float squaredDistance(const float3& a, const float3& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ float atomicMaxFloat(float* address, float value) {
    int* addressAsInt = reinterpret_cast<int*>(address);
    int old = *addressAsInt;

    while (__int_as_float(old) < value) {
        const int assumed = old;
        old = atomicCAS(addressAsInt, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }

    return __int_as_float(old);
}

__global__ void histoKernel(const uchar* gray, int size, int* histogram) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&histogram[gray[idx]], 1);
    }
}

__global__ void thresholdKernel(const uchar* gray, uchar* mask, int size, int threshold) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mask[idx] = (gray[idx] >= threshold) ? 255 : 0;
    }
}

__global__ void binaryColorStatsKernel(
    const uchar* originalBgr,
    const uchar* mask,
    int size,
    unsigned long long* sums,
    unsigned int* counts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    const int bucket = (mask[idx] > 0) ? 1 : 0;
    const int pixelBase = idx * 3;

    atomicAdd(&sums[bucket * 3 + 0], static_cast<unsigned long long>(originalBgr[pixelBase + 0]));
    atomicAdd(&sums[bucket * 3 + 1], static_cast<unsigned long long>(originalBgr[pixelBase + 1]));
    atomicAdd(&sums[bucket * 3 + 2], static_cast<unsigned long long>(originalBgr[pixelBase + 2]));
    atomicAdd(&counts[bucket], 1u);
}

__global__ void applyBinaryPaletteKernel(const uchar* mask, uchar* outBgr, int size, const uchar* paletteBgr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    const int bucket = (mask[idx] > 0) ? 1 : 0;
    const int outBase = idx * 3;
    const int paletteBase = bucket * 3;
    outBgr[outBase + 0] = paletteBgr[paletteBase + 0];
    outBgr[outBase + 1] = paletteBgr[paletteBase + 1];
    outBgr[outBase + 2] = paletteBgr[paletteBase + 2];
}

__global__ void kmeansAssignKernel(
    const float3* data,
    int size,
    const float3* centroids,
    int k,
    int* labels,
    float3* newCentroidsSum,
    int* newCentroidsCount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float3 pixel = data[idx];
        int best = 0;
        float minDist = 1e30f;

        for (int c = 0; c < k; ++c) {
            const float dist = squaredDistance(pixel, centroids[c]);
            if (dist < minDist) {
                minDist = dist;
                best = c;
            }
        }

        labels[idx] = best;
        atomicAdd(&newCentroidsSum[best].x, pixel.x);
        atomicAdd(&newCentroidsSum[best].y, pixel.y);
        atomicAdd(&newCentroidsSum[best].z, pixel.z);
        atomicAdd(&newCentroidsCount[best], 1);
    }
}

__global__ void kmeansUpdateKernel(
    float3* centroids,
    const float3* newCentroidsSum,
    const int* newCentroidsCount,
    int k,
    int* changed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        const int count = newCentroidsCount[idx];
        if (count > 0) {
            const float3 oldCenter = centroids[idx];
            const float3 newCenter = make_float3(
                newCentroidsSum[idx].x / static_cast<float>(count),
                newCentroidsSum[idx].y / static_cast<float>(count),
                newCentroidsSum[idx].z / static_cast<float>(count)
            );

            if (fabsf(oldCenter.x - newCenter.x) > 0.1f ||
                fabsf(oldCenter.y - newCenter.y) > 0.1f ||
                fabsf(oldCenter.z - newCenter.z) > 0.1f) {
                *changed = 1;
            }

            centroids[idx] = newCenter;
        }
    }
}

__global__ void kmeansApplyKernel(const float3* centroids, const int* labels, int size, uchar* outBgr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float3 center = centroids[labels[idx]];
        const int outBase = idx * 3;
        outBgr[outBase + 0] = static_cast<uchar>(fminf(fmaxf(center.x, 0.0f), 255.0f));
        outBgr[outBase + 1] = static_cast<uchar>(fminf(fmaxf(center.y, 0.0f), 255.0f));
        outBgr[outBase + 2] = static_cast<uchar>(fminf(fmaxf(center.z, 0.0f), 255.0f));
    }
}

__global__ void fcmAccumulateCentroidKernel(
    const float3* data,
    const float* membership,
    int size,
    int k,
    float m,
    float* centroidSums,
    float* centroidWeights) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    const float3 pixel = data[idx];
    const int membershipBase = idx * k;

    for (int cluster = 0; cluster < k; ++cluster) {
        const float u = membership[membershipBase + cluster];
        const float w = powf(u, m);
        atomicAdd(&centroidSums[cluster * 3 + 0], w * pixel.x);
        atomicAdd(&centroidSums[cluster * 3 + 1], w * pixel.y);
        atomicAdd(&centroidSums[cluster * 3 + 2], w * pixel.z);
        atomicAdd(&centroidWeights[cluster], w);
    }
}

__global__ void fcmFinalizeCentroidKernel(
    const float* centroidSums,
    const float* centroidWeights,
    int k,
    float3* centroids) {
    const int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= k) {
        return;
    }

    const float weight = centroidWeights[cluster];
    if (weight > 0.0f) {
        centroids[cluster] = make_float3(
            centroidSums[cluster * 3 + 0] / weight,
            centroidSums[cluster * 3 + 1] / weight,
            centroidSums[cluster * 3 + 2] / weight
        );
    }
}

__global__ void fcmUpdateMembershipKernel(
    const float3* data,
    const float3* centroids,
    const float* oldMembership,
    float* newMembership,
    int size,
    int k,
    float membershipExponent,
    float* maxDiff) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    float distances[kMaxFcmClustersForCuda];
    int zeroDistanceCount = 0;
    const float3 pixel = data[idx];
    const int membershipBase = idx * k;

    for (int cluster = 0; cluster < k; ++cluster) {
        distances[cluster] = squaredDistance(pixel, centroids[cluster]);
        if (distances[cluster] <= kDistanceEpsilon) {
            zeroDistanceCount++;
        }
    }

    if (zeroDistanceCount > 0) {
        const float sharedMembership = 1.0f / static_cast<float>(zeroDistanceCount);
        for (int cluster = 0; cluster < k; ++cluster) {
            const float updated = (distances[cluster] <= kDistanceEpsilon) ? sharedMembership : 0.0f;
            newMembership[membershipBase + cluster] = updated;
            atomicMaxFloat(maxDiff, fabsf(updated - oldMembership[membershipBase + cluster]));
        }
        return;
    }

    for (int cluster = 0; cluster < k; ++cluster) {
        float denominator = 0.0f;
        const float distanceToCluster = distances[cluster];
        for (int other = 0; other < k; ++other) {
            denominator += powf(distanceToCluster / distances[other], membershipExponent);
        }

        const float updated = 1.0f / denominator;
        newMembership[membershipBase + cluster] = updated;
        atomicMaxFloat(maxDiff, fabsf(updated - oldMembership[membershipBase + cluster]));
    }
}

__global__ void fcmApplyKernel(const float* membership, const float3* centroids, int size, int k, uchar* outBgr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) {
        return;
    }

    const int membershipBase = idx * k;
    int bestCluster = 0;
    float bestMembership = membership[membershipBase + 0];
    for (int cluster = 1; cluster < k; ++cluster) {
        const float candidate = membership[membershipBase + cluster];
        if (candidate > bestMembership) {
            bestMembership = candidate;
            bestCluster = cluster;
        }
    }

    const float3 center = centroids[bestCluster];
    const int outBase = idx * 3;
    outBgr[outBase + 0] = static_cast<uchar>(fminf(fmaxf(center.x, 0.0f), 255.0f));
    outBgr[outBase + 1] = static_cast<uchar>(fminf(fmaxf(center.y, 0.0f), 255.0f));
    outBgr[outBase + 2] = static_cast<uchar>(fminf(fmaxf(center.z, 0.0f), 255.0f));
}

__global__ void meanShiftKernel(const float3* in, float3* out, int width, int height, float hs, float hr) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width) {
        const int idx = y * width + x;
        const float3 current = in[idx];
        float3 sumColor = make_float3(0.0f, 0.0f, 0.0f);
        float weight = 0.0f;

        const int minY = max(0, y - static_cast<int>(hs));
        const int maxY = min(height - 1, y + static_cast<int>(hs));
        const int minX = max(0, x - static_cast<int>(hs));
        const int maxX = min(width - 1, x + static_cast<int>(hs));

        for (int yy = minY; yy <= maxY; ++yy) {
            for (int xx = minX; xx <= maxX; ++xx) {
                const float3 neighbor = in[yy * width + xx];
                const float spatialDistance = static_cast<float>((y - yy) * (y - yy) + (x - xx) * (x - xx));
                const float colorDistance = squaredDistance(current, neighbor);

                if (spatialDistance <= hs * hs && colorDistance <= hr * hr) {
                    sumColor.x += neighbor.x;
                    sumColor.y += neighbor.y;
                    sumColor.z += neighbor.z;
                    weight += 1.0f;
                }
            }
        }

        if (weight > 0.0f) {
            out[idx] = make_float3(sumColor.x / weight, sumColor.y / weight, sumColor.z / weight);
        } else {
            out[idx] = current;
        }
    }
}

__global__ void float3ToBgrKernel(const float3* in, uchar* outBgr, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float3 pixel = in[idx];
        const int outBase = idx * 3;
        outBgr[outBase + 0] = static_cast<uchar>(fminf(fmaxf(pixel.x, 0.0f), 255.0f));
        outBgr[outBase + 1] = static_cast<uchar>(fminf(fmaxf(pixel.y, 0.0f), 255.0f));
        outBgr[outBase + 2] = static_cast<uchar>(fminf(fmaxf(pixel.z, 0.0f), 255.0f));
    }
}

__device__ uchar quantizedChannelDevice(float value, float step) {
    const float safeStep = fmaxf(step, 1.0f);
    const float quantized = roundf(value / safeStep) * safeStep;
    return static_cast<uchar>(fminf(fmaxf(quantized, 0.0f), 255.0f));
}

__global__ void quantizedFloat3ToBgrKernel(const float3* in, uchar* outBgr, int size, float colorBandwidth) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        const float3 pixel = in[idx];
        const int outBase = idx * 3;

        outBgr[outBase + 0] = quantizedChannelDevice(pixel.x, colorBandwidth);
        outBgr[outBase + 1] = quantizedChannelDevice(pixel.y, colorBandwidth);
        outBgr[outBase + 2] = quantizedChannelDevice(pixel.z, colorBandwidth);
    }
}
} // namespace

std::pair<cv::Mat, double> OtsuGPU::execute(const cv::Mat& inputImage) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    uchar* dGray = nullptr;
    uchar* dMask = nullptr;
    uchar* dOriginalBgr = nullptr;
    uchar* dColorized = nullptr;
    uchar* dPalette = nullptr;
    int* dHistogram = nullptr;
    unsigned long long* dColorSums = nullptr;
    unsigned int* dColorCounts = nullptr;

    try {
        const cv::Mat originalBgrInput = ensureBgr8(inputImage);
        const cv::Mat grayInput = makeGray8(inputImage);
        cv::Mat originalBgr = originalBgrInput.isContinuous() ? originalBgrInput : originalBgrInput.clone();
        cv::Mat gray = grayInput.isContinuous() ? grayInput : grayInput.clone();

        const int size = gray.rows * gray.cols;
        if (size <= 0) {
            throw std::runtime_error("Input image is empty.");
        }

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaMalloc(&dGray, static_cast<size_t>(size) * sizeof(uchar)));
        CUDA_CHECK(cudaMalloc(&dMask, static_cast<size_t>(size) * sizeof(uchar)));
        CUDA_CHECK(cudaMalloc(&dOriginalBgr, static_cast<size_t>(size) * 3u * sizeof(uchar)));
        CUDA_CHECK(cudaMalloc(&dColorized, static_cast<size_t>(size) * 3u * sizeof(uchar)));
        CUDA_CHECK(cudaMalloc(&dPalette, 6u * sizeof(uchar)));
        CUDA_CHECK(cudaMalloc(&dHistogram, 256u * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dColorSums, 6u * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&dColorCounts, 2u * sizeof(unsigned int)));

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(dGray, gray.ptr<uchar>(0), static_cast<size_t>(size) * sizeof(uchar), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dOriginalBgr, originalBgr.ptr<uchar>(0), static_cast<size_t>(size) * 3u * sizeof(uchar), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dHistogram, 0, 256u * sizeof(int)));

        const int blockSize = 256;
        const int gridSize = (size + blockSize - 1) / blockSize;

        histoKernel<<<gridSize, blockSize>>>(dGray, size, dHistogram);
        CUDA_CHECK(cudaGetLastError());

        int histogram[256] = { 0 };
        CUDA_CHECK(cudaMemcpy(histogram, dHistogram, 256u * sizeof(int), cudaMemcpyDeviceToHost));
        const int threshold = computeOtsuThreshold(histogram, size);

        thresholdKernel<<<gridSize, blockSize>>>(dGray, dMask, size, threshold);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(dColorSums, 0, 6u * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(dColorCounts, 0, 2u * sizeof(unsigned int)));
        binaryColorStatsKernel<<<gridSize, blockSize>>>(dOriginalBgr, dMask, size, dColorSums, dColorCounts);
        CUDA_CHECK(cudaGetLastError());

        unsigned long long hostSums[6] = { 0, 0, 0, 0, 0, 0 };
        unsigned int hostCounts[2] = { 0, 0 };
        CUDA_CHECK(cudaMemcpy(hostSums, dColorSums, 6u * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hostCounts, dColorCounts, 2u * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        const cv::Scalar overallMean = cv::mean(originalBgr);
        const uchar fallback[3] = {
            static_cast<uchar>(overallMean[0]),
            static_cast<uchar>(overallMean[1]),
            static_cast<uchar>(overallMean[2])
        };

        uchar palette[6];
        for (int bucket = 0; bucket < 2; ++bucket) {
            const unsigned int count = hostCounts[bucket];
            for (int channel = 0; channel < 3; ++channel) {
                if (count > 0) {
                    palette[bucket * 3 + channel] = static_cast<uchar>(hostSums[bucket * 3 + channel] / count);
                } else {
                    palette[bucket * 3 + channel] = fallback[channel];
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(dPalette, palette, 6u * sizeof(uchar), cudaMemcpyHostToDevice));
        applyBinaryPaletteKernel<<<gridSize, blockSize>>>(dMask, dColorized, size, dPalette);
        CUDA_CHECK(cudaGetLastError());

        cv::Mat result(gray.rows, gray.cols, CV_8UC3);
        CUDA_CHECK(cudaMemcpy(result.ptr<uchar>(0), dColorized, static_cast<size_t>(size) * 3u * sizeof(uchar), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        CUDA_CHECK(cudaFree(dGray));
        CUDA_CHECK(cudaFree(dMask));
        CUDA_CHECK(cudaFree(dOriginalBgr));
        CUDA_CHECK(cudaFree(dColorized));
        CUDA_CHECK(cudaFree(dPalette));
        CUDA_CHECK(cudaFree(dHistogram));
        CUDA_CHECK(cudaFree(dColorSums));
        CUDA_CHECK(cudaFree(dColorCounts));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return std::make_pair(result, static_cast<double>(milliseconds));
    } catch (...) {
        if (dGray) cudaFree(dGray);
        if (dMask) cudaFree(dMask);
        if (dOriginalBgr) cudaFree(dOriginalBgr);
        if (dColorized) cudaFree(dColorized);
        if (dPalette) cudaFree(dPalette);
        if (dHistogram) cudaFree(dHistogram);
        if (dColorSums) cudaFree(dColorSums);
        if (dColorCounts) cudaFree(dColorCounts);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        throw;
    }
}

KMeansGPU::KMeansGPU(int clusters, int maxIter) : k(clusters), maxIterations(maxIter) {}

std::pair<cv::Mat, double> KMeansGPU::execute(const cv::Mat& inputImage) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float3* dData = nullptr;
    float3* dCentroids = nullptr;
    float3* dNewCentroidsSum = nullptr;
    int* dNewCentroidsCount = nullptr;
    int* dLabels = nullptr;
    int* dChanged = nullptr;
    uchar* dOutBgr = nullptr;

    try {
        const cv::Mat bgrInput = ensureBgr8(inputImage);
        cv::Mat data32;
        bgrInput.convertTo(data32, CV_32FC3);
        if (!data32.isContinuous()) {
            data32 = data32.clone();
        }

        const int rows = data32.rows;
        const int cols = data32.cols;
        const int size = rows * cols;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaMalloc(&dData, static_cast<size_t>(size) * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dCentroids, static_cast<size_t>(k) * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dNewCentroidsSum, static_cast<size_t>(k) * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dNewCentroidsCount, static_cast<size_t>(k) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dLabels, static_cast<size_t>(size) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dChanged, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&dOutBgr, static_cast<size_t>(size) * 3u * sizeof(uchar)));

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(dData, data32.ptr<cv::Vec3f>(0), static_cast<size_t>(size) * sizeof(float3), cudaMemcpyHostToDevice));

        std::vector<float3> hostCentroids(static_cast<size_t>(k));
        const float3* hostData = reinterpret_cast<const float3*>(data32.ptr<cv::Vec3f>(0));
        for (int i = 0; i < k; ++i) {
            hostCentroids[static_cast<size_t>(i)] = hostData[std::rand() % size];
        }
        CUDA_CHECK(cudaMemcpy(dCentroids, hostCentroids.data(), static_cast<size_t>(k) * sizeof(float3), cudaMemcpyHostToDevice));

        const int blockSize = 256;
        const int gridSize = (size + blockSize - 1) / blockSize;
        const int centroidGridSize = (k + blockSize - 1) / blockSize;

        int changed = 1;
        int iteration = 0;
        while (changed && iteration < maxIterations) {
            changed = 0;
            CUDA_CHECK(cudaMemcpy(dChanged, &changed, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(dNewCentroidsSum, 0, static_cast<size_t>(k) * sizeof(float3)));
            CUDA_CHECK(cudaMemset(dNewCentroidsCount, 0, static_cast<size_t>(k) * sizeof(int)));

            kmeansAssignKernel<<<gridSize, blockSize>>>(dData, size, dCentroids, k, dLabels, dNewCentroidsSum, dNewCentroidsCount);
            CUDA_CHECK(cudaGetLastError());
            kmeansUpdateKernel<<<centroidGridSize, blockSize>>>(dCentroids, dNewCentroidsSum, dNewCentroidsCount, k, dChanged);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&changed, dChanged, sizeof(int), cudaMemcpyDeviceToHost));
            iteration++;
        }

        kmeansApplyKernel<<<gridSize, blockSize>>>(dCentroids, dLabels, size, dOutBgr);
        CUDA_CHECK(cudaGetLastError());

        cv::Mat result(rows, cols, CV_8UC3);
        CUDA_CHECK(cudaMemcpy(result.ptr<uchar>(0), dOutBgr, static_cast<size_t>(size) * 3u * sizeof(uchar), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        CUDA_CHECK(cudaFree(dData));
        CUDA_CHECK(cudaFree(dCentroids));
        CUDA_CHECK(cudaFree(dNewCentroidsSum));
        CUDA_CHECK(cudaFree(dNewCentroidsCount));
        CUDA_CHECK(cudaFree(dLabels));
        CUDA_CHECK(cudaFree(dChanged));
        CUDA_CHECK(cudaFree(dOutBgr));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return std::make_pair(result, static_cast<double>(milliseconds));
    } catch (...) {
        if (dData) cudaFree(dData);
        if (dCentroids) cudaFree(dCentroids);
        if (dNewCentroidsSum) cudaFree(dNewCentroidsSum);
        if (dNewCentroidsCount) cudaFree(dNewCentroidsCount);
        if (dLabels) cudaFree(dLabels);
        if (dChanged) cudaFree(dChanged);
        if (dOutBgr) cudaFree(dOutBgr);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        throw;
    }
}

FCMGPU::FCMGPU(int clusters, int maxIter, float fuzziness, float epsilon)
    : k(clusters), maxIterations(maxIter), m(fuzziness), eps(epsilon) {}

std::pair<cv::Mat, double> FCMGPU::execute(const cv::Mat& inputImage) {
    if (k < 2) {
        throw std::invalid_argument("FCM requires at least 2 clusters.");
    }
    if (k > kMaxFcmClustersForCuda) {
        throw std::invalid_argument("FCM GPU implementation supports at most 32 clusters.");
    }
    if (m <= 1.0f) {
        throw std::invalid_argument("FCM fuzziness parameter m must be greater than 1.");
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float3* dData = nullptr;
    float3* dCentroids = nullptr;
    float* dMembership = nullptr;
    float* dNextMembership = nullptr;
    float* dCentroidSums = nullptr;
    float* dCentroidWeights = nullptr;
    float* dMaxDiff = nullptr;
    uchar* dOutBgr = nullptr;

    try {
        const cv::Mat bgrInput = ensureBgr8(inputImage);
        cv::Mat data32;
        bgrInput.convertTo(data32, CV_32FC3);
        if (!data32.isContinuous()) {
            data32 = data32.clone();
        }

        const int rows = data32.rows;
        const int cols = data32.cols;
        const int size = rows * cols;
        const size_t membershipCount = static_cast<size_t>(size) * static_cast<size_t>(k);

        std::vector<float> hostMembership(membershipCount, 0.0f);
        std::mt19937 rng(1337u);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        for (int i = 0; i < size; ++i) {
            float rowSum = 0.0f;
            for (int cluster = 0; cluster < k; ++cluster) {
                const float value = distribution(rng);
                hostMembership[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(cluster)] = value;
                rowSum += value;
            }
            for (int cluster = 0; cluster < k; ++cluster) {
                hostMembership[static_cast<size_t>(i) * static_cast<size_t>(k) + static_cast<size_t>(cluster)] /= rowSum;
            }
        }

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaMalloc(&dData, static_cast<size_t>(size) * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dCentroids, static_cast<size_t>(k) * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dMembership, membershipCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dNextMembership, membershipCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCentroidSums, static_cast<size_t>(k) * 3u * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dCentroidWeights, static_cast<size_t>(k) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dMaxDiff, sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOutBgr, static_cast<size_t>(size) * 3u * sizeof(uchar)));

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(dData, data32.ptr<cv::Vec3f>(0), static_cast<size_t>(size) * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dMembership, hostMembership.data(), membershipCount * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dCentroids, 0, static_cast<size_t>(k) * sizeof(float3)));

        const int blockSize = 256;
        const int gridSize = (size + blockSize - 1) / blockSize;
        const int centroidGridSize = (k + blockSize - 1) / blockSize;
        const float membershipExponent = 1.0f / (m - 1.0f);

        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            CUDA_CHECK(cudaMemset(dCentroidSums, 0, static_cast<size_t>(k) * 3u * sizeof(float)));
            CUDA_CHECK(cudaMemset(dCentroidWeights, 0, static_cast<size_t>(k) * sizeof(float)));
            CUDA_CHECK(cudaMemset(dMaxDiff, 0, sizeof(float)));

            fcmAccumulateCentroidKernel<<<gridSize, blockSize>>>(dData, dMembership, size, k, m, dCentroidSums, dCentroidWeights);
            CUDA_CHECK(cudaGetLastError());
            fcmFinalizeCentroidKernel<<<centroidGridSize, blockSize>>>(dCentroidSums, dCentroidWeights, k, dCentroids);
            CUDA_CHECK(cudaGetLastError());
            fcmUpdateMembershipKernel<<<gridSize, blockSize>>>(
                dData,
                dCentroids,
                dMembership,
                dNextMembership,
                size,
                k,
                membershipExponent,
                dMaxDiff);
            CUDA_CHECK(cudaGetLastError());

            float hostMaxDiff = 0.0f;
            CUDA_CHECK(cudaMemcpy(&hostMaxDiff, dMaxDiff, sizeof(float), cudaMemcpyDeviceToHost));
            std::swap(dMembership, dNextMembership);
            if (hostMaxDiff < eps) {
                break;
            }
        }

        fcmApplyKernel<<<gridSize, blockSize>>>(dMembership, dCentroids, size, k, dOutBgr);
        CUDA_CHECK(cudaGetLastError());

        cv::Mat result(rows, cols, CV_8UC3);
        CUDA_CHECK(cudaMemcpy(result.ptr<uchar>(0), dOutBgr, static_cast<size_t>(size) * 3u * sizeof(uchar), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        CUDA_CHECK(cudaFree(dData));
        CUDA_CHECK(cudaFree(dCentroids));
        CUDA_CHECK(cudaFree(dMembership));
        CUDA_CHECK(cudaFree(dNextMembership));
        CUDA_CHECK(cudaFree(dCentroidSums));
        CUDA_CHECK(cudaFree(dCentroidWeights));
        CUDA_CHECK(cudaFree(dMaxDiff));
        CUDA_CHECK(cudaFree(dOutBgr));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return std::make_pair(result, static_cast<double>(milliseconds));
    } catch (...) {
        if (dData) cudaFree(dData);
        if (dCentroids) cudaFree(dCentroids);
        if (dMembership) cudaFree(dMembership);
        if (dNextMembership) cudaFree(dNextMembership);
        if (dCentroidSums) cudaFree(dCentroidSums);
        if (dCentroidWeights) cudaFree(dCentroidWeights);
        if (dMaxDiff) cudaFree(dMaxDiff);
        if (dOutBgr) cudaFree(dOutBgr);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        throw;
    }
}

MeanShiftGPU::MeanShiftGPU(float spatialBandwidth, float colorBandwidth, int maxIter)
    : hs(spatialBandwidth), hr(colorBandwidth), maxIterations(maxIter) {}

std::pair<cv::Mat, double> MeanShiftGPU::execute(const cv::Mat& inputImage) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    float3* dIn = nullptr;
    float3* dOut = nullptr;
    uchar* dOutBgr = nullptr;

    try {
        const cv::Mat bgrInput = ensureBgr8(inputImage);
        cv::Mat data32;
        bgrInput.convertTo(data32, CV_32FC3);
        if (!data32.isContinuous()) {
            data32 = data32.clone();
        }

        const int width = data32.cols;
        const int height = data32.rows;
        const int size = width * height;

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaMalloc(&dIn, static_cast<size_t>(size) * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dOut, static_cast<size_t>(size) * sizeof(float3)));
        CUDA_CHECK(cudaMalloc(&dOutBgr, static_cast<size_t>(size) * 3u * sizeof(uchar)));

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(dIn, data32.ptr<cv::Vec3f>(0), static_cast<size_t>(size) * sizeof(float3), cudaMemcpyHostToDevice));

        const dim3 blockSize(16, 16);
        const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        const int colorBlockSize = 256;
        const int colorGridSize = (size + colorBlockSize - 1) / colorBlockSize;

        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            meanShiftKernel<<<gridSize, blockSize>>>(dIn, dOut, width, height, hs, hr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(dIn, dOut, static_cast<size_t>(size) * sizeof(float3), cudaMemcpyDeviceToDevice));
        }

        quantizedFloat3ToBgrKernel<<<colorGridSize, colorBlockSize>>>(dOut, dOutBgr, size, hr);
        CUDA_CHECK(cudaGetLastError());

        cv::Mat result(height, width, CV_8UC3);
        CUDA_CHECK(cudaMemcpy(result.ptr<uchar>(0), dOutBgr, static_cast<size_t>(size) * 3u * sizeof(uchar), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        CUDA_CHECK(cudaFree(dIn));
        CUDA_CHECK(cudaFree(dOut));
        CUDA_CHECK(cudaFree(dOutBgr));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        return std::make_pair(result, static_cast<double>(milliseconds));
    } catch (...) {
        if (dIn) cudaFree(dIn);
        if (dOut) cudaFree(dOut);
        if (dOutBgr) cudaFree(dOutBgr);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        throw;
    }
}

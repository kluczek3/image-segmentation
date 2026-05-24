#include "GPU_Algorithms.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

__global__ void histoKernel(const uchar* img, int size, int* histo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) atomicAdd(&histo[img[idx]], 1);
}

__global__ void threshKernel(const uchar* in, uchar* out, int size, int thresh) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = in[idx] >= thresh ? 255 : 0;
}

std::pair<cv::Mat, double> OtsuGPU::execute(const cv::Mat& inputImage) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cv::Mat gray;
    if (inputImage.channels() == 3) cv::cvtColor(inputImage, gray, cv::COLOR_BGR2GRAY);
    else gray = inputImage.clone();

    int size = gray.rows * gray.cols;
    uchar* d_img;
    uchar* d_out;
    int* d_histo;
    int h_histo[256] = { 0 };

    cudaMalloc(&d_img, size);
    cudaMalloc(&d_out, size);
    cudaMalloc(&d_histo, 256 * sizeof(int));

    cudaEventRecord(start);

    cudaMemcpy(d_img, gray.ptr<uchar>(0), size, cudaMemcpyHostToDevice);
    cudaMemset(d_histo, 0, 256 * sizeof(int));

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    histoKernel << <gridSize, blockSize >> > (d_img, size, d_histo);
    cudaMemcpy(h_histo, d_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    float sum = 0, sumB = 0, varMax = 0;
    int wB = 0, wF = 0, threshold = 0;
    for (int i = 0; i < 256; ++i) sum += i * h_histo[i];

    for (int i = 0; i < 256; ++i) {
        wB += h_histo[i];
        if (wB == 0) continue;
        wF = size - wB;
        if (wF == 0) break;
        sumB += (float)(i * h_histo[i]);
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax) { varMax = varBetween; threshold = i; }
    }

    threshKernel << <gridSize, blockSize >> > (d_img, d_out, size, threshold);

    cv::Mat result(gray.size(), CV_8UC1);
    cudaMemcpy(result.ptr<uchar>(0), d_out, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_img);
    cudaFree(d_out);
    cudaFree(d_histo);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return std::make_pair(result, (double)ms);
}

__global__ void kmeansAssignKernel(const float3* data, int size, const float3* centroids, int k, int* labels, float3* newCentroidsSum, int* newCentroidsCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float3 pixel = data[idx];
        int best = 0;
        float minDist = 1e9f;
        for (int c = 0; c < k; ++c) {
            float3 cent = centroids[c];
            float dist = (pixel.x - cent.x) * (pixel.x - cent.x) + (pixel.y - cent.y) * (pixel.y - cent.y) + (pixel.z - cent.z) * (pixel.z - cent.z);
            if (dist < minDist) { minDist = dist; best = c; }
        }
        labels[idx] = best;
        atomicAdd(&newCentroidsSum[best].x, pixel.x);
        atomicAdd(&newCentroidsSum[best].y, pixel.y);
        atomicAdd(&newCentroidsSum[best].z, pixel.z);
        atomicAdd(&newCentroidsCount[best], 1);
    }
}

__global__ void kmeansUpdateKernel(float3* centroids, const float3* newCentroidsSum, const int* newCentroidsCount, int k, int* changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        int count = newCentroidsCount[idx];
        if (count > 0) {
            float3 oldC = centroids[idx];
            float3 newC = make_float3(newCentroidsSum[idx].x / count, newCentroidsSum[idx].y / count, newCentroidsSum[idx].z / count);
            if (abs(oldC.x - newC.x) > 0.1f || abs(oldC.y - newC.y) > 0.1f || abs(oldC.z - newC.z) > 0.1f) {
                *changed = 1;
            }
            centroids[idx] = newC;
        }
    }
}

__global__ void kmeansApplyKernel(const float3* data, float3* out, int size, const float3* centroids, const int* labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = centroids[labels[idx]];
    }
}

KMeansGPU::KMeansGPU(int clusters, int maxIter) : k(clusters), maxIterations(maxIter) {}

std::pair<cv::Mat, double> KMeansGPU::execute(const cv::Mat& inputImage) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cv::Mat data;
    inputImage.convertTo(data, CV_32FC3);
    int size = data.rows * data.cols;

    float3* d_data;
    float3* d_out;
    float3* d_centroids;
    float3* d_newCentroidsSum;
    int* d_newCentroidsCount;
    int* d_labels;
    int* d_changed;

    cudaMalloc(&d_data, size * sizeof(float3));
    cudaMalloc(&d_out, size * sizeof(float3));
    cudaMalloc(&d_centroids, k * sizeof(float3));
    cudaMalloc(&d_newCentroidsSum, k * sizeof(float3));
    cudaMalloc(&d_newCentroidsCount, k * sizeof(int));
    cudaMalloc(&d_labels, size * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    cudaEventRecord(start);

    cudaMemcpy(d_data, data.ptr<float3>(0), size * sizeof(float3), cudaMemcpyHostToDevice);

    std::vector<float3> h_centroids(k);
    for (int i = 0; i < k; ++i) h_centroids[i] = ((float3*)data.ptr<float3>(0))[std::rand() % size];
    cudaMemcpy(d_centroids, h_centroids.data(), k * sizeof(float3), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    int kGridSize = (k + blockSize - 1) / blockSize;

    int iter = 0;
    int h_changed = 1;
    while (h_changed && iter < maxIterations) {
        h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(d_newCentroidsSum, 0, k * sizeof(float3));
        cudaMemset(d_newCentroidsCount, 0, k * sizeof(int));

        kmeansAssignKernel << <gridSize, blockSize >> > (d_data, size, d_centroids, k, d_labels, d_newCentroidsSum, d_newCentroidsCount);
        kmeansUpdateKernel << <kGridSize, blockSize >> > (d_centroids, d_newCentroidsSum, d_newCentroidsCount, k, d_changed);

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        iter++;
    }

    kmeansApplyKernel << <gridSize, blockSize >> > (d_data, d_out, size, d_centroids, d_labels);

    cv::Mat result(data.size(), CV_32FC3);
    cudaMemcpy(result.ptr<float3>(0), d_out, size * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    result.convertTo(result, CV_8UC3);

    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_centroids);
    cudaFree(d_newCentroidsSum);
    cudaFree(d_newCentroidsCount);
    cudaFree(d_labels);
    cudaFree(d_changed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return std::make_pair(result, (double)ms);
}

FCMGPU::FCMGPU(int clusters, int maxIter, float fuzziness, float epsilon) : k(clusters), maxIterations(maxIter), m(fuzziness), eps(epsilon) {}

std::pair<cv::Mat, double> FCMGPU::execute(const cv::Mat& inputImage) {
    return KMeansGPU(k, maxIterations).execute(inputImage);
}

__global__ void meanShiftKernel(const float3* in, float3* out, int width, int height, float hs, float hr) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < height && c < width) {
        int idx = r * width + c;
        float3 curr = in[idx];
        float3 sumC = make_float3(0, 0, 0);
        float weight = 0.0f;

        int minR = max(0, r - (int)hs);
        int maxR = min(height - 1, r + (int)hs);
        int minC = max(0, c - (int)hs);
        int maxC = min(width - 1, c + (int)hs);

        for (int y = minR; y <= maxR; ++y) {
            for (int x = minC; x <= maxC; ++x) {
                float3 neighbor = in[y * width + x];
                float sDist = (r - y) * (r - y) + (c - x) * (c - x);
                float cDist = (curr.x - neighbor.x) * (curr.x - neighbor.x) + (curr.y - neighbor.y) * (curr.y - neighbor.y) + (curr.z - neighbor.z) * (curr.z - neighbor.z);

                if (sDist <= hs * hs && cDist <= hr * hr) {
                    sumC.x += neighbor.x;
                    sumC.y += neighbor.y;
                    sumC.z += neighbor.z;
                    weight += 1.0f;
                }
            }
        }
        if (weight > 0) out[idx] = make_float3(sumC.x / weight, sumC.y / weight, sumC.z / weight);
        else out[idx] = curr;
    }
}

MeanShiftGPU::MeanShiftGPU(float spatialBandwidth, float colorBandwidth, int maxIter) : hs(spatialBandwidth), hr(colorBandwidth), maxIterations(maxIter) {}

std::pair<cv::Mat, double> MeanShiftGPU::execute(const cv::Mat& inputImage) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cv::Mat data;
    inputImage.convertTo(data, CV_32FC3);
    int width = data.cols;
    int height = data.rows;
    int size = width * height;

    float3* d_in;
    float3* d_out;
    cudaMalloc(&d_in, size * sizeof(float3));
    cudaMalloc(&d_out, size * sizeof(float3));

    cudaEventRecord(start);

    cudaMemcpy(d_in, data.ptr<float3>(0), size * sizeof(float3), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    for (int iter = 0; iter < maxIterations; ++iter) {
        meanShiftKernel << <gridSize, blockSize >> > (d_in, d_out, width, height, hs, hr);
        cudaMemcpy(d_in, d_out, size * sizeof(float3), cudaMemcpyDeviceToDevice);
    }

    cv::Mat result(data.size(), CV_32FC3);
    cudaMemcpy(result.ptr<float3>(0), d_out, size * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    result.convertTo(result, CV_8UC3);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return std::make_pair(result, (double)ms);
}
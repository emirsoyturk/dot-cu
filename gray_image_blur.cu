#include <iostream>
#include <opencv2/opencv.hpp>

#define BLUR_SIZE 10
#define BLOCK_SIZE 16

__global__
void gray_image_blur(unsigned char *input, unsigned char *output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) {
        return;
    }

    int sum = 0;
    int count = 0;

    for (int i = -BLUR_SIZE; i <= BLUR_SIZE; i++) {
        for (int j = -BLUR_SIZE; j <= BLUR_SIZE; j++) {
            int r = row + i;
            int c = col + j;

            if (r < 0 || r >= height || c < 0 || c >= width) {
                continue;
            }

            sum += input[r * width + c];
            count++;
        }
    }

    output[row * width + col] = sum / count;
}

int main(int argc, char **argv) {
    std::string imagePath = argv[1];
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    int width = image.cols;
    int height = image.rows;
    int imageSize = width * height * sizeof(unsigned char);

    unsigned char *d_input, *d_output;
    cudaMalloc((void **)&d_input, imageSize);
    cudaMalloc((void **)&d_output, imageSize);

    cudaMemcpy(d_input, image.data, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
      (width + BLOCK_SIZE - 1) / BLOCK_SIZE,
      (height + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    gray_image_blur<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    cudaDeviceSynchronize();

    unsigned char *output = new unsigned char[width * height];
    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cv::Mat blurredImage(height, width, CV_8UC1, output);
    cv::imwrite("blurred_image.jpg", blurredImage);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] output;

    return 0;
}

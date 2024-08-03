#define WIDTH 1920
#define HEIGHT 1920
#define BLOCK_SIZE 32
#define BACKGROUND_COLOR_X 160
#define BACKGROUND_COLOR_Y 10
#define BACKGROUND_COLOR_Z 40

#include <iostream>
#include <cuda_runtime.h>
#include "raytracer.h"
#include <opencv2/opencv.hpp>

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ bool intersect_sphere(const Sphere& sphere, const Ray& ray, float& t) {
    float3 oc = make_float3(ray.origin.x - sphere.center.x, ray.origin.y - sphere.center.y, ray.origin.z - sphere.center.z);
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return false;
    } else {
        t = (-b - sqrt(discriminant)) / (2.0f * a);
        return true;
    }
}

__device__ bool intersect_cube(Cube& cube, Ray& ray, float& t) {
    float3 min = cube.center - make_float3(cube.size.x / 2.0f, cube.size.y / 2.0f, cube.size.z / 2.0f);
    float3 max = cube.center + make_float3(cube.size.x / 2.0f, cube.size.y / 2.0f, cube.size.z / 2.0f);

    float tmin = (min.x - ray.origin.x) / ray.direction.x;
    float tmax = (max.x - ray.origin.x) / ray.direction.x;

    if (tmin > tmax) {
      swap(tmin, tmax);
    }

    float tymin = (min.y - ray.origin.y) / ray.direction.y;
    float tymax = (max.y - ray.origin.y) / ray.direction.y;

    if (tymin > tymax) {
      swap(tymin, tymax);
    }

    if ((tmin > tymax) || (tymin > tmax)) {
        return false;
    }

    if (tymin > tmin) {
        tmin = tymin;
    }

    if (tymax < tmax) {
        tmax = tymax;
    }

    float tzmin = (min.z - ray.origin.z) / ray.direction.z;
    float tzmax = (max.z - ray.origin.z) / ray.direction.z;

    if (tzmin > tzmax) {
      swap(tzmin, tzmax);
    }

    if ((tmin > tzmax) || (tzmin > tmax)) {
        return false;
    }

    if (tzmin > tmin) {
        tmin = tzmin;
    }

    if (tzmax < tmax) {
        tmax = tzmax;
    }

    t = tmin;

    if (t < 0) {
        t = tmax;
        if (t < 0) {
            return false;
        }
    }

    return true;
}

__global__ void kernel(unsigned char* d_output, Sphere* spheres, int num_spheres, Cube* cubes, int num_cubes) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) {
      return;
    }

    int index = (y * WIDTH + x) * 3;

    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 direction = make_float3(
        (x - WIDTH / 2) / (float)WIDTH,
        (y - HEIGHT / 2) / (float)HEIGHT,
        -1.0f
    );

    Ray ray = {origin, direction};

    float t;
    bool hit = false;
    float3 color = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < num_spheres; i++) {
        if (intersect_sphere(spheres[i], ray, t)) {
            color = spheres[i].color;
            hit = true;
        }
    }

    for (int i = 0; i < num_cubes; i++) {
        if (intersect_cube(cubes[i], ray, t)) {
            color = cubes[i].color;
            hit = true;
        }
    }

    if (hit) {
        d_output[index] = (unsigned char)(color.x * 255);
        d_output[index + 1] = (unsigned char)(color.y * 255);
        d_output[index + 2] = (unsigned char)(color.z * 255);
    } else {
        d_output[index] = BACKGROUND_COLOR_X;
        d_output[index + 1] = BACKGROUND_COLOR_Y;
        d_output[index + 2] = BACKGROUND_COLOR_Z;
    }
}

void raytrace() {
    std::cout << "Ray tracing started!" << std::endl;

    int imageSize = WIDTH * HEIGHT * 3;
    unsigned char *image = new unsigned char[imageSize];

    unsigned char *d_output;
    cudaMalloc(&d_output, imageSize);

    Sphere h_spheres[] = {
        {make_float3(100.0f / WIDTH, 0.0f / HEIGHT, -1.0f), 0.2f, make_float3(0.3f, 0.4f, 0.0f)},
        {make_float3(1.0f / WIDTH, 0.0f / HEIGHT, -2.0f), 0.2f, make_float3(0.6f, 0.2f, 0.1f)}
    };

    Cube h_cubes[] = {
        {make_float3(-500.0f / WIDTH, -500.0f / HEIGHT, -1.0f), make_float3(0.4f, 0.2f, 0.2f), make_float3(0.3f, 0.4f, 0.0f)},
        {make_float3(500.0f / WIDTH, 100.0f / HEIGHT, -2.0f), make_float3(0.2f, 0.2f, 0.2f), make_float3(0.1f, 0.2f, 0.8f)}
    };

    int num_spheres = sizeof(h_spheres) / sizeof(Sphere);
    int num_cubes = sizeof(h_cubes) / sizeof(Cube);

    Sphere *d_spheres;
    cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere));
    cudaMemcpy(d_spheres, h_spheres, num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice);

    Cube *d_cubes;
    cudaMalloc(&d_cubes, num_cubes * sizeof(Cube));
    cudaMemcpy(d_cubes, h_cubes, num_cubes * sizeof(Cube), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
      (WIDTH + blockSize.x - 1) / blockSize.x,
      (HEIGHT + blockSize.y - 1) / blockSize.y
    );

    kernel<<<gridSize, blockSize>>>(d_output, d_spheres, num_spheres, d_cubes, num_cubes);
    cudaDeviceSynchronize();

    cudaMemcpy(image, d_output, imageSize, cudaMemcpyDeviceToHost);

    cv::Mat output(HEIGHT, WIDTH, CV_8UC3, image);
    cv::imwrite("output.jpg", output);

    delete[] image;
    cudaFree(d_output);
    cudaFree(d_spheres);

    std::cout << "Ray tracing completed! Image saved as output.ppm" << std::endl;
}
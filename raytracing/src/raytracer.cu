#define WIDTH 1024
#define HEIGHT 1024
#define BLOCK_SIZE 16
#define BACKGROUND_COLOR_X 10
#define BACKGROUND_COLOR_Y 10
#define BACKGROUND_COLOR_Z 10

#include <iostream>
#include <cuda_runtime.h>
#include "raytracer.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <sstream>

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float3 operator*(const float3& a, const float& b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator*(const float& a, const float3& b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float3 normalize(const float3& a) {
    float len = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    return make_float3(a.x / len, a.y / len, a.z / len);
}

__device__ float3 reflect(const float3& I, const float3& N) {
    return I - 2 * dot(N, I) * N;
}

__device__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
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

__device__ float3 compute_lighting(const Ray& ray, const Sphere& sphere, const float3& hit_point, const Light* lights, int num_lights) {
    float3 diffuse_color = make_float3(0.0f, 0.0f, 0.0f);

    float3 normal = normalize(hit_point - sphere.center);
    float3 view_dir = normalize(ray.origin - hit_point);

    for (int i = 0; i < num_lights; ++i) {
        const Light& light = lights[i];
        float3 light_dir = normalize(light.position - hit_point);
        float diff = fmaxf(dot(normal, light_dir), 0.0f);
        diffuse_color = diffuse_color + (diff * light.color * sphere.color * light.intensity);
    }

    float3 color = diffuse_color;

    color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);

    return color;
}

__device__ float3 trace_ray(const Ray& ray, Sphere* spheres, int num_spheres, Light* lights, int num_lights, int MAX_DEPTH) {
    float3 color = make_float3(BACKGROUND_COLOR_X / 255.0f, BACKGROUND_COLOR_Y / 255.0f, BACKGROUND_COLOR_Z / 255.0f);
    Ray current_ray = ray;
    int depth = 0;

    float current_reflectance = 1.0f;

    while (depth < MAX_DEPTH) {
        float t;
        float3 hit_point, normal;
        int hit_index = -1;
        float closest_t = FLT_MAX;

        for (int i = 0; i < num_spheres; i++) {
            if (intersect_sphere(spheres[i], current_ray, t) && t < closest_t) {
                closest_t = t;
                hit_index = i;
            }
        }

        if (hit_index == -1) {
            color = make_float3(BACKGROUND_COLOR_X / 255.0f, BACKGROUND_COLOR_Y / 255.0f, BACKGROUND_COLOR_Z / 255.0f);
            break;
        }

        hit_point = current_ray.origin + closest_t * current_ray.direction;
        normal = normalize(hit_point - spheres[hit_index].center);
        color = compute_lighting(current_ray, spheres[hit_index], hit_point, lights, num_lights);

        if (spheres[hit_index].reflectivity > 0 && depth < MAX_DEPTH) {
            float3 reflection_dir = reflect(current_ray.direction, normal);

            Ray reflected_ray;
            reflected_ray.origin = hit_point + 0.001f * normal;
            reflected_ray.direction = -reflection_dir;

            float3 reflection_color = trace_ray(reflected_ray, spheres, num_spheres, lights, num_lights, MAX_DEPTH - 1);
            color = (1.0f - spheres[hit_index].reflectivity * current_reflectance) * color + spheres[hit_index].reflectivity * current_reflectance * reflection_color;
        } else {
            color = make_float3(BACKGROUND_COLOR_X / 255.0f, BACKGROUND_COLOR_Y / 255.0f, BACKGROUND_COLOR_Z / 255.0f);
            break;
        }
        // current_reflectance *= spheres[hit_index].reflectivity;
        depth++;
    }

    color.x = fminf(fmaxf(color.x, 0.0f), 1.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 1.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 1.0f);

    return color;
}


__global__
void kernel(unsigned char* d_output, Sphere* spheres, int num_spheres, Light* lights, int num_lights, int max_depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    int index = (y * WIDTH + x) * 3;

    float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    float3 direction = normalize(make_float3(
        (x - WIDTH / 2) / (float)WIDTH,
        (y - HEIGHT / 2) / (float)HEIGHT,
        -1.0f
    ));

    Ray ray = { origin, direction };
    float3 color = trace_ray(ray, spheres, num_spheres, lights, num_lights, max_depth);

    d_output[index] = (unsigned char)(color.x * 255);
    d_output[index + 1] = (unsigned char)(color.y * 255);
    d_output[index + 2] = (unsigned char)(color.z * 255);
}

void render_frame(const std::vector<SphereData>& spheres, const std::vector<LightData>& lights, int max_depth, int frame_number) {
    int imageSize = WIDTH * HEIGHT * 3;
    unsigned char* image = new unsigned char[imageSize];

    unsigned char* d_output;
    cudaMalloc(&d_output, imageSize);

    int num_spheres = spheres.size();
    Sphere* d_spheres;
    cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere));
    cudaMemcpy(d_spheres, spheres.data(), num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice);

    int num_lights = lights.size();
    Light* d_lights;
    cudaMalloc(&d_lights, num_lights * sizeof(Light));
    cudaMemcpy(d_lights, lights.data(), num_lights * sizeof(Light), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(
      (WIDTH + blockSize.x - 1) / blockSize.x,
      (HEIGHT + blockSize.y - 1) / blockSize.y
    );

    kernel<<<gridSize, blockSize>>>(d_output, d_spheres, num_spheres, d_lights, num_lights, max_depth);
    cudaDeviceSynchronize();

    cudaMemcpy(image, d_output, imageSize, cudaMemcpyDeviceToHost);

    cv::Mat output(HEIGHT, WIDTH, CV_8UC3, image);
    std::string filename = "frames/output_" + std::to_string(frame_number) + ".jpg";
    cv::imwrite(filename, output);

    delete[] image;
    cudaFree(d_output);
    cudaFree(d_spheres);
    cudaFree(d_lights);
}

void raytrace() {
    int max_depth;
    std::cout << "Ray tracing started!" << std::endl;

    std::ifstream scene_file("scene.txt");
    if (!scene_file.is_open()) {
        std::cerr << "Failed to open scene file." << std::endl;
        return;
    }

    std::string line;
    bool reading_frame = false;
    bool reading_spheres = false;
    bool reading_lights = false;
    int frame_number = 0;
    std::vector<SphereData> spheres;
    std::vector<LightData> lights;

    while (std::getline(scene_file, line)) {
        if (line.empty() || line[0] == '#') {
            if (line.find("# Frame") != std::string::npos) {
                if (reading_frame) {
                    render_frame(spheres, lights, max_depth, frame_number);
                    frame_number++;
                    spheres.clear();
                    lights.clear();
                }
                reading_frame = true;
            } else if (line.find("# MAX_DEPTH") != std::string::npos) {
                std::getline(scene_file, line);
                max_depth = std::stoi(line);
            } else if (line.find("# Spheres") != std::string::npos) {
                reading_spheres = true;
                reading_lights = false;
            } else if (line.find("# Lights") != std::string::npos) {
                reading_lights = true;
                reading_spheres = false;
            }
            continue;
        }

        std::istringstream iss(line);
        if (reading_spheres) {
            SphereData sphere;
            iss >> sphere.center.x >> sphere.center.y >> sphere.center.z;
            iss >> sphere.radius;
            iss >> sphere.color.x >> sphere.color.y >> sphere.color.z;
            iss >> sphere.reflectivity;
            spheres.push_back(sphere);
        } else if (reading_lights) {
            LightData light;
            iss >> light.position.x >> light.position.y >> light.position.z;
            iss >> light.color.x >> light.color.y >> light.color.z;
            iss >> light.intensity;
            lights.push_back(light);
        }
    }

    if (reading_frame) {
        render_frame(spheres, lights, max_depth, frame_number);
    }

    scene_file.close();
    std::cout << "Ray tracing completed! Frames saved in the 'frames' directory." << std::endl;
}
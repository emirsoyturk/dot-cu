#ifndef RAYTRACER_H
#define RAYTRACER_H

struct Sphere {
    float3 center;
    float radius;
    float3 color;
    float reflectivity;
};

struct Cube {
    float3 center;
    float3 size;
    float3 color;
};

struct Ray {
    float3 origin;
    float3 direction;
};

struct Light {
    float3 position;
    float3 color;
};

struct SphereData {
    float3 center;
    float radius;
    float3 color;
    float reflectivity;
};

struct LightData {
    float3 position;
    float3 color;
};

__device__ bool intersect_sphere(const Sphere& sphere, const Ray& ray, float& t);

__device__ bool intersect_cube(const Cube& cube, const Ray& ray, float& t);

__global__ void kernel(unsigned char* d_output, Sphere* spheres, int num_spheres);

void raytrace();

#endif

#include <iostream>
#include "raytracer.h"

int main() {
    cudaSetDevice(0);

    raytrace();

    cudaDeviceReset();

    return 0;
}
#pragma once
#include <cmath>
#include <limits>

#define pi 3.1415926535897932385

__host__ __device__ inline double clamp(double x, double min, double max) {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

__constant__ double infinity = std::numeric_limits<double>::infinity();


__host__ __device__ inline double degrees_to_radians(double degrees) {
	return degrees * pi / 180;
}

__host__ __device__ inline double ffmin(double a, double b) { return a <= b ? a : b; }
__host__ __device__ inline double ffmax(double a, double b) { return a >= b ? a : b; }

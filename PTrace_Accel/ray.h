#pragma once
#include "vec3.h"

class ray {
public:
	__device__ ray(){}
	__device__ ray(const vec3& origin, const vec3& direction) : o(origin), d(direction) {}

	__device__ vec3 origin() const { return o; }
	__device__ vec3 direction() const { return d; }

	__device__ vec3 at(double t) const {
		return o + t * d;
	}

	vec3 o, d;
};

__device__ vec3 ray_to_background(ray r) {
	auto t = 0.5 * (unit_vector(r.d)[1] + 1.0);
	return ((1 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0));
}

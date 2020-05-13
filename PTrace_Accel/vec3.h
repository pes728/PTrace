#pragma once
#include <cmath>
#include "PMath.h"
class vec3 {
public:
	__host__ __device__ vec3() : e{ 0,0,0 } {}
	
	__host__ __device__ vec3(float x, float y, float z) : e{ x, y, z } {}

	__host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], - e[2]); }

	__host__ __device__ float operator[](unsigned int i) const { return e[i]; }
	__host__ __device__ float& operator[](unsigned int i) { return e[i]; }

	__host__ __device__ vec3& operator+=(const vec3 &v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	__host__ __device__ vec3& operator-=(const vec3& v) {
		e[0] -= v.e[0];
		e[1] -= v.e[1];
		e[2] -= v.e[2];
		return *this;
	}

	__host__ __device__ vec3& operator*=(const vec3& v) {
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];
		return *this;
	}

	__host__ __device__ vec3& operator/=(const vec3& v) {
		e[0] /= v.e[0];
		e[1] /= v.e[1];
		e[2] /= v.e[2];
		return *this;
	}

	__host__ __device__ float length() const {
		return sqrt(length_squared());
	}

	__host__ __device__ float length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	float e[3];
};

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& u, const vec3& v) {
	return vec3(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]);
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}


__host__ __device__ inline vec3 operator*(const vec3& v, double t) {
	return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, double t) {
	return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline bool operator==(const vec3& u, const vec3& v) {
	return u.e[0] == v.e[0] && u.e[1] == v.e[1] && u.e[2] == v.e[2];
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

__host__ __device__ vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}
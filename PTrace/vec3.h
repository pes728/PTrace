#pragma once
#include <cmath>
#include "color.h"
#include "PMath.h"
class vec3 {
public:
	vec3() : e{ 0,0,0 } {}
	
	vec3(double x, double y, double z) : e{ x, y, z } {}

	vec3 operator-() const { return vec3(-e[0], -e[1], - e[2]); }

	double operator[](unsigned int i) const { return e[i]; }
	double& operator[](unsigned int i) { return e[i]; }

	vec3& operator+=(const vec3 &v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	vec3& operator-=(const vec3& v) {
		e[0] -= v.e[0];
		e[1] -= v.e[1];
		e[2] -= v.e[2];
		return *this;
	}

	vec3& operator*=(const vec3& v) {
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];
		return *this;
	}

	vec3& operator/=(const vec3& v) {
		e[0] /= v.e[0];
		e[1] /= v.e[1];
		e[2] /= v.e[2];
		return *this;
	}

	double length() const {
		return sqrt(length_squared());
	}

	double length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	color to_color(unsigned int samples_per_pixel) {
		auto scale = 1.0 / samples_per_pixel;

		return color(256 * clamp(sqrt(e[0] * scale), 0.0, 0.999), 256 * clamp(sqrt(e[1] * scale), 0.0, 0.999), 256 * clamp(sqrt(e[2] * scale), 0.0, 0.999), 255);
	}

	inline static vec3 random() {
		return vec3(random_double(), random_double(), random_double());
	}

	inline static vec3 random(double min, double max) {
		return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
	}

	double e[3];
};

inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator/(const vec3& u, const vec3& v) {
	return vec3(u.e[0] / v.e[0], u.e[1] / v.e[1], u.e[2] / v.e[2]);
}

inline vec3 operator*(double t, const vec3& v) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}


inline vec3 operator*(const vec3& v, double t) {
	return t * v;
}

inline vec3 operator/(const vec3& v, double t) {
	return (1 / t) * v;
}

inline double dot(const vec3& u, const vec3& v) {
	return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline bool operator==(const vec3& u, const vec3& v) {
	return u.e[0] == v.e[0] && u.e[1] == v.e[1] && u.e[2] == v.e[2];
}

inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

vec3 random_in_unit_sphere() {
	while (1) {
		vec3 v(random_double(-1, 1), random_double(-1, 1), random_double(-1, 1));
		if (v == vec3()) continue;
		return v;
	}
}

vec3 random_unit_vector() {
	auto a = random_double(0, 2 * pi);
	auto z = random_double(-1, 1);
	auto r = sqrt(1 - z * z);
	return vec3(r * cos(a), r* sin(a), z);
}

vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}
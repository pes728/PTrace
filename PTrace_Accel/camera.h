#pragma once
#include "vec3.h"
#include "ray.h"
#include "PMath.h"


class camera {
public:
	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect) {
		vec3 u, v, w;
		float theta = vfov * pi / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * u - half_height * v - w;
		horizontal = 2 * half_width * u;
		vertical = 2 * half_height * v;
	}

	__device__ ray get_ray(double u, double v) const {
		return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
	}


	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
};
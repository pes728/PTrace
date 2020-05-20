#pragma once
#include "hittable.h"
#include "ray.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
	vec3 p;
	do {
		p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
	} while (p.length_squared() >= 1.0f);
	return p;
}

__device__ float schlick(float cosine, float ref_idx) {
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * pow((1.0f - cosine), 0.5f);
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted){
	vec3 uv = unit_vector(v);
	float dt = dot(uv, n);
	float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
	if (discriminant > 0) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	return false;
}


class material {
public:
	__device__ virtual bool scatter(const ray& r, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;
};


class lambertian : public material {
public:
	__device__ lambertian(const vec3& a) : albedo(a) {}

	__device__ virtual bool scatter(const ray& r, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
		vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
		scattered = ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}

	vec3 albedo;
};

class metal : public material {
public:
	__device__ metal(const vec3& a, float fuzz) : albedo(a), fuzz(fuzz) {}

	__device__ virtual bool scatter(const ray& r, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
		vec3 reflected = reflect(unit_vector(r.d), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
		attenuation = albedo;
		return (dot(scattered.d, rec.normal) > 0);
	}

	float fuzz;
	vec3 albedo;
};

class dielectric : public material {
public:
	__device__ dielectric(float ri) : ref_idx(ri){}
	__device__ virtual bool scatter(const ray& r, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
		vec3 outward_normal;
		vec3 reflected = reflect(r.d, rec.normal);
		float ni_over_nt;
		attenuation = vec3(1.0, 1.0, 1.0);
		vec3 refracted;
		float reflect_prob;
		float cosine;
		if (dot(r.d, rec.normal) > 0.0f) {
			outward_normal = -rec.normal;
			ni_over_nt = ref_idx;
			cosine = dot(r.d, rec.normal) / r.d.length();
			cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
		}
		else {
			outward_normal = rec.normal;
			ni_over_nt = 1.0f / ref_idx;
			cosine = -dot(r.d, rec.normal) / r.d.length();
		}
		if (refract(r.d, rec.normal, ni_over_nt, refracted))
			reflect_prob = schlick(cosine, ref_idx);
		else
			reflect_prob = 1.0f;
		
		if (curand_uniform(local_rand_state) < reflect_prob)
			scattered = ray(rec.p, reflected);
		else
			scattered = ray(rec.p, refracted);
		return true;
	}	


	float ref_idx;
};

class normal : public material {
public:
	__device__ normal() {}

	__device__ virtual bool scatter(const ray& r, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
		scattered = ray(vec3(), vec3());
		attenuation = (0.5 * (rec.normal + vec3(1, 1, 1)));
		return true;
	}
};
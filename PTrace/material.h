#pragma once
#include "hittable.h"
class material {
public:
	virtual bool scatter(const ray& r, const hit_record& rec, vec3& attenuation, ray& scattered) const = 0;
};


class lambertian : public material {
public:
	lambertian(const vec3& a) : albedo(a) {}

	virtual bool scatter(const ray& r, const hit_record& rec, vec3& attenuation, ray& scattered) const {
		vec3 scatter_direction = rec.normal + random_unit_vector();
		scattered = ray(rec.p, scatter_direction);
		attenuation = albedo;
		return true;
	}

	vec3 albedo;
};

class metal : public material {
public:
	metal(const vec3& a, double fuzz) : albedo(a), fuzz(fuzz) {}

	virtual bool scatter(const ray& r, const hit_record& rec, vec3& attenuation, ray& scattered) const {
		vec3 reflected = reflect(unit_vector(r.d), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
		attenuation = albedo;
		return (dot(scattered.d, rec.normal) > 0);
	}

	double fuzz;
	vec3 albedo;
};

class normal : public material {
public:
	normal() {}

	virtual bool scatter(const ray& r, const hit_record& rec, vec3& attenuation, ray& scattered) const {
		scattered = ray(vec3(), vec3());
		attenuation = (0.5 * (rec.normal + vec3(1, 1, 1)));
		return true;
	}
};
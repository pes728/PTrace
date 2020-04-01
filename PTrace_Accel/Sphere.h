#pragma once
#include "ray.h"
#include "hittable.h"

class sphere : public hittable{
public:
	sphere() {}
	sphere(vec3 cen, double radius, std::shared_ptr<material> mat_ptr) : center(cen), radius(radius), m_ptr(mat_ptr){}

	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;

	vec3 center;
	double radius;
    std::shared_ptr<material> m_ptr;
};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    vec3 oc = r.o - center;
    auto a = r.d.length_squared();
    auto half_b = dot(oc, r.d);
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant > 0) {
        auto root = sqrt(discriminant);
        auto temp = (-half_b - root) / a;
        if (temp  < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = m_ptr;
            return true;
        }
        temp = (-half_b + root) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);
            rec.mat_ptr = m_ptr;
            return true;
        }
    }
    return false;
}

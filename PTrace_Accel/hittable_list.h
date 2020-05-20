#pragma once
#include "hittable.h"
#include <memory>
#include <vector>

class hittable_list : public hittable {
public:
	__device__ hittable_list() {}
	__device__ hittable_list(hittable** l, int n) { list = l; list_size = n; }

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

	hittable** list;
	int list_size;
};



__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_t = t_max;

	for (int i = 0; i < list_size; i++) {
		if (list[i]->hit(r, t_min, closest_t, temp_rec)) {
			hit_anything = true;
			closest_t = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
 }

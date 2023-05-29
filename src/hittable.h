#ifndef HITTABLE_H 
#define HITTABLE_H
#include "ray.h"

class material;

struct hit_record { 
    point3 p;
    vec3 normal;
    double t; 
    bool is_outside;
    material *mat_ptr;

    __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) { 
        is_outside = dot(r.direction(), outward_normal) < 0;
        normal = is_outside ? outward_normal :-outward_normal;
        }   

    };

class hittable { 
    public:
        point3 p;
        vec3 normal;
        double t; 
        

        __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};
        
#endif
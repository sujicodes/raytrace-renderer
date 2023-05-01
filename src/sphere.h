#ifndef SPHERE_H 
#define SPHERE_H
#include "hittable.h" 
#include "vec3.h"
#include "vec3_utils.h"

class sphere : public hittable {
     public:
        point3 center;
        double radius;

        sphere() {}
        sphere(point3 cen, double r) : center(cen), radius(r) {};
        virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override;
};

bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const 
{ 
    auto oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius*radius;
    
    //std::cerr << "a = " << a << " b = " << b
    //    << " c = " << c << std::endl;

    
    auto discriminant = b*b - 4*a*c;
    
    
    //auto half_b = dot(oc, r.direction());
    //auto c = oc.length_squared() - radius*radius;
    //auto discriminant = half_b*half_b - a*c; 
    if (discriminant < 0) 
        return false; 

    auto root = (-b - sqrt(discriminant))/ (2.0*a);
    //auto sqrtd = sqrt(discriminant);
    // Find the nearest root that lies in the acceptable range.
    // auto root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-b + sqrt(discriminant))/ (2.0*a);
        if (root < t_min || t_max < root)
            return false;
    }
    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius; 
    rec.set_face_normal(r, outward_normal);

    return true; 
    }
#endif
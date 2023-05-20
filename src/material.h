#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h" 
#include "utils.h"

struct material{
    color albedo = color(0,0,0);
    float roughness = 0; 
    color emmission_colour = color(0,0,0);
    size_t emmission_strength = 0;
    float specular_probability = 0;
    bool is_dialetric = false;
    material(){}
    material(const color& a): albedo(a) {}
};

vec3 diffuse_ray_direction(const ray &r, const hit_record& rec){
    vec3 scatter_direction = rec.normal + random_unit_vector();
    // Catch degenerate scatter direction
    if (scatter_direction.near_zero())
        scatter_direction = rec.normal;
    return scatter_direction;
}

vec3 specular_ray_direction(const ray &r, const hit_record& rec){
    return reflect(unit_vector(r.direction()), rec.normal);
}

vec3 dialetric_ray_direction(const ray &r, const hit_record& rec, const float &index_of_refraction){
    double refraction_ratio = rec.is_outside ? (1.0/index_of_refraction) : index_of_refraction;

    vec3 unit_direction = unit_vector(r.direction());

    double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
    double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
    vec3 direction;
    bool not_refractable = refraction_ratio * sin_theta > 1.0;

    if (not_refractable || reflectance(cos_theta, refraction_ratio) > random_double())
        direction = reflect(unit_direction, rec.normal);
    else
        direction = refract(unit_direction, rec.normal, refraction_ratio);
    return direction;
}


#endif
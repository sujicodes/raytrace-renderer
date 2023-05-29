#ifndef CAMERA_H 
#define CAMERA_H
#include "utils.h"

class camera {

    point3 origin;
    point3 lower_left_corner; 
    vec3 horizontal;
    vec3 vertical;
    double lens_radius;
    vec3 u, v, w;

    public:
         __device__ camera(
            point3 lookfrom,
            point3 lookat,
            vec3   vup,
            double vertical_fov,
            double aspect_ratio,
            double focus_distance,
            double aperture
        ) {
            double theta = degrees_to_radians(vertical_fov);
            double h = tan(theta/2);
            double viewport_height = 2.0 * h;
            double viewport_width = aspect_ratio * viewport_height; 
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            origin = lookfrom;
            horizontal = focus_distance * viewport_width * u;
            vertical = focus_distance * viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_distance*w;

            lens_radius = aperture/2;
        }
         __device__ ray get_ray(double s, double t, curandState *local_rand_state) const {
            vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
            vec3 offset = u * rd.x() + v * rd.y();

            return ray(
                origin + offset, 
                lower_left_corner + s*horizontal + t*vertical - origin - offset
                );
        }
};
#endif
                    
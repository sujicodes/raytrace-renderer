#include <iostream>

#include "utils.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"

using namespace std;

void write_color(std::ostream &out, color pixel_color, int samples_per_pixel) { 
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();
    // Divide the color by the number of samples.
    auto scale = 1.0 / samples_per_pixel; r *= scale;
    g *= scale;
    b *= scale;
    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999))
        << ' ' << static_cast<int>(256 * clamp(g, 0.0, 0.999)) 
        << ' ' << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
    }

double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    auto a = dot(r.direction(), r.direction());
    auto b = 2.0 * dot(oc, r.direction());
    auto c = dot(oc, oc) - radius*radius;
    auto discriminant = b*b - 4*a*c;
    if (discriminant >= 0) {
        return (-b - sqrt(discriminant) ) / (2.0*a);
    } else {
       return -1.0;
    }
}

vec3 get_normals(const ray &r, const double t) {
    return unit_vector(r.at(t) - vec3(0,0,-1));
}


color ray_color(const ray& r, const hittable &world) {
    hit_record rec;
    
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    cerr << "t_outside if:" << t << std::endl;
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

int main() {
    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;


    // world
    hittable_list world; 
    world.add(make_shared<sphere>(point3(0,0,-1), 0.5)); 
    world.add(make_shared<sphere>(point3(0,-100.5,-1), 100));

    // Camera
    camera cam;

    // render 
    cout << "P3\n" << image_width << ' ' << image_height << "\n255\n"<< std::endl;
    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rRender Progress: " << ((image_height-j)*image_height)/100 << "% " << std::endl;
        for (int i = 0; i < image_width; ++i) {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width-1); 
                auto v = (j + random_double()) / (image_height-1); 
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world);
            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }
}
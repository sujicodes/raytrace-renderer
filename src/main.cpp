#include <iostream>
#include <tuple>

#include "utils.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"

using namespace std;

hittable_list random_scene() {
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -15; a < 15; a++) {
        for (int b = -15; b < 15; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.65) {
                    // diffuse
                    auto albedo = color(random_double(), random_double(), random_double());
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.8) {
                    sphere_material = make_shared<light>();
                    sphere_material->emmission_colour = color(random_double(), random_double(), random_double()) ;
                    sphere_material->emmission_strength = 4;
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color(random_double(), random_double(), random_double());
                    auto fuzz = random_double();
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    auto light_source = make_shared<light>();
    world.add(make_shared<sphere>(point3(0.0, 3.0, -6.0), 2.0, light_source));
    light_source->emmission_colour = color(1,1,1);
    light_source->emmission_strength = 5;
    return world;
}

tuple<double,double,double> gamma_correction(double gamma, const color &pixel_colour, int samples_per_pixel){
    double r = pixel_colour.x();
    double g = pixel_colour.y();
    double b = pixel_colour.z();
    gamma = 1/gamma;
    // Divide the color by the number of samples and apply gamma correction
    double scale = 1.0 / samples_per_pixel; 
    r = pow(scale*r, gamma);
    g = pow(scale*g, gamma);
    b = pow(scale*b, gamma);
    return make_tuple(r,g,b);
}

void write_color(std::ostream &out, const color &pixel_color, int samples_per_pixel) { 
    // Write the translated [0,255] value of each color component.
    tuple<double,double,double> corrected_colour = gamma_correction(2.5, pixel_color, samples_per_pixel);
    
    out << static_cast<int>(256 * clamp(get<0>(corrected_colour), 0.0, 0.999))
        << ' ' << static_cast<int>(256 * clamp(get<1>(corrected_colour), 0.0, 0.999)) 
        << ' ' << static_cast<int>(256 * clamp(get<2>(corrected_colour), 0.0, 0.999)) << '\n';
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


color ray_color(const ray& r, const hittable &world, int depth) {
    hit_record rec;
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0,0,0);
    //1e-8
    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;
        color emmitted_light = rec.mat_ptr->emmission_colour * rec.mat_ptr->emmission_strength;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)){
            return (emmitted_light + attenuation * ray_color(scattered, world, depth-1));
        }
        return emmitted_light;
    };

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return color(0.15, 0.15, 0.15);
    //return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

int main() {
    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 500;
    const int max_depth = 50;
    
    // world
    hittable_list world = random_scene();   
    // Camera
    point3 lookfrom(13,2,3);
    point3 lookat(0,0,0);
    vec3 vup(0,1,0);
    auto dist_to_focus = 10;
    auto aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20, aspect_ratio, dist_to_focus, aperture);

    // render 
    cout << "P3\n" << image_width << ' ' << image_height << "\n255\n"<< std::endl;
    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rRender Progress: " << ((image_height-j)/image_height)*100 << "% " << std::endl;
        for (int i = 0; i < image_width; ++i) {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width-1); 
                auto v = (j + random_double()) / (image_height-1); 
                ray r = cam.get_ray(u, v);
                pixel_color += (0.5 *ray_color(r, world, max_depth));
            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }
}
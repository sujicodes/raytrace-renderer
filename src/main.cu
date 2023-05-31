#include <iostream>
#include <tuple>
#include <curand_kernel.h>

#include "utils.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"

using namespace std;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void random_scene(hittable_list *d_world, camera **d_camera, int image_width, int image_height, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        
        hittable_list *world = d_world;
       
        auto ground_material = new material(color(0.5, 0.5, 0.5));
        ground_material->roughness=1;
        world->add(new sphere(point3(0,-1000,0), 1000, ground_material));
        
        for (int a = -15; a < 15; a++) {
            for (int b = -15; b < 15; b++) {
                auto choose_mat = curand_uniform(&local_rand_state);
                point3 center(a + 0.9*curand_uniform(&local_rand_state), 0.2, b + 0.9*curand_uniform(&local_rand_state));
                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    material *sphere_material;
                    if (choose_mat < 0.65) {
                        // diffuse
                        auto albedo = color(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state));
                        sphere_material = new material(albedo);
                        sphere_material->roughness=1;
                        world->add(new sphere(center, 0.2, sphere_material));
                    } else if (choose_mat < 0.75) {
                        sphere_material = new material();
                        sphere_material->emmission_colour = color(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state)) ;
                        sphere_material->emmission_strength = 4;
                        world->add(new sphere(center, 0.2, sphere_material));
                    } else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color(curand_uniform(&local_rand_state), curand_uniform(&local_rand_state), curand_uniform(&local_rand_state));
                        auto roughness = curand_uniform(&local_rand_state);
                        sphere_material = new material(albedo);
                        sphere_material->roughness = roughness;
                        world->add(new sphere(center, 0.2, sphere_material));
                    } else {
                        // glass
                        sphere_material = new material(color(1,1,1));
                        sphere_material->is_dialetric = true;
                        world->add(new sphere(center, 0.2, sphere_material));
                    }
                }
            }
        }
    
        auto glass = new material(color(1,1,1)); 
        glass->is_dialetric=true;
        world->add(new sphere(point3(0, 1, 0), 1.0, glass));

        auto lambert = new material(color(0.4, 0.2, 0.1));
        lambert->roughness = 1;
        world->add(new sphere(point3(-4, 1, 0), 1.0, lambert));

        auto metal = new material(color(0.7, 0.6, 0.5));
        metal->roughness = 0;
        world->add(new sphere(point3(4, 1, 0), 1.0, metal));

        auto light_source = new material(color(0,0,0));
        world->add(new sphere(point3(0.0, 3.0, -6.0), 2.0, light_source));
        light_source->emmission_colour = color(1,1,1);
        light_source->emmission_strength = 5;

        point3 lookfrom(13,2,3);
        point3 lookat(0,0,0);
        vec3 vup(0,1,0);
        auto dist_to_focus = 10;
        auto aperture = 0.1;

        *d_camera = new camera(lookfrom, lookat, vup, 20, float(image_width)/image_height, dist_to_focus, aperture);;
    }
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

void write_color(std::ostream &out, const color &pixel_color) { 
    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(pixel_color.x(), 0.0, 0.999))
        << ' ' << static_cast<int>(256 * clamp(pixel_color.y(), 0.0, 0.999)) 
        << ' ' << static_cast<int>(256 * clamp(pixel_color.z(), 0.0, 0.999)) << '\n';
    }

//double hit_sphere(const point3& center, double radius, const ray& r) {
//    vec3 oc = r.origin() - center;
//   auto a = dot(r.direction(), r.direction());
//    auto b = 2.0 * dot(oc, r.direction());
//    auto c = dot(oc, oc) - radius*radius;
//    auto discriminant = b*b - 4*a*c;
//   if (discriminant >= 0) {
//        return (-b - sqrt(discriminant) ) / (2.0*a);
//    } else {
//       return -1.0;
//    }
//}

//vec3 get_normals(const ray &r, const double t) {
//    return unit_vector(r.at(t) - vec3(0,0,-1));
//}

__device__ color ray_color(const ray& r, const hittable *world, curandState *local_rand_state) {
   
    // If we've exceeded the ray bounce limit, no more light is gathered.
    color ray_col = color(1,1,1);
    color incoming_light = color(0,0,0);

    for(size_t i = 0; i < 50; i++) {
        hit_record rec;
        //1e-8
        if (world->hit(r, 0.001, infinity, rec)) {
            ray scattered;
            color attenuation;
            color emmitted_light = rec.mat_ptr->emmission_colour * rec.mat_ptr->emmission_strength;
            vec3 ray_dir;
            if (!rec.mat_ptr->is_dialetric){
                vec3 diff_ray_dir = diffuse_ray_direction(r, rec, local_rand_state);
                vec3 spec_ray_dir = specular_ray_direction(r, rec);
                ray_dir = lerp(spec_ray_dir, diff_ray_dir, rec.mat_ptr->roughness);
            }
            else{
                ray_dir = dialetric_ray_direction(r, rec, 1.5, local_rand_state);
            }
            scattered = ray(rec.p, ray_dir);
            //return (emmitted_light +  rec.mat_ptr->albedo * ray_color(scattered, world, depth-1));
            //return emmitted_light;
            incoming_light += emmitted_light * ray_col;
            ray_col = ray_col * rec.mat_ptr->albedo;

    };
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return color(0.15, 0.15, 0.15);
    //return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int samples_per_pixel, camera **cam, hittable_list *world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < samples_per_pixel; s++) {
        double u = double(i + curand_uniform(&local_rand_state)) / max_x;
        double v = double(j + curand_uniform(&local_rand_state)) / max_y;
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += ray_color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(samples_per_pixel);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

//try and remove
__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void free_world(hittable_list *d_world, camera **d_camera) {
    d_world->clear();
    delete *d_camera;
    
}

int main() {
    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 1200;
    const int image_height = 600;
    const int samples_per_pixel = 100;
    int tx = 8;
    int ty = 8;

    int num_pixels = image_width*image_height;
    size_t fb_size = 3*num_pixels*sizeof(double);

    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    
    dim3 blocks(image_width/tx+1, image_height/ty+1);
    dim3 threads(tx,ty);
    

    // make our world of hitables & the camera
    int num_hitables = 30*30+2+3;
    hittable_list *d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable_list *)));
    checkCudaErrors(cudaMalloc((void**)(&d_world->objects), num_hitables * sizeof(shared_ptr<hittable>)));
    
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    
    random_scene<<<1,1>>>(d_world, d_camera, image_width, image_height, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // render 
    cout << "P3\n" << image_width << ' ' << image_height << "\n255\n"<< std::endl;
    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rRender Progress: " << ((image_height-j)/image_height)*100 << "% " << std::endl;
        for (int i = 0; i < image_width; ++i) {
            size_t pixel_index = j*image_width + i;
            write_color(std::cout, fb[pixel_index]);
        }
    }


    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}
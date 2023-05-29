#ifndef VEC3_UTILS_H
#define VEC3_UTILS_H

#include "vec3.h"

std::ostream& operator<<(std::ostream &out, const vec3 &v);
__host__ __device__ vec3 operator+(const vec3 &u, const vec3 &v);
__host__ __device__ vec3 operator-(const vec3 &u, const vec3 &v);
__host__ __device__ vec3 operator*(const vec3 &u, const vec3 &v);
__host__ __device__ vec3 operator*(double t, const vec3 &v);
__host__ __device__ vec3 operator*(const vec3 &v, double t);
__host__ __device__ vec3 operator/(vec3 v, double t);

__host__ __device__ double dot(const vec3 &u, const vec3 &v);
__host__ __device__ vec3 cross(const vec3 &u, const vec3 &v);
__host__ __device__ vec3 unit_vector(vec3 v);
__host__ __device__ vec3 random_in_unit_sphere(curandState *local_rand_state);
__host__ __device__ vec3 random_unit_vector(curandState *local_rand_state);
__host__ __device__ vec3 random_in_hemisphere(const vec3& normal, curandState *local_rand_state);
__host__ __device__ vec3 reflect(const vec3& v, const vec3& n);
__host__ __device__ vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat);
__host__ __device__ vec3 random_in_unit_disk(curandState *local_rand_state);

#endif
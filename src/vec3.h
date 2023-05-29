#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class vec3 {
    public:
        double e[3];

        __host__ __device__ vec3();
        __host__ __device__ vec3(double e0, double e1, double e2);

        __host__ __device__ double x() const { return e[0]; }
        __host__ __device__ double y() const { return e[1]; }
        __host__ __device__ double z() const { return e[2]; }

        __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        __host__ __device__ double operator[](int i) const { return e[i]; }
        __host__ __device__ double &operator[](int i) { return e[i]; }
        __host__ __device__ vec3 &operator+=(const vec3 &v);
        __host__ __device__ vec3 &operator*=(const double t);
        __host__ __device__ vec3 &operator/=(const double t);
        __host__ __device__ double length() const;
        __host__ __device__ double length_squared() const;
        __host__ __device__ bool near_zero() const;
        //static vec3 random();
};

// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

#endif
#ifndef UTILS_H 
#define UTILS_H
#include <cmath> 
#include <limits> 
#include <memory>
#include <cstdlib>

// Usings
using std::shared_ptr; 
using std::make_shared; 
using std::sqrt;
// Constants
__device__ const double infinity = std::numeric_limits<double>::infinity(); 
__device__ const double pi = 3.1415926535897932385;

// Utility Functions
 __device__ inline double degrees_to_radians(double degrees) { 
    return degrees * pi / 180.0;
}

inline double clamp(double x, double min, double max) { 
    if (x < min) 
        return min;
    if (x > max) 
        return max;
    return x;
}

 __device__ inline double reflectance(double cosine, double ref_idx) {
    // Use Schlick's approximation for reflectance.
    auto r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
    }

template<typename T>
 __device__ T lerp(const T& a, const T& b, float t)
{
    return a + (b - a) * t;
}

// Common Headers
#include "ray.h" 
#include "vec3.h"
#endif
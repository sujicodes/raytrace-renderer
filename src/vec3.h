#ifndef VEC3_H
#define VEC3_H

#include <cmath>
#include <iostream>

class vec3 {
    public:
        vec3();
        vec3(double e0, double e1, double e2);

        double x() const { return e[0]; }
        double y() const { return e[1]; }
        double z() const { return e[2]; }

        vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        double operator[](int i) const { return e[i]; }
        double &operator[](int i) { return e[i]; }
        vec3 &operator+=(const vec3 &v);
        vec3 &operator*=(const double t);
        vec3 &operator/=(const double t);
        double length() const;
        double length_squared() const;
    public:
        double e[3];
};

// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

#endif
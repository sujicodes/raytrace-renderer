#include "vec3.h"
#include <cmath>
#include <iostream>
#include "utils.h"

using std::sqrt;
using std::pow;


vec3::vec3()
    : e{0,0,0} {}

vec3::vec3(double e0, double e1, double e2)
    : e{e0, e1, e2} {}

vec3 &vec3::operator+=(const vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }
vec3 &vec3::operator*=(const double t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

vec3 &vec3::operator/=(const double t) {
            return *this *= 1/t;
        }

double vec3::length_squared() const {
            return pow(e[0], 2) + pow(e[1], 2) + pow(e[2], 2);
        }

double vec3::length() const {
            return sqrt(length_squared());
        }

bool vec3::near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }


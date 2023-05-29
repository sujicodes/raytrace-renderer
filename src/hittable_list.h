#ifndef HITTABLE_LIST_H 
#define HITTABLE_LIST_H
#include <memory> 
#include <vector>

#include "hittable.h"

using std::shared_ptr; 
using std::make_shared;

class hittable_list : public hittable { 
    public:

        hittable **objects;
        int size;
        
        __device__ hittable_list() : objects(nullptr), size(0) {}
        __device__ hittable_list(hittable *object) { add(object); }

        __device__ void clear() {  
            for(size_t i=0; i < size; i++) {
                delete ((sphere *)objects[i])->mat_ptr;
                delete objects[i];
            }
            if (objects != nullptr) {
                delete[] objects;
                objects = nullptr;
                }
            size = 0;
        }
        __device__ void add(hittable *object) { objects[size++] = object; }

        __device__ virtual bool hit(
        const ray& r, double t_min, double t_max, hit_record& rec) const override;
};

__device__ bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const { 
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;
    for (size_t i = 0; i <= size; i++) {
        auto object = objects[i];
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true; 
            closest_so_far = temp_rec.t; 
            rec = temp_rec;
            } 
        }
    return hit_anything; 
    }


#endif
                          
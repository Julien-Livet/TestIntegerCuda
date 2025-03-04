#ifndef CU_PAIR_CUH
#define CU_PAIR_CUH

#include <cassert>
#include <compare>

#include <cuda.h>

namespace cu
{
    template <class T1, class T2>
    struct pair
    {
        T1 first;
        T2 second;

        __device__ __host__ pair()
        {
        }

        __device__ __host__ pair(T1 const& x, T2 const& y) : first(x), second(y)
        {
        }

        __device__ __host__ pair(pair const& other) : first(other.first), second(other.second)
        {
        }

        __device__ __host__ ~pair()
        {
        }

        __device__ __host__ pair& operator=(pair const& other)
        {
            first = other.first;
            second = other.second;
        }
    };
}

#endif // CU_PAIR_CUH

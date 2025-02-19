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

        __device__ __host__ pair() = default;

        __device__ __host__ pair(T1 const& x, T2 const& y) : first{x}, second{y}
        {
        }

        __device__ __host__ ~pair() = default;
    };
}

#endif // CU_PAIR_CUH

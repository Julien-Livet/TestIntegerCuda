#ifndef CU_UTILITY_CUH
#define CU_UTILITY_CUH

namespace cu
{
    template <typename T>
    struct less
    {
        __device__ __host__ constexpr less() = default;

        __device__ __host__ constexpr bool operator()(const T& lhs, const T& rhs) const 
        {
            return lhs < rhs; // assumes that the implementation handles pointer total order
        }
    };
}

#endif // CU_UTILITY_CUH

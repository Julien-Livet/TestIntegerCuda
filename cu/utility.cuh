#ifndef CU_UTILITY_CUH
#define CU_UTILITY_CUH

namespace cu
{
    template< class T >
    __device__ __host__ constexpr std::remove_reference_t<T>&& move(T&& t) noexcept
    {
        return static_cast<typename std::remove_reference<T>::type&&>(t) ;
    }

    template <typename T>
    struct less
    {
        __device__ __host__ constexpr less()
        {
        }

        __device__ __host__ constexpr ~less()
        {
        }

        __device__ __host__ constexpr bool operator()(const T& lhs, const T& rhs) const 
        {
            return lhs < rhs; // assumes that the implementation handles pointer total order
        }
    };
}

#endif // CU_UTILITY_CUH

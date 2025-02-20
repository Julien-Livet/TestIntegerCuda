#ifndef CU_ALGORITHM_CUH
#define CU_ALGORITHM_CUH

#include <iterator>

#include "iterator.cuh"
#include "pair.cuh"
#include "utility.cuh"

namespace cu
{
    template<class T>
    __device__ __host__ const T& min(const T& a, const T& b)
    {
        return (b < a) ? b : a;
    }
    
    template<class T>
    __device__ __host__ const T& max(const T& a, const T& b)
    {
        return (a < b) ? b : a;
    }

    template<class InputIt, class OutputIt>
    __device__ __host__
    OutputIt copy(InputIt first, InputIt last,
                OutputIt d_first)
    {
        for (; first != last; (void)++first, (void)++d_first)
            *d_first = *first;
    
        return d_first;
    }

    template<class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type,
         class Compare>
    __device__ __host__
    ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
    {
        ForwardIt it;
        typename std::iterator_traits<ForwardIt>::difference_type count, step;
        count = cu::distance(first, last);
    
        while (count > 0)
        {
            it = first;
            step = count / 2;
            cu::advance(it, step);
    
            if (comp(*it, value))
            {
                first = ++it;
                count -= step + 1;
            }
            else
                count = step;
        }
    
        return first;
    }

    template<class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type>
    __device__ __host__
    ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value)
    {
        return cu::lower_bound(first, last, value, cu::less<T>());
    }

    template<class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type,
         class Compare>
    __device__ __host__
    ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
    {
        ForwardIt it;
        typename std::iterator_traits<ForwardIt>::difference_type count, step;
        count = cu::distance(first, last);
    
        while (count > 0)
        {
            it = first; 
            step = count / 2;
            cu::advance(it, step);
    
            if (!comp(value, *it))
            {
                first = ++it;
                count -= step + 1;
            } 
            else
                count = step;
        }
    
        return first;
    }

    template<class ForwardIt, class T = typename std::iterator_traits<ForwardIt>::value_type>
    __device__ __host__
    ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value)
    {
        return std::upper_bound(first, last, value, cu::less<T>());
    }

    template<class ForwardIt,
         class T = typename std::iterator_traits<ForwardIt>::value_type,
         class Compare>
    __device__ __host__
    constexpr cu::pair<ForwardIt, ForwardIt>
        equal_range(ForwardIt first, ForwardIt last, const T& value, Compare comp)
    {
        return {cu::lower_bound(first, last, value, comp),
                cu::upper_bound(first, last, value, comp)};
    }

    template<class ForwardIt,
         class T = typename std::iterator_traits<ForwardIt>::value_type>
    __device__ __host__
    constexpr cu::pair<ForwardIt, ForwardIt> 
        equal_range(ForwardIt first, ForwardIt last, const T& value)
    {
        return cu::equal_range(first, last, value, cu::less<T>());
    }
}

#endif // CU_ALGORITHM_CUH

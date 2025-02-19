#ifndef CU_ITERATOR_CUH
#define CU_ITERATOR_CUH

namespace cu
{
    namespace detail
    {
        template<class It>
        __device__ __host__ constexpr // required since C++17
        typename std::iterator_traits<It>::difference_type 
            do_distance(It first, It last, std::output_iterator_tag)
        {
            typename std::iterator_traits<It>::difference_type result = 0;
            while (first != last)
            {
                ++first;
                ++result;
            }
            return result;
        }
    
        template<class It>
        __device__ __host__ constexpr // required since C++17
        typename std::iterator_traits<It>::difference_type 
            do_distance(It first, It last, std::input_iterator_tag)
        {
            typename std::iterator_traits<It>::difference_type result = 0;
            while (first != last)
            {
                ++first;
                ++result;
            }
            return result;
        }
    
        template<class It>
        __device__ __host__ constexpr // required since C++17
        typename std::iterator_traits<It>::difference_type 
            do_distance(It first, It last, std::random_access_iterator_tag)
        {
            return last - first;
        }
    } // namespace detail
    
    template <class It>
    __device__ __host__ constexpr // since C++17
    typename std::iterator_traits<It>::difference_type 
    distance(It first, It last)
    {
        return detail::do_distance(first, last,
                                   typename std::iterator_traits<It>::iterator_category());
    }

    template<class It, class Distance>
    __device__ __host__
    constexpr void advance(It& it, Distance n)
    {
        using category = typename std::iterator_traits<It>::iterator_category;
        static_assert(std::is_base_of_v<std::input_iterator_tag, category>);
    
        auto dist = typename std::iterator_traits<It>::difference_type(n);
        if constexpr (std::is_base_of_v<std::random_access_iterator_tag, category>)
            it += dist;
        else
        {
            while (dist > 0)
            {
                --dist;
                ++it;
            }
            if constexpr (std::is_base_of_v<std::bidirectional_iterator_tag, category>)
                while (dist < 0)
                {
                    ++dist;
                    --it;
                }
        }
    }
}

#endif // CU_ITERATOR_CUH

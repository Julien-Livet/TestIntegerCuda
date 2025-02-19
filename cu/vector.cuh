#ifndef CU_VECTOR_CUH
#define CU_VECTOR_CUH

#include <cassert>
#include <compare>

#include <cuda.h>

#include "iterator.cuh"

namespace cu
{
    template <typename T>
    class vector
    {
        public:
            using value_type = T;

            class const_iterator
            {
                public:
                    using pointer = T const*;
                    using iterator_category = std::bidirectional_iterator_tag;
                    using difference_type = std::ptrdiff_t;
                    using value_type = T;

                    __device__ __host__ const_iterator(T const* ptr) : ptr_{ptr}
                    {
                    }
            
                    __device__ __host__ T const& operator*() const
                    {
                        return *ptr_;
                    }
            
                    __device__ __host__ T const* operator->() const
                    {
                        return ptr_;
                    }
            
                    __device__ __host__ const_iterator& operator++()
                    {
                        ++ptr_;

                        return *this;
                    }
            
                    __device__ __host__ const_iterator operator++(int)
                    {
                        auto temp(*this);
                        ++ptr_;
                        return temp;
                    }
            
                    __device__ __host__ const_iterator& operator--()
                    {
                        --ptr_;

                        return *this;
                    }
            
                    __device__ __host__ const_iterator operator--(int)
                    {
                        auto temp(*this);
                        --ptr_;
                        return temp;
                    }

                    __device__ __host__ const_iterator& operator+=(size_t i)
                    {
                        ptr_ += i;

                        return *this;
                    }
            
                    __device__ __host__ const_iterator operator+(size_t i) const
                    {
                        auto tmp(*this);

                        return tmp += i;
                    }
            
                    __device__ __host__ const_iterator& operator-=(size_t i)
                    {
                        ptr_ -= i;

                        return *this;
                    }
            
                    __device__ __host__ const_iterator operator-(size_t i) const
                    {
                        auto tmp(*this);

                        return tmp -= i;
                    }
            
                    __device__ __host__ bool operator!=(const_iterator const& other) const
                    {
                        return ptr_ != other.ptr_;
                    }
            
                    __device__ __host__ bool operator==(const_iterator const& other) const
                    {
                        return !(*this != other);
                    }
                
                private:
                    T const* ptr_;    
            };

            class iterator
            {
                public:
                    using pointer = T*;
                    using iterator_category = std::bidirectional_iterator_tag;
                    using difference_type = std::ptrdiff_t;
                    using value_type = T;

                    __device__ __host__ iterator(T* ptr) : ptr_{ptr}
                    {
                    }
            
                    __device__ __host__ T& operator*()
                    {
                        return *ptr_;
                    }
            
                    __device__ __host__ T const& operator*() const
                    {
                        return *ptr_;
                    }
            
                    __device__ __host__ T* operator->()
                    {
                        return ptr_;
                    }
            
                    __device__ __host__ T const* operator->() const
                    {
                        return ptr_;
                    }
            
                    __device__ __host__ iterator& operator++()
                    {
                        ++ptr_;

                        return *this;
                    }
            
                    __device__ __host__ iterator operator++(int)
                    {
                        auto temp(*this);
                        ++ptr_;
                        return temp;
                    }
            
                    __device__ __host__ iterator& operator--()
                    {
                        --ptr_;

                        return *this;
                    }
            
                    __device__ __host__ iterator operator--(int)
                    {
                        auto temp(*this);
                        --ptr_;
                        return temp;
                    }

                    __device__ __host__ iterator& operator+=(size_t i)
                    {
                        ptr_ += i;

                        return *this;
                    }
            
                    __device__ __host__ iterator operator+(size_t i) const
                    {
                        auto tmp(*this);

                        return tmp += i;
                    }
            
                    __device__ __host__ iterator& operator-=(size_t i)
                    {
                        ptr_ -= i;

                        return *this;
                    }
            
                    __device__ __host__ iterator operator-(size_t i) const
                    {
                        auto tmp(*this);

                        return tmp -= i;
                    }
            
                    __device__ __host__ bool operator!=(iterator const& other) const
                    {
                        return ptr_ != other.ptr_;
                    }
            
                    __device__ __host__ bool operator==(iterator const& other) const
                    {
                        return !(*this != other);
                    }
                
                    __device__ __host__ operator const_iterator() const
                    {
                        return const_iterator(ptr_);
                    }

                private:
                    T* ptr_;    
            };

            class const_reverse_iterator
            {
                public:
                    using pointer = T const*;
                    using iterator_category = std::bidirectional_iterator_tag;
                    using difference_type = std::ptrdiff_t;
                    using value_type = T;

                    __device__ __host__ const_reverse_iterator(T const* ptr) : ptr_{ptr}
                    {
                    }
            
                    __device__ __host__ T const& operator*() const
                    {
                        return *ptr_;
                    }
            
                    __device__ __host__ T const* operator->() const
                    {
                        return ptr_;
                    }
            
                    __device__ __host__ const_reverse_iterator& operator--()
                    {
                        ++ptr_;
                        return *this;
                    }
            
                    __device__ __host__ const_reverse_iterator operator--(int)
                    {
                        auto temp(*this);
                        ++ptr_;
                        return temp;
                    }
            
                    __device__ __host__ const_reverse_iterator& operator++()
                    {
                        --ptr_;
                        return *this;
                    }
            
                    __device__ __host__ const_reverse_iterator operator++(int)
                    {
                        auto temp(*this);
                        --ptr_;
                        return temp;
                    }
            
                    __device__ __host__ const_reverse_iterator& operator+=(size_t i)
                    {
                        ptr_ -= i;

                        return *this;
                    }
            
                    __device__ __host__ const_reverse_iterator operator+(size_t i) const
                    {
                        auto tmp(*this);

                        return tmp -= i;
                    }
            
                    __device__ __host__ const_reverse_iterator& operator-=(size_t i)
                    {
                        ptr_ += i;

                        return *this;
                    }
            
                    __device__ __host__ const_reverse_iterator operator-(size_t i) const
                    {
                        auto tmp(*this);

                        return tmp += i;
                    }
            
                    __device__ __host__ bool operator!=(const_reverse_iterator const& other) const
                    {
                        return ptr_ != other.ptr_;
                    }
            
                    __device__ __host__ bool operator==(const_reverse_iterator const& other) const
                    {
                        return !(*this != other);
                    }

                private:
                    T const* ptr_;    
            };

            class reverse_iterator
            {
                public:
                    using pointer = T*;
                    using iterator_category = std::bidirectional_iterator_tag;
                    using difference_type = std::ptrdiff_t;
                    using value_type = T;

                    __device__ __host__ reverse_iterator(T* ptr) : ptr_{ptr}
                    {
                    }
            
                    __device__ __host__ T& operator*()
                    {
                        return *ptr_;
                    }
            
                    __device__ __host__ T const& operator*() const
                    {
                        return *ptr_;
                    }
            
                    __device__ __host__ T* operator->()
                    {
                        return ptr_;
                    }
            
                    __device__ __host__ T const* operator->() const
                    {
                        return ptr_;
                    }
            
                    __device__ __host__ reverse_iterator& operator--()
                    {
                        ++ptr_;
                        return *this;
                    }
            
                    __device__ __host__ reverse_iterator operator--(int)
                    {
                        auto temp(*this);
                        ++ptr_;
                        return temp;
                    }
            
                    __device__ __host__ reverse_iterator& operator++()
                    {
                        --ptr_;
                        return *this;
                    }
            
                    __device__ __host__ reverse_iterator operator++(int)
                    {
                        auto temp(*this);
                        --ptr_;
                        return temp;
                    }
            
                    __device__ __host__ reverse_iterator& operator+=(size_t i)
                    {
                        ptr_ -= i;

                        return *this;
                    }
            
                    __device__ __host__ reverse_iterator operator+(size_t i) const
                    {
                        auto tmp(*this);

                        return tmp -= i;
                    }
            
                    __device__ __host__ reverse_iterator& operator-=(size_t i)
                    {
                        ptr_ += i;

                        return *this;
                    }
            
                    __device__ __host__ reverse_iterator operator-(size_t i) const
                    {
                        auto tmp(*this);

                        return tmp += i;
                    }
            
                    __device__ __host__ bool operator!=(reverse_iterator const& other) const
                    {
                        return ptr_ != other.ptr_;
                    }
            
                    __device__ __host__ bool operator==(reverse_iterator const& other) const
                    {
                        return !(*this != other);
                    }

                    __device__ __host__ operator const_reverse_iterator() const
                    {
                        return const_reverse_iterator(ptr_);
                    }

                private:
                    T* ptr_;    
            };

            __device__ __host__ vector() = default;

            __device__ __host__ vector(size_t count, T const& value = T()) : size_{count}, capacity_{count}
            {
#ifdef __CUDA_ARCH__
                cudaMalloc(&data_, sizeof(T) * count);
#else
                data_ = new T[count];
#endif

                for (size_t i{0}; i < size_; ++i)
                    this->operator[](i) = value;
            }

            template <class InputIt>
            __device__ __host__ vector(InputIt first, InputIt last)
            {
                resize(cu::distance(first, last));
                
                while (first != last)
                {
                    push_back(*first);
                    ++first;
                }
            }

            __device__ __host__ vector(vector const& other) : vector(other.begin(), other.end())
            {
            }

            __device__ __host__ ~vector()
            {
#ifdef __CUDA_ARCH__
                cudaFree(data_);
#else
                delete[] data_;
#endif

                data_ = nullptr;
                size_ = 0;
                capacity_ = 0;
            }

            __device__ __host__ void reserve(size_t capacity)
            {
                if (capacity <= capacity_)
                    return;

                T* d{nullptr};

#ifdef __CUDA_ARCH__
                cudaMalloc(&d, sizeof(T) * capacity);

                memcpy(d, data_, sizeof(T) * size_);

                cudaFree(data_);
#else
                d = new T[capacity];

                std::memcpy(d, data_, sizeof(T) * size_);

                delete[] data_;
#endif

                capacity_ = capacity;

                data_ = d;
            }

            __device__ __host__ void resize(size_t size)
            {
                if (size > capacity_)
                    reserve(size);
                else if (size == capacity_)
                    reserve(2 * size);

                size_ = size;
            }

            __device__ __host__ size_t capacity() const
            {
                return capacity_;
            }

            __device__ __host__ size_t size() const
            {
                return size_;
            }

            __device__ __host__ T const& operator[](size_t i) const
            {
                assert(i < size_);

                return data_[i];
            }

            __device__ __host__ T& operator[](size_t i)
            {
                assert(i < size_);

                return data_[i];
            }

            __device__ __host__ T const& at(size_t i) const
            {
                if (i >= size_)
                    throw std::out_of_range("");

                return data_[i];
            }

            __device__ __host__ T& at(size_t i)
            {
                if (i >= size_)
                    throw std::out_of_range("");

                return data_[i];
            }

            __device__ __host__ T const& front() const
            {
                return this->operator[](0);
            }

            __device__ __host__ T const& back() const
            {
                return this->operator[](size_ - 1);
            }

            __device__ __host__ T const* data() const
            {
                return data_;
            }

            __device__ __host__ T* data()
            {
                return data_;
            }

            __device__ __host__ bool empty() const
            {
                return !size_;
            }

            __device__ __host__ void clear()
            {
                resize(0);
            }

            __device__ __host__ void pop_back()
            {
                resize(size_ - 1);
            }

            __device__ __host__ void push_back(T const& value)
            {
                resize(size_ + 1);

                this->operator[](size_ - 1) = value;
            }

            __device__ __host__ void insert(const_iterator pos, T const& value)
            {
                resize(size_ + 1);

                for (size_t i{size_ - 1}; i > pos; --i)
                    this->operator[](i) = this->operator[](i - 1);

                this->operator[](cu::distance(cbegin(), pos)) = value;
            }

            template <class InputIt>
            __device__ __host__ void insert(const_iterator pos, InputIt first, InputIt last)
            {
                resize(size_ + cu::distance(first, last));

                size_t i{size_ - 1};

                for (size_t j{0}; j < cu::distance(first, last); ++j)
                    this->operator[](j) = this->operator[](j - 1);

                i = cu::distance(cbegin(), pos);

                while (first != last)
                {
                    this->operator[](i) = *first;
                    ++i;
                    ++first;
                }
            }

            __device__ __host__ iterator begin()
            {
                return iterator(data_);
            }

            __device__ __host__ iterator end()
            {
                return iterator(data_ + size_);
            }

            __device__ __host__ const_iterator begin() const
            {
                return const_iterator(data_);
            }

            __device__ __host__ const_iterator end() const
            {
                return iterator(data_ + size_);
            }

            __device__ __host__ const_iterator cbegin() const
            {
                return const_iterator(data_);
            }

            __device__ __host__ const_iterator cend() const
            {
                return iterator(data_ + size_);
            }

            __device__ __host__ reverse_iterator rbegin()
            {
                return reverse_iterator(data_ + size_ - 1);
            }
            
            __device__ __host__ reverse_iterator rend()
            {
                return reverse_iterator(data_ - 1);
            }

            __device__ __host__ const_reverse_iterator rbegin() const
            {
                return const_reverse_iterator(data_ + size_ - 1);
            }
            
            __device__ __host__ const_reverse_iterator rend() const
            {
                return const_reverse_iterator(data_ - 1);
            }

            __device__ __host__ const_reverse_iterator crbegin() const
            {
                return const_reverse_iterator(data_ + size_ - 1);
            }
            
            __device__ __host__ const_reverse_iterator crend() const
            {
                return const_reverse_iterator(data_ - 1);
            }

            __device__ __host__ auto operator<=>(vector const& other) const
            {
                for (size_t i{0}; i < cu::min(size_, other.size_); ++i)
                {
                    if (data_[i] < other.data_[i])
                        return std::strong_ordering::less;
                    if (data_[i] > other.data_[i])
                        return std::strong_ordering::greater;
                }
                
                if (size_ < other.size_)
                    return std::strong_ordering::less;
                else if (size_ > other.size_)
                    return std::strong_ordering::greater;

                return std::strong_ordering::equal;
            }

        private:
            T* data_{nullptr};
            size_t size_{0};
            size_t capacity_{0};
    };

    template <class Vector>
    __device__ __host__ auto begin(Vector& v)
    {
        return v.begin();
    }

    template <class Vector>
    __device__ __host__ auto cbegin(Vector const& v)
    {
        return v.cbegin();
    }

    template <class Vector>
    __device__ __host__ auto end(Vector& v)
    {
        return v.end();
    }

    template <class Vector>
    __device__ __host__ auto cend(Vector const& v)
    {
        return v.cend();
    }

    template <class Vector>
    __device__ __host__ auto rbegin(Vector& v)
    {
        return v.rbegin();
    }

    template <class Vector>
    __device__ __host__ auto rbegin(Vector const& v)
    {
        return v.crbegin();
    }

    template <class Vector>
    __device__ __host__ auto rend(Vector& v)
    {
        return v.rend();
    }

    template <class Vector>
    __device__ __host__ typename Vector::const_reverse_iterator rend(Vector const& v)
    {
        return v.crend();
    }

    template <typename T>
    __device__ __host__ void swap(T& a, T& b)
    {
        auto const tmp(a);
        a = b;
        b = tmp;
    }

    template<class ForwardIt1, class ForwardIt2>
    __device__ __host__ constexpr //< since C++20
    void iter_swap(ForwardIt1 a, ForwardIt2 b)
    {
        cu::swap(*a, *b);
    }

    template <class BidirIt>
    __device__ __host__ constexpr // since C++20
    void reverse(BidirIt first, BidirIt last)
    {
        using iter_cat = typename std::iterator_traits<BidirIt>::iterator_category;
    
        // Tag dispatch, e.g. calling reverse_impl(first, last, iter_cat()),
        // can be used in C++14 and earlier modes.
        if constexpr (std::is_base_of_v<std::random_access_iterator_tag, iter_cat>)
        {
            if (first == last)
                return;
    
            for (--last; first < last; (void)++first, --last)
                cu::iter_swap<BidirIt>(first, last);
        }
        else
            while (first != last && first != --last)
                cu::iter_swap<BidirIt>(first++, last);
    }
}

#endif // CU_VECTOR_CUH

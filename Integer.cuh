#ifndef INTEGER_CUH
#define INTEGER_CUH

/*!
 *  \file Integer.cuh
 *  \brief Provide a class to manage large integer numbers
 *  \author Julien LIVET
 *  \version 1.0
 *  \date 28/12/2024
 */

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstring>
#include <functional>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#if __cplusplus >= 202106L
#include <format>
#endif

#ifdef WITH_GMP
#include <gmpxx.h>
#endif

#ifdef WITH_BOOST
#include <boost/multiprecision/cpp_int.hpp>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reverse.h>

#include "cu/vector.cuh"

using longest_type = uintmax_t;

//#include "primes_3_000_000.h"
#include "primes_100.h"

#define BLOCK_SIZE 1024

template <typename T>
__global__ void Integer_invert(T* a, size_t n)                  \
{                                                               \
    size_t const idx{blockIdx.x * blockDim.x + threadIdx.x};    \
                                                                \
    if (idx < n)                                                \
        a[idx] = ~a[idx];                                       \
}

template <typename T>
__global__ void Integer_setRandom(T* a, size_t n, size_t seed)
{
    size_t const idx{blockIdx.x * blockDim.x + threadIdx.x};
    
    if (idx < n)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        a[idx] = curand(&state);
    }
}

template <>
__global__ void Integer_setRandom(unsigned long long* a, size_t n, size_t seed)
{
    size_t const idx{blockIdx.x * blockDim.x + threadIdx.x};

    if (idx < n)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        a[idx] = curand(&state);
        a[idx] <<= 32;
        a[idx] |= curand(&state);
    }
}

template <typename T, class Vector, typename Enable = void>
class Integer;
/**
template <typename T>
__global__ void Integer_previousPrime(T const* numberData, size_t numberDataSize,
                                      size_t size, bool* divisible)
{
    size_t start{blockIdx.x * blockDim.x + threadIdx.x + 1};


}
    
template <typename T>
__global__ void Integer_nextPrime(T const* numberData, size_t numberDataSize,
                                  size_t size, bool* divisible)
{
    size_t start{blockIdx.x * blockDim.x + threadIdx.x + 1};


}
**/
template <typename T>
__global__ void Integer_isPrime_trialDivision(unsigned int const* primes, size_t primesSize,
                                              T const* numberData, size_t numberDataSize,
                                              T const* sqrtLimitData, size_t sqrtLimitDataSize,
                                              bool* divisible)
{
    size_t const idx{blockIdx.x * blockDim.x + threadIdx.x};

    if (idx < primesSize && !divisible)
    {
        Integer<T, cu::vector<T> > const n(numberData,
                                           numberData + numberDataSize);
        Integer<T, cu::vector<T> > const s(sqrtLimitData,
                                           sqrtLimitData + numberDataSize);
            
        if (primes[idx] <= s && !(n % primes[idx]))
            *divisible = true;
    }
}

template <typename T, class Vector>
__device__ __host__
Integer<T, Vector> reduction(Integer<T, Vector> const& t, Integer<T, Vector> const& R,
                             Integer<T, Vector> const& n, Integer<T, Vector> const& n_)
{
    auto m(t);
    m %= R;
    m *= n_;
    m %= R;
    
    auto x(m);
    x *= n;
    x += t;
    x /= R;

    if (x < n)
        x -= n;
    
    return x;
}

template <typename T, class Vector>
__device__ __host__
Integer<T, Vector> redmulmod(Integer<T, Vector> const& a, Integer<T, Vector> b,
                             Integer<T, Vector> const& n, Integer<T, Vector> const& R,
                             Integer<T, Vector> const& n_, Integer<T, Vector> const& R2modn)
{
    auto const reda(reduction(a * R2modn, R, n, n_));
    auto const redb(reduction(b * R2modn, R, n, n_));
    auto const redc(reduction(reda * redb, R, n, n_));

    return reduction(redc, R, n, n_);
}

template <typename T, class Vector>
__device__ __host__
bool mulmod(Integer<T, Vector> const& a, Integer<T, Vector> b, Integer<T, Vector> const& m)
{
    Integer<T, Vector> x(0);
    auto y(a % m);

    while (b > 0)
    {
        if (b & 1)
        {
            x += y;
            x %= m;
        }

        y <<= 1;
        y %= m;
        b >>= 1;
    }
    
    x %= m;

    return x;
}

template <typename T, class Vector>
__device__ __host__
Integer<T, Vector> modulo(Integer<T, Vector> const& base, Integer<T, Vector> e,
                          Integer<T, Vector> const& m, Integer<T, Vector> const& R,
                          Integer<T, Vector> const& m_, Integer<T, Vector> const& R2modm)
{
    Integer<T, Vector> x(1);
    auto y(base);

    while (e > 0)
    {
        if (e & 1)
        {
            auto const x_(x);
            x = redmulmod(x, y, m, R, m_, R2modm);
            while (x < 0)
                x += m;
            assert(x == (x_ * y) % m);
        }

        auto const y_(y);
        y = redmulmod(y, y, m, R, m_, R2modm);
        while (y < 0)
            y += m;
        assert(y == (y_ * y_) % m);
        e >>= 1;
    }
    
    x %= m;

    return x;
}

template <typename T>
__global__ void Integer_isPrime_millerRabin(T const* numberData, size_t numberDataSize,
                                            T const* sData, size_t sDataSize,
                                            T const* RData, size_t RDataSize,
                                            T const* m_Data, size_t m_DataSize,
                                            T const* R2modmData, size_t R2modmDataSize,
                                            bool* divisible, size_t size)
{
    size_t const idx{blockIdx.x * blockDim.x + threadIdx.x};

    if (idx < size && !divisible)
    {
        Integer<T, cu::vector<T> > const number(numberData, numberData + numberDataSize);
        auto const n(number + 1);
        auto a(n);
        a.template setRandom<std::random_device>();
        a.setPositive();
        a %= number;
        ++a;

        Integer<T, cu::vector<T> > const R(RData, RData + RDataSize);
        Integer<T, cu::vector<T> > const m_(m_Data, m_Data + m_DataSize);
        Integer<T, cu::vector<T> > const R2modm(R2modmData, R2modmData + R2modmDataSize);

        Integer<T, cu::vector<T> > temp(sData, sData + sDataSize);
        auto mod{modulo(a, temp, n, R, m_, R2modm)};

        while (temp != number && !mod && mod != number)
        {
            mod = mulmod(mod, mod, n);
            temp <<= 1;
        }

        if (mod != number && !(temp & 1))
            *divisible = true;
    }
}

template <typename T, class Vector>
class Integer<T, Vector, typename std::enable_if<std::is_unsigned<T>::value && std::is_same<T, typename Vector::value_type>::value >::type>
{
    public:
        CONSTEXPR Integer() = default;

        template <typename S, std::enable_if_t<std::is_standard_layout_v<S> && std::is_trivial_v<S> >* = nullptr>
        CONSTEXPR Integer(S n) : isPositive_{n >= 0}
        {
            bits_.reserve(std::max(longest_type{1}, longest_type{sizeof(S) / sizeof(T)}));

            if (n < 0)
                n = -n;

            if (sizeof(T) == sizeof(S))
                bits_.push_back(n);
            else
            {
                auto const shift{longest_type{1} << std::min(sizeof(T), sizeof(S)) * 8};

                for (size_t i{0}; i < bits_.capacity(); ++i)
                {
                    bits_.push_back(n % shift);
                    n /= shift;
                }

                cu::reverse(cu::begin(bits_), cu::end(bits_));
            }

            adjust();
        }

        CONSTEXPR Integer(Vector const& bits, bool isPositive = true) : isPositive_{isPositive}, bits_{bits}
        {
            adjust();
        }

        template <size_t N>
        CONSTEXPR Integer(std::bitset<N> const& bits, bool isPositive = true) : isPositive_{isPositive}
        {
            setBits(0, bits);
        }

        CONSTEXPR Integer(std::initializer_list<T> const& bits, bool isPositive = true) : isPositive_{isPositive}
        {
            bits_.reserve(bits.size());

            for (auto const& b : bits)
                bits_.push_back(b);

            adjust();
        }

        template <class InputIt>
        __device__ __host__ CONSTEXPR Integer(InputIt begin, InputIt end, bool isPositive = true) : isPositive_{isPositive}, bits_{begin, end}
        {
            adjust();
        }

#ifdef WITH_GMP
        CONSTEXPR Integer(mpz_class const& n) : Integer(n.get_str(2), 2)
        {
        }
        
        CONSTEXPR Integer(mpz_t const n)
        {
            char* s{nullptr};
            
            gmp_asprintf(&s, "%Zd", n);
            
            *this = Integer(s, 10);
            
            free(s);
        }
#endif

#ifdef WITH_BOOST
        CONSTEXPR Integer(boost::multiprecision::cpp_int const& n)
        {
            std::ostringstream oss;
            
            oss << n;
            
            *this = Integer(oss.str(), 10);
        }
#endif

        CONSTEXPR Integer(char const* n, size_t base = 0) : Integer(std::string{n}, base)
        {
        }

        Integer(std::string n, size_t base = 0)
        {
            n.erase(std::remove_if(n.begin(), n.end(), isspace), n.end());
            n.erase(std::remove(n.begin(), n.end(), '\''), n.end());

            auto it{n.begin()};

            if (*it == '-')
            {
                isPositive_ = false;
                ++it;
            }

            if (!base)
            {
                auto s{std::string{it, n.end()}.substr(0, 2)};
                std::transform(s.begin(), s.end(), s.begin(),
                               [] (unsigned char c) { return std::tolower(c); });

                if (s[0] == 'b' || s == "0b")
                    base = 2;
                else if (s[0] == 'o' || s == "0o")
                    base = 8;
                else if (s[0] == 'x' || s == "0x")
                    base = 16;
                else
                    base = 10;
            }

            assert(2 <= base && base <= 62);

            std::string str{it, n.end()};
            std::transform(str.begin(), str.end(), str.begin(),
                           [] (unsigned char c) { return std::tolower(c); });

            if (str == "nan")
                setNan();
            else if (str == "inf")
                setInfinity();
            else
            {
                auto const isPositive{isPositive_};

                if (base == 2)
                {
                    if (std::tolower(*it) == 'b')
                        ++it;
                    else if (str.substr(0, 2) == "0b")
                        it += 2;

                    *this = 0;

                    while (it != n.end())
                    {
                        if (*it == '1')
                        {
                            *this <<= 1;
                            *this |= 1;
                        }
                        else if (*it == '0')
                            *this <<= 1;

                        ++it;
                    }
                }
                else if (base == 8)
                {
                    if (str[0] == 'o')
                        ++it;
                    else if (str.substr(0, 2) == "0o")
                        it += 2;

                    auto otherIt{n.rbegin()};
                    Integer p(1);

                    while (otherIt.base() != it)
                    {
                        if ('0' <= *otherIt && *otherIt <= '7')
                        {
                            *this += (*otherIt - '0') * p;
                            p *= base;
                        }

                        ++otherIt;
                    }
                }
                else if (base <= 10)
                {
                    auto otherIt{n.rbegin()};
                    Integer p(1);

                    while (otherIt.base() != it)
                    {
                        if ('0' <= *otherIt && *otherIt <= static_cast<char>('0' + base))
                        {
                            *this += (*otherIt - '0') * p;
                            p *= base;
                        }

                        ++otherIt;
                    }
                }
                else if (base < 16)
                {
                    auto otherIt{n.rbegin()};
                    Integer p(1);

                    while (otherIt.base() != it)
                    {
                        if ('0' <= *otherIt && *otherIt <= '9')
                        {
                            *this += (*otherIt - '0') * p;
                            p *= base;
                        }
                        else if ('a' <= std::tolower(*otherIt) && std::tolower(*otherIt) <= static_cast<char>('a' + base - 10))
                        {
                            *this += (*otherIt - 'a' + 10) * p;
                            p *= base;
                        }

                        ++otherIt;
                    }
                }
                else if (base == 16)
                {
                    if (str[0] == 'x')
                        ++it;
                    else if (str.substr(0, 2) == "0x")
                        it += 2;

                    auto otherIt{n.rbegin()};
                    Integer p(1);

                    while (otherIt.base() != it)
                    {
                        if ('0' <= *otherIt && *otherIt <= '9')
                        {
                            *this += (*otherIt - '0') * p;
                            p *= base;
                        }
                        else if ('a' <= std::tolower(*otherIt) && std::tolower(*otherIt) <= 'f')
                        {
                            *this += (*otherIt - 'a' + 10) * p;
                            p *= base;
                        }

                        ++otherIt;
                    }
                }
                else// if (base <= 62)
                {
                    auto otherIt{n.rbegin()};
                    Integer p(1);

                    while (otherIt.base() != it)
                    {
                        if ('0' <= *otherIt && *otherIt <= '9')
                        {
                            *this += (*otherIt - '0') * p;
                            p *= base;
                        }
                        else if ('a' <= *otherIt && *otherIt <= 'z')
                        {
                            *this += (*otherIt - 'a' + 10) * p;
                            p *= base;
                        }
                        else if ('A' <= *otherIt && *otherIt <= 'Z')
                        {
                            *this += (*otherIt - 'A' + 36) * p;
                            p *= base;
                        }

                        ++otherIt;
                    }
                }

                isPositive_ = isPositive;
            }

            adjust();
        }

        template <typename S>
        CONSTEXPR Integer(Integer<S, Vector> const& other)
        {
            bits_ = Vector(other.dataSize() / sizeof(T) + (other.dataSize() % sizeof(T) ? 1 : 0), 0);
            std::copy(other.data(), other.data() + other.dataSize(), reinterpret_cast<char*>(bits_.begin()) + bits_.size() * sizeof(T) - other.dataSize());
        }

        CONSTEXPR bool isPositive() const noexcept
        {
            return isPositive_;
        }

        CONSTEXPR bool isNegative() const noexcept
        {
            return !isPositive_;
        }

        CONSTEXPR auto const& bits() const noexcept
        {
            return bits_;
        }

        CONSTEXPR void invert() noexcept
        {
            T* a(nullptr);
            cudaMalloc(&a, sizeof(T) * size());
            cudaMemcpy(a, bits_.data(), sizeof(T) * bits_.size(), cudaMemcpyHostToDevice);

            size_t const blockSize{BLOCK_SIZE};
            size_t const gridSize{(bits_.size() + blockSize) / blockSize};
            
            Integer_invert<T><<<gridSize, blockSize>>>(a, bits_.size());
            
            cudaDeviceSynchronize();
            
            cudaMemcpy(bits_.data(), a, sizeof(T) * bits_.size(), cudaMemcpyDeviceToHost);

            cudaFree(a);
            
            if (autoAdjust_)
                adjust();
        }

        CONSTEXPR Integer& operator*=(Integer const& other)
        {
            auto const lhs(*this);
            auto const rhs(other);

            if (other.isNegative())
            {
                *this = -*this;

                return *this *= -other;
            }
            else if (isNan() || other.isNan())
                setNan();
            else if (!*this || !other)
                *this = 0;
            else if (isInfinity() || other.isInfinity())
            {
                setInfinity();

                if (other.isNegative())
                    isPositive_ = !isPositive_;
            }
            else
            {
                if (isPositive_ && other.isPositive_)
                {
                    if (this->template fits<longest_type>() && other.template fits<longest_type>())
                    {
                        auto const a(this->template cast<longest_type>());
                        auto const b(other.template cast<longest_type>());
                        auto const ab(a * b);

                        if (ab / b == a)
                            *this = ab;
                        else
                        {
                            auto number{[] (longest_type n) -> size_t
                                        {
                                            size_t number(0);

                                            while (n)
                                            {
                                                ++number;
                                                n >>= 1;
                                            }

                                            return number;
                                        }
                            };

                            //Karatsuba algorithm
                            size_t n{std::max(number(a), number(b))};
                            if (n % 2)
                                ++n;
                            size_t const m{n / 2};
                            Integer const x0((static_cast<longest_type>(~longest_type{0}) >> (sizeof(longest_type) * 8 - m)) & a);
                            Integer const x1((((static_cast<longest_type>(~longest_type{0}) >> (sizeof(longest_type) * 8 - m)) << m) & a) >> m);
                            Integer const y0((static_cast<longest_type>(~longest_type{0}) >> (sizeof(longest_type) * 8 - m)) & b);
                            Integer const y1((((static_cast<longest_type>(~longest_type{0}) >> (sizeof(longest_type) * 8 - m)) << m) & b) >> m);

                            assert(*this == ((x1 << m) | x0));
                            assert(other == ((y1 << m) | y0));

                            auto const z0(x0 * y0);
                            auto const z1(x1 * y0 + x0 * y1);
                            auto const z2(x1 * y1);

#ifdef WITH_GMP
                            assert(z0 == mpz_class{x0.template cast<mpz_class>() * y0.template cast<mpz_class>()});
                            assert(z1 == mpz_class{x1.template cast<mpz_class>() * y0.template cast<mpz_class>() + x0.template cast<mpz_class>() * y1.template cast<mpz_class>()});
                            assert(z2 == mpz_class{x1.template cast<mpz_class>() * y1.template cast<mpz_class>()});

                            mpz_class _2_2m{2};
                            mpz_pow_ui(_2_2m.get_mpz_t(), _2_2m.get_mpz_t(), 2 * m);
                            mpz_class _2_m{2};
                            mpz_pow_ui(_2_m.get_mpz_t(), _2_m.get_mpz_t(), m);
                            mpz_class const n1_{lhs.template cast<mpz_class>() * rhs.template cast<mpz_class>()};
                            mpz_class const n2_{z2.template cast<mpz_class>() * _2_2m
                                                + z1.template cast<mpz_class>() * _2_m
                                                + z0.template cast<mpz_class>()};
                            assert(n1_ == n2_);
#endif

                            //xy = z2 * 2^(2 * m) + z1 * 2^m + z0
                            *this = z0 + (z1 << m) + (z2 << 2 * m);
                        }
                    }
                    else if (!(rhs & 1))
                    {
                        auto r(rhs);
                        Integer shift(0);

                        while (!(r & 1))
                        {
                            r >>= 1;
                            ++shift;
                        }

                        *this <<= shift;
                        *this *= r;
                    }
                    else
                    {
                        //Karatsuba algorithm
                        //x = x1 * 2^m + x0
                        //y = y1 * 2^m + y0
                        size_t n1{number() / (sizeof(T) * 8)};
                        if (number() % (sizeof(T) * 8))
                            ++n1;
                        if (n1 % 2)
                            ++n1;
                        size_t n2{other.number() / (sizeof(T) * 8)};
                        if (other.number() % (sizeof(T) * 8))
                            ++n2;
                        if (n2 % 2)
                            ++n2;
                        size_t const n{std::max(n1, n2)};
                        size_t const m{n / 2};
                        Vector bits(m, T{0});
                        std::copy(bits_.rbegin(),
                                  bits_.rbegin() + std::min(bits_.size(), m),
                                  bits.rbegin());
                        Integer const x0(bits);
                        bits = Vector(m, T{0});
                        std::copy(bits_.rbegin() + m,
                                  bits_.rbegin() + std::min(bits_.size(), 2 * m),
                                  bits.rbegin());
                        Integer const x1(bits);
                        bits = Vector(m, T{0});
                        std::copy(other.bits_.rbegin(),
                                  other.bits_.rbegin() + std::min(other.bits_.size(), m),
                                  bits.rbegin());
                        Integer const y0(bits);
                        bits = Vector(m, T{0});
                        std::copy(other.bits_.rbegin() + m,
                                  other.bits_.rbegin() + std::min(other.bits_.size(), 2 * m),
                                  bits.rbegin());
                        Integer const y1(bits);

                        assert(*this == ((x1 << (m * sizeof(T) * 8)) | x0));
                        assert(other == ((y1 << (m * sizeof(T) * 8)) | y0));

                        auto const z0(x0 * y0);
                        auto const z1(x1 * y0 + x0 * y1);
                        auto const z2(x1 * y1);

#ifdef WITH_GMP
                        assert(z0 == mpz_class{x0.template cast<mpz_class>() * y0.template cast<mpz_class>()});
                        assert(z1 == mpz_class{x1.template cast<mpz_class>() * y0.template cast<mpz_class>() + x0.template cast<mpz_class>() * y1.template cast<mpz_class>()});
                        assert(z2 == mpz_class{x1.template cast<mpz_class>() * y1.template cast<mpz_class>()});
#endif

                        //o = m * 8 * sizeof(T)
                        //xy = z2 * 2^(2 * o) + z1 * 2^o + z0

#ifdef WITH_GMP
                        size_t const o{m * 8 * sizeof(T)};
                        mpz_class _2_2o{2};
                        mpz_pow_ui(_2_2o.get_mpz_t(), _2_2o.get_mpz_t(), 2 * o);
                        mpz_class _2_o{2};
                        mpz_pow_ui(_2_o.get_mpz_t(), _2_o.get_mpz_t(), o);
                        mpz_class const n1_{lhs.template cast<mpz_class>() * rhs.template cast<mpz_class>()};
                        mpz_class const n2_{z2.template cast<mpz_class>() * _2_2o
                                            + z1.template cast<mpz_class>() * _2_o
                                            + z0.template cast<mpz_class>()};
                        assert(n1_ == n2_);
#endif
                        *this = z0;
                        bits = Vector(z1.size() + m, T{0});
                        std::copy(z1.bits_.rbegin(), z1.bits_.rend(), bits.rbegin() + m);
                        *this += Integer(bits);
                        bits = Vector(z2.size() + 2 * m, T{0});
                        std::copy(z2.bits_.rbegin(), z2.bits_.rend(), bits.rbegin() + 2 * m);
                        *this += Integer(bits);
                    }
                }
                else
                {
                    *this *= -other;
                    *this = -*this;
                }
            }

            if (autoAdjust_)
                adjust();

#ifdef WITH_GMP
            assert(*this == mpz_class{lhs.template cast<mpz_class>() * rhs.template cast<mpz_class>()});
#endif

            return *this;
        }

        CONSTEXPR Integer& operator+=(Integer const& other)
        {
            auto const lhs(*this);
            auto const rhs(other);
            
            if (isNan() || other.isNan())
                setNan();
            else if (isInfinity() || other.isInfinity())
            {

                if ((isPositive() && other.isNegative())
                    || (isNegative() && other.isPositive()))
                    setNan();
                else
                {
                    if (other.isInfinity())
                        isPositive_ = other.isPositive_;

                    setInfinity();
                }
            }
            else
            {
                if ((isPositive() && other.isPositive())
                    || (isNegative() && other.isNegative()))
                {
                    T carry{0};
                    auto const& a(bits_);
                    auto const& b(other.bits_);
                    size_t const n{std::max(a.size(), b.size())};
                    Vector result;
                    result.reserve(n);

                    for (size_t i{0}; i < n; ++i)
                    {
                        auto const bit_a{(i < a.size()) ? a[a.size() - 1 - i] : T{0}};
                        auto const bit_b{(i < b.size()) ? b[b.size() - 1 - i] : T{0}};
                        auto const sum{static_cast<T>(bit_a + bit_b + carry)};

                        carry = (sum < bit_a || sum < bit_b);

                        result.push_back(sum);
                    }

                    if (carry)
                        result.push_back(T{1});

                    cu::reverse(cu::begin(result), cu::end(result));

                    bits_ = result;
                }
                else
                {
                    auto otherBits(other.bits_);

                    if (isPositive())
                    {
                        if (*this < -other)
                        {
                            isPositive_ = false;
                            otherBits = bits_;
                            bits_ = other.bits_;
                        }
                    }
                    else
                    {
                        if (*this > -other)
                        {
                            isPositive_ = true;
                            otherBits = bits_;
                            bits_ = other.bits_;
                        }
                    }

                    auto const& a(bits_);
                    auto const& b(otherBits);
                    size_t const n{std::max(a.size(), b.size())};
                    Vector result;
                    result.reserve(n);

                    for (size_t i{n - 1}; i <= n - 1; --i)
                    {
                        auto const ia{a.size() - 1 - i};
                        auto const ib{b.size() - 1 - i};

                        auto const bit_a{ia < a.size() ? a[ia] : T{0}};
                        auto const bit_b{ib < b.size() ? b[ib] : T{0}};

                        auto bit_result{static_cast<T>(bit_a - bit_b)};

                        if (bit_a < bit_b)
                        {
                            for (auto it{result.rbegin()}; it != result.rend(); ++it)
                            {
                                bool const stop{*it > 0};

                                *it -= 1;

                                if (stop)
                                    break;
                            }

                            bit_result = static_cast<T>(-1) - bit_b;
                            bit_result += bit_a + 1;
                        }

                        result.push_back(bit_result);
                    }

                    bits_ = result;
                }
            }

            if (autoAdjust_)
                adjust();

#ifdef WITH_GMP
            assert(this->template cast<mpz_class>() == lhs.template cast<mpz_class>() + rhs.template cast<mpz_class>());
#endif

            return *this;
        }

        CONSTEXPR Integer& operator-=(Integer const& other)
        {
            auto const lhs(*this);
            auto const rhs(other);

            *this += -other;

#ifdef WITH_GMP
            assert(this->template cast<mpz_class>() == lhs.template cast<mpz_class>() - rhs.template cast<mpz_class>());
#endif

            return *this;
        }

        CONSTEXPR Integer& operator/=(Integer const& other)
        {
            auto const lhs(*this);
            auto const rhs(other);

            if (other.isNegative())
            {
                *this = -*this;

                return *this /= -other;
            }

            auto const n(*this);

            if (!other || other.isNan())
                setNan();
            else if (other.isInfinity())
                *this = 0;
            else
            {
                if (abs() < other.abs())
                    *this = 0;
                else if (isPositive_ && other.isPositive_)
                {
                    if (this->template fits<longest_type>() && other.template fits<longest_type>())
                        *this = this->template cast<longest_type>() / other.template cast<longest_type>();
                    else if (!(rhs & 1))
                    {
                        auto r(rhs);
                        Integer shift(0);

                        while (!(r & 1))
                        {
                            r >>= 1;
                            ++shift;
                        }

                        *this >>= shift;
                        *this /= r;
                    }
                    else
                        *this = computeQuotientBurnikelZiegler(*this, other);
                }
                else
                {
                    *this /= -other;
                    *this = -*this;
                }
            }

            assert(abs() <= n.abs());

#ifdef WITH_GMP
        assert(*this == mpz_class{lhs.template cast<mpz_class>() / rhs.template cast<mpz_class>()});
#endif

            return *this;
        }

        __device__ __host__ CONSTEXPR Integer& operator%=(Integer const& other)
        {
            auto const lhs(*this);
            auto const rhs(other);

            if (!other || other.isNan() || other.isInfinity())
                setNan();
            else
            {
                if ((isPositive_ && other.isPositive_) ||
                    (!isPositive_ && !other.isPositive_))
                {
                    if (other == 1)
                        *this = 0;
                    else if (other == 2)
                        *this &= 1;
                    else if (abs().template fits<longest_type>() && other.abs().template fits<longest_type>())
                    {
                        auto const isPositive{isPositive_};

                        *this = abs().template cast<longest_type>() % other.abs().template cast<longest_type>();

                        isPositive_ = isPositive;
                    }
                    else
                    {
                        auto const qr{computeQrBurnikelZiegler(*this, other)};

                        assert(*this == qr.first * rhs + qr.second);

                        *this = qr.second;
                    }
                }
                else
                {
                    auto const qr{computeQrBurnikelZiegler(*this, other)};

                    assert(*this == qr.first * rhs + qr.second);

                    *this = qr.second;
                }
            }

#ifdef WITH_GMP
            if (lhs > 0 && rhs > 0)
                assert(*this == mpz_class{lhs.template cast<mpz_class>() % rhs.template cast<mpz_class>()});
#endif

            assert(abs() < rhs.abs());

            return *this;
        }

        CONSTEXPR Integer& operator<<=(Integer other)
        {
            assert(other >= 0);

            if (!*this || !other)
                return *this;
            else if (isNan() || other.isNan() || isInfinity() || other.isInfinity())
            {
                setNan();
                
                return *this;
            }

            auto const s{static_cast<unsigned short>(sizeof(T) * 8)};
            auto const n(other / s);

            Vector const v(n.template cast<longest_type>(), T{0});

            bits_.insert(cu::end(bits_), cu::begin(v), cu::end(v));

            other -= n * s;

            Vector bits(bits_.size() + 1, T{0});

            std::copy(bits_.rbegin(), bits_.rend(), bits.rbegin());

            bits_ = bits;

            auto const shift{other.template cast<longest_type>()};

            if (shift)
            {
                for (auto it{cu::begin(bits_) + 1}; it != cu::end(bits_); ++it)
                {
                    longest_type const s{sizeof(T) * 8};
                    
                    if ((*it >> (s - shift)))
                        *(it - 1) |= (*it >> (s - shift));

                    *it <<= shift;
                }
            }

            if (autoAdjust_)
                adjust();

            return *this;
        }

        CONSTEXPR Integer& operator>>=(Integer other)
        {
            assert(other >= 0);

            if (!*this || !other)
                return *this;
            else if (isNan() || other.isNan() || isInfinity())
            {
                setNan();
                
                return *this;
            }
            else if (other.isInfinity())
            {
                if (other < 0)
                    setNan();
                else
                    *this = 0;
                
                return *this;
            }

            auto const s{static_cast<unsigned short>(sizeof(T) * 8)};
            auto const n(other / s);

            if (bits_.size() < n.template cast<longest_type>())
            {
                bits_ = Vector{T{0}};

                return *this;
            }

            bits_.resize(bits_.size() - n.template cast<longest_type>());

            other -= n * s;

            auto const shift{other.template cast<longest_type>()};

            if (shift)
            {
                for (auto it{bits_.rbegin()}; it != bits_.rend(); ++it)
                {
                    *it >>= shift;

                    if (it != bits_.rend() - 1 && (*(it + 1) & ((longest_type{1} << shift) - 1)))
                        *it |= (*(it +  1) & ((longest_type{1} << shift) - 1)) << (sizeof(T) * 8 - shift);
                }
            }

            if (autoAdjust_)
                adjust();

            return *this;
        }

        CONSTEXPR bool operator>=(Integer const& other) const
        {
            return !operator<(other);
        }

        CONSTEXPR bool operator>(Integer const& other) const
        {
            if (!isPositive_ && other.isPositive_)
                return false;
            else if (isNan() || other.isNan())
                return false;
            else if (other.isInfinity())
            {
                if (other.isNegative())
                {
                    if (isInfinity() && isNegative())
                        return false;
                    else
                        return true;
                }
                else
                    return false;
            }

            Vector a(std::max(bits_.size(), other.bits_.size()), T{0});
            Vector b(a);

            std::copy(bits_.rbegin(), bits_.rend(), a.rbegin());
            std::copy(other.bits_.rbegin(), other.bits_.rend(), b.rbegin());

            auto const great{a > b};

            return isPositive_ ? great : !great;
        }

        CONSTEXPR bool operator<=(Integer const& other) const
        {
            return !operator>(other);
        }

        CONSTEXPR bool operator<(Integer const& other) const
        {
            if (isPositive_ && !other.isPositive_)
                return false;
            else if (isNan() || other.isNan())
                return false;
            else if (other.isInfinity())
            {
                if (other.isPositive())
                {
                    if (isInfinity() && isPositive())
                        return false;
                    else
                        return true;
                }
                else
                    return false;
            }

            Vector a(std::max(bits_.size(), other.bits_.size()), T{0});
            Vector b(a);

            std::copy(bits_.rbegin(), bits_.rend(), a.rbegin());
            std::copy(other.bits_.rbegin(), other.bits_.rend(), b.rbegin());

            auto const less{a < b};

            return isPositive_ ? less : !less;
        }

        CONSTEXPR bool operator==(Integer const& other) const noexcept
        {
            if (isNan() && other.isNan())
                return true;
            else if (isNan() || other.isNan())
                return false;
            else if (isInfinity() && other.isInfinity())
                return (isPositive() == other.isPositive());
            else if (isInfinity() || other.isInfinity())
                return false;
            
            if (bits_.size() != other.bits_.size())
            {
                if (bits_.size() > other.bits_.size())
                {
                    for (size_t i{0}; i < bits_.size() - other.bits_.size(); ++i)
                    {
                        if (bits_[i])
                            return false;
                    }
                }
                else
                {
                    for (size_t i{0}; i < other.bits_.size() - bits_.size(); ++i)
                    {
                        if (other.bits_[i])
                            return false;
                    }
                }
            }

            bool zero{true};

            auto it1{bits_.rbegin()};
            auto it2{other.bits_.rbegin()};

            for (size_t i{0}; i < std::min(bits_.size(), other.bits_.size()); ++i)
            {
                if (*it1 != *it2)
                    return false;

                if (*it1)
                    zero = false;

                ++it1;
                ++it2;
            }

            if (isPositive_ != other.isPositive_ && !zero)
                return false;

            return true;
        }

        template <typename S>
        CONSTEXPR bool operator==(S const& other) const
        {
            return *this == Integer(other);
        }

        __device__ __host__ CONSTEXPR bool operator!=(Integer const& other) const
        {
            return !operator==(other);
        }

        template <typename S>
        CONSTEXPR bool operator!=(S const& other) const
        {
            return *this != Integer(other);
        }

        CONSTEXPR Integer operator-() const
        {
            auto x(*this);

            x.isPositive_ = !x.isPositive_;

            return x;
        }

        CONSTEXPR Integer operator~() const
        {
            auto x(*this);

            x.invert();

            return x;
        }

        CONSTEXPR operator bool() const noexcept
        {
            return !!*this;
        }

        __device__ __host__ CONSTEXPR bool operator!() const noexcept
        {
            for (auto const& b : bits_)
            {
                if (b)
                    return false;
            }

            return true;
        }

        CONSTEXPR Integer& operator--()
        {
            return *this -= 1;
        }

        CONSTEXPR Integer operator--(int)
        {
            auto x(*this);

            operator--();

            return x;
        }

        __device__ __host__ CONSTEXPR Integer& operator++()
        {
            return *this += 1;
        }

        CONSTEXPR Integer operator++(int)
        {
            auto x(*this);

            operator++();

            return x;
        }

        template <typename S>
        CONSTEXPR Integer& operator+=(S const& other)
        {
            return *this += Integer(other);
        }

        template <typename S>
        CONSTEXPR Integer& operator-=(S const& other)
        {
            return *this -= Integer(other);
        }

        template <typename S>
        CONSTEXPR Integer& operator/=(S const& other)
        {
            return *this /= Integer(other);
        }

        template <typename S>
        CONSTEXPR Integer& operator*=(S const& other)
        {
            return *this *= Integer(other);
        }

        template <typename S>
        CONSTEXPR Integer& operator%=(S const& other)
        {
            return *this %= Integer(other);
        }

        template <typename S>
        CONSTEXPR Integer& operator>>=(S const& other)
        {
            return *this >>= Integer(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator<<=(S const& other)
        {
            return *this <<= Integer(other);
        }

        CONSTEXPR Integer& operator&=(Integer const& other)
        {
            Vector bits(std::max(bits_.size(), other.bits_.size())
                                  - std::min(bits_.size(), other.bits_.size()), 0);

            if (bits_.size() > other.bits_.size())
                bits.insert(cu::end(bits), cu::begin(other.bits_), cu::end(other.bits_));
            else
                bits.insert(cu::end(bits), cu::begin(bits_), cu::end(bits_));

            Vector const& otherBits(bits_.size() > other.bits_.size() ? bits_ : other.bits_);

            for (size_t i{0}; i < std::min(bits_.size(), other.bits_.size()); ++i)
                *(bits.rbegin() + i) &= *(otherBits.rbegin() + i);

            bits_ = bits;

            if (autoAdjust_)
                adjust();

            return *this;
        }

        CONSTEXPR Integer& operator|=(Integer const& other)
        {
            Vector bits(bits_.size() > other.bits_.size() ? bits_ : other.bits_);
            Vector const& otherBits(bits_.size() > other.bits_.size() ? other.bits_ : bits_);

            for (size_t i{0}; i < otherBits.size(); ++i)
                *(bits.rbegin() + i) |= *(otherBits.rbegin() + i);

            bits_ = bits;

            if (autoAdjust_)
                adjust();

            return *this;
        }

        template <typename S>
        CONSTEXPR Integer& operator|=(S const& other)
        {
            return *this |= Integer(other);
        }

        template <typename S>
        CONSTEXPR Integer& operator&=(S const& other)
        {
            return *this &= Integer(other);
        }

        template <typename S>
        CONSTEXPR Integer& operator^=(S const& other)
        {
            return *this ^= Integer(other);
        }

        CONSTEXPR Integer& operator^=(Integer const& other)
        {
            Vector bits(bits_.size() > other.bits_.size() ? bits_ : other.bits_);
            Vector const otherBits(bits_.size() > other.bits_.size() ? other.bits_ : bits_);

            for (size_t i{0}; i < std::max(bits.size(), otherBits.size())
                                        - std::min(bits.size(), otherBits.size()); ++i)
                *(bits.rbegin() + i) ^= 0;

            for (size_t i{0}; i < otherBits.size(); ++i)
                *(bits.rbegin() + i) ^= *(otherBits.rbegin() + i);

            bits_ = bits;

            if (autoAdjust_)
                adjust();

            return *this;
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator=(S const& other)
        {
            return *this = Integer(other);
        }

        std::string toString(size_t base = 10, bool showBase = true) const
        {
            assert(2 <= base && base <= 62);

            std::string s;

            if (isNan_)
            {
                if (!isPositive_)
                    s = '-' + s;

                s += "nan";

                return s;
            }
            else if (isInfinity_)
            {
                if (!isPositive_)
                    s = '-' + s;

                s += "inf";

                return s;
            }

            if (base == 2)
            {
                s.reserve(s.size() + 2 + bits_.size() * sizeof(T) * 8);

#if __cplusplus >= 202106L
                switch (sizeof(T))
                {
                    case 1: //unsigned char
                        for (auto const& b : bits_)
                            s += std::format("{:08b}", b);
                        break;

                    case 2: //unsigned short
                        for (auto const& b : bits_)
                            s += std::format("{:016b}", b);
                        break;

                    case 4: //unsigned int, unsigned long
                        for (auto const& b : bits_)
                            s += std::format("{:032b}", b);
                        break;

                    case 8: //unsigned long long
                        for (auto const& b : bits_)
                            s += std::format("{:064b}", b);
                        break;

                    case 16:
                        for (auto const& b : bits_)
                            s += std::format("{:0128b}", b);
                        break;
                }
#else
                for (auto it{bits_.rbegin()}; it != bits_.rend(); ++it)
                {
                    auto b{*it};

                    for (size_t i{0}; i < sizeof(T) * 8; ++i)
                    {
                        s = (b & 1 ? '1' : '0') + s;
                        b >>= 1;
                    }
                }
#endif

                if (showBase)
                    s = "0b" + s;
            }
            else if (base == 8)
            {
#if __cplusplus >= 202106L
                if (bits_.size() == 1)
                    s = std::format("{:o}", bits_.back());
                else
#else
                {
                    auto number(abs());

                    if (!number)
                        s = "0";

                    while (number)
                    {
                        auto const tmp(number % 8);
                        s = std::to_string(tmp.template cast<short>()) + s;
                        number /= 8;
                    }
                }
#endif

                if (showBase)
                    s = "0o" + s;
            }
            else if (base == 10)
            {
                auto number(abs());

                if (bits_.size() == 1)
                    s = std::to_string(bits_.back());
                else
                {
                    if (!number)
                        s = "0";

                    auto const n{static_cast<T>(std::log10(static_cast<T>(~T{0})))};
                    T const b(pow(T{10}, n));

                    while (number)
                    {
                        auto const tmp(number % b);
                        std::ostringstream oss;
                        oss << std::setw(n) << std::setfill('0') << tmp.template cast<longest_type>();
                        s = oss.str() + s;
                        number /= b;
                    }

                    size_t i{0};

                    while (s[i] == '0' && i != s.size())
                        ++i;

                    if (i == s.size())
                        i = s.size() - 1;

                    s = s.substr(i);
                }
            }
            else if (2 < base && base < 16)
            {
                auto number(abs());

                if (bits_.size() == 1)
                    s = std::to_string(bits_.back());
                else
                {
                    if (!number)
                        s = "0";

                    while (number)
                    {
                        auto const tmp(number % static_cast<unsigned char>(base));
                        s = std::to_string(tmp.template cast<short>()) + s;
                        number /= static_cast<unsigned char>(base);
                    }
                }
            }
            else if (base == 16)
            {
                auto number(abs());
#if __cplusplus >= 202106L
                switch (sizeof(T))
                {
                    case 1: //unsigned char
                        for (auto const& b : bits_)
                            s += std::format("{:02x}", b);
                        break;

                    case 2: //unsigned short
                        for (auto const& b : bits_)
                            s += std::format("{:04x}", b);
                        break;

                    case 4: //unsigned int, unsigned long
                        for (auto const& b : bits_)
                            s += std::format("{:08x}", b);
                        break;

                    case 8: //unsigned long long
                        for (auto const& b : bits_)
                            s += std::format("{:016x}", b);
                        break;

                    case 16:
                        for (auto const& b : bits_)
                            s += std::format("{:032x}", b);
                        break;
                }

                if (bits_.size() == 1)
                    s = std::format("{:x}", bits_.back());
                else
#else
                {
                    if (!number)
                        s = "0";

                    while (number)
                    {
                        auto const tmp((number % 16).template cast<unsigned char>());
                        if (tmp < 10)
                            s = std::to_string(tmp) + s;
                        else
                            s = (char)('a' + tmp - 10) + s;
                        number /= 16;
                    }
                }
#endif

                if (showBase)
                    s = "0x" + s;
            }
            else if (base <= 62)
            {
                auto number(abs());

                if (!number)
                    s = "0";

                while (number)
                {
                    auto const tmp((number % base).template cast<unsigned char>());
                    if (tmp < 10)
                        s = std::to_string(tmp) + s;
                    else if (tmp - 10 < 26)
                        s = (char)('a' + tmp - 10) + s;
                    else
                        s = (char)('A' + tmp - 36) + s;
                    number /= base;
                }
            }

            if (!bits_.size() && (!s.empty() && s.back() != '0'))
                s += '0';

            if (!isPositive_)
                s = '-' + s;

            return s;
        }

        CONSTEXPR operator char() const noexcept
        {
            return cast<char>();
        }

        CONSTEXPR operator unsigned char() const noexcept
        {
            return cast<unsigned char>();
        }

        CONSTEXPR operator short() const noexcept
        {
            return cast<short>();
        }

        template <typename S>
        CONSTEXPR S cast() const
        {
            S n{0};

            size_t const iMax{std::min(std::max(longest_type{1}, longest_type{sizeof(S) / sizeof(T)}), longest_type{bits_.size()})};
            auto it{bits_.rbegin() + iMax - 1};

            for (size_t i{0}; i < iMax; ++i)
            {
                n <<= sizeof(T) * 8;
                n |= *it;

                --it;
            }

            if (!isPositive_)
                n = -n;

            return n;
        }

        CONSTEXPR operator unsigned short() const noexcept
        {
            return cast<unsigned short>();
        }

        CONSTEXPR operator int() const noexcept
        {
            return cast<int>();
        }

        CONSTEXPR operator unsigned int() const noexcept
        {
            return cast<unsigned int>();
        }

        CONSTEXPR operator long() const noexcept
        {
            return cast<long>();
        }

        CONSTEXPR operator unsigned long() const noexcept
        {
            return cast<unsigned long>();
        }

        CONSTEXPR operator long long() const noexcept
        {
            return cast<long long>();
        }

        CONSTEXPR operator unsigned long long() const noexcept
        {
            return cast<unsigned long long>();
        }

        CONSTEXPR bool isNan() const noexcept
        {
            return isNan_;
        }

        CONSTEXPR void setNan() noexcept
        {
            isNan_ = true;
            isInfinity_ = false;
            bits_.resize(0);
        }

        CONSTEXPR bool isInfinity() const noexcept
        {
            return isInfinity_;
        }

        CONSTEXPR void setInfinity() noexcept
        {
            isNan_ = false;
            isInfinity_ = true;
            bits_.resize(0);
        }

        CONSTEXPR Integer abs() const
        {
            if (isNegative())
                return -*this;

            return *this;
        }

        CONSTEXPR size_t precision() const noexcept
        {
            return bits_.size();
        }

        CONSTEXPR void setPrecision(size_t precision)
        {
            assert(precision);
            
            if (precision == bits_.size())
                return;

            Vector bits(precision, T{0});

            std::copy(bits_.rbegin(), bits_.rbegin() + std::min(bits_.size(), precision), bits.rbegin());

            bits_ = bits;
        }

        template <typename URNG>
        __device__ __host__ CONSTEXPR void setRandom()
        {
            isNan_ = false;
            isInfinity_ = false;
            
            URNG g;
                    
            isPositive_ = g() % 2;
            
            T* a(nullptr);
            cudaMalloc(&a, sizeof(T) * bits_.size());
            cudaMemcpy(a, thrust::raw_pointer_cast(bits_.data()), sizeof(T) * bits_.size(), cudaMemcpyHostToDevice);

            size_t const blockSize{BLOCK_SIZE};
            size_t const gridSize{(bits_.size() + blockSize) / blockSize};
            
            Integer_setRandom<T><<<gridSize, blockSize>>>(a, bits_.size(), g());
            
            cudaDeviceSynchronize();
            
            cudaMemcpy(thrust::raw_pointer_cast(bits_.data()), a, sizeof(T) * bits_.size(), cudaMemcpyDeviceToHost);

            cudaFree(a);
        }

        CONSTEXPR int isPrime(size_t reps = 25) const
        {
            assert(reps);
            
            if (*this < 2)
                return 0;
            else if (*this == 2)
                return 2;
            else if(!(*this & 1))
                return 0;
            else if (this->template fits<unsigned int>())
            {
                auto p{std::equal_range(primes.begin(), primes.end(), this->template cast<unsigned int>())};

                if (p.first != primes.end() && *p.first != this->template cast<unsigned int>())
                    --p.first;
                
                if (p.first != p.second && *p.first == this->template cast<unsigned int>())
                    return 2;
            }
            
            //Trial divisions
            
            auto const sqrtLimit(sqrt(*this));
            
            {
                auto const sqrtLimit(sqrt(*this));
                
                unsigned int* a(nullptr);
                cudaMalloc(&a, primes.size() * sizeof(unsigned int));
                cudaMemcpy(a, primes.data(), primes.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
                
                bool* divisible(nullptr);
                cudaMalloc(&divisible, sizeof(bool));
                cudaMemset(divisible, 0, sizeof(bool));
                
                T* numberData(nullptr);    
                cudaMalloc(&numberData, sizeof(T) * bits_.size());
                cudaMemcpy(numberData, bits_.data(), sizeof(T) * bits_.size(), cudaMemcpyHostToDevice);
                
                T* sqrtLimitData(nullptr);
                cudaMalloc(&sqrtLimitData, sizeof(T) * sqrtLimit.bits_.size());
                cudaMemcpy(sqrtLimitData, sqrtLimit.bits_.data(), sizeof(T) * sqrtLimit.bits_.size(), cudaMemcpyHostToDevice);
                
                size_t const blockSize{BLOCK_SIZE};
                size_t const gridSize{(primes.size() + blockSize) / blockSize};
                
                Integer_isPrime_trialDivision<T><<<gridSize, blockSize>>>(a, primes.size(),
                                                                          numberData,   bits_.size(),
                                                                          sqrtLimitData, sqrtLimit.bits_.size(),
                                                                          divisible);
                
                cudaDeviceSynchronize();
                
                cudaFree(a);
                cudaFree(numberData);
                cudaFree(sqrtLimitData);

                bool div;
                cudaMemcpy(&div, divisible, sizeof(bool), cudaMemcpyDeviceToHost);
                cudaFree(divisible);

                if (div)
                    return 0;
                
                if (sqrtLimit < primes.back())
                    return 2;
            }

            //Miller-Rabin tests

            auto s(*this - 1);

            while (!(s & 1))
                s >>= 1;

            auto const number(*this - 1);
            auto const& m(*this);
            Integer R(Integer(1) << m.number());

            assert(R > m);

            if (!(m & 1))
                ++R;

            while (!m.isCoprime(R))
            {
                if (!(m & 1))
                    --R;

                R <<= 1;

                if (!(m & 1))
                    ++R;
            }

            auto const R2modm((R * R) % m);
            Integer R_, m_;
            auto const d(gcdExtended(R, -m, R_, m_));

            assert(d.abs() == 1);
            assert(R * R_ - m * m_ == d);

            if (d == -1)
            {
                R_ = -R_;
                m_ = -m_;
            }
            
            {
                bool* divisible(nullptr);
                cudaMalloc(&divisible, sizeof(bool));
                cudaMemset(&divisible, 0, sizeof(bool));

                T* numberData(nullptr);
                cudaMalloc(&numberData, sizeof(T) * number.bits_.size());
                cudaMemcpy(numberData, number.bits_.data(), sizeof(T) * number.bits_.size(), cudaMemcpyHostToDevice);

                T* sData(nullptr);
                cudaMalloc(&sData, sizeof(T) * s.bits_.size());
                cudaMemcpy(sData, s.bits_.data(), sizeof(T) * s.bits_.size(), cudaMemcpyHostToDevice);
                
                T* RData(nullptr);
                cudaMalloc(&RData, sizeof(T) * R.bits_.size());
                cudaMemcpy(RData, R.bits_.data(), sizeof(T) * R.bits_.size(), cudaMemcpyHostToDevice);

                T* m_Data(nullptr);
                cudaMalloc(&m_Data, sizeof(T) * m_.bits_.size());
                cudaMemcpy(m_Data, m_.bits_.data(), sizeof(T) * m_.bits_.size(), cudaMemcpyHostToDevice);

                T* R2modmData(nullptr);
                cudaMalloc(&R2modmData, sizeof(T) * R2modm.bits_.size());
                cudaMemcpy(R2modmData, R2modm.bits_.data(), sizeof(T) * R2modm.bits_.size(), cudaMemcpyHostToDevice);

                size_t const blockSize{BLOCK_SIZE};
                size_t const gridSize{(reps + blockSize) / blockSize};
                
                Integer_isPrime_millerRabin<T><<<gridSize, blockSize>>>(numberData, number.bits_.size(),
                                                                        sData, s.bits_.size(),
                                                                        RData, R.bits_.size(),
                                                                        m_Data, m_.bits_.size(),
                                                                        R2modmData, R2modm.bits_.size(),
                                                                        divisible, reps);

                cudaDeviceSynchronize();
                
                cudaFree(numberData);
                cudaFree(sData);
                cudaFree(RData);
                cudaFree(m_Data);
                cudaFree(R2modmData);
                
                bool div;
                cudaMemcpy(&div, divisible, sizeof(bool), cudaMemcpyDeviceToHost);
                cudaFree(divisible);
                
                if (div)
                    return 0;
            }

            return 1;
        }

        __device__ __host__ CONSTEXPR void setPositive()
        {
            isPositive_ = true;
        }

        CONSTEXPR void setNegative()
        {
            isPositive_ = false;
        }

        static Integer nan()
        {
            static Integer n;
            n.setNan();

            return n;
        }

        static Integer infinity()
        {
            static Integer n;
            n.setInfinity();

            return n;
        }

        CONSTEXPR bool bit(size_t n) const noexcept
        {
            auto it{bits_.rbegin()};

            while (it != bits_.rend() && n > sizeof(T) * 8)
            {
                n -= sizeof(T) * 8;
                ++it;
            }

            if (it == bits_.rend())
                return false;

            return *it & (T{1} << n);
        }

        CONSTEXPR void setBit(size_t n, bool bit)
        {
            auto it{bits_.rbegin()};

            while (n > sizeof(T) * 8)
            {
                n -= sizeof(T) * 8;
                ++it;

                if (it == bits_.rend())
                {
                    Vector bits(bits_.size() + 1, T{0});

                    std::copy(bits_.rbegin(), bits_.rend(), bits.rbegin());

                    bits_ = bits;

                    it = bits_.rend() - 1;
                }
            }

            if (bit)
                *it |= T{1} << n;
            else
                *it &= ~(T{1} << n);
        }

        CONSTEXPR T bits(size_t n) const noexcept
        {
            if (n >= bits_.size())
                return T{0};

            return bits_[bits_.size() - 1 - n];
        }

        CONSTEXPR void setBits(size_t n, T const& bits)
        {
            if (bits_.size() < n)
            {
                Vector bits(n, T{0});

                std::copy(bits_.rbegin(), bits_.rend(), bits.rbegin());

                bits_ = bits;
            }

            bits_[bits_.size() - 1 - n] = bits;

            if (autoAdjust_)
                adjust();
        }

        template <size_t N>
        CONSTEXPR void setBits(size_t n, std::bitset<N> const& bits)
        {
            for (size_t i{0}; i < bits.size(); ++i)
                setBit(n + i, bits[i]);

            if (autoAdjust_)
                adjust();
        }

        CONSTEXPR size_t count() const noexcept
        {
            size_t count{0};

            for (auto b : bits_)
            {
                while (b)
                {
                    if (b & 1)
                        ++count;

                    b >>= 1;
                }
            }

            return count;
        }

        CONSTEXPR Integer<T, Vector> number() const noexcept
        {
            Integer<T, Vector> number(0);

            if (isNan() || isInfinity())
                return number;

            auto it{cu::begin(bits_)};

            while (!*it && it != cu::end(bits_))
                ++it;

            if (it != cu::end(bits_))
            {
                auto b{*it};

                while (b)
                {
                    ++number;
                    b >>= 1;
                }

                number += (cu::distance<typename Vector::const_iterator>(it, cu::end(bits_)) - 1) * sizeof(T) * 8;
            }

            return number;
        }

        CONSTEXPR bool isEven() const noexcept
        {
            if (isNan() || isInfinity())
                return false;
            else if (!bits_.size())
                return true;

            return !(bits_.back() & 1);
        }

        CONSTEXPR bool isOdd() const noexcept
        {
            if (!bits_.size())
                return false;

            return bits_.back() & 1;
        }

        template <typename S>
        CONSTEXPR bool fits() const
        {
            return (*this == this->template cast<S>());
        }

        CONSTEXPR Integer sign() const
        {
            if (*this < 0)
                return Integer(-1);

            return Integer(1);
        }

        CONSTEXPR void setSign(Integer const& other) noexcept
        {
            isPositive_ = other.isPositive_;
        }

        CONSTEXPR Integer previousPrime() const
        {
            if (isNan())
                return *this;
            else if (isInfinity() || *this < 2)
                return nan();
            else if (*this == 2)
                return Integer(2);
            else if (*this == 3)
                return Integer(2);
            else if (this->template fits<unsigned int>())
            {
                auto p{std::equal_range(primes.begin(), primes.end(), this->template cast<unsigned int>())};
                
                if (p.first != primes.end() && *p.first != this->template cast<unsigned int>())
                    --p.first;
                
                if (p.first != p.second && p.second != primes.end())
                {
                    if (*p.first == this->template cast<unsigned int>())
                        return Integer(*(p.first - 1));
                    else
                        return Integer(*p.first);
                }
            }

            auto n(*this);
            
            if (n % 2)
                n -= 2;
            else
                --n;

            std::mutex mutex;
            Integer p(0);
    
            auto threadFunc
            {
                [this, &p, n, &mutex] (size_t start, size_t size) -> void
                {
                    Integer number{n};
                    
                    while (p > number)
                    {
                        number = n - start * 2;
                        
                        if (number.isPrime())
                        {
                            mutex.lock();
                            if (number > p)
                                p = number;
                            mutex.unlock();
                        }
                        else
                            start += size;
                    }
                }
            };

            size_t const numThreads{std::thread::hardware_concurrency()};
            std::vector<std::thread> threads;

            for (size_t i{0}; i < numThreads; ++i)
                threads.push_back(threadFunc, i + 1, numThreads);

            for (auto& t : threads)
                t.join();

            return p;
        }

        CONSTEXPR Integer nextPrime() const
        {
            if (isNan())
                return *this;
            else if (*this < 2)
                return Integer(2);
            else if (*this == 2)
                return Integer(3);
            else if (isInfinity())
                return nan();
            else if (this->template fits<unsigned int>())
            {
                auto p{std::equal_range(primes.begin(), primes.end(), this->template cast<unsigned int>())};
                
                if (p.first != primes.end() && *p.first != this->template cast<unsigned int>())
                    --p.first;
                
                if (p.first != p.second && p.second != primes.end())
                    return Integer(*p.second);
            }

            auto n(*this);

            if (n % 2)
                n += 2;
            else
                ++n;

            std::mutex mutex;
            Integer p(Integer::infinity());
    
            auto threadFunc
            {
                [this, &p, n, &mutex] (size_t start, size_t size) -> void
                {
                    Integer number{n};
                    
                    while (number < p)
                    {
                        number = n + start * 2;
                        
                        if (number.isPrime())
                        {
                            mutex.lock();
                            if (number < p)
                                p = number;
                            mutex.unlock();
                        }
                        else
                            start += size;
                    }
                }
            };

            size_t const numThreads{std::thread::hardware_concurrency()};
            std::vector<std::thread> threads;

            for (size_t i{0}; i < numThreads; ++i)
                threads.push_back(threadFunc, i + 1, numThreads);

            for (auto& t : threads)
                t.join();

            return p;
        }

        CONSTEXPR size_t size() const noexcept
        {
            return bits_.size();
        }

        CONSTEXPR void adjust()
        {
            if (!bits_.size())
                return;

            auto it{cu::begin(bits_)};

            while (!*it && it != cu::end(bits_))
                ++it;

            if (it == cu::end(bits_))
                it = cu::end(bits_) - 1;

            if (it != cu::begin(bits_))
                bits_ = Vector{it, cu::end(bits_)};
        }

        CONSTEXPR bool isCoprime(Integer const& other) const noexcept
        {
            return gcd(*this, other) == 1;
        }

        CONSTEXPR bool autoAdjust() const noexcept
        {
            return autoAdjust_;
        }

        CONSTEXPR void setAutoAdjust(bool autoAdjust) noexcept
        {
            autoAdjust_ = autoAdjust;
        }
        
        CONSTEXPR char const* data() const noexcept
        {
            return reinterpret_cast<char const*>(bits_.data());
        }
        
        CONSTEXPR char* data() noexcept
        {
            return reinterpret_cast<char*>(bits_.data());
        }
        
        CONSTEXPR size_t dataSize() const noexcept
        {
            return sizeof(T) / sizeof(char) * bits_.size();
        }
        
        CONSTEXPR void setData(char const* data, size_t size) noexcept
        {
            std::memcpy(bits_.data(), data, std::min(size, dataSize()));
        }

    private:
        bool isPositive_{true};
        Vector bits_;
        bool isNan_{false};
        bool isInfinity_{false};
        bool autoAdjust_{true};
};

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator*(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs *= rhs;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator+(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs += rhs;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator-(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs -= rhs;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator/(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs /= rhs;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator%(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs %= rhs;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator&(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs &= rhs;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator|(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs |= rhs;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator^(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs ^= rhs;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator<<(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs <<= rhs;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> operator>>(Integer<T, Vector> lhs, Integer<T, Vector> const& rhs)
{
    return lhs >>= rhs;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline bool operator>(Integer<T, Vector> const& lhs, S const& rhs) noexcept
{
    return lhs.operator>(Integer<T, Vector>(rhs));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline bool operator>(S const& lhs, Integer<T, Vector> const& rhs) noexcept
{
    return rhs.operator<(Integer<T, Vector>(lhs));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline bool operator>=(Integer<T, Vector> const& lhs, S const& rhs) noexcept
{
    return lhs.operator>=(Integer<T, Vector>(rhs));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline bool operator>=(S const& lhs, Integer<T, Vector> const& rhs) noexcept
{
    return rhs.operator<=(Integer<T, Vector>(lhs));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline bool operator<(Integer<T, Vector> const& lhs, S const& rhs) noexcept
{
    return lhs.operator<(Integer<T, Vector>(rhs));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline bool operator<(S const& lhs, Integer<T, Vector> const& rhs) noexcept
{
    return rhs.operator>(Integer<T, Vector>(lhs));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline bool operator<=(Integer<T, Vector> const& lhs, S const& rhs) noexcept
{
    return lhs.operator<=(Integer<T, Vector>(rhs));
}

template <typename T, typename S, class Vector>
__device__ __host__ CONSTEXPR inline bool operator<=(S const& lhs, Integer<T, Vector> const& rhs) noexcept
{
    return rhs.operator>=(Integer<T, Vector>(lhs));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline bool operator==(S const& lhs, Integer<T, Vector> const& rhs) noexcept
{
    return rhs.operator==(Integer<T, Vector>(lhs));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline bool operator!=(S const& lhs, Integer<T, Vector> const& rhs) noexcept
{
    return rhs.operator!=(Integer<T, Vector>(lhs));
}

template <typename T, typename S, class Vector>
__device__ __host__ CONSTEXPR inline Integer<T, Vector> operator+(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs += Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator+(S const& lhs, Integer<T, Vector> rhs)
{
    return rhs += Integer<T, Vector>(lhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator-(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs -= Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator-(S const& lhs, Integer<T, Vector> rhs)
{
    return -(rhs -= Integer<T, Vector>(lhs));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator/(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs /= Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator/(S const& lhs, Integer<T, Vector> const& rhs)
{
    return Integer<T, Vector>(lhs) /= rhs;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator*(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs *= Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator*(S const& lhs, Integer<T, Vector> const& rhs)
{
    return Integer<T, Vector>(lhs) *= rhs;
}

template <typename T, typename S, class Vector>
__device__ __host__ CONSTEXPR inline Integer<T, Vector> operator%(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs %= Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator%(S const& lhs, Integer<T, Vector> const& rhs)
{
    return Integer<T, Vector>(lhs) %= rhs;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator<<(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs <<= Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator<<(S const& lhs, Integer<T, Vector> const& rhs)
{
    return Integer<T, Vector>(lhs) <<= rhs;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator>>(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs >>= Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator>>(S const& lhs, Integer<T, Vector> const& rhs)
{
    return Integer<T, Vector>(lhs) >>= rhs;
}

template <typename T, typename S, class Vector>
__device__ __host__ CONSTEXPR inline Integer<T, Vector> operator&(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs &= Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator&(S const& lhs, Integer<T, Vector> const& rhs)
{
    return Integer<T, Vector>(lhs) &= rhs;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator|(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs |= Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator|(S const& lhs, Integer<T, Vector> const& rhs)
{
    return Integer<T, Vector>(lhs) |= rhs;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator^(Integer<T, Vector> lhs, S const& rhs)
{
    return lhs ^= Integer<T, Vector>(rhs);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> operator^(S const& lhs, Integer<T, Vector> const& rhs)
{
    return Integer<T, Vector>(lhs) ^= rhs;
}

template <typename T, class Vector>
CONSTEXPR inline std::ostream& operator<<(std::ostream& os, Integer<T, Vector> const& n)
{
    bool const showBase(os.flags() & std::ios_base::showbase);
    
    if (os.flags() & std::ios_base::oct)
        return os << n.toString(8, showBase);
    else if (os.flags() & std::ios_base::hex)
        return os << n.toString(16, showBase);
    else
        return os << n.toString(10, showBase);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> gcd(Integer<T, Vector> const& a, Integer<T, Vector> const& b)
{
    if (a.isNan() || b.isNan() || a.isInfinity() || b.isInfinity())
        return Integer<T, Vector>::nan();
    else if (a < 0)
        return gcd(a.abs(), b);
    else if (b < 0)
        return gcd(a, b.abs());
    else if (a < b)
        return gcd(b, a);
    else if (!a)
        return b;
    else if (!b)
        return a;
    else if (a.isEven() && b.isEven())
        return 2 * gcd(a >> 1, b >> 1);
    else if (a.isOdd() && b.isEven())
        return gcd(a, b >> 1);
    else if (a.isEven() && b.isOdd())
        return gcd(a >> 1, b);
    else //if (a.isOdd() && b.isOdd())
        return gcd((a - b) >> 1, b);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> gcd(Integer<T, Vector> const& a, S const& b)
{
    return gcd(a, Integer<T, Vector>(b));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> gcd(S const& a, Integer<T, Vector> const& b)
{
    return gcd(Integer<T, Vector>(a), b);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> lcm(Integer<T, Vector> const& a, Integer<T, Vector> const& b)
{
    return (a * b).abs() / gcd(a, b);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> lcm(Integer<T, Vector> const& a, S const& b)
{
    return lcm(a, Integer<T, Vector>(b));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> lcm(S const& a, Integer<T, Vector> const& b)
{
    return lcm(Integer<T, Vector>(a), b);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector>
gcdExtended(Integer<T, Vector> a, Integer<T, Vector> b,
            Integer<T, Vector>& u, Integer<T, Vector>& v)
{
    if (a.isNan() || b.isNan() || a.isInfinity() || b.isInfinity())
    {
        u.setNan();
        v.setNan();
        
        return Integer<T, Vector>::nan();
    }
    
    if (!a && !b)
        return Integer<T, Vector>(0);

    Integer<T, Vector> r1(a), u1(1), v1(0);
    Integer<T, Vector> r2(b), u2(0), v2(1);
    Integer<T, Vector> q, r_temp, u_temp, v_temp;

    while (r2 != 0)
    {
        q = r1 / r2;
        r_temp = r1 - q * r2;
        u_temp = u1 - q * u2;
        v_temp = v1 - q * v2;

        r1 = r2;
        u1 = u2;
        v1 = v2;
        r2 = r_temp;
        u2 = u_temp;
        v2 = v_temp;
    }

    u = u1;
    v = v1;

    return r1;
}

template <typename T, typename S1, typename S2, class Vector>
CONSTEXPR inline Integer<T, Vector> gcdExtended(S1 const& a, S2 const& b,
                                                Integer<T, Vector>& u,
                                                Integer<T, Vector>& v)
{
    return gcdExtended(Integer<T, Vector>(a), Integer<T, Vector>(b), u, v);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> factorial(Integer<T, Vector> const& n)
{
    if (n.isNan() || n.isInfinity())
        return n;

    assert(n >= 0);

    if (n == 0)
        return Integer<T, Vector>(1);

    return n * factorial(n - 1);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> doubleFactorial(Integer<T, Vector> const& n)
{
    return factorial(factorial(n));
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> multiFactorial(Integer<T, Vector> const& n,
                                                   Integer<T, Vector> const& m)
{
    return pow(factorial(n), m);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> multiFactorial(Integer<T, Vector> const& n, S const& m)
{
    return multiFactorial(n, Integer<T, Vector>(m));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> multiFactorial(S const& n, Integer<T, Vector> const& m)
{
    return multiFactorial(Integer<T, Vector>(n), m);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> pow(Integer<T, Vector> base, Integer<T, Vector> exp)
{
    assert(exp >= 0);

    if (base.isInfinity() || base.isNan())
        return base;
    else if (exp.isNan() || exp.isInfinity())
        return exp;
    else if (base < 0)
    {
        auto n(pow(base.abs(), exp));

        if (exp & 1)
            n = -n;

        return n;
    }
    else if (base == 2)
        return Integer<T, Vector>(1) << exp;

    Integer<T, Vector> result(1);

    for (;;)
    {
        if (exp & 1)
            result *= base;

        exp >>= 1;

        if (!exp)
            break;

        base *= base;
    }

    return result;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> pow(Integer<T, Vector> const& base, S const& exp)
{
    return pow(base, Integer<T, Vector>(exp));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> pow(S const& base, Integer<T, Vector> const& exp)
{
    return pow(Integer<T, Vector>(base), exp);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> powm(Integer<T, Vector> base,
                                         Integer<T, Vector> exp,
                                         Integer<T, Vector> const& mod)
{
    assert(exp >= 0);

    Integer<T, Vector> result(1);

    auto base_mod(base % mod);

    while (exp > 0)
    {
        if ((exp & 1) == 1)
        {
            result *= base_mod;
            result %= mod;
        }

        base_mod *= base_mod;
        base_mod %= mod;

        exp >>= 1;
    }

    return result;
}

template <typename T, typename S, typename U, class Vector>
CONSTEXPR inline Integer<T, Vector> powm(Integer<T, Vector> const& base,
                                         S const& exp, U const& mod)
{
    return powm(base, Integer<T, Vector>(exp), Integer<T, Vector>(mod));
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> abs(Integer<T, Vector> const& n)
{
    return n.abs();
}

template <typename T, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQr(Integer<T, Vector> const& dividend, Integer<T, Vector> const& divisor)
{
    if (!divisor)
        return {Integer<T, Vector>::nan(), Integer<T, Vector>::nan()};
    else if (!dividend)
        return {Integer<T, Vector>{0}, Integer<T, Vector>{0}};
    else if (dividend.isNan())
        return {dividend, dividend};
    else if (divisor.isNan())
        return {divisor, divisor};
    else if (dividend.isInfinity() || divisor.isInfinity())
        return {Integer<T, Vector>::nan(), Integer<T, Vector>::nan()};
    else if (divisor.abs() > dividend.abs())
        return {Integer<T, Vector>{0}, dividend};
    else if (dividend < 0 && divisor < 0)
    {
        auto qr{computeQr(-dividend, -divisor)};

        if (qr.second)
        {
            ++qr.first;
            qr.second += divisor;
        }

        return qr;
    }
    else if (dividend > 0 && divisor < 0)
    {
        auto qr{computeQr(dividend, -divisor)};

        qr.first = -qr.first;

        if (qr.second)
        {
            qr.first -= 1;
            qr.second += divisor;
        }

        return qr;
    }
    else if (dividend < 0 && divisor > 0)
    {
        auto qr{computeQr(-dividend, divisor)};

        qr.first = -qr.first;

        if (qr.second)
        {
            qr.first -= 1;
            qr.second = divisor - qr.second;
        }

        return qr;
    }

    Integer<T, Vector> start(1);
    auto end(dividend);

    while (start <= end)
    {
        auto mid(end + start);
        mid >>= 1;

        auto n(dividend - divisor * mid);

        if (n > divisor)
            start = mid + 1;
        else if (n < 0)
            end = mid - 1;
        else
        {
            if (n == divisor)
            {
                ++mid;
                n = 0;
            }

            return {mid, n};
        }
    }

    return {Integer<T, Vector>(0), dividend};
}

template <typename T, typename S, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQr(Integer<T, Vector> const& dividend, S const& divisor)
{
    return computeQr(dividend, Integer<T, Vector>(divisor));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQr(S const& dividend, Integer<T, Vector> const& divisor)
{
    return computeQr(Integer<T, Vector>(dividend), divisor);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector>
computeQuotient(Integer<T, Vector> const& dividend,
                Integer<T, Vector> const& divisor)
{
    if (!divisor)
        return Integer<T, Vector>::nan();
    else if (!dividend)
        return Integer<T, Vector>{0};
    else if (dividend.isNan())
        return dividend;
    else if (divisor.isNan())
        return divisor;
    else if (dividend.isInfinity() || divisor.isInfinity())
        return Integer<T, Vector>::nan();
    else if (divisor.abs() > dividend.abs())
        return Integer<T, Vector>{0};
    else if (dividend < 0 && divisor < 0)
        return computeQuotient(-dividend, -divisor);
    else if (dividend > 0 && divisor < 0)
        return -computeQuotient(dividend, -divisor);
    else if (dividend < 0 && divisor > 0)
        return -computeQuotient(-dividend, divisor);

    Integer<T, Vector> start(1);
    auto end(dividend);

    while (start <= end)
    {
        auto mid(end + start);
        mid >>= 1;

        auto n(dividend - divisor * mid);

        if (n > divisor)
            start = mid + 1;
        else if (n < 0)
            end = mid - 1;
        else
        {
            if (n == divisor)
            {
                ++mid;
                n = 0;
            }

            return mid;
        }
    }

    return Integer<T, Vector>(0);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector>
computeQuotient(Integer<T, Vector> const& dividend, S const& divisor)
{
    return computeQuotient(dividend, Integer<T, Vector>(divisor));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector>
computeQuotient(S const& dividend, Integer<T, Vector> const& divisor)
{
    return computeQuotient(Integer<T, Vector>(dividend), divisor);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> fibonacci(Integer<T, Vector> n)
{
    assert(n >= 0);

    if (!n)
        return 0;
    else if (n == 1)
        return 1;

    n -= 1;

    Integer<T, Vector> fn_2(0);
    Integer<T, Vector> fn_1(1);
    Integer<T, Vector> fn;

    while (n)
    {
        fn = fn_1 + fn_2;
        fn_2 = fn_1;
        fn_1 = fn;
        --n;
    }

    return fn;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> primorial(Integer<T, Vector> n)
{
    Integer<T, Vector> result(1);
    Integer<T, Vector> number(2);

    while (number <= n)
    {
        result *= number;
        number = number.nextPrime();
    }

    return result;
}

template <typename T, class Vector>
CONSTEXPR inline int jacobi(Integer<T, Vector> const& a, Integer<T, Vector> n)
{
    assert(n > 0 && n.isOdd());

    int result(1);
    Integer<T, Vector> prime(2);

    while (n != 1)
    {
        if (!(n % prime))
        {
            n /= prime;
            result *= legendre(a, prime);
        }
        else
            prime = prime.nextPrime();
    }

    return result;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline int jacobi(Integer<T, Vector> const& a, S const& n)
{
    return jacobi(a, Integer<T, Vector>(n));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline int jacobi(S const& a, Integer<T, Vector> const& n)
{
    return jacobi(Integer<T, Vector>(a), n);
}

template <typename T, class Vector>
CONSTEXPR inline int legendre(Integer<T, Vector> const& a, Integer<T, Vector> const& p)
{
    assert(p.isPrime());

    if (!(a % p))
        return 0;
    else
    {
        bool isResidue{false};

        if (p == 2)
            isResidue = true;
        else
            isResidue = (powm(a, (p - 1) / 2, p) == 1);

        if (isResidue)
            return 1;
        else
            return -1;
    }
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> legendre(Integer<T, Vector> const& a, S const& p)
{
    return legendre(a, Integer<T, Vector>(p));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> legendre(S const& a, Integer<T, Vector> const& p)
{
    return legendre(Integer<T, Vector>(a), p);
}

template <typename T, class Vector>
CONSTEXPR inline int kronecker(Integer<T, Vector> const& a, Integer<T, Vector> const& b)
{
    if (a == b)
        return 1;
    else
        return 0;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline int kronecker(Integer<T, Vector> const& a, S const& b)
{
    return kronecker(a, Integer<T,Vector>(b));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline int kronecker(S const& a, Integer<T, Vector> const& b)
{
    return kronecker(Integer<T, Vector>(a), b);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> binomial(Integer<T, Vector> const& n,
                                             Integer<T, Vector> const& k)
{
    assert(n >= 0 && k >= 0);

    return factorial(n) / (factorial(k) * factorial(n - k));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> binomial(Integer<T, Vector> const& n, S const& k)
{
    return binomial(n, Integer<T, Vector>(k));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> binomial(S const& n, Integer<T, Vector> const& k)
{
    return binomial(Integer<T, Vector>(n), k);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> sqrt(Integer<T, Vector> const& n)
{
    if (n < 0)
        return Integer<T, Vector>::nan();
    else if (!n || n == 1 || n.isNan() || n.isInfinity())
        return n;

    Integer<T, Vector> lo(1), hi(n);
    Integer<T, Vector> res(1);

    while (lo <= hi)
    {
        auto mid(lo + hi);
        mid >>= 1;

        if (mid * mid <= n)
        {
            res = mid;
            lo = mid + 1;
        }
        else
            hi = mid - 1;
    }
    
    return res;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> root(Integer<T, Vector> const& x,
                                         Integer<T, Vector> const& n)
{
    assert(n > 0);
    
    if (x < 0)
        return Integer<T, Vector>::nan();
    else if (!x || x == 1 || x.isNan() || x.isInfinity())
        return x;
    else if (n == 1)
        return x;
    else if (n.isNan() || n.isInfinity())
        return Integer<T, Vector>::nan();

    Integer<T, Vector> lo(1), hi(x);
    Integer<T, Vector> res(1);

    while (lo <= hi)
    {
        auto mid(lo + hi);
        mid >>= 1;

        if (pow(mid, n) <= x)
        {
            res = mid;
            lo = mid + 1;
        }
        else
            hi = mid - 1;
    }

    return res;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> root(S const& x, Integer<T, Vector> const& n)
{
    return root(Integer<T, Vector>(x), n);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector> root(Integer<T, Vector> const& x, S const& n)
{
    return root(x, Integer<T, Vector>(n));
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector>
computeQuotientBinary(Integer<T, Vector> dividend, Integer<T, Vector> const& divisor)
{
    if (!divisor)
        return Integer<T, Vector>::nan();
    else if (!dividend)
        return Integer<T, Vector>{0};
    else if (dividend.isNan())
        return dividend;
    else if (divisor.isNan())
        return divisor;
    else if (dividend.isInfinity() || divisor.isInfinity())
        return Integer<T, Vector>::nan();
    else if (divisor.abs() > dividend.abs())
        return Integer<T, Vector>{0};
    else if (dividend < 0 && divisor > 0)
        return -computeQuotientBinary(-dividend, divisor);
    else if (dividend > 0 && divisor < 0)
        return -computeQuotientBinary(dividend, -divisor);
    else if (dividend < 0 && divisor < 0)
        return computeQuotientBinary(-dividend, -divisor);

    Integer<T, Vector> quotient(0);
    auto tempDivisor(divisor);
    Integer<T, Vector> bit(1);

    while (dividend >= (tempDivisor << 1))
    {
        tempDivisor <<= 1;
        bit <<= 1;
    }

    while (bit >= 1)
    {
        if (dividend >= tempDivisor)
        {
            dividend -= tempDivisor;
            quotient += bit;
        }

        tempDivisor >>= 1;
        bit >>= 1;
    }

    return quotient;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector>
computeQuotientBinary(Integer<T, Vector> const& dividend, S const& divisor)
{
    return computeQuotientBinary(dividend, Integer<T, Vector>(divisor));
}

template <typename T, typename S,class Vector>
CONSTEXPR inline Integer<T, Vector>
computeQuotientBinary(S const& dividend, Integer<T, Vector> const& divisor)
{
    return computeQuotientBinary(Integer<T, Vector>(dividend), divisor);
}

template <typename T, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQrBinary(Integer<T, Vector> const& dividend,
                Integer<T, Vector> const& divisor)
{
    std::pair<Integer<T, Vector>, Integer<T, Vector> > qr{computeQuotientBinary(dividend, divisor), Integer<T, Vector>(0)};
    qr.second = dividend.abs() - qr.first.abs() * divisor.abs();

    if (dividend < 0 && divisor < 0)
    {
        if (qr.second)
        {
            ++qr.first;
            qr.second += divisor;
        }
    }
    else if (dividend > 0 && divisor < 0)
    {
        if (qr.second)
        {
            qr.first -= 1;
            qr.second += divisor;
        }
    }
    else if (dividend < 0 && divisor > 0)
    {
        if (qr.second)
        {
            qr.first -= 1;
            qr.second = divisor - qr.second;
        }
    }

    return qr;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQrBinary(Integer<T, Vector> const& dividend, S const& divisor)
{
    return computeQrBinary(dividend, Integer<T, Vector>(divisor));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQrBinary(S const& dividend, Integer<T, Vector> const& divisor)
{
    return computeQrBinary(Integer<T, Vector>(dividend), divisor);
}

template <typename T, typename S, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQrByDivision(Integer<T, Vector> const& dividend, S const& divisor)
{
    return computeQrByDivision(dividend, Integer<T, Vector>(divisor));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQrByDivision(S const& dividend, Integer<T, Vector> const& divisor)
{
    return computeQrByDivision(Integer<T, Vector>(dividend), divisor);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector>
computeQuotientBurnikelZiegler(Integer<T, Vector> dividend,
                               Integer<T, Vector> const& divisor)
{
    if (!divisor)
        return Integer<T, Vector>::nan();
    else if (!dividend)
        return Integer<T, Vector>{0};
    else if (dividend.isNan())
        return dividend;
    else if (divisor.isNan())
        return divisor;
    else if (dividend.isInfinity() || divisor.isInfinity())
        return Integer<T, Vector>::nan();
    else if (divisor.abs() > dividend.abs())
        return Integer<T, Vector>{0};

    std::pair<Integer<T, Vector>, Integer<T, Vector> > qr;
    qr = computeQrBurnikelZiegler(dividend, divisor);

    if (dividend < 0 && divisor < 0)
    {
        if (qr.second)
            --qr.first;
    }
    else if (dividend > 0 && divisor < 0)
    {
        if (qr.second)
            ++qr.first;
    }
    else if (dividend < 0 && divisor > 0)
    {
        if (qr.second)
            ++qr.first;
    }

    return qr.first;
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector>
computeQuotientBurnikelZiegler(Integer<T, Vector> const& dividend, S const& divisor)
{
    return computeQuotientBurnikelZiegler(dividend, Integer<T, Vector>(divisor));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline Integer<T, Vector>
computeQuotientBurnikelZiegler(S const& dividend, Integer<T, Vector> const& divisor)
{
    return computeQuotientBurnikelZiegler(Integer<T, Vector>(dividend), divisor);
}

template <typename T, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQrBurnikelZiegler(Integer<T, Vector> const& dividend,
                         Integer<T, Vector> const& divisor)
{
    if (!divisor)
        return {Integer<T, Vector>::nan(), Integer<T, Vector>::nan()};
    else if (!dividend)
        return {Integer<T, Vector>{0}, Integer<T, Vector>{0}};
    else if (dividend.isNan())
        return {dividend, dividend};
    else if (divisor.isNan())
        return {divisor, divisor};
    else if (dividend.isInfinity() || divisor.isInfinity())
        return {Integer<T, Vector>::nan(), Integer<T, Vector>::nan()};
    else if (divisor.abs() > dividend.abs())
        return {Integer<T, Vector>(0), dividend};
    else if (divisor.abs() == 1)
        return {divisor.sign() * dividend, Integer<T, Vector>(0)};
    else if (dividend < 0 && divisor < 0)
    {
        auto qr{computeQrBurnikelZiegler(-dividend, -divisor)};

        if (qr.second)
        {
            ++qr.first;
            qr.second += divisor;
        }

        return qr;
    }
    else if (dividend > 0 && divisor < 0)
    {
        auto qr{computeQrBurnikelZiegler(dividend, -divisor)};

        qr.first = -qr.first;

        if (qr.second)
        {
            --qr.first;
            qr.second += divisor;
        }

        return qr;
    }
    else if (dividend < 0 && divisor > 0)
    {
        auto qr{computeQrBurnikelZiegler(-dividend, divisor)};

        qr.first = -qr.first;

        if (qr.second)
        {
            --qr.first;
            qr.second = divisor - qr.second;
        }

        return qr;
    }

    std::function<void(std::vector<Integer<T, Vector> >&, Integer<T, Vector> const&,
                       Integer<T, Vector> const&,
                       Integer<T, Vector> const&, Integer<T, Vector> const&)> inner1;

    inner1 =
    [&inner1] (std::vector<Integer<T, Vector> >& a_digits, Integer<T, Vector> const& x,
               Integer<T, Vector> const& L,
               Integer<T, Vector> const& R, Integer<T, Vector> const& n) -> void
    {
        if (L + 1 == R)
        {
            a_digits[L] = x;
            return;
        }

        auto mid(L);
        mid += R;
        mid >>= 1;
        auto shift(mid);
        shift -= L;
        shift *= n;
        auto upper(x >> shift);
        auto lower(upper);
        lower <<= shift;
        lower ^= x;
        inner1(a_digits, lower, L, mid, n);
        inner1(a_digits, upper, mid, R, n);
    };

    auto _int2digits
    {
        [&inner1] (Integer<T, Vector> const& a,
                   Integer<T, Vector> const& n) -> std::vector<Integer<T, Vector> >
        {
            assert(a >= 0);

            if (!a)
                return std::vector<Integer<T, Vector> >{Integer<T, Vector>(0)};

            std::vector<Integer<T, Vector> > a_digits((a.number() + n - 1).template cast<longest_type>() / n, Integer<T, Vector>(0));

            if (a)
                inner1(a_digits, a, Integer<T, Vector>(0), Integer<T, Vector>(a_digits.size()), n);

            return a_digits;
        }
    };

    std::function<Integer<T, Vector>(std::vector<Integer<T, Vector> > const&, Integer<T, Vector> const&,
                                     Integer<T, Vector> const&, Integer<T, Vector> const&)> inner2;

    inner2 =
    [&inner2] (std::vector<Integer<T, Vector> > const& digits, Integer<T, Vector> const& L,
               Integer<T, Vector> const& R, Integer<T, Vector> const& n) -> Integer<T, Vector>
    {
        if (L + 1 == R)
           return digits[L];

        auto mid(L);
        mid += R;
        mid >>= 1;
        auto shift(mid);
        shift -= L;
        shift *= n;

        return (inner2(digits, mid, R, n) << shift) + inner2(digits, L, mid, n);
    };

    auto _digits2int
    {
        [&inner2] (std::vector<Integer<T, Vector> > const& digits, Integer<T, Vector> const& n) -> Integer<T, Vector>
        {
            if (!digits.size())
                return Integer<T, Vector>(0);

            return inner2(digits, Integer<T, Vector>(0), Integer<T, Vector>(digits.size()), n);
        }
    };

    std::function<std::pair<Integer<T, Vector>, Integer<T, Vector> >(Integer<T, Vector>, Integer<T, Vector>, Integer<T, Vector>)> _div2n1n;
    std::function<std::pair<Integer<T, Vector>, Integer<T, Vector> >(Integer<T, Vector> const&, Integer<T, Vector> const&,
                                                                     Integer<T, Vector> const&, Integer<T, Vector> const&,
                                                                     Integer<T, Vector> const&, Integer<T, Vector> const&)> _div3n2n;

    _div2n1n =
    [&_div3n2n] (Integer<T, Vector> a, Integer<T, Vector> b,
                 Integer<T, Vector> n) -> std::pair<Integer<T, Vector>, Integer<T, Vector> >
    {
        if (a.template fits<longest_type>() && b.template fits<longest_type>())
            return {Integer<T, Vector>(a.template cast<longest_type>() / b.template cast<longest_type>()),
                    Integer<T, Vector>(a.template cast<longest_type>() % b.template cast<longest_type>())};

        auto pad(n & 1);

        if (pad)
        {
            a <<= 1;
            b <<= 1;
            ++n;
        }

        auto const half_n(n >> 1);
        Integer<T, Vector> mask(1);
        mask <<= half_n;
        --mask;
        auto const b1(b >> half_n);
        auto const b2(b & mask);
        auto tmp(a);
        tmp >>= half_n;
        tmp &= mask;
        auto[q1, r] = _div3n2n(a >> n, tmp, b, b1, b2, half_n);
        auto[q2, r2] = _div3n2n(r, a & mask, b, b1, b2, half_n);
        r = r2;

        if (pad)
            r >>= 1;
        
        q1 <<= half_n;
        q1 |= q2;

        return {q1, r};
    };

    _div3n2n =
    [&_div2n1n] (Integer<T, Vector> const& a12, Integer<T, Vector> const& a3,
                 Integer<T, Vector> const& b, Integer<T, Vector> const& b1,
                 Integer<T, Vector> const& b2, Integer<T, Vector> const& n) -> std::pair<Integer<T, Vector>, Integer<T, Vector> >
    {
        Integer<T, Vector> q, r;

        if (a12 >> n == b1)
        {
            q = 1;
            q <<= n;
            --q;
            r = b1;
            r <<= n;
            r -= a12;
            r -= b1;
            if (r.isNegative())
                r.setPositive();
            else
                r.setNegative();
        }
        else
        {
            auto const p{_div2n1n(a12, b1, n)};
            q = p.first;
            r = p.second;
        }

        r <<= n;
        r |= a3;
        r -= q * b2;

        while (r < 0)
        {
            --q;
            r += b;
        }

        return {q, r};
    };

    auto const n{divisor.number()};
    auto const a_digits(_int2digits(dividend, n));

    Integer<T, Vector> r(0);
    Integer<T, Vector> q(0);
    std::vector<Integer<T, Vector> > q_digits;

    for (auto it{a_digits.rbegin()}; it != a_digits.rend(); ++it)
    {
        r <<= n;
        r += *it;
        auto[q_digit, r_] = _div2n1n(r, divisor, n);
        r = r_;
        q_digits.push_back(q_digit);
    }

    cu::reverse(cu::begin(q_digits), cu::end(q_digits));

    q = _digits2int(q_digits, n);

    return {q, r};
}

template <typename T, typename S, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQrBurnikelZiegler(Integer<T, Vector> const& dividend, S const& divisor)
{
    return computeQrBurnikelZiegler(dividend, Integer<T, Vector>(divisor));
}

template <typename T, typename S, class Vector>
CONSTEXPR inline std::pair<Integer<T, Vector>, Integer<T, Vector> >
computeQrBurnikelZiegler(S const& dividend, Integer<T, Vector> const& divisor)
{
    return computeQrBurnikelZiegler(Integer<T, Vector>(dividend), divisor);
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> const& min(Integer<T, Vector> const& a,
                                               Integer<T, Vector> const& b)
{
    return a < b ? a : b;
}

template <typename T, class Vector>
CONSTEXPR inline Integer<T, Vector> const& max(Integer<T, Vector> const& a,
                                               Integer<T, Vector> const& b)
{
    return a > b ? a : b;
}

#endif // INTEGER_CUH

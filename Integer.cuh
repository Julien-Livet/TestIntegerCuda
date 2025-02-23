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

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cu/algorithm.cuh"
#include "cu/pair.cuh"
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

template <typename T, typename Enable = void>
class Integer;

template <typename T>
__global__ void Integer_isPrime_trialDivision(unsigned int const* primes, size_t primesSize,
                                              T const* numberData, size_t numberDataSize,
                                              T const* sqrtLimitData, size_t sqrtLimitDataSize,
                                              bool* divisible)
{
    size_t const idx{blockIdx.x * blockDim.x + threadIdx.x};

    if (idx < primesSize && !*divisible)
    {
        Integer<T> const n(numberData,
                           numberData + numberDataSize);
        Integer<T> const s(sqrtLimitData,
                           sqrtLimitData + numberDataSize);
        
        if (primes[idx] <= s && !(n % primes[idx]))
            *divisible = true;
    }
}

template <typename T>
__device__ __host__
Integer<T> reduction(Integer<T> const& t, Integer<T> const& R,
                     Integer<T> const& n, Integer<T> const& n_)
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

template <typename T>
__device__ __host__
Integer<T> redmulmod(Integer<T> const& a, Integer<T> b,
                     Integer<T> const& n, Integer<T> const& R,
                     Integer<T> const& n_, Integer<T> const& R2modn)
{
    auto const reda(reduction(a * R2modn, R, n, n_));
    auto const redb(reduction(b * R2modn, R, n, n_));
    auto const redc(reduction(reda * redb, R, n, n_));

    return reduction(redc, R, n, n_);
}

template <typename T>
__device__ __host__
bool mulmod(Integer<T> const& a, Integer<T> b, Integer<T> const& m)
{
    Integer<T> x(0);
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

template <typename T>
__device__ __host__
Integer<T> modulo(Integer<T> const& base, Integer<T> e,
                  Integer<T> const& m, Integer<T> const& R,
                  Integer<T> const& m_, Integer<T> const& R2modm)
{
    Integer<T> x(1);
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
        Integer<T> const number(numberData, numberData + numberDataSize);
        auto const n(number + 1);
        auto a(n);
        a.template setRandom<std::random_device>();
        a.setPositive();
        a %= number;
        ++a;

        Integer<T> const R(RData, RData + RDataSize);
        Integer<T> const m_(m_Data, m_Data + m_DataSize);
        Integer<T> const R2modm(R2modmData, R2modmData + R2modmDataSize);

        Integer<T> temp(sData, sData + sDataSize);
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

template <typename T>
class Integer<T, typename std::enable_if<std::is_unsigned<T>::value && std::is_same<T, typename cu::vector<T>::value_type>::value >::type>
{
    public:
        __device__ __host__ CONSTEXPR Integer() = default;

        template <typename S, std::enable_if_t<std::is_standard_layout_v<S> && std::is_trivial_v<S> >* = nullptr>
        __device__ __host__ CONSTEXPR Integer(S n)
        {
            bits_.reserve(cu::max(longest_type{1}, longest_type{sizeof(S) / sizeof(T)}));
            
            if constexpr (std::is_signed<S>::value)
            {
                isPositive_ = (n >= 0);

                if (n < 0)
                    n = -n;
            }
            
            if constexpr (sizeof(T) == sizeof(S))
                bits_.push_back(n);
            else
            {
                auto const shift{longest_type{1} << cu::min(sizeof(T), sizeof(S)) * 8};
                
                for (size_t i{0}; i < bits_.capacity(); ++i)
                {
                    bits_.push_back(n % shift);
                    n /= shift;
                }
                
                cu::reverse(cu::begin(bits_), cu::end(bits_));
            }
            
            adjust();
        }

        __device__ __host__ CONSTEXPR Integer(cu::vector<T> const& bits, bool isPositive = true) : isPositive_{isPositive}, bits_{bits}
        {
            adjust();
        }

        template <size_t N>
        __device__ __host__ CONSTEXPR Integer(std::bitset<N> const& bits, bool isPositive = true) : isPositive_{isPositive}
        {
            setBits(0, bits);
        }

        __host__ CONSTEXPR Integer(std::initializer_list<T> const& bits, bool isPositive = true) : isPositive_{isPositive}
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

        __host__ CONSTEXPR Integer(char const* n, size_t base = 0) : Integer(std::string{n}, base)
        {
        }

        __host__ Integer(std::string n, size_t base = 0)
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
        __host__ CONSTEXPR Integer(Integer<S> const& other)
        {
            bits_ = cu::vector<T>(other.dataSize() / sizeof(T) + (other.dataSize() % sizeof(T) ? 1 : 0), 0);
            cu::copy(other.data(), other.data() + other.dataSize(), reinterpret_cast<char*>(bits_.begin()) + bits_.size() * sizeof(T) - other.dataSize());
        }

        __device__ __host__ CONSTEXPR bool isPositive() const noexcept
        {
            return isPositive_;
        }

        __device__ __host__ CONSTEXPR bool isNegative() const noexcept
        {
            return !isPositive_;
        }

        __device__ __host__ CONSTEXPR auto const& bits() const noexcept
        {
            return bits_;
        }

        __host__ CONSTEXPR void invert() noexcept
        {
            T* a(nullptr);
            auto r{cudaMalloc(&a, sizeof(T) * size())};
            assert(r == cudaSuccess);
            assert(a);
            r = cudaMemcpy(a, bits_.data(), sizeof(T) * bits_.size(), cudaMemcpyHostToDevice);
            assert(r == cudaSuccess);

            size_t const blockSize{BLOCK_SIZE};
            size_t const gridSize{(bits_.size() + blockSize) / blockSize};
            
            Integer_invert<T><<<gridSize, blockSize>>>(a, bits_.size());
            
            cudaDeviceSynchronize();
            
            r = cudaMemcpy(bits_.data(), a, sizeof(T) * bits_.size(), cudaMemcpyDeviceToHost);
            assert(r == cudaSuccess);

            r = cudaFree(a);
            assert(r == cudaSuccess);
            
            if (autoAdjust_)
                adjust();
        }

        __device__ __host__ CONSTEXPR Integer& operator*=(Integer const& other)
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
                            size_t n{cu::max(number(a), number(b))};
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
                        size_t const n{cu::max(n1, n2)};
                        size_t const m{n / 2};
                        cu::vector<T> bits(m, T{0});
                        cu::copy(bits_.rbegin(),
                                  bits_.rbegin() + cu::min(bits_.size(), m),
                                  bits.rbegin());
                        Integer const x0(bits);
                        bits = cu::vector<T>(m, T{0});
                        cu::copy(bits_.rbegin() + m,
                                  bits_.rbegin() + cu::min(bits_.size(), 2 * m),
                                  bits.rbegin());
                        Integer const x1(bits);
                        bits = cu::vector<T>(m, T{0});
                        cu::copy(other.bits_.rbegin(),
                                  other.bits_.rbegin() + cu::min(other.bits_.size(), m),
                                  bits.rbegin());
                        Integer const y0(bits);
                        bits = cu::vector<T>(m, T{0});
                        cu::copy(other.bits_.rbegin() + m,
                                  other.bits_.rbegin() + cu::min(other.bits_.size(), 2 * m),
                                  bits.rbegin());
                        Integer const y1(bits);

                        assert(*this == ((x1 << (m * sizeof(T) * 8)) | x0));
                        assert(other == ((y1 << (m * sizeof(T) * 8)) | y0));

                        auto const z0(x0 * y0);
                        auto const z1(x1 * y0 + x0 * y1);
                        auto const z2(x1 * y1);

                        //o = m * 8 * sizeof(T)
                        //xy = z2 * 2^(2 * o) + z1 * 2^o + z0

                        *this = z0;
                        bits = cu::vector<T>(z1.size() + m, T{0});
                        cu::copy(z1.bits_.rbegin(), z1.bits_.rend(), bits.rbegin() + m);
                        *this += Integer(bits);
                        bits = cu::vector<T>(z2.size() + 2 * m, T{0});
                        cu::copy(z2.bits_.rbegin(), z2.bits_.rend(), bits.rbegin() + 2 * m);
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

            return *this;
        }

        __device__ __host__ CONSTEXPR Integer& operator+=(Integer const& other)
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
                    size_t const n{cu::max(a.size(), b.size())};
                    cu::vector<T> result;
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
                    size_t const n{cu::max(a.size(), b.size())};
                    cu::vector<T> result;
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

            return *this;
        }

        __device__ __host__ CONSTEXPR Integer& operator-=(Integer const& other)
        {
            auto const lhs(*this);
            auto const rhs(other);

            *this += -other;

            return *this;
        }

        __device__ __host__ CONSTEXPR Integer& operator/=(Integer const& other)
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

            assert(abs() < rhs.abs());

            return *this;
        }

        __device__ __host__ CONSTEXPR Integer& operator<<=(Integer other)
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

            cu::vector<T> const v(n.template cast<longest_type>(), T{0});

            bits_.insert(cu::end(bits_), cu::begin(v), cu::end(v));

            other -= n * s;

            cu::vector<T> bits(bits_.size() + 1, T{0});

            cu::copy(bits_.rbegin(), bits_.rend(), bits.rbegin());

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

        __device__ __host__ CONSTEXPR Integer& operator>>=(Integer other)
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
                bits_.clear();

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

        __device__ __host__ CONSTEXPR bool operator>=(Integer const& other) const
        {
            return !operator<(other);
        }

        __device__ __host__ CONSTEXPR bool operator>(Integer const& other) const
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

            cu::vector<T> a(cu::max(bits_.size(), other.bits_.size()), T{0});
            cu::vector<T> b(a);

            cu::copy(bits_.rbegin(), bits_.rend(), a.rbegin());
            cu::copy(other.bits_.rbegin(), other.bits_.rend(), b.rbegin());

            auto const great{a > b};

            return isPositive_ ? great : !great;
        }

        __device__ __host__ CONSTEXPR bool operator<=(Integer const& other) const
        {
            return !operator>(other);
        }

        __device__ __host__ CONSTEXPR bool operator<(Integer const& other) const
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

            cu::vector<T> a(cu::max(bits_.size(), other.bits_.size()), T{0});
            cu::vector<T> b(a);

            cu::copy(bits_.rbegin(), bits_.rend(), a.rbegin());
            cu::copy(other.bits_.rbegin(), other.bits_.rend(), b.rbegin());

            auto const less{a < b};

            return isPositive_ ? less : !less;
        }

        __device__ __host__ CONSTEXPR bool operator==(Integer const& other) const noexcept
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

            for (size_t i{0}; i < cu::min(bits_.size(), other.bits_.size()); ++i)
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
        __device__ __host__ CONSTEXPR bool operator==(S const& other) const
        {
            return *this == Integer(other);
        }

        __device__ __host__ CONSTEXPR bool operator!=(Integer const& other) const
        {
            return !operator==(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR bool operator!=(S const& other) const
        {
            return *this != Integer(other);
        }

        __device__ __host__ CONSTEXPR Integer operator-() const
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

        __device__ __host__ CONSTEXPR operator bool() const noexcept
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

        __device__ __host__ CONSTEXPR Integer& operator--()
        {
            return *this -= 1;
        }

        __device__ __host__ CONSTEXPR Integer operator--(int)
        {
            auto x(*this);

            operator--();

            return x;
        }

        __device__ __host__ CONSTEXPR Integer& operator++()
        {
            return *this += 1;
        }

        __device__ __host__ CONSTEXPR Integer operator++(int)
        {
            auto x(*this);

            operator++();

            return x;
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator+=(S const& other)
        {
            return *this += Integer(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator-=(S const& other)
        {
            return *this -= Integer(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator/=(S const& other)
        {
            return *this /= Integer(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator*=(S const& other)
        {
            return *this *= Integer(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator%=(S const& other)
        {
            return *this %= Integer(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator>>=(S const& other)
        {
            return *this >>= Integer(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator<<=(S const& other)
        {
            return *this <<= Integer(other);
        }

        __device__ __host__ CONSTEXPR Integer& operator&=(Integer const& other)
        {
            cu::vector<T> bits(cu::max(bits_.size(), other.bits_.size())
                                  - cu::min(bits_.size(), other.bits_.size()), 0);

            if (bits_.size() > other.bits_.size())
                bits.insert(cu::end(bits), cu::begin(other.bits_), cu::end(other.bits_));
            else
                bits.insert(cu::end(bits), cu::begin(bits_), cu::end(bits_));

            cu::vector<T> const& otherBits(bits_.size() > other.bits_.size() ? bits_ : other.bits_);

            for (size_t i{0}; i < cu::min(bits_.size(), other.bits_.size()); ++i)
                *(bits.rbegin() + i) &= *(otherBits.rbegin() + i);

            bits_ = bits;

            if (autoAdjust_)
                adjust();

            return *this;
        }

        __device__ __host__ CONSTEXPR Integer& operator|=(Integer const& other)
        {
            cu::vector<T> bits(bits_.size() > other.bits_.size() ? bits_ : other.bits_);
            cu::vector<T> const& otherBits(bits_.size() > other.bits_.size() ? other.bits_ : bits_);

            for (size_t i{0}; i < otherBits.size(); ++i)
                *(bits.rbegin() + i) |= *(otherBits.rbegin() + i);

            bits_ = bits;

            if (autoAdjust_)
                adjust();

            return *this;
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator|=(S const& other)
        {
            return *this |= Integer(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator&=(S const& other)
        {
            return *this &= Integer(other);
        }

        template <typename S>
        __device__ __host__ CONSTEXPR Integer& operator^=(S const& other)
        {
            return *this ^= Integer(other);
        }

        __device__ __host__ CONSTEXPR Integer& operator^=(Integer const& other)
        {
            cu::vector<T> bits(bits_.size() > other.bits_.size() ? bits_ : other.bits_);
            cu::vector<T> const otherBits(bits_.size() > other.bits_.size() ? other.bits_ : bits_);

            for (size_t i{0}; i < cu::max(bits.size(), otherBits.size())
                                        - cu::min(bits.size(), otherBits.size()); ++i)
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

        __host__ std::string toString(size_t base = 10, bool showBase = true) const
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

        __device__ __host__ CONSTEXPR operator char() const noexcept
        {
            return cast<char>();
        }

        __device__ __host__ CONSTEXPR operator unsigned char() const noexcept
        {
            return cast<unsigned char>();
        }

        __device__ __host__ CONSTEXPR operator short() const noexcept
        {
            return cast<short>();
        }

        template <typename S>
        __device__ __host__ CONSTEXPR S cast() const
        {
            S n{0};

            size_t const iMax{cu::min(cu::max(longest_type{1}, longest_type{sizeof(S) / sizeof(T)}), longest_type{bits_.size()})};
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

        __device__ __host__ CONSTEXPR operator unsigned short() const noexcept
        {
            return cast<unsigned short>();
        }

        __device__ __host__ CONSTEXPR operator int() const noexcept
        {
            return cast<int>();
        }

        __device__ __host__ CONSTEXPR operator unsigned int() const noexcept
        {
            return cast<unsigned int>();
        }

        __device__ __host__ CONSTEXPR operator long() const noexcept
        {
            return cast<long>();
        }

        __device__ __host__ CONSTEXPR operator unsigned long() const noexcept
        {
            return cast<unsigned long>();
        }

        __device__ __host__ CONSTEXPR operator long long() const noexcept
        {
            return cast<long long>();
        }

        __device__ __host__ CONSTEXPR operator unsigned long long() const noexcept
        {
            return cast<unsigned long long>();
        }

        __device__ __host__ CONSTEXPR bool isNan() const noexcept
        {
            return isNan_;
        }

        __device__ __host__ CONSTEXPR void setNan() noexcept
        {
            isNan_ = true;
            isInfinity_ = false;
            bits_.resize(0);
        }

        __device__ __host__ CONSTEXPR bool isInfinity() const noexcept
        {
            return isInfinity_;
        }

        __device__ __host__ CONSTEXPR void setInfinity() noexcept
        {
            isNan_ = false;
            isInfinity_ = true;
            bits_.resize(0);
        }

        __device__ __host__ CONSTEXPR Integer abs() const
        {
            if (isNegative())
                return -*this;

            return *this;
        }

        __device__ __host__ CONSTEXPR size_t precision() const noexcept
        {
            return bits_.size();
        }

        __device__ __host__ CONSTEXPR void setPrecision(size_t precision)
        {
            assert(precision);
            
            if (precision == bits_.size())
                return;

            cu::vector<T> bits(precision, T{0});

            cu::copy(bits_.rbegin(), bits_.rbegin() + cu::min(bits_.size(), precision), bits.rbegin());

            bits_ = bits;
        }

        template <typename URNG>
        __device__ __host__ CONSTEXPR void setRandom()
        {
            isNan_ = false;
            isInfinity_ = false;

#ifdef __CUDA_ARCH__
            size_t const idx{blockIdx.x * blockDim.x + threadIdx.x};

            curandState state;
            curand_init(clock64(), idx, 0, &state);

            isPositive_ = curand(&state) % 2;
#else
            URNG g;
                    
            isPositive_ = g() % 2;
#endif

            T* a(nullptr);
            auto r{cudaMalloc(&a, sizeof(T) * bits_.size())};
            assert(r == cudaSuccess);
            assert(a);
            
#ifdef __CUDA_ARCH__
            memcpy(a, bits_.data(), sizeof(T) * bits_.size());
#else
            r = cudaMemcpy(a, bits_.data(), sizeof(T) * bits_.size(), cudaMemcpyHostToDevice);
            assert(r == cudaSuccess);
#endif

            size_t const blockSize{BLOCK_SIZE};
            size_t const gridSize{(bits_.size() + blockSize) / blockSize};

#ifdef __CUDA_ARCH__
            Integer_setRandom<T><<<gridSize, blockSize>>>(a, bits_.size(), curand(&state));
#else
            Integer_setRandom<T><<<gridSize, blockSize>>>(a, bits_.size(), g());
#endif

#ifdef __CUDA_ARCH__
            memcpy(bits_.data(), a, sizeof(T) * bits_.size());
#else
            cudaDeviceSynchronize();

            r = cudaMemcpy(bits_.data(), a, sizeof(T) * bits_.size(), cudaMemcpyDeviceToHost);
            assert(r == cudaSuccess);
#endif

            r = cudaFree(a);
            assert(r == cudaSuccess);
        }

        __device__ CONSTEXPR int isPrime(unsigned int const* primes, size_t primesSize,
                                         size_t reps = 25) const
        {
            assert(reps);
            
            if (bits_.empty() || isNan() || isInfinity())
                return 0;
            else if (*this < 2)
                return 0;
            else if (*this == 2)
                return 2;
            else if(!(*this & 1))
                return 0;
            else if (this->template fits<unsigned int>())
            {
                auto p{cu::equal_range(primes, primes + primesSize, this->template cast<unsigned int>())};
                
                if (p.first != primes + primesSize && *p.first != this->template cast<unsigned int>())
                    --p.first;
                
                if (p.first != p.second && *p.first == this->template cast<unsigned int>())
                    return 2;
            }
            
            //Trial divisions
            
            {
                auto const sqrtLimit(sqrt(*this));
                
                bool* divisible(nullptr);
                auto r{cudaMalloc(&divisible, sizeof(bool))};
                assert(r == cudaSuccess);
                assert(divisible);
                *divisible = false;
                
                T* numberData(nullptr);    
                r = cudaMalloc(&numberData, sizeof(T) * bits_.size());
                assert(r == cudaSuccess);
                assert(numberData);
                memcpy(numberData, bits_.data(), sizeof(T) * bits_.size());
                
                T* sqrtLimitData(nullptr);
                r = cudaMalloc(&sqrtLimitData, sizeof(T) * sqrtLimit.bits_.size());
                assert(r == cudaSuccess);
                assert(sqrtLimitData);
                memcpy(sqrtLimitData, sqrtLimit.bits_.data(), sizeof(T) * sqrtLimit.bits_.size());
                
                size_t const blockSize{BLOCK_SIZE};
                size_t const gridSize{(primesSize + blockSize) / blockSize};
                
                Integer_isPrime_trialDivision<T><<<gridSize, blockSize>>>(primes, primesSize,
                                                                          numberData, bits_.size(),
                                                                          sqrtLimitData, sqrtLimit.bits_.size(),
                                                                          divisible);

                r = cudaFree(numberData);
                assert(r == cudaSuccess);
                r = cudaFree(sqrtLimitData);
                assert(r == cudaSuccess);

                if (*divisible)
                {
                    r = cudaFree(divisible);
                    assert(r == cudaSuccess);

                    return 0;
                }

                r = cudaFree(divisible);
                assert(r == cudaSuccess);
                
                if (sqrtLimit < primes[primesSize - 1])
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
                *divisible = false;

                T* numberData(nullptr);
                cudaMalloc(&numberData, sizeof(T) * number.bits_.size());
                memcpy(numberData, number.bits_.data(), sizeof(T) * number.bits_.size());

                T* sData(nullptr);
                cudaMalloc(&sData, sizeof(T) * s.bits_.size());
                memcpy(sData, s.bits_.data(), sizeof(T) * s.bits_.size());
                
                T* RData(nullptr);
                cudaMalloc(&RData, sizeof(T) * R.bits_.size());
                memcpy(RData, R.bits_.data(), sizeof(T) * R.bits_.size());

                T* m_Data(nullptr);
                cudaMalloc(&m_Data, sizeof(T) * m_.bits_.size());
                memcpy(m_Data, m_.bits_.data(), sizeof(T) * m_.bits_.size());

                T* R2modmData(nullptr);
                cudaMalloc(&R2modmData, sizeof(T) * R2modm.bits_.size());
                memcpy(R2modmData, R2modm.bits_.data(), sizeof(T) * R2modm.bits_.size());

                size_t const blockSize{BLOCK_SIZE};
                size_t const gridSize{(reps + blockSize) / blockSize};
                
                Integer_isPrime_millerRabin<T><<<gridSize, blockSize>>>(numberData, number.bits_.size(),
                                                                        sData, s.bits_.size(),
                                                                        RData, R.bits_.size(),
                                                                        m_Data, m_.bits_.size(),
                                                                        R2modmData, R2modm.bits_.size(),
                                                                        divisible, reps);

                auto r{cudaFree(numberData)};
                assert(r == cudaSuccess);
                r = cudaFree(sData);
                assert(r == cudaSuccess);
                r = cudaFree(RData);
                assert(r == cudaSuccess);
                r = cudaFree(m_Data);
                assert(r == cudaSuccess);
                r = cudaFree(R2modmData);
                assert(r == cudaSuccess);
                
                if (*divisible)
                {
                    r = cudaFree(divisible);
                    assert(r == cudaSuccess);

                    return 0;
                }

                r = cudaFree(divisible);
                assert(r == cudaSuccess);
            }

            return 1;
        }

        __host__ CONSTEXPR int isPrime(size_t reps = 25) const
        {
            assert(reps);
            
            (*this < 2);
            
            if (bits_.empty() || isNan() || isInfinity())
                return 0;
            else if (*this < 2)
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
            
            auto isPrimeDivisible
            {
                [](Integer const& n, unsigned int prime) -> bool
                {
                    return !(n % prime);
                }
            };

            //Trial divisions

            {
                auto const sqrtLimit(sqrt(*this));
                
                std::atomic<bool> divisible(false);
        
                auto threadFunc
                {
                    [this, &divisible, &sqrtLimit, &isPrimeDivisible] (size_t start, size_t end) -> void
                    {
                        for (size_t i{start}; i < end && !divisible.load(); ++i)
                        {
                            if (primes[i] > sqrtLimit)
                                break;
                            
                            if (isPrimeDivisible(*this, primes[i]))
                            {
                                divisible.store(true);
                                return;
                            }
                        }
                    }
                };

                size_t const numThreads{std::thread::hardware_concurrency()};
                size_t const chunkSize{primes.size() / numThreads};
                std::vector<std::thread> threads;

                for (size_t i{0}; i < numThreads; ++i)
                {
                    size_t const start{i * chunkSize};
                    size_t const end{(i == numThreads - 1) ? primes.size() : (i + 1) * chunkSize};
                    threads.emplace_back(threadFunc, start, end);
                }

                for (auto& t : threads)
                    t.join();
                
                if (divisible.load())
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
                std::atomic<bool> divisible(false);
            
                auto threadFunc
                {
                    [this, &divisible,
                     &R, &m_, &R2modm, &number, &s]
                    (size_t start, size_t end) -> void
                    {
                        for (size_t i{start}; i < end && !divisible.load(); ++i)
                        {
                            auto a(*this);
                            a.template setRandom<std::random_device>();
                            a.setPositive();
                            a %= number;
                            ++a;

                            auto temp{s};
                            auto mod{modulo(a, temp, *this, R, m_, R2modm)};

                            while (temp != number && !mod && mod != number)
                            {
                                mod = mulmod(mod, mod, *this);
                                temp <<= 1;
                            }

                            if (mod != number && !(temp & 1))
                            {
                                divisible.store(true);
                                return;
                            }
                        }
                    }
                };
                
                size_t const numThreads{std::thread::hardware_concurrency()};
                size_t const chunkSize{reps / numThreads};
                std::vector<std::thread> threads;

                for (size_t i{0}; i < numThreads; ++i)
                {
                    size_t const start{i * chunkSize};
                    size_t const end{(i == numThreads - 1) ? reps : (i + 1) * chunkSize};
                    threads.emplace_back(threadFunc, start, end);
                }

                for (auto& t : threads)
                    t.join();

                if (divisible.load())
                    return 0;
            }

            return 1;
        }

        __device__ __host__ CONSTEXPR void setPositive()
        {
            isPositive_ = true;
        }

        __device__ __host__ CONSTEXPR void setNegative()
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

        __device__ __host__ CONSTEXPR bool bit(size_t n) const noexcept
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

        __device__ __host__ CONSTEXPR void setBit(size_t n, bool bit)
        {
            auto it{bits_.rbegin()};

            while (n > sizeof(T) * 8)
            {
                n -= sizeof(T) * 8;
                ++it;

                if (it == bits_.rend())
                {
                    cu::vector<T> bits(bits_.size() + 1, T{0});

                    cu::copy(bits_.rbegin(), bits_.rend(), bits.rbegin());

                    bits_ = bits;

                    it = bits_.rend() - 1;
                }
            }

            if (bit)
                *it |= T{1} << n;
            else
                *it &= ~(T{1} << n);
        }

        __device__ __host__ CONSTEXPR T bits(size_t n) const noexcept
        {
            if (n >= bits_.size())
                return T{0};

            return bits_[bits_.size() - 1 - n];
        }

        __device__ __host__ CONSTEXPR void setBits(size_t n, T const& bits)
        {
            if (bits_.size() < n)
            {
                cu::vector<T> bits(n, T{0});

                cu::copy(bits_.rbegin(), bits_.rend(), bits.rbegin());

                bits_ = bits;
            }

            bits_[bits_.size() - 1 - n] = bits;

            if (autoAdjust_)
                adjust();
        }

        template <size_t N>
        __device__ __host__ CONSTEXPR void setBits(size_t n, std::bitset<N> const& bits)
        {
            for (size_t i{0}; i < bits.size(); ++i)
                setBit(n + i, bits[i]);

            if (autoAdjust_)
                adjust();
        }

        __device__ __host__ CONSTEXPR size_t count() const noexcept
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

        __device__ __host__ CONSTEXPR Integer<T> number() const noexcept
        {
            Integer<T> number(0);

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

                number += (cu::distance(it, cu::end(bits_)) - 1) * sizeof(T) * 8;
            }

            return number;
        }

        __device__ __host__ CONSTEXPR bool isEven() const noexcept
        {
            if (isNan() || isInfinity())
                return false;
            else if (!bits_.size())
                return true;

            return !(bits_.back() & 1);
        }

        __device__ __host__ CONSTEXPR bool isOdd() const noexcept
        {
            if (!bits_.size())
                return false;

            return bits_.back() & 1;
        }

        template <typename S>
        __device__ __host__ CONSTEXPR bool fits() const
        {
            return (*this == this->template cast<S>());
        }

        __device__ __host__ CONSTEXPR Integer sign() const
        {
            if (*this < 0)
                return Integer(-1);

            return Integer(1);
        }

        __device__ __host__ CONSTEXPR void setSign(Integer const& other) noexcept
        {
            isPositive_ = other.isPositive_;
        }

        __device__ __host__ CONSTEXPR Integer previousPrime() const
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

        __device__ __host__ CONSTEXPR Integer nextPrime() const
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
            Integer p(2 * n);
    
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

        __device__ __host__ CONSTEXPR size_t size() const noexcept
        {
            return bits_.size();
        }

        __device__ __host__ CONSTEXPR void adjust()
        {
            if (!bits_.size())
                return;
            
            auto it{cu::begin(bits_)};
            
            while (!*it && it != cu::end(bits_))
                ++it;
            
            if (it == cu::end(bits_))
                it = cu::end(bits_) - 1;
            
            if (it != cu::begin(bits_))
                bits_ = cu::vector<T>(it, cu::end(bits_));
        }

        __device__ __host__ CONSTEXPR bool isCoprime(Integer const& other) const noexcept
        {
            return gcd(*this, other) == 1;
        }

        __device__ __host__ CONSTEXPR bool autoAdjust() const noexcept
        {
            return autoAdjust_;
        }

        __device__ __host__ CONSTEXPR void setAutoAdjust(bool autoAdjust) noexcept
        {
            autoAdjust_ = autoAdjust;
        }
        
        __device__ __host__ CONSTEXPR char const* data() const noexcept
        {
            return reinterpret_cast<char const*>(bits_.data());
        }
        
        __device__ __host__ CONSTEXPR char* data() noexcept
        {
            return reinterpret_cast<char*>(bits_.data());
        }
        
        __device__ __host__ CONSTEXPR size_t dataSize() const noexcept
        {
            return sizeof(T) / sizeof(char) * bits_.size();
        }
        
        __device__ __host__ CONSTEXPR void setData(char const* data, size_t size) noexcept
        {
            std::memcpy(bits_.data(), data, cu::min(size, dataSize()));
        }

    private:
        bool isPositive_{true};
        cu::vector<T> bits_;
        bool isNan_{false};
        bool isInfinity_{false};
        bool autoAdjust_{true};
};

using Integerc = Integer<unsigned char>;
using Integers = Integer<unsigned short>;
using Integeri = Integer<unsigned int>;
using Integerl = Integer<unsigned long>;
using Integerll = Integer<unsigned long long>;
using Integer8 = Integer<uint8_t>;
using Integer16 = Integer<uint16_t>;
using Integer32 = Integer<uint32_t>;
using Integer64 = Integer<uint64_t>;

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator*(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs *= rhs;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator+(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs += rhs;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator-(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs -= rhs;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator/(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs /= rhs;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator%(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs %= rhs;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator&(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs &= rhs;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator|(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs |= rhs;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator^(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs ^= rhs;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator<<(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs <<= rhs;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator>>(Integer<T> lhs, Integer<T> const& rhs)
{
    return lhs >>= rhs;
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline bool operator>(Integer<T> const& lhs, S const& rhs) noexcept
{
    return lhs.operator>(Integer<T>(rhs));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline bool operator>(S const& lhs, Integer<T> const& rhs) noexcept
{
    return rhs.operator<(Integer<T>(lhs));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline bool operator>=(Integer<T> const& lhs, S const& rhs) noexcept
{
    return lhs.operator>=(Integer<T>(rhs));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline bool operator>=(S const& lhs, Integer<T> const& rhs) noexcept
{
    return rhs.operator<=(Integer<T>(lhs));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline bool operator<(Integer<T> const& lhs, S const& rhs) noexcept
{
    return lhs.operator<(Integer<T>(rhs));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline bool operator<(S const& lhs, Integer<T> const& rhs) noexcept
{
    return rhs.operator>(Integer<T>(lhs));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline bool operator<=(Integer<T> const& lhs, S const& rhs) noexcept
{
    return lhs.operator<=(Integer<T>(rhs));
}

template <typename T, typename S>
__device__ __host__
CONSTEXPR inline bool operator<=(S const& lhs, Integer<T> const& rhs) noexcept
{
    return rhs.operator>=(Integer<T>(lhs));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline bool operator==(S const& lhs, Integer<T> const& rhs) noexcept
{
    return rhs.operator==(Integer<T>(lhs));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline bool operator!=(S const& lhs, Integer<T> const& rhs) noexcept
{
    return rhs.operator!=(Integer<T>(lhs));
}

template <typename T, typename S>
__device__ __host__
CONSTEXPR inline Integer<T> operator+(Integer<T> lhs, S const& rhs)
{
    return lhs += Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator+(S const& lhs, Integer<T> rhs)
{
    return rhs += Integer<T>(lhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator-(Integer<T> lhs, S const& rhs)
{
    return lhs -= Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator-(S const& lhs, Integer<T> rhs)
{
    return -(rhs -= Integer<T>(lhs));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator/(Integer<T> lhs, S const& rhs)
{
    return lhs /= Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator/(S const& lhs, Integer<T> const& rhs)
{
    return Integer<T>(lhs) /= rhs;
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator*(Integer<T> lhs, S const& rhs)
{
    return lhs *= Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator*(S const& lhs, Integer<T> const& rhs)
{
    return Integer<T>(lhs) *= rhs;
}

template <typename T, typename S>
__device__ __host__
CONSTEXPR inline Integer<T> operator%(Integer<T> lhs, S const& rhs)
{
    return lhs %= Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator%(S const& lhs, Integer<T> const& rhs)
{
    return Integer<T>(lhs) %= rhs;
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator<<(Integer<T> lhs, S const& rhs)
{
    return lhs <<= Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator<<(S const& lhs, Integer<T> const& rhs)
{
    return Integer<T>(lhs) <<= rhs;
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator>>(Integer<T> lhs, S const& rhs)
{
    return lhs >>= Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator>>(S const& lhs, Integer<T> const& rhs)
{
    return Integer<T>(lhs) >>= rhs;
}

template <typename T, typename S>
__device__ __host__
CONSTEXPR inline Integer<T> operator&(Integer<T> lhs, S const& rhs)
{
    return lhs &= Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator&(S const& lhs, Integer<T> const& rhs)
{
    return Integer<T>(lhs) &= rhs;
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator|(Integer<T> lhs, S const& rhs)
{
    return lhs |= Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator|(S const& lhs, Integer<T> const& rhs)
{
    return Integer<T>(lhs) |= rhs;
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator^(Integer<T> lhs, S const& rhs)
{
    return lhs ^= Integer<T>(rhs);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> operator^(S const& lhs, Integer<T> const& rhs)
{
    return Integer<T>(lhs) ^= rhs;
}

template <typename T>
__host__ 
CONSTEXPR inline std::ostream& operator<<(std::ostream& os, Integer<T> const& n)
{
    bool const showBase(os.flags() & std::ios_base::showbase);
    
    if (os.flags() & std::ios_base::oct)
        return os << n.toString(8, showBase);
    else if (os.flags() & std::ios_base::hex)
        return os << n.toString(16, showBase);
    else
        return os << n.toString(10, showBase);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> gcd(Integer<T> const& a, Integer<T> const& b)
{
    if (a.isNan() || b.isNan() || a.isInfinity() || b.isInfinity())
    {
        Integer<T> n;
        n.setNan();

        return n;
    }
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

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> gcd(Integer<T> const& a, S const& b)
{
    return gcd(a, Integer<T>(b));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> gcd(S const& a, Integer<T> const& b)
{
    return gcd(Integer<T>(a), b);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> lcm(Integer<T> const& a, Integer<T> const& b)
{
    return (a * b).abs() / gcd(a, b);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> lcm(Integer<T> const& a, S const& b)
{
    return lcm(a, Integer<T>(b));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> lcm(S const& a, Integer<T> const& b)
{
    return lcm(Integer<T>(a), b);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T>
gcdExtended(Integer<T> a, Integer<T> b,
            Integer<T>& u, Integer<T>& v)
{
    if (a.isNan() || b.isNan() || a.isInfinity() || b.isInfinity())
    {
        u.setNan();
        v.setNan();
        
        Integer<T> n;
        n.setNan();

        return n;
    }
    
    if (!a && !b)
        return Integer<T>(0);

    Integer<T> r1(a), u1(1), v1(0);
    Integer<T> r2(b), u2(0), v2(1);
    Integer<T> q, r_temp, u_temp, v_temp;

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

template <typename T, typename S1, typename S2>
__device__ __host__ 
CONSTEXPR inline Integer<T> gcdExtended(S1 const& a, S2 const& b,
                                                Integer<T>& u,
                                                Integer<T>& v)
{
    return gcdExtended(Integer<T>(a), Integer<T>(b), u, v);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> factorial(Integer<T> const& n)
{
    if (n.isNan() || n.isInfinity())
        return n;

    assert(n >= 0);

    if (n == 0)
        return Integer<T>(1);

    return n * factorial(n - 1);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> doubleFactorial(Integer<T> const& n)
{
    return factorial(factorial(n));
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> multiFactorial(Integer<T> const& n,
                                                   Integer<T> const& m)
{
    return pow(factorial(n), m);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> multiFactorial(Integer<T> const& n, S const& m)
{
    return multiFactorial(n, Integer<T>(m));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> multiFactorial(S const& n, Integer<T> const& m)
{
    return multiFactorial(Integer<T>(n), m);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> pow(Integer<T> base, Integer<T> exp)
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
        return Integer<T>(1) << exp;

    Integer<T> result(1);

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

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> pow(Integer<T> const& base, S const& exp)
{
    return pow(base, Integer<T>(exp));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> pow(S const& base, Integer<T> const& exp)
{
    return pow(Integer<T>(base), exp);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> powm(Integer<T> base,
                                         Integer<T> exp,
                                         Integer<T> const& mod)
{
    assert(exp >= 0);

    Integer<T> result(1);

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

template <typename T, typename S, typename U>
__device__ __host__ 
CONSTEXPR inline Integer<T> powm(Integer<T> const& base,
                                         S const& exp, U const& mod)
{
    return powm(base, Integer<T>(exp), Integer<T>(mod));
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> abs(Integer<T> const& n)
{
    return n.abs();
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline cu::pair<Integer<T>, Integer<T> >
computeQr(Integer<T> const& dividend, Integer<T> const& divisor)
{
    if (!divisor)
    {
        Integer<T> n;
        n.setNan();

        return {n, n};
    }
    else if (!dividend)
        return {Integer<T>{0}, Integer<T>{0}};
    else if (dividend.isNan())
        return {dividend, dividend};
    else if (divisor.isNan())
        return {divisor, divisor};
    else if (dividend.isInfinity() || divisor.isInfinity())
    {
        Integer<T> n;
        n.setNan();

        return {n, n};
    }
    else if (divisor.abs() > dividend.abs())
        return {Integer<T>{0}, dividend};
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

    Integer<T> start(1);
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

    return {Integer<T>(0), dividend};
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline cu::pair<Integer<T>, Integer<T> >
computeQr(Integer<T> const& dividend, S const& divisor)
{
    return computeQr(dividend, Integer<T>(divisor));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline cu::pair<Integer<T>, Integer<T> >
computeQr(S const& dividend, Integer<T> const& divisor)
{
    return computeQr(Integer<T>(dividend), divisor);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T>
computeQuotient(Integer<T> const& dividend,
                Integer<T> const& divisor)
{
    if (!divisor)
    {
        Integer<T> n;
        n.setNan();

        return n;
    }
    else if (!dividend)
        return Integer<T>{0};
    else if (dividend.isNan())
        return dividend;
    else if (divisor.isNan())
        return divisor;
    else if (dividend.isInfinity() || divisor.isInfinity())
    {
        Integer<T> n;
        n.setNan();

        return n;
    }
    else if (divisor.abs() > dividend.abs())
        return Integer<T>{0};
    else if (dividend < 0 && divisor < 0)
        return computeQuotient(-dividend, -divisor);
    else if (dividend > 0 && divisor < 0)
        return -computeQuotient(dividend, -divisor);
    else if (dividend < 0 && divisor > 0)
        return -computeQuotient(-dividend, divisor);

    Integer<T> start(1);
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

    return Integer<T>(0);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T>
computeQuotient(Integer<T> const& dividend, S const& divisor)
{
    return computeQuotient(dividend, Integer<T>(divisor));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T>
computeQuotient(S const& dividend, Integer<T> const& divisor)
{
    return computeQuotient(Integer<T>(dividend), divisor);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> fibonacci(Integer<T> n)
{
    assert(n >= 0);

    if (!n)
        return 0;
    else if (n == 1)
        return 1;

    n -= 1;

    Integer<T> fn_2(0);
    Integer<T> fn_1(1);
    Integer<T> fn;

    while (n)
    {
        fn = fn_1 + fn_2;
        fn_2 = fn_1;
        fn_1 = fn;
        --n;
    }

    return fn;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> primorial(Integer<T> n)
{
    Integer<T> result(1);
    Integer<T> number(2);

    while (number <= n)
    {
        result *= number;
        number = number.nextPrime();
    }

    return result;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline int jacobi(Integer<T> const& a, Integer<T> n)
{
    assert(n > 0 && n.isOdd());

    int result(1);
    Integer<T> prime(2);

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

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline int jacobi(Integer<T> const& a, S const& n)
{
    return jacobi(a, Integer<T>(n));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline int jacobi(S const& a, Integer<T> const& n)
{
    return jacobi(Integer<T>(a), n);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline int legendre(Integer<T> const& a, Integer<T> const& p)
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

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> legendre(Integer<T> const& a, S const& p)
{
    return legendre(a, Integer<T>(p));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> legendre(S const& a, Integer<T> const& p)
{
    return legendre(Integer<T>(a), p);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline int kronecker(Integer<T> const& a, Integer<T> const& b)
{
    if (a == b)
        return 1;
    else
        return 0;
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline int kronecker(Integer<T> const& a, S const& b)
{
    return kronecker(a, Integer<T,cu::vector<T>>(b));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline int kronecker(S const& a, Integer<T> const& b)
{
    return kronecker(Integer<T>(a), b);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> binomial(Integer<T> const& n,
                                             Integer<T> const& k)
{
    assert(n >= 0 && k >= 0);

    return factorial(n) / (factorial(k) * factorial(n - k));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> binomial(Integer<T> const& n, S const& k)
{
    return binomial(n, Integer<T>(k));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> binomial(S const& n, Integer<T> const& k)
{
    return binomial(Integer<T>(n), k);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> sqrt(Integer<T> const& n)
{
    if (n < 0)
    {
        Integer<T> n;
        n.setNan();

        return n;
    }
    else if (!n || n == 1 || n.isNan() || n.isInfinity())
        return n;

    Integer<T> lo(1), hi(n);
    Integer<T> res(1);

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

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> root(Integer<T> const& x,
                                         Integer<T> const& n)
{
    assert(n > 0);
    
    if (x < 0)
    {
        Integer<T> n;
        n.setNan();

        return n;
    }
    else if (!x || x == 1 || x.isNan() || x.isInfinity())
        return x;
    else if (n == 1)
        return x;
    else if (n.isNan() || n.isInfinity())
    {
        Integer<T> n;
        n.setNan();

        return n;
    }

    Integer<T> lo(1), hi(x);
    Integer<T> res(1);

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

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> root(S const& x, Integer<T> const& n)
{
    return root(Integer<T>(x), n);
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T> root(Integer<T> const& x, S const& n)
{
    return root(x, Integer<T>(n));
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T>
computeQuotientBinary(Integer<T> dividend, Integer<T> const& divisor)
{
    if (!divisor)
    {
        Integer<T> n;
        n.setNan();

        return n;
    }
    else if (!dividend)
        return Integer<T>{0};
    else if (dividend.isNan())
        return dividend;
    else if (divisor.isNan())
        return divisor;
    else if (dividend.isInfinity() || divisor.isInfinity())
    {
        Integer<T> n;
        n.setNan();

        return n;
    }
    else if (divisor.abs() > dividend.abs())
        return Integer<T>{0};
    else if (dividend < 0 && divisor > 0)
        return -computeQuotientBinary(-dividend, divisor);
    else if (dividend > 0 && divisor < 0)
        return -computeQuotientBinary(dividend, -divisor);
    else if (dividend < 0 && divisor < 0)
        return computeQuotientBinary(-dividend, -divisor);

    Integer<T> quotient(0);
    auto tempDivisor(divisor);
    Integer<T> bit(1);

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

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T>
computeQuotientBinary(Integer<T> const& dividend, S const& divisor)
{
    return computeQuotientBinary(dividend, Integer<T>(divisor));
}

template <typename T, typename S,class cu::vector<T>>
__device__ __host__ 
CONSTEXPR inline Integer<T>
computeQuotientBinary(S const& dividend, Integer<T> const& divisor)
{
    return computeQuotientBinary(Integer<T>(dividend), divisor);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline cu::pair<Integer<T>, Integer<T> >
computeQrBinary(Integer<T> const& dividend,
                Integer<T> const& divisor)
{
    cu::pair<Integer<T>, Integer<T> > qr{computeQuotientBinary(dividend, divisor), Integer<T>(0)};
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

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline cu::pair<Integer<T>, Integer<T> >
computeQrBinary(Integer<T> const& dividend, S const& divisor)
{
    return computeQrBinary(dividend, Integer<T>(divisor));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline cu::pair<Integer<T>, Integer<T> >
computeQrBinary(S const& dividend, Integer<T> const& divisor)
{
    return computeQrBinary(Integer<T>(dividend), divisor);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T>
computeQuotientBurnikelZiegler(Integer<T> dividend,
                               Integer<T> const& divisor)
{
    if (!divisor)
    {
        Integer<T> n;
        n.setNan();

        return n;
    }
    else if (!dividend)
        return Integer<T>{0};
    else if (dividend.isNan())
        return dividend;
    else if (divisor.isNan())
        return divisor;
    else if (dividend.isInfinity() || divisor.isInfinity())
    {
        Integer<T> n;
        n.setNan();

        return n;
    }
    else if (divisor.abs() > dividend.abs())
        return Integer<T>{0};

    cu::pair<Integer<T>, Integer<T> > qr;
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

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T>
computeQuotientBurnikelZiegler(Integer<T> const& dividend, S const& divisor)
{
    return computeQuotientBurnikelZiegler(dividend, Integer<T>(divisor));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline Integer<T>
computeQuotientBurnikelZiegler(S const& dividend, Integer<T> const& divisor)
{
    return computeQuotientBurnikelZiegler(Integer<T>(dividend), divisor);
}

template <typename T>
__device__ __host__
void inner1(cu::vector<Integer<T> >& a_digits, Integer<T> const& x,
            Integer<T> const& L,
            Integer<T> const& R, Integer<T> const& n)
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
}

template <typename T>
__device__ __host__
cu::vector<Integer<T> > _int2digits(Integer<T> const& a,
                                             Integer<T> const& n)
{
    assert(a >= 0);

    if (!a)
        return cu::vector<Integer<T> >{Integer<T>(0)};

    cu::vector<Integer<T> > a_digits(((a.number() + n - 1).template cast<longest_type>() / n).template cast<longest_type>(), Integer<T>(0));

    if (a)
        inner1(a_digits, a, Integer<T>(0), Integer<T>(a_digits.size()), n);

    return a_digits;
}

template <typename T>
__device__ __host__
Integer<T> inner2(cu::vector<Integer<T> > const& digits, Integer<T> const& L,
                          Integer<T> const& R, Integer<T> const& n)
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
}

template <typename T>
__device__ __host__ 
cu::pair<Integer<T>, Integer<T> >
_div2n1n(Integer<T> a, Integer<T> b,
         Integer<T> n)
{
    if (a.template fits<longest_type>() && b.template fits<longest_type>())
        return {Integer<T>(a.template cast<longest_type>() / b.template cast<longest_type>()),
                Integer<T>(a.template cast<longest_type>() % b.template cast<longest_type>())};

    auto pad(n & 1);

    if (pad)
    {
        a <<= 1;
        b <<= 1;
        ++n;
    }

    auto const half_n(n >> 1);
    Integer<T> mask(1);
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
}

template <typename T>
__device__ __host__ 
cu::pair<Integer<T>, Integer<T> >
_div3n2n(Integer<T> const& a12, Integer<T> const& a3,
         Integer<T> const& b, Integer<T> const& b1,
         Integer<T> const& b2, Integer<T> const& n)
{
    Integer<T> q, r;

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
}

template <typename T>
__device__ __host__ 
Integer<T> _digits2int(cu::vector<Integer<T> > const& digits,
                               Integer<T> const& n)
{
    if (!digits.size())
        return Integer<T>(0);

    return inner2(digits, Integer<T>(0), Integer<T>(digits.size()), n);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline cu::pair<Integer<T>, Integer<T> >
computeQrBurnikelZiegler(Integer<T> const& dividend,
                         Integer<T> const& divisor)
{
    if (!divisor)
    {
        Integer<T> n;
        n.setNan();

        return {n, n};
    }
    else if (!dividend)
        return {Integer<T>{0}, Integer<T>{0}};
    else if (dividend.isNan())
        return {dividend, dividend};
    else if (divisor.isNan())
        return {divisor, divisor};
    else if (dividend.isInfinity() || divisor.isInfinity())
    {
        Integer<T> n;
        n.setNan();

        return {n, n};
    }
    else if (divisor.abs() > dividend.abs())
        return {Integer<T>(0), dividend};
    else if (divisor.abs() == 1)
        return {divisor.sign() * dividend, Integer<T>(0)};
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

    auto const n{divisor.number()};
    auto const a_digits(_int2digits(dividend, n));

    Integer<T> r(0);
    Integer<T> q(0);
    cu::vector<Integer<T> > q_digits;

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

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline cu::pair<Integer<T>, Integer<T> >
computeQrBurnikelZiegler(Integer<T> const& dividend, S const& divisor)
{
    return computeQrBurnikelZiegler(dividend, Integer<T>(divisor));
}

template <typename T, typename S>
__device__ __host__ 
CONSTEXPR inline cu::pair<Integer<T>, Integer<T> >
computeQrBurnikelZiegler(S const& dividend, Integer<T> const& divisor)
{
    return computeQrBurnikelZiegler(Integer<T>(dividend), divisor);
}

__host__ inline Integerc operator""_zc(char const* str)
{
    return Integerc(str);
}

__host__ inline Integers operator""_zs(char const* str)
{
    return Integers(str);
}

__host__ inline Integeri operator""_zi(char const* str)
{
    return Integeri(str);
}

__host__ inline Integerl operator""_zl(char const* str)
{
    return Integerl(str);
}

__host__ inline Integerll operator""_zll(char const* str)
{
    return Integerll(str);
}

__host__ inline Integer8 operator""_z8(char const* str)
{
    return Integer8(str);
}

__host__ inline Integer16 operator""_z16(char const* str)
{
    return Integer16(str);
}

__host__ inline Integer32 operator""_z32(char const* str)
{
    return Integer32(str);
}

__host__ inline Integer<longest_type> operator""_z(char const* str)
{
    return Integer<longest_type>(str);
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> const& min(Integer<T> const& a,
                                               Integer<T> const& b)
{
    return a < b ? a : b;
}

template <typename T>
__device__ __host__ 
CONSTEXPR inline Integer<T> const& max(Integer<T> const& a,
                                               Integer<T> const& b)
{
    return a > b ? a : b;
}

#endif // INTEGER_CUH

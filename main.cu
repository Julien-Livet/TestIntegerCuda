#include <iostream>

#include "Integer.cuh"

template <typename T>
__global__ void isPrime(T const* nData, size_t nDataSize, unsigned int const* p, size_t primesSize, int* isPrime, size_t reps = 25)
{printf("yo0\n");
    Integer<T> const n(nData, nData + nDataSize);
printf("yo1\n");
    *isPrime = n.isPrime(p, primesSize, reps);printf("yo2\n");
}

int main()
{
    unsigned int* p(nullptr);
    auto r{cudaMalloc(&p, sizeof(unsigned int) * primes.size())};
    assert(r == cudaSuccess);
    assert(p);
    r = cudaMemcpy(p, primes.data(), sizeof(unsigned int) * primes.size(), cudaMemcpyHostToDevice);
    assert(r == cudaSuccess);

    int* prime(nullptr);
    r = cudaMalloc(&prime, sizeof(int));
    assert(r == cudaSuccess);
    assert(prime);

    using T = uint64_t;
    
    {
        auto const n(23 * 29_z);

        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        T* nData(nullptr);    
        r = cudaMalloc(&nData, sizeof(T) * n.bits().size());
        assert(r == cudaSuccess);
        assert(nData);
        r = cudaMemcpy(nData, n.bits().data(), sizeof(T) * n.bits().size(), cudaMemcpyHostToDevice);
        assert(r == cudaSuccess);
        
        t = std::chrono::steady_clock::now();std::cout << "hey6" << std::endl;
        isPrime<T><<<1, 1>>>(nData, n.bits().size(), p, primes.size(), prime);
std::cout << "hey7" << std::endl;
        cudaDeviceSynchronize();
std::cout << "hey8" << std::endl;
        int pr;
        r = cudaMemcpy(&pr, prime, sizeof(int), cudaMemcpyDeviceToHost);
        assert(r == cudaSuccess);

        std::cout << pr << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        cudaFree(nData);

        return 0;
    }
/*
    {
        auto const n(56062005704198360319209_z);
        
        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;
    }

    {
        auto const n(4113101149215104800030529537915953170486139623539759933135949994882770404074832568499_z);

        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;
    }

    {
        Integer64 n;
        n.setPrecision(1024 / 64);
        n.setRandom<std::random_device>();
        n.setPositive();
        if (!(n % 2))
            ++n;

        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;
    }
*/
    r = cudaFree(p);
    assert(r == cudaSuccess);
    r = cudaFree(prime);
    assert(r == cudaSuccess);

    return 0;
}

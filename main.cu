#include <iostream>

#include "Integer.cuh"

template <typename T>
__global__ void isPrime(T const* nData, size_t nDataSize, unsigned int const* p, size_t primesSize, int* isPrime, size_t reps = 25)
{
    Integer<T> const n(nData, nData + nDataSize);

    *isPrime = n.isPrime(p, primesSize, reps);
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

    r = cudaDeviceSetLimit(cudaLimitStackSize, 256 * 256);
    assert(r == cudaSuccess);

    r = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 256);
    assert(r == cudaSuccess);
    
    using T = uintmax_t;
    
    {
        std::cout << "Block #1" << std::endl;
        cudaMemset(prime, -1, sizeof(int));

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
        
        t = std::chrono::steady_clock::now();
        isPrime<T><<<1, 1>>>(nData, n.bits().size(), p, primes.size(), prime);

        cudaDeviceSynchronize();

        int pr(-1);
        r = cudaMemcpy(&pr, prime, sizeof(int), cudaMemcpyDeviceToHost);
        assert(r == cudaSuccess);

        std::cout << pr << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        cudaFree(nData);
    }

    {
        std::cout << "Block #2" << std::endl;
        cudaMemset(prime, -1, sizeof(int));

        auto const n(1299709 * 1299721_z);

        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        T* nData(nullptr);    
        r = cudaMalloc(&nData, sizeof(T) * n.bits().size());
        assert(r == cudaSuccess);
        assert(nData);
        r = cudaMemcpy(nData, n.bits().data(), sizeof(T) * n.bits().size(), cudaMemcpyHostToDevice);
        assert(r == cudaSuccess);
        
        t = std::chrono::steady_clock::now();
        isPrime<T><<<1, 1>>>(nData, n.bits().size(), p, primes.size(), prime);

        cudaDeviceSynchronize();

        int pr(-1);
        r = cudaMemcpy(&pr, prime, sizeof(int), cudaMemcpyDeviceToHost);
        assert(r == cudaSuccess);

        std::cout << pr << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        cudaFree(nData);
    }

    {
        std::cout << "Block #3" << std::endl;
        cudaMemset(prime, -1, sizeof(int));
    
        auto const n(56062005704198360319209_z);

        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        T* nData(nullptr);    
        r = cudaMalloc(&nData, sizeof(T) * n.bits().size());
        assert(r == cudaSuccess);
        assert(nData);
        r = cudaMemcpy(nData, n.bits().data(), sizeof(T) * n.bits().size(), cudaMemcpyHostToDevice);
        assert(r == cudaSuccess);
        
        t = std::chrono::steady_clock::now();
        isPrime<T><<<1, 1>>>(nData, n.bits().size(), p, primes.size(), prime);

        cudaDeviceSynchronize();

        int pr(-1);
        r = cudaMemcpy(&pr, prime, sizeof(int), cudaMemcpyDeviceToHost);
        assert(r == cudaSuccess);

        std::cout << pr << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        cudaFree(nData);
    }

    {
        std::cout << "Block #4" << std::endl;
        cudaMemset(prime, -1, sizeof(int));

        auto const n(4113101149215104800030529537915953170486139623539759933135949994882770404074832568499_z);

        T* nData(nullptr);    
        r = cudaMalloc(&nData, sizeof(T) * n.bits().size());
        assert(r == cudaSuccess);
        assert(nData);
        r = cudaMemcpy(nData, n.bits().data(), sizeof(T) * n.bits().size(), cudaMemcpyHostToDevice);
        assert(r == cudaSuccess);
        
        auto const t{std::chrono::steady_clock::now()};
        isPrime<T><<<1, 1>>>(nData, n.bits().size(), p, primes.size(), prime);

        cudaDeviceSynchronize();

        int pr(-1);
        r = cudaMemcpy(&pr, prime, sizeof(int), cudaMemcpyDeviceToHost);
        assert(r == cudaSuccess);

        std::cout << pr << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        cudaFree(nData);
    }

    {
        std::cout << "Block #5" << std::endl;
        cudaMemset(prime, -1, sizeof(int));

        Integer64 n;
        n.setPrecision(1024 / 64);
        n.setRandom<std::random_device>();
        n.setPositive();
        if (!(n % 2))
            ++n;

            uint64_t* nData(nullptr);    
        r = cudaMalloc(&nData, sizeof(uint64_t) * n.bits().size());
        assert(r == cudaSuccess);
        assert(nData);
        r = cudaMemcpy(nData, n.bits().data(), sizeof(uint64_t) * n.bits().size(), cudaMemcpyHostToDevice);
        assert(r == cudaSuccess);
        
        auto const t{std::chrono::steady_clock::now()};
        isPrime<uint64_t><<<1, 1>>>(nData, n.bits().size(), p, primes.size(), prime);

        cudaDeviceSynchronize();

        int pr(-1);
        r = cudaMemcpy(&pr, prime, sizeof(int), cudaMemcpyDeviceToHost);
        assert(r == cudaSuccess);

        std::cout << pr << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        cudaFree(nData);
    }

    r = cudaFree(p);
    assert(r == cudaSuccess);
    r = cudaFree(prime);
    assert(r == cudaSuccess);

    return 0;
}

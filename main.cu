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
    unsigned int* p;
    cudaMalloc(&p, sizeof(unsigned int) * primes.size());
    cudaMemcpy(p, primes.data(), sizeof(unsigned int) * primes.size(), cudaMemcpyHostToDevice);

    int* prime;
    cudaMalloc(&prime, sizeof(int));

    using T = uint64_t;
    
    {
        auto const n(23 * 29_z);

        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        T* nData(nullptr);    
        cudaMalloc(&nData, sizeof(T) * n.bits().size());
        cudaMemcpy(nData, n.bits().data(), sizeof(T) * n.bits().size(), cudaMemcpyHostToDevice);
        
        t = std::chrono::steady_clock::now();std::cout << "hey6" << std::endl;
        isPrime<T><<<1, 1>>>(nData, n.bits().size(), p, primes.size(), prime);
std::cout << "hey7" << std::endl;
        cudaDeviceSynchronize();
std::cout << "hey8" << std::endl;
        int pr;
        cudaMemcpy(&pr, prime, sizeof(int), cudaMemcpyDeviceToHost);

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
    cudaFree(p);
    cudaFree(prime);

    return 0;
}

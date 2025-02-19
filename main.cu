#include <iostream>

#include "Integer.cuh"

template <typename T>
__global__ void isPrime(T const* nData, size_t nDataSize, unsigned int const* p, size_t primesSize, int* isPrime, size_t reps = 25)
{
    Integer<T, cu::vector<T> > const n(nData, nData + nDataSize);
    
    *isPrime = n.isPrime(p, primesSize, reps);
}

int main()
{
    unsigned int* p;
    cudaMalloc(&p, sizeof(unsigned int) * primes.size());
    cudaMemcpy(p, primes.data(), sizeof(unsigned int) * primes.size(), cudaMemcpyHostToDevice);

    int* prime;
    cudaMalloc(&prime, sizeof(int));

    using T = unsigned long long;
    using Z = Integer<T, std::vector<T> >;
    
    {
        Z const n(23 * 29);
        
        T* nData(nullptr);    
        cudaMalloc(&nData, sizeof(T) * n.bits().size());
        memcpy(nData, n.bits().data(), sizeof(T) * n.bits().size());
        
        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        t = std::chrono::steady_clock::now();
        isPrime<T><<<1, 1>>>(nData, n.bits().size(), p, primes.size(), prime);

        int pr;
        cudaMemcpy(&pr, prime, sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << pr << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;

        cudaFree(nData);

        return 0;
    }
/*
    {
        std::string const s("56062005704198360319209");
        Z const n(s);
        
        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;
    }

    {
        std::string const s("4113101149215104800030529537915953170486139623539759933135949994882770404074832568499");
        Z const n(s);

        auto t{std::chrono::steady_clock::now()};
        std::cout << n.isPrime() << std::endl;
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t).count() << " ms" << std::endl;
    }

    {
        Z n;
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

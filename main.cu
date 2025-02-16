#include <cassert>
#include <iostream>
#include <vector>

#include <thrust/host_vector.h>

#include "Integer.cuh"

int main()
{
    using Z = Integer<unsigned long long, thrust::host_vector<unsigned long long> >;

    {
        std::cout << "hop0" << std::endl;
        Z bigNum;
        bigNum.setPrecision(pow(Z(2), 24).cast<size_t>());
        //bigNum.setPrecision(4);
        std::cout << "hop1" << std::endl;
        bigNum = ~bigNum;
        std::cout << "hop2" << std::endl;
        bigNum.setRandom<std::random_device>();
        std::cout << "hop3" << std::endl;
        bigNum.setPositive();
        std::cout << "hop4" << std::endl;
        if (!(bigNum % 2))
            --bigNum;
        std::cout << "hop5" << std::endl;
    }
    
    auto const bigPrime{Z(111)};
    std::cout << bigPrime.isPrime() << std::endl;
/*
    auto const bigPrime{Z(4113101149215104800030529537915953170486139623539759933135949994882770404074832568499)};
    std::cout << bigPrime.isPrime() << std::endl;
*/
    return 0;
}

#include <iostream>

#include "Integer.cuh"

int main()
{
    using Z = Integer<unsigned long long, std::vector<unsigned long long> >;

    std::string const s("56062005704198360319209");
    Z const n(s);

    std::cout << n.isPrime() << std::endl;

    return 0;
}
